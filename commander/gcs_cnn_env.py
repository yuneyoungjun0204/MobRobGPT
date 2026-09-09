"""commander/gcs_cnn_env.py — GCS 실텔레메트리 아군 + 로스백 실적선을 함께 쓰는 CNN 정책 환경.

`commander/replay_cnn_env.py::ReplayCnnEnv`(적=로스백 실측, 아군=시뮬 물리)와
`commander/ros2_unet_env.py::ROS2CnnEnv`(아군·적 전부 자체 ROS2 센서) 사이의 조합이다.

  적  : ReplayCnnEnv 그대로 -- 로스백(BagEnemyReplay) 재생, 그물 재보급 타이머도 상속.
  아군: ROS2CnnEnv 의 `_advance_and_paint`와 동일한 로직(§defense_env.py `_micro`와
        1:1 대응)이지만, 센서 대신 GCS `/api/state`에서 위치·헤딩을 읽는다
        (commander/gcs_bridge.py::GcsAllyLink). 시뮬 물리(`_micro`)는 돌리지 않는다
        -- 실제 보트가 기동하고, GCS가 그 결과를 텔레메트리로 되돌려준다.

매 tick, 그 배의 "현재 활성 경유점"(`route[ptr]`, ENU 미터로 변환)을 GCS
`/gcs`-호환 goto 명령으로 보낸다(source="rl") -- 이게 `docs/contracts.md` §4가 말하는
"RL이 0.5~2Hz로 계속 보낸다"는 그 지령이다. 그물 전개 시작/종료는 GCS로 보내지 않고
`NetDeploySink`로만 알린다(gcs_bridge.py 모듈독스트링 참고 -- 액추에이터 명령 금지).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from boatattack_sim.env import cnn_map as CM

from .bag_replay import BagEnemyReplay
from .gcs_bridge import GcsAllyLink, GcsRequestError, NetDeploySink
from .replay_cnn_env import ReplayCnnEnv


class GcsBagCnnEnv(ReplayCnnEnv):
    """적=로스백 리플레이(real), 아군=GCS 실텔레메트리(real)로 구동하는 CNN 점수맵 정책 환경."""

    def __init__(
        self,
        ckpt: str,
        bag: BagEnemyReplay,
        span_real: float,
        ally_link: GcsAllyLink,
        *,
        net_sink: Optional[NetDeploySink] = None,
        publish_hz: float = 2.0,
        ally_speed_real: float = 0.3,
        enemy_mode: str = "wave",
        device: str = "cpu",
        nets_per_ship: int = 3,
        geo: tuple[float, float] | None = None,
        enu_origin: tuple[float, float] | None = None,
        net_reload_period_real: float | None = None,
    ):
        super().__init__(
            ckpt, bag, span_real, ally_speed_real=ally_speed_real,
            enemy_mode=enemy_mode, device=device, nets_per_ship=nets_per_ship,
            geo=geo, enu_origin=enu_origin,
            net_reload_period_real=net_reload_period_real)
        if ally_link.n_allies != self.P:
            raise ValueError(
                f"ally_link has {ally_link.n_allies} vehicle_ids but the checkpoint "
                f"policy expects P={self.P} allies")
        self.ally_link = ally_link
        self.net_sink = net_sink or NetDeploySink()
        if publish_hz <= 0:
            raise ValueError("publish_hz must be > 0")
        self._publish_period_real = 1.0 / float(publish_hz)
        self._have_gcs = False
        self._prev_pos = self.a_pos[0].copy()
        # 실시간(wall-clock) 기준 -- `_t_real`(시뮬 시각)로 재던 초판은 --no-realtime 이나
        # LLM 재배정으로 tick 이 밀리는 동안 스로틀이 같이 멈춰 GCS 를 과다/과소 호출했다
        # (아키텍트 검토 B3). publish 는 GCS 로 나가는 실제 명령이므로 실제 시계를 쓴다.
        self._next_publish_wall = 0.0

    # ── GCS 텔레메트리 주입 ─────────────────────────────────────────────
    def _ingest_gcs_allies(self) -> bool:
        """`/api/state` 스냅샷 -> a_pos/a_hdg/a_alive. 실패하거나 못 받으면 False.

        살아있는(alive) 배만 위치/헤딩을 덮어쓴다 -- 죽은 배를 원점 등 스냅샷의 0값으로
        덮으면 그 좌표가 sim 격자의 임의의 한 점으로 투영돼(`enu_to_sim`), 도착 판정과
        (더 심각하게는) 다른 배들의 클러스터링/배정 관측에 유령 위치로 섞여 들어간다
        (아키텍트 검토 B1 -- 실행으로 확인된 결함). 마지막으로 알려진 값을 그대로 들고
        있는 편이 훨씬 안전하다 -- `commander/ros2_sensor_bridge.py`의 영구 저장소와
        같은 방식이다.
        """
        try:
            snap = self.ally_link.ally_snapshot()
        except Exception as exc:                      # pragma: no cover -- network edge
            print(f"[gcs_bridge] ally telemetry poll failed: {exc}")
            return False
        alive = snap.alive
        if alive.any():
            sim_pos = self.scale.enu_to_sim(snap.pos[alive])
            # 적과 동일한 맵 경계 클리핑(replay_cnn_env.py:101-103, §F2) -- 안 하면
            # 래스터 관측·배정이 지도 밖 아군을 서로 다른 위치로 취급한다.
            lo = CM.origin(self.cfg)
            hi = lo + 2.0 * CM.extent_m(self.cfg)
            self.a_pos[0][alive] = np.clip(sim_pos, lo, hi)
            self.a_hdg[0][alive] = snap.hdg[alive]     # 이미 nav 규약(0=N, CW+) -- 변환 불필요
        self.a_alive[0] = alive
        return True

    # ── WP 도착 + 그물 부설 진행 (ros2_unet_env.py::_advance_and_paint 와 1:1 대응) ──
    def _advance_and_paint(self) -> None:
        cfg = self.cfg
        moved = np.hypot(*(self.a_pos[0] - self._prev_pos).T)
        self._prev_pos = self.a_pos[0].copy()

        ptr_c = np.clip(self.ptr, 0, self.Kw - 1)
        route_t = np.take_along_axis(self.route, ptr_c[:, :, None, None], axis=2)[:, :, 0, :]
        self.net_end = route_t.copy()
        # `& self.a_alive`: a dead/disconnected ship's a_pos is held at its last known
        # value (never a phantom zero, per _ingest_gcs_allies), but it must not be able
        # to register "arrival" -- or a real position it once had could coincide with a
        # freshly (re)assigned route and spuriously finish/advance a net leg it isn't
        # actually flying (architect review B1).
        arrived = (np.hypot(self.a_pos[..., 0] - route_t[..., 0],
                            self.a_pos[..., 1] - route_t[..., 1]) <= cfg.arrive_radius
                   ) & self.a_alive

        cur_net = np.take_along_axis(self.net_mask, ptr_c[..., None], axis=2)[..., 0]
        start = (cur_net & (~self.leg_netted) & (~self.doing_net)
                 & self.a_alive & (self.a_nets > 0))
        if start.any():
            self.doing_net |= start
            self.leg_netted |= start
            self.paint_dist[start] = 0.0
            self.net_start[start] = self.a_pos[start]
            self.a_nets -= start.astype(np.int64)
            self.stats["nets_used"] += int(start.sum())
            for p in np.where(start[0])[0]:
                self.net_sink.set(self.ally_link.vehicle_ids[p], True)

        painting = self.doing_net & (self.paint_dist < cfg.net_max_len) & self.a_alive
        self.paint_dist += np.where(painting, moved[None, :], 0.0)
        self._paint(painting)

        finish = self.doing_net & ((self.paint_dist >= cfg.net_max_len) | arrived)
        if finish.any():
            self.doing_net &= ~finish
            self._rasterize_net(self.net_start, self.net_end, finish)
            self.prev_on_inst |= finish
            for p in np.where(finish[0])[0]:
                self.net_sink.set(self.ally_link.vehicle_ids[p], False)

        arr_route = (~self.doing_net) & arrived
        advance = arr_route & (self.ptr < self.Kw - 1)
        if arr_route.any():
            ai, aj = np.where(arr_route)
            self.wp_reached[ai, aj, ptr_c[ai, aj]] = True
        self.ptr = np.where(advance, self.ptr + 1, self.ptr)
        self.leg_netted = np.where(advance, False, self.leg_netted)

    # ── GCS로 현재 활성 경유점 송신 (0.5~2Hz 지속 스트림 계약) ──
    def _publish_to_gcs(self) -> None:
        now = time.monotonic()
        if now < self._next_publish_wall:
            return
        self._next_publish_wall = now + self._publish_period_real
        ptr_all = np.clip(self.ptr[0], 0, self.Kw - 1)
        for p in range(self.P):
            if not bool(self.a_alive[0, p]):
                continue                          # 위치를 모르는 배는 명령하지 않는다
            east, north = self.scale.sim_to_enu(self.route[0, p, ptr_all[p]])
            try:
                verdict = self.ally_link.submit_goto(
                    self.ally_link.vehicle_ids[p], east=float(east), north=float(north))
            except GcsRequestError as exc:
                # 한 번의 TCP 타임아웃/거부로 전체 루프를 죽이지 않는다 -- gcs 의
                # hold->fade->release 가 짧은 공백을 흡수하도록 설계돼 있다
                # (docs/contracts.md §4). 다음 tick에서 다시 시도한다(아키텍트 검토 B2).
                print(f"[gcs_bridge] {self.ally_link.vehicle_ids[p]} goto POST failed: {exc}")
                continue
            if not verdict.get("accepted", False):
                print(f"[gcs_bridge] {self.ally_link.vehicle_ids[p]} goto refused: "
                      f"{verdict.get('reason')}: {verdict.get('detail', '')}")

    @property
    def ready(self) -> bool:
        """True once GCS has reported every registered ally alive at least once.

        Sticky, same as `_have_gcs` (this just exposes it) -- once every hull has been
        seen, later single-vehicle dropouts are `a_alive` holding last-known (B1), not a
        return to "not ready". Callers must not treat a `step()` call as a decision tick
        while this is False, or `_micro_ct` -- which does not advance during the wait --
        makes `decision_idx % replan_period == 0` trivially true on every pass: bogus
        decisions get logged and a completed LLM future gets immediately resubmitted
        (architect review: S5 follow-up, found after the S5 fix itself was approved).
        """
        return self._have_gcs

    def missing_ally_ids(self) -> list[str]:
        """vehicle_ids GCS is not currently reporting alive for -- so a caller waiting
        on `ready` can say what it's waiting for instead of hanging silently."""
        return [vid for vid, alive in zip(self.ally_link.vehicle_ids, self.a_alive[0])
                if not alive]

    # ── 운용 루프 ────────────────────────────────────────────────────
    def step(self):
        if bool(self.done[0]):
            self.t[0] = 0
            self.done[0] = False
        if self._t_real >= self._next_reload_real:
            self.a_nets[0] = self.cfg.nets_per_ship
            self._next_reload_real += self.net_reload_period_real

        if not self._ingest_gcs_allies():
            self._inject_replay_enemies()
            return self.get_frame()                # 텔레메트리 미수신 -- 마지막 상태 유지
        if not self._have_gcs:
            if not bool(self.a_alive[0].all()):
                # 전 척이 다 켜지기 전엔 결정/발행을 시작하지 않는다 -- 일부만 보이는
                # 상태에서 첫 결정이 굳어지는 문제를 피한다(replay_cnn_env.py의
                # first_common_time_sec/§F3와 같은 원칙, 아키텍트 검토 S5).
                self._inject_replay_enemies()
                return self.get_frame()
            self._have_gcs = True
            self._prev_pos = self.a_pos[0].copy()
            # Falls through to decide-then-advance below on this same tick, at
            # `_micro_ct == 0` -- with whatever `_plan` is currently set (typically
            # None -> idle, since no caller could have submitted a replan before seeing
            # `env.ready` go True, which only happens on this method's *return*). That
            # first idle decision is expected and safe; callers should watch `env.ready`
            # (right after calling `step()`, not before) to fire an immediate one-off
            # replan rather than waiting a full `replan_period` for it. Returning early
            # here instead (an earlier attempt at this fix) does NOT help: `_micro_ct`
            # would stay 0, so a caller's post-step "did a decision batch just finish"
            # check -- which also keys off `_micro_ct == 0` -- fires on this transition
            # tick even though no decision happened, inflating its decision counter by
            # one and misaligning every later replan-period check instead.

        # decide-then-advance (not ros2_unet_env.py's advance-then-decide): this matches
        # the sim's own ordering in replay_cnn_env.py/_micro -- a fresh decision's route
        # is what this tick's arrival/net check should measure against, not last tick's.
        if self._micro_ct % self.cfg.decision_period == 0:
            self._rl_decide()
            self._sprev = {k: 0.0 for k in self._SK}
        self._advance_and_paint()
        # captures/breaches/ally_collisions only ever change inside `_micro`, which this
        # class never calls (real hulls move themselves) -- this loop is a structural
        # no-op here, kept only so a future edit to ReplayCnnEnv.step()'s bookkeeping
        # stays visible as a diff against this copy (architect review: keep-in-sync note).
        for k in self._SK:
            cur = float(self._ev[k][0]) if self._ev is not None else 0.0
            self.stats[k] += cur - self._sprev.get(k, 0.0)
            self._sprev[k] = cur
        self._micro_ct += 1
        self._t_real += self.scale.dt_real
        self._inject_replay_enemies()
        self._publish_to_gcs()
        self.stats["survived"] = int(self.e_alive[0].sum())
        return self.get_frame()

    def reset(self, seed=None):
        super().reset(seed)
        self._have_gcs = False
        self._prev_pos = self.a_pos[0].copy()
        self._next_publish_wall = 0.0


__all__ = ["GcsBagCnnEnv"]
