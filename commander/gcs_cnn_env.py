"""commander/gcs_cnn_env.py — GCS 실텔레메트리 아군(+ 적) 을 함께 쓰는 CNN 정책 환경.

세 조합이 있다:
  `GcsBagCnnEnv`  적=로스백(BagEnemyReplay) 재생,        아군=GCS 실텔레메트리(real)
  `GcsLiveCnnEnv` 적=GCS `/api/state`(role=target, real), 아군=GCS 실텔레메트리(real)

`GcsLiveCnnEnv`가 필요한 이유 (2026-09-10 세션에서 실측으로 확인된 결함) —
`commander/live_enemy_ros2.py::LiveEnemyReplay`(§`--enemy-source live`)는 raw ROS2 토픽
`/usv/usv{i}/pose`를 그대로 구독하는데, 이 값은 **로스백 자기 자신의 녹화-로컬 프레임**이지
GCS datum 기준이 아니다(`bag_enemy_relay.py`의 module docstring이 그렇게 명시한다: "local to
the bag's own recording"). 반면 아군은 `GcsAllyLink`를 통해 GCS `/api/state`에서 읽는데, 이
값은 GCS가 이미 `gcs.datum` 기준으로 정확히 재투영해서 보고하는 값이다. 두 좌표계가 다른데
같은 `enu_origin`을 나눠 쓰면(둘 다 `SimScale.enu_to_sim`에 그대로 들어간다) 적 위치가 아군
위치에서 수 km 어긋나 지도 밖으로 클리핑된다(실측: 로스백 raw pose ≈(-4.6,-1.1) vs 아군
centroid 기준 enu_origin ≈(4944,3445) → 적이 매 tick 지도 모서리에 찍힘).

GCS `/api/state`는 (`bag_enemy_relay.py --source pose`가 이미 registry datum 기준으로 재투영해
MAVLink로 보낸 값을 GCS가 다시 datum 기준 ned로 계산하므로) **이미 아군과 같은 좌표계**다.
그래서 `GcsLiveCnnEnv`는 로스백을 아예 거치지 않고, 적도 `GcsAllyLink`로 읽는다(대상
vehicle_id만 role=target 인 것들로 바꿔서) — 좌표계 불일치가 구조적으로 사라진다.

두 클래스 모두 매 tick, 그 배의 "현재 활성 경유점"(`route[ptr]`, ENU 미터로 변환)을 GCS
`/gcs`-호환 goto 명령으로 보낸다(source="rl") -- 이게 `docs/contracts.md` §4가 말하는
"RL이 0.5~2Hz로 계속 보낸다"는 그 지령이다. 그물 전개 시작/종료는 GCS로 보내지 않고
`NetDeploySink`로만 알린다(gcs_bridge.py 모듈독스트링 참고 -- 액추에이터 명령 금지).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from boatattack_sim.env import cnn_map as CM
from boatattack_sim.env.scaling import SimScale

from .bag_replay import BagEnemyReplay
from .gcs_bridge import GcsAllyLink, GcsRequestError, NetDeploySink, WaypointSink
from .replay_cnn_env import ReplayCnnEnv
from .unet_bridge import CommandedCnnEnv


class _GcsAllyTelemetryMixin:
    """GCS `/api/state` 아군 텔레메트리 수신 + WP 진행/그물 + goto 발행.

    적 소스(로스백 재생이냐 GCS `/api/state`냐)와 완전히 무관하다 -- 아군 쪽 로직은
    `GcsBagCnnEnv`/`GcsLiveCnnEnv` 양쪽에서 바이트 단위로 동일해야 하므로(둘 다 같은 실보트를
    같은 방식으로 관제한다) 믹스인으로 한 번만 정의한다. 이 믹스인을 쓰는 클래스는
    `self.ally_link`(`GcsAllyLink`), `self.net_sink`(`NetDeploySink`),
    `self.waypoint_sink`(`WaypointSink`), `self.scale`(`SimScale`),
    `self._publish_period_real`, `self._next_publish_wall`, `self._prev_pos`
    를 자기 `__init__`에서 준비해 둬야 한다.
    """

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

    # ── GCS로 현재 활성 경유점 송신 (0.5~2Hz 지속 스트림 계약) + 경유점 2개 묶음 발행 ──
    def _publish_to_gcs(self) -> None:
        now = time.monotonic()
        if now < self._next_publish_wall:
            return
        self._next_publish_wall = now + self._publish_period_real
        ptr_all = np.clip(self.ptr[0], 0, self.Kw - 1)
        for p in range(self.P):
            if not bool(self.a_alive[0, p]):
                continue                          # 위치를 모르는 배는 명령하지 않는다
            vid = self.ally_link.vehicle_ids[p]
            east, north = self.scale.sim_to_enu(self.route[0, p, ptr_all[p]])
            try:
                verdict = self.ally_link.submit_goto(vid, east=float(east), north=float(north))
            except GcsRequestError as exc:
                # 한 번의 TCP 타임아웃/거부로 전체 루프를 죽이지 않는다 -- gcs 의
                # hold->fade->release 가 짧은 공백을 흡수하도록 설계돼 있다
                # (docs/contracts.md §4). 다음 tick에서 다시 시도한다(아키텍트 검토 B2).
                print(f"[gcs_bridge] {vid} goto POST failed: {exc}")
                continue
            if not verdict.get("accepted", False):
                print(f"[gcs_bridge] {vid} goto refused: "
                      f"{verdict.get('reason')}: {verdict.get('detail', '')}")
            # goto(위 -- HTTP, 활성 경유점 1개)와 별개로, 이 배에 배정된 경유점 wp1+wp2를
            # 한 메시지로 묶어 ROS2에도 발행한다. `self.Kw`(=cfg.transit_wp)는 route 배열의
            # 최대 슬롯 수일 뿐 실제 배정과 다르다 -- RRT가 뽑은 실제 경로 길이 L이 Kw보다
            # 짧으면 남는 슬롯은 마지막 점을 그대로 반복해 채운다(defense_env.py::
            # apply_rrt_routes, "마지막 점 반복(도달 후 정지)"). 이 프로젝트의 "wp1/wp2"
            # 개념(RUN_GUIDE.md §12, gcs_cnn_env.py 모듈독스트링)은 늘 2개이므로, Kw 전체가
            # 아니라 앞 2개만 쓴다 -- 안 그러면 뒤쪽 중복 슬롯이 "서로 다른 경유점"인 것처럼
            # 구독자에게 잘못 보인다(실측: Kw=6인 체크포인트에서 뒤 4개가 wp2와 완전히 같은
            # 값으로 찍히는 걸 확인).
            waypoints_enu = [tuple(self.scale.sim_to_enu(self.route[0, p, k]))
                             for k in range(min(2, self.Kw))]
            # ptr_all은 self.Kw(패딩 포함 슬롯 수, 위에서 봤듯 실제론 6까지 감) 기준으로
            # 클립돼 있다 -- wp2 도착 후에도 arrived가 계속 True라 몇 micro-step 안에
            # ptr이 Kw-1까지 올라간다(패딩 슬롯도 "도착"으로 잡히므로). waypoints_enu는
            # 위에서 이미 앞 2개로 잘랐으므로, active_index도 그 길이에 맞춰 clamp해야
            # 구독자가 waypoints[active_index]를 그대로 인덱싱해도 IndexError가 안 난다.
            active_idx = min(int(ptr_all[p]), len(waypoints_enu) - 1)
            self.waypoint_sink.publish(vid, waypoints_enu, active_index=active_idx)

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


class GcsBagCnnEnv(_GcsAllyTelemetryMixin, ReplayCnnEnv):
    """적=로스백 리플레이(real), 아군=GCS 실텔레메트리(real)로 구동하는 CNN 점수맵 정책 환경."""

    def __init__(
        self,
        ckpt: str,
        bag: BagEnemyReplay,
        span_real: float,
        ally_link: GcsAllyLink,
        *,
        net_sink: Optional[NetDeploySink] = None,
        waypoint_sink: Optional[WaypointSink] = None,
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
        self.waypoint_sink = waypoint_sink or WaypointSink()
        if publish_hz <= 0:
            raise ValueError("publish_hz must be > 0")
        self._publish_period_real = 1.0 / float(publish_hz)
        self._have_gcs = False
        self._prev_pos = self.a_pos[0].copy()
        # 실시간(wall-clock) 기준 -- `_t_real`(시뮬 시각)로 재던 초판은 --no-realtime 이나
        # LLM 재배정으로 tick 이 밀리는 동안 스로틀이 같이 멈춰 GCS 를 과다/과소 호출했다
        # (아키텍트 검토 B3). publish 는 GCS 로 나가는 실제 명령이므로 실제 시계를 쓴다.
        self._next_publish_wall = 0.0

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


class GcsLiveCnnEnv(_GcsAllyTelemetryMixin, CommandedCnnEnv):
    """적·아군 모두 GCS `/api/state`(둘 다 real)로 구동하는 CNN 점수맵 정책 환경.

    `GcsBagCnnEnv`의 자매 클래스다 -- 로스백을 아예 안 쓴다. `enemy_link`는 role=target 으로
    등록된 GCS vehicle_id 들을 가리키는 `GcsAllyLink`(읽기만 쓴다, `submit_goto`는 절대
    안 부른다 -- 적을 지휘하지 않는다). GCS 가 이미 이 값들을 `gcs.datum` 기준으로 정확히
    재투영해서 주므로(`bag_enemy_relay.py --source pose` 가 그 경로), 아군과 완전히 같은
    좌표계를 공유한다 -- 이 파일 모듈독스트링의 좌표계 불일치 설명 참고.
    """

    def __init__(
        self,
        ckpt: str,
        span_real: float,
        ally_link: GcsAllyLink,
        enemy_link: GcsAllyLink,
        *,
        net_sink: Optional[NetDeploySink] = None,
        waypoint_sink: Optional[WaypointSink] = None,
        publish_hz: float = 2.0,
        ally_speed_real: float = 0.3,
        enemy_mode: str = "wave",
        device: str = "cpu",
        nets_per_ship: int = 3,
        geo: tuple[float, float] | None = None,
        enu_origin: tuple[float, float] | None = None,
        net_reload_period_real: float | None = None,
    ):
        super().__init__(ckpt, enemy_mode=enemy_mode, device=device, geo=geo,
                          nets_per_ship=nets_per_ship)
        self.reset(seed=0)                  # 배열 할당 + 육지 캐시 로드. 이후 물리는 안 돌린다
        if ally_link.n_allies != self.P:
            raise ValueError(
                f"ally_link has {ally_link.n_allies} vehicle_ids but the checkpoint "
                f"policy expects P={self.P} allies")
        if enemy_link.n_allies > self.M:
            raise ValueError(
                f"enemy_link has {enemy_link.n_allies} vehicle_ids but the checkpoint "
                f"policy only has M={self.M} enemy slots")
        self.ally_link = ally_link
        self.enemy_link = enemy_link
        self.net_sink = net_sink or NetDeploySink()
        self.waypoint_sink = waypoint_sink or WaypointSink()
        if publish_hz <= 0:
            raise ValueError("publish_hz must be > 0")
        self._publish_period_real = 1.0 / float(publish_hz)
        origin = (0.0, 0.0) if enu_origin is None else enu_origin
        self.scale = SimScale(self.cfg, span_real=float(span_real),
                              v_ally_real=float(ally_speed_real), enu_origin=origin)
        if net_reload_period_real is None:
            net_reload_period_real = max(1, self.cfg.nets_per_ship) * self.scale.period_real * 6.0
        self.net_reload_period_real = float(net_reload_period_real)
        if self.net_reload_period_real <= 0:
            raise ValueError("net_reload_period_real 은 0보다 커야 합니다.")
        self._t_real = 0.0
        self._next_reload_real = self._t_real + self.net_reload_period_real
        self._have_gcs = False
        self._prev_pos = self.a_pos[0].copy()
        self._next_publish_wall = 0.0
        self._last_span_warn_wall = 0.0
        # e_pos/e_hdg/e_alive 는 M 슬롯 고정(§9-⑨ 규약과 동일): enemy_link 척수(n)보다 많은
        # 나머지 슬롯은 절대 살아나지 않는다 -- reset() 이 부모의 스폰 로직으로 채워놓은
        # 값을 여기서 모두 지운다(적을 아예 '모른다'는 초기 상태로).
        self.e_pos[0] = 0.0
        self.e_hdg[0] = 0.0
        self.e_alive[0] = False

    # ── 상속받은 아군-전용 인제스트를 여기서는 쓰지 않는다 ──────────────
    def _ingest_gcs_allies(self):
        """`_GcsAllyTelemetryMixin`에서 상속되지만 이 클래스에서는 **호출하면 안 된다** --
        `/api/state`를 한 번 더 폴링하면서 적(`e_pos`)은 갱신 안 하고 방치한다(아키텍트
        검토 지적). `step()`은 반드시 아래 `_ingest_gcs()`(아군+적 한 번에)만 쓴다."""
        raise NotImplementedError(
            "GcsLiveCnnEnv 는 _ingest_gcs() 로 아군·적을 한 번에 읽는다 -- "
            "_ingest_gcs_allies() 를 따로 부르면 두 번째 /api/state 폴링이 낭비되고 "
            "적 관측이 갱신 안 된 채로 남는다.")

    # ── GCS 텔레메트리 주입: 아군·적 한 번의 /api/state 로 같이 갱신 ──────
    def _ingest_gcs(self) -> bool:
        """`/api/state`를 한 번만 읽어 아군·적 스냅샷 양쪽에 재사용한다
        (`GcsAllyLink.ally_snapshot`의 `state` 인자가 정확히 이 용도로 설계돼 있다 -- 같은
        GCS 인스턴스를 두 번 폴링할 이유가 없다). 실패하면 아군·적 둘 다 마지막 상태
        유지(§B1과 동일 원칙, 적에도 똑같이 적용). `_ingest_gcs_allies`(믹스인)와 마찬가지로
        `Exception` 전체를 잡는다 -- malformed 응답(예: `ned` 필드 누락)이 여기서 새어나가면
        run_gcs_bridge.py 의 결정 루프 전체가 죽는다.
        """
        try:
            state = self.ally_link.client.get_json("/api/state")
        except Exception as exc:                       # pragma: no cover -- network edge
            print(f"[gcs_bridge] /api/state poll failed: {exc}")
            return False
        lo = CM.origin(self.cfg)
        hi = lo + 2.0 * CM.extent_m(self.cfg)

        ally = self.ally_link.ally_snapshot(state)
        if ally.alive.any():
            raw = self.scale.enu_to_sim(ally.pos[ally.alive])
            self._warn_if_clipped("아군", raw, lo, hi)
            self.a_pos[0][ally.alive] = np.clip(raw, lo, hi)
            self.a_hdg[0][ally.alive] = ally.hdg[ally.alive]
        self.a_alive[0] = ally.alive

        enemy = self.enemy_link.ally_snapshot(state)
        n = self.enemy_link.n_allies
        if enemy.alive.any():
            raw = self.scale.enu_to_sim(enemy.pos[enemy.alive])
            self._warn_if_clipped("적", raw, lo, hi)
            self.e_pos[0, :n][enemy.alive] = np.clip(raw, lo, hi)
            self.e_hdg[0, :n][enemy.alive] = enemy.hdg[enemy.alive]
        self.e_alive[0, :n] = enemy.alive
        return True

    def _warn_if_clipped(self, label: str, raw: np.ndarray, lo: float, hi: float) -> None:
        """`--span`이 실제 함대 퍼짐보다 작으면 sim 좌표가 지도 밖으로 나가 조용히
        클리핑된다 -- 2026-09-10 세션의 그 버그(적이 지도 모서리에 찍힘)를 잡아낸 신호가
        바로 이거였다(`replay_cnn_env.py`의 --span 경고와 같은 역할, 여기서는 사전
        추정치가 아니라 매 tick 실측을 본다). 5초에 한 번으로 스로틀."""
        out = (raw < lo) | (raw > hi)
        if not out.any():
            return
        now = time.monotonic()
        if now - self._last_span_warn_wall < 5.0:
            return
        self._last_span_warn_wall = now
        n_out = int(out.any(axis=-1).sum())
        print(f"[gcs_bridge] ⚠ {label} {n_out}척이 --span {self.scale.span_real:g} m 박스 "
              f"밖 -- 지도 모서리로 클리핑되고 있습니다. --span 을 키우거나 --enu-origin 을 "
              f"조정하세요.")

    # ── 운용 루프 (GcsBagCnnEnv.step 과 동형, 적 주입만 로스백 대신 GCS) ──
    def step(self):
        if bool(self.done[0]):
            self.t[0] = 0
            self.done[0] = False
        if self._t_real >= self._next_reload_real:
            self.a_nets[0] = self.cfg.nets_per_ship
            self._next_reload_real += self.net_reload_period_real

        if not self._ingest_gcs():
            return self.get_frame()                # 텔레메트리 미수신 -- 마지막 상태 유지
        if not self._have_gcs:
            if not bool(self.a_alive[0].all()):
                # GcsBagCnnEnv 와 동일 원칙(§S5) -- 아군 전원이 보이기 전엔 결정/발행을
                # 시작하지 않는다. 적은 "몇 척 보이는지"를 요구하지 않는다 -- 적이 아직
                # 하나도 안 보여도 아군은 대형을 갖출 수 있어야 한다.
                return self.get_frame()
            self._have_gcs = True
            self._prev_pos = self.a_pos[0].copy()

        if self._micro_ct % self.cfg.decision_period == 0:
            self._rl_decide()
            self._sprev = {k: 0.0 for k in self._SK}
        self._advance_and_paint()
        for k in self._SK:
            cur = float(self._ev[k][0]) if self._ev is not None else 0.0
            self.stats[k] += cur - self._sprev.get(k, 0.0)
            self._sprev[k] = cur
        self._micro_ct += 1
        self._t_real += self.scale.dt_real
        self._publish_to_gcs()
        self.stats["survived"] = int(self.e_alive[0].sum())
        return self.get_frame()

    def reset(self, seed=None):
        super().reset(seed)
        self.e_pos[0] = 0.0
        self.e_hdg[0] = 0.0
        self.e_alive[0] = False
        self._have_gcs = False
        self._prev_pos = self.a_pos[0].copy()
        self._next_publish_wall = 0.0
        self._last_span_warn_wall = 0.0
        self._t_real = 0.0
        # `__init__` 이 net_reload_period_real 을 세팅하기 *전에* 배열 할당용으로 한 번
        # reset(seed=0) 을 부르므로(위 __init__ 참고), 그 첫 호출에서는 이 속성이 아직 없다.
        if hasattr(self, "net_reload_period_real"):
            self._next_reload_real = self._t_real + self.net_reload_period_real

    @property
    def bag_time_real(self) -> float:
        """run_gcs_bridge.py 의 결정 루프가 로그·재배정 타임스탬프에 쓰는 duck-type 필드
        (`GcsBagCnnEnv`/`ReplayCnnEnv`와 이름을 맞춘다 -- 실제로는 '로스백 시각'이 아니라
        이 환경이 GCS 연동을 시작한 이후 경과한 실초다)."""
        return self._t_real

    def bag_exhausted(self) -> bool:
        """GCS 라이브 스트림은 절대 소진되지 않는다(`LiveEnemyReplay.duration_sec=inf`와
        같은 계약) -- 종료는 호출부의 `--max-decisions`나 Ctrl+C 몫이다."""
        return False


__all__ = ["GcsBagCnnEnv", "GcsLiveCnnEnv"]
