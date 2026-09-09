"""commander/replay_cnn_env.py — 로스백 실제 적선 + 가상 시뮬 아군 혼합 U-Net 정책 환경.

`ros2_unet_env.ROS2CnnEnv` 의 자매 클래스다. ROS2CnnEnv 는 아군까지 실센서로 채우지만,
여기서는 아군 실측이 없으므로(가상 생성) **아군은 건드리지 않는다** — `CommandedCnnEnv`가
상속하는 `DefenseVecEnv`의 기존 스폰 + PD-follow 물리가 그대로 아군을 가상 기동시킨다
(재구현 없음). 매 tick 끝에서 적(`e_pos`/`e_hdg`/`e_alive`)만 `BagEnemyReplay` 실데이터로
덮어써 "실제(적) + 가상(아군)" 혼합을 만든다.

맵 스케일은 체크포인트 cfg(world_size 등)를 전혀 건드리지 않고 `SimScale(span_real=...)`
로만 좌표를 옮긴다(문서 `docs/unet_model_deploy.md` §10~11) — 로스백이 8 m 짜리든 33 m
짜리든 같은 코드가 그대로 동작한다.
"""
from __future__ import annotations

import numpy as np

from boatattack_sim.env import cnn_map as CM
from boatattack_sim.env.scaling import SimScale

from .bag_replay import BagEnemyReplay
from .unet_bridge import CommandedCnnEnv


class ReplayCnnEnv(CommandedCnnEnv):
    """적=로스백 리플레이(real), 아군=시뮬 물리(virtual)로 구동하는 CNN 점수맵 정책 환경(1월드)."""

    def __init__(
        self,
        ckpt: str,
        bag: BagEnemyReplay,
        span_real: float,
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
        self.reset(seed=0)                  # 배열 할당 + 육지 캐시 로드. 이후 적 물리는 안 돌린다
        self.bag = bag
        # 모선 위치가 별도로 없으므로, 관측된 적 전체의 중심을 기본 원점(=맵 중앙)으로 삼는다
        # (§10 anchor 규약: enu_origin 은 맵 정중앙에 대응하는 실좌표).
        if enu_origin is not None:
            origin = enu_origin
        else:
            origin = bag.centroid()
            print(f"[replay] ⚠ --enu-origin 미지정 — 적선 전체 평균 위치 {tuple(round(c, 3) for c in origin)} "
                  f"를 모선(방어 중심)으로 임시 사용합니다(실제 모선 좌표를 모르므로 근사). "
                  f"실제 방어 목표 좌표를 알면 --enu-origin X Y 로 지정하세요.")
        self.scale = SimScale(self.cfg, span_real=float(span_real),
                              v_ally_real=float(ally_speed_real), enu_origin=origin)

        max_r = bag.max_radius_from(origin)
        need = 2.0 * max_r
        if need > self.scale.span_real:
            print(f"[replay] ⚠ --span {self.scale.span_real:g} m 가 로스백 실측 범위"
                  f"(원점에서 최대 {max_r:.2f} m, 필요 ≥{need:.2f} m)보다 작습니다 — "
                  f"경계 밖 적은 관측(픽셀 클립)과 배정(실좌표) 기하가 서로 어긋납니다. "
                  f"--span 을 키우거나 --enu-origin 을 조정하세요.")

        # 척마다 pose 발행 시작 시각이 몇 ms 씩 어긋난다 — 전 척이 다 켜진 시점부터 시작해야
        # 첫 결정이 일부 척 누락 상태로 굳어지지 않는다(§F3).
        self._t_real = bag.first_common_time_sec
        # 그물 재보급을 실제 경과초로 독립적으로 관리한다(§F1 후속) — cfg.max_steps(RL
        # 에피소드 길이, 학습 아티팩트일 뿐 운용과 무관)에 얹으면 "13결정 만에 소진 → 67결정
        # 방치" 처럼 재보급 주기가 --span/--nets 와 우연히 맞을 때만 쓸만해진다. 실제 초 단위
        # 주기를 독립 타이머로 두면 --nets/--net-reload-period 로 직접 튜닝 가능하다.
        # 명시적으로 안 주면 "그물 nets_per_ship 장을 다 쓰는 데 대략 6결정 걸린다"는 실측
        # 관찰(§F1 재검토)에 여유를 둔 기본값을 결정주기에서 유도한다 — 20 초 같은 고정
        # 상수는 --span 이 바뀌면 결정주기(period_real)도 바뀌어 의미가 어긋난다.
        if net_reload_period_real is None:
            net_reload_period_real = max(1, self.cfg.nets_per_ship) * self.scale.period_real * 6.0
        self.net_reload_period_real = float(net_reload_period_real)
        if self.net_reload_period_real <= 0:
            raise ValueError("net_reload_period_real 은 0보다 커야 합니다.")
        self._next_reload_real = self._t_real + self.net_reload_period_real
        self._inject_replay_enemies()

    # ── 로스백 → sim 상태 주입 ─────────────────────────────────────────
    def _inject_replay_enemies(self) -> None:
        """현재 `_t_real` 시각의 로스백 적선 상태로 e_pos/e_hdg/e_alive 를 덮어쓴다.

        M 슬롯 고정(§9-⑨ 규약): 로스백 척수가 M 보다 적으면 남는 슬롯은 e_alive=False,
        많으면 앞 M 개만 사용한다(척수가 보통 훨씬 적으므로 위협 정렬은 생략).
        """
        xy, hdg_deg, alive = self.bag.sample(self._t_real)
        M = self.M
        n = min(len(alive), M)
        self.e_pos[0] = 0.0
        self.e_hdg[0] = 0.0
        self.e_alive[0] = False
        # world_to_pix 가 어차피 점수맵 경계(center ± cnn_extent, u-net_map.pt 는 이게
        # world_size 와 우연히 같다)로 클립하므로, 여기서 e_pos 자체를 같은 경계로 클립해
        # raster 관측과 클러스터링/배정(둘 다 e_pos 를 직접 읽는다)이 항상 같은 좌표를 보게
        # 한다 — 안 그러면 지도 밖 적이 관측·배정에서 서로 다른 위치로 취급된다(§F2).
        # `[0, world_size]`를 그대로 쓰지 않고 cnn_map 에서 유도하는 이유: 다른 체크포인트는
        # `cnn_extent < world_size/2` 이거나 `center`가 맵 중앙이 아닐 수 있다.
        lo = CM.origin(self.cfg)
        hi = lo + 2.0 * CM.extent_m(self.cfg)
        self.e_pos[0, :n] = np.clip(self.scale.enu_to_sim(xy[:n]), lo, hi)
        self.e_hdg[0, :n] = hdg_deg[:n]
        self.e_alive[0, :n] = alive[:n]

    # ── 운용 루프 ────────────────────────────────────────────────────
    def step(self):
        """`CommandedDefenseEnv.step()` 과 동일한 결정+micro 루프 + 매 tick 끝 적 리플레이 주입.

        아군 물리(`_micro`)는 그대로 두되(가상 기동), `_micro` 내부에서 매 micro-step 스스로
        전진시키는 적을 매 tick 끝에서 실데이터로 덮어써 다음 결정/관측이 실제 적 위치를 보게
        한다(문서 §10 tick 레시피와 동일한 '센서로 상태 덮어쓰기' 패턴).

        `CommandedDefenseEnv.step()`의 "done → 재스폰"(`_spawn_worlds`) 분기는 여기서 쓰지
        않는다: `done`은 주로 `t >= cfg.max_steps`(RL 에피소드 길이 상한 — 로스백 전체 재생
        시간보다 훨씬 짧다) 때문에 켜지는데, 아군을 통째로 재스폰하면 가상 임무가 로스백
        중간에 끊긴다. 대신 `t`만 되감아 계속 잇는다 — 종료 시점은 `bag_exhausted()`가 결정.

        그물 재보급은 `done`(=`max_steps`, 학습용 에피소드 길이일 뿐 실제 시간과 무관한
        아티팩트)에 얹지 않고 `net_reload_period_real`(실제 초) 독립 타이머로 처리한다
        (아래) — 그렇지 않으면 재보급 주기가 `--span`에 따라 우연히 정해져, 그물을 몇
        결정 만에 다 쓰고 나머지를 방치하는 경우가 생긴다(§F1 후속 조치).

        재보급은 **`a_nets`만** 채운다. `doing_net`을 여기서 같이 끄면 마침 부설 중이던
        그물이 `_micro`의 완성 경로(`finish = doing_net & (...)`, defense_env.py)를 영영
        못 타 `net_installed`에 등록되지 않고 그냥 사라진다(그물은 이미 소비됐는데 실체가
        없는 것) — `_spawn_worlds`가 재스폰 때 다섯 개 필드(ptr·leg_netted·paint_dist·
        net_start·net_end 등)를 한꺼번에 리셋하는 것과 달리 여기선 그중 하나만 건드리므로
        어설프게 흉내내지 않는다. `a_alive`도 여기서 건드리지 않는다 — 충돌로 죽은 배를
        타이머가 그 자리에서 되살리면 "영구 손실" 이라는 불변식이 깨진다.
        """
        if bool(self.done[0]):
            self.t[0] = 0
            self.done[0] = False
        if self._t_real >= self._next_reload_real:
            self.a_nets[0] = self.cfg.nets_per_ship
            self._next_reload_real += self.net_reload_period_real
        if self._micro_ct % self.cfg.decision_period == 0:
            self._rl_decide()
            self._sprev = {k: 0.0 for k in self._SK}
        self._micro(self._ev)
        for k in self._SK:
            cur = float(self._ev[k][0])
            self.stats[k] += cur - self._sprev[k]
            self._sprev[k] = cur
        self._micro_ct += 1
        self._t_real += self.scale.dt_real
        self._inject_replay_enemies()
        self.stats["survived"] = int(self.e_alive[0].sum())
        return self.get_frame()

    def reset(self, seed=None):
        super().reset(seed)
        if hasattr(self, "bag"):
            self._t_real = self.bag.first_common_time_sec
            self._next_reload_real = self._t_real + self.net_reload_period_real
            self._inject_replay_enemies()
        else:
            self._t_real = 0.0

    @property
    def bag_time_real(self) -> float:
        return self._t_real

    def bag_exhausted(self) -> bool:
        return self._t_real > self.bag.duration_sec


__all__ = ["ReplayCnnEnv"]
