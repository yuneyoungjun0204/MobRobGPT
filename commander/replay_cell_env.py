"""commander/replay_cell_env.py — 로스백 실제 적선 + 가상 시뮬 아군 혼합 셀선택 정책 환경.

`replay_cnn_env.ReplayCnnEnv` 의 셀선택(CellPointerActor) 판이다. 같은 원칙을 따른다 —
아군은 실측이 없으므로(가상 생성) **건드리지 않는다**: `CommandedCellEnv`가 상속하는
`DefenseVecEnv`의 기존 스폰 + PD-follow 물리가 그대로 아군을 가상 기동시킨다(재구현 없음).
매 tick 끝에서 적(`e_pos`/`e_hdg`/`e_alive`)만 `BagEnemyReplay` 실데이터로 덮어써
"실제(적) + 가상(아군)" 혼합을 만든다.

셀 모델과 CNN 모델의 차이(`commander/cell_bridge.py`가 이미 처리, 여기선 손대지 않는다):
  · `CommandedCellEnv.__init__` 이 자체적으로 ally_speed·decision_period 등을 배 스케일링한다
    (학습분포 유지 목적) — `ReplayCellEnv`는 이 cfg 조정 이후의 `self.scale`만 새로 얹는다.
  · `specialized_root` 를 주면 대형(집중/양동/파상) 기하분류 라우팅이 자동으로 켜진다
    (`commander/formation_router.py::SpecializedCellRouter`) — 이 클래스는 그 인자를 그대로
    `CommandedCellEnv`에 전달할 뿐, 라우팅 로직을 재구현하지 않는다.

맵 스케일은 체크포인트 cfg(world_size 등)를 전혀 건드리지 않고 `SimScale(span_real=...)`
로만 좌표를 옮긴다 — `ReplayCnnEnv`와 동일한 계약.
"""
from __future__ import annotations

import numpy as np

from boatattack_sim.env.scaling import SimScale

from .bag_replay import BagEnemyReplay
from .cell_bridge import CommandedCellEnv


class ReplayCellEnv(CommandedCellEnv):
    """적=로스백 리플레이(real), 아군=시뮬 물리(virtual)로 구동하는 셀선택 정책 환경(1월드)."""

    def __init__(
        self,
        ckpt: str,
        bag: BagEnemyReplay,
        span_real: float,
        ally_speed_real: float = 0.3,
        enemy_mode: str = "wave",
        device: str = "cpu",
        specialized_root: str | None = None,
        enu_origin: tuple[float, float] | None = None,
        net_reload_period_real: float | None = None,
    ):
        super().__init__(ckpt, enemy_mode=enemy_mode, device=device, specialized_root=specialized_root)
        self.reset(seed=0)                  # 배열 할당 + 육지 캐시 로드. 이후 적 물리는 안 돌린다
        self.bag = bag
        if enu_origin is not None:
            origin = enu_origin
        else:
            origin = bag.centroid()
            print(f"[replay] ⚠ --enu-origin 미지정 — 적선 전체 평균 위치 {tuple(round(c, 3) for c in origin)} "
                  f"를 모선(방어 중심)으로 임시 사용합니다(실제 모선 좌표를 모르므로 근사). "
                  f"실제 방어 목표 좌표를 알면 --enu-origin X Y 로 지정하세요.")
        # ★ CommandedCellEnv.__init__ 이 이미 ally_speed 를 2배(등)로 스케일했다 — 그 결과인
        #   self.cfg 를 그대로 SimScale 이 받는다(cfg 를 다시 건드리지 않는다).
        self.scale = SimScale(self.cfg, span_real=float(span_real),
                              v_ally_real=float(ally_speed_real), enu_origin=origin)

        max_r = bag.max_radius_from(origin)
        need = 2.0 * max_r
        if need > self.scale.span_real:
            print(f"[replay] ⚠ --span {self.scale.span_real:g} m 가 로스백 실측 범위"
                  f"(원점에서 최대 {max_r:.2f} m, 필요 ≥{need:.2f} m)보다 작습니다 — "
                  f"경계 밖 적은 관측(래스터/클러스터링)과 배정(실좌표) 기하가 서로 어긋납니다. "
                  f"--span 을 키우거나 --enu-origin 을 조정하세요.")

        self._t_real = bag.first_common_time_sec
        if net_reload_period_real is None:
            net_reload_period_real = max(1, self.cfg.nets_per_ship) * self.scale.period_real * 6.0
        self.net_reload_period_real = float(net_reload_period_real)
        if self.net_reload_period_real <= 0:
            raise ValueError("net_reload_period_real 은 0보다 커야 합니다.")
        self._next_reload_real = self._t_real + self.net_reload_period_real
        self._inject_replay_enemies()

    # ── 로스백 → sim 상태 주입 (ReplayCnnEnv._inject_replay_enemies 와 동일 로직) ──
    def _inject_replay_enemies(self) -> None:
        xy, hdg_deg, alive = self.bag.sample(self._t_real)
        M = self.M
        n = min(len(alive), M)
        self.e_pos[0] = 0.0
        self.e_hdg[0] = 0.0
        self.e_alive[0] = False
        # 셀 모델은 CNN 래스터 경계 클립이 필요 없다(고정 20×20 카르테시안 후보셀 + 환형
        # 필터일 뿐 관측이 픽셀 격자로 잘리지 않는다) — world_size 경계로만 클립해 물리 격자
        # (net_installed 등, 200×200)와 좌표를 맞춘다.
        self.e_pos[0, :n] = np.clip(self.scale.enu_to_sim(xy[:n]), 0.0, self.cfg.world_size)
        self.e_hdg[0, :n] = hdg_deg[:n]
        self.e_alive[0, :n] = alive[:n]

    # ── 운용 루프 (ReplayCnnEnv.step 과 동일 원칙) ──────────────────────
    def step(self):
        """결정+micro 루프 + 매 tick 끝 적 리플레이 주입. 상세 근거는 `ReplayCnnEnv.step` 참고."""
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


__all__ = ["ReplayCellEnv"]
