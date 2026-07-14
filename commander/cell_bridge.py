"""LLM 지휘관 ↔ '셀선택' RL 정책(CellPointerActor) 실행 브릿지.

run_commander_ui.py --rl --cell 에서 사용. 잔차(wp_delta) 모델을 쓰는 CommandedDefenseEnv 와
동일 인터페이스(set_plan/step/get_frame)지만, 경로 기동을 **이산 셀선택 정책**이 담당한다.

핵심 합성(잔차 모델과 완전히 동일한 주입 지점):
  LLM(deployments: 클러스터→ally_ids, deploy_net, radius_adjust, hold)
    → plan_to_assign → _assign / _assignI 주입(_compute_assignment; 상위 클래스 재사용)
    → 셀 정책이 이 배정을 그대로 관측·존중:
        · build_cell_obs 의 own[9] = ... + 배정요격점2·할당플래그1  (소프트 유도)
        · _cell_valid_mask 가 배정 요격점 기준으로 후보셀 pruning(각도게이트+Voronoi disjoint)
          → 각 배는 '자기 클러스터 섹터' 셀만 고름 (하드 게이팅)
        · radius_adjust 로 _assignI 반경 이동 → pruning 중심 이동(더 멀리/가까이 요격)
  셀 정책은 이 '배정-조건부 pruning'으로 학습됐으므로 LLM 배정 주입은 학습분포 안(충실).

셀 특유 처리:
  · HOLD(hold_ships): 배정 -1 만으로는 셀선택이 멈추지 않으므로(잔차와 다름) 명시적으로
    경로를 제자리로 접고 그물 끔.
  · deploy_net=false: 요격 위치로 이동하되 그물 안 깖(net_mask 끔).
  · net_legs: 셀 모델엔 '레그' 개념이 없어(셀 K개가 곧 벽) 무시.
  · 명령 전(_plan None): 전원 제자리 정지(셀 미적용).
"""
from __future__ import annotations

import numpy as np
import torch

from boatattack_sim.env.defense_env import DefenseVecEnv
from boatattack_sim.model.cell_actor import load_cell_actor, cell_obs_to_torch

from .rl_bridge import CommandedDefenseEnv, build_battlefield_defense


class CommandedCellEnv(CommandedDefenseEnv):
    """LLM 배정을 존중하는 셀선택 RL 실행 환경(1월드). 경로/그물은 셀 정책이 기동."""

    def __init__(self, ckpt: str, enemy_mode: str = "diversionary", device: str = "cpu",
                 avoid_steer: bool | None = None, joint: bool = True):
        actor, cfg = load_cell_actor(ckpt, device=device)
        if not getattr(cfg, "cell_action", False):
            raise ValueError(f"{ckpt} 는 셀선택(cell_action) 정책이 아닙니다. "
                             f"잔차 정책이면 CommandedDefenseEnv 를 쓰세요.")
        # ★ 체크포인트 자체 config(world·cell 격자·evade/weave 등) 유지 — 학습분포 충실.
        #   LLM 정합을 위해서만 최소 오버라이드.
        cfg.n_clusters = 3                     # LLM 이 최대 3그룹으로 다룸(셀 정책은 클러스터수 불변)
        cfg.spawn_phase_lo = 1.0               # 웨이브 텀 설계대로(스폰 랜덤 당김 끄기)
        cfg.nets_per_ship = 3                  # ★ 그물 1개만 깔고 멈추던 문제 → 최대 3번 재전개.
        #   a_nets=3 으로 시작 → 그물 완성 후에도 a_nets>0 이라 동결(done) 안 됨 → 매 결정마다
        #   재배정·새 셀 선택·재전개(fresh route 로 leg_netted/paint_dist 리셋 = '기억 리셋').
        if avoid_steer is not None:
            cfg.avoid_steer = bool(avoid_steer)
        # super().__init__ 가 _compute_assignment 를 부를 수 있으므로 속성 선주입.
        self._actor = actor
        self._device = device
        self.gain = 1.0                        # 셀 모델은 잔차배율 무의미(호환용)
        self._plan = None
        self._plan_command = None
        self._held: set[int] = set()
        self._joint = bool(joint)
        DefenseVecEnv.__init__(self, num_worlds=1, cfg=cfg, enemy_mode=enemy_mode)
        self._actor.eval()
        self._h = self._actor.init_hidden(self.P, device)   # cell_recurrent=False 면 None
        self._mask_r = float(self._cell_half())             # joint 교차잠금 반경(격자 반칸)
        self._micro_ct = 0
        self._ev = None
        self.running = True
        self._SK = ("captures", "breaches", "ally_collisions", "nets_used")
        self.stats = {k: 0 for k in self._SK + ("survived",)}
        self._sprev = {k: 0.0 for k in self._SK}
        self.resolve_conflicts = False         # 셀 정책 greedy_joint 가 교차잠금 담당
        self._cmd_deploy = np.ones(self.P, bool)
        self._cmd_net_legs = [None] * self.P   # 셀 모델 미사용(호환용)

    # ── LLM 계획 주입: 상위 _compute_assignment 재사용 + HOLD 집합 갱신 ──
    def _compute_assignment(self, assign_pref=None):
        super()._compute_assignment(assign_pref)            # _assign/_assignI 주입 + radius_adjust
        self._held = ({int(i) for i in (self._plan.hold_ships or [])}
                      if self._plan is not None else set())

    # ── 셀 정책 결정 + 셀 특유 오버라이드 ──
    def _rl_decide(self):
        if self._plan is None:                              # 명령 전 → 전원 제자리 정지
            self._compute_assignment()                      # _assign=-1(예비) 확정
            for p in range(self.P):
                self.route[0, p, :, :] = self.a_pos[0, p]
                self.net_mask[0, p, :] = False; self.doing_net[0, p] = False
            self._ev = self.fresh_ev()
            return
        obs = self.build_cell_obs()                         # 내부서 _compute_assignment(주입) 호출
        ot = cell_obs_to_torch(obs, self._device)
        with torch.no_grad():
            p, self._h = self._actor(ot, self._h)
            g = (self._actor.greedy_joint(p, self.N, self.P,
                                          cell_world=self.cell_world, mask_radius=self._mask_r)
                 if self._joint else self._actor.greedy(p))
        cells = g["cells"].view(self.N, self.P, -1).cpu().numpy()
        self._apply_actions({"cells": cells})
        self._apply_cell_overrides()                        # HOLD 정지 + deploy_net 끔
        self._ev = self.fresh_ev()

    def _apply_cell_overrides(self):
        """셀 정책 적용 뒤 LLM 의 HOLD/deploy_net 을 반영(net_legs 는 셀 모델 미사용).

        ★ NET LOCK-IN: 이미 그물을 전개 중(doing_net=True)인 배는 HOLD/deploy_net=false 로도
        절대 중단하지 않는다 — 중단하면 그물이 '설치하다 미완성 정지'로 낭비된다(그런 케이스 0).
        전개중 배는 완성될 때까지 건드리지 않고, 아직 시작 안 한 배에만 HOLD/미투척을 적용."""
        w = 0
        for p in range(self.P):
            if bool(self.doing_net[w, p]):                  # 전개중 → lock-in(중단 금지)
                continue
            if p in self._held:                             # HOLD: 제자리 정지 + 그물 끔
                self.route[w, p, :, :] = self.a_pos[w, p]
                self.net_mask[w, p, :] = False; self.doing_net[w, p] = False
            elif int(self._assign[w, p]) >= 0 and not self._cmd_deploy[p]:
                self.net_mask[w, p, :] = False              # 이동만, 그물 안 깖
                self.doing_net[w, p] = False


__all__ = ["CommandedCellEnv", "build_battlefield_defense"]
