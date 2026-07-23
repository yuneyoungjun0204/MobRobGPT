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
from boatattack_sim.env.config import RewardCfg
from boatattack_sim.model.cell_actor import load_cell_actor, cell_obs_to_torch

from .rl_bridge import CommandedDefenseEnv, build_battlefield_defense


class CommandedCellEnv(CommandedDefenseEnv):
    """LLM 배정을 존중하는 셀선택 RL 실행 환경(1월드). 경로/그물은 셀 정책이 기동."""

    def __init__(self, ckpt: str, enemy_mode: str = "diversionary", device: str = "cpu",
                 avoid_steer: bool | None = None, joint: bool = True,
                 specialized_root: str | None = None,
                 world_size: float | None = None):
        # ★ specialized_root 주면: 공격양상 기하분류 → 집중/양동/파상 특화 셀 정책 라우팅.
        #   (단일 ckpt 대신 3종 특화모델을 로드해 매 결정 대형에 맞는 모델로 추론)
        self._router = None
        if specialized_root:
            from .formation_router import SpecializedCellRouter
            self._router = SpecializedCellRouter(specialized_root, device=device)
            actor, cfg = self._router.actors["concentrated"], self._router.cfg   # cfg 원천 + 폴백 actor
        else:
            actor, cfg = load_cell_actor(ckpt, device=device)
        if not getattr(cfg, "cell_action", False):
            raise ValueError(f"{ckpt} 는 셀선택(cell_action) 정책이 아닙니다. "
                             f"잔차 정책이면 CommandedDefenseEnv 를 쓰세요.")
        # ★ 스케일 변환(선택): world_size 를 주면 모든 길이를 비례 축소해 그 크기 실험장으로.
        #   관측이 길이/길이 비율이라 정규화 입력이 수학적으로 동일 → 재학습 불필요.
        #   (다른 오버라이드보다 **먼저** 적용 — 이후 값들은 배율·개수라 순서 무관.)
        rcfg = RewardCfg()
        if world_size is not None:
            _s = float(world_size) / float(cfg.world_size)
            cfg.apply_scale(_s)
            rcfg.apply_scale(_s)     # 보상 길이항(영향반경·배정 cost[m])도 같이
        # ★ 체크포인트 자체 config(world·cell 격자·evade/weave 등) 유지 — 학습분포 충실.
        #   LLM 정합을 위해서만 최소 오버라이드.
        cfg.n_clusters = 3                     # LLM 이 최대 3그룹으로 다룸(셀 정책은 클러스터수 불변)
        cfg.spawn_phase_lo = 1.0               # 웨이브 텀 설계대로(스폰 랜덤 당김 끄기)
        cfg.transit_wp = cfg.cell_nets         # ★ WP 개수를 cell_nets와 일치시킴 (6→2)
        cfg.nets_per_ship = 3                  # ★ 그물 1개만 깔고 멈추던 문제 → 최대 3번 재전개.
        #   a_nets=3 으로 시작 → 그물 완성 후에도 a_nets>0 이라 동결(done) 안 됨 → 매 결정마다
        #   재배정·새 셀 선택·재전개(fresh route 로 leg_netted/paint_dist 리셋 = '기억 리셋').
        # ★ 전반적 속도 2배(아군·적군). enemy_speed 는 프로퍼티(=ally_speed×enemy_speed_mult)라
        #   ally_speed 만 2배로 하면 적 속도도 자동 2배 → 비율(2:3) 유지 = 요격 기하 동일, 기동만 빨라짐.
        #   회전반경 유지 위해 최대 선회각도도 함께 2배(안 그러면 넓게 돌아 요격점 오버슈트).
        cfg.ally_speed = float(getattr(cfg, "ally_speed", 6.0)) * 2.0        # 6→12 (적 9→18 자동)
        cfg.ally_max_turn = float(getattr(cfg, "ally_max_turn", 8.0)) * 2.0
        cfg.enemy_max_turn = float(getattr(cfg, "enemy_max_turn", 5.0)) * 2.0
        #   ★ 결정 주기도 절반(25→12)으로 → 결정당 이동거리를 1× 수준으로 유지(정책 학습분포 안).
        #   안 그러면 결정당 2배 이동해 오배치·요격 실패(특히 파상). '진짜 빨리감기' 효과.
        cfg.decision_period = max(1, int(round(int(getattr(cfg, "decision_period", 25)) / 2)))
        if avoid_steer is not None:
            cfg.avoid_steer = bool(avoid_steer)
        cfg.mother_keepout = True               # ★ APF는 꺼도 모선-전용 회피는 항상(모선 충돌 방지)
        # super().__init__ 가 _compute_assignment 를 부를 수 있으므로 속성 선주입.
        self._actor = actor
        self._device = device
        self.gain = 1.0                        # 셀 모델은 잔차배율 무의미(호환용)
        self._plan = None
        self._plan_command = None
        self._held: set[int] = set()
        self._joint = bool(joint)
        DefenseVecEnv.__init__(self, num_worlds=1, cfg=cfg, rcfg=rcfg, enemy_mode=enemy_mode)
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
        self._last_cells = None                # 시각화용 최근 선택 셀 [P,K]
        self._formation = None                 # 라우터가 고른 현재 대형(시각화/로그용)

    # ── LLM 계획 주입: 상위 _compute_assignment 재사용 + HOLD 집합 갱신 ──
    def _compute_assignment(self, assign_pref=None):
        super()._compute_assignment(assign_pref)            # _assign/_assignI 주입 + radius_adjust
        # HOLD 집합 = 계획의 hold_ships 중 '실제로 미배정(assign<0)'인 배만.
        #   → plan_to_assign 의 '최소 1대 활성' 보장으로 강제 배정된 배는 HOLD 제외(동결 안 함).
        if self._plan is not None:
            self._held = {int(i) for i in (self._plan.hold_ships or [])
                          if int(self._assign[0, int(i)]) < 0}
        else:
            self._held = set()

    # ── 후보셀에서 '이미 그물 깔린 곳' 제외 (행동공간에서 아예 배제) ──
    def _cell_valid_mask(self):
        """베이스 pruning 위에, net_installed(설치된 그물) 격자에 걸리는 후보셀을 무효화한다.
        → 재전개(최대 3회) 배가 기존/팀원 그물 위에 중복으로 다시 깔지 않고 새 위치로 커버 확대.
        전부 무효로 굶는 배는 원복(그물 위 아니면 어차피 안 굶음; 크래시 방지)."""
        base = super()._cell_valid_mask()                   # [N,P,C] True=무효
        ni = self.net_installed[0]
        if not ni.any():
            return base
        G = ni.shape[0]; cell = self.cfg.world_size / G
        ii, jj = np.where(ni)                               # 설치 그물 격자셀
        netxy = np.stack([(ii + 0.5) * cell, (jj + 0.5) * cell], axis=1)   # [M,2] 그물셀 중심
        cw = self.cell_world                                # [C,2]
        # ★ 250m 하한은 길이라 스케일을 곱해야 한다. 안 곱하면 33m 수조에서
        #   R=250m > 맵 전체 → 모든 후보셀이 배제돼 정책이 아무것도 못 고른다.
        R = max(float(self._cell_half()), 250.0 * float(getattr(self.cfg, "scale", 1.0)))
        # 각 후보셀 → 가장 가까운 그물셀 거리 < R 이면 무효(그물 깔린 곳)
        d2 = ((cw[:, None, 0] - netxy[None, :, 0]) ** 2
              + (cw[:, None, 1] - netxy[None, :, 1]) ** 2)   # [C,M]
        occ = d2.min(1) < (R * R)                           # [C]
        mask = base | occ[None, None, :]                    # 그 후보셀은 모든 배에게 무효
        short = (~mask).sum(2) < int(self.cfg.cell_nets)    # 유효셀 부족한 배 → 원복(굶음 방지)
        if short.any():
            mask = np.where(short[..., None], base, mask)
        return mask

    # ── 셀 정책 결정 + 셀 특유 오버라이드 ──
    def _rl_decide(self):
        if self._plan is None:                              # 명령 전 → 전원 제자리 정지
            self._compute_assignment()                      # _assign=-1(예비) 확정
            for p in range(self.P):
                self.route[0, p, :, :] = self.a_pos[0, p]
                self.net_mask[0, p, :] = False; self.doing_net[0, p] = False
            self._ev = self.fresh_ev()
            return
        # ★ 특화 라우팅: 공격양상 기하분류 → 그 대형 특화 정책 선택(없으면 단일 self._actor)
        if self._router is not None:
            self._formation = self._router.select(self)     # 'concentrated'|'diversionary'|'wave'
            actor = self._router.actors[self._formation]
        else:
            self._formation = None
            actor = self._actor
        obs = self.build_cell_obs()                         # 내부서 _compute_assignment(주입) 호출
        ot = cell_obs_to_torch(obs, self._device)
        with torch.no_grad():
            p, self._h = actor(ot, self._h)
            g = (actor.greedy_joint(p, self.N, self.P,
                                    cell_world=self.cell_world, mask_radius=self._mask_r)
                 if self._joint else actor.greedy(p))
        cells = g["cells"].view(self.N, self.P, -1).cpu().numpy()
        self._last_cells = cells[0].copy()                  # 시각화용(선택 셀)
        self._apply_actions({"cells": cells})
        self._apply_cell_overrides()                        # HOLD 정지 + deploy_net 끔
        self._ev = self.fresh_ev()

    def cell_viz(self):
        """UI 오버레이용 후보셀 데이터: 전체후보 / 배별유효 / 그물배제 / 선택셀."""
        ours = self._cell_valid_mask()[0]                       # net배제 포함(True=무효)
        base = DefenseVecEnv._cell_valid_mask(self)[0]          # net배제 전
        excluded = (~base) & ours                              # 유효였는데 그물로 배제
        return {
            "world": self.cell_world,
            "valid": [np.where(~ours[p])[0] for p in range(self.P)],
            "excluded": [np.where(excluded[p])[0] for p in range(self.P)],
            "selected": self._last_cells,
        }

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
