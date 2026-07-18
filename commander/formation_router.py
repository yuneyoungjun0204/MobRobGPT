"""공격 양상(포메이션) 기하 분류 + 대형 특화 셀 정책 라우팅.

관측된 적 공간분포로 현재 공격이 집중/양동/파상 중 무엇인지 **기하학적으로 판정**하고,
그 대형에 특화 학습된 셀 정책(30_model/<mode>/best.pt)을 골라 추론에 쓴다.

왜 기하인가 (LLM 아님): 포메이션 분류는 적 위치의 각/반경 분포로 결정되는 명확한 3분류라
  - 비용 0(마이크로초)·완전 결정적(같은 상태→같은 판정, 변동 0)
  - world_mode 정답 대비 ~100% 정확 (feature 분리도 우수: 아래 임계 참고)
반면 LLM 분류는 매 결정 호출비용 + 미세 변동 위험이 있어, 이 crisp 한 작업엔 과하다.

기하 특징 (모선 중심 상대):
  R       = 적 방위의 원형 평균결과길이 (1=한 방향 집중, 0=전방위 분산)
  rspread = 적 반경의 상대 표준편차 (파상=여러 랭크 → 큼)
  nclu    = 각 클러스터 수 (양동/파상=여러, 집중=1)
판정: R<R_DIV → diversionary / (rspread>RS_WAVE or nclu>=2) → wave / else concentrated
"""
from __future__ import annotations

import os
import numpy as np
import torch

from boatattack_sim.env import clustering
from boatattack_sim.model.cell_actor import load_cell_actor, cell_obs_to_torch

MODES = ("concentrated", "diversionary", "wave")

# 분류 임계 (검증된 feature 분포 기준: 집중 R=1.0/rs=0.04, 양동 R=0.02, 파상 R=0.97/rs=0.24)
R_DIV = 0.55        # R 이보다 낮으면 = 전방위 분산 = 양동
RS_WAVE = 0.12      # 각 집중인데 반경 퍼짐 크면 = 랭크 = 파상


def classify_formation(env, w: int = 0) -> str:
    """월드 w 의 현재 적 분포 → 'concentrated' | 'diversionary' | 'wave' (기하 결정적)."""
    al = env.e_alive[w]
    if int(al.sum()) == 0:
        return "concentrated"                                   # 적 없음 → 기본
    c = env.center
    dx = env.e_pos[w, al, 0] - c[0]
    dy = env.e_pos[w, al, 1] - c[1]
    ang = np.arctan2(dx, dy)
    rad = np.hypot(dx, dy)
    R = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))     # 각 집중도
    rspread = float(rad.std() / max(rad.mean(), 1.0))               # 반경 상대 퍼짐
    cl = clustering.cluster_by_gaps_vec(env.e_pos, env.e_alive, env.e_hdg, c,
                                        env.cfg.enemy_speed, 4, env.cfg.cluster_gap_deg)
    nclu = int(cl["active"][w].sum())
    if R < R_DIV:                                               # 전방위 분산
        return "diversionary"
    if rspread > RS_WAVE or nclu >= 2:                          # 각 집중 + 여러 랭크
        return "wave"
    return "concentrated"                                        # 각 집중 + 단일 랭크


class SpecializedCellRouter:
    """대형 특화 셀 정책 3종을 로드하고, 분류 결과에 맞는 정책으로 추론(greedy_joint)."""

    def __init__(self, root: str, device: str = "cpu"):
        self.device = device
        self.actors = {}
        self.cfg = None
        for m in MODES:
            path = os.path.join(root, m, "best.pt")
            actor, cfg = load_cell_actor(path, device=device)
            actor.eval()
            self.actors[m] = actor
            self.cfg = cfg                                       # 3종 동일 구조 가정
        self._last_mode = None

    def select(self, env, w: int = 0) -> str:
        self._last_mode = classify_formation(env, w)
        return self._last_mode

    @torch.no_grad()
    def act(self, env, mask_radius: float = 0.0):
        """현재 대형 분류 → 해당 특화 정책으로 셀 선택(greedy_joint). 반환 cells[N,P,K]."""
        mode = self.select(env)
        actor = self.actors[mode]
        p, _ = actor(cell_obs_to_torch(env.build_cell_obs(), self.device))
        if mask_radius > 0:
            g = actor.greedy_joint(p, env.N, env.P, cell_world=env.cell_world, mask_radius=mask_radius)
        else:
            g = actor.greedy(p)
        return g["cells"].view(env.N, env.P, -1).cpu().numpy(), mode


__all__ = ["classify_formation", "SpecializedCellRouter", "MODES"]
