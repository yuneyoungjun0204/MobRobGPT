"""배분 계획 의미 검증 (스키마 통과 후) — 두 어댑터 공용.

가벼운 검증만: 실존 클러스터 · 중복 없음 · n_ships>=1.
투입 합계가 아군수를 넘어도 실패시키지 않는다(브릿지가 클램프) → 폴백 남발 방지.
"""
from .schema import BattlefieldState, CommanderPlan


def _validate_deployments(plan: CommanderPlan, state: BattlefieldState) -> None:
    cluster_ids = {c.id for c in state.enemy_clusters}
    seen = set()
    for d in plan.deployments:
        if d.cluster_id not in cluster_ids:
            raise ValueError(f"존재하지 않는 클러스터 배분: {d.cluster_id}")
        if d.cluster_id in seen:
            raise ValueError(f"클러스터 중복 배분: {d.cluster_id}")
        seen.add(d.cluster_id)
        if d.n_ships < 1:
            raise ValueError(f"n_ships<1 (cluster {d.cluster_id})")
    # 빈 deployments(전원 예비)도 유효.

    # HOLD 대상은 실존 아군 id 여야 함(범위 밖은 조용히 무시하지 말고 잡음).
    ally_ids = {a.id for a in state.allies}
    for i in getattr(plan, "hold_ships", None) or []:
        if i not in ally_ids:
            raise ValueError(f"존재하지 않는 아군 HOLD 지정: {i}")
