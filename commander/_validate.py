"""배분 계획 의미 검증 (스키마 통과 후) — 두 어댑터 공용.

가벼운 검증만: 실존 클러스터·아군 · 클러스터 중복 없음 · 한 배가 두 클러스터에 중복 배정 안 됨.
ally_ids 를 비워도 유효(브릿지가 효율/안전 기준으로 대신 채움) → 폴백 남발 방지.
"""
from .schema import BattlefieldState, CommanderPlan


def _validate_deployments(plan: CommanderPlan, state: BattlefieldState) -> None:
    cluster_ids = {c.id for c in state.enemy_clusters}
    ally_ids = {a.id for a in state.allies}
    seen_cluster = set()
    seen_ship = set()
    for d in plan.deployments:
        if d.cluster_id not in cluster_ids:
            raise ValueError(f"존재하지 않는 클러스터 배분: {d.cluster_id}")
        if d.cluster_id in seen_cluster:
            raise ValueError(f"클러스터 중복 배분: {d.cluster_id}")
        seen_cluster.add(d.cluster_id)
        for aid in d.ally_ids:
            if aid not in ally_ids:
                raise ValueError(f"존재하지 않는 아군 배정: {aid} (cluster {d.cluster_id})")
            if aid in seen_ship:
                raise ValueError(f"아군 {aid} 가 두 클러스터에 중복 배정")
            seen_ship.add(aid)
    # 빈 deployments(전원 예비)·빈 ally_ids(시스템이 대신 선택)도 유효.

    # HOLD 대상은 실존 아군 id 여야 함(범위 밖은 조용히 무시하지 말고 잡음).
    for i in getattr(plan, "hold_ships", None) or []:
        if i not in ally_ids:
            raise ValueError(f"존재하지 않는 아군 HOLD 지정: {i}")
