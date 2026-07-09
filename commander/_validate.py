"""배분 계획 정제/검증 (스키마 통과 후) — 두 어댑터 공용.

핵심 원칙: LLM 의 사소한 실수(중복 ally, 없는 id 등)로 **전체 계획을 버리고 전원 정지**하면
안 된다. 그래서 검증(예외) 대신 **정제(sanitize)** 를 기본으로 쓴다 — 잘못된 부분만 고쳐
항상 적용 가능한 계획을 돌려준다. _validate_deployments 는 진단/테스트용으로 남긴다.
"""
from .schema import BattlefieldState, CommanderPlan, ClusterDeployment


def sanitize_plan(plan: CommanderPlan, state: BattlefieldState) -> CommanderPlan:
    """LLM 계획을 '항상 적용 가능'하게 정제(예외 없음):
    - 존재하지 않는 클러스터·중복 클러스터 배분 제거
    - 존재하지 않는 아군 id 제거, 한 배가 두 클러스터에 중복되면 첫 클러스터만 유지
      (뒤 클러스터는 ally_ids 가 비어도 브릿지가 효율/안전 기준으로 다른 배를 채움)
    - hold_ships 도 실존 id 로 정리·중복 제거
    """
    cluster_ids = {c.id for c in state.enemy_clusters}
    ally_ids = {a.id for a in state.allies}
    seen_cluster: set = set()
    seen_ship: set = set()
    new_deps = []
    for d in plan.deployments:
        if d.cluster_id not in cluster_ids or d.cluster_id in seen_cluster:
            continue
        seen_cluster.add(d.cluster_id)
        ids = []
        for aid in (d.ally_ids or []):
            if aid in ally_ids and aid not in seen_ship:
                seen_ship.add(aid); ids.append(aid)
        new_deps.append(ClusterDeployment(cluster_id=d.cluster_id, ally_ids=ids,
                                          deploy_net=d.deploy_net, net_legs=d.net_legs))
    holds = list(dict.fromkeys(i for i in (plan.hold_ships or []) if i in ally_ids))
    return CommanderPlan(deployments=new_deps, hold_ships=holds, rationale=plan.rationale)


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
