"""오프라인 테스트용 휴리스틱 배정 (라이브 어댑터는 사용 안 함, demo --fallback / diag 검증용).

위협(척수) 비례로 클러스터별 투입 척수를 정한다. 경로·그물은 시뮬이 기하로 생성.
"""
from .schema import BattlefieldState, CommanderPlan, ClusterDeployment


def heuristic_plan(state: BattlefieldState) -> CommanderPlan:
    """최소 병력·전 클러스터 커버: 위협 큰 순으로 클러스터당 1척씩(아군 수까지). 나머지 예비."""
    allies = state.allies
    clusters = sorted(state.enemy_clusters, key=lambda c: c.count, reverse=True)
    P = len(allies)
    if not clusters or P == 0:
        return CommanderPlan(deployments=[], rationale="적 클러스터 없음 → 전원 예비.")

    covered = clusters[:P]          # 아군 수까지 위협 큰 클러스터부터 1척씩
    uncovered = clusters[P:]        # 아군보다 클러스터가 많으면 나머지는 불가피하게 미커버
    # ally_ids 는 비워둔다 → plan_to_assign 이 효율/안전 복합점수로 배를 대신 선택.
    deployments = [ClusterDeployment(cluster_id=cl.id, ally_ids=[]) for cl in covered]
    committed = len(deployments)
    rat = (f"[휴리스틱] 최소방어: 클러스터당 1척으로 {committed}개 커버 "
           f"(투입 {committed}/{P}척, 예비 {P - committed}척).")
    if uncovered:
        rat += f" 아군 부족으로 미커버: C{[c.id for c in uncovered]}."
    return CommanderPlan(deployments=deployments, rationale=rat)
