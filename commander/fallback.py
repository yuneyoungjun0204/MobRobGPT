"""오프라인 테스트용 휴리스틱 경로 생성기 (라이브 어댑터는 사용 안 함).

LLM 없이 파이프라인을 시험할 때 쓰는 간단한 기하 경로: 위협비례로 배를 클러스터에 붙이고,
각 배마다 자기 위치 → 클러스터 방향 6-WP 직선 경로, 바깥 3개 구간에 그물.
※ 시뮬 자체 _build_cluster_path 만큼 정교하지 않음. demo --fallback / diag 검증용.
"""
from .schema import BattlefieldState, CommanderPlan, ShipRoute, Waypoint


def heuristic_plan(state: BattlefieldState) -> CommanderPlan:
    allies = state.allies
    clusters = sorted(state.enemy_clusters, key=lambda c: c.count, reverse=True)
    P = len(allies)
    if not clusters or P == 0:
        return CommanderPlan(routes=[], rationale="적 클러스터 없음 → 전원 예비.")

    # 위협(척수) 비례 좌석 배분 → 클러스터 id 시퀀스
    total = sum(c.count for c in clusters) or 1
    raw = [c.count / total * P for c in clusters]
    seats = [int(x) for x in raw]
    for i in sorted(range(len(clusters)), key=lambda i: raw[i] - seats[i], reverse=True)[:P - sum(seats)]:
        seats[i] += 1
    if sum(seats) == 0:
        seats[0] = 1
    slots = []
    for ci, s in enumerate(seats):
        slots.extend([clusters[ci]] * s)

    cx, cy = state.mothership.pos.x, state.mothership.pos.y
    r_net = 0.75 * state.constraints.max_intercept_radius   # 도달가능 링(요격 가능 반경)

    routes = []
    for a, cl in zip(allies, slots):
        ax, ay = a.pos.x, a.pos.y
        # 클러스터 방위의 도달가능 링 위 요격점(먼 클러스터 중심이 아니라!)
        dx, dy = cl.center.x - cx, cl.center.y - cy
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        tx, ty = cx + dx / d * r_net, cy + dy / d * r_net
        wps = [Waypoint(x=ax + (tx - ax) * ((s + 1) / 6.0),
                        y=ay + (ty - ay) * ((s + 1) / 6.0),
                        deploy_net=(s >= 3))          # 바깥 절반 구간에 그물
               for s in range(6)]
        routes.append(ShipRoute(ally_id=a.id, waypoints=wps))

    committed = len(routes)
    rationale = (f"[휴리스틱] 위협비례 투입 {committed}/{P}척, 배별 6-WP 직선 경로 + 바깥 3구간 그물.")
    return CommanderPlan(routes=routes, rationale=rationale)
