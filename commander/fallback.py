"""오프라인/LLM 불가 시의 폴백 지휘관 (demo --fallback / diag 검증 · ollama 미설치 환경).

위협(척수) 큰 클러스터부터 1척씩 배정한다. 경로·그물은 시뮬/정책이 기하로 생성.

★ 이 모듈은 '지휘관'이지 배정 보정층이 아니다. 배정 파이프라인이 mode="llm"(순수 지휘관
  배정)으로 바뀐 뒤로 **빈 `ally_ids` 는 '시스템이 대신 고름'이 아니라 '아무도 안 보냄'** 을
  뜻한다. 따라서 폴백 지휘관도 LLM 과 동일하게 **배를 직접 지목해야** 한다.
  (예전에는 비워두고 plan_to_assign 의 헝가리안에 위임했다 — 그 위임처가 이제 없다.)
"""
from . import geometry as GEO
from .schema import BattlefieldState, CommanderPlan, ClusterDeployment


def _pick(state: BattlefieldState, clusters):
    """클러스터별로 배 1척씩 지목 — 요격점까지 이동거리+선회 최소, 경로 교차 회피.

    전역 최소비용 매칭(헝가리안). 탐욕은 한 배가 먼저 가져가면 다른 배가 더 나은 짝을 놓쳐
    총 이동거리·선회가 커진다. scipy 가 없으면 탐욕으로 내려간다.
    비용·기하는 commander/geometry.py 를 쓴다 — plan_to_assign 과 **같은 기준**이다.
    """
    W = state.constraints.world_size
    avail = [a for a in state.allies if a.alive]
    if not avail or not clusters:
        return {}
    icept = GEO.intercepts(state)
    cls = [c for c in clusters if c.id in icept][:len(avail)]
    if not cls:
        return {}

    # sticky(연속성): 현재 담당이면 비용 차감 → 타겟이 매 결정 뒤바뀌는 것 억제.
    #   id 뿐 아니라 **방위**로도 같은 무리를 인정한다(GEO.sticky_bonus 주석 참조).
    cost = [[GEO.ship_cost(a, icept[cl.id], [], W) - GEO.sticky_bonus(a, cl, W)
             for cl in cls] for a in avail]
    try:
        import numpy as _np
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(_np.asarray(cost, dtype=float))
        pairs = list(zip([int(r) for r in rows], [int(c) for c in cols]))
    except Exception:                                      # scipy 없으면 탐욕
        pairs = []
        order = sorted((cost[i][j], i, j) for i in range(len(avail)) for j in range(len(cls)))
        ur, uc = set(), set()
        for _, i, j in order:
            if i not in ur and j not in uc:
                pairs.append((i, j)); ur.add(i); uc.add(j)

    # 교차 해소·효율 보정은 sim_bridge 와 **같은 함수**를 쓴다(배 id 순 정규화 → 같은 답).
    refined = GEO.refine_pairs([(avail[i], cls[j].id) for i, j in pairs], icept, W,
                               clusters={c.id: c for c in state.enemy_clusters})
    return {c: a.id for a, c in refined}



def heuristic_plan(state: BattlefieldState) -> CommanderPlan:
    """최소 병력·전 클러스터 커버: 위협 큰 순으로 클러스터당 1척씩(아군 수까지). 나머지 예비."""
    allies = [a for a in state.allies if a.alive]
    clusters = sorted(state.enemy_clusters, key=lambda c: c.count, reverse=True)
    P = len(allies)
    if not clusters or P == 0:
        return CommanderPlan(deployments=[], rationale="적 클러스터 없음 → 전원 예비.")

    covered = clusters[:P]          # 아군 수까지 위협 큰 클러스터부터 1척씩
    uncovered = clusters[P:]        # 아군보다 클러스터가 많으면 나머지는 불가피하게 미커버
    picks = _pick(state, covered)   # ★ 배를 직접 지목(빈 ally_ids 는 '아무도 안 보냄'이므로)
    by_cluster = {c: [a] for c, a in picks.items()}

    # ── ★ 잉여 전력 배정: 1:1 매칭 뒤 남은 배를 '가장 과부하인' 클러스터에 덧붙인다 ──
    #   활성 클러스터가 아군보다 적으면(실측: 결정의 46%가 클러스터 1개 / 아군 3척) 남는 배가
    #   전부 영구 정지한다. DefenseVecEnv._compute_assignment 의 잉여배정과 같은 취지다.
    #   요격점 층 분리(반경 assign_layer_dt · 방위 assign_layer_dbear)는 주입 시점
    #   (rl_bridge._inject)에서 붙으므로 여기서는 '누구를 어느 클러스터에' 만 정한다.
    used = set(picks.values())
    idle = [a for a in state.allies
            if a.alive and a.id not in used and a.nets_remaining > 0]
    if idle and by_cluster:
        cmap = {c.id: c for c in state.enemy_clusters}
        cnt = {c.id: c.count for c in state.enemy_clusters}
        W = state.constraints.world_size
        for a in sorted(idle, key=lambda x: x.id):
            # 부하 = 적 수 /(1+배정수). 여기에 **연속성**을 얹는다 — 안 얹으면 잉여 배가
            # 매 결정 부하 순위만 보고 담당을 갈아탄다(실측: 전환의 53%가 잉여 배였다).
            # sticky_bonus 를 부하 단위로 환산: 보너스/W → 0.45 또는 0.30 만큼 순위를 올린다.
            def score(c):
                base = cnt.get(c, 0) / (1.0 + len(by_cluster[c]))
                return base + GEO.sticky_bonus(a, cmap[c], W) / W if c in cmap else base
            cid = max(by_cluster, key=score)
            by_cluster[cid].append(a.id)

    deployments = [ClusterDeployment(cluster_id=c, ally_ids=sorted(v))
                   for c, v in sorted(by_cluster.items())]
    committed = sum(len(v) for v in by_cluster.values())
    rat = (f"[휴리스틱] 클러스터 {len(by_cluster)}개 커버, 투입 {committed}/{P}척 "
           f"(예비 {P - committed}척). 지목: "
           + ", ".join(f"C{c}<-USV{v}" for c, v in sorted(by_cluster.items())) + ".")
    if uncovered:
        rat += f" 아군 부족으로 미커버: C{[c.id for c in uncovered]}."
    return CommanderPlan(deployments=deployments, rationale=rat)
