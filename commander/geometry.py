"""배정 기하 — 요격점·선분교차·배 비용. 순수 함수만(외부 의존 없음).

`sim_bridge` 와 `fallback` 이 **같은 기준**으로 짝을 짓도록 여기에 단일 소스로 둔다.
sim_bridge 는 boatattack_sim 을 import 하므로, 폴백 지휘관이 그걸 끌어오지 않게
공통부를 이 모듈로 분리했다. numpy·scipy 도 쓰지 않는다.
"""
import math


def intercept_point(cx, cy, mx, my, v_a, v_e, r_cap):
    """모선-클러스터 선상의 요격 지점. 반경 = v_a·d/(v_a+v_e), max_intercept 로 캡."""
    dx, dy = cx - mx, cy - my
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (mx, my)
    r = min(v_a * d / (v_a + v_e), r_cap)
    return (mx + dx / d * r, my + dy / d * r)


def segments_cross(a, b, c, d):
    """선분 ab, cd 가 교차하면 True (두 배의 직선 경로 충돌 위험 판정)."""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    return (ccw(a, b, c) * ccw(a, b, d) < 0) and (ccw(c, d, a) * ccw(c, d, b) < 0)


def ship_cost(a, ipt, assigned_pairs, W):
    """아군 a 를 요격점 ipt 로 보낼 때의 '효율+안전' 비용(작을수록 좋음).

    효율: 요격점까지 이동거리(≈도착시간) + 선회량 + 그물 보유.
    안전: 이미 배정된 배들의 경로와 교차하면(충돌 위험) 큰 페널티.
    """
    px, py = a.pos.x, a.pos.y
    ix, iy = ipt
    travel = math.hypot(ix - px, iy - py)                       # 효율: 이동거리
    desired = math.degrees(math.atan2(ix - px, iy - py)) % 360.0
    turn = abs(((desired - a.heading + 180.0) % 360.0) - 180.0)  # 효율: 선회량[deg]
    turn_pen = (turn / 180.0) * (W * 0.06)
    nets_pen = 0.0 if a.nets_remaining > 0 else (W * 2.0)         # 그물 없으면 사실상 배제
    cross_pen = sum(W * 0.7 for (p, q) in assigned_pairs
                    if segments_cross((px, py), (ix, iy), p, q))  # 안전: 경로 교차
    return travel + turn_pen + nets_pen + cross_pen


def intercepts(state):
    """전장 상태 → {cluster_id: 요격점(x,y)}. sim_bridge/fallback 공통."""
    mx, my = state.mothership.pos.x, state.mothership.pos.y
    con = state.constraints
    v_a = max(con.ally_speed, 1e-6)
    return {c.id: intercept_point(c.center.x, c.center.y, mx, my, v_a,
                                  max(c.approach_speed, 1e-6), con.max_intercept_radius)
            for c in state.enemy_clusters}


def sticky_bonus(ally, cl, W, same=0.45, near_deg=25.0, near=0.30):
    """연속성 보너스(비용에서 뺄 값). 클수록 현재 담당을 유지한다.

    ★ 클러스터 id 만 보면 안 된다 — id 는 매 결정 **방위 정렬 순서로 새로 매겨진다**
      (`clustering.cluster_by_gaps_vec`). 적 한 척이 죽거나 무리 방위가 바뀌면 같은 무리의
      id 가 0↔1 로 뒤바뀌고, 배는 같은 적을 쫓는데도 '담당이 사라졌다'며 타겟을 갈아탄다.
      실측(2026-08-20): 타겟 전환의 **39%가 이 id 재부여 때문**이었다.

    그래서 **기하로도** 판정한다:
      · id 가 같으면                                    → `same` (강한 연속성)
      · 직전 담당 방위(`ally.assigned_bearing`)와 이 클러스터 방위가 `near_deg` 안이면
        → `near` (id 는 바뀌었어도 같은 무리로 간주)
    """
    W = float(W)
    if ally.assigned_cluster is None:
        return 0.0
    if ally.assigned_cluster == cl.id:
        return W * same
    b_prev = getattr(ally, "assigned_bearing", None)
    if b_prev is None:
        return 0.0
    if abs(((cl.bearing - float(b_prev) + 180.0) % 360.0) - 180.0) <= near_deg:
        return W * near                     # 방위가 거의 같다 = 사실상 같은 무리
    return 0.0


def refine_pairs(pairs, icept, W, eps_ratio=0.1, clusters=None):
    """2-opt: 배 쌍의 담당 클러스터를 맞바꿔 총 (이동거리+선회+경로교차)가 뚜렷이 줄면 스왑.

    clusters — {cluster_id: EnemyCluster}. 주면 연속성(sticky_bonus)도 비용에 반영해
    직전 담당을 유지하는 쪽을 선호한다. 없으면 순수 효율 기준.

    pairs — [(ally_obj, cluster_id)]. **배 id 오름차순으로 정규화해 순회**하므로 호출자가
    어떤 순서로 넘기든 결과가 같다(같은 입력 → 같은 국소최적). 이 정규화가 없으면
    sim_bridge 와 fallback 이 같은 배정에서 서로 다른 답에 수렴한다.

    근소차는 연속성 위해 유지(임계 W*eps_ratio) → 경로 교차(큰 페널티)나 명백한 비효율만 고침.
    반환: 스왑이 반영된 새 [(ally_obj, cluster_id)] (입력 리스트는 건드리지 않는다).
    """
    ps = sorted(pairs, key=lambda ac: ac[0].id)
    eps = W * eps_ratio

    def seg(i):
        a, c = ps[i]
        return ((a.pos.x, a.pos.y), icept[c])

    improved, guard = True, 0
    while improved and guard < 20:
        improved = False
        guard += 1
        for x in range(len(ps)):
            for y in range(x + 1, len(ps)):
                (a1, c1), (a2, c2) = ps[x], ps[y]
                if c1 == c2 or c1 not in icept or c2 not in icept:
                    continue
                s1, s2 = seg(x), seg(y)
                # ★ 연속성도 비용에 넣는다 — 안 넣으면 매칭 단계에서 sticky 로 지킨 짝을
                #   2-opt 가 근소한 효율 차로 되돌려 타겟이 매 결정 흔들린다.
                #   clusters 를 받은 경우에만(없으면 순수 효율 2-opt).
                st_cur = st_swp = 0.0
                if clusters:
                    st_cur = (sticky_bonus(a1, clusters[c1], W)
                              + sticky_bonus(a2, clusters[c2], W))
                    st_swp = (sticky_bonus(a1, clusters[c2], W)
                              + sticky_bonus(a2, clusters[c1], W))
                cur = (ship_cost(a1, icept[c1], [s2], W)
                       + ship_cost(a2, icept[c2], [s1], W)) - st_cur
                t1 = ((a1.pos.x, a1.pos.y), icept[c2])
                t2 = ((a2.pos.x, a2.pos.y), icept[c1])
                swp = (ship_cost(a1, icept[c2], [t2], W)
                       + ship_cost(a2, icept[c1], [t1], W)) - st_swp
                if swp < cur - eps:
                    ps[x], ps[y] = (a1, c2), (a2, c1)
                    improved = True
    return ps


__all__ = ["intercept_point", "segments_cross", "ship_cost", "intercepts",
           "refine_pairs", "sticky_bonus"]
