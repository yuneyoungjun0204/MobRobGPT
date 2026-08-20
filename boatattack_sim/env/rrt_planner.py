"""
boatattack_sim/env/rrt_planner.py — RRT* 경로 플래너 (휴리스틱 경로생성, 시각화/검증용)

순수 numpy. 단일 (start, goal, 장애물격자) → 충돌 없는 경로(가변 길이) + 트리.
장애물 = net_installed 격자(깔린 그물벽) + 모선 death-disk. 좌표=world meter, 격자 인덱스 (x→i, y→j).

★ 이 버전은 **시각화/직접확인용**: 월드 1개·배 1척 단위로 진짜 RRT 트리를 키운다(WP 개수 제한 없음).
  학습 핫루프 벡터화(N월드)·액션 안전투영은 plan(Phase A/F)에서 별도 구현.
"""
import numpy as np


# ── 충돌 프리미티브 ──────────────────────────────────────────────────
def seg_blocked(a, b, occ, cell, G, center, mother_r, n_samples=None):
    """선분 a→b 가 설치 그물 셀(occ) 또는 모선 death-disk(반경 mother_r)를 지나면 True.
    a,b [2] world. occ [G,G] bool. 샘플 간격 ≤ cell 로 띠 터널링 방지."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    n = n_samples or max(2, int(np.ceil(d / cell)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]                 # [n,2]
    ci = np.clip((pts[:, 0] / cell).astype(np.int64), 0, G - 1)
    cj = np.clip((pts[:, 1] / cell).astype(np.int64), 0, G - 1)
    if occ[ci, cj].any():
        return True
    dm = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    return bool((dm < mother_r).any())


# ── RRT* (단일 start→goal, 가변 노드) ────────────────────────────────
def rrt_star(start, goal, occ, cell, G, center, mother_r, world_size,
             step=300.0, goal_bias=0.05, max_iter=1500, rewire_radius=650.0,
             goal_tol=150.0, pad=900.0, seed=0):
    """RRT* 경로계획. 반환 dict:
      nodes[K,2], parent[K], cost[K], path[L,2](world, start→goal 순),
      goal_idx, reached(bool: 실제 goal 연결 성공).
    sampling box = bbox(start,goal) 패딩(pad). RRT* rewire 로 경로 단축."""
    rng = np.random.default_rng(seed)
    start = np.asarray(start, float); goal = np.asarray(goal, float)
    lo = np.clip(np.minimum(start, goal) - pad, 0.0, world_size)
    hi = np.clip(np.maximum(start, goal) + pad, 0.0, world_size)

    cap = max_iter + 2
    nodes = np.zeros((cap, 2)); parent = np.full(cap, -1, np.int64); cost = np.zeros(cap)
    nodes[0] = start; n = 1
    reached = False; goal_idx = -1

    for _ in range(max_iter):
        q = goal if rng.random() < goal_bias else lo + rng.random(2) * (hi - lo)
        cur = nodes[:n]
        ni = int(np.argmin(((cur - q) ** 2).sum(1)))                 # 최근접 노드
        npt = nodes[ni]; vec = q - npt; dist = float(np.hypot(vec[0], vec[1]))
        if dist < 1e-6:
            continue
        new = npt + vec / dist * min(step, dist)                     # steer (range=step)
        if seg_blocked(npt, new, occ, cell, G, center, mother_r):
            continue
        # RRT*: rewire_radius 안에서 비용 최소 부모 선택
        dd = np.hypot(nodes[:n, 0] - new[0], nodes[:n, 1] - new[1])
        cand = np.where(dd < rewire_radius)[0]
        best_par = ni; best_cost = cost[ni] + float(np.hypot(*(new - npt)))
        for c in cand:
            cc = cost[c] + float(np.hypot(*(new - nodes[c])))
            if cc < best_cost and not seg_blocked(nodes[c], new, occ, cell, G, center, mother_r):
                best_par = int(c); best_cost = cc
        ni_new = n
        nodes[n] = new; parent[n] = best_par; cost[n] = best_cost; n += 1
        # rewire: 이웃을 new 경유가 더 싸면 부모 교체
        for c in cand:
            cc = best_cost + float(np.hypot(*(nodes[c] - new)))
            if cc < cost[c] and not seg_blocked(new, nodes[c], occ, cell, G, center, mother_r):
                parent[c] = ni_new; cost[c] = cc
        # 목표 연결 시도
        if float(np.hypot(*(new - goal))) <= goal_tol and \
                not seg_blocked(new, goal, occ, cell, G, center, mother_r):
            nodes[n] = goal; parent[n] = ni_new
            cost[n] = best_cost + float(np.hypot(*(goal - new)))
            goal_idx = n; n += 1; reached = True
            break

    if goal_idx < 0:                                                 # 미도달 → 목표 최근접 노드
        goal_idx = int(np.argmin(((nodes[:n] - goal) ** 2).sum(1)))

    path = []; i = goal_idx
    while i >= 0:
        path.append(nodes[i].copy()); i = int(parent[i])
    path = np.array(path[::-1]) if path else start[None, :]
    return {"nodes": nodes[:n].copy(), "parent": parent[:n].copy(), "cost": cost[:n].copy(),
            "path": path, "goal_idx": goal_idx, "reached": reached}


def shortcut(path, occ, cell, G, center, mother_r):
    """Theta*식 LOS 단축: 비인접 정점을 충돌 없이 직접 이으면 중간 정점 제거(가변 길이)."""
    path = np.asarray(path, float)
    if len(path) < 3:
        return path
    out = [path[0]]; i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and seg_blocked(path[i], path[j], occ, cell, G, center, mother_r):
            j -= 1
        out.append(path[j]); i = j
    return np.array(out)


# ── env 1월드 전체 배에 대해 경로계획 (시각화 다리) ──────────────────
def plan_env_world(env, w=0, net_halflen=None, **rrt_kw):
    """배정된 배마다 **그물 살포 기동** 경로계획 (단순 점 도달 X):
      목표 = 적→모선 코리도에 **수직인 요격 벽**(중심=_assignI 요격반경, 길이=2·net_halflen).
      RRT 가 벽의 가까운 끝점 S 까지 **벽 회피 transit** 경로를 만들고, 이어서 S→E **수직 sweep**
      (net_max_len 단위 분할)으로 그물을 깔며 코리도를 가로지른다.
    반환 {p: {nodes,parent(transit 트리), route[M,2], net_wp[M], netline(A,B), start, goal(net_center),
             S, E, reached, short(=route 그리기호환)}}."""
    occ = env.net_installed[w]; cell = env.cell; G = env.G
    center = env.center; mr = env.cfg.mothership_radius; ws = env.cfg.world_size
    nml = float(net_halflen if net_halflen is not None else env.cfg.net_max_len)
    out = {}
    for p in range(env.P):
        if env._assign[w, p] < 0 or not env.a_alive[w, p]:
            continue
        start = env.a_pos[w, p].copy(); cent = env._assign_cent[w, p].copy()
        rel = cent - center; D = float(np.hypot(rel[0], rel[1]))
        if D < 1e-6:
            continue
        rad = rel / D; perp = np.array([-rad[1], rad[0]])            # 코리도축 / 수직축
        net_center = env._assignI[w, p].copy()                      # 요격반경(코리도상, 도달가능)
        A = net_center - perp * nml; B = net_center + perp * nml     # 벽 양 끝점
        S, E = (A, B) if np.hypot(*(A - start)) <= np.hypot(*(B - start)) else (B, A)
        # RRT transit: start→S (설치 그물벽 회피)
        seed = int(env.t[w]) * 131 + p
        r = rrt_star(start, S, occ, cell, G, center, mr, ws, seed=seed, **rrt_kw)
        transit = shortcut(r["path"], occ, cell, G, center, mr)      # [...,2] start→S
        # sweep: S→E 를 net_max_len 으로 분할(그물 leg)
        Lsw = float(np.hypot(*(E - S)))
        nseg = max(1, int(np.ceil(Lsw / env.cfg.net_max_len)))
        sweep = np.array([S + (E - S) * t for t in np.linspace(0.0, 1.0, nseg + 1)])  # [nseg+1,2]
        route = np.vstack([transit, sweep[1:]])                     # transit(…→S) + sweep 내부/끝
        net_wp = np.zeros(len(route), bool)
        net_wp[len(transit):] = True                                # sweep 도착 WP = 그물 leg
        out[p] = {"nodes": r["nodes"], "parent": r["parent"], "route": route, "short": route,
                  "net_wp": net_wp, "netline": (A, B), "start": start, "goal": net_center,
                  "S": S, "E": E, "reached": r["reached"]}
    return out


# ── 단독 데모: 합성 그물벽 한 줄을 두고 RRT 가 우회하는지 그림 저장 ──
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ws = 11000.0; G = 200; cell = ws / G
    center = np.array([ws / 2, ws / 2]); mother_r = 260.0
    occ = np.zeros((G, G), bool)
    # 합성 벽: start(좌하)와 goal(우상) 사이를 가로지르는 수직 띠 (x≈6000, y 4000~7000)
    xi = int(6000 / cell)
    yj0, yj1 = int(4000 / cell), int(7000 / cell)
    for di in range(-2, 3):                                          # 폭 ~5셀(≈275m)
        occ[np.clip(xi + di, 0, G - 1), yj0:yj1] = True

    start = np.array([4200.0, 4500.0]); goal = np.array([7600.0, 6500.0])
    r = rrt_star(start, goal, occ, cell, G, center, mother_r, ws,
                 step=300.0, max_iter=2500, seed=7)
    sp = shortcut(r["path"], occ, cell, G, center, mother_r)
    print(f"[rrt demo] reached={r['reached']} nodes={len(r['nodes'])} "
          f"raw_path={len(r['path'])} short_path={len(sp)}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(occ.T, origin="lower", extent=[0, ws, 0, ws], cmap="Greys",
              alpha=0.5, aspect="equal")
    segs = [[r["nodes"][int(r["parent"][i])], r["nodes"][i]]
            for i in range(len(r["nodes"])) if r["parent"][i] >= 0]
    ax.add_collection(LineCollection(segs, colors="#26C6DA", linewidths=0.4, alpha=0.3))
    ax.plot(r["path"][:, 0], r["path"][:, 1], "-", color="#FFB74D", lw=1.2, alpha=0.7, label="raw RRT*")
    ax.plot(sp[:, 0], sp[:, 1], "-o", color="#00E676", lw=2.4, ms=5, label=f"shortcut ({len(sp)} wp)")
    ax.scatter(*start, c="#42A5F5", s=90, marker="s", label="start", zorder=5)
    ax.scatter(*goal, c="#EF5350", s=120, marker="*", label="goal", zorder=5)
    ax.add_patch(plt.Circle(center, mother_r, color="#FFD54F", alpha=0.4))
    ax.set_xlim(3000, 9000); ax.set_ylim(3000, 8000); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("RRT* — 그물벽 우회 경로생성 (데모)")
    out = os.path.join(os.path.dirname(__file__), "..", "..", "rrt_demo.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("[rrt demo] saved ->", os.path.abspath(out))
