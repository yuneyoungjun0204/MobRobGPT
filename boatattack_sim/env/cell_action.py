"""
boatattack_sim/env/cell_action.py — 셀선택 행동공간 기하 (pointer 어텐션용)

· make_cells: 모선중심 **극좌표 후보셀**(방위 sector × 반경 band) → world 좌표 [C,2] (에피소드 불변, 정적).
· build_routes_from_cells: 선택한 cell_nets개 셀 → route[N,P,Kw,2] + net_mask[N,P,Kw].
    각 셀 c 에서 코리도(적→모선) **수직 벽**을 net_max_len 로 깔도록 (a=c-perp·half, b=c+perp·half) 2WP 구성.
    route = [a0,b0,a1,b1,...], net leg = b_k (a_k→b_k 구간 = 그물 벽, 코리도 차단).
방위 규약은 defense_env.build_obs 와 동일: bearing = atan2(dx,dy) (0=+y, 90=+x) → world = c + r·(sin,cos).
"""
import numpy as np


def make_cells(cfg, center):
    """후보셀 world 좌표 반환. cell_world[C,2], cell_polar[C,2]=(r, bearing_deg).
    cfg.cell_grid == "cartesian": 모선중심 정사각 격자(cell_cart_n) → **요격환형[r_min,r_max]만** 필터.
    그 외(polar): 방위(cell_bearings)×반경(cell_bands) 극좌표 격자."""
    c = np.asarray(center, np.float64)
    grid = getattr(cfg, "cell_grid", "polar")
    if grid == "annulus":                                        # ★ 균일간격 환형: 반경마다 방위수 = 2πr/s
        s = float(cfg.cell_spacing)
        radii = np.arange(cfg.cell_r_min, cfg.cell_r_max + 1e-6, s)
        xs, ys, rs, bs = [], [], [], []
        for r in radii:
            nb = max(1, int(round(2.0 * np.pi * r / s)))         # 큰 반경일수록 방위 많이 → 호간격 s 일정
            th = (2.0 * np.pi / nb) * np.arange(nb)
            xs.append(c[0] + r * np.sin(th)); ys.append(c[1] + r * np.cos(th))
            rs.append(np.full(nb, r)); bs.append(np.degrees(th))
        cell_world = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=-1)
        cell_polar = np.stack([np.concatenate(rs), np.concatenate(bs)], axis=-1)
        return cell_world, cell_polar
    if grid == "cartesian":
        n = int(cfg.cell_cart_n); R = float(cfg.cell_r_max)
        xs = np.linspace(-R, R, n)                               # 모선기준 [-r_max, r_max]
        X, Y = np.meshgrid(xs, xs, indexing="ij")               # [n,n]
        rr = np.hypot(X, Y)                                      # 모선거리
        keep = (rr >= cfg.cell_r_min) & (rr <= cfg.cell_r_max)  # 요격환형만
        px = (c[0] + X)[keep]; py = (c[1] + Y)[keep]
        cell_world = np.stack([px, py], axis=-1)                # [C,2]
        r = rr[keep]; brg = np.degrees(np.arctan2(X[keep], Y[keep])) % 360.0
        cell_polar = np.stack([r, brg], axis=-1)
        return cell_world, cell_polar
    nb = int(cfg.cell_bearings); nr = int(cfg.cell_bands)
    brg = (2.0 * np.pi / nb) * np.arange(nb)                     # [nb] rad (0=+y)
    radii = np.linspace(cfg.cell_r_min, cfg.cell_r_max, nr)      # [nr] m
    B, R = np.meshgrid(brg, radii, indexing="ij")               # [nb,nr]
    x = c[0] + R * np.sin(B)                                     # sin→x (규약 일치)
    y = c[1] + R * np.cos(B)                                     # cos→y
    cell_world = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)   # [C,2]
    cell_polar = np.stack([R.reshape(-1), np.degrees(B.reshape(-1))], axis=-1)
    return cell_world, cell_polar


def build_routes_from_cells(cell_pts, a_pos, a_alive, center, cfg, Kw):
    """선택 셀 → **연결 폴리라인 그물벽** route/net_mask.
      cell_pts[N,P,K,2] (K=cell_nets), a_pos[N,P,2], a_alive[N,P].
    route WP = 배와 가까운 순으로 정렬한 K개 셀. leg0(배→첫셀)=transit(0),
      leg1..K-1(셀_{k-1}→셀_k)=그물(1) → 셀들을 잇는 **연속 벽**(부채꼴 스윕 재현).
    죽은 배 = 정지(hold), net 0."""
    N, P = a_pos.shape[:2]
    K = cell_pts.shape[2]
    # 배에서 가까운 셀부터 방문하도록 정렬 → 벽이 배 진행방향으로 연속
    d2 = ((cell_pts - a_pos[:, :, None, :]) ** 2).sum(-1)        # [N,P,K]
    order = np.argsort(d2, axis=-1)                             # [N,P,K]
    cp = np.take_along_axis(cell_pts, order[..., None], axis=2)  # [N,P,K,2] 정렬
    route = np.empty((N, P, Kw, 2))
    L = min(K, Kw)
    route[:, :, :L, :] = cp[:, :, :L, :]
    if L < Kw:
        route[:, :, L:, :] = cp[:, :, L - 1:L, :]              # 남으면 마지막 셀 반복(정지)
    route = np.clip(route, 0.0, cfg.world_size)
    net_mask = np.zeros((N, P, Kw), bool)
    net_mask[:, :, 1:L] = True                                 # leg1..K-1 = 셀 잇는 그물벽
    hold = np.broadcast_to(a_pos[:, :, None, :], (N, P, Kw, 2))
    al = a_alive[..., None, None]
    route = np.where(al, route, hold)
    net_mask = net_mask & a_alive[..., None]
    return route, net_mask.astype(np.int64)
