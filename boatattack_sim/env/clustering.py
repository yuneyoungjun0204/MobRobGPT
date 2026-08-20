"""
boatattack_sim/env/clustering.py — 모선 기준 각도 클러스터 (centroid / spread)

순수 numpy. 적을 모선 기준 방위각 빈(sector)으로 묶어 '위협 군집'의
중심·각도스프레드·규모·접근속도를 요약한다. obs 의 적 관측 = 이 클러스터 set
(BoatAttack 설계 채용: 무리 방위+스프레드가 곧 그물 위치·폭 신호).
"""
import numpy as np


def enemy_clusters(enemy_xy, alive, center, n_clusters: int = 3):
    """
    적을 모선 기준 방위각으로 n_clusters 개 군집(각도 빈)으로 묶어 요약.
    반환 [n_clusters, 4] = (centroid_bearing_deg, spread_deg, count, mean_dist).
    살아있는 적 없으면 0.
    """
    enemy_xy = np.atleast_2d(np.asarray(enemy_xy, dtype=np.float64))
    alive = np.asarray(alive, dtype=bool)
    out = np.zeros((n_clusters, 4), dtype=np.float64)
    if enemy_xy.shape[0] == 0 or not alive.any():
        return out
    c = np.asarray(center, dtype=np.float64)
    d = enemy_xy[alive] - c
    brg = (np.degrees(np.arctan2(d[:, 0], d[:, 1]))) % 360.0       # 0..360
    dist = np.hypot(d[:, 0], d[:, 1])
    # 360°를 n_clusters 빈으로 분할
    bins = np.floor(brg / (360.0 / n_clusters)).astype(int) % n_clusters
    for k in range(n_clusters):
        m = bins == k
        if not m.any():
            continue
        b = brg[m]
        # 원형 평균 (방위각)
        cx, cy = np.cos(np.deg2rad(b)).mean(), np.sin(np.deg2rad(b)).mean()
        cb = np.degrees(np.arctan2(cy, cx)) % 360.0
        spread = float(np.sqrt(max(0.0, 1.0 - np.hypot(cx, cy))) * 90.0)  # 0=집중,↑=퍼짐
        out[k] = [cb, spread, m.sum(), dist[m].mean()]
    return out


def enemy_clusters_vec(e_pos, e_alive, e_hdg, center, enemy_speed,
                       n_clusters: int = 4):
    """
    N월드 벡터화 각도 클러스터. 모선 기준 방위 빈(360/n_clusters°)으로 적을 묶어 요약.
      e_pos [N,M,2], e_alive [N,M], e_hdg [N,M]
    반환 dict (모두 [N, n_clusters] 또는 [N,n_clusters,2]):
      centroid [N,K,2] 멤버 평균 위치 (world)
      count    [N,K]   멤버 수
      spread_deg [N,K] 각도 범위(빈 내부라 wrap 불요): max−min bearing
      approach [N,K]   모선 방향 평균 접근속도 (m/step)
      active   [N,K]   count>0
    """
    e_pos = np.asarray(e_pos, np.float64); e_alive = np.asarray(e_alive, bool)
    N, M = e_pos.shape[:2]; K = n_clusters
    c = np.asarray(center, np.float64)
    dx = e_pos[..., 0] - c[0]; dy = e_pos[..., 1] - c[1]
    brg = np.degrees(np.arctan2(dx, dy)) % 360.0                  # [N,M]
    binw = 360.0 / K
    bink = (np.floor(brg / binw).astype(np.int64)) % K           # [N,M]
    # 적 접근속도(모선 방향 성분)
    evx = np.sin(np.deg2rad(e_hdg)) * enemy_speed
    evy = np.cos(np.deg2rad(e_hdg)) * enemy_speed
    emx = c[0] - e_pos[..., 0]; emy = c[1] - e_pos[..., 1]
    emn = np.hypot(emx, emy) + 1e-6
    approach = (evx * emx + evy * emy) / emn                      # [N,M]

    cent = np.zeros((N, K, 2)); cnt = np.zeros((N, K))
    spread = np.zeros((N, K)); appr = np.zeros((N, K))
    for k in range(K):
        m = (bink == k) & e_alive                                # [N,M]
        c_k = m.sum(1).astype(np.float64)                        # [N]
        cnt[:, k] = c_k
        den = np.maximum(c_k, 1.0)
        cent[:, k, 0] = (e_pos[..., 0] * m).sum(1) / den
        cent[:, k, 1] = (e_pos[..., 1] * m).sum(1) / den
        bmax = np.where(m, brg, -np.inf).max(1)
        bmin = np.where(m, brg, np.inf).min(1)
        spread[:, k] = np.where(c_k > 0, bmax - bmin, 0.0)
        appr[:, k] = np.where(c_k > 0, (approach * m).sum(1) / den, 0.0)
    return {"centroid": cent, "count": cnt, "spread_deg": spread,
            "approach": appr, "active": cnt > 0}


def cluster_by_gaps_vec(e_pos, e_alive, e_hdg, center, enemy_speed,
                        n_clusters: int = 4, gap_deg: float = 40.0):
    """N월드 벡터화 **적응형 각도-간격(gap) 클러스터링** (고정 90° 빈 대체).

    적을 모선 기준 방위로 원형 정렬 후, **큰 각도 빈틈에서 잘라** 무리로 묶는다:
      · 무리 사이 큰 gap(>gap_deg)에서만 분할 → 한 방향 무리는 통째로 1클러스터(빈 경계가
        무리를 둘로 쪼개는 고정-빈 아티팩트 제거), 멀리 떨어진 무리만 별도 클러스터.
      · 클러스터 수 B = clip(#(gap>gap_deg), 1, K) — 데이터에 맞춰 1..K개로 적응(중복 배정 방지).
      · 가장 큰 gap 을 원형 '이음새'로 열고, 그 다음 큰 gap 들(상위 B개)을 경계로 라벨 부여.
    반환: enemy_clusters_vec 와 동일 dict + **labels[N,M]**(적별 클러스터 id, 죽음/미배정=-1).
      (labels 로 leak/멤버십을 일관 계산 — 빈 재계산 불일치 제거.)
    """
    e_pos = np.asarray(e_pos, np.float64); e_alive = np.asarray(e_alive, bool)
    N, M = e_pos.shape[:2]; K = int(n_clusters)
    c = np.asarray(center, np.float64); ar = np.arange(N)
    dx = e_pos[..., 0] - c[0]; dy = e_pos[..., 1] - c[1]
    brg = np.degrees(np.arctan2(dx, dy)) % 360.0                     # [N,M]
    cnt_alive = e_alive.sum(1).astype(np.int64)                      # [N]

    # 원형 정렬(살아있는 적을 방위 오름차순; 죽은 적은 뒤로)
    key = np.where(e_alive, brg, 1e9)
    order = np.argsort(key, axis=1)                                  # [N,M] 정렬→원본 idx
    sb = np.take_along_axis(brg, order, axis=1)                      # 정렬된 방위 [N,M]

    # 인접 gap(정렬 위치 p 의 gap = 점 p→p+1) + wrap(마지막→처음). [N,M], 무효=-1
    gaps = np.full((N, M), -1.0)
    cg = sb[:, 1:] - sb[:, :-1]                                      # [N,M-1] (오름차순→≥0)
    cons_valid = (np.arange(1, M)[None, :] < cnt_alive[:, None])     # p+1 가 살아있음
    gaps[:, :M - 1] = np.where(cons_valid, cg, -1.0)
    rows = np.where(cnt_alive >= 1)[0]
    if rows.size:
        last = np.take_along_axis(sb, np.clip(cnt_alive[:, None] - 1, 0, M - 1), 1)[:, 0]
        wrap = sb[:, 0] + 360.0 - last                              # 마지막→처음(360 넘어)
        gaps[rows, cnt_alive[rows] - 1] = wrap[rows]

    # B = 클러스터 수: 큰 gap 개수(>gap_deg) 를 [1, min(K, #점)] 로 클립
    B = np.clip((gaps > gap_deg).sum(1), 1, K)
    B = np.minimum(B, np.maximum(cnt_alive, 1))                      # [N]
    # 경계 = 상위 B개 gap (가장 큰 것=이음새). desc 정렬 인덱스로 정확히 B개 선택(동점 안전).
    desc = np.argsort(-np.where(gaps >= 0, gaps, -1.0), axis=1)      # [N,M]
    boundary = np.zeros((N, M), bool)
    for k in range(M):
        sel = k < B
        if sel.any():
            boundary[ar[sel], desc[sel, k]] = True
    seam = desc[:, 0]                                               # 가장 큰 gap 위치(원형 시작점)

    # 라벨링: 이음새 다음 점부터 원형으로 걸으며 경계 gap 통과 시 클러스터 id 증가
    cntc = np.maximum(cnt_alive, 1)
    lbl_sorted = np.full((N, M), -1, np.int64)
    cur = np.zeros(N, np.int64)
    for k in range(M):
        valid = k < cnt_alive
        posk = (seam + 1 + k) % cntc
        if k >= 1:
            prev = (seam + k) % cntc
            cur = cur + (boundary[ar, prev] & valid).astype(np.int64)
        if valid.any():
            rr = ar[valid]
            lbl_sorted[rr, posk[rr]] = cur[rr]
    lbl_sorted = np.clip(lbl_sorted, -1, K - 1)
    labels = np.full((N, M), -1, np.int64)
    np.put_along_axis(labels, order, lbl_sorted, axis=1)
    labels = np.where(e_alive, labels, -1)

    # 적 접근속도(모선 방향 성분)
    evx = np.sin(np.deg2rad(e_hdg)) * enemy_speed
    evy = np.cos(np.deg2rad(e_hdg)) * enemy_speed
    emx = c[0] - e_pos[..., 0]; emy = c[1] - e_pos[..., 1]
    emn = np.hypot(emx, emy) + 1e-6
    approach = (evx * emx + evy * emy) / emn                         # [N,M]

    cent = np.zeros((N, K, 2)); cnt = np.zeros((N, K))
    spread = np.zeros((N, K)); appr = np.zeros((N, K))
    br = np.deg2rad(brg)
    for k in range(K):
        m = (labels == k) & e_alive                                 # [N,M]
        ck = m.sum(1).astype(np.float64); cnt[:, k] = ck
        den = np.maximum(ck, 1.0)
        cent[:, k, 0] = (e_pos[..., 0] * m).sum(1) / den
        cent[:, k, 1] = (e_pos[..., 1] * m).sum(1) / den
        cosb = (np.cos(br) * m).sum(1) / den
        sinb = (np.sin(br) * m).sum(1) / den
        R = np.hypot(cosb, sinb)                                    # 원형 결집도
        spread[:, k] = np.where(ck > 0, np.sqrt(np.clip(1.0 - R, 0.0, 1.0)) * 90.0, 0.0)
        appr[:, k] = np.where(ck > 0, (approach * m).sum(1) / den, 0.0)
    return {"centroid": cent, "count": cnt, "spread_deg": spread,
            "approach": appr, "active": cnt > 0, "labels": labels}
