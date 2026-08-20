"""
boatattack_sim/env/cnn_map.py — CNN 점수맵 좌표·마스크 규약 (단일 소스)

셀선택(pointer)의 정적 후보셀 대신, 맵 전체를 H×W 점수맵으로 보고 그 위에서 WP 를 뽑는다.
**world↔pix 변환은 오직 이 모듈에만 있다.** 환경·모델·렌더러가 전부 여기를 import 한다
(규약이 두 군데 있으면 90° 회전/상하반전 버그가 반드시 난다).

규약 (nav):  x=East, y=North, hdg 0°=North, CW+.  bearing = atan2(dx,dy).
배열 축 순서: **[..., ix, iy] = [..., x, y]** — env/grid.py 의 painted[i=x//cell, j=y//cell] 와 동일.
  → 물리격자(200²) → 점수맵(30²) 다운샘플이 전치 없이 맞는다(200%30≠0 → reduceat 일반경로).
  → matplotlib 으로 그릴 때만 imshow(arr.T, origin="lower", extent=extent(cfg)).

  px_size = 2*cnn_extent / cnn_grid_n            (기본 12600/30 = 420.0 m)
  x0, y0  = center - cnn_extent                  (기본 (0,0) = 맵 좌하단)
  world→pix: ix = clip(floor((x-x0)/px), 0, n-1)
  pix→world: x  = x0 + (ix+0.5)*px  (+ dx*px*0.5,  dx∈[-1,1] 서브픽셀)
  flat     : f  = ix*n + iy
"""
import numpy as np


# ── 기본 기하 ────────────────────────────────────────────────────────

def grid_n(cfg) -> int:
    return int(getattr(cfg, "cnn_grid_n", 30))


def extent_m(cfg) -> float:
    """모선중심 반폭(m). None/0 이면 world_half."""
    e = getattr(cfg, "cnn_extent", None)
    return float(e) if e else float(cfg.world_size) * 0.5


def px_size(cfg) -> float:
    return 2.0 * extent_m(cfg) / grid_n(cfg)


def origin(cfg):
    """점수맵 좌하단 world 좌표 (x0, y0)."""
    c = np.asarray(cfg.center, np.float64)
    return c - extent_m(cfg)


def extent(cfg):
    """matplotlib imshow(extent=...) 용 (x0, x1, y0, y1)."""
    x0, y0 = origin(cfg)
    s = 2.0 * extent_m(cfg)
    return (float(x0), float(x0 + s), float(y0), float(y0 + s))


# ── 변환 ─────────────────────────────────────────────────────────────

def world_to_pix(cfg, xy):
    """world [...,2] → (ix, iy) 정수 인덱스 [...]. 맵 밖은 가장자리로 clip."""
    xy = np.asarray(xy, np.float64)
    o = origin(cfg); p = px_size(cfg); n = grid_n(cfg)
    ij = np.floor((xy - o) / p).astype(np.int64)
    ij = np.clip(ij, 0, n - 1)
    return ij[..., 0], ij[..., 1]


def pix_to_world(cfg, ix, iy, off=None):
    """(ix, iy) [...] → 픽셀 중심 world [...,2]. off[...,2]∈[-1,1] 이면 ±반픽셀 서브픽셀 이동."""
    o = origin(cfg); p = px_size(cfg)
    x = o[0] + (np.asarray(ix, np.float64) + 0.5) * p
    y = o[1] + (np.asarray(iy, np.float64) + 0.5) * p
    xy = np.stack([x, y], axis=-1)
    if off is not None:
        xy = xy + np.clip(np.asarray(off, np.float64), -1.0, 1.0) * (p * 0.5)
    return xy


def ij_to_flat(cfg, ix, iy):
    n = grid_n(cfg)
    return np.asarray(ix, np.int64) * n + np.asarray(iy, np.int64)


def flat_to_ij(cfg, f):
    n = grid_n(cfg)
    f = np.asarray(f, np.int64)
    return f // n, f % n


def flat_to_world(cfg, f, off=None):
    ix, iy = flat_to_ij(cfg, f)
    return pix_to_world(cfg, ix, iy, off)


# ── 픽셀 중심 좌표 캐시 (정적: config 당 1회) ────────────────────────

_CACHE = {}


def _key(cfg):
    c = np.asarray(cfg.center, np.float64)
    return (grid_n(cfg), extent_m(cfg), float(c[0]), float(c[1]))


def pixel_centers(cfg):
    """픽셀 중심 world 좌표 (XX[n,n], YY[n,n]) — 축 순서 [ix, iy]."""
    k = _key(cfg) + ("centers",)
    if k not in _CACHE:
        n = grid_n(cfg); o = origin(cfg); p = px_size(cfg)
        ax = o[0] + (np.arange(n) + 0.5) * p
        ay = o[1] + (np.arange(n) + 0.5) * p
        _CACHE[k] = np.meshgrid(ax, ay, indexing="ij")
    return _CACHE[k]


def pixel_polar(cfg):
    """픽셀 중심의 모선기준 (반경 R[n,n], 방위 B[n,n] deg, bearing=atan2(dx,dy))."""
    k = _key(cfg) + ("polar",)
    if k not in _CACHE:
        XX, YY = pixel_centers(cfg)
        c = np.asarray(cfg.center, np.float64)
        dx = XX - c[0]; dy = YY - c[1]
        _CACHE[k] = (np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360.0)
    return _CACHE[k]


def annulus_mask(cfg):
    """요격 환형 [cell_r_min, cell_r_max] 픽셀 마스크 [n,n] bool (정적)."""
    k = _key(cfg) + ("ann", float(cfg.cell_r_min), float(cfg.cell_r_max))
    if k not in _CACHE:
        R, _ = pixel_polar(cfg)
        _CACHE[k] = (R >= cfg.cell_r_min) & (R <= cfg.cell_r_max)
    return _CACHE[k]


def origin_mask(cfg, radius=None):
    """모선 disk 마스크 [n,n] bool (정적)."""
    r = float(cfg.mothership_radius if radius is None else radius)
    k = _key(cfg) + ("moth", r)
    if k not in _CACHE:
        R, _ = pixel_polar(cfg)
        _CACHE[k] = R <= max(r, px_size(cfg) * 0.5)
    return _CACHE[k]


def coord_channels(cfg):
    """CoordConv 채널 [Cc,n,n] float32: 정규화 x, y (+ cnn_coord_r 이면 모선거리 r). 정적."""
    use_r = bool(getattr(cfg, "cnn_coord_r", False))
    k = _key(cfg) + ("coord", use_r)
    if k not in _CACHE:
        XX, YY = pixel_centers(cfg)
        c = np.asarray(cfg.center, np.float64); e = extent_m(cfg)
        ch = [(XX - c[0]) / e, (YY - c[1]) / e]
        if use_r:
            R, _ = pixel_polar(cfg)
            ch.append(np.clip(R / e, 0.0, 1.0))
        _CACHE[k] = np.stack(ch, axis=0).astype(np.float32)
    return _CACHE[k]


# ── 스탬프 (개체 → 픽셀 블록) ────────────────────────────────────────

def stamp_offsets(radius_px: int):
    """(2R+1)² 이웃 오프셋 [S,2] + 가우시안 가중 [S]. radius_px=0 → 단일 픽셀."""
    R = int(max(radius_px, 0))
    d = np.arange(-R, R + 1)
    DX, DY = np.meshgrid(d, d, indexing="ij")
    off = np.stack([DX.reshape(-1), DY.reshape(-1)], axis=-1)
    sig = max(R, 1) * 0.7
    w = np.exp(-(off[:, 0] ** 2 + off[:, 1] ** 2) / (2.0 * sig ** 2))
    return off, w


def stamp_max(out, cfg, pts, valid, radius_px=1, weight=1.0):
    """out[N,n,n] 에 pts[N,K,2] world 위치를 (2R+1)² 블록으로 max-스탬프 (valid[N,K] 만).
    개체 수가 적어(N·K ≲ 수천) np.maximum.at 비용이 무시할 수준이다."""
    n = grid_n(cfg)
    ix, iy = world_to_pix(cfg, pts)                       # [N,K]
    off, w = stamp_offsets(radius_px)                     # [S,2],[S]
    N, Kk = ix.shape
    ni = np.broadcast_to(np.arange(N)[:, None, None], (N, Kk, len(off)))
    xx = np.clip(ix[:, :, None] + off[None, None, :, 0], 0, n - 1)
    yy = np.clip(iy[:, :, None] + off[None, None, :, 1], 0, n - 1)
    ww = np.broadcast_to(w[None, None, :], (N, Kk, len(off))) * float(weight)
    m = np.broadcast_to(valid[:, :, None], (N, Kk, len(off)))
    np.maximum.at(out, (ni[m], xx[m], yy[m]), ww[m])
    return out


def scatter_count(cfg, N, pts, valid, weights=None):
    """pts[N,K,2] → 픽셀별 합계 [N,n,n] (weights 없으면 개수). bincount 로 완전 벡터화."""
    n = grid_n(cfg)
    ix, iy = world_to_pix(cfg, pts)
    flat = (np.arange(N)[:, None] * (n * n) + ix * n + iy).reshape(-1)
    v = np.asarray(valid, bool).reshape(-1)
    w = (np.ones(flat.shape) if weights is None
         else np.asarray(weights, np.float64).reshape(-1))
    acc = np.bincount(flat[v], weights=w[v], minlength=N * n * n)
    return acc.reshape(N, n, n)


# ── 물리격자(200²) → 점수맵(n²) 다운샘플 ─────────────────────────────

def _reduce_plan(cfg, G):
    """물리격자 셀 → 점수맵 픽셀 매핑을 정렬해 reduceat 계획으로 캐시."""
    k = _key(cfg) + ("plan", int(G))
    if k in _CACHE:
        return _CACHE[k]
    n = grid_n(cfg)
    cell = float(cfg.world_size) / G
    gi = (np.arange(G) + 0.5) * cell
    GX, GY = np.meshgrid(gi, gi, indexing="ij")
    ix, iy = world_to_pix(cfg, np.stack([GX, GY], axis=-1))
    tgt = (ix * n + iy).reshape(-1)                      # [G*G] 각 물리셀의 목표 픽셀
    order = np.argsort(tgt, kind="stable")
    st = tgt[order]
    starts = np.concatenate([[0], np.nonzero(np.diff(st))[0] + 1])
    groups = st[starts]                                   # 실제 채워지는 픽셀 flat idx
    _CACHE[k] = (order, starts, groups)
    return _CACHE[k]


def downsample_max(cfg, arr):
    """물리격자 [N,G,G] bool/float → 점수맵 [N,n,n] float32 (블록 max).
    extent=world_half 이고 G%n==0 이면 reshape-maxpool 고속 경로."""
    a = np.asarray(arr)
    N, G = a.shape[0], a.shape[1]
    n = grid_n(cfg)
    aligned = (abs(extent_m(cfg) - float(cfg.world_size) * 0.5) < 1e-9) and (G % n == 0)
    if aligned:
        f = G // n
        return a.reshape(N, n, f, n, f).max(axis=(2, 4)).astype(np.float32)
    order, starts, groups = _reduce_plan(cfg, G)
    v = a.reshape(N, G * G)[:, order].astype(np.float32)
    red = np.maximum.reduceat(v, starts, axis=1)          # [N,#groups]
    out = np.zeros((N, n * n), np.float32)
    out[:, groups] = red
    return out.reshape(N, n, n)


# ── 원판 마스크 (자기회귀 중복 방지 / cross-ship 잠금) ────────────────

def reach_keep(cfg, mask, ix, iy, reach_m):
    """mask[...,n,n] 에서 (ix,iy) 로부터 reach_m **바깥** 픽셀을 False 로 (도달성 상한).

    ★ disk_exclude 의 대칭짝이다. disk_exclude 는 '너무 가까운' 하한(cnn_dup_r)을 막고,
      이건 '너무 먼' 상한(net_max_len)을 막는다. 그물벽은 두 선택 픽셀을 잇는 선분인데
      실제 도색은 net_max_len 에서 끊기므로(defense_env: paint_dist >= net_max_len),
      간격이 그보다 크면 **벽이 목표 픽셀에 닿지 못한다**.
      실측(수정 전): 정책 선택의 88.6 % 가 450 m 를 초과했고, 의도한 벽의 73 % 만 도색됐다.
    reach_m <= 0 이면 무동작(하위호환)."""
    if reach_m <= 0:
        return mask
    n = grid_n(cfg); p = px_size(cfg)
    rp2 = (float(reach_m) / p) ** 2
    ii = np.arange(n)
    d2 = ((ii[:, None] - np.asarray(ix)[..., None, None]) ** 2
          + (ii[None, :] - np.asarray(iy)[..., None, None]) ** 2)
    return mask & (d2 <= rp2)


def disk_exclude(cfg, mask, ix, iy, radius_m):
    """mask[...,n,n] 에서 (ix,iy)[...] 중심 radius_m 안쪽 픽셀을 False 로.
    ix/iy 의 shape 는 mask 의 앞쪽 축과 일치해야 한다."""
    if radius_m <= 0:
        return mask
    n = grid_n(cfg); p = px_size(cfg)
    rp2 = (float(radius_m) / p) ** 2
    ii = np.arange(n)
    d2 = ((ii[:, None] - np.asarray(ix)[..., None, None]) ** 2      # [...,n,1]
          + (ii[None, :] - np.asarray(iy)[..., None, None]) ** 2)   # [...,1,n] → [...,n,n]
    return mask & (d2 > rp2)
