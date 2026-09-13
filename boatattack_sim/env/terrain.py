"""
boatattack_sim/env/terrain.py — 실제 해역 육지(섬) 마스크 + APF 척력장 (단일 소스)

`eval/basemap.py` 가 배경 '그림'만 깔던 것을, **물리·관측·행동에 들어가는 데이터**로 승격한다.
앵커(`cfg.geo_lat`, `cfg.geo_lon`) 중심 `world_size` 정사각 박스의 육지를 bool 마스크로 만든다.

판정 소스 사다리 ★ (`cfg.land_source = "auto"` 는 위에서부터 성공할 때까지 내려간다)

    1. `vector`    — **OSM Overpass 벡터**(`natural=coastline`, `place=island|islet`).
                     섬의 위치·모양·크기가 좌표 폴리곤으로 온다. **이미지 분류가 아예 없다**
                     → 구름·라벨 글자·행정경계선이 육지로 잡힐 여지가 원천적으로 없다.
    2. `tile`      — OSM 표준 타일 RGB. 단 `~수역` 이 아니라 **육지 팔레트 화이트리스트**로
                     좁힌다(아래 '오탐 억제' 참고).
    3. `satellite` — 위성(Esri) RGB 에서 **진한 초록**만 육지로. 구름(고휘도·저채도)·
                     얕은 바다(청록)를 배제하는 대신 사구·암반 섬 일부는 놓친다.
                     정보 손실을 감수하고 오탐을 없애는 최후 폴백.

오탐 억제 — 왜 `~water` 를 버렸나 ★
    구 구현은 OSM 타일에서 `~수역색` 을 전부 육지로 봤다. 수역은 단일색(#aad3df)이라
    깔끔해 보이지만, 바다 위 **지명 라벨의 흰 후광·글자 획**과 항로/행정경계선까지
    '수역색이 아닌 것' 으로 딸려 들어온다(opening 으로도 글자 뭉치는 살아남는다).
    지금은 **육지로 알려진 색만 통과**시키고(화이트리스트), 도로·하천 구멍은 closing 으로 메운다.
    원칙: **놓치는 섬 < 없는 섬** — 사용자 요구대로 미검출을 감수하고 오탐을 막는다.

좌표 규약
    반환 마스크의 축은 **`[ix, iy] = [x=East, y=North]`** — `env/grid.py` 의
    `painted[i=x//cell, j=y//cell]`, `env/cnn_map.py` 와 동일. 그릴 때만 전치한다.
    래스터 이미지는 `[row=북→남, col=서→동]` 이므로 `img[::-1, :].T` 로 옮긴다.

네트워크
    최초 1회만. `env/_terrain_cache/` 에 소스 해상도 마스크를 캐시한다.
    전부 실패하면 **all-False(=육지 없음)** 를 돌려주고 경고만 찍는다 — 오프라인에서도 학습은 돈다.
"""
import math
import os

import numpy as np

from .config import SimConfig, DEFAULT_CONFIG

OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
SAT_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/tile/{z}/{y}/{x}")
# ★ 전 지구 미러만 쓴다. 지역 미러(overpass.osm.ch 등)는 **200 OK + 빈 결과** 를 돌려줘
#   '이 해역엔 섬이 없다' 로 조용히 오판하게 만든다(실측: 남해 질의에 elements=0).
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
_UA = {"User-Agent": "boatattack_sim/0.1 (maritime defense sim; land mask)"}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_terrain_cache")

# OSM 기본 스타일 육지 팔레트 (화이트리스트). 도로 흰색/노랑은 **일부러 뺐다** —
# 바다 위 라벨 후광이 흰색이라 같이 들어온다. 섬 안 도로 구멍은 closing 이 메운다.
_OSM_LAND_RGB = (
    (242, 239, 233),   # 기본 육지 배경 landuse 없는 곳
    (173, 209, 158),   # forest / wood
    (205, 235, 176),   # grass / park / meadow
    (200, 215, 171),   # scrub
    (238, 240, 213),   # farmland
    (224, 223, 223),   # residential
    (218, 219, 205),   # allotments / orchard 계열
    (245, 233, 198),   # sand / beach
    (238, 229, 220),   # bare rock / heath
    (217, 208, 201),   # building
)
_OSM_LAND_TOL = 14

_CACHE = {}          # (lat, lon, world, zoom, ...) → 배열

# 벡터 결과가 이 비율을 넘으면 '해안선 링 반전' 으로 보고 실패 처리한다.
# 해상 방어 시나리오의 앵커는 바다가 주인공이므로, 육지가 과반인 박스는 어차피 쓸 수 없다.
_INVERSION_FRAC = 0.60


# ── 형태학 (scipy 없이도 도는 4-이웃 침식/팽창) ────────────────────────

def _erode(a, k=1):
    for _ in range(k):
        p = np.pad(a, 1, constant_values=False)
        a = a & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return a


def _dilate(a, k=1):
    for _ in range(k):
        p = np.pad(a, 1, constant_values=False)
        a = a | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return a


def _close(a, k=1):
    """팽창→침식. 섬 안의 도로·하천·라벨 구멍을 메운다(면적은 대체로 보존)."""
    return _erode(_dilate(a, k), k) if k > 0 else a


def _open(a, k=1):
    """침식→팽창. 선·점 같은 얇은 오탐을 지운다(덩어리는 살아남는다)."""
    return _dilate(_erode(a, k), k) if k > 0 else a


def _fill_holes(a):
    """섬 내부에 갇힌 구멍(팔레트에 없는 색으로 렌더된 마을·과수원 등)을 메운다."""
    try:
        from scipy.ndimage import binary_fill_holes
        return np.asarray(binary_fill_holes(a), bool)
    except Exception:
        return a


# ── 좌표 ──────────────────────────────────────────────────────────────

def _lonlat_to_xfrac_yfrac(lon, lat, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _bounds(lat, lon, world_m):
    half = world_m / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return lon - dlon, lon + dlon, lat - dlat, lat + dlat      # W, E, S, N


def _lonlat_to_local(lon, lat, anchor_lat, anchor_lon, world_m):
    """경위도 → 로컬 미터 (박스 좌하단이 원점, x=East, y=North). `_bounds` 와 동일 근사."""
    mx = 111_320.0 * math.cos(math.radians(anchor_lat))
    x = (np.asarray(lon, np.float64) - anchor_lon) * mx + world_m * 0.5
    y = (np.asarray(lat, np.float64) - anchor_lat) * 111_320.0 + world_m * 0.5
    return x, y


def _src_n(world_m, zoom):
    """소스 해상도(한 변 px). 타일 소스의 m/px 와 맞춰 둔다(z13 ≈ 16 m/px @위도 35)."""
    mpp = 156543.03392 * math.cos(math.radians(35.0)) / (2.0 ** zoom)
    return int(np.clip(round(world_m / max(mpp, 1e-6)), 192, 2048))


# ══════════════════════════════════════════════════════════════════════
#  소스 1: OSM Overpass 벡터 (섬 폴리곤) — 기본
# ══════════════════════════════════════════════════════════════════════

def _overpass(query, timeout=90):
    """Overpass 미러를 순서대로 시도. 전부 실패하면 None."""
    try:
        import requests
    except Exception as e:
        print(f"[terrain] requests 없음 → 벡터 소스 건너뜀 ({e})")
        return None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, timeout=timeout, headers=_UA)
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            print(f"[terrain] overpass 실패({url.split('/')[2]}): {ex}")
    return None


def _fetch_ways(lat, lon, world_m, pad, cache_dir=_CACHE_DIR):
    """앵커 박스(+pad) 의 해안선/섬 way 를 로컬 미터 좌표 폴리라인으로 반환.

    반환 `(coast, outer, inner)`
        coast — `natural=coastline` way. **진행방향 왼쪽이 육지** 규약을 쓴다.
        outer — 면(面)으로 그려진 섬의 바깥 경계 polyline.
        inner — 그 안의 내수면(호수) 경계 polyline.

    ★ 섬을 way 로만 찾으면 놓친다. 실측(소매물도): 큰 섬이 `place=island` **릴레이션**
      (멤버 way 는 무태그)이라 `way["place"="island"]` 에 안 걸렸고, `natural=coastline`
      도 없었다 → 위성엔 뻔히 보이는 섬이 마스크에서 통째로 빠졌다. 릴레이션 멤버
      geometry 까지 받아야 한다.

    원본 응답은 `_terrain_cache/ways_*.json` 에 남긴다 — 임계·여유를 바꿔 다시 만들 때
    Overpass 를 다시 두드리지 않는다(공용 API 예의이자 오프라인 재현성).
    """
    import json

    w, e, s, n = _bounds(lat, lon, world_m * (1.0 + 2.0 * pad))
    bb = f"{s},{w},{n},{e}"
    q = (f"[out:json][timeout:90];"
         f'(way["natural"="coastline"]({bb});'
         f' way["place"~"^(island|islet|archipelago)$"]({bb});'
         f' way["natural"="land"]({bb});'
         f' rel["place"~"^(island|islet|archipelago)$"]({bb});'
         f' rel["natural"="land"]({bb}););'
         f"out geom;")
    import hashlib
    qh = hashlib.sha1(q.encode("utf-8")).hexdigest()[:8]      # 질의가 바뀌면 캐시도 갈린다
    jpath = os.path.join(cache_dir,
                         f"ways_{lat:.5f}_{lon:.5f}_{int(world_m)}_p{pad:.2f}_{qh}.json")
    data = None
    if os.path.exists(jpath):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    if data is None:
        data = _overpass(q)
        if data is not None:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                with open(jpath, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception:
                pass
    if data is None:
        return None, None, None

    def _pts(g):
        x, y = _lonlat_to_local([p["lon"] for p in g], [p["lat"] for p in g],
                                lat, lon, world_m)
        return np.stack([x, y], 1)

    coast, outer, inner = [], [], []
    for el in data.get("elements", []):
        if el.get("type") == "relation":
            for m in el.get("members", []):
                g = m.get("geometry") or []
                if m.get("type") != "way" or len(g) < 2:
                    continue
                (inner if m.get("role") == "inner" else outer).append(_pts(g))
            continue
        g = el.get("geometry") or []
        if len(g) < 3:
            continue
        pts = _pts(g)
        if (el.get("tags") or {}).get("natural") == "coastline":
            coast.append(pts)
        else:
            outer.append(pts)
    return coast, outer, inner


def _join_chains(ways, eps=0.5):
    """끝점을 공유하는 way 를 이어 붙여 긴 사슬로. (섬 하나가 여러 way 로 쪼개져 있다)"""
    chains = [np.asarray(w, np.float64) for w in ways]
    changed = True
    while changed:
        changed = False
        heads = {}
        for i, c in enumerate(chains):
            heads.setdefault((round(c[0, 0] / eps), round(c[0, 1] / eps)), []).append(i)
        used = set()
        out = []
        for i, c in enumerate(chains):
            if i in used:
                continue
            cur = c
            while True:
                if np.hypot(*(cur[0] - cur[-1])) < eps:       # 이미 닫힘
                    break
                k = (round(cur[-1, 0] / eps), round(cur[-1, 1] / eps))
                nxt = [j for j in heads.get(k, []) if j not in used and j != i]
                if not nxt:
                    break
                j = nxt[0]
                used.add(j)
                cur = np.concatenate([cur, chains[j][1:]], 0)
                changed = True
            used.add(i)
            out.append(cur)
        chains = out
    return chains


def _clip_chain(pts, box):
    """개곡선을 박스로 자른다 → 박스 안 부분곡선 목록(끝점이 경계 위에 놓인다)."""
    W, E, S, N = box
    out, cur = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        t0, t1 = 0.0, 1.0
        ok = True
        for p, q in ((-d[0], a[0] - W), (d[0], E - a[0]),
                     (-d[1], a[1] - S), (d[1], N - a[1])):
            if abs(p) < 1e-12:
                if q < 0:
                    ok = False
                    break
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
        if not ok or t0 > t1:
            if len(cur) > 1:
                out.append(np.asarray(cur))
            cur = []
            continue
        pa, pb = a + d * t0, a + d * t1
        if not cur or np.hypot(*(np.asarray(cur[-1]) - pa)) > 1e-6:
            if len(cur) > 1:
                out.append(np.asarray(cur))
            cur = [pa]
        cur.append(pb)
    if len(cur) > 1:
        out.append(np.asarray(cur))
    return out


def _box_t(p, box):
    """박스 경계 위 점 → 둘레 파라미터 t∈[0,4) (반시계: (W,S)→(E,S)→(E,N)→(W,N))."""
    W, E, S, N = box
    x, y = float(p[0]), float(p[1])
    d = (abs(y - S), abs(x - E), abs(N - y), abs(x - W))
    i = int(np.argmin(d))
    if i == 0:
        return (x - W) / (E - W)
    if i == 1:
        return 1.0 + (y - S) / (N - S)
    if i == 2:
        return 2.0 + (E - x) / (E - W)
    return 3.0 + (N - y) / (N - S)


def _box_corner(k, box):
    W, E, S, N = box
    return ((W, S), (E, S), (E, N), (W, N))[int(k) % 4]


def _link_on_box(segs, box):
    """★ 열린 해안 세그먼트들을 박스 둘레로 **서로 이어** 링을 만든다.

    각 세그먼트는 경계에서 시작해 경계에서 끝난다. 해안선 규약이 '진행방향 왼쪽이 육지'
    이므로, 이탈점에서 **반시계로 가장 가까운 (다른) 세그먼트의 진입점**으로 이어야
    링 전체에서 육지가 왼쪽에 남는다.

    ⚠ 세그먼트마다 독립으로 '내 이탈점 → 내 진입점' 을 닫으면(구 `_close_on_box` 단독 사용)
      중간에 있는 다른 세그먼트의 진입점을 지나쳐 **박스를 통째로 감싼다.**
      실측(백령도 남, 박스 357 km²): 3 세그먼트가 각각 +352 / +29 / +333 km² 를 육지로
      주장해 맵 전체가 육지가 됐다. 소스는 성공했는데 결과만 뒤집히는 무음 오류였다.
      올바른 연결은 seg1 → seg2 → seg0 → (짧은 호) → seg1 의 **단일 링**이다.
    """
    n = len(segs)
    t_in = [_box_t(c[0], box) for c in segs]
    t_out = [_box_t(c[-1], box) for c in segs]
    rings, used = [], [False] * n
    for s0 in range(n):
        if used[s0]:
            continue
        parts, i = [], s0
        for _ in range(n + 1):                       # 최대 n 번이면 반드시 닫힌다
            used[i] = True
            parts.append(segs[i])
            j = min(range(n), key=lambda k: (t_in[k] - t_out[i]) % 4.0)
            span = (t_in[j] - t_out[i]) % 4.0
            corners, c = [], math.floor(t_out[i]) + 1.0
            while c - t_out[i] < span - 1e-9:
                corners.append(_box_corner(c, box))
                c += 1.0
            if corners:
                parts.append(np.asarray(corners, np.float64))
            if j == s0 or used[j]:
                break
            i = j
        rings.append(np.concatenate(parts + [segs[s0][:1]], 0))
    return rings


def _close_on_box(chain, box):
    """열린 해안선을 박스 둘레를 따라 **반시계로** 이어 닫는다.

    해안선 규약은 '진행방향 왼쪽이 육지'. 박스 둘레를 반시계로 돌면 박스 안쪽이 왼쪽이므로,
    끝점→시작점을 반시계로 이으면 링 전체에서 육지가 왼쪽 = 링이 육지를 반시계로 감싼다.
    → 부호 있는 면적이 양수면 육지, 음수면 (섬 안의) 물.
    """
    t0 = _box_t(chain[-1], box)
    t1 = _box_t(chain[0], box)
    span = (t1 - t0) % 4.0
    ring = [chain]
    corners = []
    c = math.floor(t0) + 1.0
    while c - t0 < span - 1e-9:
        corners.append(_box_corner(c, box))
        c += 1.0
    if corners:
        ring.append(np.asarray(corners, np.float64))
    ring.append(chain[:1])
    return np.concatenate(ring, 0)


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _rings_from_coast(coast, box):
    """해안선 way 목록 → (육지 링, 물 링) 폴리곤 목록."""
    land, hole = [], []
    rings, open_segs = [], []
    for ch in _join_chains(coast):
        if np.hypot(*(ch[0] - ch[-1])) < 1.0:        # 이미 닫힌 섬 → 그대로
            rings.append(ch)
        else:                                        # 박스를 가로지르는 해안선 조각
            open_segs += [c for c in _clip_chain(ch, box) if len(c) > 1]
    if open_segs:
        # ★ 조각들을 **함께** 이어야 한다. 하나씩 닫으면 박스를 통째로 감싼다(_link_on_box 주석).
        rings += _link_on_box(open_segs, box)
    for p in rings:
        if p is None or len(p) < 4:
            continue
        if np.hypot(*(p[0] - p[-1])) > 1e-6:
            p = np.concatenate([p, p[:1]], 0)
        (land if _signed_area(p) > 0 else hole).append(p)
    return land, hole


def _rings_from_areas(parts):
    """면(面) polyline 조각 → 닫힌 링만 (방향 무시). 안 닫히면 버리고 경고.

    릴레이션 멤버는 bbox 와 무관하게 **전체 geometry** 가 오므로 정상 데이터는 반드시 닫힌다.
    해안선과 달리 방향 규약이 없어 열린 조각은 안팎을 판정할 수 없다 — 지어내느니 버린다.
    """
    rings, dropped = [], 0
    for c in _join_chains(parts):
        if len(c) >= 4 and np.hypot(*(c[0] - c[-1])) < 1.0:
            rings.append(np.concatenate([c, c[:1]], 0)
                         if np.hypot(*(c[0] - c[-1])) > 1e-6 else c)
        else:
            dropped += 1
    if dropped:
        print(f"[terrain] 면 조각 {dropped}개가 닫히지 않아 버렸다(부분 데이터)")
    return rings


def _fill(polys, n, world_m):
    """폴리곤 목록 → [n,n] bool (축 [x,y], 셀 중심 판정). 폴리곤 bbox 안만 검사."""
    out = np.zeros((n, n), bool)
    if not polys:
        return out
    try:
        from matplotlib.path import Path
    except Exception as e:
        print(f"[terrain] matplotlib 없음 → 벡터 래스터화 불가 ({e})")
        return None
    c = (np.arange(n) + 0.5) * (world_m / n)
    for p in polys:
        i0 = int(np.clip(np.searchsorted(c, p[:, 0].min()) - 1, 0, n - 1))
        i1 = int(np.clip(np.searchsorted(c, p[:, 0].max()) + 1, 0, n))
        j0 = int(np.clip(np.searchsorted(c, p[:, 1].min()) - 1, 0, n - 1))
        j1 = int(np.clip(np.searchsorted(c, p[:, 1].max()) + 1, 0, n))
        if i1 <= i0 or j1 <= j0:
            continue
        gx, gy = np.meshgrid(c[i0:i1], c[j0:j1], indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        inside = Path(p).contains_points(pts).reshape(i1 - i0, j1 - j0)
        out[i0:i1, j0:j1] |= inside
    return out


def land_polygons(cfg: SimConfig = DEFAULT_CONFIG, pad=0.25):
    """OSM 벡터 섬 폴리곤을 **로컬 미터 좌표 그대로** 반환 `(land_rings, hole_rings)`.

    래스터화 이전의 원천 데이터다 — `eval/land_check.py` 가 '무엇을 받아서 무엇으로 칠했나'
    를 나란히 보여줄 때 쓴다. 실패 시 `([], [])`.
    """
    coast, outer, inner = _fetch_ways(cfg.geo_lat, cfg.geo_lon, cfg.world_size, pad)
    if coast is None:
        return [], []
    P = float(cfg.world_size) * pad
    box = (-P, float(cfg.world_size) + P, -P, float(cfg.world_size) + P)
    land_rings, hole_rings = _rings_from_coast(coast, box)
    land_rings += _rings_from_areas(outer)             # place=island 등 면으로 그려진 섬
    hole_rings += _rings_from_areas(inner)             # 그 안의 내수면
    return land_rings, hole_rings


def _land_mask_vector(lat, lon, world_m, zoom, margin_px, pad=0.25):
    """OSM 벡터(해안선·섬 폴리곤) → 소스 해상도 육지 마스크 [H,H] bool. 실패 시 None."""
    coast, outer, inner = _fetch_ways(lat, lon, world_m, pad)
    if coast is None:
        return None
    P = world_m * pad
    box = (-P, world_m + P, -P, world_m + P)
    land_rings, hole_rings = _rings_from_coast(coast, box)
    land_rings += _rings_from_areas(outer)             # place=island 등 면으로 그려진 섬
    hole_rings += _rings_from_areas(inner)             # 그 안의 내수면
    if not land_rings and not hole_rings:
        print("[terrain] 벡터: 이 박스에 해안선이 없다 → 육지 0 (외해)")
        return np.zeros((_src_n(world_m, zoom),) * 2, bool)

    n = _src_n(world_m, zoom)
    land = _fill(land_rings, n, world_m)
    if land is None:
        return None
    holes = _fill(hole_rings, n, world_m)
    if holes is not None:
        land &= ~holes
    if margin_px > 0:
        land = _dilate(land, margin_px)                # 해안 여유
    # ── ★ 반전 검출 (2026-08-19) ────────────────────────────────────
    #   해안선 링 닫기가 틀어지면 박스 전체가 육지로 칠해진다. 크래시도 경고도 없이
    #   '벡터 소스 OK' 를 찍고 결과만 뒤집히므로, 여기서 못 잡으면 아무도 모른다.
    #   (실측: 백령도·흑산도·손죽도가 나란히 100.00 % / 158.8 km² 로 나왔다 —
    #    서로 다른 좌표가 같은 값을 낸 것이 유일한 단서였다.)
    #   해상 시나리오에서 박스의 대부분이 육지인 앵커는 애초에 쓸 수 없으므로,
    #   임계를 넘으면 **성공으로 위장하지 말고 실패**시켜 래스터 폴백으로 내려보낸다.
    frac = float(land.mean())
    if frac > _INVERSION_FRAC:
        print(f"[terrain] ! 벡터 반전 의심 — 육지 {frac*100:.1f} % > "
              f"{_INVERSION_FRAC*100:.0f} % (해안선 링이 박스를 통째로 감쌌을 가능성). "
              f"벡터 실패 처리 → 다음 소스로")
        return None
    print(f"[terrain] 벡터 소스 OK — 링 {len(land_rings)}개(+내수면 {len(hole_rings)}) "
          f"→ {n}² px 중 육지 {int(land.sum())} ({land.mean()*100:.2f} %)")
    return land


# ══════════════════════════════════════════════════════════════════════
#  소스 2·3: 래스터 타일 (OSM 표준 / 위성) — 폴백
# ══════════════════════════════════════════════════════════════════════

def _fetch_tiles(lat, lon, world_m, zoom, url, esri_order=False):
    """앵커 박스를 덮는 타일을 받아 스티칭 후 크롭 → [H,W,3] uint8. 실패 시 None."""
    try:
        import io

        import requests
        from PIL import Image
    except Exception as e:
        print(f"[terrain] requests/PIL 없음 ({e})")
        return None
    w, e, s, n = _bounds(lat, lon, world_m)
    x0f, y0f = _lonlat_to_xfrac_yfrac(w, n, zoom)
    x1f, y1f = _lonlat_to_xfrac_yfrac(e, s, zoom)
    xt0, yt0 = int(math.floor(x0f)), int(math.floor(y0f))
    xt1, yt1 = int(math.floor(x1f)), int(math.floor(y1f))
    nx, ny = xt1 - xt0 + 1, yt1 - yt0 + 1
    if nx * ny > 64:
        print(f"[terrain] 타일 과다({nx}x{ny}) → zoom 을 낮추세요.")
        return None
    try:
        mos = Image.new("RGB", (nx * 256, ny * 256))
        sess = requests.Session()
        sess.headers.update(_UA)
        for ix in range(nx):
            for iy in range(ny):
                r = sess.get(url.format(z=zoom, x=xt0 + ix, y=yt0 + iy), timeout=15)
                r.raise_for_status()
                mos.paste(Image.open(io.BytesIO(r.content)).convert("RGB"),
                          (ix * 256, iy * 256))
    except Exception as ex:
        print(f"[terrain] 타일 다운로드 실패 ({ex})")
        return None
    px0, py0 = (x0f - xt0) * 256.0, (y0f - yt0) * 256.0
    px1, py1 = (x1f - xt0) * 256.0, (y1f - yt0) * 256.0
    return np.asarray(mos.crop((int(px0), int(py0),
                                int(math.ceil(px1)), int(math.ceil(py1)))), np.uint8)


def _land_mask_tile(lat, lon, world_m, zoom, open_k, margin_px):
    """OSM 표준 타일 → **육지 팔레트 화이트리스트** 로 마스크. 실패 시 None.

    구 구현의 `~수역색` 은 바다 위 라벨 후광·글자·경계선까지 육지로 삼켰다.
    '알려진 육지색만 통과' 로 뒤집고, 섬 안 도로 구멍은 closing 으로 메운다.
    """
    img = _fetch_tiles(lat, lon, world_m, zoom, OSM_URL)
    if img is None:
        return None
    a = img.astype(np.int16)
    land_img = np.zeros(img.shape[:2], bool)
    for rgb in _OSM_LAND_RGB:
        m = np.ones(img.shape[:2], bool)
        for ch, v in enumerate(rgb):
            m &= np.abs(a[..., ch] - v) <= _OSM_LAND_TOL
        land_img |= m
    land_img = _close(land_img, 3)                  # 도로·하천으로 끊긴 육지 잇기
    land_img = _fill_holes(land_img)                # 섬 안 팔레트 미등록 색 구멍 메우기
    land_img = _open(land_img, max(open_k, 1))      # 얇은 선·점 오탐 제거
    if margin_px > 0:
        land_img = _dilate(land_img, margin_px)
    return np.ascontiguousarray(land_img[::-1, :].T)


def _land_mask_satellite(lat, lon, world_m, zoom, open_k, margin_px):
    """위성(Esri) RGB → **진한 초록만** 육지. 실패 시 None.

    구름(고휘도·저채도)·얕은 바다(청록: B≥G)를 배제한다. 사구·암반·건조지 섬은 놓친다 —
    '없는 섬을 만드느니 있는 섬을 놓친다' 는 원칙에 따른 의도적 손실이다.
    """
    img = _fetch_tiles(lat, lon, world_m, zoom, SAT_URL)
    if img is None:
        return None
    a = img.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    v = a.max(-1)
    sat = v - a.min(-1)
    green = ((g > r + 8) & (g > b + 8)      # 초록 우세 (바다는 B 우세, 구름은 무채색)
             & (g >= 25) & (g <= 170)       # '진한' 초록 — 고휘도 구름/파도 제외
             & (v <= 190) & (sat >= 18))    # 저채도(구름·안개) 제외
    green = _close(green, 3)                # 그림자·능선으로 끊긴 섬 내부 잇기
    green = _fill_holes(green)              # 마을·나지 등 초록 아닌 내부 메우기
    green = _open(green, max(open_k, 2))    # 파도 반짝임·조류 오탐 제거
    if margin_px > 0:
        green = _dilate(green, margin_px)
    return np.ascontiguousarray(green[::-1, :].T)


# ── 소스 사다리 + 디스크 캐시 ─────────────────────────────────────────

def fetch_land_mask(lat, lon, world_m, zoom=13, open_k=2, margin_px=1,
                    source="auto", cache_dir=_CACHE_DIR):
    """앵커 박스의 육지 마스크 [H,H] bool (축 = [x, y]). 전부 실패하면 None.

    source : "auto"(벡터→타일→위성) | "vector" | "tile" | "satellite" | "none"
    open_k    : opening 반복(래스터 소스 전용). 선 성분 제거 강도.
    margin_px : 마스크 팽창(해안 여유). 소스 1 px ≈ 16 m @z13.
    """
    src = str(source or "auto").lower()
    if src == "none":
        return None
    key = (f"land_{src}_{lat:.5f}_{lon:.5f}_{int(world_m)}"
           f"_z{zoom}_o{open_k}_m{margin_px}.npy")
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass

    ladder = {"auto": ("vector", "tile", "satellite")}.get(src, (src,))
    out = None
    for s in ladder:
        if s == "vector":
            out = _land_mask_vector(lat, lon, world_m, zoom, margin_px)
        elif s == "tile":
            out = _land_mask_tile(lat, lon, world_m, zoom, open_k, margin_px)
        elif s == "satellite":
            out = _land_mask_satellite(lat, lon, world_m, zoom, open_k, margin_px)
        else:
            print(f"[terrain] 알 수 없는 land_source={s!r}")
            continue
        if out is not None:
            if s != ladder[0]:
                print(f"[terrain] ! 1순위 소스 실패 → {s} 폴백으로 판정했다")
            break
    if out is None:
        print("[terrain] 모든 소스 실패 → 육지 없음으로 진행")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    try:
        np.save(path, out)
    except Exception:
        pass
    return out


# ── 임의 해상도로 면적비 다운샘플 ─────────────────────────────────────

def _frac_pool(a, n):
    """[H,W] bool → [n,n] float, 블록 내 True 면적비.

    ★ 정사각을 가정하면 안 된다. 타일 모자이크를 박스로 자를 때 반올림 때문에
      가로·세로가 1~3 px 어긋나는 해역이 있다(실측: 신안_홍도북 804x803,
      거제_매물도동 802x803). 예전에는 h 하나로 인덱스를 만들어 그런 해역에서
      "array is not broadcastable" 로 죽었고, 그 결과 KOREA_SITES 6곳 중 2곳이
      쓰이지 못했다. 축마다 따로 비율 인덱스를 만든다.
    """
    h, w = a.shape[:2]
    iy = np.arange(h) * n // h
    ix = np.arange(w) * n // w
    s = np.zeros((n, n))
    c = np.zeros((n, n))
    np.add.at(s, (iy[:, None], ix[None, :]), a.astype(np.float64))
    np.add.at(c, (iy[:, None], ix[None, :]), 1.0)
    return s / np.maximum(c, 1.0)


def _key(cfg, n):
    return (round(float(cfg.geo_lat), 5), round(float(cfg.geo_lon), 5),
            float(cfg.world_size), int(cfg.basemap_zoom),
            str(getattr(cfg, "land_source", "auto")),
            int(getattr(cfg, "land_open_k", 2)), int(getattr(cfg, "land_margin_px", 1)),
            float(getattr(cfg, "land_threshold", 0.30)), int(n))


def land_mask(cfg: SimConfig = DEFAULT_CONFIG, n: int = None):
    """해상도 n×n 의 육지 마스크 [n,n] bool (축 = [x, y]).

    `cfg.land_obstacle` 이 False 면 항상 all-False (기능 자체를 끈 상태).
    임계 `cfg.land_threshold` = 픽셀 내 육지 면적비. 0.3 이면 '30 % 이상 육지면 육지'.
    """
    n = int(n or cfg.grid_size)
    if not bool(getattr(cfg, "land_obstacle", False)):
        return np.zeros((n, n), bool)
    k = _key(cfg, n) + ("mask",)
    if k in _CACHE:
        return _CACHE[k]
    src = fetch_land_mask(cfg.geo_lat, cfg.geo_lon, cfg.world_size,
                          zoom=int(cfg.basemap_zoom),
                          open_k=int(getattr(cfg, "land_open_k", 2)),
                          margin_px=int(getattr(cfg, "land_margin_px", 1)),
                          source=str(getattr(cfg, "land_source", "auto")))
    if src is None:
        out = np.zeros((n, n), bool)
    else:
        out = _frac_pool(src, n) > float(getattr(cfg, "land_threshold", 0.30))
    _CACHE[k] = out
    return out


# ── APF 척력장 (정적 → 1회 계산 후 O(1) 조회) ─────────────────────────

def repulsion_field(cfg: SimConfig = DEFAULT_CONFIG, n: int = None):
    """육지 APF 척력장. 반환 `(dist [n,n] m, away [n,n,2] 단위벡터)`.

    dist = 가장 가까운 육지까지 거리(m). 육지 내부는 0.
    away = 그 육지에서 **멀어지는** 방향 단위벡터. 육지가 없으면 (inf, 0).

    육지가 정적이므로 **에피소드·월드와 무관하게 1회만 계산**하고 캐시한다.
    적/아군 스텝에서는 자기 셀을 인덱싱하는 O(1) 조회만 든다.
    """
    n = int(n or cfg.grid_size)
    k = _key(cfg, n) + ("apf",)
    if k in _CACHE:
        return _CACHE[k]
    land = land_mask(cfg, n)
    cell = float(cfg.world_size) / n
    if not land.any():
        out = (np.full((n, n), np.inf), np.zeros((n, n, 2)))
        _CACHE[k] = out
        return out

    try:
        from scipy.ndimage import distance_transform_edt
        d_px, (ni, nj) = distance_transform_edt(~land, return_indices=True)
    except Exception:                                    # scipy 없으면 브루트포스 폴백
        li, lj = np.nonzero(land)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        d_px = np.full((n, n), np.inf)
        ni = np.zeros((n, n), np.int64)
        nj = np.zeros((n, n), np.int64)
        for s in range(0, len(li), 4096):               # 메모리 상한용 청크
            dx = ii[..., None] - li[None, None, s:s + 4096]
            dy = jj[..., None] - lj[None, None, s:s + 4096]
            dd = np.hypot(dx, dy)
            am = dd.argmin(-1)
            dm = np.take_along_axis(dd, am[..., None], -1)[..., 0]
            better = dm < d_px
            d_px = np.where(better, dm, d_px)
            ni = np.where(better, li[s:s + 4096][am], ni)
            nj = np.where(better, lj[s:s + 4096][am], nj)

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vx = (ii - ni).astype(np.float64)                    # 최근접 육지 → 자기 (=멀어지는 방향)
    vy = (jj - nj).astype(np.float64)
    mag = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ax = np.where(mag > 0, vx / mag, 0.0)
        ay = np.where(mag > 0, vy / mag, 0.0)
    out = (d_px * cell, np.stack([ax, ay], axis=-1))
    _CACHE[k] = out
    return out


def summary(cfg: SimConfig = DEFAULT_CONFIG, n: int = None):
    """진단용 한 줄 요약 (학습 배너·check_setup 에서 호출)."""
    n = int(n or cfg.grid_size)
    m = land_mask(cfg, n)
    src = getattr(cfg, "land_source", "auto")
    if not m.any():
        return f"land: 없음 (land_obstacle={getattr(cfg, 'land_obstacle', False)}, source={src})"
    cell = float(cfg.world_size) / n
    area = m.sum() * cell * cell / 1e6
    return (f"land: {int(m.sum())}/{n*n} cell ({m.mean()*100:.2f}%, {area:.1f} km²) "
            f"@{cell:.0f} m · src={src} · anchor({cfg.geo_lat}, {cfg.geo_lon}) "
            f"thr={cfg.land_threshold}")


# ══════════════════════════════════════════════════════════════════════
#  ★ 지형 프리셋 (다중 해역 로테이션)
# ══════════════════════════════════════════════════════════════════════
#  단일 앵커로 학습하면 정책이 "그 섬 하나의 배치"를 외운다 — 지형 일반화가 안 된다.
#  그렇다고 매 에피소드 임의 좌표를 뽑으면 (a) 네트워크 폭주 (b) 모선이 뭍에서 출발하거나
#  섬이 아예 없거나 육지가 절반인 말이 안 되는 판이 섞인다.
#
#  그래서 **사전 선정 + 로테이션**을 쓴다. 후보 해역을 `eval/land_scout.py` 로 측정해
#  아래 기준을 전부 통과한 곳만 남기고, 월드마다 그중 하나를 균등 추출한다.
#    · 모선(중앙)에서 가장 가까운 육지까지 ≥ 2.5 km  → 출발 직후 좌초 불가
#    · 박스 육지 면적비 0.5 ~ 10 %                    → 없지도, 넘치지도 않게
#    · 요격 환형 잠식 ≤ 8 % · 최악 30° 섹터 ≤ 35 %    → 한 방위가 통째로 막히지 않게
#    · 최대 연결 섬 ≥ 0.05 km²                        → 암초 몇 점이 아니라 '섬'
#
#  각 항목의 실측값은 목록 주석에 그대로 박아 둔다 (재정찰 없이 대조 가능).
#  실측 2026-08-18 (`land_scout --grid 200`, world 12.6 km). 20곳 중 통과 4곳.
#  ⚠ 남은 16곳이 전부 '탈락'인 것은 아니다 — 7곳은 Overpass 429/504 로 **못 재서** 육지 0 으로
#    찍혔다(로그의 "모든 소스 실패"). 재정찰하면 합격지가 늘 수 있다.
#  ⚠ 기존 기본 앵커 통영_매물도서(34.625,128.52)는 모선여유 2305 m 로 탈락선(2500) 바로 아래다.
#    챔피언은 그 해역에서 학습했다 — 지금 빠진 건 기준이 보수적이라서지 그 판이 나빠서가 아니다.
KOREA_SITES = [
    # ★ 2026-08-19 재정찰. 1차 선정(0.0~0.5 %)은 **섬이 요격 환형 밖**이라 지형이 배경 그림이
    #   됐다(아군이 섬 2 km 이내로 간 적 0 회). 그래서 환형 잠식 **하한 1 %** 를 추가하고
    #   모선여유 하한을 2500→1500 m 로 낮춰 다시 골랐다. 아래 값이 그 결과다.
    #   비교: 1차 5곳 환형 잠식 평균 0.2 % → 아래 4곳 평균 2.8 %.
    ("통영_매물도서", 34.6250, 128.5200),   # 육지 2.24% · 여유 2305m · 환형 1.93% · 섹터 16.6%
    ("통영_연화도서", 34.6200, 128.1900),   # 육지 7.47% · 여유 3536m · 환형 2.05% · 섹터 15.1%
    ("신안_홍도북",   34.7400, 125.2000),   # 육지 3.16% · 여유 1607m · 환형 3.19% · 섹터 34.0%
    ("거제_매물도동", 34.6300, 128.6000),   # 육지 2.52% · 여유 1746m · 환형 4.08% · 섹터 22.3%
    ("통영_매물도북", 34.6650, 128.5450),   # 육지 6.62% · 여유 2437m · 환형 5.98% · 섹터 33.5%
                                          #   ★환형 잠식 최대 = 지형이 교전에 가장 깊이 개입.
                                          #   1차엔 Overpass 실패로 '육지 0%' 로 찍혀 탈락했었다.
    # ★ 외해(육지 0 %). 지형이 있는 판만 학습하면 정책이 "섬은 항상 있다"를 전제로 굳는다.
    #   ⚠ 이 항목만 min_frac·min_ann 을 의도적으로 위반한다. 다른 0 % 후보와 달리 여기는
    #     소스 성공 + "해안선이 없다" 로 확인된 진짜 외해다(Overpass 실패가 아니다).
    ("포항_호미곶동", 36.0800, 129.7000),   # 육지 0.00% (외해) — 지형 없는 케이스 대조군
    #
    # 탈락 기록 (같은 실수를 반복하지 않기 위해 남긴다)
    #   · 통영_욕지도남/여수_거문도북/부안_위도서/제주_추자도서 — 1차 합격지였으나
    #     환형 잠식 0.00~0.53 % 로 섬이 교전권 밖. 화면에서도 서로 구별되지 않았다.
    #   · 옹진_백령도남(환형 0.00%) · 신안_흑산도북(모선여유 223m) · 여수_손죽도근해(1159m)
    #     — 해안선 링 반전 버그를 고쳐 **처음으로 제대로 측정된** 곳들. 기준 미달이지만
    #       그전까지는 셋 다 "육지 100 %" 로 나와 정찰표를 오염시키고 있었다.
    #   · 거제_매물도동은 통영_매물도서와 중심 거리 7.3 km(박스 12.6 km)라 바다가 겹친다.
    #     다만 모선 위치가 달라 교전 기하(환형 잠식 4.08 vs 1.93 %)는 서로 다르다.
]

SITE_SETS = {}            # 이름 → [(사이트명, lat, lon)]


def _register_sites(name, sites):
    SITE_SETS[name] = list(sites)
    return SITE_SETS[name]


_register_sites("korea", KOREA_SITES)


def resolve_sites(cfg: SimConfig = DEFAULT_CONFIG):
    """`cfg.land_sites` → `[(name, lat, lon)]`. 항상 최소 1개를 돌려준다.

    · ""            → 단일 앵커 `(cfg.geo_lat, cfg.geo_lon)` (구 동작 그대로)
    · "korea" 등    → `SITE_SETS` 프리셋
    · "a,b,c"       → 프리셋 안의 사이트명을 골라 쓰는 부분집합
    · "34.6,128.5;35.1,126.2" → 좌표 직접 지정(실험용. 검증 안 거친 값이므로 권장 안 함)
    """
    s = str(getattr(cfg, "land_sites", "") or "").strip()
    if not s:
        return [("anchor", float(cfg.geo_lat), float(cfg.geo_lon))]
    if s in SITE_SETS:
        return list(SITE_SETS[s])
    if ";" in s or all(c in "0123456789.,-; " for c in s):
        out = []
        for i, part in enumerate(p for p in s.split(";") if p.strip()):
            lat, lon = (float(v) for v in part.split(",")[:2])
            out.append((f"custom{i}", lat, lon))
        if out:
            return out
    known = {n: (n, la, lo) for grp in SITE_SETS.values() for n, la, lo in grp}
    out = [known[k.strip()] for k in s.split(",") if k.strip() in known]
    if not out:
        raise ValueError(f"land_sites={s!r} 를 해석할 수 없다. "
                         f"프리셋={list(SITE_SETS)} · 사이트={list(known)}")
    return out


def site_cfg(cfg: SimConfig, site):
    """사이트 `(name, lat, lon)` 로 앵커만 바꾼 cfg 사본."""
    import dataclasses
    return dataclasses.replace(cfg, geo_lat=float(site[1]), geo_lon=float(site[2]))


def land_masks(cfg: SimConfig = DEFAULT_CONFIG, n: int = None, sites=None):
    """사이트별 육지 마스크 `[S, n, n]` bool. 사이트가 1개여도 축을 유지한다."""
    n = int(n or cfg.grid_size)
    sites = list(sites if sites is not None else resolve_sites(cfg))
    return np.stack([land_mask(site_cfg(cfg, s), n) for s in sites], axis=0)


def repulsion_fields(cfg: SimConfig = DEFAULT_CONFIG, n: int = None, sites=None):
    """사이트별 APF 척력장 `(dist [S,n,n], away [S,n,n,2])`."""
    n = int(n or cfg.grid_size)
    sites = list(sites if sites is not None else resolve_sites(cfg))
    fs = [repulsion_field(site_cfg(cfg, s), n) for s in sites]
    return (np.stack([f[0] for f in fs], 0), np.stack([f[1] for f in fs], 0))


def sites_summary(cfg: SimConfig = DEFAULT_CONFIG, n: int = None, sites=None):
    """학습 배너용 다중 사이트 한 줄 요약."""
    n = int(n or cfg.grid_size)
    sites = list(sites if sites is not None else resolve_sites(cfg))
    if len(sites) == 1:
        return summary(site_cfg(cfg, sites[0]), n)
    ms = land_masks(cfg, n, sites)
    cell = float(cfg.world_size) / n
    fr = ms.reshape(len(sites), -1).mean(1) * 100.0
    ar = ms.reshape(len(sites), -1).sum(1) * cell * cell / 1e6
    body = " · ".join(f"{s[0]} {f:.2f}%/{a:.1f}km²" for s, f, a in zip(sites, fr, ar))
    return (f"land: 사이트 {len(sites)}곳 로테이션 (평균 {fr.mean():.2f}%, "
            f"{fr.min():.2f}~{fr.max():.2f}%) @{cell:.0f} m — {body}")
