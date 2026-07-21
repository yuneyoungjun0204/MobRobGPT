"""
boatattack_sim/env/formations.py — 적 스폰 포메이션 + 아군 초기 배치

순수 numpy. 적은 **맵 가장자리(perimeter)에서 모선(중앙)을 향해** 출발한다.
포메이션 4종(Concentrated/Wave/Diversionary/Random) 제공, 기본 = Random edge.
"""
import numpy as np

from .config import SimConfig, DEFAULT_CONFIG


def _heading_to_center(pos: np.ndarray, center) -> np.ndarray:
    """각 위치에서 모선(center)을 향하는 heading(deg, nav)."""
    d = np.asarray(center, dtype=np.float64) - pos
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0


def _edge_points(n: int, center, radius: float, rng, jitter: float = 250.0) -> np.ndarray:
    """중심에서 radius(m) 원주에 n개 점을 방위 균등 분포(산발). 맵 크기와 decouple.
    (옛 '맵 4변 둘레' 방식은 맵 확대 시 모서리=9km 로 튀어 enemy_spawn_radius 원주로 교체.)"""
    b = rng.uniform(0.0, 360.0, size=n)
    r = np.deg2rad(b)
    pts = center[None, :] + radius * np.stack([np.sin(r), np.cos(r)], axis=1)
    if jitter:
        pts = pts + rng.normal(0.0, jitter, size=(n, 2))
    return pts


def _bearing_point(center, bearing_deg, radius):
    """모선(center)에서 방위 bearing_deg 방향으로 radius(m) 떨어진 점 (nav: x=sin,y=cos)."""
    r = np.deg2rad(bearing_deg)
    return np.array([center[0] + np.sin(r) * radius,
                     center[1] + np.cos(r) * radius], dtype=np.float64)


def _grouped_spawn(center, world, margin, rng, *, bearings, counts,
                   radii, jitter):
    """방위별 '뭉친' 그룹 스폰의 공통 헬퍼.
      bearings [G] 각 그룹 방위(deg), counts [G] 그룹별 적 수,
      radii    [G] 그룹별 모선과의 거리(m), jitter 그룹 내부 산포 반경(m).
    각 그룹은 자기 방위의 한 점 주위에 좁게 모여(가우시안 jitter) 하나의 클러스터를 이룬다.
    반환 pos [Σcounts, 2] (margin 안으로 clip)."""
    lo, hi = margin, world - margin
    pts = []
    for b, cnt, rad in zip(bearings, counts, radii):
        if cnt <= 0:
            continue
        gc = _bearing_point(center, b, rad)                 # 그룹 중심
        off = rng.normal(0.0, jitter, size=(int(cnt), 2))   # 빽빽한 산포
        pts.append(gc[None, :] + off)
    pos = np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))
    return np.clip(pos, lo, hi)


def _dr_scale(base: float, frac: float, rng) -> float:
    """도메인 랜덤화: base 를 ±frac 범위로 곱셈 흔들기 (frac=0.3 → [0.7,1.3]×base)."""
    return float(base) * rng.uniform(1.0 - frac, 1.0 + frac)


def _split_counts(M: int, G: int, weights=None) -> np.ndarray:
    """M개 적을 G개 그룹에 (가중)분배. 합=M 보장."""
    if weights is None:
        weights = np.ones(G)
    w = np.asarray(weights, np.float64); w = w / w.sum()
    base = np.floor(w * M).astype(int)
    while base.sum() < M:                       # 나머지를 큰 그룹부터 채움
        base[np.argmax(w * M - base)] += 1
    return base


def spawn_enemies(cfg: SimConfig = DEFAULT_CONFIG, rng=None, mode: str = "random"):
    """
    적 초기 상태 생성. 반환: (pos [M,2], hdg [M], phase [M]).
      mode:
        "random"       가장자리 무작위 분산
        "concentrated" 한 변에 집중 (한 방향 돌파)
        "wave"         두 변에서 시차 웨이브 (여기선 위치만; 시차는 위빙 phase로)
        "diversionary" 다수 양동 + 소수 반대편
    phase = 위빙 위상 (적별 상이).
    """
    rng = rng or np.random.default_rng(cfg.seed)
    M = cfg.n_enemies
    world = cfg.world_size; margin = cfg.enemy_spawn_margin
    center = np.array(cfg.center, dtype=np.float64)
    # 비-wave 기본 스폰 반경: enemy_spawn_radius(맵과 decouple) 우선, 없으면 가장자리.
    R0 = getattr(cfg, "enemy_spawn_radius", None) or (world * 0.5 - margin)
    jit = getattr(cfg, "enemy_group_jitter", 150.0 * getattr(cfg, "scale", 1.0))
    G0 = max(1, getattr(cfg, "enemy_spawn_groups", 3))
    b0 = rng.uniform(0, 360)                        # 전체 방위 무작위 회전
    # ★ 도메인 랜덤화: 매 스폰마다 포메이션 '구조 파라미터'를 ±frac 흔들어 매번 다른 변형.
    dr = getattr(cfg, "domain_rand", False)
    drf = getattr(cfg, "domain_rand_frac", 0.0) if dr else 0.0
    # ★ jitter 는 DR 로 '키우지 않는다': 키우면 한 그룹이 여러 덩어리로 쪼개져 "클러스터 과다"
    #   (diversionary 3그룹→4~7개)를 유발 → 규제 취지와 반대. 응집 위해 cfg 고정값 유지.

    if mode == "concentrated":
        # 집중: 한 방위에 전원 한 덩어리 (단일 클러스터, 한 방향 강행돌파)
        pos = _grouped_spawn(center, world, margin, rng,
                             bearings=[b0], counts=[M], radii=[R0], jitter=jit)
    elif mode == "diversionary":
        # 양동: 세 방위에서 동시 분산 도래 (4/3/3, 3 클러스터). 한쪽 7 대신 고르게 나눠
        #   여러 방향 압박 → 배정/충돌회피가 더 까다로움.
        cnt = _split_counts(M, 3, weights=[0.4, 0.3, 0.3])     # M=10 → 4/3/3
        # ★ DR: 두 분산 그룹의 방위(기본 130/230)를 ±angle 흔들어 양동 형상 변형.
        da = getattr(cfg, "domain_rand_angle", 0.0) if dr else 0.0
        a1 = 130.0 + (rng.uniform(-da, da) if dr else 0.0)
        a2 = 230.0 + (rng.uniform(-da, da) if dr else 0.0)
        pos = _grouped_spawn(center, world, margin, rng,
                             bearings=[b0, (b0 + a1) % 360, (b0 + a2) % 360],
                             counts=cnt, radii=[R0, R0, R0], jitter=jit)
    elif mode == "wave":
        # 파상: 한 방위에서 여러 단(rank)이 거리차로 줄지어 도달(시차 클러스터).
        #   ★ near(가장 가까운 파, 먼저 도달)부터 +gap 씩 '바깥(뒤)'으로 미뤄 진짜 파상 구현.
        #   near=5000, gap=600, ranks=3 → 5000/5600/6200. 가까운 파도 ≥5km(near), 후속은 6km+.
        nr = max(1, getattr(cfg, "enemy_wave_ranks", 3))
        gap = getattr(cfg, "enemy_wave_gap", 1000.0 * getattr(cfg, "scale", 1.0))
        near = getattr(cfg, "enemy_wave_near", 4000.0 * getattr(cfg, "scale", 1.0))
        if dr:                                                        # ★ DR: 파상 '모양'만 변형(거리는 고정)
            ranks_opts = getattr(cfg, "domain_rand_ranks", (nr,))
            nr = int(rng.choice(ranks_opts))                         # 단(rank) 수 {2,3}
            gap = _dr_scale(gap, drf, rng)                           # 단 간격 ±frac
            # ★ near(첫 파 거리)는 흔들지 않는다: 흔들면 wave 만 다른 포메이션(~5450)보다 확 가까이
            #   (3600~4700) 와서 거리 불일치·근접 과다. 정형 near(4000) 고정 → 모양만 변형.
        far = near + (nr - 1) * gap                                   # 가장 먼 파 (예: 6000)
        rmax = world * 0.5 - margin
        far = min(far, rmax)                                          # 맵 안으로
        cnt = _split_counts(M, nr)
        # ★ 3-4-3: 가장 큰 그룹을 '중앙 방위'에 배치(치우침 방지 + 가운데 집중). nr=3,M=10 → [3,4,3].
        co = np.argsort(np.abs(np.arange(nr) - (nr - 1) / 2.0))      # 중앙 rank 부터
        cnt2 = np.empty(nr, np.int64); cnt2[co] = np.sort(cnt)[::-1]; cnt = cnt2
        # ★ 방위 균등 분산(7:3 치우침 방지): rank 별로 b0 ± spread 균등 → 각 rank 가 distinct 방위.
        spread = getattr(cfg, "enemy_wave_spread", 18.0)
        if dr:
            spread = _dr_scale(spread, drf, rng)                     # 단 간 부채 퍼짐 ±frac
        bearings = [(b0 + (k - (nr - 1) / 2.0) * spread) % 360 for k in range(nr)]
        radii = [min(rmax, near + k * gap) for k in range(nr)]        # 바깥으로 stagger
        pos = _grouped_spawn(center, world, margin, rng,
                             bearings=bearings, counts=cnt, radii=radii, jitter=jit)
        # ★ 양방향 radial clamp: jitter 가 끌어도 모든 적을 [near, far](예: 4~6km) 이내로 보장.
        v = pos - center[None, :]
        dd = np.hypot(v[:, 0], v[:, 1])
        scale = np.clip(dd, near, far) / np.maximum(dd, 1e-6)         # near↓ 바깥으로, far↑ 안쪽으로
        pos = center[None, :] + v * scale[:, None]
    elif mode == "grouped":
        # 기본 그룹전: G개 방위에 고르게(±무작위) 나눠 뭉쳐 온다
        bearings = [(b0 + 360.0 * g / G0 + rng.uniform(-20, 20)) % 360
                    for g in range(G0)]
        cnt = _split_counts(M, G0)
        pos = _grouped_spawn(center, world, margin, rng,
                             bearings=bearings, counts=cnt,
                             radii=[R0] * G0, jitter=jit)
    else:  # random — 느슨한 2~3 그룹 (옛 '완전 균등 산발'은 클러스터 3~4개로 정신없음 → 약화).
        #   2~3 방위에 고르게(±무작위) 나눠 느슨히 뭉침(jitter 약간 큼) → 클러스터 2~3개로 규제.
        g = int(rng.choice((2, 3)))
        bearings = [(b0 + 360.0 * i / g + rng.uniform(-30, 30)) % 360 for i in range(g)]
        cnt = _split_counts(M, g)
        pos = _grouped_spawn(center, world, margin, rng,
                             bearings=bearings, counts=cnt,
                             radii=[R0] * g, jitter=jit * 1.2)   # 느슨한 산포(그룹보다 약간 퍼짐)

    # 커리큘럼: 스폰 위치를 중심 쪽으로 frac 배 당김 (frac<1 → 더 가까이, 접촉 빠름)
    frac = getattr(cfg, "enemy_spawn_frac", 1.0)
    if frac < 1.0:
        pos = center + (pos - center) * frac

    hdg = _heading_to_center(pos, center)
    phase = rng.uniform(0, 2 * np.pi, size=M)
    return pos, hdg, phase


def spawn_allies(cfg: SimConfig = DEFAULT_CONFIG, rng=None):
    """
    아군 초기 상태 생성: **모선 바로 아래에 한 줄**로 옆(East)으로만 띄워 배치.
    전원 heading = cfg.ally_heading (기본 180°=South=아래) 로 모선 아래에 붙어 대기.
    반환: (pos [P,2], hdg [P]).
    """
    P = cfg.n_allies
    cx, cy = cfg.center
    # 옆(East)으로 균등 분산, 중앙(cx) 기준 대칭
    offs = (np.arange(P) - (P - 1) / 2.0) * cfg.ally_side_spacing
    x = cx + offs
    y = np.full(P, cy - cfg.ally_row_gap)          # 모선 중심에서 아래로
    pos = np.stack([x, y], axis=1).astype(np.float64)
    hdg = np.full(P, cfg.ally_heading, dtype=np.float64)   # 전원 아래 바라봄
    return pos, hdg
