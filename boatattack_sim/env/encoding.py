"""
boatattack_sim/env/encoding.py — 액션 디코드 / 관측 정규화 단일 소스 (전이 계약 SoT)

순수 numpy (torch 무의존). 학습·시뮬·Unity export 가 동일 함수를 써서
[-1,1] 정규화 좌표 ↔ world 좌표 변환을 정확히 동일하게 재현한다.
(항해사모사 encoding.py 의 encode/decode 대칭쌍 패턴 차용.)

핵심 불변식: **격자 해상도와 무관**. 액션은 저차원 [-1,1] 좌표이며,
            여기서 world(m)로 디코드한 뒤에야 grid.py 가 rasterize 한다.
"""
import numpy as np


# ── 액션 좌표 디코드/인코드 (맵 기준 절대 정규화) ────────────────────
#   가장 단순한 1차 규약: [-1,1] → 맵 전체 [0, world_size].
#   (후속: 모선/클러스터 기준 상대 디코드로 교체 가능 — 같은 SoT만 유지하면 됨.)

def decode_wp(norm_xy: np.ndarray, world_size: float) -> np.ndarray:
    """[-1,1]^2 정규화 WP → world 좌표 (m).
    norm_xy : [..., 2] in [-1,1]
    반환    : [..., 2] in [0, world_size]
    """
    norm_xy = np.clip(np.asarray(norm_xy, dtype=np.float64), -1.0, 1.0)
    return (norm_xy * 0.5 + 0.5) * world_size


def encode_wp(world_xy: np.ndarray, world_size: float) -> np.ndarray:
    """world 좌표 (m) → [-1,1]^2 (decode_wp 의 역함수)."""
    world_xy = np.asarray(world_xy, dtype=np.float64)
    return np.clip(world_xy / world_size * 2.0 - 1.0, -1.0, 1.0)


# ── 관측 정규화 (BoatAttack Unity 패리티 — 공식·부호 규약 동일) ──────
#   ★ 단일 소스: obs 인코딩·전이 export 가 모두 이 함수들을 쓴다.
#   k(관심반경)·맵 스케일만 우리 10km 에 맞춰 config 에서 주입 (형 패리티, A 방식).
#   좌표: nav (x=East, y=North, hdg 0=North CW+) ≡ Unity (x, z, 0=+Z CW+), z↔y.

def norm_range(d: np.ndarray, k: float) -> np.ndarray:
    """거리 정규화 — **멀수록 1**.  d/(|d|+k) ∈ [0,1).  d=k 에서 0.5.
    유리함수 → 근거리 해상도↑·원거리 포화. 부호전환 없음(안정)."""
    d = np.asarray(d, dtype=np.float64)
    return d / (np.abs(d) + k)


def norm_close(d: np.ndarray, k: float) -> np.ndarray:
    """거리 정규화 — **가까울수록 1**.  k/(d+k) ∈ (0,1].  d=k 에서 0.5.  (모선 거리용)"""
    d = np.asarray(d, dtype=np.float64)
    return k / (d + k)


# norm_dist: 하위호환 별칭 (= norm_range)
norm_dist = norm_range


def signed_bearing(fwd: np.ndarray, to: np.ndarray) -> np.ndarray:
    """방위각 — **sqrt 압축 + 좌(−)/우(+) 부호**.  반환 [-1,1].
      0 = 정면, ±1 = 정후방.  sqrt(ang/180) 로 정면 부근 분해능↑ (조준/차단 민감).
    ★ 부호(우현+/좌현−): 우리 nav 는 **우수계(x=East,y=North)** 라 Unity 좌수계(x,z)와
      cross 부호가 반대 → 여기서 한 번만 매핑(cross<0 = 우현 = +).  (좌표 규약 통일 지점)
    fwd, to : [...,2] 벡터 (정규화 불필요)."""
    fwd = np.asarray(fwd, dtype=np.float64); to = np.asarray(to, dtype=np.float64)
    fn = np.linalg.norm(fwd, axis=-1); tn = np.linalg.norm(to, axis=-1)
    denom = fn * tn
    dot = (fwd * to).sum(-1)
    cosang = np.clip(np.divide(dot, denom, out=np.zeros_like(denom * 1.0),
                               where=denom > 1e-12), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))                 # 0..180
    n = np.sqrt(ang / 180.0)                            # sqrt 압축 (정면=0)
    cross = fwd[..., 0] * to[..., 1] - fwd[..., 1] * to[..., 0]
    return np.where(cross <= 0, n, -n)                  # 우현(+)/좌현(−), nav 우수계 기준


def heading_cossin(from_deg: np.ndarray, to_deg: np.ndarray):
    """상대 헤딩차 → **(cos, sin) 2채널**.  각도 모호성 제거(유일 표현).
    반환: (cos δ, sin δ),  δ = wrap180(to − from)."""
    d = np.deg2rad((np.asarray(to_deg, np.float64) - np.asarray(from_deg, np.float64)
                    + 180.0) % 360.0 - 180.0)
    return np.cos(d), np.sin(d)


def spread_norm(spread_deg: np.ndarray) -> np.ndarray:
    """클러스터 각도 스프레드 정규화.  clip(spread/90, 0, 1).  (그물 폭 신호)"""
    return np.clip(np.asarray(spread_deg, np.float64) / 90.0, 0.0, 1.0)


def norm_angle_sincos(deg: np.ndarray) -> np.ndarray:
    """각도(deg) → (sin, cos) 쌍. (절대 heading 인코딩용; 헤딩'차'는 heading_cossin 사용)"""
    r = np.deg2rad(np.asarray(deg, dtype=np.float64))
    return np.stack([np.sin(r), np.cos(r)], axis=-1)


# ── 액션 → 경로 디코드 (egocentric 상대, 벡터화) ─────────────────────
#   임의 leading dim(...)에 대해 동작. env(N,P)·단일에이전트 모두 같은 함수 사용 (SoT).

def decode_plan(wp, ship_xy, ship_hdg, net_after, net_dir,
                wp_max_len, net_max_len, world_size):
    """
    egocentric 상대 액션 → 절대 경유 WP 좌표 + 그물 시작/끝.
      wp        [...,K,2]  각 슬롯 [-1,1]² (방향·크기). |v|·wp_max_len 으로 클램프.
      ship_xy   [...,2]    선박 현재 위치 (체인 시작)
      ship_hdg  [...]      (미사용; 크기 0이면 제자리라 방향 무관)
      net_after [...]      그물 삽입 위치 0..K (0=선박앞, k=k번째 WP 뒤)
      net_dir   [...,2]    그물 방향 ([-1,1]²; 길이는 net_max_len 고정)
    반환:
      transit_xy [...,K,2] 절대 경유 WP (직전점 기준 체인, 간격 ≤ wp_max_len)
      net_start  [...,2]   그물 시작점 (net_after 위치)
      net_end    [...,2]   그물 끝점 = start + dir·net_max_len
    """
    wp = np.asarray(wp, dtype=np.float64)
    ship_xy = np.asarray(ship_xy, dtype=np.float64)
    K = wp.shape[-2]
    prev = ship_xy.copy()
    out = np.empty_like(wp)
    for k in range(K):
        v = wp[..., k, :]                                   # [...,2]
        nrm = np.linalg.norm(v, axis=-1, keepdims=True)     # [...,1]
        mag = np.minimum(nrm, 1.0) * wp_max_len             # |v|≤1 → 0..wp_max_len
        dirv = v / np.maximum(nrm, 1e-9)                    # 단위방향 (|v|≈0이면 mag≈0)
        p = np.clip(prev + dirv * mag, 0.0, world_size)
        out[..., k, :] = p
        prev = p
    # 그물 시작 = net_after 위치의 점 (0=선박, else transit[net_after-1])
    na = np.asarray(net_after)
    idx = np.clip(na - 1, 0, K - 1)
    gathered = np.take_along_axis(out, idx[..., None, None], axis=-2)[..., 0, :]
    net_start = np.where((na == 0)[..., None], ship_xy, gathered)
    nd = np.asarray(net_dir, dtype=np.float64)
    ndn = nd / np.maximum(np.linalg.norm(nd, axis=-1, keepdims=True), 1e-9)
    net_end = np.clip(net_start + ndn * net_max_len, 0.0, world_size)
    return out, net_start, net_end
