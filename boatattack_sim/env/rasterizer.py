"""
boatattack_sim/env/rasterizer.py — 상태 → 멀티채널 래스터 관측 (CNN 점수맵 모드)

가변 개체 수(적 생멸·아군 소멸)를 "칠하기"로 흡수해 입력 크기를 (C,H,W) 로 고정한다.
순열 불변이 공짜로 확보되고, 국소 기하(접근 코리도·차단선)를 CNN 이 등변적으로 학습한다.

반환 구조 — **전역/배별 분리**가 핵심이다. 전역 채널은 배와 무관하므로 월드당 1회만 계산하고,
P 축 복제는 numpy 에서 하지 않는다(모델에서 torch expand 로 무복사 브로드캐스트).

  gmap  [N, Cg, H, W] float32   전역 공통 채널
  smap  [N, P, Cs, H, W] float32 배별 채널 (self_marker / self_intercept / self_valid)
  own   [N, P, F] float32       래스터로 표현하기 낭비인 스칼라 (위치·헤딩·그물·요격점)
  valid [N, P, H, W] bool       정책 마스크 (= smap 의 self_valid 와 동일 배열)

축 순서는 [ix, iy] = [x, y] (env/cnn_map.py 규약). 그릴 때만 전치.
"""
import numpy as np

from . import cnn_map as CM
from . import kinematics as K


OWN_F = 8   # 자기 pos2 · heading2 · 남은그물1 · doing_net1 · 배정요격점2


def global_channel_names(cfg):
    names = ["enemy_presence"]
    if getattr(cfg, "cnn_ch_enemy_vel", True):
        names += ["enemy_vx", "enemy_vy"]
    if getattr(cfg, "cnn_ch_enemy_threat", True):
        names += ["enemy_threat"]
    if getattr(cfg, "cnn_ch_ally", True):
        names += ["ally_presence"]
    if getattr(cfg, "cnn_ch_net", True):
        names += ["net_installed"]
    if getattr(cfg, "cnn_ch_origin", True):
        names += ["origin"]
    if getattr(cfg, "cnn_ch_annulus", True):
        names += ["annulus"]
    if _land_on(cfg):
        names += ["land"]
    return names


def _land_on(cfg):
    """육지 채널이 켜졌는가 — 마스터 스위치(land_obstacle)와 채널 플래그 둘 다 필요."""
    return bool(getattr(cfg, "land_obstacle", False)) and bool(getattr(cfg, "cnn_ch_land", True))


def ship_channel_names(cfg):
    names = ["self_marker", "self_intercept", "self_valid"]
    if getattr(cfg, "cnn_autoreg_refeed", False):
        names += ["partial_placement"]
    return names


def coord_channel_names(cfg):
    """CoordConv 채널 이름. r 은 x,y 에서 conv 로 합성 가능해 기본 제외(cnn_coord_r)."""
    if not getattr(cfg, "cnn_coord", True):
        return []
    names = ["coord_x", "coord_y"]
    if getattr(cfg, "cnn_coord_r", False):
        names += ["coord_r"]
    return names


def channel_names(cfg):
    """모델 입력 순서 그대로의 전체 채널 이름 (전역 → 배별 → CoordConv)."""
    return global_channel_names(cfg) + ship_channel_names(cfg) + coord_channel_names(cfg)


def obs_channels(cfg):
    """(Cg, Cs, Ccoord) — 모델이 stem in_channels 를 잡는 데 쓴다."""
    return (len(global_channel_names(cfg)), len(ship_channel_names(cfg)),
            len(coord_channel_names(cfg)))


# ── 전역 채널 ────────────────────────────────────────────────────────

def build_global_map(env):
    """전역 공통 채널 [N, Cg, H, W] float32."""
    cfg = env.cfg; N, M = env.N, env.M
    n = CM.grid_n(cfg)
    c = np.asarray(cfg.center, np.float64)
    chans = []

    # 0 enemy_presence — 픽셀 내 살아있는 적 수 / n_enemies
    cnt = CM.scatter_count(cfg, N, env.e_pos, env.e_alive)             # [N,n,n]
    chans.append((cnt / max(M, 1)).astype(np.float32))

    if getattr(cfg, "cnn_ch_enemy_vel", True):
        eh = K.heading_vec(env.e_hdg)                                   # [N,M,2] (sin,cos)
        sx = CM.scatter_count(cfg, N, env.e_pos, env.e_alive, eh[..., 0])
        sy = CM.scatter_count(cfg, N, env.e_pos, env.e_alive, eh[..., 1])
        den = np.maximum(cnt, 1.0)
        chans.append((sx / den).astype(np.float32))
        chans.append((sy / den).astype(np.float32))

    if getattr(cfg, "cnn_ch_enemy_threat", True):
        # 모선 도달 임박도: 가까울수록 1. 픽셀당 max (적이 겹쳐도 가장 위험한 값)
        d = np.hypot(env.e_pos[..., 0] - c[0], env.e_pos[..., 1] - c[1])   # [N,M]
        thr = 1.0 - np.clip(d / (cfg.enemy_speed * cfg.max_steps + 1e-6), 0.0, 1.0)
        out = np.zeros((N, n, n))
        ix, iy = CM.world_to_pix(cfg, env.e_pos)
        ni = np.broadcast_to(np.arange(N)[:, None], (N, M))
        m = env.e_alive
        np.maximum.at(out, (ni[m], ix[m], iy[m]), thr[m])
        chans.append(out.astype(np.float32))

    if getattr(cfg, "cnn_ch_ally", True):
        ac = CM.scatter_count(cfg, N, env.a_pos, env.a_alive)
        chans.append((ac / max(env.P, 1)).astype(np.float32))

    if getattr(cfg, "cnn_ch_net", True):
        chans.append(CM.downsample_max(cfg, env.net_installed))

    if getattr(cfg, "cnn_ch_origin", True):
        om = CM.origin_mask(cfg).astype(np.float32)
        chans.append(np.broadcast_to(om, (N, n, n)))

    if getattr(cfg, "cnn_ch_annulus", True):
        am = CM.annulus_mask(cfg).astype(np.float32)
        chans.append(np.broadcast_to(am, (N, n, n)))

    if _land_on(cfg):
        # ★ env 가 __init__ 에서 사이트별로 캐시해 둔 것을 재사용(에피소드 내 불변).
        #   지형 로테이션이 켜지면 월드마다 해역이 다르므로 사이트축을 월드로 갈라 준다.
        lm = env.land_map[env.world_site].astype(np.float32)         # [N,n,n]
        chans.append(np.broadcast_to(lm, (N, n, n)))

    return np.ascontiguousarray(np.stack(chans, axis=1), dtype=np.float32)


# ── 배별 채널 ────────────────────────────────────────────────────────

def build_ship_map(env, valid):
    """배별 채널 [N, P, Cs, H, W] float32. valid[N,P,n,n] = 정책 마스크."""
    cfg = env.cfg; N, P = env.N, env.P
    n = CM.grid_n(cfg)
    r = int(getattr(cfg, "cnn_marker_r", 1))
    Cs = len(ship_channel_names(cfg))
    out = np.zeros((N, P, Cs, n, n), np.float32)

    for p in range(P):                       # P ≤ 8 — 이 축만 파이썬 루프(스탬프 인덱싱 단순화)
        alive = env.a_alive[:, p]
        CM.stamp_max(out[:, p, 0], cfg, env.a_pos[:, p][:, None, :], alive[:, None], r)
        asg = (env._assign[:, p] >= 0) & alive
        CM.stamp_max(out[:, p, 1], cfg, env._assignI[:, p][:, None, :], asg[:, None], r)
    out[:, :, 2] = valid.astype(np.float32)
    if getattr(cfg, "cnn_autoreg_refeed", False):
        out[:, :, 3] = 0.0                   # partial_placement — 디코딩 중 모델이 갱신
    return out


def build_own(env):
    """own 스칼라 [N, P, OWN_F] float32."""
    cfg = env.cfg
    c = np.asarray(cfg.center, np.float64); e = CM.extent_m(cfg)
    apos = (env.a_pos - c) / e                                   # [N,P,2]
    ah = K.heading_vec(env.a_hdg)                                # [N,P,2]
    asg = (env._assign >= 0).astype(np.float64)                  # [N,P]
    In = (env._assignI - c[None, None, :]) / e * asg[..., None]  # 미배정=0
    own = np.stack([apos[..., 0], apos[..., 1], ah[..., 0], ah[..., 1],
                    env.a_nets / max(cfg.nets_per_ship, 1),
                    env.doing_net.astype(np.float64),
                    In[..., 0], In[..., 1]], axis=-1)
    return np.ascontiguousarray(own, dtype=np.float32)


def build_cnn_obs(env):
    """CNN 점수맵 관측 전체. env._cnn_valid_mask() 와 마스크를 공유한다
    (휴리스틱 BC 타깃이 -inf 픽셀에 떨어지지 않게 하는 유일한 보장)."""
    env._compute_assignment()                # 배정 요격점 갱신(마스크·own·렌더 일관)
    valid = env._cnn_valid_mask()            # [N,P,n,n] bool
    return {"gmap": build_global_map(env),
            "smap": build_ship_map(env, valid),
            "own": build_own(env),
            "valid": valid}
