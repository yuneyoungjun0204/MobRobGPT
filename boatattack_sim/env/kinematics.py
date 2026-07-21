"""
boatattack_sim/env/kinematics.py — 운동학 (PD 경로추종 + 적 전진) + 기하 유틸

순수 numpy (torch 무의존). 점질량 + 선회율/속도 한계 PD. **물리 복제 금지.**
좌표계: nav (x=East, y=North, hdg 0°=North, CW+). 이동 = (sin·v, cos·v).

차용(항해사모사 colregs.py): wrap180, compute_cpa, NeighborTracker(EMA 속도추정).
"""
import numpy as np


# ── 기하 유틸 (순수) ─────────────────────────────────────────────────

def wrap180(deg):
    """각도를 (-180, 180] 으로 정규화. 스칼라/배열 모두."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def bearing_deg(src_xy: np.ndarray, dst_xy: np.ndarray):
    """src → dst 방위각 (deg, nav: 0=North, CW+). 벡터화 [K]."""
    d = np.asarray(dst_xy, dtype=np.float64) - np.asarray(src_xy, dtype=np.float64)
    return np.degrees(np.arctan2(d[..., 0], d[..., 1]))


def heading_vec(hdg_deg):
    """heading(deg) → 단위 진행벡터 [...,2] = (sin, cos)."""
    r = np.deg2rad(np.asarray(hdg_deg, dtype=np.float64))
    return np.stack([np.sin(r), np.cos(r)], axis=-1)


def compute_cpa(p_rel, v_rel):
    """상대위치/상대속도로 CPA. 반환 (d_cpa, t_cpa). v_rel≈0이면 (현재거리, inf).
    스칼라 입력 가정 (선박쌍 1개). 벡터 평가는 호출부에서 broadcast."""
    p = np.asarray(p_rel, dtype=np.float64); v = np.asarray(v_rel, dtype=np.float64)
    vv = float(v @ v)
    d_now = float(np.hypot(p[0], p[1]))
    if vv < 1e-9:
        return d_now, float("inf")
    t = -float(p @ v) / vv
    if t < 0:
        return d_now, t
    c = p + v * t
    return float(np.hypot(c[0], c[1])), t


# ── PD 경로 추종 (아군) ──────────────────────────────────────────────

def pd_follow(pos, hdg, target, speed, max_turn, dt,
              turn_gain=0.6, slow_min=0.30, arrive_radius=50.0):
    """
    아군 1스텝 PD 추종 (벡터화 가능: pos/target [K,2], hdg/speed [K]).
      · 목표 WP 방위로 비례선회(±max_turn) — 배는 천천히 돈다.
      · 선회각↑ → cos 감속 (급선회 시 느려짐, 하한 slow_min).
      · arrive_radius 도착 시 arrived=True (상위에서 다음 WP 로 전환).
    반환: (pos_new [K,2], hdg_new [K], arrived [K] bool)
    """
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    target = np.atleast_2d(np.asarray(target, dtype=np.float64))
    hdg = np.atleast_1d(np.asarray(hdg, dtype=np.float64))
    speed = np.atleast_1d(np.asarray(speed, dtype=np.float64))

    d = target - pos
    dist = np.hypot(d[:, 0], d[:, 1])
    brg = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    err = wrap180(brg - hdg)                          # heading 오차 (deg)

    turn = np.clip(turn_gain * err, -max_turn, max_turn)
    hdg_new = (hdg + turn * dt) % 360.0

    slow = np.clip(np.cos(np.deg2rad(err)), slow_min, 1.0)   # 급선회 감속
    step = speed * slow * dt
    step = np.minimum(step, dist)                     # WP 오버슈트 방지

    hr = np.deg2rad(hdg_new)
    pos_new = pos + np.stack([np.sin(hr), np.cos(hr)], axis=1) * step[:, None]
    arrived = dist <= arrive_radius
    return pos_new, hdg_new, arrived


# ── 적 전진 (모선 방향 등속 + 위빙) ──────────────────────────────────

def enemy_step(pos, hdg, target_xy, speed, max_turn, t,
               weave_amp=14.0, weave_period=32.0, phase=None, dt=1.0, evade=None):
    """
    적 1스텝 전진 (벡터화: pos [M,2], hdg [M]).
      · 모선(target_xy) 방위로 선회(±max_turn) + 사인 위빙 + evade(그물 회피 조향 deg, [M]).
    반환: (pos_new [M,2], hdg_new [M])
    """
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    hdg = np.atleast_1d(np.asarray(hdg, dtype=np.float64))
    M = pos.shape[0]
    if M == 0:
        return pos, hdg
    speed = np.broadcast_to(np.asarray(speed, dtype=np.float64), (M,))
    tgt = np.broadcast_to(np.asarray(target_xy, dtype=np.float64), (M, 2))
    if phase is None:
        phase = np.zeros(M)
    ev = np.zeros(M) if evade is None else np.broadcast_to(np.asarray(evade, np.float64), (M,))

    d = tgt - pos
    brg = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    weave = weave_amp * np.sin(2.0 * np.pi * t / weave_period + phase)
    err = wrap180(brg + weave + ev - hdg)
    turn = np.clip(err, -max_turn, max_turn)
    hdg_new = (hdg + turn * dt) % 360.0

    hr = np.deg2rad(hdg_new)
    pos_new = pos + np.stack([np.sin(hr), np.cos(hr)], axis=1) * (speed * dt)[:, None]
    return pos_new, hdg_new


# ── 타선/적 속도 추정 (EMA + 최근접 게이팅) ──────────────────────────

class NeighborTracker:
    """ID 없는 적 위치열에서 절대속도 추정 (최근접 매칭 + EMA 평활).
    (항해사모사 NeighborTracker 차용 — 분류·CPA 경계 깜빡임 완화.)"""

    def __init__(self, gate_m: float = 600.0, vel_ema: float = 0.3):
        self.gate = gate_m
        self.alpha = vel_ema
        self.tracks = []     # [{pos, vel}]

    def update(self, positions):
        out = []
        used = [False] * len(self.tracks)
        new_tracks = []
        for p in positions:
            best, bi = self.gate, -1
            for ti, tr in enumerate(self.tracks):
                if used[ti]:
                    continue
                dd = float(np.hypot(p[0] - tr["pos"][0], p[1] - tr["pos"][1]))
                if dd < best:
                    best, bi = dd, ti
            if bi >= 0:
                prev = self.tracks[bi]["pos"]; pv = self.tracks[bi]["vel"]
                raw = (p[0] - prev[0], p[1] - prev[1])
                vel = (self.alpha * raw[0] + (1 - self.alpha) * pv[0],
                       self.alpha * raw[1] + (1 - self.alpha) * pv[1])
                used[bi] = True
                new_tracks.append({"pos": p, "vel": vel}); out.append((p, vel))
            else:
                new_tracks.append({"pos": p, "vel": (0.0, 0.0)})
                out.append((p, (0.0, 0.0)))
        self.tracks = new_tracks
        return out

    def reset(self):
        self.tracks = []
