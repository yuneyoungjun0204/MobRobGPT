"""
boatattack_sim/env/scaling.py — 스케일 계약 (실세계 ↔ 시뮬 단위)

**관측은 무차원이다.** 래스터 채널·own 스칼라는 전부 `extent`·`n_enemies`·`P`·
`enemy_speed·max_steps` 같은 **config 상수로 나눈 비율**이라, 맵이 12.6 km 든 12.6 m 든
같은 기하면 같은 텐서가 나온다 (`tests/test_scale.py` 가 k=1e-3 에서 이를 검증한다).

따라서 수조든 실해역이든 **모델을 다시 학습할 필요가 없다.** 필요한 것은 두 가지뿐이다.

  ① 길이 스케일 S   [sim-m / real-m]  = world_size / (실제 운용 박스 한 변)
  ② 시간 스케일 dt  [s / sim-step]    = cfg.ally_speed / (실제 아군 속도 × S)

②가 ①에서 따라 나오는 게 핵심이다. **모델을 스케일하는 게 아니라 시계를 스케일한다.**
아군이 sim 한 스텝에 `ally_speed`(=6) sim-m 를 가야 하므로, 실제 속도 v[m/s] 로 그만큼
가는 데 걸리는 실시간이 곧 한 스텝의 길이다.

두 가지 쓰임이 있다.

  · `SimScale`      — **운용용**. config 는 학습 당시 그대로 두고 센서만 sim 단위로 옮긴다.
                      권장 경로다 (학습된 기하를 한 톨도 안 건드린다).
  · `scale_config`  — **검증용**. config 자체를 k배 해도 관측이 동일함을 보이는 데 쓴다.
                      운용에서 이걸 쓸 이유는 없다 — 이득 없이 위험만 는다.

⚠ 스케일을 맞춰도 **무차원 비(Π)가 다르면 정책은 off-distribution 이다.**
  `SimScale.report()` 가 그 비들을 실제 설비값과 대조해 찍는다.
"""
import dataclasses
import math

import numpy as np

from .config import SimConfig, RewardCfg


# ── 차원 분류 ────────────────────────────────────────────────────────
# 길이(m). 전부 같은 계수로 곱해야 기하가 상사(similar)로 유지된다.
# ★ cell_size 는 property(world_size/grid_size)라 자동으로 따라온다 — 넣으면 안 된다.
#   center 도 property(world_size/2) 라 마찬가지.
LENGTH_FIELDS = (
    "world_size", "cnn_extent", "action_grid_half",
    "cell_spacing", "cell_r_min", "cell_r_max",
    "cnn_gate_r", "cnn_dup_r", "cnn_joint_mask_r",
    "mothership_radius", "moback_size", "ship_len", "ship_wid", "enemy_size",
    "enemy_evade_look", "enemy_spawn_margin", "enemy_spawn_radius",
    "enemy_group_jitter", "enemy_wave_gap", "enemy_wave_near",
    "arrive_radius", "ally_row_gap", "ally_side_spacing",
    "ally_collision_radius", "ally_mother_radius",
    "net_max_len", "net_probe_range",
    "avoid_steer_cap", "wp_repel_r", "wp_repel_mother_r",
    "route_step", "wp_adjust_max", "fan_curve_max",
    "aux_radial_max", "aux_lateral_max",
    "norm_k_enemy", "norm_k_mother", "norm_k_ally",
    "enemy_land_range",
)

# 속도(m/step). 길이와 **같은 계수**로 곱한다 (dt 는 무차원 취급이므로).
SPEED_FIELDS = ("ally_speed",)

# 보상/배정 쪽에도 미터 단위가 있다. 놓치면 배정이 스케일마다 달라진다.
REWARD_LENGTH_FIELDS = (
    "place_scale", "clear_net_r", "avoid_r", "avoid_dmin",
    "mother_avoid_r", "assign_sticky_bonus",
)

# 스케일하면 **안 되는** 것들 — 무차원이거나 개수/각도/픽셀이다.
#   grid_size, cnn_grid_n, net_width(cell), cnn_marker_r(px), land_margin_px,
#   max_steps, dt, decision_period, n_allies, n_enemies, nets_per_ship,
#   *_frac, *_mult, *_gain, *_deg, *_angle, cell_off_scale, spawn_phase_lo


def scale_config(cfg: SimConfig, k: float, rcfg: RewardCfg = None):
    """모든 길이·속도를 k배 한 config 사본. **검증용**(운용은 `SimScale` 을 써라).

    k=1e-3 이면 12.6 km 맵이 12.6 m 수조가 된다. 관측은 바뀌지 않아야 한다.
    반환: `cfg2` 또는 `(cfg2, rcfg2)` (rcfg 를 주면).
    """
    k = float(k)
    upd = {f: getattr(cfg, f) * k for f in LENGTH_FIELDS if hasattr(cfg, f)}
    upd.update({f: getattr(cfg, f) * k for f in SPEED_FIELDS if hasattr(cfg, f)})
    cfg2 = dataclasses.replace(cfg, **upd)
    # cell_size 는 world_size/grid_size 와 반드시 일치해야 한다(물리격자 해상도).
    assert abs(cfg2.cell_size - cfg2.world_size / cfg2.grid_size) < 1e-9 * max(cfg2.world_size, 1.0), \
        "cell_size != world_size/grid_size — 물리격자가 어긋난다"
    if rcfg is None:
        return cfg2
    rupd = {f: getattr(rcfg, f) * k for f in REWARD_LENGTH_FIELDS if hasattr(rcfg, f)}
    return cfg2, dataclasses.replace(rcfg, **rupd)


# ── 운용용 좌표/시간 브리지 ──────────────────────────────────────────

class SimScale:
    """실세계 ↔ 시뮬 단위 고정 affine + 시계.

    config 는 **학습 당시 그대로** 두고, 센서 값만 sim 단위로 옮긴다.
    한 번 만들면 운용 내내 재사용한다 (매 tick 새로 잡으면 관측이 흔들려 정책이 진동한다).

    Parameters
    ----------
    cfg : SimConfig            체크포인트에서 복원한 그대로
    span_real : float          sim 맵 한 변(world_size)에 대응시킬 **실제 거리(m)**
                               실해역이면 12600, 10 m 수조면 10.0
    v_ally_real : float        실제 아군 순항속도 [m/s]
    anchor : (lat, lon) | None 위경도를 쓸 때의 앵커. 맵 정중앙(=모선)에 대응한다.
                               None 이면 위경도 변환 없이 로컬 ENU 만 쓴다.
    enu_origin : (x, y)        로컬 ENU 를 쓸 때 맵 정중앙에 대응할 실좌표 원점 [m]
    """

    def __init__(self, cfg, span_real, v_ally_real, anchor=None, enu_origin=(0.0, 0.0)):
        self.cfg = cfg
        self.span_real = float(span_real)
        self.v_ally_real = float(v_ally_real)
        self.anchor = None if anchor is None else (float(anchor[0]), float(anchor[1]))
        self.enu_origin = (float(enu_origin[0]), float(enu_origin[1]))

        self.S = float(cfg.world_size) / self.span_real          # sim-m per real-m
        self.half = float(cfg.world_size) * 0.5                  # sim 중앙 = 모선
        # 한 sim step 이 실제 몇 초인가 — 아군이 스텝당 ally_speed sim-m 를 가야 한다
        self.dt_real = float(cfg.ally_speed) / max(self.v_ally_real * self.S, 1e-12)
        self.period_real = self.dt_real * float(cfg.decision_period)

        if self.anchor is not None:
            self.mpd_lat = 111_320.0
            self.mpd_lon = 111_320.0 * math.cos(math.radians(self.anchor[0]))
        else:
            self.mpd_lat = self.mpd_lon = None

    # ── 위치 ─────────────────────────────────────────────────────────
    def enu_to_sim(self, xy):
        """실 로컬 ENU [...,2] (m, x=East y=North) → sim world [...,2]."""
        a = np.asarray(xy, np.float64)
        return (a - np.asarray(self.enu_origin)) * self.S + self.half

    def sim_to_enu(self, xy):
        a = np.asarray(xy, np.float64)
        return (a - self.half) / self.S + np.asarray(self.enu_origin)

    def ll_to_sim(self, lat, lon):
        """WGS84 → sim world [...,2]. 앵커 위도로 고정한 등거리 근사."""
        if self.anchor is None:
            raise ValueError("anchor 없이 위경도 변환을 쓸 수 없다")
        la0, lo0 = self.anchor
        x = (np.asarray(lon, np.float64) - lo0) * self.mpd_lon
        y = (np.asarray(lat, np.float64) - la0) * self.mpd_lat
        return np.stack([x, y], -1) * self.S + self.half

    def sim_to_ll(self, xy):
        """sim world [...,2] → (lat, lon)."""
        if self.anchor is None:
            raise ValueError("anchor 없이 위경도 변환을 쓸 수 없다")
        la0, lo0 = self.anchor
        a = (np.asarray(xy, np.float64) - self.half) / self.S
        return la0 + a[..., 1] / self.mpd_lat, lo0 + a[..., 0] / self.mpd_lon

    # ── 속도/길이 편의 ───────────────────────────────────────────────
    def speed_to_sim(self, v_real):
        """실 속도 [m/s] → sim 속도 [sim-m/step]."""
        return np.asarray(v_real, np.float64) * self.S * self.dt_real

    def len_to_sim(self, d_real):
        return np.asarray(d_real, np.float64) * self.S

    def len_to_real(self, d_sim):
        return np.asarray(d_sim, np.float64) / self.S

    # ── 무차원 비 점검 ───────────────────────────────────────────────
    def pi_groups(self):
        """학습 무대가 정의하는 무차원 비들. 실 설비가 이 값들을 못 맞추면 off-distribution."""
        c = self.cfg
        e = float(getattr(c, "cnn_extent", None) or c.world_size * 0.5)
        va = float(c.ally_speed)
        turn = math.radians(float(c.ally_max_turn))
        r_turn = va * float(c.dt) / turn if turn > 0 else float("inf")
        return {
            "이동/결정 (va·T/extent)":      va * c.decision_period / e,
            "속도비 (ve/va)":               float(c.enemy_speed_mult),
            "선회반경/extent":              r_turn / e,
            "선회반경/선체길이":            r_turn / float(c.ship_len),
            "그물길이/extent":              float(c.net_max_len) / e,
            "환형 내/외 (r_min,r_max)/extent": (float(c.cell_r_min) / e, float(c.cell_r_max) / e),
            "적스폰반경/extent":            float(c.enemy_spawn_radius) / e,
            "픽셀/extent":                  2.0 / int(getattr(c, "cnn_grid_n", 50)),
            "선체길이/extent":              float(c.ship_len) / e,
            "충돌반경/extent":              float(c.ally_collision_radius) / e,
            "도착반경/extent":              float(c.arrive_radius) / e,
            "threat 분모 (ve·max_steps/extent)": c.enemy_speed * c.max_steps / e,
        }

    def report(self, r_turn_real=None, v_enemy_real=None, loa_real=None):
        """실 설비값을 주면 학습 무대와 대조해 사람이 읽을 표를 만든다.
        r_turn_real: 실제 선회반경[m] · v_enemy_real: 실제 위협 속도[m/s] · loa_real: 실제 선체길이[m]"""
        c = self.cfg
        pi = self.pi_groups()
        L = [f"[scale] S = {self.S:.4g} sim-m/real-m  (맵 {c.world_size:.0f} sim-m ↔ 실제 {self.span_real:g} m)",
             f"[scale] dt = {self.dt_real:.4g} s/step  ·  결정주기 = {self.period_real:.4g} s "
             f"({c.decision_period} step)",
             f"[scale] 실제 1 픽셀 = {self.len_to_real(2 * (getattr(c,'cnn_extent',None) or c.world_size*0.5) / c.cnn_grid_n):.4g} m",
             f"[scale] 실제 그물 최대길이 = {self.len_to_real(c.net_max_len):.4g} m",
             f"[scale] 실제 요격 환형 = {self.len_to_real(c.cell_r_min):.4g} ~ "
             f"{self.len_to_real(c.cell_r_max):.4g} m",
             "[scale] ── 무차원 비 (학습 무대) ──"]
        for k, v in pi.items():
            L.append(f"[scale]   {k:34s} = " +
                     (", ".join(f"{x:.4g}" for x in v) if isinstance(v, tuple) else f"{v:.4g}"))
        warn = []
        if v_enemy_real is not None:
            got = v_enemy_real / self.v_ally_real
            if abs(got - c.enemy_speed_mult) > 0.15 * c.enemy_speed_mult:
                warn.append(f"속도비 {got:.2f} vs 학습 {c.enemy_speed_mult:.2f} — 재학습 검토")
        if r_turn_real is not None:
            need = self.len_to_real(pi["선회반경/extent"] *
                                    (getattr(c, "cnn_extent", None) or c.world_size * 0.5))
            if r_turn_real > need * 1.3:
                warn.append(f"선회반경 {r_turn_real:.3g} m > 허용 {need:.3g} m "
                            f"— WP 추종이 학습과 달라진다 (ally_max_turn 재조정 또는 재학습)")
        if loa_real is not None:
            got = self.len_to_sim(loa_real) / c.ship_len
            if not (0.6 < got < 1.7):
                warn.append(f"선체길이 비 {got:.2f}× — 충돌·포획 판정 스케일이 어긋난다")
        for w in warn:
            L.append(f"[scale] ⚠ {w}")
        if not warn:
            L.append("[scale] ✓ 대조한 항목은 학습 무대와 정합")
        return "\n".join(L)
