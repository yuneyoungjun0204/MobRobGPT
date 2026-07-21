"""
boatattack_sim/env/reward.py — 보상 (Unity 패리티 + GRPO용 포텐셜 shaping)

순수 numpy. GRPO는 후보 플랜당 **스칼라 1개**(decision_period 윈도우 누적 return)를 쓴다.
→ Unity DefenseRewardCalculator 패리티 '이벤트'(정답)에 potential-based shaping(Φ)을 더해
   25스텝 윈도우에도 변별 가능한 밀집 신호를 만든다.

설계 근거:
  · GRPO advantage = (r_i − mean(r))/std(r)  (그룹 상대, critic 없음) → per-step 보상 불필요.
  · 분산≈0(윈도우 내 무사건) 그룹은 신호 0 → shaping 으로 변별력 확보.
  · Φ shaping 은 potential-based(γΦ(s')−Φ(s)) 라 최적 정책을 바꾸지 않음(Ng et al. 1999).
  · 윈도우 끝 Φ = 잔여위협 추정 → 장기 신용 보강.
"""
import numpy as np

from .config import RewardCfg, DEFAULT_REWARD


# ── 위협 포텐셜 Φ (밀집 shaping의 핵심) ──────────────────────────────

def threat_potential(enemy_xy, alive, center, world_half: float,
                     cfg: RewardCfg = DEFAULT_REWARD) -> float:
    """살아있는 적이 모선에 가까울수록 위협↑.  Φ = −w · Σ 근접도.
      근접도_i = clip(1 − dist_i / world_half, 0, 1)  (모선에서 멀면 0, 붙으면 1)
    포획/저지로 적이 사라지거나 멀어지면 Φ가 오른다(=덜 위험) → +shaping.
    """
    enemy_xy = np.atleast_2d(np.asarray(enemy_xy, dtype=np.float64))
    alive = np.asarray(alive, dtype=bool)
    if enemy_xy.shape[0] == 0 or not alive.any():
        return 0.0
    c = np.asarray(center, dtype=np.float64)
    d = np.hypot(enemy_xy[:, 0] - c[0], enemy_xy[:, 1] - c[1])
    prox = np.clip(1.0 - d / max(world_half, 1e-6), 0.0, 1.0)
    return float(-cfg.w_threat * (prox * alive).sum())


# ── 결정 윈도우 보상 (후보 플랜 1개의 스칼라 return) ─────────────────

def window_reward(events: dict,
                  phi_start: float, phi_end: float,
                  cfg: RewardCfg = DEFAULT_REWARD,
                  *, steps: int = 25,
                  nets_total: int = 9,
                  terminal: bool = False,
                  n_remaining: int = 0) -> float:
    """
    decision_period 동안 일어난 일(events)을 스칼라 보상으로 합산 (팀 공유).

    events 키 (윈도우 동안의 증분):
      captures, breaches, ally_collisions, obstacle_collisions, nets_used,
      path_dist_norm (이동거리/이론최대, 0~1),
      wp_good, wp_bad  (도달가능·커버 / 도달불가·중복 개수)
    phi_start/phi_end : 윈도우 시작/끝의 위협 포텐셜 Φ.
    terminal/n_remaining : 윈도우에서 에피소드 종료 시 전멸(+)/잔존(−) 처리.
    """
    e = events
    r = 0.0
    # ── Unity 패리티 이벤트 ──
    r += cfg.r_capture        * e.get("captures", 0)
    r += cfg.r_breach         * e.get("breaches", 0)
    r += cfg.r_ally_collision * e.get("ally_collisions", 0)
    r += cfg.r_obstacle       * e.get("obstacle_collisions", 0)
    r += cfg.time_penalty     * steps
    # ── 경로/효율/그물/WP 품질 ──
    r += -cfg.w_path * float(e.get("path_dist_norm", 0.0))
    r += -cfg.w_net  * (e.get("nets_used", 0) / max(nets_total, 1))
    r += cfg.w_wp_good * e.get("wp_good", 0) - cfg.w_wp_bad * e.get("wp_bad", 0)
    # ── GRPO 밀집화: potential-based shaping ──
    r += cfg.gamma * phi_end - phi_start
    # ── 종료 보너스/페널티 ──
    if terminal:
        r += cfg.r_wipeout if n_remaining == 0 else cfg.r_survive * n_remaining
    return float(r)


def individual_capture_bonus(captures_by_ship, cfg: RewardCfg = DEFAULT_REWARD):
    """배별 포획 개별 보너스 [P]. 팀 보상에 더해 분산 정책 신용배분 보강."""
    return cfg.r_capture_indiv * np.asarray(captures_by_ship, dtype=np.float64)


# ── 벡터화 윈도우 보상 (N 월드 일괄, VecEnv/GRPO 용) ─────────────────

def window_reward_vec(captures, breaches, ally_collisions, nets_used,
                      path_dist_norm, phi_start, phi_end,
                      done, n_remaining, cfg: RewardCfg = DEFAULT_REWARD,
                      *, steps: int = 25, nets_total: int = 9,
                      obstacle_collisions=0.0):
    """[N] 배열 입력 → 팀 보상 [N]. window_reward 와 동일 공식의 벡터화판."""
    captures = np.asarray(captures, np.float64)
    breaches = np.asarray(breaches, np.float64)
    coll = np.asarray(ally_collisions, np.float64)
    obst = np.asarray(obstacle_collisions, np.float64)
    nu = np.asarray(nets_used, np.float64)
    pdn = np.asarray(path_dist_norm, np.float64)
    phi0 = np.asarray(phi_start, np.float64)
    phi1 = np.asarray(phi_end, np.float64)
    done = np.asarray(done, bool)
    n_rem = np.asarray(n_remaining, np.float64)

    r = (cfg.r_capture * captures + cfg.r_breach * breaches
         + cfg.r_ally_collision * coll + cfg.r_obstacle * obst
         + cfg.time_penalty * steps
         - cfg.w_path * pdn - cfg.w_net * nu          # 그물 1개당 절대 비용(낭비 억제)
         + cfg.gamma * phi1 - phi0)
    term = np.where(n_rem == 0, cfg.r_wipeout, cfg.r_survive * n_rem)
    r = r + np.where(done, term, 0.0)
    return r


def threat_potential_vec(enemy_xy, alive, center, world_half: float,
                         cfg: RewardCfg = DEFAULT_REWARD):
    """[N] 위협 포텐셜. enemy_xy [N,M,2], alive [N,M]."""
    enemy_xy = np.asarray(enemy_xy, np.float64)
    alive = np.asarray(alive, bool)
    c = np.asarray(center, np.float64)
    d = np.hypot(enemy_xy[..., 0] - c[0], enemy_xy[..., 1] - c[1])    # [N,M]
    prox = np.clip(1.0 - d / max(world_half, 1e-6), 0.0, 1.0)
    return (-cfg.w_threat * (prox * alive).sum(axis=1)).astype(np.float64)
