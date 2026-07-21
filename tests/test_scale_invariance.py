"""스케일 불변 검증 — 12.6km 해상 vs 33m 수조에서 정규화 관측이 동일한가?

같은 seed 로 두 스케일의 env 를 굴려 build_cell_obs() 출력을 직접 비교한다.
정규화가 전부 길이/길이라면 s 가 약분돼 **완전히 동일**해야 한다.
"""
import numpy as np
from boatattack_sim.env.config import SimConfig, RewardCfg
from boatattack_sim.env.defense_env import DefenseVecEnv

SEED = 7
N = 8            # 월드 수
STEPS = 6        # 결정 스텝 수


def make(scale_ws=None, mode="wave"):
    if scale_ws is None:
        cfg, r = SimConfig(), RewardCfg()
    else:
        base = SimConfig()
        s = scale_ws / base.world_size
        cfg = SimConfig.at_scale(world_size=scale_ws)
        r = RewardCfg(); r.apply_scale(s)
    cfg.cell_action = True
    cfg.n_allies = 3
    env = DefenseVecEnv(num_worlds=N, cfg=cfg, rcfg=r, enemy_mode=mode, seed=SEED)
    env.reset(seed=SEED)
    return env


def compare(mode):
    big = make(None, mode)
    small = make(33.0, mode)
    worst = {}
    for t in range(STEPS):
        ob, os_ = big.build_cell_obs(), small.build_cell_obs()
        assert ob.keys() == os_.keys()
        for k in ob:
            a, b = np.asarray(ob[k]), np.asarray(os_[k])
            if a.dtype == bool:
                d = float((a != b).mean())          # 마스크는 불일치 비율
            else:
                d = float(np.max(np.abs(a - b)))    # 실수는 최대 절대오차
            worst[k] = max(worst.get(k, 0.0), d)
        # 동일한 행동을 양쪽에 적용 (셀 인덱스는 무차원이라 그대로 공유 가능)
        C = big.n_cells
        rng = np.random.default_rng(100 + t)
        act = {"cells": rng.integers(0, C, size=(N, big.P, big.cfg.cell_nets))}
        big.step(act); small.step(act)
    return worst


print("=" * 62)
print(" 스케일 불변 검증 : world 12600m  vs  33m  (s = 1/381.8)")
print("=" * 62)
ok = True
for mode in ("wave", "grouped", "diversionary"):
    try:
        w = compare(mode)
    except Exception as e:
        print(f"\n[{mode}] 실행 실패: {type(e).__name__}: {e}")
        ok = False
        continue
    print(f"\n[{mode}]  (정규화 관측 최대 절대오차)")
    for k, v in sorted(w.items()):
        tag = "OK " if v < 1e-9 else ("~  " if v < 1e-6 else "X  ")
        if v >= 1e-9:
            ok = False
        print(f"   {tag} {k:12s} {v:.3e}")

print()
print("=" * 62)
print(" 결과:", "완전 동일 — 스케일 불변 성립" if ok else "불일치 발견 — 아래 항목 조사 필요")
print("=" * 62)
