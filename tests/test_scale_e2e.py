"""스케일 불변 종단 검증 — 실제 셀 정책을 12.6km / 33m 두 스케일에서 한 에피소드 굴려
포획/돌파/충돌이 같게 나오는지 본다. (관측이 동일하면 결정도 동일 → 결과도 동일해야 함)"""
import numpy as np, torch
from boatattack_sim.env.config import SimConfig, RewardCfg
from boatattack_sim.env.defense_env import DefenseVecEnv
from boatattack_sim.model.cell_actor import CellPointerActor
from commander.cell_bridge import load_cell_actor, cell_obs_to_torch

CKPT = "30_model/wave/best.pt"
SEED = 3
N = 16
MODE = "wave"


def build(ws=None):
    actor, cfg = load_cell_actor(CKPT, device="cpu")
    r = RewardCfg()
    if ws is not None:
        s = ws / cfg.world_size
        cfg.apply_scale(s)
        r.apply_scale(s)
    cfg.n_allies = 3
    env = DefenseVecEnv(num_worlds=N, cfg=cfg, rcfg=r, enemy_mode=MODE, seed=SEED)
    env.reset(seed=SEED)
    return env, actor


def run(ws, steps=90):
    env, actor = build(ws)
    acc = {k: 0.0 for k in ("captures", "breaches",
                            "obstacle_collisions", "ally_collisions", "net_touches")}
    for _ in range(steps):
        ob = env.build_cell_obs()
        ot = cell_obs_to_torch(ob, "cpu")
        with torch.no_grad():
            pr, _ = actor(ot, None)
            g = actor.greedy_joint(pr, env.N, env.P,
                                   cell_world=None, mask_radius=0)
        cells = g["cells"].view(env.N, env.P, -1).cpu().numpy()
        _, _, _, _, info = env.step({"cells": cells})
        for k in acc:
            acc[k] += float(info[k].sum())
    return {k: round(v, 6) for k, v in acc.items()}


base = run(None)
small = run(33.0)
print("=" * 58)
print(" 종단 검증 — 동일 정책 / 동일 seed / 서로 다른 스케일")
print("=" * 58)
print(f"  12600m : {base}")
print(f"     33m : {small}")
print("=" * 58)
print(" 결과:", "동일 — 스케일 불변" if base == small else "차이 발생")
