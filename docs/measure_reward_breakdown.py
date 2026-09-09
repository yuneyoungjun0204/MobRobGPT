# -*- coding: utf-8 -*-
"""보상 항목별 실제 기여 측정 — 가중치가 아니라 '무엇이 그래디언트를 끄는가'.

env._rwd_breakdown 은 _reward() 안에서 항목별 월드평균을 채운다.
rollout_eval 은 snapshot/restore 라 상태를 바꾸지 않으므로, 매 결정마다
정책이 고른 행동으로 한 번 굴려 보상 분해를 읽어도 에피소드가 오염되지 않는다.
"""
import os, sys, json
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commander.unet_bridge import CommandedCnnEnv
from commander.rl_bridge import build_battlefield_defense
from commander.fallback import heuristic_plan

CKPT = "boatattack_sim/models/u-net_map.pt"


def run(episodes=6, steps=900, replan=25, seed0=100, mode="diversionary"):
    env = CommandedCnnEnv(CKPT, enemy_mode=mode, device="cpu")
    acc = defaultdict(list)
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        last = -10 ** 9
        for t in range(steps):
            if t - last >= replan:
                env.set_plan(heuristic_plan(build_battlefield_defense(env)))
                last = t
            env.step()
            if env._last_pix is None:
                continue
            if t % env.cfg.decision_period:
                continue                                   # 결정 시점만
            act = {"pix": env._last_pix[None]}
            if env._last_off is not None:
                act["offset"] = env._last_off[None]
            env.rollout_eval(act, period=180)   # 학습과 동일한 eval_period
            for k, v in env._rwd_breakdown.items():
                acc[k].append(v)
    return env, acc


if __name__ == "__main__":
    env, acc = run()
    n = len(acc["total"])
    print(f"결정 표본 {n} (6 에피소드 × diversionary)\n")
    rows = []
    for k, v in acc.items():
        a = np.array(v)
        rows.append((k, a.mean(), np.abs(a).mean(), a.std(), (a != 0).mean()))
    tot_abs = sum(r[2] for r in rows if r[0] != "total")
    rows.sort(key=lambda r: -r[2])
    print(f"{'항목':12s} {'평균':>9s} {'|평균|':>9s} {'표준편차':>9s} {'비영0%':>7s} {'|기여| 점유':>9s}")
    for k, m, am, sd, nz in rows:
        share = "" if k == "total" else f"{100*am/tot_abs:7.1f}%"
        print(f"{k:12s} {m:+9.3f} {am:9.3f} {sd:9.3f} {100*nz:6.1f}% {share:>9s}")
    json.dump({k: dict(mean=float(np.mean(v)), abs_mean=float(np.mean(np.abs(v))),
                       std=float(np.std(v)), nonzero=float(np.mean(np.array(v) != 0)))
               for k, v in acc.items()},
              open("reward_breakdown.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
