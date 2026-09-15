# -*- coding: utf-8 -*-
"""15채널 관측 전개도 — 논문 Fig.(fig:observation) 용, 5x3 격자, 흰 배경, 영문 라벨.

왜 새로 만드는가
    기존 fig_channels.png 는 디버그 대시보드였다(검은 배경, 4x4 에 15장이라 한 칸 빔,
    한국어 제목, 패널마다 min/max/비영 픽셀 수, 100x100 렌더라 캡션에 해명이 필요).
    논문 도판은 (1) 실제 학습 설정인 50x50 그대로, (2) 15 = 5x3 으로 딱 맞게,
    (3) 채널 이름만 영문으로, (4) 흰 배경에 절제된 컬러맵.

컬러맵 규칙
    부호 없는 채널(존재·마스크·위협·거리) : Greys   — 0 흰색, 값이 클수록 검게
    부호 있는 채널(v_x, v_y, coord_x, coord_y) : RdBu_r — 0 흰색, 음 파랑, 양 빨강
    흰 배경에서 '없음'이 흰색이라야 성긴 채널(적 8픽셀)이 점으로 보인다.

축 규약은 cnn_map 단일 소스를 따른다 — 배열 [ix,iy] 를 전치해 origin='lower' 로 그린다.

사용:
    python tools/make_channel_figure.py                       # 기본: 통영_매물도북, wave, seed 18, t=150
    python tools/make_channel_figure.py --site 신안_홍도북 --enemy diversionary --seed 1 --t 200
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

#: 코드 채널 이름 → 도판 라벨. 코드 이름은 캡션에서 대응시키므로 짧은 영문만 둔다.
LABEL = {
    "enemy_presence": "enemy presence",
    "enemy_vx": "enemy $v_x$",
    "enemy_vy": "enemy $v_y$",
    "enemy_threat": "enemy threat",
    "ally_presence": "ally presence",
    "net_installed": "net installed",
    "origin": "origin",
    "annulus": "annulus",
    "land": "land",
    "self_marker": "self marker",
    "self_intercept": "self intercept",
    "self_valid": "self valid",
    "coord_x": "coord $x$",
    "coord_y": "coord $y$",
    "coord_r": "coord $r$",
}
SIGNED = {"enemy_vx", "enemy_vy", "coord_x", "coord_y"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="15채널 관측 전개도(논문용)")
    p.add_argument("--site", default="통영_매물도북",
                   help="해역. 환형 안에 육지가 있어야 land 채널이 비지 않는다(매물도북 5.98%%)")
    p.add_argument("--enemy", default="wave")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--t", type=int, default=150, help="관측을 찍을 step")
    p.add_argument("--ship", type=int, default=0, help="배별 채널을 보일 방어정 번호")
    p.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    p.add_argument("--nets", type=int, default=3)
    p.add_argument("--out", default=os.path.join(REPO, "논문초안", "figs_snak", "fig_channels"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from boatattack_sim.env import cnn_map as CM
    from boatattack_sim.env.rasterizer import channel_names, obs_channels
    from commander.fallback import heuristic_plan
    from commander.rl_bridge import build_battlefield_defense
    from commander.sim_bridge import plan_to_assign
    from commander.unet_bridge import CommandedCnnEnv
    from tools.make_commander_gif import resolve_site

    lat, lon, site_label = resolve_site(args.site)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO, args.ckpt)
    env = CommandedCnnEnv(ckpt, enemy_mode=args.enemy, nets_per_ship=args.nets,
                          geo=None if lat is None else (lat, lon))
    env.reset(seed=args.seed)
    env.running = True

    # 배정을 주입해야 self_intercept / self_valid 가 채워진다(휴리스틱 계획으로 충분).
    last = -10 ** 9
    for _ in range(args.t):
        t = int(env.t.flat[0])
        if t - last >= 60:
            bf = build_battlefield_defense(env, command="Capture all enemies")
            plan = heuristic_plan(bf)
            plan_to_assign(plan, bf, mode="llm")
            env.set_plan(plan, "Capture all enemies")
            last = t
        if bool(env.done.flat[0]):
            break
        env.step()

    obs = env.build_cnn_obs()
    cfg = env.cfg
    names = channel_names(cfg)
    Cg, Cs, Cc = obs_channels(cfg)
    gmap = np.asarray(obs["gmap"][0])                 # [Cg,n,n]
    smap = np.asarray(obs["smap"][0, args.ship])      # [Cs,n,n]
    coord = np.asarray(CM.coord_channels(cfg))        # [Cc,n,n]
    chans = np.concatenate([gmap, smap, coord], axis=0)
    assert chans.shape[0] == len(names) == 15, (chans.shape, names)
    n = chans.shape[-1]
    print(f"[ch] site={site_label} enemy={args.enemy} seed={args.seed} t={int(env.t.flat[0])} "
          f"grid={n}x{n} ship={args.ship}")
    for nm, a in zip(names, chans):
        print(f"      {nm:<16s} min={a.min():+.2f} max={a.max():+.2f} nz={int((a != 0).sum())}")

    # ── 5x3 ──
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 8,
                         "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.55), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.06, hspace=0.22)
    for k, (ax, nm, a) in enumerate(zip(axes.ravel(), names, chans)):
        if nm in SIGNED:
            v = float(max(abs(a.min()), abs(a.max()), 1e-6))
            im = ax.imshow(a.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                           interpolation="nearest")
        else:
            im = ax.imshow(a.T, origin="lower", cmap="Greys", vmin=0.0,
                           vmax=float(max(a.max(), 1e-6)), interpolation="nearest")
        ax.set_title(f"{k + 1:02d}  {LABEL.get(nm, nm)}", fontsize=7.5, pad=2.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#888888")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[ch] 저장: {args.out}.png / .pdf  ({n}x{n}, 5x3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
