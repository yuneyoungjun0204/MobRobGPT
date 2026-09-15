# -*- coding: utf-8 -*-
"""논문판 도판 — 유효 마스크와 픽셀 지목 (Fig. validmask). 흑백·영문·최소 텍스트.

`scoremap_vs_regression.py`(슬라이드판)의 **데이터 경로를 그대로 재사용**한다:
같은 체크포인트, 같은 대표 프레임(seed 100, t 61, ship 0), 같은 마스크 캐스케이드.
집계 수치(전 결정 평균 후보 비율·표본 수)는 슬라이드판이 남긴 why_scoremap_stats.json 에서
읽는다 --- 4 에피소드 재수집 없이 그림만 다시 그린다.

바뀐 것은 표현뿐이다.
  (a) 전체 격자: 놓을 수 없는 칸은 회색 빗금, 놓을 수 있는 칸은 흰색. 육지는 진회색.
      적·방어정·모선은 검정 마커. 확대 범위는 검정 파선 사각형.
  (b) 후보 영역 확대: 점수맵을 회색조(로그)로, 선택된 두 픽셀과 그물벽은 검정.
  (c) 후보 축소 캐스케이드: 가로 막대 5개, 마지막만 검정.
  텍스트는 패널 라벨·짧은 영문 소제목·막대 숫자·단계 이름뿐. 설명 문장은 캡션의 몫이다.

렌더:
    python docs/diagrams/scoremap_vs_regression_paper.py
출력:
    논문_그래프/8_논문판/figP9_valid_mask.{pdf,png}
    논문초안/figs_snak/fig_valid_mask.{pdf,png}
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boatattack_sim.env import cnn_map as CM                                   # noqa: E402
from boatattack_sim.eval.paper_figs import (MONO, _panel_label, paper_figsize,  # noqa: E402
                                            save, use_paper_style)
from docs.diagrams.scoremap_vs_regression import mask_cascade, replay_to       # noqa: E402

STATS = os.path.join(_HERE, "why_scoremap_stats.json")
OUT_DIR = os.path.join(_ROOT, "논문_그래프", "8_논문판")
PAPER_DIR = os.path.join(_ROOT, "논문초안", "figs_snak")

#: 캐스케이드 단계의 영문 이름 (slide 판의 한글 키와 같은 순서)
STEP_EN = ["All cells", "Intercept annulus", "Land removed", "Radius gate 3 km",
           "Bearing gate ±60°"]


def _ships(ax, env, *, ms_scale=1.0):
    """적·방어정·모선을 흑백 마커로. 범례 핸들을 돌려준다."""
    ink = MONO["ink"]
    c = env.center
    ax.add_patch(Circle(c, float(env.cfg.mothership_radius) * 1.6, fc="white", ec=ink,
                        lw=0.8, zorder=6))
    e = env.e_pos[0][env.e_alive[0]]
    if len(e):
        ax.scatter(e[:, 0], e[:, 1], s=14 * ms_scale, marker="^", c=ink, lw=0, zorder=6)
    a = env.a_pos[0][env.a_alive[0]]
    if len(a):
        ax.scatter(a[:, 0], a[:, 1], s=16 * ms_scale, marker="s", c=ink, lw=0, zorder=6)
    return [Line2D([], [], marker="^", color=ink, lw=0, ms=4, label="Attacker"),
            Line2D([], [], marker="s", color=ink, lw=0, ms=4, label="Defender"),
            Line2D([], [], marker="o", color=ink, markerfacecolor="white", lw=0, ms=5,
                   label="Mothership")]


def build(env, p: int, stats: dict):
    use_paper_style()
    ink, mid, light = MONO["ink"], MONO["mid"], MONO["light"]
    cfg = env.cfg
    v = env.cnn_viz()
    valid = np.asarray(v["valid"])[p]
    prob = np.asarray(v["prob"])[0, p]
    pix = np.asarray(v["pix"])[p]
    off = None if v["offset"] is None else np.asarray(v["offset"])[p]
    n = valid.shape[0]
    ext = CM.extent(cfg)
    px = CM.px_size(cfg)
    step, _, _, _ = mask_cascade(env, p)
    cnts = [int(m.sum()) for m in step.values()]
    n2 = n * n

    fig = plt.figure(figsize=paper_figsize(nrows=1, ncols=3, ratio=1.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.10)
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))

    # ── (a) 전체 격자 ───────────────────────────────────────────────────────
    axA.set_xlim(ext[0], ext[1]); axA.set_ylim(ext[2], ext[3])
    axA.set_aspect("equal"); axA.set_xticks([]); axA.set_yticks([])
    # 놓을 수 없는 칸: 연회색 + 빗금
    bad = ~valid
    # 놓을 수 없는 칸 = 연회색 단색, 놓을 수 있는 칸 = 흰색. 빗금은 137 mm 판형에서 너무 굵다.
    axA.imshow(np.where(bad, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greys",
               vmin=0, vmax=7, zorder=1, interpolation="nearest")
    if env.has_land and env.land_map.shape[1:] == (n, n):
        land = env.land_map[env.world_site][0]
        axA.imshow(np.where(land, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greys",
                   vmin=0, vmax=1.6, zorder=3, interpolation="nearest")
    handles = _ships(axA, env)
    # 확대 범위
    ii, jj = np.where(valid)
    zx = (ext[0] + (ii.min() - 2) * px, ext[0] + (ii.max() + 3) * px)
    zy = (ext[2] + (jj.min() - 2) * px, ext[2] + (jj.max() + 3) * px)
    pad = 0.5 * max(zx[1] - zx[0], zy[1] - zy[0])
    cx, cy = 0.5 * sum(zx), 0.5 * sum(zy)
    zoom = (cx - pad, cx + pad, cy - pad, cy + pad)
    axA.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                            fill=False, ec=ink, lw=0.8, ls=(0, (3, 2)), zorder=8))
    pct_bad = 100.0 * bad.sum() / n2
    axA.set_title(f"Infeasible (grey): {pct_bad:.0f} % of grid", fontsize=7, pad=4)
    axA.legend(handles=handles, loc="lower left", fontsize=5.5, handletextpad=0.3,
               borderpad=0.4, labelspacing=0.3, frameon=True, framealpha=0.95,
               edgecolor="none")

    # ── (b) 후보 영역 확대 + 점수맵 ─────────────────────────────────────────
    axB.set_xlim(zoom[0], zoom[1]); axB.set_ylim(zoom[2], zoom[3])
    axB.set_aspect("equal"); axB.set_xticks([]); axB.set_yticks([])
    axB.imshow(np.where(bad, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greys",
               vmin=0, vmax=7, zorder=1, interpolation="nearest")
    pm = prob.max()
    if pm > 0:
        lg = np.log10(np.where(prob > 0, prob / pm, np.nan))
        sm = np.clip((lg + 4.0) / 4.0, 0.0, 1.0)              # 10^-4 ~ 1 → 0 ~ 1
        sm = np.where(valid, sm, np.nan)
        axB.imshow(sm.T, origin="lower", extent=ext, cmap="Greys", vmin=0.0, vmax=1.15,
                   zorder=3, interpolation="nearest")
    # 유효 영역 윤곽
    axB.contour(*CM.pixel_centers(cfg), valid.astype(float), levels=[0.5], colors=[mid],
                linewidths=0.6, zorder=4)
    _ships(axB, env, ms_scale=1.4)
    w = CM.flat_to_world(cfg, pix, off)
    axB.plot(w[:, 0], w[:, 1], "-", color=ink, lw=1.8, zorder=8)
    axB.scatter(w[:, 0], w[:, 1], s=18, marker="o", c="white", edgecolors=ink, lw=0.9, zorder=9)
    axB.annotate("Net wall", xy=w.mean(0), xytext=(-52, -24), textcoords="offset points",
                 fontsize=6.5, color=ink, zorder=10,
                 arrowprops=dict(arrowstyle="-", color=ink, lw=0.6, shrinkB=3))
    axB.set_title(f"Score map on {int(valid.sum())} feasible cells", fontsize=7, pad=4)
    for ax in (axA, axB):
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    # ── (c) 캐스케이드 막대 ────────────────────────────────────────────────
    y = np.arange(len(cnts))[::-1] * 1.0
    colors = [light] * (len(cnts) - 1) + [ink]
    axC.barh(y, cnts, color=colors, edgecolor=ink, linewidth=0.5, height=0.42, align="center")
    for yi, ct, nm in zip(y, cnts, STEP_EN):
        axC.text(0, yi + 0.27, nm, va="bottom", ha="left", fontsize=6.5, color=ink)
        axC.text(ct + n2 * 0.02, yi, f"{ct:,}", va="center", ha="left", fontsize=6.5, color=ink)
    axC.set_yticks([]); axC.set_xlim(0, n2 * 1.3); axC.set_xticks([])
    axC.set_ylim(-0.4, len(cnts) - 0.2)
    for s_ in ("top", "right", "bottom"):
        axC.spines[s_].set_visible(False)
    axC.grid(False)
    axC.set_title(f"Mask cascade (mean {100 * stats['frac_mean']:.1f} %, n = "
                  f"{stats['n_decisions']:,})", fontsize=7, pad=4)

    _panel_label(axA, "a", dx=-0.02, dy=1.09)
    _panel_label(axB, "b", dx=-0.02, dy=1.09)
    _panel_label(axC, "c", dx=-0.04, dy=1.09)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03)
    return fig


if __name__ == "__main__":
    stats = json.load(open(STATS, encoding="utf-8"))
    fr = stats["frame"]
    env = replay_to(fr["seed"], fr["t"])
    fig = build(env, int(stats["ship"]), stats)
    pdf = save(fig, OUT_DIR, "figP9_valid_mask")
    print("[ok]", pdf)
    for ext_ in (".pdf", ".png"):
        shutil.copyfile(os.path.join(OUT_DIR, "figP9_valid_mask" + ext_),
                        os.path.join(PAPER_DIR, "fig_valid_mask" + ext_))
    print("[ok] ->", os.path.join(PAPER_DIR, "fig_valid_mask.pdf"))
