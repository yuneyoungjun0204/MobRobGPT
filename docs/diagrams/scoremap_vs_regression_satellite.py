# -*- coding: utf-8 -*-
"""논문판 도판(위성 배경) — 유효 마스크와 픽셀 지목, 통영 매물도 북측.

`scoremap_vs_regression_paper.py`(흑백판)와 같은 구성 (a)(b)(c) 이되, (a)(b)의 배경에
Esri World Imagery 위성 타일을 깐다. 해역은 학습 6곳 중 **환형 잠식이 가장 큰**
통영 매물도 북측(34.6650, 128.5450; terrain.KOREA_SITES)이다 --- 지형이 마스크에 실제로
개입하는 모습이 보여야 위성 배경이 정보가 된다.

데이터는 흑백판과 같은 경로: 같은 체크포인트, `CommandedCnnEnv` 를 이 해역으로 고정해
휴리스틱 배정으로 굴리다가 **세 척 모두 배정 + 적 6척 이상 생존** 인 첫 결정 프레임을 쓴다.
캐스케이드 수치는 그 프레임의 실측이다.

위성 타일은 boatattack_sim/eval/_basemap_cache/ 에 npy 로 캐시한다(gitignore). 네트워크가
없으면 캐시가 있을 때만 그린다.

렌더:
    python docs/diagrams/scoremap_vs_regression_satellite.py
출력:
    논문_그래프/8_논문판/figP9b_valid_mask_sat.{pdf,png}
    논문초안/figs_snak/fig_valid_mask_sat.{pdf,png}
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge
import matplotlib.patheffects as pe

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boatattack_sim.env import cnn_map as CM                                   # noqa: E402
from boatattack_sim.env import clustering                                      # noqa: E402
from boatattack_sim.env import terrain as T                                    # noqa: E402
from boatattack_sim.eval.paper_figs import (MONO, _panel_label, paper_figsize,  # noqa: E402
                                            save, use_paper_style)
from boatattack_sim.eval.renderer import (_CV, _local_rect, _rot, _ship_poly,  # noqa: E402
                                          carrier_fit_size, draw_carrier)
from commander.fallback import heuristic_plan                                  # noqa: E402
from commander.rl_bridge import build_battlefield_defense                      # noqa: E402
from commander.unet_bridge import CommandedCnnEnv                              # noqa: E402
from docs.diagrams.scoremap_vs_regression import CKPT, mask_cascade           # noqa: E402

SITE_NAME = "통영_매물도북"
GEO = next(g for n, *g in T.KOREA_SITES if n == SITE_NAME)        # (lat, lon)
OUT_DIR = os.path.join(_ROOT, "논문_그래프", "8_논문판")
PAPER_DIR = os.path.join(_ROOT, "논문초안", "figs_snak")
CACHE = os.path.join(_ROOT, "boatattack_sim", "eval", "_basemap_cache")
STEP_EN = ["All cells", "Intercept annulus", "Land removed", "Radius gate 3 km",
           "Bearing gate ±60°"]


def satellite(cfg, zoom: int = 14):
    """월드 박스를 덮는 위성 RGB [H,W,3]. 캐시 우선, 없으면 내려받아 저장."""
    os.makedirs(CACHE, exist_ok=True)
    key = f"sat_{GEO[0]:.5f}_{GEO[1]:.5f}_{int(cfg.world_size)}_z{zoom}.npy"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return np.load(path)
    img = T._fetch_tiles(GEO[0], GEO[1], float(cfg.world_size), zoom, T.SAT_URL)
    if img is None:
        raise SystemExit("[sat] 위성 타일을 받지 못했고 캐시도 없다.")
    np.save(path, img)
    return img


def find_frame(seed0: int = 100, episodes: int = 3, steps: int = 400, replan: int = 25):
    """세 척 모두 배정 + 적 6척 이상 생존 + t>60 인 첫 결정 프레임의 env 를 돌려준다."""
    for ep in range(episodes):
        env = CommandedCnnEnv(CKPT, enemy_mode="diversionary", device="cpu", geo=GEO)
        env.reset(seed=seed0 + ep)
        last = -10 ** 9
        for t in range(steps):
            if t - last >= replan:
                env.set_plan(heuristic_plan(build_battlefield_defense(env)))
                last = t
            env.step()
            v = env.cnn_viz()
            if v["prob"] is None or v["pix"] is None:
                continue
            assign = np.asarray(v["assign"])
            if t > 60 and (assign >= 0).all() and int(env.e_alive[0].sum()) >= 6:
                print(f"[sat] frame seed={seed0 + ep} t={t}")
                return env, seed0 + ep, t
    raise SystemExit("[sat] 대표 프레임을 찾지 못했다.")


#: 운용 화면(run_commander_ui / make_commander_gif)과 같은 선체 색.
C_ENEMY, C_ALLY = "#EC407A", "#26C6DA"


class _HullHandler(HandlerPatch):
    """범례 핸들을 지도와 같은 7점 선체 다각형으로 그린다(뱃머리 → 오른쪽)."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        verts = _ship_poly(xdescent + width / 2.0, ydescent + height / 2.0, 90.0,
                           width * 0.95, height * 0.95)
        poly = Polygon(verts, closed=True, facecolor=orig_handle.get_facecolor(),
                       edgecolor="white", lw=0.5)
        poly.set_transform(trans)
        return [poly]


class _CarrierHandler(HandlerPatch):
    """범례용 미니 항공모함 --- draw_carrier 와 같은 갑판·각진갑판·아일랜드 도형(뱃머리 → 오른쪽)."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        cx, cy = xdescent + width / 2.0, ydescent + height / 2.0
        hl, hw = width * 0.48, height * 0.42
        deck = [(0, hl), (0.60 * hw, 0.84 * hl), (hw, 0.58 * hl), (hw, -hl), (-hw, -hl),
                (-hw, 0.58 * hl), (-0.60 * hw, 0.84 * hl)]
        arts = [Polygon(_rot(deck, cx, cy, 90.0), closed=True, facecolor=_CV["deck"],
                        edgecolor=_CV["edge"], lw=0.4),
                Polygon(_rot(_local_rect(-0.34 * hw, 0.04 * hl, 0.62 * hl, 0.40 * hw, 9.0),
                             cx, cy, 90.0), closed=True, facecolor=_CV["deck2"],
                        edgecolor=_CV["edge"], lw=0.3),
                Polygon(_rot(_local_rect(0.74 * hw, -0.02 * hl, 0.20 * hl, 0.18 * hw, 0.0),
                             cx, cy, 90.0), closed=True, facecolor=_CV["island"],
                        edgecolor="#11151A", lw=0.3)]
        for a in arts:
            a.set_transform(trans)
        return arts


def _ships(ax, env, *, scale=1.0):
    """UI 렌더러와 같은 글리프 --- 적·아군은 7점 선체 다각형(heading 방향), 모선은 항공모함.

    선체 실제 길이(적 160 m, 아군 ship_len)는 12.6 km 판에서 1 % 남짓이라 보이지 않으므로
    `scale` 배로 키워 그린다. 위치·방향은 실제 값이고 크기만 기호적이다.
    모선은 항공모함 그림 자체가 기호이므로 범례에 넣지 않는다(캡션이 말한다).
    """
    cfg = env.cfg
    c = env.center
    mr = float(cfg.mothership_radius)
    ax.add_patch(Circle(c, mr, facecolor="none", edgecolor="white", lw=0.6,
                        ls=(0, (2, 1.5)), alpha=0.9, zorder=5))
    n_before = len(ax.collections)
    draw_carrier(ax, c[0], c[1], float(getattr(cfg, "moback_heading", 0.0)),
                 carrier_fit_size(mr) * max(1.0, scale * 0.8), z=6)
    # draw_carrier 의 마지막 scatter 는 마스트 점(s=10 고정)이라 이 축척에선 주황 점으로만
    # 보인다 --- 항공모함이 곧 모선 기호이므로 점은 지운다.
    for coll in ax.collections[n_before:]:
        coll.remove()
    e_len, e_wid = float(cfg.enemy_size) * 1.6 * scale, float(cfg.enemy_size) * 0.55 * scale
    for k in np.where(env.e_alive[0])[0]:
        x, y = env.e_pos[0, k]
        ax.add_patch(Polygon(_ship_poly(x, y, float(env.e_hdg[0, k]), e_len, e_wid), closed=True,
                             facecolor=C_ENEMY, edgecolor="white", lw=0.5, zorder=6))
    a_len, a_wid = float(cfg.ship_len) * scale, float(cfg.ship_wid) * scale
    for k in np.where(env.a_alive[0])[0]:
        x, y = env.a_pos[0, k]
        ax.add_patch(Polygon(_ship_poly(x, y, float(env.a_hdg[0, k]), a_len, a_wid), closed=True,
                             facecolor=C_ALLY, edgecolor="white", lw=0.5, zorder=7))
    return [Patch(facecolor=C_ENEMY, label="Attacker"),
            Patch(facecolor=C_ALLY, label="Defender"),
            Patch(facecolor=_CV["deck"], label="Mothership")]


C_FOCUS = "#8E0000"     # 초점 방어정·배정 클러스터·요격점 --- 검붉은색(위성의 청록과 보색, 인쇄 안전)


def _assignment_overlay(ax, env, p: int, *, scale: float = 1.0):
    """초점 방어정 p 가 맡은 클러스터(환형 부채꼴) + 요격점(◇) + 배정 화살표 + 방어정 강조."""
    cfg = env.cfg
    c = env.center
    k = int(env._assign[0, p])
    if k < 0:
        return []
    cl = clustering.cluster_by_gaps_vec(env.e_pos, env.e_alive, env.e_hdg, c,
                                        cfg.enemy_speed, int(cfg.n_clusters),
                                        float(cfg.cluster_gap_deg))
    labels = np.asarray(cl["labels"][0])
    mem = env.e_pos[0][(labels == k) & env.e_alive[0]]
    if len(mem):
        d = mem - np.asarray(c)
        rad = np.hypot(d[:, 0], d[:, 1])
        brg = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
        cb = np.degrees(np.arctan2(np.sin(np.deg2rad(brg)).mean(),
                                   np.cos(np.deg2rad(brg)).mean()))
        dev = (brg - cb + 180.0) % 360.0 - 180.0
        apad = max(2.0, min(4.0, float(cfg.cluster_gap_deg) * 0.25))
        amin, amax = cb + dev.min() - apad, cb + dev.max() + apad
        rin, rout = max(0.0, rad.min() - 260.0), rad.max() + 260.0
        th1, th2 = 90.0 - amax, 90.0 - amin
        ax.add_patch(Wedge(c, rout, th1, th2, width=rout - rin, facecolor="none",
                           edgecolor=C_FOCUS, lw=0.9, zorder=4.6))
    I = env._assignI[0, p]
    sx, sy = env.a_pos[0, p]
    ax.annotate("", xy=(I[0], I[1]), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=C_FOCUS, lw=0.9, shrinkA=0, shrinkB=2),
                zorder=7.5)
    # 초점 방어정: 선체를 넉넉히 감싸는 검붉은 원
    a_len = float(cfg.ship_len) * scale
    ax.add_patch(Circle((sx, sy), a_len * 1.1, facecolor="none", edgecolor=C_FOCUS, lw=1.0,
                        zorder=7.2))
    return [Patch(facecolor="none", edgecolor=C_FOCUS, lw=0.6, label="Assigned cluster"),
            Line2D([], [], marker="o", color=C_FOCUS, markerfacecolor="none", lw=0, ms=2.8,
                   markeredgewidth=0.7, label="Focal defender")]


def build(env, p: int, sat: np.ndarray):
    use_paper_style()
    ink, mid, light = MONO["ink"], MONO["mid"], MONO["light"]
    cfg = env.cfg
    v = env.cnn_viz()
    valid = np.asarray(v["valid"])[p]
    prob = np.asarray(v["prob"])[0, p]
    pix = np.asarray(v["pix"])[p]
    off = None if v["offset"] is None else np.asarray(v["offset"])[p]
    n = valid.shape[0]
    n2 = n * n
    ext = CM.extent(cfg)
    px = CM.px_size(cfg)
    W = float(cfg.world_size)
    step, _, _, _ = mask_cascade(env, p)
    cnts = [int(m.sum()) for m in step.values()]
    bad = ~valid

    fig = plt.figure(figsize=paper_figsize(nrows=1, ncols=3, ratio=1.0))
    # 3열 + (c) 앞의 빈 열: y 눈금 라벨("Bearing gate ±60°")이 (b)와 겹치지 않을 자리.
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.36, 0.72], wspace=0.10,
                          left=0.01, right=0.99, top=0.86, bottom=0.10)
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in (0, 1, 3))

    def basemap(ax):
        # 위성 타일: 행 0 이 북쪽 → origin="upper", 박스 [0, W]² 에 맞춘다.
        ax.imshow(sat, extent=(0.0, W, 0.0, W), origin="upper", zorder=0,
                  interpolation="bilinear")
        # 놓을 수 없는 칸은 흰색 반투명으로 눌러 위성은 비치되 '못 쓰는 곳'임이 읽히게.
        ax.imshow(np.where(bad, 0.0, np.nan).T, origin="lower", extent=ext, cmap="Greys",
                  vmin=0, vmax=1, alpha=0.60, zorder=1, interpolation="nearest")   # 흰 베일

    # ── (a) 전체 격자 ───────────────────────────────────────────────────────
    axA.set_xlim(ext[0], ext[1]); axA.set_ylim(ext[2], ext[3])
    axA.set_aspect("equal"); axA.set_xticks([]); axA.set_yticks([])
    basemap(axA)
    handles = _ships(axA, env, scale=3.0)
    handles += _assignment_overlay(axA, env, p, scale=3.0)
    ii, jj = np.where(valid)
    zx = (ext[0] + (ii.min() - 2) * px, ext[0] + (ii.max() + 3) * px)
    zy = (ext[2] + (jj.min() - 2) * px, ext[2] + (jj.max() + 3) * px)
    pad = 0.5 * max(zx[1] - zx[0], zy[1] - zy[0])
    cx, cy = 0.5 * sum(zx), 0.5 * sum(zy)
    zoom = (cx - pad, cx + pad, cy - pad, cy + pad)
    axA.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                            fill=False, ec="white", lw=1.6, zorder=7))
    axA.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                            fill=False, ec=ink, lw=0.8, ls=(0, (3, 2)), zorder=8))
    pct_bad = 100.0 * bad.sum() / n2
    axA.set_title("Feasible region (focal defender)", fontsize=7, pad=4)
    hmap = {handles[0]: _HullHandler(), handles[1]: _HullHandler(),
            handles[2]: _CarrierHandler()}          # 나머지(부채꼴·◇)는 기본 핸들러
    axA.legend(handles=handles, loc="lower left", fontsize=3.0, handletextpad=0.3,
               borderpad=0.25, labelspacing=0.35, frameon=True, framealpha=0.92,
               edgecolor="none", handlelength=1.2, handleheight=0.55, handler_map=hmap)

    # ── (b) 후보 영역 확대 + 점수맵 ─────────────────────────────────────────
    axB.set_xlim(zoom[0], zoom[1]); axB.set_ylim(zoom[2], zoom[3])
    axB.set_aspect("equal"); axB.set_xticks([]); axB.set_yticks([])
    basemap(axB)
    pm = prob.max()
    if pm > 0:
        lg = np.log10(np.where(prob > 0, prob / pm, np.nan))
        sm = np.clip((lg + 4.0) / 4.0, 0.0, 1.0)              # 10^-4 ~ 1 → 0 ~ 1
        sm = np.where(valid, sm, np.nan)
        # 위성(청록·녹색) 위에서 구분되는 따뜻한 단색 계열. 낮은 확률은 투명하게.
        al = np.where(np.isnan(sm), 0.0, 0.15 + 0.85 * np.nan_to_num(sm))
        axB.imshow(np.nan_to_num(sm).T, origin="lower", extent=ext, cmap="YlOrRd",
                   vmin=0.0, vmax=1.0, alpha=al.T, zorder=3, interpolation="nearest")
    axB.contour(*CM.pixel_centers(cfg), valid.astype(float), levels=[0.5], colors=["white"],
                linewidths=0.9, zorder=4)
    _ships(axB, env, scale=1.6)
    _assignment_overlay(axB, env, p, scale=1.6)
    w = CM.flat_to_world(cfg, pix, off)
    axB.plot(w[:, 0], w[:, 1], "-", color=ink, lw=2.6, zorder=8,
             path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
    axB.scatter(w[:, 0], w[:, 1], s=20, marker="o", c="white", edgecolors=ink, lw=0.9,
                zorder=9)
    # 'Net wall' 글자는 뺀다 --- 두 흰 원을 잇는 선이 그물벽임은 캡션이 말한다.
    axB.set_title("Score map and selected pair", fontsize=7, pad=4)
    for ax in (axA, axB):
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    # ── (c) 캐스케이드 --- 정식 가로 막대 (y 축 = 단계, x 축 = 칸 수) ─────────
    y = np.arange(len(cnts))[::-1] * 1.0
    # 막대 다섯 개 전부 같은 사선 빗금 --- 어느 단계도 특별 취급하지 않는다.
    plt.rcParams["hatch.linewidth"] = 0.4
    bars = axC.barh(y, cnts, color="white", edgecolor=ink, linewidth=0.5, height=0.55)
    for b in bars:
        b.set_hatch("/////")
    for yi, ct in zip(y, cnts):
        axC.text(ct + n2 * 0.025, yi, f"{ct:,}", va="center", ha="left", fontsize=6.5, color=ink)
    axC.set_yticks(y)
    axC.set_yticklabels(STEP_EN, fontsize=6.5)
    axC.tick_params(axis="y", length=0, pad=2)
    axC.set_xlim(0, n2 * 1.22)
    axC.set_xticks([0, 1000, 2000])
    axC.set_xticklabels(["0", "1,000", "2,000"], fontsize=6)
    axC.set_xlabel("Number of cells", fontsize=7)
    axC.set_ylim(-0.6, len(cnts) - 0.4)
    for s_ in ("top", "right"):
        axC.spines[s_].set_visible(False)
    axC.grid(False)
    axC.set_title("Mask cascade", fontsize=7, pad=4)

    _panel_label(axA, "a", dx=-0.02, dy=1.09)
    _panel_label(axB, "b", dx=-0.02, dy=1.09)
    _panel_label(axC, "c", dx=-0.60, dy=1.09)
    return fig, dict(cascade=dict(zip(STEP_EN, cnts)), pct_bad=pct_bad, n_valid=int(valid.sum()))


if __name__ == "__main__":
    env, seed, t = find_frame()
    sat = satellite(env.cfg)
    print(f"[sat] tiles {sat.shape}")
    v = env.cnn_viz()
    p = int(np.where(np.asarray(v["assign"]) >= 0)[0][0])
    fig, info = build(env, p, sat)
    print(info)
    pdf = save(fig, OUT_DIR, "figP9b_valid_mask_sat")
    print("[ok]", pdf)
    for ext_ in (".pdf", ".png"):
        shutil.copyfile(os.path.join(OUT_DIR, "figP9b_valid_mask_sat" + ext_),
                        os.path.join(PAPER_DIR, "fig_valid_mask_sat" + ext_))
    print("[ok] ->", os.path.join(PAPER_DIR, "fig_valid_mask_sat.pdf"))
