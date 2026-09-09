# -*- coding: utf-8 -*-
"""슬라이드 도판 — 「왜 좌표 회귀가 아니라 점수맵 위 픽셀 지목인가」

실제 체크포인트(`boatattack_sim/models/u-net_map.pt`)를 돌려서 나온 값만 쓴다.
가짜 숫자·가짜 회귀 모델 없음. 논지는 측정된 것 하나다:

    행동공간의 대부분은 **물리적으로 못 쓰는 곳**이다.
    좌표 회귀는 그 경계를 출력 형식 안에 표현할 자리가 없고,
    점수맵 지목은 후보를 마스킹해 **구조적으로** 그 안에서만 고른다.

렌더:
    python docs/diagrams/scoremap_vs_regression.py
출력:
    docs/diagrams/why_scoremap.png        (2패널 대조 — 슬라이드 본문)
    docs/diagrams/why_scoremap_stats.json (도판에 찍힌 수치의 원본)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib import font_manager

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_ROOT = os.path.dirname(_ROOT)                      # 저장소 루트
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boatattack_sim.env import cnn_map as CM                      # noqa: E402
from commander.fallback import heuristic_plan                     # noqa: E402
from commander.rl_bridge import build_battlefield_defense         # noqa: E402
from commander.unet_bridge import CommandedCnnEnv                 # noqa: E402

CKPT = os.path.join(_ROOT, "boatattack_sim", "models", "u-net_map.pt")
OUT = os.path.dirname(os.path.abspath(__file__))

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

C_BAD = "#C62828"      # 못 쓰는 곳
C_OK = "#2E7D32"       # 후보
C_PICK = "#00E5FF"     # 선택 픽셀
C_NET = "#00E5FF"


# ──────────────────────────────────────────────────────────────────────────
# 1. 마스크 축소 단계별 픽셀 수 (실측)
# ──────────────────────────────────────────────────────────────────────────
def mask_cascade(env, p: int):
    """배 p 에 대해 후보가 어떤 순서로 깎이는지 픽셀 수를 센다.
    defense_env._cnn_valid_mask 와 같은 순서: 환형 → ¬육지 → 반경게이트 → 방위게이트."""
    cfg = env.cfg
    n = CM.grid_n(cfg)
    c = env.center
    XX, YY = CM.pixel_centers(cfg)

    annulus = CM.annulus_mask(cfg)
    land = None
    if env.has_land and env.land_map.shape[1:] == (n, n):
        land = env.land_map[env.world_site][0]

    Gc = env._assignI[0, p] if env._assign[0, p] >= 0 else env.a_pos[0, p]
    d = np.hypot(XX - Gc[0], YY - Gc[1])

    step = {"전체 격자": np.ones((n, n), bool)}
    step["환형(요격 가능 반경)"] = annulus.copy()
    m = annulus.copy()
    if land is not None:
        m = m & ~land
    step["- 육지"] = m.copy()
    gr = float(getattr(cfg, "cnn_gate_r", 0.0))
    if gr > 0.0:
        m = m & (d <= gr)
    step[f"- 요격점 반경 {gr:.0f} m 밖"] = m.copy()
    amax = float(getattr(cfg, "cnn_gate_angle", 180.0))
    if amax < 180.0:
        _, PB = CM.pixel_polar(cfg)
        Gb = np.degrees(np.arctan2(Gc[0] - c[0], Gc[1] - c[1]))
        db = np.abs(((PB - Gb + 180.0) % 360.0) - 180.0)
        m = m & (db <= amax)
    step[f"- 배정 코리도 ±{amax:.0f}° 밖"] = m.copy()
    return step, float(amax), float(gr), Gc


# ──────────────────────────────────────────────────────────────────────────
# 2. 대표 프레임 찾기 + 집계
# ──────────────────────────────────────────────────────────────────────────
def collect(episodes=4, steps=700, replan=25, seed0=100, mode="diversionary"):
    env = CommandedCnnEnv(CKPT, enemy_mode=mode, device="cpu")
    frac = []          # 결정마다 유효픽셀 / 전체
    dmean = []         # |E[픽셀] - argmax| — MSE 회귀가 수렴할 조건부평균이 봉우리에서 얼마나 떨어지나
    best = None        # 도판에 쓸 프레임 (배 3척 모두 배정 + 육지가 보이는 상태)
    for ep in range(episodes):
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
            valid = np.asarray(v["valid"])
            assign = np.asarray(v["assign"])
            act = np.where(assign >= 0)[0]
            if len(act) == 0:
                continue
            n2 = valid.shape[-1] * valid.shape[-2]
            prob = np.asarray(v["prob"])
            pixa = np.asarray(v["pix"])
            H, W = valid.shape[-2:]
            gi, gj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            for p in act:
                frac.append(valid[p].sum() / n2)
                pr = prob[0, p]
                sp = pr.sum()
                if sp <= 0:
                    continue
                q = pr / sp
                mi, mj = float((q * gi).sum()), float((q * gj).sum())
                k0 = int(pixa[p, 0])
                dmean.append(float(np.hypot(mi - k0 // W, mj - k0 % W)))
            # 대표 프레임: 전 척 배정 + 적이 여럿 살아있음 + 그물 벽이 실제로 그려짐
            n_alive = int(env.e_alive[0].sum())
            if best is None and len(act) == valid.shape[0] and n_alive >= 6 and t > 60:
                best = dict(ep=ep, t=t, seed=seed0 + ep)
    return env, np.array(frac), np.array(dmean), best


def replay_to(seed, t_target, replan=25, mode="diversionary"):
    """같은 시드로 다시 돌려 대상 프레임의 env 를 그대로 재현한다."""
    env = CommandedCnnEnv(CKPT, enemy_mode=mode, device="cpu")
    env.reset(seed=seed)
    last = -10 ** 9
    for t in range(t_target + 1):
        if t - last >= replan:
            env.set_plan(heuristic_plan(build_battlefield_defense(env)))
            last = t
        env.step()
    return env


# ──────────────────────────────────────────────────────────────────────────
# 3. 그리기
# ──────────────────────────────────────────────────────────────────────────
def _scene(ax, env, title, zoom=None):
    cfg = env.cfg
    c = env.center
    ext = CM.extent(cfg)
    if zoom is None:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    else:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=15, pad=10, weight="bold")
    n = CM.grid_n(cfg)
    if env.has_land and env.land_map.shape[1:] == (n, n):
        land = env.land_map[env.world_site][0]
        ax.imshow(np.where(land, 1.0, np.nan).T, origin="lower", extent=ext,
                  cmap="copper", vmin=0, vmax=1, alpha=0.55, zorder=1.5,
                  interpolation="nearest")
    ax.add_patch(Circle(c, float(cfg.mothership_radius), fc="#FDD835",
                        ec="#F57F17", lw=1.5, zorder=6))
    e = env.e_pos[0][env.e_alive[0]]
    if len(e):
        ax.scatter(e[:, 0], e[:, 1], s=42, marker="^", c="#D32F2F",
                   ec="white", lw=0.6, zorder=6, label="적 USV")
    a = env.a_pos[0][env.a_alive[0]]
    if len(a):
        ax.scatter(a[:, 0], a[:, 1], s=80, marker="s", c="#1565C0",
                   ec="white", lw=1.0, zorder=6, label="방어정")
    return ext


def build_figure(env, p, frac, dmean, out_png):
    cfg = env.cfg
    v = env.cnn_viz()
    valid = np.asarray(v["valid"])[p]
    prob = np.asarray(v["prob"])[0, p]
    pix = np.asarray(v["pix"])[p]
    off = None if v["offset"] is None else np.asarray(v["offset"])[p]
    n = valid.shape[0]
    n2 = n * n

    step, amax, gr, Gc = mask_cascade(env, p)
    n_valid = int(valid.sum())
    pct = 100.0 * n_valid / n2

    fig = plt.figure(figsize=(15.5, 8.2), dpi=170)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.20], hspace=0.10, wspace=0.06,
                          left=0.02, right=0.98, top=0.88, bottom=0.03)

    fig.suptitle("왜 좌표 회귀가 아니라 「점수맵 위 픽셀 지목」인가",
                 fontsize=22, weight="bold", y=0.975)

    # ── 좌: 좌표 회귀 ───────────────────────────────────────────────────
    axL = fig.add_subplot(gs[0, 0])
    ext = _scene(axL, env, "좌표 회귀 —  출력 = 실수 2개  (x, y)")
    bad = ~valid
    axL.imshow(np.where(bad, 1.0, np.nan).T, origin="lower", extent=ext,
               cmap="Reds", vmin=0, vmax=1.6, alpha=0.42, zorder=2,
               interpolation="nearest")
    axL.text(0.5, 0.035,
             f"붉은 영역 = 물리적으로 못 놓는 곳  ·  격자의 {100 - pct:.1f} %\n"
             "회귀 출력에는 이 경계를 표현할 자리가 없다",
             transform=axL.transAxes, ha="center", fontsize=12.5, color=C_BAD,
             weight="bold", zorder=9,
             bbox=dict(fc="white", ec=C_BAD, lw=1.2, alpha=0.92, pad=6))
    axL.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # ── 우: 점수맵 지목 (유효영역으로 확대) ─────────────────────────────
    #   점수맵은 매우 뾰족하다(조건부평균 ↔ argmax 중앙 0.3 px = 81 m). 전체 맵 축척으로는
    #   봉우리 한 점만 보이므로 후보영역 bbox 로 확대하고 색조를 로그로 잡는다.
    ii, jj = np.where(valid)
    px = CM.px_size(cfg)
    x0w, y0w = ext[0], ext[2]
    zx = (x0w + (ii.min() - 2) * px, x0w + (ii.max() + 3) * px)
    zy = (y0w + (jj.min() - 2) * px, y0w + (jj.max() + 3) * px)
    pad = 0.5 * max(zx[1] - zx[0], zy[1] - zy[0])
    cxm, cym = 0.5 * (zx[0] + zx[1]), 0.5 * (zy[0] + zy[1])
    zoom = (cxm - pad, cxm + pad, cym - pad, cym + pad)
    axR = fig.add_subplot(gs[0, 1])
    _scene(axR, env, "점수맵 지목 —  출력 = 후보 픽셀 위의 확률분포  (후보영역 확대)",
           zoom=zoom)
    axR.imshow(np.where(valid, 1.0, np.nan).T, origin="lower", extent=ext,
               cmap="Greens", vmin=0, vmax=2.2, alpha=0.35, zorder=2,
               interpolation="nearest")
    pm = prob.max()
    if pm > 0:
        #   확률 동적범위가 10^-8 까지 간다. 로그로 펴되, 낮은 픽셀은 알파를 낮춰
        #   아래 깔린 후보영역(연녹)이 비치게 한다 — 안 그러면 검은 덩어리로 보인다.
        lg = np.log10(np.where(prob > 0, prob / pm, np.nan))
        sm = np.clip((lg + 8.0) / 8.0, 0.0, 1.0)              # 10^-8 ~ 1 → 0~1
        al = np.where(np.isnan(sm), 0.0, 0.10 + 0.85 * sm)
        axR.imshow(np.nan_to_num(sm).T, origin="lower", extent=ext, cmap="magma",
                   alpha=al.T, zorder=3, interpolation="nearest", vmin=0.0, vmax=1.0)
    # 왼쪽 패널에 확대 범위 표시
    axL.add_patch(plt.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                                fill=False, ec="#00838F", lw=2.0, ls="--", zorder=8))
    axL.annotate("확대 →", (zoom[1], zoom[3]), textcoords="offset points",
                 xytext=(4, 4), color="#00838F", fontsize=11, weight="bold", zorder=9)
    w = CM.flat_to_world(cfg, pix, off)                       # [K,2]
    axR.plot(w[:, 0], w[:, 1], "-", color=C_NET, lw=3.0, zorder=8)
    axR.scatter(w[:, 0], w[:, 1], s=170, marker="X", c=C_PICK,
                ec="white", lw=1.8, zorder=9)
    for k, (x, y) in enumerate(w):
        axR.annotate(f"{k}", (x, y), textcoords="offset points", xytext=(11, 9),
                     color=C_PICK, fontsize=14, weight="bold", zorder=10,
                     path_effects=None)
    mid = w.mean(0)
    axR.annotate("그물 벽", mid, textcoords="offset points", xytext=(52, -34),
                 color="#006064", fontsize=12.5, weight="bold", zorder=10,
                 arrowprops=dict(arrowstyle="-", color="#006064", lw=1.2),
                 bbox=dict(fc="white", ec="#006064", lw=1.0, alpha=0.9, pad=3))
    px_m = CM.px_size(cfg)
    axR.text(0.5, 0.135,
             f"점수맵은 뾰족하다 — 조건부평균 ↔ argmax 중앙 {np.median(dmean)*px_m:.0f} m "
             f"({len(dmean):,} 표본).\n확률로 뭉개는 게 아니라 후보 안에서 확정적으로 고른다.",
             transform=axR.transAxes, ha="center", fontsize=10.5, color="#37474F",
             zorder=9, bbox=dict(fc="#ECEFF1", ec="#B0BEC5", alpha=0.92, pad=4))
    axR.text(0.5, 0.035,
             f"선택 가능 {n_valid} / {n2} 픽셀 ({pct:.1f} %)\n"
             "마스킹된 softmax → 부적합한 점은 확률 0, 애초에 못 고른다",
             transform=axR.transAxes, ha="center", fontsize=12.5, color=C_OK,
             weight="bold", zorder=9,
             bbox=dict(fc="white", ec=C_OK, lw=1.2, alpha=0.92, pad=6))

    # ── 하단: 후보가 깎이는 순서 + 집계 ─────────────────────────────────
    axB = fig.add_subplot(gs[1, :]); axB.axis("off")
    names = list(step.keys())
    cnts = [int(step[k].sum()) for k in names]
    x0, w_bar = 0.020, 0.132
    for i, (nm, ct) in enumerate(zip(names, cnts)):
        x = x0 + i * (w_bar + 0.026)
        h = 0.30 * ct / n2
        axB.add_patch(plt.Rectangle((x, 0.42), w_bar, max(h, 0.012),
                                    fc=C_OK if i == len(names) - 1 else "#90A4AE",
                                    ec="none", transform=axB.transAxes))
        axB.text(x + w_bar / 2, 0.36, nm, transform=axB.transAxes, ha="center",
                 va="top", fontsize=10.5)
        axB.text(x + w_bar / 2, 0.44 + max(h, 0.012), f"{ct}",
                 transform=axB.transAxes, ha="center", va="bottom",
                 fontsize=11.5, weight="bold")
        if i < len(names) - 1:
            axB.annotate("", xy=(x + w_bar + 0.030, 0.50),
                         xytext=(x + w_bar + 0.004, 0.50),
                         xycoords=axB.transAxes, textcoords=axB.transAxes,
                         arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.4))
    axB.text(0.99, 0.52,
             f"전 결정 평균 후보 {100*frac.mean():.1f} %\n"
             f"(중앙 {100*np.median(frac):.1f} % · 최소 {100*frac.min():.1f} %)",
             transform=axB.transAxes, ha="right", va="center", fontsize=12,
             weight="bold", color="#37474F",
             bbox=dict(fc="#ECEFF1", ec="#B0BEC5", pad=7))
    axB.text(0.020, 0.06,
             "후보 축소 순서 (defense_env._cnn_valid_mask) — 마지막 칸이 정책이 실제로 고를 수 있는 집합. "
             "배정 코리도는 LLM 지휘관이 정한 요격점 방위로 열린다.",
             transform=axB.transAxes, fontsize=10.5, color="#546E7A")

    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return dict(n_valid=n_valid, n_total=n2, pct=pct,
                cascade={k: int(sv.sum()) for k, sv in step.items()},
                gate_r=gr, gate_angle=amax,
                frac_mean=float(frac.mean()), frac_median=float(np.median(frac)),
                frac_min=float(frac.min()), n_decisions=int(len(frac)),
                dmean_median_px=float(np.median(dmean)),
                dmean_median_m=float(np.median(dmean) * CM.px_size(cfg)),
                dmean_n=int(len(dmean)))


if __name__ == "__main__":
    _, frac, dmean, best = collect()
    if best is None:
        raise SystemExit("대표 프레임을 찾지 못했습니다.")
    print(f"대표 프레임: seed={best['seed']} t={best['t']}")
    env = replay_to(best["seed"], best["t"])
    v = env.cnn_viz()
    act = np.where(np.asarray(v["assign"]) >= 0)[0]
    p = int(act[0])
    png = os.path.join(OUT, "why_scoremap.png")
    stats = build_figure(env, p, frac, dmean, png)
    stats["frame"] = best
    stats["ship"] = p
    json.dump(stats, open(os.path.join(OUT, "why_scoremap_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("->", png)
