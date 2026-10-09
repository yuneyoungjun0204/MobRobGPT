# -*- coding: utf-8 -*-
"""N9 --- 점수맵 첨예도 통계 (논문_그래프/9_신규제안/N9_scoremap_sharpness).

`scoremap_vs_regression.collect()` 를 그대로 돌려(체크포인트 u-net_map.pt, 양동 포메이션 4 에피소드)
결정마다 두 값을 모은다:
  (a) 유효 픽셀 비율 frac = |valid| / 2500 (x축은 격자 전체 0-100 %)
  (b) 조건부평균 ↔ argmax 거리 dmean (셀 단위, 위쪽 보조축 m) --- MSE 좌표 회귀가 수렴할 점(확률의 평균)이
      정책이 실제로 고르는 점(최빈)에서 얼마나 떨어져 있는가. 분포가 다봉이면 이 값이 커진다.
원자료는 npz 로 남겨 재렌더 때 시뮬을 다시 돌리지 않는다.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boatattack_sim.eval.paper_figs import MONO, _clean_axis, _panel_label, paper_figsize, save, use_paper_style  # noqa: E402

OUT = os.path.join(_ROOT, "논문_그래프", "9_신규제안")
RAW = os.path.join(_ROOT, "results", "scoremap_sharpness.npz")
PX_M = 252.0


def collect_raw():
    from docs.diagrams.scoremap_vs_regression import collect
    _, frac, dmean, _ = collect()
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    np.savez(RAW, frac=frac, dmean=dmean)
    return frac, dmean


def build(frac, dmean):
    use_paper_style()
    ink, light = MONO["ink"], MONO["light"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=paper_figsize(nrows=1, ncols=2, ratio=0.68))
    # (a) 격자 전체(0-100 %)를 x축으로 두어 유효 영역이 얼마나 작은지 한눈에 보이게 한다.
    pct = 100 * frac
    a1.hist(pct, bins=np.arange(0, 101, 1), color=light, edgecolor=ink, lw=0.4)
    a1.axvline(pct.mean(), color=ink, ls=(0, (4, 2)), lw=0.9)
    a1.text(pct.mean() + 2, 0.95, f"mean {pct.mean():.1f} %\n({pct.mean() / 100 * 2500:.0f} of 2,500 cells)",
            transform=a1.get_xaxis_transform(), fontsize=6, va="top", ha="left")
    a1.set_xlabel("Feasible cells per decision (% of grid)")
    a1.set_ylabel(f"Decisions (n = {len(frac):,})")
    _clean_axis(a1, xlim=(0, 100))
    a1.set_xticks([0, 25, 50, 75, 100])
    # (b) 결정이 거의 같은 셀 안에서 정해짐을 보이기 위해 셀 단위로 그리고, 위쪽에 m 보조축을 단다.
    d = dmean
    a2.hist(np.clip(d, 0, 2), bins=np.arange(0, 2.05, 0.05), color=light, edgecolor=ink, lw=0.4)
    a2.axvline(np.median(d), color=ink, ls=(0, (4, 2)), lw=0.9)
    a2.text(np.median(d) + 0.05, 0.95, f"median {np.median(d):.2f} cell", transform=a2.get_xaxis_transform(),
            fontsize=6, va="top")
    a2.axvline(1.0, color=ink, ls=":", lw=0.7)
    a2.text(1.05, 0.70, f"{100 * (d <= 1).mean():.1f} %\nwithin one cell", transform=a2.get_xaxis_transform(),
            fontsize=6, va="top")
    a2.set_xlabel("Mean-to-mode distance of score map (grid cells)")
    a2.set_ylabel("Decisions")
    _clean_axis(a2, xlim=(0, 2))
    a2.set_xticks([0, 0.5, 1, 1.5, 2])
    sec = a2.secondary_xaxis("top", functions=(lambda x: x * PX_M, lambda x: x / PX_M))
    sec.set_xticks([0, 126, 252, 378, 504])
    sec.set_xlabel("(m)", fontsize=6, labelpad=2)
    sec.tick_params(labelsize=6)
    _panel_label(a1, "a"); _panel_label(a2, "b")
    fig.tight_layout(w_pad=1.6)
    return fig


if __name__ == "__main__":
    if os.path.exists(RAW):
        z = np.load(RAW); frac, dmean = z["frac"], z["dmean"]
    else:
        frac, dmean = collect_raw()
    print("[ok]", save(build(frac, dmean), OUT, "N9_scoremap_sharpness"))
