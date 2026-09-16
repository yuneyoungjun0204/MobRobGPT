# -*- coding: utf-8 -*-
"""수렴 진단 도판(KR_real Fig. 11)의 '참조 논문 스타일' 판.

참조: 먼지 침강 실험의 Depth/Intensity 패널 그림 --- 패널마다 중앙 제목, 모든 패널에 x·y 축 라벨,
연회색 격자, 4면 박스 프레임 + 바깥 눈금, 패널 안쪽에 테두리 있는 작은 범례, 파랑/빨강/회색 +
초록 파선 기준선, 넓고 낮은 패널을 세로로 쌓은 2열 구성.

옮기지 않은 것(자료 특성): 참조는 프레임별 산점이지만 우리 자료는 800 업데이트 이동평균 시계열이라
선으로 그린다. (c) 근사 KL 은 로그축을 유지한다. 범례 항목은 패널마다 실제로 그려진 계열만 싣는다.

데이터 적재·스무딩은 paper_figs.fig_training_diag 와 같다(같은 CSV, 같은 이동평균 창).
"""
from __future__ import annotations

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paper_figs import _log_decimal, _smooth, paper_figsize, save, use_paper_style

# 참조 그림의 색 --- 파랑(주 계열)·빨강·회색, 초록 파선 기준선
REF = {"blue": "#1F77B4", "red": "#C0392B", "grey": "#B0B0B0", "green": "#2E8B57", "ink": "#000000"}


def use_ref_style() -> None:
    """use_paper_style() 위에 참조 그림의 관례를 덮어쓴다."""
    use_paper_style()
    matplotlib.rcParams.update({
        "axes.grid": True, "axes.grid.axis": "both", "axes.axisbelow": True,
        "grid.color": "#D0D0D0", "grid.linewidth": 0.5, "grid.linestyle": "-",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.linewidth": 0.6, "axes.edgecolor": REF["ink"],
        "axes.titlesize": 6.5, "axes.titleweight": "normal", "axes.titlepad": 2.0,
        "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5,
        "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 1.0,
        "legend.edgecolor": REF["ink"], "legend.facecolor": "white",
        "legend.fontsize": 4.8, "legend.borderpad": 0.3, "legend.labelspacing": 0.2,
        "legend.handlelength": 1.4,
        "patch.linewidth": 0.4,   # 범례 테두리 굵기
    })


def _ref_axis(ax, *, xlim, title: str, ylabel: str, xlabel: str = "Updates", legend: dict | None = None):
    from matplotlib.ticker import MaxNLocator
    ax.set_title(title, loc="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.margins(y=0.08)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    if ax.get_yscale() != "log":
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(**(legend or {"loc": "best"}))


def fig_training_diag_ref(diag_root: str, *, adopted_run: str | None = None, smooth: int = 10,
                          figsize=None):
    use_ref_style()
    runs = sorted(d for d in glob.glob(os.path.join(diag_root, "seed*")) if os.path.isdir(d))
    ms = [pd.read_csv(os.path.join(d, "metrics.csv")) for d in runs]
    es = [pd.read_csv(os.path.join(d, "evals.csv")) for d in runs]
    n_upd = min(len(m) for m in ms)
    ms = [m.iloc[:n_upd] for m in ms]
    base = float(np.mean([e["baseline"].iloc[0] for e in es]))
    u = ms[0]["upd"].to_numpy()
    xlim = (0, int(np.ceil(u.max() / 100.0) * 100))
    n_seed = len(runs)
    mean_label = f"Seed mean ({n_seed} seeds)"

    def series(ax, col, *, log=False):
        # 시드별 원자료(짧은 이동평균)를 회색 가는 선으로 깔고, 그 위에 시드 평균을 얹는다.
        Y = np.stack([_smooth(m[col].to_numpy(), smooth) for m in ms])
        for i, y in enumerate(Y):
            ax.plot(u, y, color=REF["grey"], lw=0.35, label="Individual seeds" if i == 0 else None)
        ax.plot(u, Y.mean(0), color=REF["blue"], lw=0.7, label=mean_label)
        if log:
            _log_decimal(ax)
        return Y

    # 2행 x 2열, 넓고 낮은 패널(참조 그림의 비례). 그래디언트 노름·보상 이득은 본문 수치로만 남긴다
    # --- 전자는 본문이 수렴 지표로 쓰지 않는다고 밝힌 양이고, 후자는 부호 반전 한 문장뿐이다.
    figsize = figsize or paper_figsize(nrows=2, ncols=2, ratio=0.42)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (a1, a2), (a3, a4) = axes

    # (a) greedy 포획률 --- 채택 런이 시드 범위 안에 드는가
    n_ev = min(len(e) for e in es)
    ue = es[0]["upd"].to_numpy()[:n_ev]
    ce = np.stack([e["eval_cap"].to_numpy()[:n_ev] for e in es])
    for i, y in enumerate(ce):
        a1.plot(ue, y, color=REF["grey"], lw=0.5, label="Individual seeds" if i == 0 else None)
    a1.plot(ue, ce.mean(0), color=REF["blue"], lw=0.8, label=mean_label)
    if adopted_run and os.path.exists(os.path.join(adopted_run, "evals.csv")):
        ea = pd.read_csv(os.path.join(adopted_run, "evals.csv"))
        a1.plot(ea["upd"], ea["eval_cap"], color=REF["red"], lw=0.7, ls=(0, (4, 2)), zorder=5,
                label="Adopted run")
    a1.axhline(base, color=REF["green"], ls=(0, (4, 2)), lw=0.7, label=f"Heuristic baseline ({base:.3f})")
    _ref_axis(a1, xlim=xlim, title="(a) Greedy capture rate", ylabel="Capture rate",
              legend={"loc": "lower right", "ncol": 2})

    # (b) 엔트로피
    series(a2, "ent_pix1")
    _ref_axis(a2, xlim=xlim, title="(b) Policy entropy", ylabel="Entropy (nat)",
              legend={"loc": "upper right", "ncol": 2})

    # (c) 근사 KL (로그축)
    series(a3, "kl", log=True)
    _ref_axis(a3, xlim=xlim, title="(c) Approx. KL per update", ylabel="Approx. KL",
              legend={"loc": "lower left", "ncol": 2})

    # (d) 파라미터 이동 거리
    series(a4, "dtheta_rel")
    a4.set_ylim(bottom=0)
    _ref_axis(a4, xlim=xlim, title="(d) Relative parameter shift", ylabel="Relative shift",
              legend={"loc": "lower right", "ncol": 2})

    fig.tight_layout(h_pad=0.8, w_pad=1.4)
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="results/train_diag")
    ap.add_argument("--adopted", default="results/train_run_20260819-203248")
    ap.add_argument("--out", default="논문_그래프/8_논문판")
    ap.add_argument("--name", default="figP8b_convergence_ref")
    a = ap.parse_args()
    print("[ok]", save(fig_training_diag_ref(a.diag, adopted_run=a.adopted), a.out, a.name))
