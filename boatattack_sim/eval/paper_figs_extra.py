# -*- coding: utf-8 -*-
"""논문 완성도 보강용 신규 도판 후보 (N1~N8) --- 논문_그래프/9_신규제안/ 에 굽는다.

전부 **기존 원자료**에서만 만든다(새 실험 없음):
    results/eval_merged/episodes.csv     720 에피소드 (8조건 x 3포메이션 x 30시드)
    results/eval_merged/llm_metrics.csv  13,547 호출 (지연·폴백·커버리지·churn·교차·휴리스틱 일치·근거 길이)
    results/train_diag/seed*/{bc,metrics}.csv  수렴 진단 3시드

스타일은 paper_figs 의 저널판(Times/HCR Batang, 137 mm 판형, 영문 라벨)을 그대로 쓴다.
지휘관 모델 색은 결과 도판(P1~P5)과 같은 CMD_COLOR, 단일 계열은 MONO.

    N1 latency_ecdf      지연 ECDF (모델별) + 25 s 결정 주기선
    N2 stability_time    churn·교차의 에피소드 시간축 추이 (모델별, 25 step 빈)
    N3 paired_diff       포메이션별 시드 대응차이 분포 (U-Net 효과, 제안 체계 효과)
    N4 metric_corr       10지표 대응차이의 Spearman 상관 히트맵
    N6 agreement         휴리스틱 일치율 (모델별) 과 llm_heur 포획률의 관계
    N7 rationale         근거 길이 분포 (모델별) 와 churn 의 관계
    N8 bc_warmup         행동복제 워밍업 손실 + 초기 300 갱신 엔트로피 (3시드)
    (N5 시간축 포획 곡선은 에피소드 내부 로그가 없어 만들 수 없다)

사용:
    python -m boatattack_sim.eval.paper_figs_extra
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import paper_names as PN
from . import stats as S
from .paper_figs import (CMD_COLOR, MONO, OI, _clean_axis, _panel_label, paper_figsize, save,
                         use_paper_style)

OUT = os.path.join("논문_그래프", "9_신규제안")
MODELS = ["qwen2.5-7b", "qwen2.5-14b", "gemini-3.5-flash-lite"]
MODEL_EN = {"qwen2.5-7b": "Qwen2.5 7B (local)", "qwen2.5-14b": "Qwen2.5 14B (local)",
            "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite (cloud)"}
FORM_EN = {"concentrated": "Concentrated", "diversionary": "Diversionary", "wave": "Wave"}
METRIC_EN = {
    "capture_rate": "Capture rate", "breaches": "Breaches", "collision_rate": "Collision rate",
    "net_touches": "Net entanglements", "nets_per_capture": "Nets per capture",
    "traveled_per_capture": "Distance per capture", "turn_sum_rad": "Total turning",
    "cap_dist_mean": "Capture distance", "cap_time_mean": "Capture time",
    "net_deploy_edist": "Enemy distance at deploy",
}
PERIOD = 25.0     # 결정 주기 (s) --- 25 step x 1 s


def _load():
    e = pd.read_csv(os.path.join("results", "eval_merged", "episodes.csv"))
    l = pd.read_csv(os.path.join("results", "eval_merged", "llm_metrics.csv"))
    return e, l


# ════════════════════════════════════════════════════════════════════════════
# N1. 지연 ECDF
# ════════════════════════════════════════════════════════════════════════════
def fig_latency_ecdf(l):
    use_paper_style()
    fig, ax = plt.subplots(figsize=paper_figsize(rel_width=0.62, ratio=0.72))
    for m in MODELS:
        x = np.sort(l.loc[l["model"] == m, "latency_s"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        over = 100.0 * (x > PERIOD).mean()
        ax.step(x, y, where="post", color=CMD_COLOR[m], lw=1.2,
                label=f"{MODEL_EN[m]} (n = {len(x):,}; >25 s: {over:.1f} %)")
    ax.axvline(PERIOD, color=MONO["mid"], ls=(0, (4, 2)), lw=0.8)
    ax.text(PERIOD * 0.93, 0.5, "decision period 25 s", fontsize=6, color=MONO["mid"], rotation=90,
            va="center", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 400)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Commander response latency (s)")
    ax.set_ylabel("Fraction of calls")
    ax.grid(False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1, bbox_to_anchor=(0.55, 0.0),
               frameon=False, fontsize=5.8)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# N2. 계획 안정성의 시간축 추이
# ════════════════════════════════════════════════════════════════════════════
def fig_stability_time(l, *, bin_steps: int = 50):
    use_paper_style()
    d = l.copy()
    d["tb"] = (d["t"] // bin_steps) * bin_steps + bin_steps / 2
    d = d[d["t"] <= 700]
    fig, axes = plt.subplots(1, 2, figsize=paper_figsize(nrows=1, ncols=2, ratio=0.66), sharex=True)
    for ax, col, ylab in zip(axes, ("churn", "crossings"),
                             ("Assignment churn", "Crossing assignments")):
        for m in MODELS:
            g = d[d["model"] == m].groupby("tb")[col]
            mu = g.mean(); n = g.size(); se = g.std() / np.sqrt(n)
            ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=CMD_COLOR[m], alpha=0.15, lw=0)
            ax.plot(mu.index, mu.values, color=CMD_COLOR[m], lw=1.2, label=MODEL_EN[m])
        ax.set_xlabel("Episode time (step)")
        ax.set_ylabel(ylab)
        _clean_axis(ax, xlim=(0, 700))
    axes[0].legend(loc="upper right", fontsize=5.8)
    _panel_label(axes[0], "a"); _panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.6)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# N3. 시드 대응차이 분포
# ════════════════════════════════════════════════════════════════════════════
def fig_paired_diff(e, *, metric: str = "capture_rate", best: str = "gemini-3.5-flash-lite"):
    use_paper_style()
    forms = ["concentrated", "diversionary", "wave"]
    pairs = [("heur_unet", "heur_heur", "Maneuver layer only (heur_unet − heur_heur)", OI["blue"]),
             (f"llm_unet@{best}", "heur_heur", "Full system (llm_unet − heur_heur)", OI["vermil"])]
    fig, axes = plt.subplots(1, 3, figsize=paper_figsize(nrows=1, ncols=3, ratio=0.95), sharey=True)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes, forms):
        pv = S.pivot_by_seed(e, metric, formation=f)
        for k, (a, b, lab, col) in enumerate(pairs):
            dlt = (pv[a] - pv[b]).dropna().to_numpy()
            est = S.paired_diff(pv[a].to_numpy(), pv[b].to_numpy())
            x0 = k
            jit = rng.uniform(-0.16, 0.16, len(dlt))
            ax.scatter(x0 + jit, dlt, s=7, color=col, alpha=0.45, lw=0, zorder=3)
            ax.errorbar([x0 + 0.32], [est.mean], yerr=[[est.mean - est.lo], [est.hi - est.mean]],
                        fmt="D", ms=3.2, color=col, ecolor=col, elinewidth=1.0, capsize=2, zorder=4)
            ax.text(x0 + 0.42, est.mean, f"{est.mean:+.3f}", ha="left", va="center", fontsize=5.8,
                    color=col)
        ax.axhline(0, color=MONO["mid"], lw=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Maneuver\nonly", "Full\nsystem"], fontsize=6.5)
        ax.set_xlim(-0.5, 1.9)
        ax.set_title(FORM_EN[f], fontsize=7, pad=3)
        ax.grid(False)
    axes[0].set_ylabel(f"Δ {METRIC_EN[metric].lower()} (seed-paired)")
    for ax, k in zip(axes, "abc"):
        _panel_label(ax, k, dx=-0.04)
    fig.tight_layout(w_pad=1.0)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# N4. 지표 대응차이 상관 히트맵
# ════════════════════════════════════════════════════════════════════════════
def fig_metric_corr(e, *, best: str = "gemini-3.5-flash-lite"):
    use_paper_style()
    keys = [k for k, *_ in PN.METRICS]
    cols = {}
    for k in keys:
        parts = []
        for f in ("concentrated", "diversionary", "wave"):   # 포메이션별 시드 쌍 → 90쌍
            pv = S.pivot_by_seed(e, k, formation=f)
            d = pv[f"llm_unet@{best}"] - pv["heur_heur"]
            d.index = [f"{f}:{i}" for i in d.index]
            parts.append(d)
        d = pd.concat(parts)
        # 개선 방향이 항상 양수가 되도록 부호 통일 (낮을수록 좋은 지표는 뒤집는다)
        cols[k] = -d if k in PN.LOWER_BETTER else d
    D = pd.DataFrame(cols).dropna()
    C = D.corr(method="spearman")
    fig, ax = plt.subplots(figsize=paper_figsize(rel_width=0.66, ratio=0.9))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(keys)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([METRIC_EN[k] for k in keys], rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels([METRIC_EN[k] for k in keys], fontsize=6)
    for i in range(n):
        for j in range(n):
            v = C.values[i, j]
            ax.text(j, i, f"{v:+.2f}" if i != j else "", ha="center", va="center", fontsize=4.8,
                    color="white" if abs(v) > 0.55 else MONO["ink"])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Spearman ρ (improvement direction)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(f"Seed-paired improvements, full system vs. baseline (n = {len(D)})", fontsize=7, pad=4)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# N6. 휴리스틱 일치율 vs 성능
# ════════════════════════════════════════════════════════════════════════════
def fig_agreement(e, l):
    use_paper_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=paper_figsize(nrows=1, ncols=2, ratio=0.72),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})
    # (a) 모델별 일치율: 값이 {0, 1/3, 2/3, 1} 로 이산이라 상자그림이 맞지 않는다 →
    #     평균 ± 95 % CI 막대 + "완전 일치" 호출 비율(빈 막대)
    xs = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        v = l.loc[l["model"] == m, "heuristic_agreement"].dropna().to_numpy()
        est = S.mean_ci(v)
        a1.bar(i - 0.18, est.mean, width=0.34, color=CMD_COLOR[m], edgecolor=MONO["ink"], lw=0.5,
               yerr=[[est.mean - est.lo], [est.hi - est.mean]], error_kw=dict(elinewidth=0.8, capsize=2))
        a1.bar(i + 0.18, float((v >= 0.999).mean()), width=0.34, color="white", edgecolor=CMD_COLOR[m],
               lw=0.9, hatch="////")
    a1.plot([], [], color=MONO["ink"], lw=4, label="Mean agreement (95 % CI)")
    a1.plot([], [], color=MONO["mid"], lw=0.9, label="Fraction of calls with full agreement")
    a1.set_xticks(xs); a1.set_xticklabels(["7B", "14B", "Gemini"], fontsize=6.5)
    a1.set_ylabel("Agreement with heuristic")
    a1.set_ylim(0, 0.8); a1.grid(False)
    a1.legend(loc="upper left", fontsize=5.6)
    # (b) 일치율 평균 vs llm_heur 포획률 (모델당 1점, 양방향 CI)
    for m in MODELS:
        ag = S.mean_ci(l.loc[l["model"] == m, "heuristic_agreement"].dropna().to_numpy())
        cr = S.mean_ci(e.loc[e["condition"] == f"llm_heur@{m}", "capture_rate"].to_numpy())
        a2.errorbar([ag.mean], [cr.mean], xerr=[[ag.mean - ag.lo], [ag.hi - ag.mean]],
                    yerr=[[cr.mean - cr.lo], [cr.hi - cr.mean]], fmt="o", ms=4, color=CMD_COLOR[m],
                    ecolor=CMD_COLOR[m], elinewidth=0.9, capsize=2, label=MODEL_EN[m])
    base = S.mean_ci(e.loc[e["condition"] == "heur_heur", "capture_rate"].to_numpy())
    a2.axhline(base.mean, color=MONO["mid"], ls=(0, (4, 2)), lw=0.8)
    a2.text(0.98, base.mean + 0.006, f"heuristic baseline {base.mean:.3f}", fontsize=5.8, color=MONO["mid"],
            transform=a2.get_yaxis_transform(), ha="right")
    a2.set_xlabel("Mean agreement with heuristic")
    a2.set_ylabel("Capture rate (LLM assignment)")
    a2.set_xlim(0.3, 0.8); a2.grid(False)
    a2.legend(loc="lower right", fontsize=5.6)
    _panel_label(a1, "a"); _panel_label(a2, "b")
    fig.tight_layout(w_pad=1.6)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# N7. 근거 길이와 churn
# ════════════════════════════════════════════════════════════════════════════
def fig_rationale(l):
    use_paper_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=paper_figsize(nrows=1, ncols=2, ratio=0.7))
    bins = np.linspace(0, 700, 36)
    for m in MODELS:
        x = l.loc[l["model"] == m, "rationale_chars"].dropna().clip(upper=700).to_numpy()
        a1.hist(x, bins=bins, histtype="step", color=CMD_COLOR[m], lw=1.1, density=True,
                label=f"{MODEL_EN[m]} (median {np.median(x):.0f})")
    a1.set_xlabel("Rationale length (characters)")
    a1.set_ylabel("Density")
    a1.grid(False)
    fig_legend = a1.get_legend_handles_labels()
    # (b) 근거 길이 분위 vs churn (모델별)
    for m in MODELS:
        d = l[l["model"] == m].dropna(subset=["rationale_chars", "churn"])
        q = pd.qcut(d["rationale_chars"], 6, duplicates="drop")
        g = d.groupby(q, observed=True)["churn"]
        xs = d.groupby(q, observed=True)["rationale_chars"].median()
        mu, se = g.mean(), g.std() / np.sqrt(g.size())
        a2.errorbar(xs.values, mu.values, yerr=1.96 * se.values, fmt="o-", ms=3, lw=1.0,
                    color=CMD_COLOR[m], capsize=1.5, label=MODEL_EN[m])
    a2.set_xlabel("Rationale length (characters, sextile median)")
    a2.set_ylabel("Assignment churn")
    a2.grid(False)
    _panel_label(a1, "a"); _panel_label(a2, "b")
    fig.tight_layout(w_pad=1.6, rect=(0, 0.08, 1, 1))
    fig.legend(*fig_legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False,
               fontsize=6)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# N8. BC 워밍업 + 초기 엔트로피
# ════════════════════════════════════════════════════════════════════════════
def fig_bc_warmup(diag_root: str = os.path.join("results", "train_diag")):
    use_paper_style()
    runs = sorted(d for d in glob.glob(os.path.join(diag_root, "seed*")) if os.path.isdir(d))
    ink, light = MONO["ink"], MONO["light"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=paper_figsize(nrows=1, ncols=2, ratio=0.66))
    B = [pd.read_csv(os.path.join(r, "bc.csv")) for r in runs]
    n = min(len(b) for b in B)
    Y = np.stack([b["loss"].to_numpy()[:n] for b in B])
    for y in Y:
        a1.plot(np.arange(n), y, color=light, lw=0.6)
    a1.plot(np.arange(n), Y.mean(0), color=ink, lw=1.3, label="Seed mean")
    a1.plot([], [], color=light, lw=0.6, label="Individual seeds")
    a1.set_xlabel("Behaviour-cloning step"); a1.set_ylabel("Cross-entropy loss")
    a1.legend(loc="upper right", fontsize=6)
    _clean_axis(a1, xlim=(0, n))
    M = [pd.read_csv(os.path.join(r, "metrics.csv")) for r in runs]
    k = 300
    E = np.stack([m["ent_pix1"].to_numpy()[:k] for m in M])
    for y in E:
        a2.plot(np.arange(k), y, color=light, lw=0.6)
    a2.plot(np.arange(k), E.mean(0), color=ink, lw=1.3)
    a2.axvline(300, color=MONO["mid"], ls=(0, (4, 2)), lw=0.8)
    a2.set_xlabel("GRPO update (first 300, BC term active)"); a2.set_ylabel("Policy entropy (nat)")
    _clean_axis(a2, xlim=(0, k))
    _panel_label(a1, "a"); _panel_label(a2, "b")
    fig.tight_layout(w_pad=1.6)
    return fig


if __name__ == "__main__":
    e, l = _load()
    out = []
    out.append(save(fig_latency_ecdf(l), OUT, "N1_latency_ecdf"))
    out.append(save(fig_stability_time(l), OUT, "N2_stability_time"))
    out.append(save(fig_paired_diff(e), OUT, "N3_paired_diff"))
    out.append(save(fig_metric_corr(e), OUT, "N4_metric_corr"))
    out.append(save(fig_agreement(e, l), OUT, "N6_agreement"))
    out.append(save(fig_rationale(l), OUT, "N7_rationale"))
    out.append(save(fig_bc_warmup(), OUT, "N8_bc_warmup"))
    for p in out:
        print("[ok]", p)
