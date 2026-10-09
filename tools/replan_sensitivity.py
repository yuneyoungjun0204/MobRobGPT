# -*- coding: utf-8 -*-
"""재계획 주기 민감도 --- 제안 체계(llm_unet@Gemini), 파상 포메이션, 시드 0–29.

results/eval_merged (replan_every=1, 25 s) 에 results/eval_replan{2,4}_gemini (50 s, 100 s) 를
붙여, 주기별 평균과 25 s 대비 시드 대응 차이의 BCa 95 % CI 를 표로 낸다.

    python tools/replan_sensitivity.py            # → 논문초안/tables/tab_replan.tex + 콘솔 요약

heur_heur / heur_unet 행은 replan 과 무관하므로 세 실행에서 같은 시드에 같은 값이어야 한다 ---
이를 결정성 검사로 함께 보고한다(불일치면 시뮬레이터가 바뀐 것이고 비교는 무효다).
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boatattack_sim.eval import paper_names as PN  # noqa: E402
from boatattack_sim.eval import stats as S  # noqa: E402

COND = "llm_unet@gemini-3.5-flash-lite"
FORM = "wave"
PERIOD_S = 25
RUNS = [  # (replan_every, episodes.csv)
    (1, os.path.join(ROOT, "results", "eval_merged", "episodes.csv")),
    (2, os.path.join(ROOT, "results", "eval_replan2_gemini", "episodes.csv")),
    (4, os.path.join(ROOT, "results", "eval_replan4_gemini", "episodes.csv")),
]
METRICS = ["capture_rate", "nets_per_capture", "turn_sum_rad", "traveled_per_capture"]
OUT_TEX = os.path.join(ROOT, "논문초안", "tables", "tab_replan.tex")


def _load(path: str, replan: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["formation"] == FORM].copy()
    if replan > 1:
        assert (df.loc[df["condition"].str.startswith("llm"), "replan_every"] == replan).all(), path
    return df


def determinism_check(frames: dict[int, pd.DataFrame]) -> list[str]:
    """비-LLM 두 조건이 세 실행에서 같은 시드에 같은 값인지."""
    notes = []
    base = frames[1]
    for r, df in frames.items():
        if r == 1:
            continue
        for cond in ("heur_heur", "heur_unet"):
            a = base[base.condition == cond].set_index("seed").sort_index()
            b = df[df.condition == cond].set_index("seed").sort_index()
            common = a.index.intersection(b.index)
            for col in ("capture_rate", "nets_used", "turn_sum_rad"):
                same = np.allclose(a.loc[common, col].to_numpy(float), b.loc[common, col].to_numpy(float), atol=1e-9)
                notes.append(f"replan{r} {cond:9s} {col:14s} n={len(common)} {'일치' if same else '★불일치'}")
    return notes


def build(frames: dict[int, pd.DataFrame]):
    rows = {}
    for r, df in frames.items():
        sub = df[df.condition == COND].set_index("seed").sort_index()
        rows[r] = sub
    seeds = sorted(set.intersection(*[set(v.index) for v in rows.values()]))
    base = rows[1].loc[seeds]
    heur = frames[1][frames[1].condition == "heur_heur"].set_index("seed").loc[seeds]
    out = []   # (period_s, n, {metric: (mean, Estimate|None)})
    for r in sorted(rows):
        sub = rows[r].loc[seeds]
        cells = {}
        for m in METRICS:
            x = sub[m].to_numpy(float)
            est = None if r == 1 else S.paired_diff(x, base[m].to_numpy(float), seed=0)
            cells[m] = (float(np.nanmean(x)), est)
        out.append((r * PERIOD_S, len(seeds), cells))
    heur_row = {m: float(np.nanmean(heur[m].to_numpy(float))) for m in METRICS}
    return out, heur_row, len(seeds)


def _fmt(m: str, v: float) -> str:
    return f"{v:.3f}" if m == "capture_rate" else (f"{v/1000:.2f}" if m == "traveled_per_capture" else f"{v:.2f}")


def _fmt_d(m: str, e: S.Estimate) -> str:
    scale = 1000.0 if m == "traveled_per_capture" else 1.0
    star = "$^{*}$" if (e.lo > 0 or e.hi < 0) else ""
    if m == "capture_rate":
        return f"{e.mean:+.3f} [{e.lo:+.3f}, {e.hi:+.3f}]{star}"
    return f"{e.mean/scale:+.2f} [{e.lo/scale:+.2f}, {e.hi/scale:+.2f}]{star}"


def to_tex(out, heur_row, n) -> str:
    head = {"capture_rate": "포획률", "nets_per_capture": "포획당 그물", "turn_sum_rad": "총 선회량 (rad)",
            "traveled_per_capture": "포획당 이동거리 (km)"}
    L = []
    L.append("% 자동 생성 — tools/replan_sensitivity.py. 손으로 고치지 말 것.")
    L.append("\\begin{table*}[tp]")
    L.append("\\centering")
    L.append("\\caption{재계획 주기 민감도 --- 제안 체계(Gemini 지휘관 + U-Net 기동), 파상 포메이션, 시드 %d개. "
             "결정 주기는 25\\,s 로 고정하고 지휘관 재계획만 1·2·4 결정마다 수행하였다. $\\Delta$ 는 25\\,s 대비 시드 대응 차이의 "
             "평균과 BCa 95\\,\\%% 신뢰구간이다.}" % n)
    L.append("\\label{tab:replan}")
    L.append("\\footnotesize")
    L.append("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%")
    L.append("\\begin{tabular}{l" + "c" * (len(METRICS) * 2) + "}")
    L.append("\\toprule")
    L.append("재계획 주기 & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{head[m]}}}" for m in METRICS) + " \\\\")
    L.append(" & " + " & ".join("평균 & $\\Delta$ [95\\,\\% CI]" for _ in METRICS) + " \\\\")
    L.append("\\midrule")
    for period, _n, cells in out:
        parts = []
        for m in METRICS:
            mean, est = cells[m]
            parts.append(_fmt(m, mean))
            parts.append("---" if est is None else _fmt_d(m, est))
        L.append(f"{period}\\,s & " + " & ".join(parts) + " \\\\")
    L.append("\\midrule")
    L.append("휴리스틱 기준선 & " + " & ".join(f"{_fmt(m, heur_row[m])} & " for m in METRICS).rstrip(" &") + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("}")
    L.append("\\\\[3pt]")
    L.append("\\begin{minipage}{\\linewidth}\\footnotesize")
    L.append("$^{*}$ 신뢰구간이 0 을 포함하지 않음. 휴리스틱 기준선(heur\\_heur)은 같은 시드의 참고값이다. "
             "지휘관 호출 수는 주기에 반비례한다(25\\,s 기준 1 : 1/2 : 1/4).")
    L.append("\\end{minipage}")
    L.append("\\end{table*}")
    return "\n".join(L) + "\n"


def main():
    frames = {r: _load(p, r) for r, p in RUNS}
    for line in determinism_check(frames):
        print("[det]", line)
    out, heur_row, n = build(frames)
    for period, _n, cells in out:
        print(f"[{period:3d}s] n={_n} " + "  ".join(
            f"{m}={cells[m][0]:.3f}" + ("" if cells[m][1] is None else f" (Δ {cells[m][1].mean:+.3f} [{cells[m][1].lo:+.3f},{cells[m][1].hi:+.3f}])")
            for m in METRICS))
    print("[heur]", {m: round(v, 3) for m, v in heur_row.items()})
    tex = to_tex(out, heur_row, n)
    io.open(OUT_TEX, "w", encoding="utf-8", newline="\n").write(tex)
    print("[ok]", OUT_TEX)


if __name__ == "__main__":
    main()
