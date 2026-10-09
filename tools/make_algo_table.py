# -*- coding: utf-8 -*-
"""학습 알고리즘 비교표 — 같은 픽셀 행동공간 위의 GRPO(채택) 대 MAPPO(시드 3런).

입력
  results/eval_merged/episodes.csv         : 논문 본 평가 (heur_heur, heur_unet=GRPO 채택 런)
  results/ablation/mappo_s{k}/episodes.csv : run_eval --skip-llm 로 잰 MAPPO 런 k (heur_unet 행이 MAPPO)
출력
  논문초안/tables/tab_algo.tex             : 10 지표 × (휴리스틱, MAPPO, GRPO, Δ[CI], d_z)
  results/ablation/algo_episodes.csv       : 세 조건을 한 CSV 로 (그림용). 조건 heur_mappo 는 3런 평균.

MAPPO 3런은 같은 평가 시드(0--29)·포메이션을 쓰므로 (seed, formation) 마다 3런의 지표를 평균해
하나의 조건 heur_mappo 로 둔다 --- 학습 시드에 따른 운을 평균으로 지운 뒤 GRPO 와 대응 비교한다.
런별 포획률은 각주에 따로 적어 분산을 숨기지 않는다.

    python tools/make_algo_table.py --runs results/ablation/mappo_s0 results/ablation/mappo_s1 results/ablation/mappo_s2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boatattack_sim.eval import paper_names as PN          # noqa: E402
from boatattack_sim.eval import stats as S                 # noqa: E402
from tools.make_paper_tables import _wrap, _sig, _sig_bonf, _write, FORM_KO   # noqa: E402

MAPPO = "heur_mappo"


def load_algo_frame(base_csv: str, runs: list[str]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """본 평가 + MAPPO 런들을 한 프레임으로. 반환 (합본, 런별 heur_unet 프레임)."""
    base = pd.read_csv(base_csv)
    keep = base[base["condition"].isin(["heur_heur", "heur_unet"])].copy()
    per_run = []
    for r in runs:
        d = pd.read_csv(os.path.join(r, "episodes.csv"))
        per_run.append(d[d["condition"] == "heur_unet"].copy())
    keys = [k for k, *_ in PN.METRICS]
    stack = pd.concat(per_run, ignore_index=True)
    mean = (stack.groupby(["seed", "formation"])[keys].mean().reset_index())
    mean["condition"] = MAPPO
    mean["assign"] = "heuristic"; mean["maneuver"] = "unet_mappo"
    mean["label"] = "휴리스틱 배정 + MAPPO 기동"; mean["backend"] = ""; mean["replan_every"] = 0
    return pd.concat([keep, mean], ignore_index=True), per_run


def tab_algo(df: pd.DataFrame, per_run: list[pd.DataFrame], out: str) -> None:
    rows = []
    for k, ko, lower, nd in PN.METRICS:
        piv = S.pivot_by_seed(df, k)
        ok = piv[["heur_heur", MAPPO, "heur_unet"]].dropna()
        h, m, g = (ok[c].to_numpy() for c in ("heur_heur", MAPPO, "heur_unet"))
        d = S.paired_diff(g, m)                       # GRPO − MAPPO
        arrow = r"$\downarrow$" if lower else r"$\uparrow$"
        fmt = "{:.%df}" % nd
        rows.append(f"{ko} {arrow} & {fmt.format(h.mean())} & {fmt.format(m.mean())} & "
                    f"{fmt.format(g.mean())} & {('{:+.%df}' % nd).format(d.mean)} "
                    f"[{('{:+.%df}' % nd).format(d.lo)}, {('{:+.%df}' % nd).format(d.hi)}]"
                    f"{_sig(d)}{_sig_bonf(g, m)} & {S.cohens_dz(g, m):+.2f} \\\\\n")
    caps = ", ".join(f"{r['capture_rate'].mean():.3f}" for r in per_run)
    tex = _wrap(
        "".join(rows),
        "같은 픽셀 행동공간 위의 학습 알고리즘 비교. 배정은 모두 휴리스틱이며 기동 정책의 "
        "액터·보상·초기화·환경 step 예산이 같고 갱신 규칙만 다르다. MAPPO 는 학습 시드 3런의 "
        "에피소드별 평균, $\\Delta$ 는 GRPO $-$ MAPPO 의 시드 대응 차이.",
        "tab:algo", "lccccc",
        "지표 & 휴리스틱 & MAPPO & GRPO(채택) & $\\Delta$ [95\\,\\% CI] & $d_z$ \\\\",
        f"$^{{*}}$ 95\\,\\% 구간이 0 을 배제. $^{{\\dagger}}$ Bonferroni 보정(99.5\\,\\%)에서도 배제. "
        f"MAPPO 런별 포획률 {caps}. 3개 포메이션 통합(각 조건 30시드).",
        wide=True)
    _write(out, "tab_algo.tex", tex)


def tab_algo_form(df: pd.DataFrame, out: str) -> None:
    """포메이션별 포획률 — Table 6(tab_maneuver)과 같은 꼴."""
    rows = []
    for f in ("concentrated", "diversionary", "wave", None):
        piv = S.pivot_by_seed(df, "capture_rate", formation=f)
        ok = piv[["heur_heur", MAPPO, "heur_unet"]].dropna()
        h, m, g = (ok[c].to_numpy() for c in ("heur_heur", MAPPO, "heur_unet"))
        dm = S.paired_diff(m, h); dg = S.paired_diff(g, h)
        name = FORM_KO.get(f, f) if f else "\\textbf{전체}"
        rows.append(f"{name} & {h.mean():.3f} & {m.mean():.3f} ({dm.mean:+.3f}{_sig(dm)}) & "
                    f"{g.mean():.3f} ({dg.mean:+.3f}{_sig(dg)}) \\\\\n")
    tex = _wrap("".join(rows),
                "포메이션별 포획률. 휴리스틱 기동 대비 MAPPO·GRPO 의 시드 대응 차이(괄호).",
                "tab:algo_form", "lccc",
                "포메이션 & 휴리스틱 & MAPPO & GRPO(채택) \\\\",
                "$^{*}$ 휴리스틱 대비 95\\,\\% 구간이 0 을 배제.", wide=False)
    _write(out, "tab_algo_form.tex", tex)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="results/eval_merged/episodes.csv")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="논문초안/tables")
    ap.add_argument("--csv", default="results/ablation/algo_episodes.csv")
    a = ap.parse_args()
    df, per_run = load_algo_frame(a.base, a.runs)
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    df.to_csv(a.csv, index=False)
    tab_algo(df, per_run, a.out)
    tab_algo_form(df, a.out)
    # 검증 출력: heur_heur 가 본 평가와 런 평가에서 같은 값인지(같은 시드 → 같아야 한다)
    base = pd.read_csv(a.base)
    hh0 = base[base["condition"] == "heur_heur"].groupby(["seed", "formation"])["capture_rate"].mean()
    for r in a.runs:
        d = pd.read_csv(os.path.join(r, "episodes.csv"))
        hh = d[d["condition"] == "heur_heur"].groupby(["seed", "formation"])["capture_rate"].mean()
        j = hh0.align(hh, join="inner")
        print(f"[check] {r}: heur_heur 재현 |Δ| max = {np.abs(j[0] - j[1]).max():.4f} "
              f"({len(j[0])} 조합)")
    print("[ok] tab_algo.tex / tab_algo_form.tex ->", a.out, "; csv ->", a.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
