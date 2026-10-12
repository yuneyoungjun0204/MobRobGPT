# -*- coding: utf-8 -*-
"""15쪽판 학습 알고리즘 비교표(영문, 한글 양식용) --- GRPO(채택) 대 MAPPO(시드 3런) 대 휴리스틱.

make_algo_table.py 와 같은 원자료·같은 대응 비교를 쓰되 15쪽판 방침에 맞춘다.
  - 지표 9개(돌파 수 제외: 포획률과 항등), Bonferroni m = 9
  - 거리 지표는 관측 픽셀 한 변 ell(=252 m) 단위
  - 포메이션별 포획률 3행을 같은 표 아래에 붙인다(Δ 는 위와 같이 GRPO − MAPPO;
    휴리스틱 대비 95 % 유의는 값 옆 ‡)
출력: LaTeX 표 조각(영문)과 표 주석을 JSON 으로 --- hwp 패치 스크립트가 읽는다.

    python tools/make_algo_table_15p.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boatattack_sim.eval import paper_names as PN          # noqa: E402
from boatattack_sim.eval import stats as S                 # noqa: E402
from tools.make_algo_table import MAPPO, load_algo_frame   # noqa: E402

ELL_M = 252.0
BONF = 1.0 - 0.05 / 9
EN = {"capture_rate": "Capture rate", "collision_rate": "Collision rate (per defender)",
      "net_touches": "Net entanglements", "nets_per_capture": "Nets per capture",
      "traveled_per_capture": "Distance per capture ($\\ell$)", "turn_sum_rad": "Total turning (rad)",
      "cap_dist_mean": "Capture distance ($\\ell$)", "cap_time_mean": "Capture time (step)",
      "net_deploy_edist": "Enemy distance at deploy ($\\ell$)"}
FORM = {"concentrated": "Concentrated", "diversionary": "Diversionary", "wave": "Wave"}
DIST = {"traveled_per_capture", "cap_dist_mean", "net_deploy_edist"}


def _excl(e):
    return e.lo > 0 or e.hi < 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/eval_merged/episodes.csv")
    ap.add_argument("--runs", nargs="+", default=[f"results/ablation/mappo_s{k}" for k in range(3)])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df, per_run = load_algo_frame(a.base, a.runs)
    rows, facts = [], {}
    for k, _ko, lower, nd in PN.METRICS:
        if k == "breaches":
            continue
        piv = S.pivot_by_seed(df, k)
        ok = piv[["heur_heur", MAPPO, "heur_unet"]].dropna()
        h, m, g = (ok[c].to_numpy() for c in ("heur_heur", MAPPO, "heur_unet"))
        sc = ELL_M if k in DIST else 1.0
        if k in DIST:
            nd = 2
        d = S.paired_diff(g / sc, m / sc)
        db = S.paired_diff(g / sc, m / sc, confidence_level=BONF)
        dz = S.cohens_dz(g, m)
        f = "{:.%df}" % nd
        fs = "{:+.%df}" % nd
        mark = ("$^{*}$" if _excl(d) else "") + ("$^{\\dagger}$" if _excl(db) else "")
        arrow = "$\\downarrow$" if lower else "$\\uparrow$"
        rows.append(f"{EN[k]} {arrow} & {f.format(h.mean() / sc)} & {f.format(m.mean() / sc)} & "
                    f"{f.format(g.mean() / sc)} & {fs.format(d.mean)} [{fs.format(d.lo)}, {fs.format(d.hi)}]"
                    f"{mark} & {dz:+.2f} \\\\")
        facts[k] = dict(h=h.mean() / sc, m=m.mean() / sc, g=g.mean() / sc, d=d.mean, lo=d.lo, hi=d.hi,
                        sig=_excl(d), bonf=_excl(db), dz=dz)
    rows.append("\\midrule")
    for fk, fen in FORM.items():
        piv = S.pivot_by_seed(df, "capture_rate", formation=fk)
        ok = piv[["heur_heur", MAPPO, "heur_unet"]].dropna()
        h, m, g = (ok[c].to_numpy() for c in ("heur_heur", MAPPO, "heur_unet"))
        dm, dg = S.paired_diff(m, h), S.paired_diff(g, h)
        d = S.paired_diff(g, m)
        db = S.paired_diff(g, m, confidence_level=BONF)
        mark = ("$^{*}$" if _excl(d) else "") + ("$^{\\dagger}$" if _excl(db) else "")
        mm = f"{m.mean():.3f}" + (" ‡" if _excl(dm) else "")
        gm = f"{g.mean():.3f}" + (" ‡" if _excl(dg) else "")
        rows.append(f"Capture rate, {fen} & {h.mean():.3f} & {mm} & {gm} & {d.mean:+.3f} "
                    f"[{d.lo:+.3f}, {d.hi:+.3f}]{mark} & {S.cohens_dz(g, m):+.2f} \\\\")
        facts["form_" + fk] = dict(h=h.mean(), m=m.mean(), g=g.mean(), m_vs_h=dm.mean, m_vs_h_sig=_excl(dm),
                                   g_vs_h=dg.mean, g_vs_h_sig=_excl(dg), d=d.mean, sig=_excl(d))
    caps = ", ".join(f"{r['capture_rate'].mean():.3f}" for r in per_run)
    tex = ("\\begin{table}\n\\caption{x}\\label{tab:algo15}\n\\begin{tabular}{lccccc}\n\\toprule\n"
           "Metric & Heuristic & MAPPO & GRPO (adopted) & $\\Delta$ [95 \\% CI] & $d_z$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    note = ("$\\Delta$, $d_z$: GRPO $-$ MAPPO, seed-paired. $^{*}$ 95 % CI excludes 0. $^{\\dagger}$ Also excludes 0 "
            "under Bonferroni correction for 9 metrics. ‡ Differs from the heuristic (95 % CI). MAPPO: mean of three "
            f"training seeds (capture rates {caps}); heuristic assignment in all columns.")
    json.dump({"tex": tex, "note": note, "facts": facts}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(tex)
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
