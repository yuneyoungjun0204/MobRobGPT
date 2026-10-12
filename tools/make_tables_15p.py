# -*- coding: utf-8 -*-
"""국방학술대회 15쪽판(sections_kdu/) 전용 결과 표 생성기.

make_paper_tables.py 와 같은 원자료·같은 통계 함수로 굽되 두 가지만 다르다.
  1) 지표에서 돌파 수를 뺀다 --- 평가 720 에피소드 모두 적 M척이 포획 또는 돌파로
     결판나므로(resolved_frac = 1) 돌파 수 = M(1 - 포획률) 이 항등식이다. 같은 정보를
     두 번 세면 Bonferroni 보정 대상만 늘어난다. 그래서 m = 9.
  2) 거리 지표(m)를 관측 픽셀 한 변 ell 단위로 적는다 --- 15쪽판은 물리 단위를 싣지 않는다.
공유 표(논문초안/tables/)는 다른 판이 쓰므로 건드리지 않고 sections_kdu/*_ell.tex 로 낸다.

    python tools/make_tables_15p.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_paper_tables as T  # noqa: E402

S = T.S
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "논문초안", "sections_kdu")
ELL_M = 252.0     # 기준 설정의 관측 픽셀 한 변(m). 원자료의 거리 단위가 m 이다.
DIST_KEYS = {"traveled_per_capture", "cap_dist_mean", "net_deploy_edist"}

T.METRICS = tuple(m for m in T.METRICS if m[0] != "breaches")
T.BONF_M = len(T.METRICS)
T.BONF_LEVEL = 1.0 - 0.05 / T.BONF_M
BONF_PCT = f"{100 * T.BONF_LEVEL:.1f}"


def _ell_rows(tex: str, value_cols: int) -> str:
    """'(m)' 이 붙은 행의 앞쪽 `value_cols` 개 칸 숫자를 ell 단위로 바꾼다(나머지 칸은 그대로)."""
    out = []
    for line in tex.split("\n"):
        if " (m) &" in line:
            label, rest = line.split(" &", 1)
            cells = rest.split("&")
            for i in range(min(value_cols, len(cells))):
                parts = re.split(r"(\$\^\{[^}]*\}\$)", cells[i])
                cells[i] = "".join(p if p.startswith("$^") else re.sub(
                    r"[+-]?\d+(?:\.\d+)?",
                    lambda m: (("+" if m.group().startswith("+") else "")
                               + f"{float(m.group()) / ELL_M:.2f}"), p) for p in parts)
            line = label.replace(" (m)", r" ($\ell$)") + " &" + "&".join(cells)
        out.append(line)
    return "\n".join(out)


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "eval_merged", "episodes.csv"), encoding="utf-8-sig")
    assert (df.captures + df.breaches == df.captures.iloc[0] + df.breaches.iloc[0]).all(), \
        "결판나지 않은 에피소드가 있다 --- 돌파 수를 지표에서 뺄 근거가 사라진다"
    head = (f"% 자동 생성 --- tools/make_tables_15p.py (지표 {T.BONF_M}개, 돌파 수 제외, 거리는 ell 단위). "
            "손으로 고치지 말 것.")
    with tempfile.TemporaryDirectory() as tmp:
        T.tab_effectsize(df, tmp)
        T.tab_interaction(df, tmp)
        es = open(os.path.join(tmp, "tab_effectsize.tex"), encoding="utf-8").read()
        it = open(os.path.join(tmp, "tab_interaction.tex"), encoding="utf-8").read()
    es = es.replace("지표 10개에 대한 Bonferroni 보정(99.5\\,\\% 구간)",
                    f"지표 {T.BONF_M}개에 대한 Bonferroni 보정({BONF_PCT}\\,\\% 구간)")
    es = _ell_rows(es, value_cols=4)      # 방향, baseline, 제안, Δ  (d_z 는 무차원)
    it = _ell_rows(it, value_cols=2)      # 총효과, 상호작용 [CI]
    for name, tex in (("tab_effectsize_ell.tex", es), ("tab_interaction_ell.tex", it)):
        tex = tex.split("\n", 1)[1]
        open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n").write(head + "\n" + tex)
        print("[ok]", name)

    # 본문 서술용: 상호작용이 95 % / 보정 수준에서 유의하지 않은 지표 수, 효과크기가 보정에서 유의성을 잃는 지표
    n95 = nb = 0
    lost = []
    for met, ko, _a, _nd in T.METRICS:
        piv = S.pivot_by_seed(df, met)
        hh, hu = piv["heur_heur"].to_numpy(), piv["heur_unet"].to_numpy()
        lh = piv["llm_heur@gemini-3.5-flash-lite"].to_numpy()
        lu = piv["llm_unet@gemini-3.5-flash-lite"].to_numpy()
        e95 = S.paired_diff(lu - lh, hu - hh)
        eb = S.paired_diff(lu - lh, hu - hh, confidence_level=T.BONF_LEVEL)
        n95 += int(e95.lo <= 0 <= e95.hi)
        nb += int(eb.lo <= 0 <= eb.hi)
        d95 = S.paired_diff(lu, hh)
        db = S.paired_diff(lu, hh, confidence_level=T.BONF_LEVEL)
        if not (d95.lo <= 0 <= d95.hi) and (db.lo <= 0 <= db.hi):
            lost.append(ko)
    print(f"[본문] 지표 {T.BONF_M}개 | 상호작용 비유의: 95 % {n95}개, 보정({BONF_PCT} %) {nb}개 "
          f"| 효과크기가 보정에서 유의성을 잃는 지표: {lost}")


if __name__ == "__main__":
    main()
