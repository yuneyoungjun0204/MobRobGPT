# -*- coding: utf-8 -*-
"""참고문헌 항목을 인용 스타일에 맞는 순서로 재배열한다.

왜 필요한가
    수동 `thebibliography` 에서는 \\bibitem 이 적힌 **물리적 순서가 곧 번호**다.
    Elsevier 번호 스타일은 본문에서 처음 인용된 순서로 번호를 매기므로, 목록이
    알파벳순이면 30개 항목의 번호가 전부 어긋난다(실제로 그랬다).
    반대로 저자-연도 스타일이면 알파벳순이 맞다.
    → 인용 스타일을 바꾸면 이 스크립트를 다시 돌려야 한다.

사용:
    python tools/sort_refs.py --order cite     # 번호 스타일 (기본)
    python tools/sort_refs.py --order alpha    # 저자-연도 스타일
    python tools/sort_refs.py --check          # 고치지 않고 어긋남만 보고 (종료코드 1)

인용 순서는 본문 파일을 논문 순서대로 읽어 \\cite* 등장 순으로 정한다.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

#: 본문 순서. 여기 없는 파일의 인용은 마지막에 붙는다.
BODY_ORDER = ("01_intro", "02_related", "03_problem", "04_method", "05_experiment",
              "06_results", "07_discussion", "08_limitations", "09_conclusion",
              "10_declarations", "99_appendix")


def strip_comments(s: str) -> str:
    out = []
    for ln in s.splitlines():
        m = re.search(r"(?<!\\)%", ln)
        out.append(ln if m is None else ln[: m.start()])
    return "\n".join(out)


def citation_order(sec: str) -> list[str]:
    """본문에서 처음 인용된 순서."""
    seen, order = set(), []
    names = list(BODY_ORDER)
    names += [os.path.splitext(f)[0] for f in sorted(os.listdir(sec))
              if f.endswith(".tex") and os.path.splitext(f)[0] not in BODY_ORDER]
    for name in names:
        p = os.path.join(sec, name + ".tex")
        if not os.path.exists(p):
            continue
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", strip_comments(io.open(p, encoding="utf-8").read())):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k and k not in seen:
                    seen.add(k)
                    order.append(k)
    return order


def split_refs(tex: str):
    """(머리말, [(키, 블록)...], 꼬리말) 로 쪼갠다."""
    i = tex.find(r"\begin{thebibliography}")
    j = tex.find(r"\end{thebibliography}")
    if i < 0 or j < 0:
        raise SystemExit("thebibliography 환경을 찾지 못했다")
    line_end = tex.find("\n", i) + 1
    body = tex[line_end:j]
    parts = re.split(r"(?=\\bibitem)", body)
    head = tex[:line_end] + parts[0]
    items = []
    for p in parts[1:]:
        m = re.match(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", p)
        if not m:
            raise SystemExit(f"bibitem 형식이 예상과 다르다: {p[:60]}")
        items.append((m.group(1), p.rstrip() + "\n\n"))
    return head, items, tex[j:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", default="논문초안")
    ap.add_argument("--order", choices=("cite", "alpha"), default="cite",
                    help="cite = 번호 스타일(최초 인용 순) · alpha = 저자-연도 스타일")
    ap.add_argument("--check", action="store_true", help="고치지 않고 어긋남만 보고")
    a = ap.parse_args()

    sec = os.path.join(a.paper, "sections")
    refs = os.path.join(sec, "refs.tex")
    tex = io.open(refs, encoding="utf-8").read()
    head, items, tail = split_refs(tex)
    keys = [k for k, _ in items]

    cited = citation_order(sec)
    missing = [k for k in cited if k not in keys]
    unused = [k for k in keys if k not in cited]
    if missing:
        print(f"[!!] 인용했으나 목록에 없음: {missing}")
    if unused:
        print(f"[!!] 목록에 있으나 인용 안 함: {unused}")

    if a.order == "cite":
        rank = {k: i for i, k in enumerate(cited)}
        want = sorted(keys, key=lambda k: rank.get(k, 10**6))
        why = "번호 스타일 — 최초 인용 순"
    else:
        want = sorted(keys, key=str.lower)
        why = "저자-연도 스타일 — 알파벳 순"

    if keys == want:
        print(f"[ok] 이미 {why} 이다 ({len(keys)}개)")
        return 1 if (missing or unused) else 0

    n = sum(1 for x, y in zip(keys, want) if x != y)
    print(f"[!!] 순서 어긋남 — {len(keys)}개 중 {n}개 자리가 다르다 (필요: {why})")
    if a.check:
        for i, (x, y) in enumerate(zip(keys, want), 1):
            if x != y:
                print(f"     [{i:2}] {x}  →  {y}")
        return 1

    by = dict(items)
    io.open(refs, "w", encoding="utf-8", newline="\n").write(
        head + "".join(by[k] for k in want) + tail)
    print(f"[ok] refs.tex 재배열 완료 — {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
