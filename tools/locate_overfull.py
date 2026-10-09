# -*- coding: utf-8 -*-
"""LaTeX 로그의 Overfull 이 **어느 파일**에서 났는지 괄호 스택으로 복원한다.

'직전에 등장한 파일명' 으로 짚으면 틀린다 — 이 저장소에서만 세 번 틀렸다.
LaTeX 은 파일을 열 때 '(경로', 닫을 때 ')' 를 로그에 찍으므로, 그 균형을 세면
경고가 난 순간 실제로 읽고 있던 파일이 나온다. 넘친 상자의 내용도 함께 보인다.

사용:
    python tools/locate_overfull.py 논문초안/USV_swarm_defense_EAAI_KR_real.log
"""
from __future__ import annotations

import io
import re
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    raw = io.open(sys.argv[1], encoding="utf-8", errors="replace").read()
    lines = raw.split("\n")
    tok = re.compile(
        r"\((?:\./)?([^()\s]*\.(?:tex|sty|cls|def|cfg|fd|clo))"
        r"|\)"
        r"|Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)([^\n]*)")
    stack: list[str] = []
    n = 0
    for m in tok.finditer(raw):
        if m.group(1):
            stack.append(m.group(1))
        elif m.group(2):
            n += 1
            where = next((f for f in reversed(stack) if "/" in f and not f.endswith(".sty")),
                         stack[-1] if stack else "?")
            # 넘친 상자의 내용은 다음 줄에 찍힌다
            ln = raw.count("\n", 0, m.start())
            content = lines[ln + 1][:140] if ln + 1 < len(lines) else ""
            print(f"[{n}] {float(m.group(2)):7.1f}pt {m.group(3).strip()}  ← {where}")
            print(f"      {content}")
        else:
            if stack:
                stack.pop()
    if n == 0:
        print("Overfull 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
