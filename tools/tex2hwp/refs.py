# -*- coding: utf-8 -*-
"""thebibliography 항목(하버드식) → 국방학술대회 양식의 참고문헌 형식.

양식: 이름, "논문제목", 저널명(이탤릭), Vol.권-호, (년도), p.111-111.
      서적: 이름, 서적명(이탤릭), 출판사, 출판지(년도).
돌려주는 값은 항목마다 [(글자, 이탤릭 여부), ...] 조각 목록이다.
"""
from __future__ import annotations

import os
import re

from .common import PAPER


def _clean(s: str) -> str:
    s = s.replace("{\\L}", "Ł").replace("\\&", "&").replace("--", "-").replace("~", " ")
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load(path: str | None = None) -> list[tuple[str, str]]:
    """(키, 본문) 목록을 실린 순서대로."""
    path = path or os.path.join(PAPER, "sections_def", "refs.tex")
    src = open(path, encoding="utf-8").read()
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
    body = src.split("\\begin{thebibliography}", 1)[1].split("\\end{thebibliography}", 1)[0]
    items = re.split(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", body)[1:]
    return [(items[k], _clean(items[k + 1])) for k in range(0, len(items), 2)]


def _pages(tail: str) -> tuple[str, str]:
    """', 55(6), pp.1136-1146.' → ('Vol.55-6', 'p.1136-1146')"""
    vol = pages = ""
    m = re.search(r"(\d+)\(([^)]*)\)", tail)
    if m:
        vol = f"Vol.{m.group(1)}-{m.group(2)}"
    else:
        m2 = re.match(r",\s*(\d+),", tail)
        if m2:
            vol = f"Vol.{m2.group(1)}"
    m = re.search(r"pp?\.\s*([\d\-–]+)", tail)
    if m:
        pages = f"p.{m.group(1)}"
    else:
        m = re.search(r",\s*(\d{4,})\.?$", tail.strip())
        if m:
            pages = f"No.{m.group(1)}"
    return vol, pages


def format_entry(text: str) -> list[tuple[str, bool]]:
    m = re.match(r"(.*?),\s*(\d{4})\.\s*(.*)", text)
    if not m:
        return [(text, False)]
    authors, year, rest = m.groups()
    parts: list[tuple[str, bool]] = [(authors + ", ", False)]
    if rest.startswith("\\textit{"):
        # 서적·웹 문서: 제목이 이탤릭
        title, after = re.match(r"\\textit\{(.*?)\}\.?\s*(.*)", rest).groups()
        after = re.sub(r"\.$", "", after).strip()
        # 서적의 '출판지: 출판사' → 양식 순서 '출판사, 출판지'
        mp = re.match(r"([^:\[]+):\s*(.+)$", after)
        if mp and "Available" not in after:
            after = f"{mp.group(2)}, {mp.group(1)}"
        parts.append((title, True))
        parts.append(((", " + after if after else "") + f" ({year}).", False))
        return parts
    mt = re.match(r"(.*?)\.\s*\\textit\{(.*?)\}(.*)", rest)
    if not mt:
        return parts + [(rest, False)]
    title, journal, tail = mt.groups()
    vol, pages = _pages(tail)
    parts.append((f"\u201c{title}\u201d, ", False))
    parts.append((journal, True))
    extra = "".join(f", {x}" for x in (vol,) if x)
    parts.append((f"{extra}, ({year})" + (f", {pages}." if pages else "."), False))
    return parts
