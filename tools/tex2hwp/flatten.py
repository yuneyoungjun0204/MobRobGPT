# -*- coding: utf-8 -*-
"""국방발표 판 마스터 tex 를 pandoc 이 읽을 수 있는 한 파일로 펼친다.

- \\input 을 재귀로 펼친다(주석 줄은 버린다).
- \\inputshortcap{캡션}{파일} 은 표 파일의 \\caption{...} 을 짧은 캡션으로 바꿔 넣는다.
- equationsmall / equationfn 은 equation 으로 되돌린다(글자 크기만 다른 환경).
- 표 안의 \\resizebox{...}{!}{ ... } 껍데기를 벗긴다.
"""
from __future__ import annotations

import os
import re

from .common import PAPER


def _strip_comments(src: str) -> str:
    out = []
    for line in src.splitlines():
        # 이스케이프되지 않은 % 부터 줄 끝까지 버린다
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def _read(rel: str) -> str:
    p = os.path.join(PAPER, rel if rel.endswith(".tex") else rel + ".tex")
    with open(p, encoding="utf-8") as f:
        return _strip_comments(f.read())


def _balanced(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' 에서 짝 맞는 '}' 까지의 내용과 그 다음 위치."""
    assert s[i] == "{"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1: j], j + 1
    raise ValueError("unbalanced")


def _replace_caption(src: str, cap: str) -> str:
    i = src.find("\\caption{")
    if i < 0:
        return src
    _, end = _balanced(src, i + len("\\caption"))
    return src[:i] + "\\caption{" + cap + "}" + src[end:]


def _strip_resizebox(src: str) -> str:
    while True:
        i = src.find("\\resizebox")
        if i < 0:
            return src
        j = i + len("\\resizebox")
        _, j = _balanced(src, j)          # 폭
        _, j = _balanced(src, j)          # 높이 {!}
        while src[j].isspace():
            j += 1
        body, k = _balanced(src, j)
        src = src[:i] + body + src[k:]


def expand(src: str) -> str:
    out, i = [], 0
    pat = re.compile(r"\\(inputshortcap|input)\{")
    while True:
        m = pat.search(src, i)
        if not m:
            out.append(src[i:])
            break
        out.append(src[i: m.start()])
        if m.group(1) == "input":
            name, i = _balanced(src, m.end() - 1)
            out.append(expand(_read(name)))
        else:
            cap, j = _balanced(src, m.end() - 1)
            name, i = _balanced(src, j)
            out.append(_replace_caption(expand(_read(name)), cap))
    return "".join(out)


def flatten(master: str) -> str:
    src = _read(master)
    pre, body = src.split("\\begin{document}", 1)
    body = body.split("\\end{document}", 1)[0]
    body = expand(body)
    body = re.sub(r"\\(begin|end)\{equation(small|fn)\}", r"\\\1{equation}", body)
    body = _strip_resizebox(body)
    # pandoc 이 쓰는 매크로 정의만 남긴다
    macros = "\n".join(l for l in pre.splitlines() if l.startswith("\\newcommand{\\capr")
                       or l.startswith("\\newcommand{\\dz") or l.startswith("\\newcommand{\\unit"))
    return macros + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
