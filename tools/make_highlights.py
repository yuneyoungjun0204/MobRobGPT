# -*- coding: utf-8 -*-
"""Highlights 를 **별도 제출 파일**로 뽑는다.

왜 필요한가
    Elsevier 투고에서 Highlights 는 원고와 별개의 제출 항목이다. elsarticle 의
    `highlights` 환경이 PDF 앞쪽에 별쪽을 만들어 주지만, 그것으로 제출 요건이
    채워지지는 않는다. 투고 시스템에 올릴 파일이 따로 필요하다.

왜 생성기로 두나
    손으로 복사해 두면 원고와 갈린다. 원고를 고치고 제출 파일을 안 고치는 일이
    반드시 생긴다. 두 파일 모두 **원고와 같은 소스**에서 뽑는다:
        국문  논문초안/sections/00c_highlights.tex        (highlights 환경)
        영문  논문초안/sections/00b_abstract_en.tex       (English Abstract 절 안)

출력
    논문초안/highlights.txt   투고 시스템에 붙여 넣는 평문 (영문이 먼저 — 영문 저널이다)
    논문초안/highlights.tex   1쪽 PDF 로 굽는 독립 문서 (첨부용)

사용:
    python tools/make_highlights.py
    xelatex -output-directory=논문초안 논문초안/highlights.tex   # PDF 가 필요하면
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys


def strip_comments(s: str) -> str:
    out = []
    for ln in s.splitlines():
        m = re.search(r"(?<!\\)%", ln)
        out.append(ln if m is None else ln[: m.start()])
    return "\n".join(out)


def items_after(tex: str, anchor: str, end: str | None = None) -> list[str]:
    """`anchor` 뒤의 \\item 본문들. itemize 를 끼든 안 끼든 받는다."""
    i = tex.find(anchor)
    if i < 0:
        return []
    body = tex[i + len(anchor):]
    j = body.find(r"\begin{itemize}")
    if j >= 0:
        k = body.find(r"\end{itemize}", j)
        body = body[j + len(r"\begin{itemize}"):k if k > 0 else len(body)]
    elif end:
        k = body.find(end)
        body = body[:k if k > 0 else len(body)]
    parts = [p.strip() for p in body.split(r"\item")[1:]]
    return [re.sub(r"\s+", " ", p).strip().rstrip("\\") for p in parts if p.strip()]


def plain(s: str) -> str:
    """LaTeX 을 벗겨 붙여 넣기 좋은 평문으로."""
    s = re.sub(r"\\emph\{([^}]*)\}|\\textbf\{([^}]*)\}|\\texttt\{([^}]*)\}",
               lambda m: next(g for g in m.groups() if g is not None), s)
    s = s.replace(r"\,", " ").replace(r"\%", "%").replace("---", "—")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()


def emit(path: str, text: str, check: bool) -> bool:
    """check 면 비교만, 아니면 쓴다. 어긋나면 False."""
    if check:
        cur = io.open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if cur == text:
            print(f"[ok] {path} 최신")
            return True
        print(f"[!!] {path} 가 원고와 다르다 — python tools/make_highlights.py 로 다시 생성할 것")
        return False
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    print(f"[ok] {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", default="논문초안")
    ap.add_argument("--check", action="store_true",
                    help="생성하지 않고, 디스크의 파일이 원고와 맞는지만 본다")
    a = ap.parse_args()
    sec = os.path.join(a.paper, "sections")

    ko_src = strip_comments(io.open(os.path.join(sec, "00c_highlights.tex"),
                                    encoding="utf-8").read())
    en_src = strip_comments(io.open(os.path.join(sec, "00b_abstract_en.tex"),
                                    encoding="utf-8").read())
    ko = [plain(x) for x in items_after(ko_src, r"\begin{highlights}", r"\end{highlights}")]
    en = [plain(x) for x in items_after(en_src, r"\textbf{Highlights}")]

    if not ko or not en:
        print(f"[!!] 항목을 찾지 못했다 — 국문 {len(ko)}개 · 영문 {len(en)}개")
        return 1
    if len(ko) != len(en):
        print(f"[주의] 국문 {len(ko)}개 · 영문 {len(en)}개 — 개수가 다르다")

    # ── ① 평문 ──────────────────────────────────────────────────────────────
    txt_body = ("Highlights\n\n"
                + "".join(f"- {x}\n" for x in en)
                + "\n\n연구 하이라이트 (국문 원고 기준 — 참고용)\n\n"
                + "".join(f"- {x}\n" for x in ko)
                + "\n※ 이 파일은 tools/make_highlights.py 가 원고에서 생성한다. "
                  "직접 고치지 말 것 — 다음 실행에서 덮어쓴다.\n")
    good = emit(os.path.join(a.paper, "highlights.txt"), txt_body, a.check)

    # ── ② 독립 문서 ─────────────────────────────────────────────────────────
    def esc(s: str) -> str:
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    tex_body = (
        "%==============================================================================\n"
        "%  Highlights — Elsevier **별도 제출 파일**\n"
        "%\n"
        "%  원고(PDF 앞쪽 별쪽)와 별개로 투고 시스템에 올리는 항목이다.\n"
        "%  ★ 손으로 고치지 말 것 — tools/make_highlights.py 가 원고에서 생성한다.\n"
        "%    원고를 고친 뒤 그 스크립트를 다시 돌리면 여기도 따라온다.\n"
        "%  컴파일: xelatex highlights.tex\n"
        "%==============================================================================\n"
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[a4paper, margin=25mm]{geometry}\n"
        "\\usepackage{kotex}\n"
        "\\usepackage{fontspec}\n"
        "\\setmainfont{TeX Gyre Termes}[Ligatures=TeX]\n"
        "\\setmainhangulfont{HCR Batang}[AutoFakeBold=2.0, AutoFakeSlant=0.2]\n"
        "\\usepackage{enumitem}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n\n"
        "\\noindent{\\Large\\bfseries Highlights}\n\n"
        "\\vspace{4pt}\n"
        "\\begin{itemize}[leftmargin=1.2em, itemsep=3pt, topsep=4pt]\n"
        + "".join(f"\\item {esc(x)}\n" for x in en) +
        "\\end{itemize}\n\n"
        "\\vspace{18pt}\n"
        "\\noindent{\\large\\bfseries 연구 하이라이트}\\;{\\small (국문 원고 기준 --- 참고용)}\n\n"
        "\\vspace{4pt}\n"
        "\\begin{itemize}[leftmargin=1.2em, itemsep=3pt, topsep=4pt]\n"
        + "".join(f"\\item {esc(x)}\n" for x in ko) +
        "\\end{itemize}\n\n"
        "\\end{document}\n")
    good = emit(os.path.join(a.paper, "highlights.tex"), tex_body, a.check) and good

    # 길이를 알린다 --- 상한 자체는 미확인 기준이므로 판정하지 않고 값만 보인다.
    print("\n항목 길이(공백 포함):")
    for tag, xs in (("영문", en), ("국문", ko)):
        print(f"  {tag}: {[len(x) for x in xs]}")
    print("  ※ 흔히 인용되는 85자 상한은 로컬 Elsevier 배포물에서 확인되지 않는다 "
          "— EAAI Guide for Authors 로 확인할 것.")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
