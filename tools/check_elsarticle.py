# -*- coding: utf-8 -*-
"""elsarticle(Elsevier) 조판 규약 기계 검사.

"형식이 어느 한 구석도 달라선 안 된다" 는 요구는 눈으로 지킬 수 없다. 규약은 대부분
**보이지 않는 곳**에 있기 때문이다 — \\address 대신 \\affiliation 키-값을 썼는지,
highlights 가 진짜 환경인지 minipage 흉내인지, keyword 를 \\sep 로 끊었는지,
선언문 제목이 Elsevier 가 찾는 영문 문구 그대로인지.

근거: c:/texlive/2026/texmf-dist/doc/latex/elsarticle/elsarticle-template-num.tex
      c:/texlive/2026/texmf-dist/doc/latex/elsarticle/elsdoc.pdf (Elsevier 배포 매뉴얼)

사용:
    python tools/check_elsarticle.py
    python tools/check_elsarticle.py --tex 논문초안/USV_swarm_defense_EAAI.tex

종료 코드 0 = 전부 통과. 하나라도 실패하면 1.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

#: Elsevier 가 제목으로 블록을 인식하는 네 절. **문구가 정확히 이것이어야 한다.**
#  순서도 규정을 따른다(본문 → 아래 넷 → 참고문헌).
DECLARATIONS = (
    "CRediT authorship contribution statement",
    "Declaration of competing interest",
    "Declaration of generative AI and AI-assisted technologies in the writing process",
    "Data availability",
)

#: frontmatter 안에서 이 순서로 나와야 한다(공식 템플릿의 배열).
FRONT_ORDER = (r"\title", r"\author", r"\affiliation",
               "begin{abstract}", "begin{highlights}", "begin{keyword}")

_FAILED: list[str] = []


def ok(m: str) -> None:
    print(f"  [ok] {m}")


def bad(m: str) -> None:
    _FAILED.append(m)
    print(f"  [!!] {m}")


def read(p: str) -> str:
    return io.open(p, encoding="utf-8").read()


def strip_comments(s: str) -> str:
    out = []
    for ln in s.splitlines():
        m = re.search(r"(?<!\\)%", ln)
        out.append(ln if m is None else ln[: m.start()])
    return "\n".join(out)


def expand(tex_path: str) -> tuple[str, str]:
    """메인 파일과, \\input 을 한 번 펼친 전체 본문을 돌려준다."""
    root = os.path.dirname(tex_path)
    main = strip_comments(read(tex_path))
    full = []
    for m in re.finditer(r"\\input\{([^}]+)\}", main):
        p = os.path.join(root, m.group(1))
        if not p.endswith(".tex"):
            p += ".tex"
        if os.path.exists(p):
            full.append(strip_comments(read(p)))
    return main, "\n".join(full)


# ══════════════════════════════════════════════════════════════════════════
def check_class(main: str) -> None:
    print("클래스")
    m = re.search(r"\\documentclass\[([^\]]*)\]\{elsarticle\}", main)
    if not m:
        bad("\\documentclass{elsarticle} 가 없다 — 자작 조판이면 규약 밖이다")
        return
    opts = [o.strip() for o in m.group(1).split(",")]
    ok(f"elsarticle · 옵션 {opts}")
    known = {"preprint", "review", "final", "1p", "3p", "5p", "twocolumn",
             "authoryear", "number", "times", "doubleblind", "longtitle",
             "10pt", "11pt", "12pt", "oneside", "twoside", "onecolumn", "sort&compress"}
    for o in opts:
        if o not in known:
            bad(f"모르는 클래스 옵션: {o} (공식 매뉴얼에 없는 옵션이다)")
    if "\\journal{" not in main:
        bad("\\journal{...} 선언이 없다")
    else:
        j = re.search(r"\\journal\{([^}]*)\}", main)
        ok(f"\\journal = {j.group(1) if j else '?'}")


def check_frontmatter(main: str, full: str) -> None:
    print("frontmatter")
    i, j = main.find(r"\begin{frontmatter}"), main.find(r"\end{frontmatter}")
    if i < 0 or j < 0:
        bad("frontmatter 환경이 없다")
        return
    # frontmatter 안에서 \input 된 조각들을 제자리에 펼친다.
    block = main[i:j]
    root = os.path.dirname(check_frontmatter.tex)
    def _inline(mm):
        p = os.path.join(root, mm.group(1))
        if not p.endswith(".tex"):
            p += ".tex"
        return strip_comments(read(p)) if os.path.exists(p) else ""
    block = re.sub(r"\\input\{([^}]+)\}", _inline, block)

    pos, last, okorder = {}, -1, True
    for tok in FRONT_ORDER:
        k = block.find(tok)
        pos[tok] = k
        if k < 0:
            bad(f"frontmatter 에 {tok} 가 없다")
            okorder = False
        elif k < last:
            bad(f"frontmatter 순서 어긋남: {tok} 가 앞 항목보다 먼저 나온다")
            okorder = False
        else:
            last = k
    if okorder:
        ok("title → author → affiliation → abstract → highlights → keyword 순서")

    if r"\address" in block:
        bad("구식 \\address 를 쓰고 있다 — \\affiliation{organization=...} 키-값이어야 한다")
    elif "organization=" in block:
        ok("\\affiliation 키-값 형식")
    else:
        bad("\\affiliation 에 organization= 키가 없다")

    for tok, why in ((r"\ead{", "교신저자 이메일(\\ead)"),
                     (r"\cortext[", "교신 표시(\\cortext)"),
                     (r"\corref{", "교신 저자 지정(\\corref)")):
        (ok if tok in block else bad)(f"{why} {'있음' if tok in block else '없음'}")

    if r"\sep" in block:
        ok("keyword 를 \\sep 로 구분")
    else:
        bad("keyword 를 \\sep 로 구분하지 않았다(쉼표는 규약 밖이다)")

    if "minipage" in block:
        bad("frontmatter 안에 minipage 가 있다 — highlights/abstract 를 흉내낸 흔적이다")


def check_declarations(main: str, full: str) -> None:
    print("선언문")
    body = main + "\n" + full
    where = []
    for t in DECLARATIONS:
        k = body.find("\\section*{" + t + "}")
        if k < 0:
            bad(f"절 제목이 규정 문구와 다르거나 없다: {t}")
        else:
            where.append((k, t))
            ok(t)
    if len(where) == len(DECLARATIONS):
        if [t for _, t in sorted(where)] == list(DECLARATIONS):
            ok("규정 순서")
        else:
            bad(f"선언문 순서 어긋남: {[t for _, t in sorted(where)]}")

    # ★ 공식 템플릿(elsarticle-template-num.tex 말미)은 \appendix 블록을
    #   \begin{thebibliography} **앞**에 둔다. 앞서 이 검사가 반대 순서를 통과시켜
    #   이탈을 놓쳤다 — 기준을 템플릿에 맞춘다.
    #   부록은 선택 요소다 — 2026-09-11 부록 도판이 본문으로 흡수되어 파일이 없어졌으므로,
    #   배선이 없으면 순서 검사에서 제외한다(주석 처리된 \input 은 strip 되어 안 잡힌다).
    want = ("09_conclusion", "10_declarations", "99_appendix", "refs")
    order = [main.find(f"\\input{{sections/{n}}}") for n in want]
    if order[2] == -1:
        want = tuple(n for n in want if n != "99_appendix")
        order = [o for o in order if o != -1] if -1 not in order[:2] + order[3:] else order
    if -1 in order:
        bad(f"본문 배선을 찾지 못했다: {dict(zip(want, order))}")
    elif order == sorted(order):
        ok("배치: " + " → ".join({"09_conclusion": "결론", "10_declarations": "선언문",
                                 "99_appendix": "부록", "refs": "참고문헌"}[n] for n in want)
           + " (템플릿 순서)")
    else:
        bad(f"배치 순서 어긋남 — 템플릿은 부록이 참고문헌 앞이다: {dict(zip(want, order))}")

    # elsarticle 에 없는 요소가 본문에 섞이지 않았는가
    if r"\section*{English Abstract}" in main + "\n" + full:
        bad("본문 뒤 제2 초록(English Abstract)은 elsarticle 에 없는 요소다")
    else:
        ok("elsarticle 에 없는 제2 초록 없음")


def check_ref_order(tex_path: str, main: str) -> None:
    """번호 스타일이면 최초 인용 순, 저자-연도면 알파벳 순이어야 한다.

    수동 thebibliography 는 \\bibitem 의 물리적 순서가 곧 번호다. 이 검사가 없어서
    번호 스타일인데 알파벳순으로 실린 목록(30개 전부 어긋남)을 놓쳤다.
    """
    print("참고문헌 순서")
    sec = os.path.join(os.path.dirname(tex_path), "sections")
    refs = os.path.join(sec, "refs.tex")
    if not os.path.exists(refs):
        bad("refs.tex 가 없다")
        return
    keys = re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", read(refs))

    authoryear = "authoryear" in (re.search(r"\\documentclass\[([^\]]*)\]", main) or
                                  type("", (), {"group": lambda *_: ""})()).group(1)
    if authoryear:
        want, why = sorted(keys, key=str.lower), "저자-연도 — 알파벳 순"
    else:
        order, seen = [], set()
        names = ["01_intro", "02_related", "03_problem", "04_method", "05_experiment",
                 "06_results", "07_discussion", "08_limitations", "09_conclusion",
                 "10_declarations", "99_appendix"]
        for n in names:
            f = os.path.join(sec, n + ".tex")
            if not os.path.exists(f):
                continue
            for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", strip_comments(read(f))):
                for k in (x.strip() for x in m.group(1).split(",")):
                    if k and k not in seen:
                        seen.add(k)
                        order.append(k)
        rank = {k: i for i, k in enumerate(order)}
        want, why = sorted(keys, key=lambda k: rank.get(k, 10**6)), "번호 — 최초 인용 순"

    if keys == want:
        ok(f"{why} ({len(keys)}개)")
    else:
        n = sum(1 for x, y in zip(keys, want) if x != y)
        bad(f"참고문헌 순서 어긋남 — {len(keys)}개 중 {n}개 ({why} 이어야 한다). "
            f"고치려면: python tools/sort_refs.py --order "
            f"{'alpha' if authoryear else 'cite'}")


def check_envs(full: str) -> None:
    print("환경")
    if r"\begin{highlights}" in full:
        n = len(re.findall(r"\\item", full.split(r"\begin{highlights}")[1]
                           .split(r"\end{highlights}")[0]))
        # 항목 수 상한은 로컬 Elsevier 배포물에 규정이 없다(통념 3~5). 개수는 알리기만 한다.
        ok(f"highlights 환경 · 항목 {n}개")
    else:
        bad("highlights 환경을 쓰지 않았다")
    (ok if r"\begin{keyword}" in full else bad)(
        "keyword 환경 " + ("있음" if r"\begin{keyword}" in full else "없음"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex", default="논문초안/USV_swarm_defense_EAAI.tex")
    a = ap.parse_args()
    if not os.path.exists(a.tex):
        print(f"원고를 찾지 못했다: {a.tex}")
        return 1
    check_frontmatter.tex = a.tex
    m, f = expand(a.tex)
    check_class(m)
    check_frontmatter(m, f)
    check_envs(m + "\n" + f)
    check_declarations(m, f)
    check_ref_order(a.tex, m)
    print()
    if _FAILED:
        print(f"형식 이탈 {len(_FAILED)}건")
        for x in _FAILED:
            print(f"  - {x}")
        return 1
    print("elsarticle 규약 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
