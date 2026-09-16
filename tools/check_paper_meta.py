# -*- coding: utf-8 -*-
"""원고 전면부·기호 규약 기계 검사.

눈으로 세지 않는다. 하이라이트 글자 수, 국문/영문 초록의 수치 일치, 기호 중복 사용,
기호표에만 있고 본문에 없는 유령 기호 — 전부 사람이 놓치는 종류다.

사용:
    python tools/check_paper_meta.py
    python tools/check_paper_meta.py --paper 논문초안        # 다른 원고 폴더

종료 코드 0 = 전부 통과. 하나라도 실패하면 1 (CI 에 걸 수 있다).
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys

#: ⚠ 아래 세 값은 **로컬 Elsevier 배포물에서 확인하지 못했다.**
#    elsdoc.tex / elsarticle.cls / 템플릿 3종 어디에도 항목 수·글자 수·단어 수 규정이 없다.
#    웹 Guide for Authors 에서 온 통념이므로 EAAI 페이지에서 직접 확인할 것.
#    확인 못 한 기준으로 빌드를 막지 않는다 --- 위반해도 warn 으로만 찍고 종료코드에는
#    넣지 않는다(아래 warn 참조).
#  덧붙여 85자 규칙은 **영문 기준**이다. 한글은 글자당 정보량이 달라 같은 잣대로 재는
#  것 자체가 의미가 약하다. 실제로 걸릴 쪽은 영문 하이라이트다.
HL_MAX_CHARS = 85
HL_MIN_ITEMS, HL_MAX_ITEMS = 3, 5
ABSTRACT_MAX_WORDS = 250

_FAILED: list[str] = []
_WARNED: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def bad(msg: str) -> None:
    _FAILED.append(msg)
    print(f"  [!!] {msg}")


def warn(msg: str) -> None:
    """확인하지 못한 규칙 위반. 알리되 종료코드에는 넣지 않는다."""
    _WARNED.append(msg)
    print(f"  [주의] {msg}")


def read(path: str) -> str:
    return io.open(path, encoding="utf-8").read()


def strip_comments(s: str) -> str:
    """줄 첫 %(주석)를 지운다. 이스케이프된 \\% 는 남긴다."""
    out = []
    for ln in s.splitlines():
        m = re.search(r"(?<!\\)%", ln)
        out.append(ln if m is None else ln[: m.start()])
    return "\n".join(out)


def items_of(tex: str, after: str) -> list[str]:
    """`after` 뒤의 \\item 본문들.

    ★ elsarticle 의 highlights 환경은 itemize 를 끼지 않고 \\item 을 바로 쓴다.
      itemize 만 찾으면 0개로 읽힌다(그렇게 한 번 놓쳤다). 두 형태를 모두 받는다.
    """
    i = tex.find(after)
    if i < 0:
        return []
    j = tex.find(r"\begin{itemize}", i)
    k = tex.find(r"\end{itemize}", j) if j >= 0 else -1
    if j >= 0 and k > j:
        body = tex[j + len(r"\begin{itemize}"):k]
    else:
        # 환경 이름을 after 에서 뽑아 그 끝까지를 본문으로 삼는다.
        m = re.match(r"\\begin\{([a-zA-Z*]+)\}", after)
        end = ("\\end{" + m.group(1) + "}") if m else None
        e = tex.find(end, i) if end else -1
        body = tex[i + len(after):e if e > 0 else len(tex)]
    parts = [p.strip() for p in body.split(r"\item")[1:]]
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


#: 인자까지 통째로 버리는 조판 명령. \vspace{6pt} 의 6 을 수치로 세면 초록이 어긋난 것처럼 보인다.
_DROP_ARG = ("vspace", "hspace", "label", "cite", "citep", "citet", "ref", "unit",
             "includegraphics", "setlength", "addcontentsline", "linespread")


def strip_tex(s: str) -> str:
    """조판 명령을 걷어내고 사람이 읽는 텍스트만 남긴다."""
    s = re.sub(r"\\S\s*\\?ref\{[^}]*\}", " ", s)
    s = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", " ", s)
    s = re.sub(r"\\(" + "|".join(_DROP_ARG) + r")\*?\s*(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)      # 남은 명령 이름 제거 (인자 내용은 남긴다)
    return s.replace("{", " ").replace("}", " ")


def numbers_in(s: str) -> list[str]:
    """수치 토큰.

    ★ 대시를 먼저 지운다. LaTeX 의 --- (em dash) 뒤에 숫자가 오면 "---8" 이 "-8" 로
      읽혀 국문의 "8" 과 어긋난 것처럼 보인다. 하이픈 복합어(15-channel)도 같은 문제다.
      부호는 $-0.192$ 처럼 대시가 아닌 자리에서만 살아남아야 한다.
    """
    s = strip_tex(s)
    s = s.replace("---", " ").replace("--", " ")
    s = re.sub(r"(?<=[A-Za-z가-힣])-(?=\d)", " ", s)     # 15-channel 류
    return re.findall(r"[-+\u2212]?\d+(?:\.\d+)?", s.replace("\u2212", "-"))


# ══════════════════════════════════════════════════════════════════════════
def check_highlights(sec: str) -> None:
    print("Highlights")
    ko = read(os.path.join(sec, "00c_highlights.tex"))
    en = read(os.path.join(sec, "00b_abstract_en.tex"))
    # 국문판은 elsarticle 의 highlights 환경, 영문판은 English Abstract 절 안의 목록이다.
    for lang, tex, anchor in (("국문", ko, r"\begin{highlights}"),
                              ("영문", en, r"\textbf{Highlights}")):
        its = items_of(strip_comments(tex), anchor)
        if not (HL_MIN_ITEMS <= len(its) <= HL_MAX_ITEMS):
            warn(f"{lang} 하이라이트 항목 {len(its)}개 "
                 f"(통념 {HL_MIN_ITEMS}~{HL_MAX_ITEMS} — 미확인 기준)")
        else:
            ok(f"{lang} 하이라이트 {len(its)}개")
        for n, it in enumerate(its, 1):
            plain = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", it)
            plain = re.sub(r"\$[^$]*\$", "x", plain).replace("\\", "")
            if len(plain) > HL_MAX_CHARS:
                warn(f"{lang} 하이라이트 {n}번 {len(plain)}자 "
                     f"(통념 {HL_MAX_CHARS}자 — 미확인 기준, 영문 기준값이다): {plain[:40]}...")
            else:
                ok(f"{lang} 하이라이트 {n}번 {len(plain)}자")


def check_abstracts(sec: str) -> None:
    print("초록")
    ko = strip_comments(read(os.path.join(sec, "00_abstract.tex")))
    en = strip_comments(read(os.path.join(sec, "00b_abstract_en.tex")))
    # 영문 초록 본문만 — 제목·주제어·Highlights 는 초록 단어 수에 들어가지 않는다.
    en_body = en.split(r"\textbf{Highlights}")[0].split(r"\textbf{Keywords}")[0]
    # 영문 제목의 마지막 문구로 자른다. 제목을 바꾸면 이 앵커도 바꿔야 한다.
    en_body = en_body.split("for Defense Against Suicide USV Swarms}")[-1]

    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", strip_tex(en_body)))
    if words > ABSTRACT_MAX_WORDS:
        warn(f"영문 초록 {words} 단어 (통념 {ABSTRACT_MAX_WORDS} — 미확인 기준)")
    else:
        ok(f"영문 초록 {words} 단어")

    a, b = sorted(numbers_in(ko)), sorted(numbers_in(en_body))
    if a != b:
        only_ko = sorted(set(a) - set(b))
        only_en = sorted(set(b) - set(a))
        bad(f"초록 수치 불일치 — 국문에만 {only_ko} / 영문에만 {only_en}")
    else:
        ok(f"초록 수치 {len(a)}개 전부 일치")

    # 국문 주제어는 elsarticle 의 keyword 환경(별도 파일)에 있고 \\sep 로 끊는다.
    kw_path = os.path.join(sec, "00d_keywords.tex")
    if os.path.exists(kw_path):
        kw = strip_comments(read(kw_path))
        i, j = kw.find(r"\begin{keyword}"), kw.find(r"\end{keyword}")
        body = kw[i + len(r"\begin{keyword}"):j] if i >= 0 and j > i else ""
        n = len([x for x in body.split(r"\sep") if len(x.strip()) > 1])
        (ok if n >= 5 else bad)(f"국문 주제어 {n}개" + ("" if n >= 5 else " (5개 이상 권장)"))
    else:
        bad("국문 주제어 파일(00d_keywords.tex)이 없다")

    i = en.find("Keywords")
    if i < 0:
        bad("영문 주제어 없음")
    else:
        tail = en[i:i + 400].split("\\vspace")[0]
        n = len([x for x in re.split(r"[;,]", tail.split(":", 1)[-1]) if len(x.strip()) > 2])
        (ok if n >= 5 else bad)(f"영문 주제어 {n}개" + ("" if n >= 5 else " (5개 이상 권장)"))


def check_declarations(sec: str) -> None:
    print("선언문")
    tex = strip_comments(read(os.path.join(sec, "10_declarations.tex")))
    # ★ 제목 문구·순서·배치는 tools/check_elsarticle.py 가 규정과 대조한다.
    #   여기서는 네 절의 존재만 본다 — 같은 검사를 두 곳에 두면 한쪽만 고치게 된다.
    need = ("CRediT authorship contribution statement",
            "Declaration of competing interest",
            "Declaration of generative AI",
            "Data availability")
    for t in need:
        if t in tex:
            ok(f"{t} 절 있음")
        else:
            bad(f"{t} 절 없음")
    n = tex.count("투고 전")
    if n:
        print(f"  [주의] 저자 확인이 필요한 표시 {n}건 — 투고 전에 반드시 채울 것")


def check_symbols(sec: str) -> None:
    """같은 문자를 두 뜻으로 쓰지 않는지, 기호표에 유령 기호가 없는지."""
    print("기호")
    tabs = os.path.join(os.path.dirname(sec), "tables")
    body = "\n".join(strip_comments(read(f))
                     for f in sorted(glob.glob(os.path.join(sec, "*.tex"))
                                     + glob.glob(os.path.join(tabs, "*.tex")))
                     if "03_problem" not in f)
    nom = strip_comments(read(os.path.join(sec, "03_problem.tex")))
    i = nom.find(r"\label{tab:nomenclature}")
    j = nom.find(r"\end{tabular}", i)
    if i < 0 or j < 0:
        bad("기호표를 찾지 못했다")
        return
    table = nom[i:j]

    syms = re.findall(r"\$(\\?[A-Za-z\\{}^_\\]+?)\$\s*&", table)
    ghosts = []
    for sym in syms:
        core = sym.strip()
        if core and core not in body and core not in nom:
            ghosts.append(core)
    if ghosts:
        bad(f"기호표에만 있고 본문에 없는 기호: {ghosts}")
    else:
        ok(f"기호표 {len(syms)}개 전부 본문에 등장")

    # 무리 수 / 경유점 수 / 후보 수가 같은 문자를 쓰지 않는지
    full = body + nom
    # 표(자동 생성분 포함)와 본문 어투를 모두 잡아야 한다. 앞서 표의
    # "적 무리 최대 수 $K$" 를 놓쳐 Table 1 에 충돌이 남아 있었다.
    for pat, why in ((r"\$K\$?\s*개의?\s*무리", "K 를 무리 수로 사용"),
                     (r"무리[^$\n]{0,12}\$K\$", "K 를 무리 수로 사용(표)"),
                     (r"후보\s*\$K", "K 를 그룹 후보 수로 사용"),
                     (r"최대\s*\$K\s*=\s*4", "K 를 무리 수로 사용")):
        hits = re.findall(pat, full)
        if hits:
            bad(f"기호 충돌 — {why} ({len(hits)}건)")
    if not _FAILED or all("기호 충돌" not in m for m in _FAILED):
        ok("K/C/G 충돌 없음")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", default="논문초안", help="원고 폴더")
    ap.add_argument("--sections", default="sections",
                    help="섹션 하위 폴더 이름 (실그림 판은 sections_real)")
    args = ap.parse_args()
    sec = os.path.join(args.paper, args.sections)
    if not os.path.isdir(sec):
        print(f"섹션 폴더가 없다: {sec}")
        return 1

    check_highlights(sec)
    check_abstracts(sec)
    check_declarations(sec)
    check_symbols(sec)

    print()
    if _WARNED:
        print(f"확인 못 한 기준에 대한 주의 {len(_WARNED)}건 "
              f"(EAAI Guide for Authors 로 확인할 것)")
        for m in _WARNED:
            print(f"  - {m}")
        print()
    if _FAILED:
        print(f"실패 {len(_FAILED)}건")
        for m in _FAILED:
            print(f"  - {m}")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
