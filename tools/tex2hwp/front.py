# -*- coding: utf-8 -*-
"""양식 hwp 의 1쪽 머리(쪽 고정 표: 제목·저자·이메일·영문 초록·Key Words)와 소속 표기 표를
원고 내용으로 바꾸고, 본문 예시를 지운 기본 문서(HWPX)를 만든다.

양식의 두 표는 쪽 기준으로 위치가 고정된 개체라서, 글자만 바꾸면 위치·테두리·간격이 그대로 남는다.
소속 표기 표는 원래 본문 2.1절 문단에 붙어 있으므로 첫 문단으로 옮긴 뒤 본문을 지운다.
"""
from __future__ import annotations

import html
import os
import re
import shutil
import zipfile

# ── 머리 내용 ─────────────────────────────────────────────────────────────────
TITLE_KO = ["군집 자폭 수상 드론 방어를 위한",
            "LLM 기반 지휘체계와 U-Net 점수맵 기반",
            "다중 에이전트 협력 강화학습"]
TITLE_EN = ["LLM-Based Command Architecture and U-Net Score-Map",
            "Multi-Agent Cooperative Reinforcement Learning",
            "for Defense Against Suicide USV Swarms"]
# (이름, 소속 번호). 영문 이름은 원고에 없어 비워 둔다 --- 양식 (4)항에 따라 채워 넣을 것.
AUTHORS = [("윤 영 준", 1), ("방 현 태", 1), ("전 유 라", 2), ("김 덕 환", 2), ("윤 원 근", 1)]
AFFIL = ["*1 충남대학교", "*2 한국과학기술원"]
EMAIL = ("(0204jun@o.cnu.ac.kr, htbang@g.cnu.ac.kr, youraj@kaist.ac.kr, dnb06116@kaist.ac.kr, "
         "wkyoun@cnu.ac.kr)")
# 국방발표 판 국문 초록(sections_def/00_abstract.tex)의 영문. 수치는 국문 초록과 같다.
ABSTRACT_EN = (
    "Swarms of low-cost suicide unmanned surface vehicles (USVs) evade conventional "
    "counters: electronic warfare cannot reach wire-guided targets, and interceptors cost more than their "
    "targets and saturate. We propose a two-layer defence architecture in which a few defenders, slower than "
    "the attackers, pre-emptively block the mothership's approach routes with nets. In the "
    "strategic layer a large language model (LLM) commander assigns each defender an enemy cluster; in the "
    "maneuver layer each defender points at a pixel of a score map that a U-Net produces over a rasterized "
    "battlefield grid to place its net. Both layers decide on the same period but differ sharply in "
    "response latency, so only the slow strategic layer is decoupled asynchronously. Pixel pointing confines "
    "the action space to the grid, imposes safety and mission constraints by masking and is "
    "invariant to swarm size, and training extends critic-free group-relative policy "
    "optimization to multiple agents. From 720 seed-paired episodes (8 conditions, 3 formations, 30 seeds) "
    "we separated the contributions of the two layers in a 2x2 design. The maneuver layer alone "
    "raised the capture rate from 0.844 to 0.886, and the full system cut nets per capture "
    "by 45 % and total turning by 47 % at the same defense level; the two contributions were additive "
    "rather than synergistic. The commander paid off only for models above a capability threshold, and its "
    "performance was explained not by assignment coverage but by plan stability across replans. In "
    "periodically replanning architectures, consistency over time matters more than local optimality."
)
KEYWORDS = ("무인수상정(Unmanned Surface Vehicle), 군집 방어(Swarm Defense), "
            "다중 에이전트 강화학습(Multi-Agent Reinforcement Learning), 임무배정(Task Assignment), "
            "대규모 언어모델(Large Language Model), 의미론적 분할(Semantic Segmentation), "
            "그룹 상대 정책 최적화(Group Relative Policy Optimization)")
FIRST_HEADING = "1. 서  론"

# 원고(마스터)별 제목. 국방발표 판 제목은 대한조선학회에 제출한 것이라 15쪽판은 따로 둔다.
# 영문 제목은 양식대로 단어 첫 글자만 대문자(4자 이하 관사·전치사·접속사 제외).
TITLES = {
    "USV_swarm_defense_국방발표_15p": (
        ["자폭 무인수상정 군집 방어를 위한",
         "LLM 지휘관과 U-Net 점수맵 기동 정책의 2계층 결합"],
        ["Coupling an LLM Commander and a U-Net Score-Map Policy",
         "for Two-Layer Defense against Suicide USV Swarms"]),
}


def _p(para_pr: int, style: int, runs: list[tuple[int, str]]) -> str:
    body = "".join(f'<hp:run charPrIDRef="{c}"><hp:t>{html.escape(t, quote=False)}</hp:t></hp:run>' if t
                   else f'<hp:run charPrIDRef="{c}"/>' for c, t in runs)
    return (f'<hp:p id="2147483648" paraPrIDRef="{para_pr}" styleIDRef="{style}" pageBreak="0" '
            f'columnBreak="0" merged="0">{body}</hp:p>')


def _title_cell(title_ko: list[str], title_en: list[str]) -> str:
    ps = [_p(18, 0, [(17, "")]), _p(18, 0, [(17, "")])]
    ps += [_p(17, 5, [(18, t)]) for t in title_ko]
    ps.append(_p(11, 0, [(0, "")]))
    ps += [_p(12, 7, [(4, t)]) for t in title_en]
    ps.append(_p(12, 6, [(10, "")]))
    runs: list[tuple[int, str]] = []
    for k, (name, aff) in enumerate(AUTHORS):
        runs.append((12, name))
        runs.append((8, f"*{aff}"))
        if k < len(AUTHORS) - 1:
            runs.append((12, ", "))
    ps.append(_p(12, 6, runs))
    ps.append(_p(11, 0, [(0, "")]))
    ps.append(_p(10, 19, [(6, EMAIL)]))
    ps.append(_p(10, 19, [(6, "")]))
    ps.append(_p(4, 9, [(0, ABSTRACT_EN)]))
    ps.append(_p(11, 0, [(0, "")]))
    ps.append(_p(5, 10, [(6, "Key Words"), (0, ": " + KEYWORDS)]))
    ps.append(_p(11, 0, [(0, "")]))
    return "".join(ps)


def _top_paras(s: str) -> list[str]:
    """최상위 <hp:p> 문단만 잘라 낸다(표 셀 안의 문단은 바깥 문단에 포함된다)."""
    out, depth, start = [], 0, 0
    for m in re.finditer(r"<hp:p[ >]|</hp:p>", s):
        if m.group(0) == "</hp:p>":
            depth -= 1
            if depth == 0:
                out.append(s[start: m.end()])
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return out


def _replace_sublist(tbl: str, new_paras: str) -> str:
    """1x1 표의 셀 subList 안 문단들을 통째로 바꾼다."""
    m = re.search(r"(<hp:subList[^>]*>)(.*)(</hp:subList>)", tbl, re.S)
    return tbl[: m.start(2)] + new_paras + tbl[m.end(2):]


def build(template_hwpx: str, out_hwpx: str, master: str = "") -> None:
    title_ko, title_en = TITLES.get(master, (TITLE_KO, TITLE_EN))
    tmp = out_hwpx + ".d"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(template_hwpx) as z:
        z.extractall(tmp)
        names = z.namelist()
    sec_path = os.path.join(tmp, "Contents", "section0.xml")
    s = open(sec_path, encoding="utf-8").read()
    s = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", s, flags=re.S)

    head_end = s.index("<hp:p ")                       # <hs:sec ...> 머리
    paras = _top_paras(s[head_end:])
    first = paras[0]
    # 소속 표기 표: '*1' 로 시작하는 문단을 담은 쪽 고정 표
    affil_para = next(p for p in paras if "*1 " in p and "<hp:tbl" in p and p is not first)
    i = affil_para.index("<hp:tbl")
    j = affil_para.index("</hp:tbl>") + len("</hp:tbl>")
    affil_tbl = affil_para[i:j]
    aff_paras = "".join(_p(11, 0, [(9, a)]) for a in AFFIL)
    affil_tbl = _replace_sublist(affil_tbl, aff_paras)

    # 제목 표(첫 문단 안의 첫 표) 내용을 바꾸고, 소속 표를 그 뒤에 붙인다
    i = first.index("<hp:tbl")
    j = first.index("</hp:tbl>") + len("</hp:tbl>")
    title_tbl = _replace_sublist(first[i:j], _title_cell(title_ko, title_en))
    first = first[:i] + title_tbl + affil_tbl + first[j:]
    first = re.sub(r"<hp:t>1\.\s*서\s*론</hp:t>", f"<hp:t>{FIRST_HEADING}</hp:t>", first)

    # 첫 장 제목 뒤 빈 줄 하나와, 본문을 이어 쓸 빈 문단 하나만 남긴다
    tail = _p(11, 0, [(0, "")]) + _p(6, 1, [(0, "")])
    s = s[:head_end] + first + tail + "</hs:sec>"
    open(sec_path, "w", encoding="utf-8").write(s)

    # '독립된 수식' 스타일(문단 모양 8)의 탭 정의 3 은 오른쪽 탭이 판면 중간에 있다. 양식 설명대로
    # 식 번호가 줄 끝에 오도록 '문단 오른쪽 끝 자동 탭'이 켜진 탭 정의 2 로 바꾼다.
    hdr_path = os.path.join(tmp, "Contents", "header.xml")
    hdr = open(hdr_path, encoding="utf-8").read()
    hdr = hdr.replace('<hh:paraPr id="8" tabPrIDRef="3"', '<hh:paraPr id="8" tabPrIDRef="2"')
    open(hdr_path, "w", encoding="utf-8").write(hdr)

    if os.path.exists(out_hwpx):
        os.remove(out_hwpx)
    with zipfile.ZipFile(out_hwpx, "w") as z:
        # mimetype 은 압축하지 않고 맨 앞에 둔다(OPC 규칙)
        z.write(os.path.join(tmp, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
        for n in names:
            if n == "mimetype":
                continue
            z.write(os.path.join(tmp, n), n, compress_type=zipfile.ZIP_DEFLATED)
    shutil.rmtree(tmp, ignore_errors=True)
