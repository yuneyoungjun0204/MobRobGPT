# -*- coding: utf-8 -*-
"""국방발표 판 LaTeX 원고 → 대학원생 국방학술대회 양식 한글(hwp) 파일.

  python -m tools.tex2hwp.convert            # 논문초안/USV_swarm_defense_국방발표.hwp (+ .pdf 미리보기)

흐름
  1) flatten: \\input 을 펼친 한 파일 → pandoc JSON AST
  2) 번호: LaTeX 컴파일 결과(.aux)의 그림·표·식·절 번호와 참고문헌 번호를 그대로 쓴다
     (먼저 xelatex 으로 국방발표 판을 컴파일해 두어야 한다)
  3) front: 양식 hwp → HWPX 로 저장해 1쪽 머리(제목·저자·영문 초록·Key Words)와 소속 표기를
     바꾸고 예시 본문을 지운 기본 문서를 만든다
  4) COM: 기본 문서를 한글로 열어 본문을 양식의 문단 스타일로 이어 쓴다
     - 장/절/항 제목: 스타일 11/12/13, 본문: 1, 독립 수식: 15(오른쪽 탭 번호),
       그림·표 제목: 16(한 줄)/17(두 줄 이상), 참고문헌: 18
     - 그림·표 제목과 표 안의 글은 양식대로 영어(en.py)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from . import en, eqn, front, refs
from .common import FIGDIR, PAPER, TEMPLATE
from .flatten import flatten
from .hwp import Hwp

MASTER = "USV_swarm_defense_국방발표"
PANDOC = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Pandoc", "pandoc.exe")
TEXT_W = 150.0     # 본문 폭(mm): B5 182 - 좌우 여백 14.5×2 = 153, 약간의 여유
COL_W = 72.0       # 2단 원고의 단 폭 그림을 1단 판면에 옮길 때의 폭(mm)
WIDE_K = 0.70      # 판면 폭 그림(figure*)은 1단 판면에서 이 비율로 줄인다(분량 절약)

# 양식 스타일 번호(논문작성 형식.hwp)
ST_NORMAL, ST_BODY, ST_CH, ST_SEC, ST_SUBSEC = 0, 1, 11, 12, 13
ST_EQ, ST_CAP1, ST_CAP2, ST_REF = 15, 16, 17, 18
# 양식 스타일별 줄간격(%) --- header.xml 의 문단 모양에서 읽은 값. 캡션 스타일만 100 %.
STYLE_LS = {ST_NORMAL: 148, ST_BODY: 148, ST_CH: 148, ST_SEC: 148, ST_SUBSEC: 148,
            ST_EQ: 148, ST_CAP1: 100, ST_CAP2: 100, ST_REF: 148}

LOG: list[str] = []


# ── 번호 ──────────────────────────────────────────────────────────────────────
def load_aux(path: str):
    labels, cites = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", line)
        if m:
            labels[m.group(1)] = m.group(2)
        m = re.match(r"\\bibcite\{([^}]*)\}\{\{(\d+)\}", line)
        if m:
            cites[m.group(1)] = int(m.group(2))
    return labels, cites


# ── AST 도우미 ────────────────────────────────────────────────────────────────
def plain(x) -> str:
    """캡션·표 칸을 '$수식$' 이 섞인 문자열로. 참조는 {라벨} 자리표시."""
    if isinstance(x, list):
        return "".join(plain(i) for i in x)
    if not isinstance(x, dict):
        return ""
    t = x.get("t")
    if t == "Str":
        return x["c"].replace("\u00a0", " ")
    if t in ("Space", "SoftBreak", "LineBreak"):
        return " "
    if t == "Math":
        return "$" + x["c"][1] + "$"
    if t == "Code":
        return x["c"][1]
    if t == "Link":
        ref = _ref_target(x)
        return "{" + ref + "}" if ref else plain(x["c"][1])
    if t in ("Plain", "Para"):
        return plain(x["c"])
    if "c" in x:
        return plain(x["c"])
    return ""


def _ref_target(link) -> str | None:
    attrs = dict(link["c"][0][2])
    return attrs.get("reference")


class Writer:
    def __init__(self, hwp: Hwp, labels: dict, cites: dict):
        self.h = hwp
        self.labels = labels
        self.cites = cites
        self.eq_no = 0
        self.tab_labels = [k for k, _ in sorted(((k, int(v)) for k, v in labels.items()
                                                 if k.startswith("tab:") and v.isdigit()),
                                                key=lambda kv: kv[1])]
        self.tab_i = 0
        self.first_header_done = False
        self.sec = [0, 0, 0]
        self.refs_path = None

    # ── 글자 수준 ────────────────────────────────────────────────────────────
    def num(self, label: str) -> str:
        if label not in self.labels:
            LOG.append(f"unknown label {label}")
            return "??"
        return self.labels[label]

    def cite_text(self, keys: list[str]) -> str:
        nums = sorted(self.cites.get(k.strip(), 0) for k in keys)
        for k in keys:
            if k.strip() not in self.cites:
                LOG.append(f"unknown cite {k}")
        return "[" + ", ".join(str(n) for n in nums) + "]"

    def math_inline(self, tex: str, base: int = 900) -> None:
        self.h.equation(eqn.convert(tex), base)

    def mixed(self, s: str, base: int = 900, bold: bool = False) -> None:
        """'$수식$' 과 '{라벨}' 이 섞인 문자열을 입력한다."""
        for part in re.split(r"(\$[^$]*\$)", s):
            if not part:
                continue
            if part.startswith("$") and part.endswith("$") and len(part) > 1:
                # 밑글자 없는 위첨자 표식($^{*}$, $^{\dagger}$)은 수식으로 넣으면 비어 보인다
                m = re.fullmatch(r"\^\{?(\\dagger|\\ast|\*)\}?", part[1:-1].strip())
                if m:
                    self.h.char_shape(super=1)
                    self.h.text("†" if "dagger" in m.group(1) else "*")
                    self.h.char_shape(super=0)
                    continue
                self.math_inline(part[1:-1], base)
            else:
                part = re.sub(r"\{([a-z]+:[^}]+)\}",
                              lambda m: ("(" + self.num(m.group(1)) + ")") if m.group(1).startswith("eq:")
                              else self.num(m.group(1)), part)
                self.h.text(part)

    def inlines(self, xs: list, base: int = 900) -> None:
        buf: list[str] = []

        def flush():
            if buf:
                self.h.text("".join(buf))
                buf.clear()

        for x in xs:
            t = x["t"]
            if t == "Str":
                buf.append(x["c"].replace("\u00a0", " "))
            elif t in ("Space", "SoftBreak"):
                buf.append(" ")
            elif t == "LineBreak":
                flush()
                self.h.run("BreakLine")
            elif t == "Math":
                flush()
                if x["c"][0]["t"] == "DisplayMath":
                    raise RuntimeError("display math must be split before inlines()")
                self.math_inline(x["c"][1], base)
            elif t in ("Emph", "Strong", "Superscript", "Subscript"):
                flush()
                key = {"Emph": "italic", "Strong": "bold", "Superscript": "super", "Subscript": "sub"}[t]
                self.h.char_shape(**{key: 1})
                self.inlines(x["c"], base)
                self.h.char_shape(**{key: 0})
            elif t == "Code":
                buf.append(x["c"][1])
            elif t == "Link":
                ref = _ref_target(x)
                if ref:
                    rtype = dict(x["c"][0][2]).get("reference-type", "ref")
                    n = self.num(ref)
                    buf.append(f"({n})" if rtype == "eqref" else n)
                else:
                    flush()
                    self.inlines(x["c"][1], base)
            elif t == "Cite":
                keys = [c["citationId"] for c in x["c"][0]]
                buf.append(self.cite_text(keys))
            elif t == "RawInline":
                m = re.match(r"\\cite[pt]?\{([^}]*)\}", x["c"][1].replace("\n", " "))
                if m:
                    buf.append(self.cite_text(m.group(1).split(",")))
                else:
                    LOG.append(f"raw inline dropped: {x['c'][1][:60]}")
            elif t == "Quoted":
                q = ("\u2018", "\u2019") if x["c"][0]["t"] == "SingleQuote" else ("\u201c", "\u201d")
                buf.append(q[0])
                flush()
                self.inlines(x["c"][1], base)
                buf.append(q[1])
            elif t in ("Span", "SmallCaps", "Underline", "Strikeout"):
                flush()
                self.inlines(x["c"][1] if t == "Span" else x["c"], base)
            elif t == "Note":
                LOG.append("footnote dropped")
            else:
                LOG.append(f"inline {t} dropped")
        flush()

    # ── 문단 수준 ────────────────────────────────────────────────────────────
    def new_para(self, style: int) -> None:
        self.h.para()
        self.h.style(style)
        # 스타일을 적용해도 앞 문단에 직접 준 문단 모양(줄간격·다음 문단과 함께)은 새 문단으로
        # 이어진다. 양식 스타일의 값으로 되돌린다.
        self.h.para_shape(KeepWithNext=0, LineSpacing=STYLE_LS.get(style, 148))

    def blank(self, keep: bool = False, half: bool = False) -> None:
        self.new_para(ST_NORMAL)
        if keep:
            self.h.para_shape(KeepWithNext=1)
        if half:                            # 그림·표 둘레 빈 줄은 반 줄(분량 절약)
            # 글자 크기를 줄이면 다음 문단까지 이어지므로 줄간격만 줄인다
            self.h.para_shape(LineSpacing=50)

    def paragraph(self, xs: list, style: int = ST_BODY) -> None:
        """디스플레이 수식에서 문단을 나눈다. 수식 뒤 이어지는 글은 들여쓰기 없이(스타일 0)."""
        chunk: list = []
        cur_style = style
        for x in xs + [None]:
            if x is None or (x["t"] == "Math" and x["c"][0]["t"] == "DisplayMath"):
                # 앞뒤 공백만 남은 조각은 버린다
                while chunk and chunk[0]["t"] in ("Space", "SoftBreak"):
                    chunk.pop(0)
                while chunk and chunk[-1]["t"] in ("Space", "SoftBreak"):
                    chunk.pop()
                if chunk:
                    self.new_para(cur_style)
                    self.inlines(chunk)
                chunk = []
                if x is not None:
                    self.display_math(x["c"][1])
                    cur_style = ST_NORMAL
            else:
                chunk.append(x)

    def display_math(self, tex: str) -> None:
        self.eq_no += 1
        lab = eqn.label_of(tex)
        if lab and self.labels.get(lab) != str(self.eq_no):
            LOG.append(f"equation number mismatch {lab}: aux {self.labels.get(lab)} vs {self.eq_no}")
        self.blank()
        self.new_para(ST_EQ)
        self.h.equation(eqn.convert(tex), 900)
        self.h.tab()
        self.h.text(f"({self.eq_no})")
        self.blank()

    def header(self, level: int, attr, xs: list) -> None:
        title = plain(xs).strip()
        label = attr[0]
        unnumbered = "unnumbered" in attr[1]
        if unnumbered:
            numtxt = ""
        else:
            self.sec[level - 1] += 1
            for k in range(level, 3):
                self.sec[k] = 0
            numtxt = ".".join(str(n) for n in self.sec[:level])
            if label in self.labels and self.labels[label] != numtxt:
                LOG.append(f"section number mismatch {label}: aux {self.labels[label]} vs {numtxt}")
        if level == 1 and not self.first_header_done:
            self.first_header_done = True     # '1. 서 론' 은 기본 문서에 이미 있다
            return
        if level == 1:
            if re.fullmatch(r"[가-힣]{2}", title):
                title = title[0] + "  " + title[1]
            self.blank()
            self.new_para(ST_CH)
            self.h.para_shape(KeepWithNext=1)
            self.h.text(f"{numtxt}. {title}" if numtxt else title)
            self.blank()
        else:
            self.blank()
            self.new_para(ST_SEC if level == 2 else ST_SUBSEC)
            self.h.para_shape(KeepWithNext=1)
            self.h.text(f"{numtxt} {title}" if numtxt else title)

    # ── 그림 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _fig_width(attrs: dict) -> float:
        w = attrs.get("width", "\\columnwidth")
        m = re.match(r"([\d.]+)%", w)
        if m:
            return TEXT_W * WIDE_K * float(m.group(1)) / 100
        m = re.match(r"([\d.]*)\\(textwidth|columnwidth|linewidth)", w)
        if m:
            k = float(m.group(1) or 1)
            return (TEXT_W * WIDE_K if m.group(2) == "textwidth" else COL_W) * k
        return COL_W

    def caption(self, prefix: str, text: str, keep_next: bool = False) -> None:
        style = ST_CAP1 if len(prefix) + len(re.sub(r"\$[^$]*\$", "xxxx", text)) < 95 else ST_CAP2
        self.new_para(style)
        if keep_next:                       # 표 제목은 표와 같은 쪽에
            self.h.para_shape(KeepWithNext=1)
        self.h.char_shape(bold=1)
        self.h.text(prefix)
        self.h.char_shape(bold=0)
        self.h.text(" ")
        self.mixed(text)

    def figure(self, x) -> None:
        attr, _cap, body = x["c"]
        label = attr[0]
        imgs = []

        def find(y):
            if isinstance(y, dict):
                if y.get("t") == "Image":
                    imgs.append(y)
                for v in y.values():
                    find(v)
            elif isinstance(y, list):
                for v in y:
                    find(v)
        find(body)
        self.blank(keep=True, half=True)
        for img in imgs:
            src = img["c"][2][0]
            path = os.path.join(FIGDIR, src)
            if not os.path.exists(path):
                LOG.append(f"missing image {src}")
                continue
            self.new_para(ST_CAP1)
            self.h.para_shape(KeepWithNext=1)     # 그림과 그 아래 제목을 같은 쪽에
            self.h.picture(path, min(self._fig_width(dict(img["c"][0][2])), TEXT_W))
        cap = en.FIG_CAP.get(label)
        if cap is None:
            LOG.append(f"figure caption not translated: {label}")
            cap = plain(_cap[1])
        self.caption(f"Fig. {self.num(label)}", cap)
        self.blank(half=True)

    # ── 표 ───────────────────────────────────────────────────────────────────
    def _cell_text(self, cell) -> str:
        s = plain(cell[4]).strip()
        s = re.sub(r"\s+", " ", s)
        if re.search(r"[가-힣]", s):
            if s in en.CELL:
                return en.CELL[s]
            LOG.append(f"cell not translated: {s}")
        return s

    def table(self, x, label: str) -> None:
        _attr, _cap, colspecs, thead, tbodies, tfoot = x["c"]
        ncol = len(colspecs)
        rows = list(thead[1])
        for tb in tbodies:
            rows += list(tb[2]) + list(tb[3])
        rows += list(tfoot[1])
        nhead = len(thead[1])
        # 행 병합(rowspan)을 격자로 펼친다
        grid: list[list] = []
        merges = []
        occupied: dict[tuple[int, int], bool] = {}
        for r, row in enumerate(rows):
            grid.append([""] * ncol)
            c = 0
            for cell in row[1]:
                while occupied.get((r, c)):
                    c += 1
                rs, cs = cell[2], cell[3]
                grid[r][c] = self._cell_text(cell)
                for dr in range(rs):
                    for dc in range(cs):
                        occupied[(r + dr, c + dc)] = True
                if rs > 1 or cs > 1:
                    merges.append((r, c, rs, cs))
                c += cs
        nrow = len(grid)
        # 열 폭: 칸 글자 수에 비례(수식은 명령어를 뺀 글자 수로 셈)
        def wlen(s: str) -> float:
            s2 = re.sub(r"\$([^$]*)\$",
                        # 수식은 첨자·괄호 때문에 같은 글자 수의 본문보다 넓다
                        lambda m: "x" * max(1, int(1.5 * len(re.sub(r"\\[a-zA-Z]+|[{}_^\\ ]", "", m.group(1))))), s)
            return max(3.0, min(len(s2), 60))
        weights = [max(wlen(grid[r][c]) for r in range(nrow)) for c in range(ncol)]
        natural = sum(weights) * 2.1 + 4 * ncol    # 8pt 기준 글자당 약 2 mm --- 좁으면 행이 줄바꿈되어 표가 길어진다
        total = min(TEXT_W - 6, max(natural * 1.15, 70.0))   # 여유를 두어 칸 안 줄바꿈을 줄인다
        if total > 0.8 * (TEXT_W - 6):     # 판면 폭에 가까운 표는 판면 폭을 다 쓴다
            total = TEXT_W - 6
        # 한글 표의 열 폭은 셀 안 여백(좌우 1.8 mm)을 따로 더한다 --- 내용 폭만 나눠 준다
        content = total - 3.6 * ncol
        widths = [content * w / sum(weights) for w in weights]
        aligns = []
        for spec in colspecs:
            a = spec[0]["t"]
            aligns.append({"AlignLeft": 1, "AlignRight": 2, "AlignCenter": 3}.get(a, 3))

        cap = en.TAB_CAP.get(label)
        if cap is None:
            LOG.append(f"table caption not translated: {label}")
            cap = plain(_cap[1])
        self.blank(keep=True, half=True)
        self.caption(f"Table {self.num(label)}", cap, keep_next=True)
        self.new_para(ST_CAP1)

        H = self.h.h
        tc = H.HParameterSet.HTableCreation
        H.HAction.GetDefault("TableCreate", tc.HSet)
        tc.Rows, tc.Cols = nrow, ncol
        tc.WidthType, tc.HeightType = 2, 0
        tc.WidthValue = H.MiliToHwpUnit(total)
        tc.CreateItemArray("ColWidth", ncol)
        for i, w in enumerate(widths):
            tc.ColWidth.SetItem(i, H.MiliToHwpUnit(w))
        tc.TableProperties.TreatAsChar = 1
        tc.TableProperties.Width = H.MiliToHwpUnit(total)
        H.HAction.Execute("TableCreate", tc.HSet)
        # 생성 인자의 TreatAsChar 가 무시되는 경우가 있다 --- 표 개체에 직접 건다
        tbl = H.ParentCtrl
        if tbl is not None:
            pset = tbl.Properties
            pset.SetItem("TreatAsChar", 1)
            tbl.Properties = pset

        for r in range(nrow):
            for c in range(ncol):
                self.h.para_shape(AlignType=3 if r < nhead else aligns[c], LineSpacing=120,
                                  Indentation=0, LeftMargin=0)
                self.h.char_shape(height=800, bold=1 if r < nhead else 0)
                if grid[r][c]:
                    self.mixed(grid[r][c], base=800)
                if not (r == nrow - 1 and c == ncol - 1):
                    self.h.run("TableRightCell")
        # 병합은 뒤에서부터(앞 칸의 순회 순서가 바뀌지 않게)
        for (r, c, rs, cs) in sorted(merges, reverse=True):
            self.h.run("TableColBegin")
            self.h.run("TableColPageUp")
            for _ in range(r * ncol + c):
                self.h.run("TableRightCell")
            self.h.run("TableCellBlock")
            self.h.run("TableCellBlockExtend")
            for _ in range(rs - 1):
                self.h.run("TableLowerCell")
            for _ in range(cs - 1):
                self.h.run("TableRightCell")
            self.h.run("TableMergeCell")
            self.h.run("Cancel")
        self.h.run("Cancel")
        H.Run("CloseEx")
        self.h.doc_end()
        note = en.NOTES.get(label)
        if note:
            self.new_para(ST_NORMAL)
            self.h.char_shape(height=800)
            self.mixed(note, base=800)
            self.h.char_shape(height=900)
        self.blank(half=True)

    # ── 블록 순회 ────────────────────────────────────────────────────────────
    def blocks(self, bs: list) -> None:
        # 제목 바로 뒤의 그림은 다음 문단 뒤로 보낸다. LaTeX 에서는 떠다니는 개체라 문제가 없지만
        # 글자처럼 놓으면 제목만 쪽 끝에 남고 그림이 다음 쪽으로 밀린다.
        bs = list(bs)
        for i in range(len(bs) - 2):
            if bs[i]["t"] == "Header" and bs[i + 1]["t"] == "Figure" and bs[i + 2]["t"] == "Para":
                bs[i + 1], bs[i + 2] = bs[i + 2], bs[i + 1]
        for b in bs:
            t = b["t"]
            if t == "Header":
                self.header(b["c"][0], b["c"][1], b["c"][2])
            elif t in ("Para", "Plain"):
                if plain(b["c"]).strip():
                    self.paragraph(b["c"])
            elif t == "Figure":
                self.figure(b)
            elif t == "Table":
                label = self.tab_labels[self.tab_i] if self.tab_i < len(self.tab_labels) else "tab:?"
                self.tab_i += 1
                self.table(b, label)
            elif t == "Div":
                classes = b["c"][0][1]
                if "thebibliography" in classes:
                    self.bibliography()
                elif "minipage" in classes:
                    pass                       # 표 주석은 en.NOTES 의 영문으로 대신한다
                else:
                    self.blocks(b["c"][1])
            elif t == "BulletList":
                for item in b["c"]:
                    for blk in item:
                        if blk["t"] in ("Plain", "Para"):
                            self.new_para(ST_BODY)
                            self.h.text("○ ")
                            self.inlines(blk["c"])
                        else:
                            self.blocks([blk])
            elif t == "OrderedList":
                for k, item in enumerate(b["c"][1], 1):
                    for blk in item:
                        self.new_para(ST_BODY)
                        self.h.text(f"({k}) ")
                        self.inlines(blk.get("c", []))
            elif t == "BlockQuote":
                self.blocks(b["c"])
            elif t == "CodeBlock":
                self.blank()
                for line in b["c"][1].splitlines():
                    self.new_para(ST_NORMAL)
                    self.h.text(line)
                self.blank()
            elif t in ("HorizontalRule", "RawBlock"):
                pass
            else:
                LOG.append(f"block {t} dropped")

    def bibliography(self) -> None:
        self.blank()
        self.new_para(ST_CH)
        self.h.text("참고문헌")
        self.blank()
        for k, (key, text) in enumerate(refs.load(self.refs_path), 1):
            if self.cites.get(key) != k:
                LOG.append(f"reference order mismatch {key}")
            self.new_para(ST_REF)
            self.h.text(f"[{k}] ")
            for s, it in refs.format_entry(text):
                if it:
                    self.h.char_shape(italic=1)
                    self.h.text(s)
                    self.h.char_shape(italic=0)
                else:
                    self.h.text(s)


def main(master: str = MASTER) -> None:
    out_name = master
    work = tempfile.mkdtemp(prefix="tex2hwp_")
    flat = os.path.join(work, "flat.tex")
    open(flat, "w", encoding="utf-8").write(flatten(master))
    # 참고문헌 파일은 마스터가 \input 하는 것(sections_def/refs, sections_kdu/refs ...)
    msrc = open(os.path.join(PAPER, master + ".tex"), encoding="utf-8").read()
    refs_rel = re.search(r"^\\input\{(sections_\w+)/refs\}", msrc, re.M).group(1)
    ast_path = os.path.join(work, "ast.json")
    subprocess.run([PANDOC, "-f", "latex", "-t", "json", flat, "-o", ast_path], check=True)
    ast = json.load(open(ast_path, encoding="utf-8"))
    labels, cites = load_aux(os.path.join(PAPER, master + ".aux"))

    h = Hwp(visible=False)
    h.h.SetMessageBoxMode(0x00020021)
    try:
        tpl_hwpx = os.path.join(work, "template.hwpx")
        h.open(TEMPLATE, "HWP")
        h.save_as(tpl_hwpx, "HWPX")
        base = os.path.join(work, "base.hwpx")
        front.build(tpl_hwpx, base, master)
        h.open(base, "HWPX")
        h.doc_end()
        w = Writer(h, labels, cites)
        w.refs_path = os.path.join(PAPER, refs_rel, "refs.tex")
        # 첫 블록 전에 커서는 빈 본문 문단에 있다 --- 첫 문단은 그 자리에 쓴다
        h.run("DeleteBack")       # 빈 문단 하나를 지워 new_para() 가 그 자리를 다시 만들게 한다
        w.blocks(ast["blocks"])
        out = os.path.join(PAPER, out_name + ".hwp")
        h.save_as(out, "HWP")
        h.save_as(os.path.join(PAPER, out_name + "_hwp미리보기.pdf"), "PDF")
        print("[ok]", out)
    finally:
        h.quit()
    if eqn.UNKNOWN:
        LOG.append("unknown latex commands: " + ", ".join(sorted(eqn.UNKNOWN)))
    log_path = os.path.join(PAPER, out_name + "_hwp변환로그.txt")
    open(log_path, "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    print(f"[log] {len(LOG)} lines -> {log_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
