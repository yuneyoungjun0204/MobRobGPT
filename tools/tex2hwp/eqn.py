# -*- coding: utf-8 -*-
"""LaTeX 수식 → 한글 수식 편집기 스크립트.

원고가 쓰는 범위(분수·근호·첨자·그리스 문자·cases·aligned·크기 조절 괄호·노름 등)만 다룬다.
한글 2022 에서 렌더링을 확인한 키워드만 쓴다 --- pm 은 인식되지 않아 +- 로, \\mathcal 은 대응
글꼴이 없어 일반 글자로 낸다. 모르는 명령은 백슬래시만 떼어 내고 UNKNOWN 에 남긴다.
"""
from __future__ import annotations

import re

UNKNOWN: set[str] = set()

SYM = {
    # 관계·연산
    "le": "<=", "leq": "<=", "ge": ">=", "geq": ">=", "ne": "!=", "neq": "!=",
    "in": "in", "notin": "notin", "to": "->", "rightarrow": "->", "leftarrow": "<-",
    "Rightarrow": "=>", "times": "times", "cdot": "cdot", "pm": "+-", "mp": "-+",
    "approx": "approx", "sim": "sim", "propto": "propto", "equiv": "==", "infty": "inf",
    "partial": "partial", "nabla": "nabla", "dagger": "dagger", "circ": "circ",
    "cdots": "cdots", "ldots": "ldots", "dots": "ldots", "uparrow": "uparrow",
    "downarrow": "downarrow", "forall": "forall", "exists": "exists", "mid": "|",
    "prime": "prime", "ast": "*", "star": "*", "setminus": "-", "cup": "cup", "cap": "cap",
    "subset": "subset", "subseteq": "subseteq", "emptyset": "emptyset",
    # 큰 연산자·함수
    "sum": "sum", "prod": "prod", "int": "int", "min": "min", "max": "max", "log": "log",
    "exp": "exp", "lim": "lim", "sin": "sin", "cos": "cos", "tan": "tan", "arg": 'rm{"arg"}',
    "argmax": 'rm{"argmax"}', "argmin": 'rm{"argmin"}',
    # 구분자(단독)
    "lVert": "LEFT ‖", "rVert": "RIGHT ‖", "lvert": "LEFT |", "rvert": "RIGHT |",
    "langle": "LEFT langle", "rangle": "RIGHT rangle", "lfloor": "LEFT lfloor", "rfloor": "RIGHT rfloor",
    "lceil": "LEFT lceil", "rceil": "RIGHT rceil",
    "|": "‖", "{": "lbrace", "}": "rbrace", "%": "%", "&": "&", "#": "#", "_": "_",
    # 간격
    ",": "`", ";": "~", ":": "~", "!": "", " ": "~", "quad": "~~", "qquad": "~~~~",
}
GREEK = set("""alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi
omicron pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega ell
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega""".split())
ACCENT = {"hat": "hat", "widehat": "hat", "tilde": "tilde", "widetilde": "tilde", "bar": "bar",
          "overline": "bar", "vec": "vec", "dot": "dot", "ddot": "ddot"}
SIZE = {"big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr"}
DELIM = {"(": "(", ")": ")", "[": "[", "]": "]", "|": "|", ".": "",
         "\\{": "lbrace", "\\}": "rbrace", "\\|": "‖", "\\lVert": "‖", "\\rVert": "‖",
         "\\lvert": "|", "\\rvert": "|", "\\langle": "langle", "\\rangle": "rangle",
         "\\lfloor": "lfloor", "\\rfloor": "rfloor", "\\lceil": "lceil", "\\rceil": "rceil"}

_TOK = re.compile(r"\\[a-zA-Z]+\*?|\\.|[{}^_&]|\d+(?:\.\d+)?|\s+|.", re.S)


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall(s)]


class _P:
    def __init__(self, toks: list[str]):
        self.t = toks
        self.i = 0

    def peek(self):
        while self.i < len(self.t) and self.t[self.i].isspace():
            self.i += 1
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def group_raw(self) -> list[str]:
        """다음 인자를 토큰 목록으로(중괄호면 안쪽, 아니면 토큰 하나)."""
        tok = self.next()
        if tok != "{":
            return [tok] if tok is not None else []
        depth, out = 1, []
        while self.i < len(self.t):
            tok = self.t[self.i]
            self.i += 1
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(tok)
        return out

    def raw_text(self) -> str:
        return "".join(self.group_raw())

    def opt(self):
        """[...] 선택 인자를 건너뛰고 내용을 돌려준다."""
        if self.peek() == "[":
            self.next()
            out = []
            while self.i < len(self.t) and self.t[self.i] != "]":
                out.append(self.t[self.i])
                self.i += 1
            self.i += 1
            return "".join(out)
        return None


def _conv_tokens(toks: list[str]) -> str:
    return _expr(_P(toks), stop=None)


def _text(s: str) -> str:
    s = s.replace("\\,", " ").replace("~", " ").replace("\\_", "_").replace("\\%", "%")
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("{", "").replace("}", "")
    # \text{if } 처럼 글 앞뒤의 공백은 의미가 있다 --- 한글 수식은 일반 공백을 무시하므로 ~ 로 남긴다
    pre = "~" if s[:1].isspace() else ""
    post = "~" if s[-1:].isspace() else ""
    s = s.strip()
    return f'{pre}rm{{"{s}"}}{post}' if s else ""


def _env(p: _P, name: str) -> str:
    # \begin{name} ... \end{name} 의 내용을 토큰으로 모은다
    depth, body = 1, []
    while p.i < len(p.t):
        tok = p.t[p.i]
        if tok == "\\begin":
            depth += 1
        if tok == "\\end":
            depth -= 1
            if depth == 0:
                p.i += 1
                p.group_raw()
                break
        body.append(tok)
        p.i += 1
    rows = []
    cur: list[str] = []
    j = 0
    while j < len(body):
        tok = body[j]
        if tok == "\\\\":
            rows.append(cur)
            cur = []
            # \\[3pt] 같은 간격 인자 건너뛰기
            k = j + 1
            while k < len(body) and body[k].isspace():
                k += 1
            if k < len(body) and body[k] == "[":
                while k < len(body) and body[k] != "]":
                    k += 1
                j = k
        else:
            cur.append(tok)
        j += 1
    if any(not t.isspace() for t in cur):
        rows.append(cur)
    inner = " # ".join(_conv_tokens(r) for r in rows)
    if name == "cases":
        return f"cases{{{inner}}}"
    if name in ("aligned", "align", "align*", "split", "gathered", "array", "matrix", "pmatrix"):
        return f"eqalign{{{inner}}}"
    return inner


def _atom(p: _P) -> str | None:
    tok = p.next()
    if tok is None:
        return None
    if tok == "{":
        p.i -= 1
        inner = _conv_tokens(p.group_raw())
        return "{" + inner + "}"
    if tok in ("^", "_"):
        arg = _atom(p) or ""
        if not arg.startswith("{"):
            arg = "{" + arg + "}"
        return f"{tok}{arg}"
    if tok == "&":
        return "&"
    if not tok.startswith("\\"):
        if tok in ("'",):
            return "prime"
        return tok
    name = tok[1:]
    if name in ("frac", "dfrac", "tfrac"):
        a = _conv_tokens(p.group_raw())
        b = _conv_tokens(p.group_raw())
        return f"{{{a}}} over {{{b}}}"
    if name == "sqrt":
        n = p.opt()
        a = _conv_tokens(p.group_raw())
        return f"root {{{n}}} of {{{a}}}" if n else f"sqrt {{{a}}}"
    if name in ("mathrm", "textrm", "operatorname", "text", "textnormal", "mathsf", "texttt", "textit"):
        raw = p.raw_text()
        if re.fullmatch(r"[A-Za-z0-9.\-/ ]+", raw) is None and "\\" in raw:
            return "rm{" + _conv_tokens(tokenize(raw)) + "}"
        return _text(raw)
    if name in ("mathbf", "boldsymbol", "bm", "textbf"):
        return "bold{" + _conv_tokens(p.group_raw()) + "}"
    if name in ("mathcal", "mathscr", "mathbb", "mathit"):
        return "{" + _conv_tokens(p.group_raw()) + "}"
    if name in ACCENT:
        return f"{ACCENT[name]}{{{_conv_tokens(p.group_raw())}}}"
    if name == "ensuremath":
        return "{" + _conv_tokens(p.group_raw()) + "}"
    if name == "unit":
        return "` " + _text(p.raw_text())
    if name in ("left", "right"):
        d = p.next()
        if d == "\\":  # 안전장치
            d = p.next()
        sym = DELIM.get(d, d.lstrip("\\") if d else "")
        if not sym:
            return ""
        return ("LEFT " if name == "left" else "RIGHT ") + sym
    if name in SIZE:
        d = p.next()
        return DELIM.get(d, d.lstrip("\\") if d else "") or ""
    if name == "begin":
        env = p.raw_text()
        if env in ("equation", "equation*"):
            return ""
        return _env(p, env)
    if name == "end":
        p.group_raw()
        return ""
    if name in ("label", "nonumber", "notag", "tag"):
        if name in ("label", "tag"):
            p.group_raw()
        return ""
    if name in ("displaystyle", "textstyle", "scriptstyle", "limits", "nolimits", "small",
                "footnotesize", "normalsize"):
        return ""
    if name in GREEK:
        return name
    if name in SYM:
        return SYM[name]
    if len(name) == 1:  # \, \; \{ 등
        return SYM.get(name, name)
    UNKNOWN.add(name)
    return name


def _expr(p: _P, stop) -> str:
    out = []
    while True:
        tok = p.peek()
        if tok is None or tok == stop:
            break
        a = _atom(p)
        if a is None:
            break
        if a:
            out.append(a)
    s = " ".join(out)
    s = re.sub(r"\s+([_^])", r" \1", s)
    return s


def convert(latex: str) -> str:
    s = latex.strip()
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    return _conv_tokens(tokenize(s)).strip()


def label_of(latex: str) -> str | None:
    m = re.search(r"\\label\{([^}]*)\}", latex)
    return m.group(1) if m else None
