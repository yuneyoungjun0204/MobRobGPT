# -*- coding: utf-8 -*-
"""논문 제출용 도판 — Elsevier(EAAI) 조판을 전제로 다시 그린 판.

`plots.py` 는 실험을 돌리면서 눈으로 확인하려고 만든 **탐색용** 도판이다. 정보는 맞지만
조판 관습이 논문과 다르다. 이 모듈은 같은 CSV·같은 통계 함수에서 **논문판**만 만든다.
`plots.py` 는 건드리지 않는다(다른 스크립트 30여 개가 의존한다).

탐색용과 무엇이 다른가 — 전부 이유가 있는 차이다
    ① 그림 안 제목을 없앤다. 제목은 캡션의 몫이다(Elsevier artwork 지침). 그림 안 제목은
       조판되면 캡션과 두 번 말하게 되고, 도판을 축소할 때 가장 먼저 못 읽게 된다.
    ② 조건 8개에 색 8개를 주지 않는다. 색 = 지휘관, 무늬 = 기동 계층으로 **두 요인을 따로**
       인코딩한다. 범례가 8줄에서 4+2 줄로 줄고, 2x2 설계가 그림에서 바로 보인다.
    ③ forest 는 막대가 아니라 **점추정 + 신뢰구간**이다. 구간 없는 forest 는 forest 가 아니다.
       (기존 figJ 는 막대에 별표만 붙어 있었다.)
    ④ 이중 y축을 쓰지 않는다. 두 축의 눈금을 어떻게 맞추느냐로 상관이 있어 보이게도 없어
       보이게도 만들 수 있어 심사에서 지적되는 형식이다. 산점도로 바꾼다.
    ⑤ 대응표본 설계에 **비대응 CI 를 겹쳐 그리지 않는다.** 셀별 CI 는 서로 겹치는데 대응차이는
       유의한 상황이 흔해, 그림이 표를 부정하는 것처럼 읽힌다. 불확실성은 대응차이 패널로 뺀다.
    ⑥ 색은 Okabe-Ito(색각 이상 안전) + 흑백 인쇄용 무늬. 얇은 축선(0.7 pt), 위/오른쪽 축 제거.
    ⑦ 최종 배치 크기(1단 88 mm / 2단 180 mm)로 만들고 본문 대비 8 pt 로 맞춘다. 그림을
       LaTeX 에서 늘이거나 줄이면 글자 크기가 본문과 어긋난다.

통계는 새로 만들지 않는다
    전부 `stats.py` 의 대응표본 BCa 부트스트랩을 그대로 쓴다. forest 의 d_z 구간은
    평균차의 BCa CI 를 sd(d) 로 나눈 것이다 — sd 는 재표집하지 않는 상수이므로 이 변환은
    단조이고, 따라서 그림의 구간과 본문 표의 구간이 **같은 검정**을 가리킨다.

사용:
    python -m boatattack_sim.eval.paper_figs --csv results/eval_merged
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from . import paper_names as PN
from . import plots as P
from . import stats as S

# ── 배치 크기 (Elsevier: 1단 90 mm, 1.5단 140 mm, 2단 190 mm) ───────────────
MM = 1 / 25.4
W1, W15, W2 = 88 * MM, 140 * MM, 180 * MM

# ── Okabe-Ito ───────────────────────────────────────────────────────────────
OI = {
    "black":  "#000000", "orange": "#E69F00", "sky":    "#56B4E9",
    "green":  "#009E73", "yellow": "#F0E442", "blue":   "#0072B2",
    "vermil": "#D55E00", "purple": "#CC79A7", "grey":   "#7F7F7F",
}
#: 다중비교 보정 수준 — 지표 10개에 대한 Bonferroni. tools/make_paper_tables.py 와 같은 값을
#  써야 표의 단검(dagger)과 그림의 판정이 일치한다.
BONF_M = 10
BONF_LEVEL = 1.0 - 0.05 / BONF_M

#: 색 = 지휘관 계층. 휴리스틱(=언어모델 없음)은 무채색으로 두어 "기준선"임을 나타낸다.
CMD_COLOR = {
    "": OI["grey"],
    "qwen2.5-7b": OI["vermil"],
    "qwen2.5-14b": OI["orange"],
    "gpt-4o-mini": OI["purple"],
    "gemini-3.5-flash-lite": OI["blue"],
}
CMD_LABEL = {"": "휴리스틱 배정"}
#: 무늬 = 기동 계층. 흑백 인쇄에서 색이 죽어도 이 축은 남는다.
MNV_HATCH = {"heur": "", "unet": "///"}
MNV_LABEL = {"heur": "휴리스틱 기동", "unet": "U-Net 점수맵"}

_INK = "#222222"
_GRID = "#DCDCDC"


def use_paper_style() -> None:
    """rcParams 를 논문판으로 바꾼다. 이 모듈의 함수는 전부 이걸 먼저 부른다."""
    for kf in P._KFONTS:
        if kf in P._avail:
            matplotlib.rcParams["font.family"] = kf
            break
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": _INK,
        "axes.labelcolor": _INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "text.color": _INK,
        "xtick.color": _INK, "ytick.color": _INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "grid.color": _GRID, "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def _ygrid(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def _xgrid(ax):
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def _T(s: str) -> str:
    """폰트에 없을 수 있는 글리프를 치환(plots.safe_text 와 같은 규약)."""
    return P.safe_text(s)


def _M(metric: str, *, unit: bool = True) -> str:
    """지표 표기. **논문 표와 같은 이름**을 쓴다(paper_names 가 정본).

    plots.metric_label 은 탐색용 어휘라 표와 어긋난다 --- "그물 걸림"(표) 대
    "그물 접촉"(탐색용) 처럼. 독자가 표와 그림을 오가며 같은 양인지 확인하게 만들면 안 된다.
    """
    return _T(PN.label(metric, unit=unit))


def save(fig, outdir: str, name: str) -> str:
    """PDF(제출용) + PNG(미리보기) 로 저장하고 PDF 경로를 돌려준다."""
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, name + ".pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(outdir, name + ".png"))
    plt.close(fig)
    return pdf


# ════════════════════════════════════════════════════════════════════════════
# P1. 조건 x 포메이션 — 2x2 설계를 색/무늬로 분리 인코딩
# ════════════════════════════════════════════════════════════════════════════
def _split(cond: str):
    """조건 태그를 (지휘관 백엔드, 기동 계층) 으로 쪼갠다."""
    base = P._base_key(cond)
    mnv = "unet" if base.endswith("_unet") else "heur"
    return P._base_model(P._backend_of(cond)), mnv


def fig_conditions(df, metric: str = "capture_rate", *, seed: int = 0,
                   figsize=(W2, 2.45)):
    """조건 x 포메이션 막대. 색 = 지휘관, 무늬 = 기동 계층.

    기존 figA 는 조건 8개에 색 8개를 배정하고 범례를 4열 x 2행으로 그림 위에 얹어
    도판 높이의 30 %를 범례가 먹었다. 여기서는 요인 두 개를 따로 인코딩해 범례가
    4(지휘관) + 2(기동) 로 줄고, 같은 색 안에서 무늬만 다른 쌍이 곧 U-Net 효과가 된다.
    """
    use_paper_style()
    # 같은 지휘관의 (휴리스틱, U-Net) 쌍이 반드시 이웃하도록 정렬한다. 이 순서가 아니면
    # U-Net 효과가 그림에서 쌍 비교로 읽히지 않는다. 지휘관은 능력 오름차순.
    rank = {m: i for i, m in enumerate(P.MODEL_ORDER)}
    conds = sorted(P._conds(df),
                   key=lambda c: (rank.get(_split(c)[0], -1) if _split(c)[0] else -1,
                                  _split(c)[1] == "unet"))
    forms = [f for f in ("concentrated", "diversionary", "wave") if f in set(df["formation"])]
    groups = forms + ["ALL"]

    fig, ax = plt.subplots(figsize=figsize)
    n = len(conds)
    width = 0.86 / n
    xs = np.arange(len(groups))
    for i, c in enumerate(conds):
        cmd, mnv = _split(c)
        means, los, his = [], [], []
        for g in groups:
            sub = df if g == "ALL" else df[df["formation"] == g]
            piv = S.pivot_by_seed(sub, metric)
            if c not in piv:
                means.append(np.nan); los.append(0.0); his.append(0.0); continue
            est = S.mean_ci(piv[c].to_numpy(), seed=seed)
            means.append(est.mean)
            los.append(0.0 if not np.isfinite(est.lo) else max(0.0, est.mean - est.lo))
            his.append(0.0 if not np.isfinite(est.hi) else max(0.0, est.hi - est.mean))
        off = (i - (n - 1) / 2) * width
        ax.bar(xs + off, means, width * 0.92,
               yerr=np.vstack([los, his]),
               color=CMD_COLOR.get(cmd, OI["grey"]), edgecolor="white",
               linewidth=0.45, hatch=MNV_HATCH[mnv], zorder=3,
               error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6,
                         "ecolor": _INK, "zorder": 4})

    ax.set_xticks(xs)
    ax.set_xticklabels([_T(P.FORM_KO.get(g, g)) for g in groups])
    ax.set_ylabel(_M(metric))
    ax.set_xlabel(_T("적 공격 포메이션"))
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _ygrid(ax)

    cmds = list(dict.fromkeys(_split(c)[0] for c in conds))
    h1 = [Patch(facecolor=CMD_COLOR.get(m, OI["grey"]), edgecolor="white", linewidth=0.45,
                label=_T(CMD_LABEL.get(m) or P.model_label(m))) for m in cmds]
    h2 = [Patch(facecolor="#FFFFFF", edgecolor=_INK, linewidth=0.6, hatch=MNV_HATCH[k],
                label=_T(MNV_LABEL[k])) for k in ("heur", "unet")]
    leg1 = ax.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.0, 1.005),
                     ncol=len(h1), title=_T("지휘관(전략) 계층"), title_fontsize=7)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=h2, loc="lower right", bbox_to_anchor=(1.0, 1.005),
                     ncol=2, title=_T("기동 계층"), title_fontsize=7)
    leg2._legend_box.align = "left"
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P2. 다지표 forest — 점추정 + 신뢰구간 (막대 아님)
# ════════════════════════════════════════════════════════════════════════════
def fig_forest(df, *, treatment: str | None = None, baseline: str = "heur_heur",
               groups=P.METRIC_GROUPS, seed: int = 0, width=W2):
    """제안 조건 vs 기준선의 전 지표 효과크기 forest.

    기존 figJ 의 문제: 이름은 forest 인데 **구간이 없었다**. 채운 막대에 별표만 붙어 있어
    "얼마나 확실한가"가 그림에 없다. 여기서는 점추정 + 95 % 구간을 그린다.

    d_z 의 구간은 어떻게 얻었나
        평균차 d 의 BCa CI 를 sd(d) 로 나눈다. sd 는 재표집하지 않는 상수이므로 이 변환은
        단조이고, 그림의 구간이 0 을 배제하는 것과 본문 표의 평균차 CI 가 0 을 배제하는 것이
        **같은 사건**이 된다. 그림과 표가 다른 검정을 말하는 사고를 막는다.

    부호는 항상 '좋아진 방향이 +' 로 통일한다(낮을수록 좋은 지표는 뒤집는다).
    """
    use_paper_style()
    conds = P._conds(df)
    if treatment is None:
        cand = [c for c in conds if P._base_key(c) == "llm_unet"] or \
               [c for c in conds if c != baseline]
        if not cand:
            raise ValueError("비교할 조건이 없다")
        treatment = cand[0]

    rows = []          # (그룹, 라벨, dz, lo, hi, 유의)
    for g, ms in groups:
        for m in ms:
            if m not in df.columns:
                continue
            piv = S.pivot_by_seed(df, m)
            if treatment not in piv or baseline not in piv:
                continue
            x, y = piv[treatment].to_numpy(), piv[baseline].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            d = x[ok] - y[ok]
            sd = d.std(ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                continue
            est = S.paired_diff(x, y, seed=seed)
            eb = S.paired_diff(x, y, confidence_level=BONF_LEVEL, seed=seed)
            dz, lo, hi = est.mean / sd, est.lo / sd, est.hi / sd
            blo, bhi = eb.lo / sd, eb.hi / sd
            if m in PN.LOWER_BETTER:
                dz, lo, hi = -dz, -hi, -lo
                blo, bhi = -bhi, -blo
            rows.append((g, _M(m, unit=False), dz, lo, hi,
                         bool(est.significant), blo, bhi,
                         bool(eb.lo > 0 or eb.hi < 0)))
    if not rows:
        raise ValueError("forest 에 그릴 지표가 없다")

    h = 0.235 * len(rows) + 0.95
    fig, ax = plt.subplots(figsize=(width, h))
    ys = np.arange(len(rows))[::-1]

    for ref in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(ref, color="#EFEFEF", lw=0.5, zorder=0)
    ax.axvline(0, color=_INK, lw=0.7, zorder=1)

    for y, (g, lab, dz, lo, hi, sig, blo, bhi, bsig) in zip(ys, rows):
        col = OI["blue"] if dz >= 0 else OI["vermil"]
        # 보정 구간을 먼저(옅고 가늘게) — 95 % 구간이 그 위에 얹힌다.
        ax.plot([blo, bhi], [y, y], color=col, lw=0.6, alpha=0.42, zorder=2,
                solid_capstyle="butt")
        for e in (blo, bhi):
            ax.plot([e, e], [y - 0.10, y + 0.10], color=col, lw=0.6, alpha=0.42, zorder=2)
        ax.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=3,
                solid_capstyle="butt")
        for e in (lo, hi):                       # 구간 끝의 세로 마감선
            ax.plot([e, e], [y - 0.16, y + 0.16], color=col, lw=0.8, zorder=3)
        # 속을 채우는 기준은 **보정 구간**이다 — 표의 단검과 같은 판정.
        ax.plot([dz], [y], marker="D", ms=3.6, color=col, zorder=4,
                markerfacecolor=col if bsig else "white",
                markeredgecolor=col, markeredgewidth=0.9)

    # 지표 라벨 + 그룹 구분선
    ax.set_yticks(ys)
    ax.set_yticklabels([_T(r[1]) for r in rows])
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)
    prev = None
    for y, r in zip(ys, rows):
        if prev is not None and r[0] != prev:
            ax.axhline(y + 0.5, color="#E4E4E4", lw=0.5, zorder=0)
        prev = r[0]

    xmax = max(2.2, max(r[7] for r in rows) * 1.06)
    xmin = min(-0.5, min(r[6] for r in rows) * 1.06)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(_T("효과크기 ") + r"$d_z$" + _T("  (+ = 제안이 개선된 방향)"))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _xgrid(ax)
    ax.grid(axis="x", color="#F2F2F2")

    # 오른쪽 수치 열 — 그림만 보고도 값을 인용할 수 있게 한다.
    tx = 1.015
    ax.text(tx, (ys.max() + 0.75 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
            r"$d_z$" + _T(" [95% CI]"), transform=ax.transAxes, fontsize=6.8,
            va="bottom", ha="left", color="#555555")
    for y, (g, lab, dz, lo, hi, sig, blo, bhi, bsig) in zip(ys, rows):
        ax.annotate(f"{dz:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("" if bsig else "  n.s."),
                    xy=(tx, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=6.8,
                    color=_INK if bsig else "#8A8A8A")

    # 왼쪽 그룹 이름 — 지표군을 한 번에 훑게 한다.
    gx = -0.30
    seen = {}
    for y, r in zip(ys, rows):
        seen.setdefault(r[0], []).append(y)
    for g, yy in seen.items():
        ax.annotate(_T(g), xy=(gx, float(np.mean(yy))),
                    xycoords=("axes fraction", "data"), va="center", ha="left",
                    fontsize=6.8, color="#666666", rotation=0)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P3. 2x2 상호작용 — 셀 평균(왼쪽) + 대응 대비(오른쪽)
# ════════════════════════════════════════════════════════════════════════════
def fig_interaction(df, metric: str = "capture_rate", *, backend: str = "",
                    seed: int = 0, figsize=(W2, 2.35)):
    """(a) 셀 평균 프로파일, (b) 대응표본 대비 3개.

    왜 (a) 에 오차막대를 안 그리나
        이 설계는 시드 대응표본이다. 셀별 CI 는 **비대응** 구간이라 시나리오 난이도의
        분산을 그대로 포함해 넓다. 대응차이는 유의한데 셀 CI 는 크게 겹치는 상황이
        흔하고, 그러면 그림이 표를 부정하는 것처럼 읽힌다. 불확실성은 전부 (b) 로 보낸다 —
        (b) 의 구간이 이 논문이 실제로 검정한 양이다.
    """
    use_paper_style()
    suf = f"@{backend}" if backend else ""
    cells = {
        ("heur", "heur"): "heur_heur",
        ("heur", "unet"): "heur_unet",
        ("llm", "heur"): f"llm_heur{suf}",
        ("llm", "unet"): f"llm_unet{suf}",
    }
    piv = S.pivot_by_seed(df, metric)
    miss = [c for c in cells.values() if c not in piv]
    if miss:
        raise ValueError(f"상호작용에 필요한 조건이 없다: {miss}")
    v = {k: piv[c].to_numpy() for k, c in cells.items()}

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.46})

    # ── (a) 프로파일 ────────────────────────────────────────────────────────
    # 선이 둘뿐이라 범례 대신 선 끝에 직접 이름을 붙인다 — 범례 상자가 데이터 위에
    # 얹히는 것을 피하고, 눈이 색을 대조하지 않아도 된다.
    xs = np.array([0.0, 1.0])
    for cmd, col, mk, lab, dy in (
            ("heur", OI["grey"], "o", "휴리스틱 배정", -9),
            # 모델 이름은 캡션이 말한다. 선 옆 라벨이 길면 옆 패널을 침범한다.
            ("llm", CMD_COLOR.get(backend, OI["blue"]), "s", "LLM 지휘관", 6)):
        ys = [float(np.nanmean(v[(cmd, "heur")])), float(np.nanmean(v[(cmd, "unet")]))]
        axa.plot(xs, ys, marker=mk, color=col, lw=1.1, ms=4.2, zorder=3)
        for x, y in zip(xs, ys):
            axa.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, dy), ha="center", va="bottom" if dy > 0 else "top",
                         fontsize=6.5, color=col)
        axa.annotate(_T(lab), (xs[-1], ys[-1]), textcoords="offset points",
                     xytext=(7, 0), ha="left", va="center", fontsize=6.5, color=col)
    axa.set_xticks(xs)
    axa.set_xticklabels([_T("휴리스틱 기동"), _T("U-Net 점수맵")])
    axa.set_xlim(-0.22, 1.18)
    axa.set_ylabel(_M(metric))
    axa.margins(y=0.16)
    _ygrid(axa)
    axa.set_title("(a)", loc="left", fontsize=8, color="#555555")

    # ── (b) 대응 대비 ───────────────────────────────────────────────────────
    contrasts = [
        (_T("U-Net 효과 | 휴리스틱 지휘관"), v[("heur", "unet")], v[("heur", "heur")]),
        (_T("U-Net 효과 | LLM 지휘관"),      v[("llm", "unet")],  v[("llm", "heur")]),
    ]
    labs, ms_, los, his, sigs = [], [], [], [], []
    for lab, a, b in contrasts:
        e = S.paired_diff(a, b, seed=seed)
        labs.append(lab); ms_.append(e.mean); los.append(e.lo); his.append(e.hi)
        sigs.append(bool(e.significant))
    # 상호작용 = 차이의 차이. 같은 시드끼리 빼므로 이것도 대응표본이다.
    e = S.paired_diff(v[("llm", "unet")] - v[("llm", "heur")],
                      v[("heur", "unet")] - v[("heur", "heur")], seed=seed)
    labs.append(_T("상호작용 (차이의 차이)"))
    ms_.append(e.mean); los.append(e.lo); his.append(e.hi); sigs.append(bool(e.significant))

    ys = np.arange(len(labs))[::-1]
    axb.axvline(0, color=_INK, lw=0.7, zorder=1)
    for y, m, lo, hi, sig in zip(ys, ms_, los, his, sigs):
        col = OI["blue"] if sig else "#8A8A8A"
        axb.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=3)
        for x in (lo, hi):
            axb.plot([x, x], [y - 0.13, y + 0.13], color=col, lw=0.8, zorder=3)
        axb.plot([m], [y], marker="D", ms=3.6, color=col, zorder=4,
                 markerfacecolor=col if sig else "white", markeredgewidth=0.9)
        axb.annotate(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]", xy=(1.02, y),
                     xycoords=("axes fraction", "data"), va="center", ha="left",
                     fontsize=6.5, color=_INK if sig else "#8A8A8A")
    # 행 이름표는 y축 바깥이 아니라 **구간선 바로 위 패널 안쪽**에 둔다.
    # 바깥에 두면 긴 한글 이름표가 왼쪽 패널 (a) 영역으로 흘러들어, (a) 의 계열 이름과
    # 붙어 정반대 조건의 부제처럼 읽힌다(실제로 그렇게 읽혔다).
    axb.set_yticks(ys); axb.set_yticklabels([""] * len(labs))
    axb.set_ylim(-0.6, len(labs) - 0.25)
    axb.tick_params(axis="y", length=0)
    axb.spines["left"].set_visible(False)
    for y, lab in zip(ys, labs):
        axb.annotate(lab, xy=(0.012, y + 0.22), xycoords=("axes fraction", "data"),
                     va="bottom", ha="left", fontsize=6.6, color=_INK)
    axb.set_xlabel(_M(metric, unit=False) + _T(" 대응차이 (95% BCa CI)"))
    _xgrid(axb)
    axb.set_title("(b)", loc="left", fontsize=8, color="#555555")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P4. 능력 사다리
# ════════════════════════════════════════════════════════════════════════════
def fig_ladder(df, *, metric: str = "capture_rate", seed: int = 0,
               maneuver: str = "heur", figsize=(W15, 2.55)):
    """지휘관 능력축 x 포메이션. 점선 = 같은 포메이션의 휴리스틱 배정 기준선.

    기존 figM2 대비: matplotlib 기본 tab10 대신 Okabe-Ito, 점마다 부트스트랩 CI 추가,
    기준선은 범례 대신 오른쪽 끝에 직접 이름을 붙인다(색 대조를 눈으로 시키지 않는다).
    """
    use_paper_style()
    ms = P.models_in(df)
    if len(ms) < 2:
        raise ValueError("사다리 그림에는 모델 2종 이상이 필요하다. 있는 것: %s" % ms)
    forms = [f for f in ("concentrated", "diversionary", "wave") if f in set(df["formation"])]
    fcol = {"concentrated": OI["blue"], "diversionary": OI["vermil"], "wave": OI["green"]}
    fmk = {"concentrated": "o", "diversionary": "s", "wave": "^"}

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(ms), dtype=float)
    bases = {}
    for f in forms:
        ys, los, his, base = [], [], [], np.nan
        for m in ms:
            x, b = P._model_capture(df, m, metric, f, maneuver=maneuver)
            if x is None:
                ys.append(np.nan); los.append(0.0); his.append(0.0)
            else:
                est = S.mean_ci(np.asarray(x, float), seed=seed)
                ys.append(est.mean)
                los.append(0.0 if not np.isfinite(est.lo) else max(0.0, est.mean - est.lo))
                his.append(0.0 if not np.isfinite(est.hi) else max(0.0, est.hi - est.mean))
            if b is not None:
                base = float(np.mean(b))
        c = fcol.get(f, OI["grey"])
        ax.errorbar(xs, ys, yerr=np.vstack([los, his]), marker=fmk.get(f, "o"),
                    color=c, lw=1.1, ms=4.0, elinewidth=0.6, capsize=1.5,
                    capthick=0.6, label=_T(P.FORM_KO.get(f, f)), zorder=3)
        if np.isfinite(base):
            ax.axhline(base, ls=(0, (1.6, 1.6)), lw=0.7, color=c, zorder=2)
            bases[f] = base

    # 기준선 이름은 축 **밖** 오른쪽에 붙인다. 축 안에 두면 점선 위에 글자가 얹히고,
    # 기준선끼리 가까우면(양동 0.79 · 파상 0.75) 두 글자가 겹친다 → 겹치면 밀어낸다.
    order = sorted(bases.items(), key=lambda kv: kv[1])
    span = (max(bases.values()) - min(bases.values())) or 1.0
    prev = -1e9
    for f, b in order:
        yy = max(b, prev + 0.055 * span)
        ax.annotate(_T(P.FORM_KO.get(f, f) + " 기준선"),
                    xy=(1.012, yy), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=6.3, color=fcol.get(f, OI["grey"]))
        prev = yy

    ax.set_xticks(xs)
    ax.set_xticklabels([_T(P.model_label(P._base_model(m))) for m in ms])
    ax.set_xlim(-0.25, xs[-1] + 0.25)
    ax.set_xlabel(_T("지휘관 언어모델 능력 (낮음 → 높음)"))
    ax.set_ylabel(_M(metric))
    ax.legend(loc="lower right", ncol=len(forms), fontsize=6.8,
              title=_T("적 포메이션"), title_fontsize=6.8)
    _ygrid(ax)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P5. 계획 품질 -> 성능 (이중 축 제거, 산점도)
# ════════════════════════════════════════════════════════════════════════════
def fig_plan_quality(df, llm_df, *, metric: str = "capture_rate", seed: int = 0,
                     maneuver: str = "heur", figsize=(W2, 2.15)):
    """계획 품질 지표(가로) vs 성능(세로). 모델 하나가 점 하나다.

    기존 figN 은 막대(품질) 위에 붉은 선(성능)을 **이중 y축**으로 겹쳤다. 이중 축은 두 축의
    범위를 어떻게 잡느냐로 상관을 만들어 낼 수 있어 심사에서 지적되는 형식이다. 두 양의
    관계를 보이려는 것이므로 두 양을 그냥 두 축에 놓는다 — 그러면 '커버리지는 성능과 함께
    가지 않고 churn 은 간다'가 축 조작 없이 그대로 보인다.

    ★ 회귀선·상관계수는 얹지 않는다. 모델이 3종이라 n=3 이다.
    """
    use_paper_style()
    if llm_df is None or len(llm_df) == 0:
        raise ValueError("계획 품질 그림에는 llm_metrics 가 필요하다")
    col = "model" if "model" in llm_df.columns else "backend"
    have = set(llm_df[col].astype(str))
    ms = [m for m in P.models_in(df) if m in have]
    if len(ms) < 2:
        raise ValueError("계획품질 그림에는 모델 2종 이상이 필요하다. 있는 것: %s" % ms)

    perf, plo, phi = [], [], []
    for m in ms:
        x, _ = P._model_capture(df, m, metric, None, maneuver=maneuver)
        est = S.mean_ci(np.asarray(x, float), seed=seed) if x is not None else None
        perf.append(est.mean if est else np.nan)
        plo.append(0.0 if not est or not np.isfinite(est.lo) else max(0.0, est.mean - est.lo))
        phi.append(0.0 if not est or not np.isfinite(est.hi) else max(0.0, est.hi - est.mean))

    panels = (("coverage", "커버리지 (배정된 무리 비율)", "높을수록 좋음"),
              ("churn", "churn (재계획마다 표적을 바꾼 배)", "낮을수록 좋음"),
              ("crossings", "경로 교차 (두 배의 요격경로)", "낮을수록 좋음"))
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    tags = "abc"
    for k, (ax, (q, lab, dirn)) in enumerate(zip(axes, panels)):
        qv, qlo, qhi = [], [], []
        for m in ms:
            g = llm_df[llm_df[col].astype(str) == m]
            a = g[q].astype(float).dropna().to_numpy() if q in g.columns else np.array([])
            if a.size == 0:
                qv.append(np.nan); qlo.append(0.0); qhi.append(0.0); continue
            est = S.mean_ci(a, seed=seed)
            qv.append(est.mean)
            qlo.append(0.0 if not np.isfinite(est.lo) else max(0.0, est.mean - est.lo))
            qhi.append(0.0 if not np.isfinite(est.hi) else max(0.0, est.hi - est.mean))
        lo_x, hi_x = np.nanmin(qv), np.nanmax(qv)
        mid = 0.5 * (lo_x + hi_x)
        for i, m in enumerate(ms):
            c = CMD_COLOR.get(P._base_model(m), OI["grey"])
            ax.errorbar([qv[i]], [perf[i]],
                        xerr=[[qlo[i]], [qhi[i]]], yerr=[[plo[i]], [phi[i]]],
                        marker="o", ms=4.2, color=c, elinewidth=0.6,
                        capsize=1.4, capthick=0.6, zorder=3)
            # 오른쪽에 있는 점은 왼쪽으로 라벨을 붙인다 — 안 그러면 축 밖으로 나간다.
            right = qv[i] > mid
            ax.annotate(_T(P.model_label(P._base_model(m))), (qv[i], perf[i]),
                        textcoords="offset points",
                        xytext=(-7 if right else 7, -1.5),
                        ha="right" if right else "left", va="center",
                        fontsize=6.2, color=c)
        ax.set_xlabel(_T(lab) + "\n" + _T("(" + dirn + ")"), fontsize=6.8, linespacing=1.4)
        if k == 0:
            ax.set_ylabel(_M(metric))
        ax.set_ylim(0.55, 1.0)
        ax.margins(x=0.30)
        ax.grid(zorder=0); ax.set_axisbelow(True)
        ax.set_title(f"({tags[k]})", loc="left", fontsize=8, color="#555555")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P6. 학습 곡선 — 논문이 평가하는 그 런 하나만
# ════════════════════════════════════════════════════════════════════════════
def fig_training_curve(run_dir: str, *, figsize=(W2, 3.6)):
    """GRPO 학습 경과. **평가된 런 하나만** 그린다.

    탐색용 그림(fig_traincurve.png)은 두 런을 겹쳐 놓고 내부 실험 이름("dmin 스크리닝 C",
    "cnn_dmin_best", "warm-start 체인")을 그대로 노출했다. 논문에는 최종 구조의 학습 경과만
    싣고, 곡선의 뜻은 캡션이 말한다.

    입력: run_dir/metrics.csv (20 업데이트마다) + run_dir/evals.csv (greedy 평가).
    """
    import pandas as pd
    use_paper_style()
    m = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    e = pd.read_csv(os.path.join(run_dir, "evals.csv"))
    base = float(e["baseline"].iloc[0]) if "baseline" in e else np.nan
    best_i = int(e["eval_cap"].idxmax())

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    (a1, a2), (a3, a4) = axes
    c_train, c_eval, c_best = OI["blue"], OI["blue"], OI["vermil"]

    # (a) 포획률 — 학습 중(옅게) + greedy 평가(표식) + 기준선
    mt = m[m["upd"] > 0]                      # upd=0 은 롤아웃 전 0 --- 곡선을 눌러 버린다
    a1.plot(mt["upd"], mt["cap_rate"], color=c_train, lw=0.8, alpha=0.45,
            label=_T("학습 중 (후보 평균)"))
    a1.plot(e["upd"], e["eval_cap"], color=c_eval, lw=1.1, marker="o", ms=4.5,
            markerfacecolor="white", markeredgewidth=1.0, label=_T("greedy 평가"))
    a1.plot([e["upd"].iloc[best_i]], [e["eval_cap"].iloc[best_i]], marker="*", ms=11,
            color=c_best, linestyle="none", zorder=5,
            label=_T(f"최고점 {e['eval_cap'].iloc[best_i]:.3f} (채택 가중치)"))
    if np.isfinite(base):
        a1.axhline(base, color=OI["grey"], ls=(0, (2.5, 2)), lw=0.9,
                   label=_T(f"휴리스틱 기준선 {base:.3f}"))
    a1.set_ylabel(_T("포획률"))
    a1.set_ylim(0.40, 0.90)
    a1.legend(loc="lower left", fontsize=6.4, ncol=2)
    a1.set_title("(a)", loc="left", fontsize=8, color="#555555")
    _ygrid(a1)

    # (b) 보상 — 후보 평균 vs 후보 최대 (둘의 간격 = 그룹 내 변별력)
    a2.plot(m["upd"], m["R"], color=OI["blue"], lw=1.0, label=_T("후보 평균 R"))
    a2.plot(m["upd"], m["best"], color=OI["blue"], lw=0.8, ls=(0, (3, 2)),
            label=_T("후보 최대 R"))
    a2.fill_between(m["upd"], m["R"], m["best"], color=OI["blue"], alpha=0.10, lw=0)
    a2.set_ylabel(_T("윈도우 보상"))
    a2.legend(loc="lower right", fontsize=6.4)
    a2.set_title("(b)", loc="left", fontsize=8, color="#555555")
    _ygrid(a2)

    # (c) 학습신호 유효율
    a3.plot(m["upd"], m["valid"], color=OI["green"], lw=1.0)
    a3.set_ylim(0, 1.0)
    a3.set_ylabel(_T("학습신호 유효 월드 비율"))
    a3.set_xlabel(_T("누적 업데이트"))
    a3.set_title("(c)", loc="left", fontsize=8, color="#555555")
    _ygrid(a3)

    # (d) 손실
    a4.plot(m["upd"], m["loss"], color=OI["grey"], lw=1.0)
    a4.axhline(0, color=_INK, lw=0.5)
    a4.set_ylabel(_T("손실 (정책경사 + 엔트로피)"))
    a4.set_xlabel(_T("누적 업데이트"))
    a4.set_title("(d)", loc="left", fontsize=8, color="#555555")
    _ygrid(a4)

    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P7. 학습률·엔트로피 스케줄 — 학습 코드가 실제로 쓴 값
# ════════════════════════════════════════════════════════════════════════════
def fig_lr_schedule(run_dir: str | None = None, *, updates=800, lr=1e-4, eta_ratio=0.05,
                    ent0=0.01, ent_end_ratio=0.1, figsize=(W15, 2.1)):
    """단일 주기 코사인 학습률(재시작 없음) + 선형 감쇠 엔트로피 계수.

    값은 grpo_cnn.py 의 CosineAnnealingLR(T_max=updates, eta_min=lr*0.05) 와
    ent_c = ent0*(1-(1-end_ratio)*upd/updates) 를 그대로 재현한다. run_dir 를 주면
    evals.csv 의 greedy 평가 시점을 학습률 곡선 위에 표시한다.
    """
    use_paper_style()
    u = np.arange(0, updates + 1)
    eta_min = lr * eta_ratio
    lr_t = eta_min + (lr - eta_min) * 0.5 * (1.0 + np.cos(np.pi * u / updates))
    ent_t = ent0 * (1.0 - (1.0 - ent_end_ratio) * u / updates)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    a1.plot(u, lr_t * 1e4, color=OI["blue"], lw=1.2)
    a1.axhline(eta_min * 1e4, color=OI["grey"], ls=(0, (2.5, 2)), lw=0.8)
    a1.text(0, eta_min * 1e4 + 0.03, _T(r"$\eta_{\min}=0.05\,\eta_0$"),
            ha="left", va="bottom", fontsize=6.5, color=OI["grey"])
    if run_dir and os.path.exists(os.path.join(run_dir, "evals.csv")):
        import pandas as pd
        e = pd.read_csv(os.path.join(run_dir, "evals.csv"))
        ue = e["upd"].to_numpy()
        le = eta_min + (lr - eta_min) * 0.5 * (1.0 + np.cos(np.pi * ue / updates))
        a1.plot(ue, le * 1e4, "o", ms=4.5, color=OI["blue"], markerfacecolor="white",
                markeredgewidth=1.0, label=_T("greedy 평가 시점"))
        bi = int(e["eval_cap"].idxmax())
        a1.plot([ue[bi]], [le[bi] * 1e4], "*", ms=11, color=OI["vermil"], zorder=5,
                label=_T("채택 가중치"))
        a1.legend(loc="upper right", fontsize=6.4)
    a1.set_ylabel(_T(r"학습률 $\eta_t$ ($\times 10^{-4}$)"))
    a1.set_xlabel(_T("누적 업데이트"))
    a1.set_ylim(0, lr * 1e4 * 1.08)
    a1.set_title("(a)", loc="left", fontsize=8, color="#555555")
    _ygrid(a1)

    a2.plot(u, ent_t * 1e2, color=OI["green"], lw=1.2)
    a2.set_ylabel(_T(r"엔트로피 계수 $\beta_t$ ($\times 10^{-2}$)"))
    a2.set_xlabel(_T("누적 업데이트"))
    a2.set_ylim(0, ent0 * 1e2 * 1.08)
    a2.set_title("(b)", loc="left", fontsize=8, color="#555555")
    _ygrid(a2)

    fig.tight_layout(w_pad=1.8)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# P8. 수렴 진단 — 채택 런과 같은 설정으로 다시 돌린 3 시드의 학습 동역학
# ════════════════════════════════════════════════════════════════════════════
def fig_training_diag(diag_root: str, *, adopted_run: str | None = None,
                      smooth: int = 20, figsize=(W2, 4.6)):
    """수렴 근거 도판. 입력은 tools/train_diag.py 가 만든 seed*/{metrics,evals}.csv.

    채택 런(figP6)은 20 업데이트마다 보상·포획률·손실만 남겨 "학습이 수렴했는가"에 답할
    지표가 없다. 이 그림은 **같은 설정을 3 시드로 다시 돌려** 표준적인 수렴 진단을 보인다:
      (a) greedy 포획률 --- 100 업데이트마다, 시드별 얇은 선 + 평균 굵은 선 + 휴리스틱 기준선
      (b) 정책 엔트로피(1단계 픽셀 분포, nat) --- 균일분포 상한에서 얼마나 내려왔는가
      (c) 업데이트 간 근사 KL(old‖new, k3 추정량) --- 정책이 한 걸음에 얼마나 움직이는가
      (d) 그래디언트 노름(클리핑 전) --- 클립 한계 1.0 대비
      (e) 표본 후보 보상 − 휴리스틱 후보 보상(같은 전장, 같은 창) --- 0 을 넘으면 정책의
          표본이 평균적으로 휴리스틱을 앞선다
      (f) 파라미터 이동 거리 ||θ_t − θ_BC|| / ||θ_BC|| --- 포화하면 정지점에 닿은 것
    굵은 선은 시드 평균, 띠는 시드 간 최소~최대(3 시드라 표준오차 대신 범위를 그대로 보인다).
    시계열은 창 `smooth` 업데이트 이동평균이다(매 업데이트 원자료는 CSV 에 있다).
    """
    import glob
    import pandas as pd
    use_paper_style()
    runs = sorted(d for d in glob.glob(os.path.join(diag_root, "seed*")) if os.path.isdir(d))
    ms = [pd.read_csv(os.path.join(d, "metrics.csv")) for d in runs]
    es = [pd.read_csv(os.path.join(d, "evals.csv")) for d in runs]
    n_upd = min(len(m) for m in ms)
    ms = [m.iloc[:n_upd] for m in ms]
    base = float(np.mean([e["baseline"].iloc[0] for e in es]))

    def _sm(x):
        return pd.Series(x).rolling(smooth, min_periods=1, center=True).mean().to_numpy()

    def band(ax, col, color, *, log=False, label=None):
        u = ms[0]["upd"].to_numpy()
        Y = np.stack([_sm(m[col].to_numpy()) for m in ms])
        ax.fill_between(u, Y.min(0), Y.max(0), color=color, alpha=0.18, lw=0)
        ax.plot(u, Y.mean(0), color=color, lw=1.1, label=label)
        if log:
            # 로그축 기본 포매터는 mathtext 10^{-1} 을 쓰는데 한글 폰트에 U+2212 가 없어
            # 지수 부호가 깨진다 → 십진 표기(0.1, 0.01)로 바꾼다.
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        return Y

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    (a1, a2, a3), (a4, a5, a6) = axes
    c = OI["blue"]

    # (a) greedy 포획률
    for e in es:
        a1.plot(e["upd"], e["eval_cap"], color=c, lw=0.6, alpha=0.4, marker="o", ms=2.2)
    ue = es[0]["upd"].to_numpy()
    n_ev = min(len(e) for e in es)
    ce = np.stack([e["eval_cap"].to_numpy()[:n_ev] for e in es])
    a1.plot(ue[:n_ev], ce.mean(0), color=c, lw=1.4, marker="o", ms=3.8,
            markerfacecolor="white", markeredgewidth=1.0, label=_T("greedy 평가 (시드 평균)"))
    a1.axhline(base, color=OI["grey"], ls=(0, (2.5, 2)), lw=0.9,
               label=_T(f"휴리스틱 기준선 {base:.3f}"))
    if adopted_run and os.path.exists(os.path.join(adopted_run, "evals.csv")):
        ea = pd.read_csv(os.path.join(adopted_run, "evals.csv"))
        a1.plot(ea["upd"], ea["eval_cap"], linestyle="none", marker="*", ms=8,
                color=OI["vermil"], zorder=5, label=_T("채택 런 (Fig. 학습 경과)"))
    a1.set_ylabel(_T("greedy 포획률"))
    a1.legend(loc="lower right", fontsize=6.2)
    a1.set_title("(a)", loc="left", fontsize=8, color="#555555")
    _ygrid(a1)

    # (b) 정책 엔트로피 --- 균일분포 상한 log(유효픽셀 중앙값 273)
    band(a2, "ent_pix1", c)
    h_max = np.log(273.0)
    a2.axhline(h_max, color=OI["grey"], ls=(0, (2.5, 2)), lw=0.9)
    a2.text(n_upd * 0.98, h_max - 0.08, _T("균일분포 (273 px)"), ha="right", va="top",
            fontsize=6.2, color=OI["grey"])
    a2.set_ylabel(_T("정책 엔트로피 (nat)"))
    a2.set_title("(b)", loc="left", fontsize=8, color="#555555")
    _ygrid(a2)

    # (c) 근사 KL
    band(a3, "kl", c, log=True)
    a3.set_ylabel(_T("업데이트 간 근사 KL"))
    a3.set_title("(c)", loc="left", fontsize=8, color="#555555")
    _ygrid(a3)

    # (d) 그래디언트 노름
    band(a4, "grad_norm", c, log=True)
    a4.axhline(1.0, color=OI["grey"], ls=(0, (2.5, 2)), lw=0.9)
    a4.text(n_upd * 0.98, 1.0 * 1.15, _T("클립 한계 1.0"), ha="right", va="bottom",
            fontsize=6.2, color=OI["grey"])
    a4.set_ylabel(_T("그래디언트 노름 (클리핑 전)"))
    a4.set_xlabel(_T("누적 업데이트"))
    a4.set_title("(d)", loc="left", fontsize=8, color="#555555")
    _ygrid(a4)

    # (e) 표본 후보의 휴리스틱 후보 대비 보상 이득 (R_k − R_heur 의 평균, k≥1)
    band(a5, "gain_heur", OI["green"])
    a5.axhline(0.0, color=OI["grey"], ls=(0, (2.5, 2)), lw=0.9)
    a5.set_ylabel(_T("표본 후보 보상 − 휴리스틱 후보 보상"))
    a5.set_xlabel(_T("누적 업데이트"))
    a5.set_title("(e)", loc="left", fontsize=8, color="#555555")
    _ygrid(a5)

    # (f) 파라미터 이동 거리
    band(a6, "dtheta_rel", OI["vermil"])
    a6.set_ylabel(_T(r"$\|\theta_t-\theta_{\rm BC}\| \,/\, \|\theta_{\rm BC}\|$"))
    a6.set_xlabel(_T("누적 업데이트"))
    a6.set_ylim(bottom=0)
    a6.set_title("(f)", loc="left", fontsize=8, color="#555555")
    _ygrid(a6)

    fig.tight_layout(h_pad=1.0, w_pad=1.4)
    return fig


def diag_summary(diag_root: str) -> dict:
    """본문·캡션에 쓸 수치. 앞 100 / 뒤 100 업데이트 구간의 시드 평균."""
    import glob
    import pandas as pd
    runs = sorted(d for d in glob.glob(os.path.join(diag_root, "seed*")) if os.path.isdir(d))
    ms = [pd.read_csv(os.path.join(d, "metrics.csv")) for d in runs]
    es = [pd.read_csv(os.path.join(d, "evals.csv")) for d in runs]
    out = {"n_seeds": len(runs)}
    for col in ["ent_pix1", "top1", "kl", "grad_norm", "win_heur", "gain_heur", "dtheta_rel",
                "valid", "R"]:
        head = np.mean([m[col].iloc[:100].mean() for m in ms])
        tail = np.mean([m[col].iloc[-100:].mean() for m in ms])
        out[col] = (float(head), float(tail))
    out["eval_first"] = float(np.mean([e["eval_cap"].iloc[0] for e in es]))
    out["eval_last"] = float(np.mean([e["eval_cap"].iloc[-1] for e in es]))
    out["eval_best"] = [float(e["eval_cap"].max()) for e in es]
    out["eval_best_upd"] = [int(e["upd"].iloc[int(e["eval_cap"].idxmax())]) for e in es]
    out["baseline"] = float(es[0]["baseline"].iloc[0])
    return out


# ════════════════════════════════════════════════════════════════════════════
def save_paper_figs(df, llm_df=None, outdir: str = "논문_그래프/8_논문판",
                    *, backend: str = "", seed: int = 0) -> list[str]:
    """논문에 실리는 결과 도판 5장을 한꺼번에 만든다."""
    if not backend:
        ms = P.models_in(df)
        backend = ms[-1] if ms else ""
    out = []
    out.append(save(fig_conditions(df, seed=seed), outdir, "figP1_conditions"))
    out.append(save(fig_forest(df, seed=seed), outdir, "figP2_forest"))
    out.append(save(fig_interaction(df, backend=backend, seed=seed),
                    outdir, "figP3_interaction"))
    out.append(save(fig_ladder(df, seed=seed), outdir, "figP4_ladder"))
    if llm_df is not None and len(llm_df):
        out.append(save(fig_plan_quality(df, llm_df, seed=seed),
                        outdir, "figP5_plan_quality"))
    return out


if __name__ == "__main__":
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="results/eval_merged",
                    help="episodes.csv / llm_metrics.csv 가 있는 폴더")
    ap.add_argument("--outdir", default="논문_그래프/8_논문판")
    ap.add_argument("--backend", default="", help="상호작용 그림에 쓸 지휘관 태그")
    ap.add_argument("--train-run", default="results/train_run_20260819-203248",
                    help="학습 곡선 원자료 폴더 (metrics.csv + evals.csv)")
    ap.add_argument("--diag-root", default="results/train_diag",
                    help="tools/train_diag.py 출력 폴더 (seed*/metrics.csv)")
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(args.csv, "episodes.csv"))
    lp = os.path.join(args.csv, "llm_metrics.csv")
    ld = pd.read_csv(lp) if os.path.exists(lp) else None
    for p in save_paper_figs(d, ld, args.outdir, backend=args.backend):
        print("[ok]", p)
    if os.path.isdir(args.train_run):
        print("[ok]", save(fig_training_curve(args.train_run), args.outdir, "figP6_traincurve"))
        print("[ok]", save(fig_lr_schedule(args.train_run), args.outdir, "figP7_lr_schedule"))
    if os.path.isdir(args.diag_root):
        print("[ok]", save(fig_training_diag(args.diag_root, adopted_run=args.train_run),
                           args.outdir, "figP8_convergence"))
        print(diag_summary(args.diag_root))
