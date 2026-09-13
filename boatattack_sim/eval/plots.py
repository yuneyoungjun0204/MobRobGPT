"""논문 도판 생성기 — harness 가 낸 CSV 하나에서 그림 전부를 만든다.

원칙
    ① 숫자는 CSV 에서만 온다. 함수 인자로 값을 받아 그리는 경로를 만들지 않는다(전사 오류 차단).
    ② 평균 막대에는 **반드시** 오차막대(부트스트랩 CI)를 붙인다. 평균만 그린 그림은
       "그래서 유의한가"에 답을 못 해 심사에서 지적된다.
    ③ 벡터(PDF)로 저장한다. 논문은 확대해도 안 깨져야 한다. PNG 는 미리보기용 부산물.
    ④ 흑백 인쇄를 가정해 색 + 해칭을 같이 쓴다. 색만으로 구분되는 그림은 인쇄에서 죽는다.

사용:
    from boatattack_sim.eval import plots
    plots.save_all(df, outdir="논문초안/figs_snak")
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                      # 헤드리스 저장 전용 (창을 띄우지 않는다)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as _fm

from . import stats as S

# ── 한글 폰트 (renderer.py 와 같은 사다리) ──
_KFONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Batang")
_avail = {f.name for f in _fm.fontManager.ttflist}
for _kf in _KFONTS:
    if _kf in _avail:
        matplotlib.rcParams["font.family"] = _kf
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType 임베드 (논문 제출 요구사항인 경우가 많다)
matplotlib.rcParams["ps.fonttype"] = 42

# ★ 한글 폰트(Malgun Gothic 등)에 없는 글리프를 쓰면 matplotlib 이 경고를 stderr 로 내고
#   해당 글자가 두부(□)로 찍힌다. 특히 U+2212(MINUS SIGN)는 한글 워드프로세서에서 복사한
#   문자열에 잘 섞여 들어온다. 라벨을 만들 때 이 함수를 통과시켜 ASCII 로 바꾼다.
#   (경고가 stderr 로 나가면 PowerShell 파이프에서 exit 1 로 잡혀 실행이 실패로 보인다.)
_GLYPH_FIX = {
    "−": "-",   # MINUS SIGN → 하이픈
    "–": "-",   # EN DASH
    "—": "-",   # EM DASH  (제목의 '—' 는 폰트에 있으나 안전하게 통일)
    "×": "x",   # MULTIPLICATION SIGN
}


def safe_text(s: str) -> str:
    """폰트에 없을 수 있는 기호를 ASCII 로 치환. 축·범례 라벨에 쓴다."""
    for a, b in _GLYPH_FIX.items():
        s = s.replace(a, b)
    return s

# ── 조건 표기 (harness.MATRIX_2x2 와 순서·키가 일치해야 한다) ──
COND_ORDER = ("heur_heur", "heur_unet", "llm_heur", "llm_unet")
COND_SHORT = {
    "heur_heur": "휴리스틱\n+휴리스틱",
    "heur_unet": "휴리스틱\n+U-Net",
    "llm_heur":  "LLM\n+휴리스틱",
    "llm_unet":  "LLM\n+U-Net",
}
#: baseline 은 회색, 제안은 진한 색. 단일 요인 조건은 중간 톤.
COND_COLOR = {"heur_heur": "#9E9E9E", "heur_unet": "#5C9BD5",
              "llm_heur": "#ED9A4B", "llm_unet": "#2E5E8E"}
COND_HATCH = {"heur_heur": "", "heur_unet": "//", "llm_heur": "\\\\", "llm_unet": "xx"}

FORM_KO = {"concentrated": "집중", "diversionary": "양동", "wave": "파상", "ALL": "전체"}

METRIC_KO = {
    "capture_rate": "포획률",
    "captures": "포획 수",
    "breaches": "돌파 수",
    "resolved_frac": "교전 종결률",
    # 안전
    "collisions": "충돌 건수 (아군+모선)",
    "collision_rate": "충돌률 (척당)",
    "ally_losses": "아군 손실",
    "loss_rate": "손실률 (척당)",
    "net_touches": "그물 접촉",
    "mother_dmin": "모선 최소접근 (m)",
    "ally_dmin": "아군 최소이격 (m)",
    # 자원 효율
    "nets_used": "사용 그물 수",
    "nets_per_capture": "포획 1척당 그물",
    "captures_per_net": "그물 1장당 포획",
    # 기동 비용
    "traveled_m": "총 이동거리 (m)",
    "traveled_per_capture": "포획 1척당 이동 (m)",
    "turn_sum_rad": "총 선회량 (rad)",
    # 교전 품질
    "cap_dist_mean": "평균 포획거리 (m)",
    "cap_time_mean": "평균 포획시각 (step)",
    "cap_time_frac": "제압 소요 비율",
    "net_deploy_edist": "전개시 적 근접도 (m)",
}

#: ★ 낮을수록 좋은 지표. 그림의 화살표·레이더 반전·표의 개선 방향 판정에 쓴다.
#   이 목록이 틀리면 "개선"과 "악화"가 뒤집혀 표시된다 — 지표를 추가할 때 반드시 갱신할 것.
LOWER_BETTER = frozenset({
    "breaches", "collisions", "collision_rate", "ally_losses", "loss_rate", "net_touches",
    "nets_used", "nets_per_capture", "traveled_m", "traveled_per_capture", "turn_sum_rad",
    "cap_time_mean", "cap_time_frac",
    # 그물은 물에 고정되지 않아 표류한다 → 적에 붙여 뿌릴수록 유리 = 작을수록 좋다.
    "net_deploy_edist",
})


def better_dir(metric: str) -> str:
    """그림 라벨용: 이 지표는 어느 방향이 좋은가."""
    return "낮을수록 좋음" if metric in LOWER_BETTER else "높을수록 좋음"


#: ── 영문 표기 (해외 학회·영문 논문용) ──────────────────────────────────────
#   한글 맵과 키를 1:1로 맞춰 둔다. 지표를 추가하면 양쪽 모두 갱신할 것 —
#   한쪽만 채우면 영문 그림에 키 이름이 그대로 노출된다.
METRIC_EN = {
    "capture_rate": "Capture rate",
    "captures": "Captures",
    "breaches": "Breaches",
    "resolved_frac": "Engagement resolution",
    "collisions": "Collisions (ally + mothership)",
    "collision_rate": "Collision rate (per ship)",
    "ally_losses": "Ally losses",
    "loss_rate": "Loss rate (per ship)",
    "net_touches": "Net contacts",
    "mother_dmin": "Min. range to mothership (m)",
    "ally_dmin": "Min. inter-ally range (m)",
    "nets_used": "Nets used",
    "nets_per_capture": "Nets per capture",
    "captures_per_net": "Captures per net",
    "traveled_m": "Total distance (m)",
    "traveled_per_capture": "Distance per capture (m)",
    "turn_sum_rad": "Total turning (rad)",
    "cap_dist_mean": "Mean capture range (m)",
    "cap_time_mean": "Mean capture time (step)",
    "cap_time_frac": "Normalized capture time",
    "net_deploy_edist": "Enemy range at net deployment (m)",
}
COND_SHORT_EN = {
    "heur_heur": "Heuristic + Heuristic",
    "heur_unet": "Heuristic + U-Net",
    "llm_heur":  "LLM + Heuristic",
    "llm_unet":  "LLM + U-Net",
}
GROUP_EN = {
    "방어 성과": "Defense", "안전": "Safety", "자원 효율": "Resource",
    "기동 비용": "Maneuver cost", "교전 품질": "Engagement quality",
}
FORM_EN = {"concentrated": "Concentrated", "diversionary": "Diversionary",
           "wave": "Wave", "ALL": "All"}


def metric_label(metric: str, lang: str = "ko") -> str:
    """지표 표시명. lang='en' 이면 영문."""
    if lang == "en":
        return METRIC_EN.get(metric, metric)
    return METRIC_KO.get(metric, metric)


def cond_label(cond: str, lang: str = "ko", *, newline: bool = True) -> str:
    """조건 표시명. 백엔드 태그가 있으면 괄호로 덧붙인다."""
    base_key = _base_key(cond)
    if lang == "en":
        base = COND_SHORT_EN.get(base_key, base_key)
        if newline:
            # 세로 막대 축에서는 한 줄이면 이웃 라벨과 붙는다 → '+' 에서 접는다.
            base = base.replace(" + ", "\n+ ")
    else:
        base = COND_SHORT.get(base_key, base_key).replace("\n", "")
    b = _backend_of(cond)
    if not b:
        return base
    return f"{base}\n({b})" if newline else f"{base} ({b})"


#: LLM 계획 품질 지표 한글명 (commander/llm_metrics.py 의 컬럼).
LLM_METRIC_KO = {
    "latency_s": "응답 지연 (s)",
    "fallback": "폴백률",
    "coverage": "클러스터 커버리지",
    "churn": "배정 요동 (churn)",
    "crossings": "교차 배정 수",
    "n_hold": "HOLD 척수",
    "heuristic_agreement": "휴리스틱 일치율",
}


#: 백엔드 접미사(`llm_unet@openai`)를 붙였을 때 쓰는 색. 지휘관별로 톤을 나눈다.
BACKEND_TINT = {
    # 조건 접미사(`llm_unet@qwen2.5-7b`)에 실제로 붙는 것은 harness.slugify_model 이
    # 만든 **모델 태그**다. 백엔드 이름(openai/ollama)도 남겨 둔다 — 예전 CSV 호환.
    "qwen2.5-7b": "#B07AA1",            # 로컬 소형
    "qwen2.5-14b": "#7B4FA0",           # 로컬 중형 (같은 계열이라 같은 색조의 진한 톤)
    "gemini-3.5-flash-lite": "#2E8B7A",  # 클라우드
    "gpt-4o-mini": "#2E5E8E",
    "openai": "#2E5E8E", "ollama": "#7B4FA0", "gemini": "#2E8B7A", "": None,
}

#: 모델 태그 → 그림에 찍을 이름. 없으면 태그를 그대로 쓴다.
MODEL_LABEL = {
    "qwen2.5-7b": "Qwen2.5 7B",
    "qwen2.5-14b": "Qwen2.5 14B",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-4o-mini": "GPT-4o mini",
}

#: 능력 순서(작은 것 → 큰 것). 모델 비교 그림의 x 축 순서를 고정한다 —
#  "능력이 오르면 성능이 오른다"를 보이려면 축이 능력 순이어야 한다.
MODEL_ORDER = ("qwen2.5-7b", "qwen2.5-14b", "gpt-4o-mini", "gemini-3.5-flash-lite")


def model_label(tag: str) -> str:
    return MODEL_LABEL.get(tag, tag)


#: 조건 태그에 실리는 배정모드 접미사 → 모드 이름. harness.ASSIGN_MODE_TAG 의 역방향이다.
#  (여기 하드코딩하는 이유: plots 는 harness 를 import 하지 않는다 — 그림만 그리는 층이
#   시뮬 층에 의존하면 CSV 만 있으면 되는 --from-csv 경로가 깨진다.)
MODE_SUFFIX = {"+hyb": "hybrid", "+code": "code"}
MODE_LABEL = {"llm": "순수 LLM", "hybrid": "LLM+코드보완", "code": "코드 재최적화"}
MODE_LABEL_EN = {"llm": "LLM only", "hybrid": "LLM + code fill", "code": "code re-opt"}


def _mode_of(tag: str) -> str:
    """모델 태그에서 배정모드를 읽는다. 접미사가 없으면 기본 "llm"."""
    for suf, mode in MODE_SUFFIX.items():
        if tag.endswith(suf):
            return mode
    return "llm"


def _base_model(tag: str) -> str:
    """모델 태그에서 배정모드 접미사를 뗀 순수 모델 이름."""
    for suf in MODE_SUFFIX:
        if tag.endswith(suf):
            return tag[: -len(suf)]
    return tag


def models_in(df, *, mode: str | None = "llm") -> list[str]:
    """CSV 에 있는 지휘관 모델 태그를 능력 순서로. 수집이 덜 끝난 것은 뺀다.

    `mode="llm"`(기본) 이면 **배정모드 변형(`+hyb`/`+code`)을 뺀다.** 능력축 그림
    (figM*)에 hybrid 변형이 섞이면 "능력이 오르면 성능이 오른다"는 축이 무너진다 —
    같은 모델이 모드만 달리해 두 번 서기 때문이다. 모드 비교는 figO 가 따로 한다.
    `mode=None` 이면 전부 돌려준다.
    """
    d = S.drop_incomplete(df)
    tags = {_backend_of(c) for c in d["condition"].unique() if "@" in c}
    if mode is not None:
        tags = {t for t in tags if _mode_of(t) == mode}
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda t: (rank.get(_base_model(t), 99), t))


def _base_key(cond: str) -> str:
    """`llm_unet@openai` → `llm_unet`. 백엔드 접미사를 떼어 기본 조건을 얻는다."""
    return cond.split("@", 1)[0]


def _backend_of(cond: str) -> str:
    return cond.split("@", 1)[1] if "@" in cond else ""


def _short(cond: str) -> str:
    """그림에 찍는 라벨. 백엔드가 있으면 줄을 하나 더 쓴다."""
    base = COND_SHORT.get(_base_key(cond), _base_key(cond))
    b = _backend_of(cond)
    return f"{base}\n({b})" if b else base


def _color(cond: str) -> str:
    b = _backend_of(cond)
    if b and BACKEND_TINT.get(b):
        return BACKEND_TINT[b]
    return COND_COLOR.get(_base_key(cond), "#777777")


def _hatch(cond: str) -> str:
    return COND_HATCH.get(_base_key(cond), "")


def _conds(df, *, complete_only: bool = True):
    """CSV 에 실제로 있는 조건만 정해진 순서로. 백엔드 변형은 기본 조건 뒤에 이어 붙인다.

    `complete_only=True`(기본) 면 수집이 덜 끝난 조건을 뺀다. 안 빼면 집계(pivot_by_seed)는
    이미 그 조건을 제외했는데 축에는 자리가 남아 **빈 막대**가 그려지고 라벨만 겹친다.
    """
    d = S.drop_incomplete(df) if complete_only else df
    have = list(dict.fromkeys(d["condition"].tolist()))
    rank = {c: i for i, c in enumerate(COND_ORDER)}
    return sorted(have, key=lambda c: (rank.get(_base_key(c), 99), _backend_of(c)))


#: 그림 → 하위 폴더. "이 그림이 어떤 질문에 답하는가"로 가른다.
#  save_all 이 이 표를 따라 outdir 아래에 나눠 저장하므로, 재생성해도 분류가 유지된다.
#  접두사 매칭이라 figA 는 figA2..figA7 까지, figI 는 figI2..figI5 까지 함께 걸린다.
FIG_GROUPS = (
    ("1_조건비교",  ("figA", "figE")),          # 조건(2x2) x 포메이션 지표 막대
    ("2_요약패널",  ("figI", "figJ")),          # 여러 지표를 한 장에
    ("3_계층효과",  ("figB", "figC", "figD")),  # 배정 x 기동 분해 · 대응비교
    ("4_모델비교",  ("figG", "figH", "figK", "figM")),   # 지휘관 모델 축
    ("5_구조도",    ("unet_",)),                # 아키텍처 도판
    ("7_메커니즘",  ("figN", "figO")),          # 왜 그런가 — 계획품질·배정권한 ablation
)
#: 어디에도 안 걸리는 그림이 가는 곳. 새 그림을 추가하고 분류를 깜빡해도 사라지지 않는다.
FIG_GROUP_OTHER = "9_기타"
#: 애니메이션(GIF/MP4). make_commander_gif.py 가 여기에 넣는다.
FIG_GROUP_VIDEO = "6_영상"


def fig_group(name: str) -> str:
    """그림 이름 → 하위 폴더 이름."""
    for grp, prefixes in FIG_GROUPS:
        if any(name.startswith(pre) for pre in prefixes):
            return grp
    return FIG_GROUP_OTHER


def _save(fig, outdir: str, name: str, *, png: bool = True,
          group: bool = True) -> list[str]:
    """그림 저장. `group=True`(기본) 면 fig_group(name) 하위 폴더에 넣는다."""
    if group:
        outdir = os.path.join(outdir, fig_group(name))
    os.makedirs(outdir, exist_ok=True)
    paths = []
    p = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(p, bbox_inches="tight"); paths.append(p)
    if png:
        p2 = os.path.join(outdir, f"{name}.png")
        fig.savefig(p2, dpi=200, bbox_inches="tight"); paths.append(p2)
    plt.close(fig)
    return paths


# ── Fig A. 조건 × 포메이션 그룹 막대 (본표의 그림 버전) ─────────────────────
def fig_condition_bars(df, metric: str = "capture_rate", *, seed: int = 0,
                       figsize=(7.2, 3.4)):
    """포메이션별로 4조건을 나란히. **오차막대 = 부트스트랩 95% CI**.

    논문의 주력 그림. 어느 포메이션에서 이기고 어디서 안 이기는지가 한눈에 보인다.
    """
    conds = _conds(df)
    forms = [f for f in ("concentrated", "diversionary", "wave")
             if f in set(df["formation"].unique())]
    fig, ax = plt.subplots(figsize=figsize)
    W = 0.8 / max(len(conds), 1)
    x = np.arange(len(forms) + 1)          # +1 = ALL

    for i, c in enumerate(conds):
        means, los, his = [], [], []
        for f in forms + ["ALL"]:
            piv = S.pivot_by_seed(df, metric, formation=None if f == "ALL" else f)
            est = (S.mean_ci(piv[c].to_numpy(), seed=seed) if c in piv
                   else S.Estimate(np.nan, np.nan, np.nan, 0))
            means.append(est.mean)
            los.append(est.mean - est.lo if np.isfinite(est.lo) else 0.0)
            his.append(est.hi - est.mean if np.isfinite(est.hi) else 0.0)
        ax.bar(x + (i - (len(conds) - 1) / 2) * W, means, W * 0.92,
               yerr=[los, his], capsize=2.5, label=_short(c).replace("\n", " "),
               color=_color(c), hatch=_hatch(c), edgecolor="white", linewidth=0.6,
               error_kw={"elinewidth": 0.9, "ecolor": "#333"})

    ax.set_xticks(x)
    ax.set_xticklabels([FORM_KO.get(f, f) for f in forms] + ["전체"])
    ax.set_ylabel(METRIC_KO.get(metric, metric))
    ax.set_xlabel("적 포메이션")
    # 범례는 축 위로 뺀다 — 포획률은 1.0 근처에 몰려서 축 안에 두면 막대를 가린다.
    # ★ 제목 pad 는 **범례 행 수에 따라** 벌린다. 고정 pad(26)는 1행 기준이라,
    #   조건이 5개 이상이면 범례가 2행이 되면서 제목과 겹쳐 둘 다 못 읽게 된다
    #   (8조건 수집에서 실제로 발생 — 논문 주력 그림이 판독 불가였다).
    ncol = min(len(conds), 4)
    nrow = int(np.ceil(len(conds) / max(ncol, 1)))
    ax.legend(fontsize=8, ncol=ncol, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{METRIC_KO.get(metric, metric)} — 조건별 (오차막대: 부트스트랩 95% CI)",
                 fontsize=9.5, loc="left", pad=12 + 15 * nrow)
    fig.tight_layout()
    return fig


# ── Fig B. baseline 대비 대응차이 forest plot ───────────────────────────────
def fig_paired_forest(df, metric: str = "capture_rate", *, baseline: str = "heur_heur",
                      seed: int = 0, figsize=(6.4, 3.2)):
    """baseline 대비 **대응표본 차이**와 CI. 0 선을 넘지 않으면 유의하지 않다.

    심사자가 가장 먼저 보는 그림이다 — "얼마나 좋아졌고, 그게 확실한가"에 직접 답한다.
    """
    conds = [c for c in _conds(df) if c != baseline]
    forms = [f for f in ("concentrated", "diversionary", "wave")
             if f in set(df["formation"].unique())] + ["ALL"]

    labels, mus, los, his, cols = [], [], [], [], []
    for f in forms:
        piv = S.pivot_by_seed(df, metric, formation=None if f == "ALL" else f)
        if baseline not in piv:
            continue
        base = piv[baseline].to_numpy()
        for c in conds:
            if c not in piv:
                continue
            d = S.paired_diff(piv[c].to_numpy(), base, seed=seed)
            # COND_SHORT 는 이미 '+' 를 품고 개행으로 줄을 나눈다 → 개행만 지운다(++ 방지).
            labels.append(f"{FORM_KO.get(f, f)} · {_short(c).replace(chr(10), ' ')}")
            mus.append(d.mean)
            los.append(d.mean - d.lo if np.isfinite(d.lo) else 0.0)
            his.append(d.hi - d.mean if np.isfinite(d.hi) else 0.0)
            cols.append(_color(c))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.30 * len(labels) + 1.0)))
    ax.axvline(0, color="#C62828", lw=1.0, ls="--", zorder=1)
    ax.errorbar(mus, y, xerr=[los, his], fmt="none", ecolor="#555", elinewidth=1.1,
                capsize=3, zorder=2)
    ax.scatter(mus, y, c=cols, s=34, zorder=3, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {METRIC_KO.get(metric, metric)}  (baseline 대비, 대응표본)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("baseline 대비 개선폭 — CI가 0선을 넘으면 유의하지 않음",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    return fig


# ── Fig C. 2×2 상호작용 ────────────────────────────────────────────────────
def fig_interaction(df, metric: str = "capture_rate", *, seed: int = 0, backend: str = "",
                    figsize=(5.4, 3.4)):
    """두 계층이 서로를 돕는가. 두 선이 **평행이 아니면** 상호작용이 있다.

    제안의 핵심 주장("계층 분리가 단순 합 이상")을 직접 검증하는 그림이다.
    평행하게 나오면 그것도 결과다 — 두 기여가 독립적이라고 정직하게 쓰면 된다.

    `backend` 를 주면 그 지휘관의 LLM 조건(`llm_*@backend`)으로 분해한다.
    여러 백엔드를 한 CSV 에 담았을 때 백엔드마다 한 장씩 뽑기 위한 것이다.
    """
    piv = S.pivot_by_seed(df, metric)
    sfx = f"@{backend}" if backend else ""
    lk_h, lk_u = f"llm_heur{sfx}", f"llm_unet{sfx}"
    need = ["heur_heur", "heur_unet", lk_h, lk_u]
    if any(c not in piv for c in need):
        raise ValueError(f"상호작용 그림에는 {need} 가 모두 필요하다. "
                         f"있는 것: {list(piv.columns)}")

    fig, ax = plt.subplots(figsize=figsize)
    x = [0, 1]
    llm_lab = f"LLM 지휘관 ({backend})" if backend else "LLM 지휘관"
    for assign, keys, col, mk in (("휴리스틱 배정", ("heur_heur", "heur_unet"), "#9E9E9E", "o"),
                                  (llm_lab, (lk_h, lk_u), _color(lk_u), "s")):
        m, lo, hi = [], [], []
        for k in keys:
            e = S.mean_ci(piv[k].to_numpy(), seed=seed)
            m.append(e.mean)
            lo.append(e.mean - e.lo if np.isfinite(e.lo) else 0.0)
            hi.append(e.hi - e.mean if np.isfinite(e.hi) else 0.0)
        ax.errorbar(x, m, yerr=[lo, hi], marker=mk, color=col, capsize=3,
                    lw=1.8, ms=7, label=assign)

    inter = S.interaction(df, metric, seed=seed, backend=backend)["interaction"]
    ax.set_xticks(x); ax.set_xticklabels(["휴리스틱 기동", "U-Net 점수맵"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(METRIC_KO.get(metric, metric))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    sig = "유의" if inter.significant else "유의하지 않음"
    ax.set_title(f"계층 상호작용 — Δ={inter.mean:+.3f} "
                 f"[{inter.lo:+.3f}, {inter.hi:+.3f}] ({sig})", fontsize=9.5, loc="left")
    fig.tight_layout()
    return fig


# ── Fig D. 시드별 대응 산점도 ──────────────────────────────────────────────
def fig_paired_scatter(df, metric: str = "capture_rate", *, baseline: str = "heur_heur",
                       treatment: str = "llm_unet", figsize=(3.6, 3.6)):
    """시드마다 점 하나. 대각선 위 = 제안이 이긴 시드.

    평균이 숨기는 것을 드러낸다 — "평균은 올랐지만 절반의 시드에서는 졌다"가 보인다.
    심사자의 신뢰를 크게 얻는 그림이라 넣을 값어치가 있다.
    """
    piv = S.pivot_by_seed(df, metric)
    if baseline not in piv or treatment not in piv:
        raise ValueError(f"필요 조건 없음: {baseline}, {treatment}")
    b = piv[baseline].to_numpy(); t = piv[treatment].to_numpy()
    ok = np.isfinite(b) & np.isfinite(t)
    b, t = b[ok], t[ok]

    fig, ax = plt.subplots(figsize=figsize)
    lim = [min(b.min(), t.min()), max(b.max(), t.max())]
    pad = 0.05 * (lim[1] - lim[0] + 1e-9)
    lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, color="#C62828", ls="--", lw=1.0, zorder=1)
    win = t > b
    ax.scatter(b[win], t[win], s=26, c="#2E5E8E", alpha=0.8, zorder=2,
               edgecolor="white", linewidth=0.5, label=f"제안 우세 ({win.sum()})")
    ax.scatter(b[~win], t[~win], s=26, c="#B0653C", alpha=0.8, zorder=2, marker="v",
               edgecolor="white", linewidth=0.5, label=f"baseline 우세 ({(~win).sum()})")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"baseline {METRIC_KO.get(metric, metric)}")
    ax.set_ylabel(f"제안 {METRIC_KO.get(metric, metric)}")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("시드별 대응 비교 (같은 전장)", fontsize=9.5, loc="left")
    fig.tight_layout()
    return fig


# ── Fig E. 다지표 레이더 (트레이드오프) ────────────────────────────────────
#: 레이더 기본 축 — 성과·안전·자원·기동·품질을 한 장에 모은다.
RADAR_DEFAULT = ("capture_rate", "resolved_frac", "cap_dist_mean", "cap_time_mean",
                 "net_deploy_edist", "collision_rate", "nets_per_capture",
                 "traveled_per_capture")


def fig_tradeoff_radar(df, *, metrics=RADAR_DEFAULT, lower_better=None, figsize=(5.2, 5.2)):
    """여러 지표를 한 장에. 각 축은 조건 간 min-max 로 정규화하고,
    **낮을수록 좋은 지표는 뒤집어** 항상 '바깥 = 우수'가 되게 한다.

    포획률만 보고하면 "그물을 마구 써서 올린 것 아닌가"라는 질문에 답을 못 한다.
    자원·기동·안전을 같이 그려야 방어가 성립한다.
    """
    lower_better = LOWER_BETTER if lower_better is None else frozenset(lower_better)
    keys = [m for m in metrics if m in df.columns]
    conds = _conds(df)
    vals = {c: [] for c in conds}
    for m in keys:
        piv = S.pivot_by_seed(df, m)
        col = {c: float(np.nanmean(piv[c].to_numpy())) if c in piv else np.nan for c in conds}
        arr = np.array([col[c] for c in conds], float)
        rng = np.nanmax(arr) - np.nanmin(arr)
        if not np.isfinite(rng) or rng <= 0:
            norm = np.full_like(arr, 0.5)
        else:
            norm = (arr - np.nanmin(arr)) / rng
            if m in lower_better:
                norm = 1.0 - norm            # 낮을수록 좋은 지표는 뒤집어 '바깥=좋음' 통일
        for i, c in enumerate(conds):
            vals[c].append(norm[i])

    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    for c in conds:
        v = vals[c] + vals[c][:1]
        ax.plot(ang, v, lw=1.6, color=_color(c), label=_short(c).replace("\n", " "))
        ax.fill(ang, v, color=_color(c), alpha=0.10)
    ax.set_xticks(ang[:-1])
    # 축 라벨에 ↓ 를 달아 '이 축은 원래 낮을수록 좋은 값을 뒤집어 그렸다'를 명시한다.
    ax.set_xticklabels([METRIC_KO.get(k, k).split(" (")[0] + ("↓" if k in lower_better else "")
                        for k in keys], fontsize=7.5)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["", "", ""], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.30, 1.13), frameon=False)
    ax.set_title("다지표 트레이드오프 (바깥=우수)\n↓ 표시 = 원래 낮을수록 좋은 지표(반전)",
                 fontsize=8.5, pad=18)
    fig.tight_layout()
    return fig


# ── Fig I. 다지표 패널 — 모든 관점을 한 장에 ────────────────────────────────
#: 논문 결과 절에 그대로 들어갈 지표 묶음. (그룹명, [지표들])
METRIC_GROUPS = (
    ("방어 성과", ["capture_rate", "breaches"]),
    ("안전",      ["collision_rate", "net_touches"]),
    ("자원 효율", ["nets_per_capture"]),
    ("기동 비용", ["traveled_per_capture", "turn_sum_rad"]),
    ("교전 품질", ["cap_dist_mean", "cap_time_mean", "net_deploy_edist"]),
)


def fig_metric_panels(df, *, groups=METRIC_GROUPS, seed: int = 0, ncols: int = 3,
                      panel=(2.45, 2.15)):
    """지표 하나당 작은 막대 패널. 조건 전체를 모든 관점에서 한 장에 본다.

    포획률 단일 지표로는 "그물을 퍼부어 올린 것 아닌가", "배가 부딪히지 않았나",
    "얼마나 돌아다녔나"에 답할 수 없다. 이 그림이 그 질문들을 한꺼번에 닫는다.
    각 패널 제목에 **개선 방향(↑/↓)** 을 적어 오독을 막는다.
    """
    keys = [m for _, ms in groups for m in ms if m in df.columns]
    if not keys:
        raise ValueError("그릴 지표가 없다 — CSV 컬럼을 확인하라")
    conds = _conds(df)
    nrow = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrow, ncols, figsize=(panel[0] * ncols, panel[1] * nrow))
    axes = np.atleast_1d(axes).ravel()

    grp_of = {m: g for g, ms in groups for m in ms}
    for ax, m in zip(axes, keys):
        means, los, his, cols = [], [], [], []
        for c in conds:
            piv = S.pivot_by_seed(df, m)
            e = (S.mean_ci(piv[c].to_numpy(), seed=seed) if c in piv
                 else S.Estimate(np.nan, np.nan, np.nan, 0))
            means.append(e.mean)
            los.append(e.mean - e.lo if np.isfinite(e.lo) else 0.0)
            his.append(e.hi - e.mean if np.isfinite(e.hi) else 0.0)
            cols.append(_color(c))
        ax.bar(range(len(conds)), means, 0.68, yerr=[los, his], capsize=2,
               color=cols, hatch=[_hatch(c) for c in conds],
               edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 0.8})
        arrow = "↓" if m in LOWER_BETTER else "↑"
        ax.set_title(f"{METRIC_KO.get(m, m)} {arrow}", fontsize=8.5, loc="left")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([_short(c) for c in conds], fontsize=5.6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # 그룹을 색 띠로 표시 — 어느 관점의 지표인지 눈으로 묶인다.
        ax.text(0.995, 1.02, grp_of.get(m, ""), transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color="#666")

    for ax in axes[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("다지표 평가 — ↑ 높을수록 / ↓ 낮을수록 좋음 (오차막대: 부트스트랩 95% CI)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ── Fig F. 학습 곡선 (metrics.csv) ─────────────────────────────────────────
def fig_training_curve(csv_paths: dict[str, str], *, metric: str = "cap_rate",
                       smooth: int = 20, figsize=(6.4, 3.2)):
    """학습 중 metrics.csv 의 곡선. `csv_paths` = {범례이름: 경로}.

    이동평균으로 매끈하게 만들되 **원 곡선을 옅게 함께** 그린다 —
    스무딩만 보이면 실제 분산을 숨기는 그림이 된다.
    """
    import pandas as pd
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    for i, (name, path) in enumerate(csv_paths.items()):
        d = pd.read_csv(path)
        if metric not in d.columns:
            raise ValueError(f"{path} 에 '{metric}' 컬럼이 없다. 있는 것: {list(d.columns)}")
        x = d["upd"] if "upd" in d.columns else np.arange(len(d))
        y = d[metric].astype(float)
        col = cmap(i % 10)
        ax.plot(x, y, color=col, alpha=0.20, lw=0.7)
        ax.plot(x, y.rolling(max(1, smooth), min_periods=1).mean(), color=col, lw=1.6,
                label=name)
    ax.set_xlabel("업데이트"); ax.set_ylabel(METRIC_KO.get(metric, metric))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"학습 곡선 (이동평균 {smooth}, 원 곡선 옅게)", fontsize=9.5, loc="left")
    fig.tight_layout()
    return fig


# ── Fig G. LLM 지휘관 계측 ─────────────────────────────────────────────────
def fig_llm_metrics(llm_df, *, figsize=(7.2, 2.9)):
    """LLM 계층 단독 지표 3패널: 응답지연 분포 · 계획 품질 · 휴리스틱 일치율.

    포획률과 독립이라 "LLM 이 무엇을 했는가"를 따로 주장할 수 있다.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ① 지연 분포 + p50/p95
    ax = axes[0]
    lat = llm_df["latency_s"].dropna().to_numpy()
    if lat.size:
        ax.hist(lat, bins=min(24, max(5, lat.size // 3)), color="#5C9BD5",
                edgecolor="white", linewidth=0.5)
        for q, c, ls in ((0.50, "#2E5E8E", "-"), (0.95, "#C62828", "--")):
            v = float(np.quantile(lat, q))
            ax.axvline(v, color=c, ls=ls, lw=1.2, label=f"p{int(q*100)}={v:.2f}s")
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("응답 지연 (s)"); ax.set_ylabel("호출 수")
    ax.set_title("① 지휘관 응답 지연", fontsize=9, loc="left")

    # ② 계획 품질 (0~1 지표들)
    ax = axes[1]
    qk = [("coverage", "커버리지"), ("churn", "churn"),
          ("heuristic_agreement", "휴리스틱\n일치율")]
    qk = [(k, n) for k, n in qk if k in llm_df.columns]
    m = [float(llm_df[k].mean(skipna=True)) for k, _ in qk]
    e = [float(llm_df[k].std(skipna=True)) for k, _ in qk]
    ax.bar(range(len(qk)), m, yerr=e, capsize=3, color=["#2E5E8E", "#ED9A4B", "#9E9E9E"][:len(qk)],
           edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.9})
    ax.set_xticks(range(len(qk))); ax.set_xticklabels([n for _, n in qk], fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("비율")
    ax.set_title("② 계획 품질 (막대: 평균±SD)", fontsize=9, loc="left")

    # ③ 폴백 / 교차 배정 (개수 지표)
    ax = axes[2]
    fb = float(llm_df["fallback"].mean()) if "fallback" in llm_df else np.nan
    cx = float(llm_df["crossings"].mean()) if "crossings" in llm_df else np.nan
    hold = float(llm_df["n_hold"].mean()) if "n_hold" in llm_df else np.nan
    ax.bar(["폴백률", "교차 배정", "HOLD 수"], [fb, cx, hold],
           color=["#C62828", "#ED9A4B", "#9E9E9E"], edgecolor="white", linewidth=0.6)
    ax.set_title("③ 신뢰성 지표", fontsize=9, loc="left")

    for a in axes:
        a.grid(axis="y", alpha=0.25, linewidth=0.5); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ── Fig H. 지휘관 백엔드 비교 (로컬 vs GPT) ────────────────────────────────
def backends_in(df) -> list[str]:
    """CSV 에 들어 있는 LLM 백엔드 목록."""
    return sorted({_backend_of(c) for c in df["condition"].unique() if _backend_of(c)})


def fig_backend_compare(df, metric: str = "capture_rate", *, seed: int = 0,
                        base: str = "llm_unet", figsize=(6.6, 3.2)):
    """같은 기동 계층에서 **지휘관만 바꿨을 때**의 차이.

    논문에서 답해야 하는 실질 질문: "고가의 상용 API 가 꼭 필요한가, 로컬 소형 모델로도 되는가."
    왼쪽은 절대 성능(휴리스틱 배정 기준선 포함), 오른쪽은 백엔드 간 대응차이다.
    """
    bks = backends_in(df)
    if len(bks) < 2:
        raise ValueError(f"백엔드 비교에는 2개 이상이 필요하다. 있는 것: {bks}")
    piv = S.pivot_by_seed(df, metric)
    keys = [f"{base}@{b}" for b in bks if f"{base}@{b}" in piv]
    if len(keys) < 2:
        raise ValueError(f"조건 {base}@* 가 2개 미만이다: {list(piv.columns)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ① 절대 성능 — 휴리스틱 배정 기준선을 점선으로 같이 그린다
    m, lo, hi, cols = [], [], [], []
    for k in keys:
        e = S.mean_ci(piv[k].to_numpy(), seed=seed)
        m.append(e.mean)
        lo.append(e.mean - e.lo if np.isfinite(e.lo) else 0.0)
        hi.append(e.hi - e.mean if np.isfinite(e.hi) else 0.0)
        cols.append(_color(k))
    ax1.bar(range(len(keys)), m, 0.6, yerr=[lo, hi], capsize=3, color=cols,
            edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.9})
    ref = "heur_unet" if "heur_unet" in piv else None
    if ref:
        rv = float(np.nanmean(piv[ref].to_numpy()))
        ax1.axhline(rv, color="#C62828", ls="--", lw=1.0,
                    label=f"휴리스틱 배정 ({rv:.3f})")
        ax1.legend(fontsize=7.5, frameon=False)
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels([_backend_of(k) for k in keys], fontsize=8.5)
    ax1.set_ylabel(METRIC_KO.get(metric, metric))
    ax1.set_title("① 지휘관별 절대 성능", fontsize=9, loc="left")

    # ② 백엔드 간 대응차이 (첫 백엔드 기준)
    ref_k = keys[0]
    labs, mus, los2, his2 = [], [], [], []
    for k in keys[1:]:
        d = S.paired_diff(piv[k].to_numpy(), piv[ref_k].to_numpy(), seed=seed)
        # 유니코드 U+2212 대신 ASCII 하이픈 — Malgun Gothic 에 U+2212 글리프가 없다.
        labs.append(f"{_backend_of(k)}\n- {_backend_of(ref_k)}")
        mus.append(d.mean)
        los2.append(d.mean - d.lo if np.isfinite(d.lo) else 0.0)
        his2.append(d.hi - d.mean if np.isfinite(d.hi) else 0.0)
    ax2.axhline(0, color="#C62828", ls="--", lw=1.0)
    ax2.errorbar(range(len(labs)), mus, yerr=[los2, his2], fmt="o", ms=7,
                 color="#2E5E8E", capsize=4, elinewidth=1.1)
    ax2.set_xticks(range(len(labs))); ax2.set_xticklabels(labs, fontsize=8)
    ax2.set_xlim(-0.6, len(labs) - 0.4)
    ax2.set_ylabel(f"Δ {METRIC_KO.get(metric, metric)}")
    ax2.set_title("② 대응차이 (0 포함 = 차이 미확인)", fontsize=9, loc="left")

    for a in (ax1, ax2):
        a.grid(axis="y", alpha=0.25, linewidth=0.5); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_llm_contribution(df, metric: str = "capture_rate", *, seed: int = 0,
                         figsize=(6.6, 3.4)):
    """**LLM 지휘관을 넣어서 좋아졌는가** — 배정 계층의 순수 기여만 뽑는다.

    기동 계층을 고정한 채 배정만 바꾼 두 쌍의 대응차이를 나란히 그린다:
        휴리스틱 기동에서:  llm_heur − heur_heur
        U-Net 기동에서  :  llm_unet − heur_unet
    두 막대가 모두 0 위에 있고 CI 가 0 을 배제해야 "LLM 이 기여했다"고 쓸 수 있다.
    지휘관이 여러 개면 지휘관마다 한 쌍씩 그린다.
    """
    piv = S.pivot_by_seed(df, metric)
    tags = backends_in(df)
    if not tags:
        raise ValueError("LLM 조건이 없다 (--skip-llm 으로 돌렸다면 이 그림은 못 만든다)")

    labels, mus, los, his, cols, sigs = [], [], [], [], [], []
    for t in tags:
        for man, lk, hk in (("휴리스틱 기동", f"llm_heur@{t}", "heur_heur"),
                            ("U-Net 기동",   f"llm_unet@{t}", "heur_unet")):
            if lk not in piv or hk not in piv:
                continue
            d = S.paired_diff(piv[lk].to_numpy(), piv[hk].to_numpy(), seed=seed)
            labels.append(f"{t}\n{man}")
            mus.append(d.mean)
            los.append(d.mean - d.lo if np.isfinite(d.lo) else 0.0)
            his.append(d.hi - d.mean if np.isfinite(d.hi) else 0.0)
            cols.append(BACKEND_TINT.get(t) or ("#2E5E8E" if "U-Net" in man else "#ED9A4B"))
            sigs.append(d.significant)
    if not labels:
        raise ValueError("배정 기여를 계산할 조건 쌍이 없다")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="#C62828", ls="--", lw=1.0, zorder=1)
    ax.bar(x, mus, 0.6, yerr=[los, his], capsize=3, color=cols, edgecolor="white",
           linewidth=0.6, zorder=2, error_kw={"elinewidth": 0.9})
    for i, (v, s) in enumerate(zip(mus, sigs)):
        if s:
            ax.text(i, v + (his[i] if v >= 0 else -los[i]), "*", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    # ASCII 하이픈 — Malgun Gothic 에 U+2212(MINUS SIGN) 글리프가 없어 경고가 뜬다.
    ax.set_ylabel(f"Δ {METRIC_KO.get(metric, metric)}  (LLM 배정 - 휴리스틱 배정)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("LLM 지휘관의 순수 기여 — 기동 계층 고정, 배정만 교체\n"
                 "(* = CI가 0을 배제. 0 근처면 배정에서 LLM 이득이 확인되지 않은 것)",
                 fontsize=9, loc="left")
    fig.tight_layout()
    return fig


def fig_llm_quality_ci(llm_df, *, by: str = "backend", seed: int = 0, figsize=(7.4, 3.6)):
    """지휘관별 계획 품질을 **CI 와 함께**. 평균 막대만으로는 유의성을 못 읽는다.

    지표마다 스케일이 달라 패널을 나누고, 각 패널 제목에 개선 방향을 적는다.
    `heuristic_agreement` 는 방향이 없다 — 1.0 이면 LLM 이 불필요하고 0.0 이면 통제 불능이라
    중간이 정상이다. 그래서 '방향 없음'으로 표시한다.
    """
    tab = S.compare_llm(llm_df, by=by, seed=seed)
    if tab.empty:
        raise ValueError("비교할 LLM 지표가 없다")
    ms = [m for m, _ in S.LLM_METRICS if m in set(tab["metric"])]
    groups = sorted(tab[by].dropna().unique())
    ncol = 4
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0], figsize[1] * nrow / 2))
    axes = np.atleast_1d(axes).ravel()
    dirmap = dict(S.LLM_METRICS)
    for ax, m in zip(axes, ms):
        sub = tab[tab["metric"] == m].set_index(by).reindex(groups)
        mu = sub["mean"].to_numpy(float)
        lo = np.where(np.isfinite(sub["lo"]), mu - sub["lo"], 0.0)
        hi = np.where(np.isfinite(sub["hi"]), sub["hi"] - mu, 0.0)
        ax.bar(range(len(groups)), mu, 0.6, yerr=[lo, hi], capsize=3,
               color=[BACKEND_TINT.get(g) or "#9E9E9E" for g in groups],
               edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.9})
        d = dirmap.get(m)
        arrow = {"lower": " ↓", "higher": " ↑"}.get(d, " (방향 없음)")
        ax.set_title(LLM_METRIC_KO.get(m, m) + arrow, fontsize=8, loc="left")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=6.5, rotation=12)
        ax.tick_params(axis="y", labelsize=7)
        if m == "latency_s":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(ms):]:
        ax.set_visible(False)
    fig.suptitle("지휘관별 계획 품질 (오차막대: 부트스트랩 95% CI)", fontsize=9.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_llm_backend_metrics(llm_df, *, figsize=(6.8, 3.0)):
    """백엔드별 지휘관 계측 비교: 지연 · 폴백률 · 계획 품질.

    성능이 같다면 **지연과 비용**이 선택 근거가 된다 — 그 근거를 그림으로 만든다.
    """
    if "backend" not in llm_df.columns or llm_df["backend"].nunique() < 2:
        raise ValueError("llm_metrics 에 backend 컬럼이 2종 이상 있어야 한다")
    bks = sorted(llm_df["backend"].dropna().unique())
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    data = [llm_df.loc[llm_df["backend"] == b, "latency_s"].dropna().to_numpy() for b in bks]
    data = [d for d in data if d.size]
    if data:
        bp = ax.boxplot(data, labels=bks[:len(data)], patch_artist=True, widths=0.55)
        for patch, b in zip(bp["boxes"], bks):
            patch.set_facecolor(BACKEND_TINT.get(b) or "#9E9E9E"); patch.set_alpha(0.65)
        ax.set_yscale("log")
        # ★ 로그축 기본 눈금은 mathtext 로 10^{-1} 을 그리는데, 그 지수의 U+2212 가
        #   한글 폰트에 없어 두부(□)로 찍히고 stderr 경고가 난다
        #   (axes.unicode_minus 는 mathtext 에 적용되지 않는다).
        #   지연 축은 0.1 / 1 / 10 처럼 평문 숫자가 논문에서도 읽기 쉬우므로 바꾼다.
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("응답 지연 (s, 로그축)")
    ax.set_title("① 지연 분포", fontsize=9, loc="left")

    ax = axes[1]
    fb = [float(llm_df.loc[llm_df["backend"] == b, "fallback"].mean()) for b in bks]
    ax.bar(bks, fb, 0.55, color=[BACKEND_TINT.get(b) or "#9E9E9E" for b in bks],
           edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.05); ax.set_ylabel("폴백률")
    ax.set_title("② 실패율 (낮을수록 좋음)", fontsize=9, loc="left")

    ax = axes[2]
    qk = [("coverage", "커버리지"), ("churn", "churn"), ("heuristic_agreement", "일치율")]
    qk = [(k, n) for k, n in qk if k in llm_df.columns]
    w = 0.8 / max(len(bks), 1)
    for i, b in enumerate(bks):
        sub = llm_df[llm_df["backend"] == b]
        v = [float(sub[k].mean(skipna=True)) for k, _ in qk]
        ax.bar(np.arange(len(qk)) + (i - (len(bks) - 1) / 2) * w, v, w * 0.9,
               color=BACKEND_TINT.get(b) or "#9E9E9E", edgecolor="white",
               linewidth=0.6, label=b)
    ax.set_xticks(range(len(qk))); ax.set_xticklabels([n for _, n in qk], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("③ 계획 품질", fontsize=9, loc="left")

    for a in axes:
        a.grid(axis="y", alpha=0.25, linewidth=0.5); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ── Fig I2. 핵심 5지표 — 위 2 / 아래 3 대칭 배치 ───────────────────────────
#: 발표·본문용 축약본. 교전 품질·기동 비용은 부록(figI)으로 미루고,
#  "얼마나 막았나(방어) · 얼마나 안전했나(안전) · 얼마나 아꼈나(자원)"만 남긴다.
# -- Fig M. 지휘관 모델만의 비교 --------------------------------------------
#: 조건(2x2)이 아니라 **모델 축**으로 자른다. 2x2 그림들이 "계층이 기여하는가"를
#  묻는다면 이쪽은 "어느 지휘관을 쓸 것인가"를 묻는다 - 논문에서 서로 다른 질문이다.
def _model_capture(df, model: str, metric: str, formation, *, maneuver: str = "heur"):
    """(모델, 포메이션)의 대응표본 벡터와 baseline 벡터. 짝이 안 맞으면 (None, None)."""
    piv = S.pivot_by_seed(df, metric, formation=formation)
    col = "llm_%s@%s" % (maneuver, model)
    if col not in piv.columns or "heur_heur" not in piv.columns:
        return None, None
    sub = piv[["heur_heur", col]].dropna()
    if len(sub) < 3:
        return None, None
    return sub[col].to_numpy(float), sub["heur_heur"].to_numpy(float)


def model_table(df, llm_df=None, *, metric: str = "capture_rate", seed: int = 0,
                maneuver: str = "heur"):
    """지휘관 모델별 한 줄 요약표(DataFrame).

    `maneuver="heur"` 가 기본이다 - 기동 계층을 휴리스틱으로 고정해야 **배정 품질만**
    남는다. U-Net 기동과 섞으면 두 계층의 효과가 뒤엉켜 모델 비교가 흐려진다.

    포메이션별 성능 + 전체 대응차이(vs heur_heur) + LLM 계측(지연/폴백/품질).
    """
    import pandas as pd
    forms = sorted(df["formation"].unique())
    rows = []
    for m in models_in(df):
        r = {"model": model_label(m), "tag": m}
        for f in forms:
            x, b = _model_capture(df, m, metric, f, maneuver=maneuver)
            r[f] = float(np.mean(x)) if x is not None else np.nan
            r[f + "_base"] = float(np.mean(b)) if b is not None else np.nan
        x, b = _model_capture(df, m, metric, None, maneuver=maneuver)
        if x is not None:
            e = S.paired_diff(x, b, seed=seed)
            r.update({"ALL": float(np.mean(x)), "d_mean": e.mean, "d_lo": e.lo,
                      "d_hi": e.hi, "dz": S.cohens_dz(x, b), "n": int(len(x)),
                      "sig": bool(e.lo > 0 or e.hi < 0)})
        else:
            r.update({"ALL": np.nan, "d_mean": np.nan, "d_lo": np.nan,
                      "d_hi": np.nan, "dz": np.nan, "n": 0, "sig": False})
        if llm_df is not None and len(llm_df) and "model" in llm_df.columns:
            q = llm_df[llm_df["model"] == m]
            if len(q):
                lat = pd.to_numeric(q["latency_s"], errors="coerce").dropna()
                r["latency_p50"] = float(lat.median()) if len(lat) else np.nan
                r["latency_p95"] = float(lat.quantile(0.95)) if len(lat) else np.nan
                for k in ("fallback", "coverage", "churn", "crossings",
                          "heuristic_agreement"):
                    if k in q.columns:
                        r[k] = float(pd.to_numeric(q[k], errors="coerce").mean(skipna=True))
                r["n_calls"] = int(len(q))
        rows.append(r)
    return pd.DataFrame(rows)


def model_pairwise(df, *, metric: str = "capture_rate", seed: int = 0,
                   maneuver: str = "heur"):
    """모델 두 개씩 **같은 시드로 짝지어** 비교. baseline 을 거치지 않은 직접 대조다."""
    import itertools
    import pandas as pd
    piv = S.pivot_by_seed(df, metric, formation=None)
    rows = []
    for a, b in itertools.combinations(models_in(df), 2):
        ca = "llm_%s@%s" % (maneuver, a)
        cb = "llm_%s@%s" % (maneuver, b)
        if ca not in piv.columns or cb not in piv.columns:
            continue
        sub = piv[[ca, cb]].dropna()
        if len(sub) < 3:
            continue
        # b - a : MODEL_ORDER 가 능력 순이므로 "능력을 키웠을 때의 증분"이 된다.
        x, y = sub[cb].to_numpy(float), sub[ca].to_numpy(float)
        e = S.paired_diff(x, y, seed=seed)
        rows.append({"a": model_label(a), "b": model_label(b),
                     "mean_a": float(y.mean()), "mean_b": float(x.mean()),
                     "d_mean": e.mean, "d_lo": e.lo, "d_hi": e.hi,
                     "dz": S.cohens_dz(x, y), "n": int(len(sub)),
                     "sig": bool(e.lo > 0 or e.hi < 0)})
    return pd.DataFrame(rows)


def fig_model_compare(df, llm_df=None, *, metric: str = "capture_rate", seed: int = 0,
                      maneuver: str = "heur", figsize=(7.8, 5.4)):
    """지휘관 모델 비교 4패널: 포메이션별 성능 / baseline 대비 / 지연 / 계획 품질.

    x 축은 MODEL_ORDER(능력 순)로 고정한다 - 사다리가 눈에 보여야 한다.
    """
    ms = models_in(df)
    if len(ms) < 2:
        raise ValueError("모델 비교에는 2종 이상이 필요하다. 있는 것: %s" % ms)
    forms = sorted(df["formation"].unique())
    t = model_table(df, llm_df, metric=metric, seed=seed, maneuver=maneuver)
    t = t[t["tag"].isin(ms)]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    cols = [BACKEND_TINT.get(m) or "#9E9E9E" for m in ms]
    xs = np.arange(len(ms))
    names = [safe_text(model_label(m)) for m in ms]

    def _v(m, k):
        v = t.loc[t["tag"] == m, k]
        return float(v.iloc[0]) if len(v) and v.notna().iloc[0] else np.nan

    # (1) 포메이션별 성능 + 휴리스틱 기준선
    ax = axes[0]
    w = 0.8 / max(len(forms), 1)
    for i, f in enumerate(forms):
        bars = ax.bar(xs + (i - (len(forms) - 1) / 2) * w, [_v(m, f) for m in ms],
                      w * 0.9, label=safe_text(FORM_KO.get(f, f)), edgecolor="white", linewidth=0.6)
        # 기준선은 **막대와 같은 색**으로 긋는다. 회색 하나로 그으면 어느 포메이션의
        # 기준인지 알 수 없어 "넘었나 못 넘었나"를 읽을 수 없다.
        b = np.nanmean([_v(m, f + "_base") for m in ms])
        if np.isfinite(b):
            ax.axhline(b, ls=":", lw=1.0, color=bars[0].get_facecolor(),
                       alpha=0.9, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7, rotation=12)
    ax.set_ylabel(safe_text(metric_label(metric)))
    ax.set_ylim(0, 1.28)          # 범례가 막대를 덮지 않도록 위를 비운다
    ax.legend(fontsize=6.5, frameon=False, ncol=len(forms), loc="upper center",
              borderaxespad=0.2)
    ax.set_title(safe_text("(1) 포메이션별 (점선 = 같은 색의 휴리스틱 기준)"),
                 fontsize=9, loc="left")

    # (2) baseline 대비 대응차이 + CI
    ax = axes[1]
    dm = np.array([_v(m, "d_mean") for m in ms])
    lo = np.array([_v(m, "d_lo") for m in ms])
    hi = np.array([_v(m, "d_hi") for m in ms])
    ax.bar(xs, dm, 0.55, color=cols, edgecolor="white", linewidth=0.6)
    ax.errorbar(xs, dm, yerr=np.abs(np.vstack([dm - lo, hi - dm])), fmt="none",
                ecolor="#333333", capsize=3, lw=1)
    ax.axhline(0, color="#333333", lw=0.9)
    for i, m in enumerate(ms):
        sig = t.loc[t["tag"] == m, "sig"]
        if len(sig) and bool(sig.iloc[0]) and np.isfinite(hi[i]):
            ax.annotate("*", (xs[i], hi[i]), ha="center", va="bottom", fontsize=11)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7, rotation=12)
    ax.set_ylabel(safe_text("vs 휴리스틱 배정"), fontsize=8.5, labelpad=1)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.margins(y=0.18)            # * 표시가 축 밖으로 나가지 않게
    ax.set_title(safe_text("(2) 대응차이 (* = CI가 0 배제)"), fontsize=9, loc="left")

    # (3) 지연
    ax = axes[2]
    if "latency_p50" in t.columns and t["latency_p50"].notna().any():
        ax.bar(xs - 0.18, [_v(m, "latency_p50") for m in ms], 0.34, color=cols,
               edgecolor="white", linewidth=0.6, label="p50")
        ax.bar(xs + 0.18, [_v(m, "latency_p95") for m in ms], 0.34, color=cols,
               edgecolor="white", linewidth=0.6, alpha=0.5, label="p95")
        ax.set_ylabel(safe_text("응답 지연 (s)"))
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, safe_text("LLM 계측 없음"), ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7, rotation=12)
    ax.set_title(safe_text("(3) 응답 지연 (낮을수록 좋음)"), fontsize=9, loc="left")

    # (4) 계획 품질
    ax = axes[3]
    qk = [("coverage", "커버리지"), ("churn", "churn"),
          ("heuristic_agreement", "일치율")]
    qk = [(k, n) for k, n in qk if k in t.columns and t[k].notna().any()]
    if qk:
        w2 = 0.8 / max(len(ms), 1)
        for i, m in enumerate(ms):
            ax.bar(np.arange(len(qk)) + (i - (len(ms) - 1) / 2) * w2,
                   [_v(m, k) for k, _ in qk], w2 * 0.9,
                   color=BACKEND_TINT.get(m) or "#9E9E9E", edgecolor="white",
                   linewidth=0.6, label=safe_text(model_label(m)))
        ax.set_xticks(range(len(qk)))
        ax.set_xticklabels([safe_text(n) for _, n in qk], fontsize=8)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5, frameon=False)
    else:
        ax.text(0.5, 0.5, safe_text("LLM 계측 없음"), ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_title(safe_text("(4) 계획 품질"), fontsize=9, loc="left")

    for a in axes:
        a.grid(axis="y", alpha=0.25, linewidth=0.5); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(safe_text("지휘관 모델 비교 (기동 계층 = %s 고정)" % maneuver),
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    return fig


def fig_model_ladder(df, *, metric: str = "capture_rate", seed: int = 0,
                     maneuver: str = "heur", figsize=(6.4, 3.6)):
    """능력 사다리 - 포메이션마다 모델 순서대로 선을 잇는다.

    막대보다 이쪽이 "능력이 오르면 성능이 오른다"를 직접 보여준다. 휴리스틱 수평선을
    같이 그려 **어디서 교차하는지**가 논문의 주장이 된다.
    """
    ms = models_in(df)
    if len(ms) < 2:
        raise ValueError("사다리 그림에는 모델 2종 이상이 필요하다. 있는 것: %s" % ms)
    forms = sorted(df["formation"].unique())
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(ms))
    marks = ("o", "s", "^", "D", "v")
    for i, f in enumerate(forms):
        ys, base = [], np.nan
        for m in ms:
            x, b = _model_capture(df, m, metric, f, maneuver=maneuver)
            ys.append(float(np.mean(x)) if x is not None else np.nan)
            if b is not None:
                base = float(np.mean(b))
        line, = ax.plot(xs, ys, marker=marks[i % len(marks)], lw=1.6, ms=5,
                        label=safe_text(FORM_KO.get(f, f)))
        if np.isfinite(base):
            ax.axhline(base, ls=":", lw=1.0, color=line.get_color(), alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([safe_text(model_label(m)) for m in ms], fontsize=8)
    ax.set_xlabel(safe_text("지휘관 능력 (좌 -> 우)"))
    ax.set_ylabel(safe_text(metric_label(metric)))
    ax.set_title(safe_text("능력 사다리 (점선 = 같은 색 포메이션의 휴리스틱 배정)"),
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


# -- figN: 계획 품질 -> 성능 (메커니즘) --------------------------------------
#: 계획 품질 지표와 방향. (지표, 낮을수록 좋은가, 라벨)
PLAN_QUALITY = (
    ("coverage",  False, "커버리지\n(배정된 클러스터 비율)"),
    ("churn",     True,  "churn\n(재계획마다 타겟 바꾼 배)"),
    ("crossings", True,  "경로 교차\n(두 배의 요격경로 교차)"),
)


def fig_plan_quality(df, llm_df, *, metric: str = "capture_rate", seed: int = 0,
                     maneuver: str = "heur", figsize=(7.6, 3.1)):
    """계획 품질이 성능을 예측하는가 — 지휘관 축으로 나란히 본다.

    왜 이 그림이 필요한가
        모델 비교 그림(figM*)은 **순위**만 준다. "어느 모델이 낫다"는 모델이 바뀌면
        낡는다. 이 그림은 **무엇이 성능을 만드는가**에 답한다.

        핵심은 패널 간 대조다: coverage(막을 클러스터를 실제로 막았나)는 성능을
        예측하지 못하고, churn(계획이 얼마나 흔들리나)·crossings 는 단조로 따라간다.
        7B 는 커버리지가 가장 높은데 성능이 가장 나쁘다 — 안 막아서 지는 게 아니라
        **반쯤 깐 그물을 버리고 타겟을 갈아타서** 진다.

    ★ 회귀선·상관계수를 얹지 않는다. 모델이 3종뿐이라 n=3 이고, 거기에 상관계수를
      찍으면 없는 통계적 근거를 주장하는 셈이 된다. 이 그림은 기술통계다.
    """
    if llm_df is None or len(llm_df) == 0:
        raise ValueError("계획 품질 그림에는 llm_metrics 가 필요하다")
    col = "model" if "model" in llm_df.columns else "backend"
    have = set(llm_df[col].astype(str))
    ms = [m for m in models_in(df) if m in have]
    if len(ms) < 2:
        raise ValueError("계획품질 그림에는 모델 2종 이상이 필요하다. 있는 것: %s" % ms)

    # 성능은 기동 계층을 고정해 지휘관 축만 남긴다 — figM* 와 같은 규약.
    perf = []
    for m in ms:
        x, _ = _model_capture(df, m, metric, None, maneuver=maneuver)
        perf.append(float(np.mean(x)) if x is not None else np.nan)

    fig, axes = plt.subplots(1, len(PLAN_QUALITY), figsize=figsize)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(ms))
    for ax, (q, lower_better, lab) in zip(axes, PLAN_QUALITY):
        vals, los, his = [], [], []
        for m in ms:
            g = llm_df[llm_df[col].astype(str) == m]
            v = g[q].astype(float).dropna().to_numpy() if q in g.columns else np.array([])
            if v.size == 0:
                vals.append(np.nan); los.append(0.0); his.append(0.0); continue
            est = S.mean_ci(v, seed=seed)
            vals.append(est.mean)
            los.append(max(0.0, est.mean - est.lo))
            his.append(max(0.0, est.hi - est.mean))
        cols = [BACKEND_TINT.get(_base_model(m)) or "#777777" for m in ms]
        ax.bar(xs, vals, 0.62, yerr=np.vstack([los, his]), color=cols,
               edgecolor="white", linewidth=0.6, capsize=2.5,
               error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
        arrow = " (낮을수록 좋음)" if lower_better else " (높을수록 좋음)"
        ax.set_title(safe_text(lab + arrow), fontsize=7.8, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels([safe_text(model_label(_base_model(m))) for m in ms],
                           fontsize=6.8, rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # 성능을 같은 패널에 겹친다 — 눈이 두 축을 따로 훑지 않아도 관계가 보인다.
        ax2 = ax.twinx()
        ax2.plot(xs, perf, marker="o", ms=5, lw=1.5, color="#C62828", zorder=5)
        ax2.set_ylim(0.0, 1.05)
        ax2.tick_params(axis="y", labelsize=6.5, colors="#C62828")
        for sp in ("top", "left"):
            ax2.spines[sp].set_visible(False)
        ax2.spines["right"].set_color("#C62828")
    fig.suptitle(safe_text("계획 품질 -> 성능 (붉은 선 = " + metric_label(metric)
                           + "): 커버리지는 예측하지 못하고, 계획 안정성이 예측한다"),
                 fontsize=9.2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# -- figO: 배정 권한(assign_mode) ablation ------------------------------------
def fig_assign_mode(df, *, metric: str = "capture_rate", seed: int = 0,
                    maneuver: str = "heur", baseline: str = "heur_heur",
                    figsize=(6.8, 3.6)):
    """배정 권한 ablation — 코드 가드레일이 약한 모델을 구제하는가.

    본 조건은 mode="llm"(코드 보완 일절 없음)이다. 이 선택은 약한 모델에 불리하게
    작용하므로 "일부러 불리하게 두지 않았나"라는 반론이 반드시 나온다. hybrid
    (LLM 명시분은 잠그고 빈 클러스터만 코드가 채움)를 **같은 시드**로 돌려 직접 답한다.

    읽는 법
        · 회복하면    -> 코드 가드레일이 작은 로컬 모델을 실용 가능하게 만든다(설계 기여)
        · 회복 안 하면 -> 커버리지를 메워도 안 되므로 churn 이 진범임이 확정된다(figN 강화)

    ★ 대응표본이므로 두 모드가 **같은 시드**를 가진 행만 쓴다(dropna). 시드가 어긋난
      모델을 섞으면 차이에 시나리오 난이도가 실려 비교가 무효가 된다.
    """
    piv = S.pivot_by_seed(df, metric)
    base = "llm_" + maneuver
    pairs = []
    for m in models_in(df, mode="llm"):
        a, b = base + "@" + m, base + "@" + m + "+hyb"
        if a in piv.columns and b in piv.columns:
            ok = piv[[a, b]].dropna()
            if len(ok) >= 3:
                pairs.append((m, ok[a].to_numpy(), ok[b].to_numpy()))
    if not pairs:
        raise ValueError("배정모드 ablation 에는 같은 모델의 llm/hybrid 조건이 둘 다 "
                         "필요하다 (results/eval_hybrid 수집이 끝나야 한다)")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    xs = np.arange(len(pairs))
    w = 0.36
    for j, (lab, hatch, colr) in enumerate((("순수 LLM", "", "#8FA8BF"),
                                            ("LLM+코드보완", "//", "#2E5E8E"))):
        vals, los, his = [], [], []
        for _, a, b in pairs:
            v = a if j == 0 else b
            est = S.mean_ci(v, seed=seed)
            vals.append(est.mean)
            los.append(max(0.0, est.mean - est.lo))
            his.append(max(0.0, est.hi - est.mean))
        axL.bar(xs + (j - 0.5) * w, vals, w, yerr=np.vstack([los, his]),
                label=safe_text(lab), hatch=hatch, capsize=2.5, color=colr,
                edgecolor="white", linewidth=0.6,
                error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
    if baseline in piv.columns:
        bm = float(np.nanmean(piv[baseline].to_numpy()))
        axL.axhline(bm, ls="--", lw=1.1, color="#C62828")
        axL.text(len(pairs) - 0.45, bm, safe_text(" 휴리스틱 baseline"), fontsize=7,
                 color="#C62828", va="bottom", ha="right")
    axL.set_xticks(xs)
    axL.set_xticklabels([safe_text(model_label(m)) for m, _, _ in pairs],
                        fontsize=7.5, rotation=12, ha="right")
    axL.set_ylabel(safe_text(metric_label(metric)))
    axL.set_title(safe_text("배정 권한별 성능"), fontsize=9, loc="left")
    axL.legend(fontsize=7, frameon=False, loc="lower right")

    # 오른쪽: 대응차이(hybrid - llm) + CI. 0 을 지나면 '구제 못 함'이다.
    for i, (m, a, b) in enumerate(pairs):
        est = S.paired_diff(b, a, seed=seed)
        c = "#2E7D32" if est.lo > 0 else ("#C62828" if est.hi < 0 else "#777777")
        axR.errorbar(est.mean, i,
                     xerr=[[max(0.0, est.mean - est.lo)], [max(0.0, est.hi - est.mean)]],
                     fmt="o", ms=5, color=c, elinewidth=1.3, capsize=3)
    axR.axvline(0, ls="--", lw=1.0, color="#555555")
    axR.set_yticks(xs)
    axR.set_yticklabels([safe_text(model_label(m)) for m, _, _ in pairs], fontsize=7.5)
    axR.set_ylim(-0.6, len(pairs) - 0.4)
    axR.set_xlabel(safe_text("코드보완 - 순수 LLM (대응차이, 95% CI)"), fontsize=8)
    axR.set_title(safe_text("구제 효과 (CI 가 0 을 지나면 구제 실패)"), fontsize=9, loc="left")
    for ax in (axL, axR):
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def model_table_tex(df, llm_df=None, *, metric: str = "capture_rate",
                    seed: int = 0, maneuver: str = "heur") -> str:
    """model_table 을 논문 표(LaTeX)로. 손으로 옮기지 말 것."""
    t = model_table(df, llm_df, metric=metric, seed=seed, maneuver=maneuver)
    forms = [c for c in t.columns if c + "_base" in t.columns]
    head = " & ".join(["지휘관"] + forms + ["전체", r"$\Delta$ vs 휴리스틱", "$d_z$",
                                            "지연 p50", "폴백"])
    out = [r"\begin{table}[t]\centering",
           r"\caption{지휘관 모델별 비교 (기동 계층 = %s 고정, 평균 및 부트스트랩 95\%% CI). "
           r"$^{*}$ = 대응차이의 CI가 0을 배제}" % maneuver,
           r"\begin{tabular}{l" + "r" * (len(forms) + 5) + "}", r"\hline",
           head + r" \\", r"\hline"]
    for _, r in t.iterrows():
        star = r"$^{*}$" if r.get("sig") else ""
        cells = [str(r["model"])]
        cells += ["%.3f" % r[f] if np.isfinite(r[f]) else "--" for f in forms]
        cells.append("%.3f" % r["ALL"] if np.isfinite(r["ALL"]) else "--")
        cells.append(("%+.3f [%+.3f, %+.3f]%s" % (r["d_mean"], r["d_lo"], r["d_hi"], star))
                     if np.isfinite(r["d_mean"]) else "--")
        cells.append("%+.2f" % r["dz"] if np.isfinite(r["dz"]) else "--")
        cells.append("%.2f s" % r["latency_p50"]
                     if np.isfinite(r.get("latency_p50", np.nan)) else "--")
        cells.append("%.4f" % r["fallback"]
                     if np.isfinite(r.get("fallback", np.nan)) else "--")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


CORE5 = (
    ("capture_rate", "방어 성과"),
    ("breaches", "방어 성과"),
    ("collision_rate", "안전"),
    ("net_touches", "안전"),
    ("nets_per_capture", "자원 효율"),
)


def fig_core_panels(df, *, metrics=CORE5, seed: int = 0, figsize=(7.6, 5.4)):
    """핵심 지표 5개를 **위 2 / 아래 3** 으로 대칭 배치.

    2행 6열 격자에 폭 3짜리 2개(위)와 폭 2짜리 3개(아래)를 얹으면 두 행의 총 폭이
    같아져 좌우 여백 없이 가운데 정렬된다. (subplots 로는 이 배치가 안 나온다.)
    """
    keys = [(m, g) for m, g in metrics if m in df.columns]
    if len(keys) != 5:
        raise ValueError(f"5개 지표가 필요하다. 사용 가능: {[m for m, _ in keys]}")
    conds = _conds(df)

    fig = plt.figure(figsize=figsize)
    # gridspec 으로 폭이 다른 행을 만들면 tight_layout 이 맞지 않는다(경고 후 배치가 틀어짐).
    #   → 여백을 여기서 직접 확정하고 tight_layout 은 쓰지 않는다.
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=0.62,
                          left=0.07, right=0.985, top=0.885, bottom=0.10)
    #   위: (0,0-2) (0,3-5)   아래: (1,0-1) (1,2-3) (1,4-5)
    axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]),
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
            fig.add_subplot(gs[1, 4:6])]

    for ax, (m, grp) in zip(axes, keys):
        piv = S.pivot_by_seed(df, m)
        means, los, his, cols = [], [], [], []
        for c in conds:
            e = (S.mean_ci(piv[c].to_numpy(), seed=seed) if c in piv
                 else S.Estimate(np.nan, np.nan, np.nan, 0))
            means.append(e.mean)
            los.append(e.mean - e.lo if np.isfinite(e.lo) else 0.0)
            his.append(e.hi - e.mean if np.isfinite(e.hi) else 0.0)
            cols.append(_color(c))
        ax.bar(range(len(conds)), means, 0.66, yerr=[los, his], capsize=2.5,
               color=cols, hatch=[_hatch(c) for c in conds],
               edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.9})
        arrow = "↓" if m in LOWER_BETTER else "↑"
        ax.set_title(f"{METRIC_KO.get(m, m)} {arrow}", fontsize=9.5, loc="left", pad=4)
        ax.set_xticks(range(len(conds)))
        # 아래 행(폭 2칸)은 위 행(폭 3칸)보다 좁아 같은 글씨면 라벨이 서로 붙는다.
        #   백엔드 태그를 괄호째 줄바꿈해 두 줄로 접고 글씨를 한 단계 줄인다.
        narrow = ax.get_subplotspec().colspan.stop - ax.get_subplotspec().colspan.start <= 2
        ax.set_xticklabels([_short(c) for c in conds], fontsize=5.6 if narrow else 6.4)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.995, 1.015, grp, transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom", color="#666")

    fig.suptitle("핵심 지표 — ↑ 높을수록 / ↓ 낮을수록 좋음 "
                 "(오차막대: 부트스트랩 95% CI)", fontsize=10, x=0.012, y=0.972,
                 ha="left")
    return fig


# ── Fig I4. 핵심 4지표 · 2×2 ───────────────────────────────────────────────
#: 충돌률을 뺀 4지표. 충돌률은 이 데이터에서 유일하게 제안이 불리하고 CI 도 넓어,
#  '유의하게 개선된 것만' 보이는 요약 그림에는 넣지 않는다(전체는 figI/figI3 에 있다).
CORE4 = (
    ("capture_rate", "방어 성과"),
    ("breaches", "방어 성과"),
    ("net_touches", "안전"),
    ("nets_per_capture", "자원 효율"),
)


def fig_core4(df, *, metrics=CORE4, seed: int = 0, lang: str = "en",
              baseline: str = "heur_heur", figsize=(7.2, 5.2), annotate: bool = True):
    """핵심 4지표를 2×2 로. 세로 막대, 조건 4개.

    2×2 는 지표 수와 격자가 정확히 맞아 빈 칸이 없다 — 5지표(figI2)에서 필요했던
    비대칭 배치가 여기서는 불필요하다.

    같은 4지표를 **세로로 쭉 내리는 가로막대**로 보려면 `fig_metric_barh(df,
    metrics=CORE4)` 를 쓴다.
    """
    keys = [(m, g) for m, g in metrics if m in df.columns]
    if len(keys) != 4:
        raise ValueError(f"4개 지표가 필요하다. 사용 가능: {[m for m, _ in keys]}")
    conds = _conds(df)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for ax, (m, grp) in zip(axes, keys):
        piv = S.pivot_by_seed(df, m)
        base = piv[baseline].to_numpy() if baseline in piv else None
        mus, los, his, cols, sig = [], [], [], [], []
        for c in conds:
            e = (S.mean_ci(piv[c].to_numpy(), seed=seed) if c in piv
                 else S.Estimate(np.nan, np.nan, np.nan, 0))
            mus.append(e.mean)
            los.append(e.mean - e.lo if np.isfinite(e.lo) else 0.0)
            his.append(e.hi - e.mean if np.isfinite(e.hi) else 0.0)
            cols.append(_color(c))
            sig.append(bool(base is not None and c != baseline and c in piv
                            and S.paired_diff(piv[c].to_numpy(), base,
                                              seed=seed).significant))
        x = np.arange(len(conds))
        ax.bar(x, mus, 0.66, yerr=[los, his], capsize=3, color=cols,
               hatch=[_hatch(c) for c in conds], edgecolor="white",
               linewidth=0.7, error_kw={"elinewidth": 1.0, "ecolor": "#333"})
        top = float(np.nanmax(np.add(mus, his)))
        if annotate:
            for xi, v, h, s in zip(x, mus, his, sig):
                if not np.isfinite(v):
                    continue
                ax.text(xi, v + h + top * 0.035, f"{v:.3g}" + ("*" if s else ""),
                        ha="center", va="bottom", fontsize=7, color="#222")
            ax.set_ylim(0, top * 1.22)

        arrow = "↓" if m in LOWER_BETTER else "↑"
        better = ("lower is better" if m in LOWER_BETTER else "higher is better") \
            if lang == "en" else ("낮을수록 좋음" if m in LOWER_BETTER else "높을수록 좋음")
        g_lab = GROUP_EN.get(grp, grp) if lang == "en" else grp
        ax.set_title(f"{metric_label(m, lang)}  {arrow}", fontsize=10, loc="left", pad=4)
        ax.text(1.0, 1.015, f"{g_lab} · {better}", transform=ax.transAxes,
                fontsize=6.8, ha="right", va="bottom", color="#666")
        ax.set_xticks(x)
        ax.set_xticklabels([cond_label(c, lang) for c in conds], fontsize=6.6)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", alpha=0.24, linewidth=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    title = ("Core metrics (error bars: bootstrap 95% CI; "
             "* paired difference vs baseline excludes 0)" if lang == "en"
             else "핵심 지표 (오차막대: 부트스트랩 95% CI; "
                  "* baseline 대비 대응차이가 0을 배제)")
    fig.suptitle(title, fontsize=9.5, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


# ── Fig I3. 가로 막대 · 지표를 위에서 아래로 나열 ──────────────────────────
def fig_metric_barh(df, *, metrics=CORE5, seed: int = 0, lang: str = "en",
                    baseline: str = "heur_heur", panel_h: float = 1.02,
                    width: float = 7.2, annotate: bool = True):
    """지표를 **위에서 아래로** 쌓고, 각 지표 안에서 조건을 **가로 막대**로 비교한다.

    지표마다 단위와 범위가 달라(포획률 0~1, 돌파 0~2, 충돌률 0~0.15) 한 축에 겹쳐 그릴
    수 없다. 그래서 지표당 패널 하나씩을 세로로 쌓되, 각 패널은 x 축을 따로 갖는다.
    조건 순서(위→아래 = baseline→제안)는 모든 패널에서 동일해 세로로 읽힌다.

    막대 끝의 숫자는 평균, 오차막대는 부트스트랩 95% CI, * 는 baseline 대비 대응차이의
    CI 가 0 을 배제함을 뜻한다.
    """
    keys = [(m, g) for m, g in metrics if m in df.columns]
    if not keys:
        raise ValueError("그릴 지표가 없다 — CSV 컬럼을 확인하라")
    conds = _conds(df)
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(width, panel_h * n + 0.9))
    axes = np.atleast_1d(axes)

    #: 위에서 아래로 baseline → 제안 순으로 읽히게 y 축을 뒤집는다.
    y = np.arange(len(conds))[::-1]

    for ax, (m, grp) in zip(axes, keys):
        piv = S.pivot_by_seed(df, m)
        base = piv[baseline].to_numpy() if baseline in piv else None
        mus, los, his, cols, sig = [], [], [], [], []
        for c in conds:
            e = (S.mean_ci(piv[c].to_numpy(), seed=seed) if c in piv
                 else S.Estimate(np.nan, np.nan, np.nan, 0))
            mus.append(e.mean)
            los.append(e.mean - e.lo if np.isfinite(e.lo) else 0.0)
            his.append(e.hi - e.mean if np.isfinite(e.hi) else 0.0)
            cols.append(_color(c))
            sig.append(bool(base is not None and c != baseline
                            and c in piv
                            and S.paired_diff(piv[c].to_numpy(), base,
                                              seed=seed).significant))
        ax.barh(y, mus, 0.68, xerr=[los, his], capsize=2.5, color=cols,
                hatch=[_hatch(c) for c in conds], edgecolor="white",
                linewidth=0.6, error_kw={"elinewidth": 0.9, "ecolor": "#333"})

        if annotate:
            span = max(np.nanmax(np.add(mus, his)), 1e-9)
            for yi, v, h, s in zip(y, mus, his, sig):
                if not np.isfinite(v):
                    continue
                txt = f"{v:.3g}" + ("*" if s else "")   # ASCII: U+2731 은 한글폰트에 없다
                ax.text(v + h + span * 0.02, yi, txt, va="center", ha="left",
                        fontsize=6.6, color="#222")
            ax.set_xlim(0, span * 1.26)     # 주석이 잘리지 않도록 오른쪽 여유

        arrow = "↓" if m in LOWER_BETTER else "↑"
        better = ("lower is better" if m in LOWER_BETTER else "higher is better") \
            if lang == "en" else ("낮을수록 좋음" if m in LOWER_BETTER else "높을수록 좋음")
        g_lab = GROUP_EN.get(grp, grp) if lang == "en" else grp
        ax.set_title(f"{metric_label(m, lang)}  {arrow}", fontsize=9, loc="left", pad=3)
        ax.text(1.0, 1.06, f"{g_lab} · {better}", transform=ax.transAxes,
                fontsize=6.6, ha="right", va="bottom", color="#666")
        ax.set_yticks(y)
        ax.set_yticklabels([cond_label(c, lang, newline=False) for c in conds],
                           fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.22, linewidth=0.5); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    title = ("Multi-metric comparison (error bars: bootstrap 95% CI; "
             "* paired difference vs baseline excludes 0)" if lang == "en"
             else "다지표 비교 (오차막대: 부트스트랩 95% CI; "
                  "* baseline 대비 대응차이가 0을 배제)")
    fig.suptitle(title, fontsize=9.5, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / (panel_h * n + 0.9)))
    return fig


# ── Fig J. 다지표 효과 forest (표준화 차이) ────────────────────────────────
def fig_multimetric_forest(df, *, treatment: str | None = None, baseline: str = "heur_heur",
                           groups=METRIC_GROUPS, seed: int = 0, figsize=(6.6, 5.2)):
    """제안 vs baseline 을 **모든 지표에서** 한 장에. 단위가 다르므로 효과크기(d_z)로 그린다.

    d_z 는 대응차이를 그 표준편차로 나눈 값이라 단위가 없어 지표끼리 겹쳐 볼 수 있다.
    부호는 **항상 '좋아진 방향이 +'** 가 되도록 낮을수록 좋은 지표에서 뒤집는다 —
    안 뒤집으면 같은 그림 안에서 +가 어떤 축은 개선, 어떤 축은 악화를 뜻해 읽을 수 없다.
    """
    conds = _conds(df)
    if treatment is None:
        cand = [c for c in conds if _base_key(c) == "llm_unet"] or \
               [c for c in conds if c != baseline]
        if not cand:
            raise ValueError("비교할 조건이 없다")
        treatment = cand[0]
    keys = [m for _, ms in groups for m in ms if m in df.columns]

    labels, dzs, cols, sigs = [], [], [], []
    for g, ms in groups:
        for m in ms:
            if m not in keys:
                continue
            piv = S.pivot_by_seed(df, m)
            if treatment not in piv or baseline not in piv:
                continue
            x, y = piv[treatment].to_numpy(), piv[baseline].to_numpy()
            dz = S.cohens_dz(x, y)
            d = S.paired_diff(x, y, seed=seed)
            if m in LOWER_BETTER:          # 좋아진 방향을 + 로 통일
                dz = -dz
            labels.append(f"{METRIC_KO.get(m, m).split(' (')[0]}  [{g}]")
            dzs.append(dz)
            sigs.append(d.significant)
            cols.append("#2E5E8E" if (np.isfinite(dz) and dz > 0) else "#B0653C")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(figsize[0], max(3.0, 0.34 * len(labels) + 1.2)))
    ax.axvline(0, color="#C62828", lw=1.0, ls="--", zorder=1)
    for ref in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):     # Cohen 관례선
        ax.axvline(ref, color="#DDD", lw=0.6, zorder=0)
    ax.barh(y, dzs, 0.62, color=cols, edgecolor="white", linewidth=0.5, zorder=2)
    for i, (v, s) in enumerate(zip(dzs, sigs)):
        if s and np.isfinite(v):
            ax.text(v + (0.05 if v >= 0 else -0.05), i, "*", va="center",
                    ha="left" if v >= 0 else "right", fontsize=11, color="#222")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("효과크기 $d_z$  (+ = 제안이 개선된 방향, * = CI가 0을 배제)")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{_short(treatment).replace(chr(10), ' ')} vs "
                 f"{_short(baseline).replace(chr(10), ' ')} — 전 지표 효과크기",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    return fig


def multimetric_table(df, *, groups=METRIC_GROUPS, seed: int = 0,
                      baseline: str = "heur_heur") -> str:
    """전 지표 × 전 조건 요약 LaTeX 표. 논문 결과 절의 본표.

    각 지표 행에 개선 방향(↑/↓)을 달아 표만 봐도 오독이 안 되게 한다.
    """
    conds = _conds(df)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{다지표 평가 결과 (평균 $\pm$ 부트스트랩 95\% CI 반폭). "
             r"$\uparrow$ 높을수록, $\downarrow$ 낮을수록 우수. "
             r"$^{*}$ = baseline 대비 대응차이의 CI가 0을 배제}",
             r"\begin{tabular}{ll" + "c" * len(conds) + "}", r"\hline",
             "관점 & 지표 & " + " & ".join(
                 _short(c).replace("\n", " ") for c in conds) + r" \\", r"\hline"]
    for g, ms in groups:
        first = True
        for m in ms:
            if m not in df.columns:
                continue
            piv = S.pivot_by_seed(df, m)
            arrow = r"$\downarrow$" if m in LOWER_BETTER else r"$\uparrow$"
            cells = []
            for c in conds:
                if c not in piv:
                    cells.append("---"); continue
                e = S.mean_ci(piv[c].to_numpy(), seed=seed)
                half = (e.hi - e.lo) / 2 if np.isfinite(e.lo) else np.nan
                txt = (f"{e.mean:.3g}$\\pm${half:.2g}" if np.isfinite(half)
                       else f"{e.mean:.3g}")
                if c != baseline and baseline in piv:
                    d = S.paired_diff(piv[c].to_numpy(), piv[baseline].to_numpy(), seed=seed)
                    if d.significant:
                        txt += r"$^{*}$"
                cells.append(txt)
            gcell = g if first else ""
            first = False
            lines.append(f"{gcell} & {METRIC_KO.get(m, m).split(' (')[0]} {arrow} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ── 일괄 저장 ──────────────────────────────────────────────────────────────
def save_all(df, outdir: str = "논문_그래프", *, llm_df=None,
             metric: str = "capture_rate", seed: int = 0, verbose: bool = True) -> list[str]:
    """가능한 그림을 전부 만든다. 조건이 모자라 못 그리는 그림은 건너뛰고 이유를 찍는다."""
    made: list[str] = []

    def _try(name, fn):
        try:
            made.extend(_save(fn(), outdir, name))
            if verbose:
                print(f"  [ok]   {name}")
        except Exception as e:
            if verbose:
                print(f"  [skip] {name}: {e}")

    bks = backends_in(df)
    _try("figA_condition_bars", lambda: fig_condition_bars(df, metric, seed=seed))
    _try("figA2_breaches", lambda: fig_condition_bars(df, "breaches", seed=seed))
    _try("figA3_collision_rate", lambda: fig_condition_bars(df, "collision_rate", seed=seed))
    _try("figA4_nets_per_capture",
         lambda: fig_condition_bars(df, "nets_per_capture", seed=seed))
    _try("figA5_traveled_per_capture",
         lambda: fig_condition_bars(df, "traveled_per_capture", seed=seed))
    _try("figA6_cap_time", lambda: fig_condition_bars(df, "cap_time_mean", seed=seed))
    _try("figA7_net_deploy_edist",
         lambda: fig_condition_bars(df, "net_deploy_edist", seed=seed))
    _try("figI_metric_panels", lambda: fig_metric_panels(df, seed=seed))
    _try("figI2_core_panels", lambda: fig_core_panels(df, seed=seed))
    _try("figI3_core_barh_en", lambda: fig_metric_barh(df, seed=seed, lang="en"))
    _try("figI4_core4_en", lambda: fig_core4(df, seed=seed, lang="en"))
    _try("figI5_core4_barh_en",
         lambda: fig_metric_barh(df, metrics=CORE4, seed=seed, lang="en",
                                 panel_h=1.15, width=7.4))
    _try("figJ_multimetric_forest", lambda: fig_multimetric_forest(df, seed=seed))
    _try("figB_paired_forest", lambda: fig_paired_forest(df, metric, seed=seed))
    # 상호작용: 백엔드가 여럿이면 백엔드마다 한 장
    if bks:
        for b in bks:
            _try(f"figC_interaction_{b}",
                 lambda b=b: fig_interaction(df, metric, seed=seed, backend=b))
    else:
        _try("figC_interaction", lambda: fig_interaction(df, metric, seed=seed))
    for b in (bks or [""]):
        tr = f"llm_unet@{b}" if b else "llm_unet"
        _try(f"figD_paired_scatter{('_' + b) if b else ''}",
             lambda tr=tr: fig_paired_scatter(df, metric, treatment=tr))
    _try("figE_tradeoff_radar", lambda: fig_tradeoff_radar(df))
    if len(bks) >= 2:
        _try("figH_backend_compare", lambda: fig_backend_compare(df, metric, seed=seed))
    if bks:
        _try("figK_llm_contribution", lambda: fig_llm_contribution(df, metric, seed=seed))
    if llm_df is not None and len(llm_df):
        _try("figG_llm_metrics", lambda: fig_llm_metrics(llm_df))
        # 모델 태그가 있으면 그쪽으로 가른다(같은 백엔드 안의 7b vs 14b 를 보기 위해).
        by = "model" if ("model" in llm_df.columns
                         and llm_df["model"].nunique() >= llm_df.get(
                             "backend", llm_df["model"]).nunique()) else "backend"
        _try("figK2_llm_quality_ci", lambda: fig_llm_quality_ci(llm_df, by=by, seed=seed))
        _try("figH2_llm_backend_metrics", lambda: fig_llm_backend_metrics(llm_df))
    # 모델 축 그림 — llm_df 가 없어도 성능 패널은 그려진다(계측 패널만 비운다).
    _try("figM_model_compare", lambda: fig_model_compare(df, llm_df, metric=metric,
                                                         seed=seed))
    _try("figM2_model_ladder", lambda: fig_model_ladder(df, metric=metric, seed=seed))
    # 메커니즘·ablation — 데이터가 모자라면 _try 가 [skip] 을 찍고 넘어간다.
    _try("figN_plan_quality",
         lambda: fig_plan_quality(df, llm_df, metric=metric, seed=seed))
    _try("figO_assign_mode", lambda: fig_assign_mode(df, metric=metric, seed=seed))
    return made


def paper_table(df, metric: str = "capture_rate", *, seed: int = 0) -> str:
    """논문 본표(LaTeX). summarize() 결과를 그대로 조판한다 — 손으로 옮기지 말 것."""
    import pandas as pd
    s = S.summarize(df, metric, seed=seed)
    s = s[s["formation"] == "ALL"]
    order = {c: i for i, c in enumerate(COND_ORDER)}
    s = s.sort_values("condition", key=lambda col: col.map(order))
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{METRIC_KO.get(metric, metric)} — 조건별 (평균과 부트스트랩 95\% CI, "
        rf"대응표본 차이는 baseline 대비)}}",
        r"\begin{tabular}{lccc}", r"\hline",
        r"조건 & 평균 [95\% CI] & $\Delta$ vs baseline & $d_z$ \\", r"\hline",
    ]
    for _, r in s.iterrows():
        lab = _short(r["condition"]).replace("\n", " ")
        mean = (f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
                if np.isfinite(r["lo"]) else f"{r['mean']:.3f}")
        if np.isfinite(r["d_mean"]):
            star = r"$^{*}$" if r["sig"] else ""
            dd = f"{r['d_mean']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]{star}"
            dz = f"{r['dz']:.2f}"
        else:
            dd, dz = "---", "---"
        lines.append(f"{lab} & {mean} & {dd} & {dz} \\\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\begin{flushleft}\footnotesize $^{*}$ CI가 0을 포함하지 않음."
              r"\end{flushleft}",
              r"\end{table}"]
    return "\n".join(lines)



