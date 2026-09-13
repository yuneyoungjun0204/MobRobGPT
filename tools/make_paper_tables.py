# -*- coding: utf-8 -*-
"""논문 표 생성기 — results/eval_merged/*.csv 에서 LaTeX 표 조각을 굽는다.

왜 스크립트인가
    논문_그래프/README 의 원칙과 같다: **숫자를 손으로 옮겨 적는 경로를 만들지 않는다.**
    본문은 \\input{tables/xxx} 만 하므로, 평가를 다시 돌리면 표가 자동으로 맞는다.
    손으로 옮기면 CSV 와 어긋나고, 어긋난 것을 아무도 알아채지 못한다.

사용:
    python tools/make_paper_tables.py
    python tools/make_paper_tables.py --src results/eval_merged --out 논문초안/tables
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# tools/ 에서 직접 실행하면 sys.path[0] 이 tools/ 라 boatattack_sim 을 못 찾는다.
# 리포 루트를 먼저 넣는다(어느 작업 디렉터리에서 불러도 동작하도록).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boatattack_sim.eval.stats as S  # noqa: E402

#: 본문에서 쓰는 지휘관. 능력 순서 고정 — 표의 행 순서가 곧 주장의 순서다.
MODELS = ("qwen2.5-7b", "qwen2.5-14b", "gemini-3.5-flash-lite")
MODEL_KO = {
    "qwen2.5-7b": "Qwen2.5 7B (로컬)",
    "qwen2.5-14b": "Qwen2.5 14B (로컬)",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite (클라우드)",
}
COND_KO = {
    "heur_heur": "휴리스틱 배정 + 휴리스틱 기동 (baseline)",
    "heur_unet": "휴리스틱 배정 + U-Net 점수맵",
    "llm_heur": "LLM 지휘관 + 휴리스틱 기동",
    "llm_unet": "LLM 지휘관 + U-Net 점수맵 (제안)",
}
FORM_KO = {"concentrated": "집중", "diversionary": "양동", "wave": "파상"}

#: 방향 화살표는 수식모드로 낸다 — 한글 폰트(DotumChe/HCR 계열)에 U+2191/U+2193
#  글리프가 없으면 XeTeX 이 경고 없이 빈칸을 찍는다. 표에서 방향이 사라지면
#  "낮을수록 좋음" 지표가 악화로 읽힌다.
UP, DN = r"$\uparrow$", r"$\downarrow$"

#: (지표, 한국어 이름, 방향, 소수 자릿수). 방향 = 개선이 어느 쪽인가.
#  ★ 이름·방향·자릿수의 정본은 boatattack_sim/eval/paper_names.py 다. 표와 그림이 같은
#    지표를 다른 이름으로 부르는 일을 막으려고 한 곳에서만 정의한다.
from boatattack_sim.eval import paper_names as _PN                      # noqa: E402

METRICS = tuple((k, ko, DN if lb else UP, nd) for k, ko, lb, nd in _PN.METRICS)


def _esc(s: str) -> str:
    """LaTeX 특수문자 이스케이프. 지표 이름에 _ 가 들어가면 수식모드로 새어 깨진다."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _sig(est) -> str:
    """CI 가 0 을 포함하지 않으면 별표. 포함하면 빈 칸 — '유의하지 않음'을 숨기지 않는다."""
    return r"$^{*}$" if (est.lo > 0 or est.hi < 0) else ""


#: 다중비교 보정 — 지표 10개에 대한 Bonferroni. alpha 0.05/10 → 99.5 % 구간.
#  왜 m=10 인가: 이 표가 한 조건에 대해 지표 10개를 동시에 검정하기 때문이다.
#  조건 수까지 곱하지 않는 이유는 조건별 비교가 각각 독립된 질문이기 때문이며,
#  이 선택을 캡션에 밝힌다(숨은 자유도를 남기지 않는다).
BONF_M = 10
BONF_LEVEL = 1.0 - 0.05 / BONF_M


def _sig_bonf(x, b) -> str:
    """Bonferroni 보정 수준에서도 CI 가 0 을 배제하면 단검표."""
    e = S.paired_diff(x, b, confidence_level=BONF_LEVEL)
    return r"$^{\dagger}$" if (e.lo > 0 or e.hi < 0) else ""


def _wrap(body: str, caption: str, label: str, colspec: str, header: str,
          note: str = "", wide: bool = False, tight: bool = False) -> str:
    """표 조각 하나. `wide=True` 면 table* (2단 조판에서 양단 걸침).

    ★ 본문이 2단이므로 열이 많은 표는 좁은 단에 안 들어간다. 이 선택을 여기 두는 이유:
      생성된 .tex 를 손으로 table* 로 고치면 다음 재생성에서 지워진다.
    """
    env = "table*" if wide else "table"
    # ★ 판면보다 넓을 때만 줄인다. 2단 양단걸침(≈170 mm)에 맞춘 표가 elsarticle
    #   preprint 판면(≈137 mm)에서는 넘친다. 조판 판형이 바뀌어도 표를 다시 짜지
    #   않도록, 들어가면 원래 크기 그대로 두고 넘칠 때만 축소하는 관용구를 쓴다.
    fit_a = "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
    fit_b = "}\n"
    # 열이 많은 표는 열 간격을 좁혀 판면에 넣는다. 축소(\resizebox)로 덮으면 글자가
    # 본문보다 작아지고 괘선까지 얇아진다 --- 심사자가 보는 판에서만 그렇게 된다.
    tc = "\\setlength{\\tabcolsep}{4pt}%\n" if tight else ""
    # threeparttable 의존을 피한다 — 표 폭에 맞춘 단순 문단으로 낸다.
    # (tablenotes 는 threeparttable 환경 밖에서 쓰면 컴파일 에러가 난다.)
    n = ("\\\\[3pt]\n\\begin{minipage}{\\linewidth}\\footnotesize\n"
         + note + "\n\\end{minipage}\n") if note else ""
    return (
        "% 자동 생성 — tools/make_paper_tables.py. 손으로 고치지 말 것.\n"
        f"\\begin{{{env}}}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\footnotesize\n"
        f"{tc}{fit_a}"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n"
        f"{fit_b}"
        f"{n}\\end{{{env}}}\n")


# ── 표 0: 시나리오 파라미터 (체크포인트 config 에서 직접) ───────────────────
#: (config 키, 한국어 이름, 단위, 포맷). None 키는 파생값 — 아래에서 따로 계산한다.
SCENARIO = (
    ("n_enemies",   "적 USV 수 $M$",            "척",   "{:.0f}"),
    ("n_allies",    "방어정 수 $P$",            "척",   "{:.0f}"),
    ("enemy_speed", "적 속력",                  "m/s",  "{:.1f}"),
    ("ally_speed",  "방어정 속력",              "m/s",  "{:.1f}"),
    ("world_size",  "교전 해역 한 변",          "m",    "{:.0f}"),
    ("net_max_len", "그물벽 최대 길이",         "m",    "{:.0f}"),
    ("cnn_extent",  "관측 격자 반폭",           "m",    "{:.0f}"),
    ("cnn_grid_n",  "관측 격자 해상도",         "칸",   "{:.0f}"),
    # ★ 무리 수는 $C$ 다. $K$ 는 경유점 수(식 2·3), $G$ 는 학습의 그룹 후보 수 —
    #   boatattack_sim/eval/paper_names.py 머리말과 본문 기호표(tab:nomenclature) 참조.
    ("n_clusters",  "적 무리 최대 수 $C$",      "개",   "{:.0f}"),
)


def tab_scenario(ckpt_path, out, nets_eval=3):
    """시나리오 표를 **학습 체크포인트 config 에서 직접** 굽는다.

    왜 손으로 안 쓰는가: 초안에는 그물벽 길이가 150 m 로 적혀 있었으나 실제 config 는
    450 m 였다(3배 오차). 이런 상수는 한 번 잘못 적히면 아무도 다시 확인하지 않는다.

    ★ nets_per_ship 주의 — 학습 config 는 1 이지만 평가·배포는 3 이다. 정책은 결정마다
      그물벽 '하나'(픽셀 2개 → 선분 1개)를 놓으므로 결정당 행동공간이 보유량과 무관하고,
      따라서 1장으로 학습한 정책이 재학습 없이 3장 설정에서 동작한다. 표에는 둘 다 적는다.
    """
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    d = cfg if isinstance(cfg, dict) else vars(cfg)
    rows = []
    for key, ko, unit, fmt in SCENARIO:
        if key not in d:
            continue
        rows.append(f"{ko} & {fmt.format(float(d[key]))} & {unit} \\\\\n")
    px = 2.0 * float(d["cnn_extent"]) / float(d["cnn_grid_n"])
    rows.append(f"관측 픽셀 한 변 & {px:.0f} & m \\\\\n")
    rows.append(f"방어정당 그물 (학습 / 평가) & {int(d['nets_per_ship'])} / {nets_eval} & 장 \\\\\n")
    n_par = sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel"))
    rows.append(f"정책 파라미터 수 & {n_par:,} & 개 \\\\\n")

    tex = _wrap(
        "".join(rows),
        "교전 시나리오와 관측 설정. 값은 학습 체크포인트의 설정에서 직접 읽었다.",
        "tab:scenario", "lrl",
        "항목 & 값 & 단위 \\\\",
        "방어정 속력이 적 속력보다 낮다 --- 추격이 성립하지 않으므로 접근 회랑의 선제 "
        "차단이 유일한 대응이다. 그물 보유량은 학습 시 1장, 평가·배포 시 3장이다: "
        "정책은 결정마다 그물벽 하나를 놓으므로 결정당 행동공간이 보유량과 무관하고, "
        "1장으로 학습한 정책이 재학습 없이 3장 설정에서 동작한다.")
    _write(out, "tab_scenario.tex", tex)


# ── 표 1: 주 결과 (포획률 8조건) ─────────────────────────────────────────────
def tab_main(df, out):
    piv = S.pivot_by_seed(df, "capture_rate")
    base = piv["heur_heur"].to_numpy()
    rows = []
    for c in _conds_ordered(df):
        v = piv[c].to_numpy()
        m = S.mean_ci(v)
        # ★ _esc 를 밖에서 걸면 열 구분자 & 까지 이스케이프된다. _cond_cols 가
        #   각 칸을 따로 이스케이프해 돌려준다.
        line = f"{_cond_cols(c)} & {m.mean:.3f} [{m.lo:.3f}, {m.hi:.3f}]"
        if c == "heur_heur":
            line += " & --- & --- \\\\\n"
        else:
            d = S.paired_diff(v, base)
            line += (f" & {d.mean:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]{_sig(d)}"
                     f" & {S.cohens_dz(v, base):+.2f} \\\\\n")
        rows.append(line)
    body = "".join(rows)
    tex = _wrap(
        body,
        "조건별 포획률. 평균과 부트스트랩 95\\,\\% 신뢰구간, baseline 대비 대응표본 차이. "
        "3개 포메이션 통합(각 조건 30시드).",
        "tab:main", "llccc",
        "지휘관(배정) & 기동 & 포획률 [95\\,\\% CI] & $\\Delta$ vs baseline & $d_z$ \\\\",
        "$^{*}$ 신뢰구간이 0 을 포함하지 않음. $d_z$ 는 대응표본 효과크기. "
        "첫 두 열이 $2\\times2$ 설계의 두 축이다 --- 같은 지휘관 행 두 개를 비교하면 "
        "기동 계층의 기여가, 같은 기동 열을 비교하면 지휘관의 기여가 나온다. "
        "Qwen2.5 는 로컬(단일 RTX 4070) 서빙, Gemini 는 클라우드 API 다.",
        wide=True, tight=True)
    _write(out, "tab_main.tex", tex)


# ── 표 2: 전 지표 효과크기 ──────────────────────────────────────────────────
def tab_effectsize(df, out, treatment="llm_unet@gemini-3.5-flash-lite"):
    rows = []
    for met, ko, arrow, nd in METRICS:
        try:
            piv = S.pivot_by_seed(df, met)
            if treatment not in piv.columns or "heur_heur" not in piv.columns:
                continue
            ok = piv[[treatment, "heur_heur"]].dropna()
            x, b = ok[treatment].to_numpy(), ok["heur_heur"].to_numpy()
        except Exception:
            continue
        d = S.paired_diff(x, b)
        dz = S.cohens_dz(x, b)
        rows.append((abs(dz), f"{_esc(ko)} & {arrow} & {b.mean():.{nd}f} & "
                              f"{x.mean():.{nd}f} & "
                              f"{d.mean:+.{nd}f}{_sig(d)}{_sig_bonf(x, b)} & "
                              f"{dz:+.2f} \\\\\n"))
    # |dz| 내림차순 — 포획률이 맨 아래로 가면서 "가장 둔한 지표"임이 표에서 바로 보인다.
    body = "".join(r for _, r in sorted(rows, key=lambda t: -t[0]))
    tex = _wrap(
        body,
        "전 지표 효과크기 (제안 = LLM 지휘관 + U-Net 점수맵, Gemini). "
        "$|d_z|$ 내림차순. \\textbf{포획률이 가장 둔한 지표이며, 다중비교 보정에서 "
        "유의성을 잃는 유일한 방어 성과 지표다.}",
        "tab:effectsize", "lccccc",
        "지표 & 방향 & baseline & 제안 & $\\Delta$ & $d_z$ \\\\",
        "방향 $\\uparrow$ = 높을수록 좋음, $\\downarrow$ = 낮을수록 좋음. "
        "$^{*}$ 95\\,\\% 신뢰구간이 0 을 포함하지 않음. "
        "$^{\\dagger}$ 지표 10개에 대한 Bonferroni 보정(99.5\\,\\% 구간)에서도 0 을 배제함 "
        "--- 즉 단검이 없는 행은 보정을 견디지 못한다.",
        wide=True)
    _write(out, "tab_effectsize.tex", tex)


# ── 표 3: 포메이션별 천장 효과 ──────────────────────────────────────────────
def tab_ceiling(df, out, treatment="llm_unet@gemini-3.5-flash-lite"):
    rows = []
    for f in ("concentrated", "diversionary", "wave"):
        piv = S.pivot_by_seed(df, "capture_rate", formation=f)
        if treatment not in piv.columns:
            continue
        ok = piv[[treatment, "heur_heur"]].dropna()
        b, x = ok["heur_heur"].to_numpy(), ok[treatment].to_numpy()
        d = S.paired_diff(x, b)
        rows.append(f"{FORM_KO.get(f, f)} & {b.mean():.3f} & {1 - b.mean():.3f} & "
                    f"{x.mean():.3f} & {d.mean:+.3f}{_sig(d)} \\\\\n")
    tex = _wrap(
        "".join(rows),
        "포메이션별 천장 효과. 집중 포메이션은 baseline 이 이미 0.997 이라 "
        "개선 여지가 0.003 뿐이다 --- 이 포메이션 단독으로 결론을 내면 안 된다.",
        "tab:ceiling", "lcccc",
        "포메이션 & baseline & 남은 여유 & 제안 & $\\Delta$ \\\\",
        "$^{*}$ 신뢰구간이 0 을 포함하지 않음.")
    _write(out, "tab_ceiling.tex", tex)


# ── 표 3-2: 기동 계층 단독 기여 (LLM 없이) ──────────────────────────────────
def tab_maneuver(df, out):
    """U-Net 점수맵 기동 계층이 **LLM 없이** 내는 기여를 포메이션별로.

    이 표가 논문에서 갖는 위치가 특수하다: `heur_unet` 은 LLM 을 전혀 쓰지 않으므로
    **폐쇄형 API 에 의존하지 않는 재현 가능한 기여**다. 지휘관 모델이 단종되어도
    이 결과는 남는다. 그래서 결과 절의 첫 표로 놓는다.
    """
    rows = []
    for f in ("concentrated", "diversionary", "wave", None):
        piv = S.pivot_by_seed(df, "capture_rate", formation=f)
        if "heur_unet" not in piv.columns:
            continue
        ok = piv[["heur_unet", "heur_heur"]].dropna()
        b, x = ok["heur_heur"].to_numpy(), ok["heur_unet"].to_numpy()
        d = S.paired_diff(x, b)
        name = FORM_KO.get(f, f) if f else "\\textbf{전체}"
        rows.append(f"{name} & {b.mean():.3f} & {x.mean():.3f} & "
                    f"{d.mean:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]{_sig(d)} & "
                    f"{S.cohens_dz(x, b):+.2f} \\\\\n")
    tex = _wrap(
        "".join(rows),
        "기동 계층(U-Net 점수맵 + GRPO)의 단독 기여. 두 조건 모두 배정은 휴리스틱이며 "
        "\\textbf{LLM 을 전혀 사용하지 않는다} --- 따라서 이 결과는 외부 API 에 의존하지 "
        "않고 재현 가능하다. 파상 포메이션에서 가장 크다.",
        "tab:maneuver", "lcccc",
        "포메이션 & 휴리스틱 기동 & U-Net 기동 & $\\Delta$ [95\\,\\% CI] & $d_z$ \\\\",
        "$^{*}$ 신뢰구간이 0 을 포함하지 않음. 전체 행은 3개 포메이션 통합.",
        wide=True)   # CI 열 때문에 단폭을 47pt 넘겨 여백을 침범했다
    _write(out, "tab_maneuver.tex", tex)


# ── 표 4: 2x2 상호작용 분해 ─────────────────────────────────────────────────
def tab_interaction(df, out, backend="gemini-3.5-flash-lite"):
    rows, n_add, n_tot = [], 0, 0
    for met, ko, _arrow, nd in METRICS:
        try:
            r = S.interaction(df, met, backend=backend)
        except Exception:
            continue
        t, i = r["total"], r["interaction"]
        additive = not (i.lo > 0 or i.hi < 0)
        n_tot += 1
        n_add += int(additive)
        verdict = "가산" if additive else ("상승" if i.mean > 0 else "상쇄")
        rows.append(f"{_esc(ko)} & {t.mean:+.{nd}f} & "
                    f"{i.mean:+.{nd}f} [{i.lo:+.{nd}f}, {i.hi:+.{nd}f}] & {verdict} \\\\\n")
    tex = _wrap(
        "".join(rows),
        "2$\\times$2 상호작용 분해 (Gemini 지휘관). "
        f"{n_tot}개 지표 중 \\textbf{{{n_add}개}}에서 상호작용 신뢰구간이 0 을 포함한다 "
        "--- 두 계층의 기여는 \\textbf{가산적}이다.",
        "tab:interaction", "lccc",
        "지표 & 총효과 & 상호작용 [95\\,\\% CI] & 판정 \\\\",
        "상호작용 $=$ (LLM+U-Net $-$ LLM+휴리스틱) $-$ (휴리스틱+U-Net $-$ baseline). "
        "신뢰구간이 0 을 포함하면 두 계층이 서로를 돕지도 잡아먹지도 않는다.",
        wide=True)
    _write(out, "tab_interaction.tex", tex)


# ── 표 5: 지휘관 계획 품질 ──────────────────────────────────────────────────
def tab_planquality(df, llm_df, out):
    col = "model" if "model" in llm_df.columns else "backend"
    piv = S.pivot_by_seed(df, "capture_rate")
    rows = []
    for m in MODELS:
        g = llm_df[llm_df[col].astype(str) == m]
        if g.empty:
            continue
        def mv(k):
            return g[k].astype(float).dropna().mean() if k in g.columns else np.nan
        key = f"llm_heur@{m}"
        cap = piv[key].dropna().mean() if key in piv.columns else np.nan
        rows.append(f"{MODEL_KO.get(m, m)} & {mv('latency_s'):.2f} & "
                    f"{mv('coverage'):.3f} & {mv('churn'):.3f} & "
                    f"{mv('crossings'):.3f} & {mv('fallback'):.3f} & {cap:.3f} \\\\\n")
    tex = _wrap(
        "".join(rows),
        "지휘관 계획 품질과 성능. 계획 품질은 시뮬레이션 결과와 독립적으로, "
        "계획이 산출된 시점에 측정한다. \\textbf{7B 는 커버리지가 가장 높은데 "
        "포획률은 가장 낮다} --- 실패 원인은 미배정이 아니다.",
        "tab:planquality", "lcccccc",
        "지휘관 & 지연 (s) & 커버리지 $\\uparrow$ & churn $\\downarrow$ & "
        "교차 $\\downarrow$ & 폴백 $\\downarrow$ & 포획률 $\\uparrow$ \\\\",
        "커버리지 $=$ 배정된 활성 클러스터 비율. churn $=$ 재계획 간 담당이 바뀐 배의 비율. "
        "교차 $=$ 두 방어정의 요격 경로가 교차한 비율. 포획률은 기동 계층을 "
        "휴리스틱으로 고정한 조건(LLM 지휘관 + 휴리스틱 기동)의 값이다.",
        wide=True, tight=True)
    _write(out, "tab_planquality.tex", tex)


# ── 보조 ────────────────────────────────────────────────────────────────────
def _cond_cols(c: str) -> str:
    """조건 태그를 '지휘관 & 기동' 두 칸으로. 한 칸에 몰면 열이 판면을 넘는다."""
    base, _, tag = c.partition("@")
    # ★ "(로컬)/(클라우드)" 를 여기서 떼어 각주로 보낸다. 이 괄호가 열 폭을 결정해
    #   판면을 넘기고, 그러면 표 전체가 축소돼 글자가 본문보다 작아진다.
    cmd = MODEL_KO.get(tag, tag).split(" (")[0] if tag else "휴리스틱"
    mnv = "U-Net 점수맵" if base.endswith("_unet") else "휴리스틱"
    if base == "heur_heur":
        cmd += " (baseline)"
    return f"{_esc(cmd)} & {_esc(mnv)}"


def _cond_ko(c: str) -> str:
    base, _, tag = c.partition("@")
    ko = COND_KO.get(base, base)
    return f"{ko} --- {MODEL_KO.get(tag, tag)}" if tag else ko


def _conds_ordered(df):
    order = ("heur_heur", "heur_unet", "llm_heur", "llm_unet")
    have = list(dict.fromkeys(S.drop_incomplete(df)["condition"].tolist()))
    rank = {c: i for i, c in enumerate(order)}
    mrank = {m: i for i, m in enumerate(MODELS)}
    return sorted(have, key=lambda c: (rank.get(c.partition("@")[0], 99),
                                       mrank.get(c.partition("@")[2], 99)))


def _write(out: str, name: str, tex: str):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    print(f"  [ok] {p}  ({len(tex)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/eval_merged")
    ap.add_argument("--out", default="논문초안/tables")
    ap.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt",
                    help="시나리오 표를 읽어올 체크포인트. 결과를 낸 것과 같아야 한다.")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.src, "episodes.csv"), encoding="utf-8-sig")
    lp = os.path.join(a.src, "llm_metrics.csv")
    llm_df = pd.read_csv(lp, encoding="utf-8-sig") if os.path.exists(lp) else None
    print(f"[표] 원자료 {len(df)} 에피소드 → {a.out}")

    if os.path.exists(a.ckpt):
        tab_scenario(a.ckpt, a.out)
    else:
        print(f"  [skip] tab_scenario: 체크포인트 없음 {a.ckpt}")
    tab_main(df, a.out)
    tab_maneuver(df, a.out)
    tab_effectsize(df, a.out)
    tab_ceiling(df, a.out)
    tab_interaction(df, a.out)
    if llm_df is not None:
        tab_planquality(df, llm_df, a.out)
    print("[표] 완료")


if __name__ == "__main__":
    main()
