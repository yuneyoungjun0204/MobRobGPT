"""통계 — 시드 대응표본(paired) 부트스트랩 CI, 효과크기, 2×2 상호작용.

논문에 "제안이 baseline보다 높다"를 쓰려면 **불확실성**이 같이 있어야 한다.
평균만 적힌 표는 심사에서 반드시 지적된다.

왜 paired 인가
    같은 seed 는 조건이 달라도 같은 전장이다. 조건 간 차이 d_k = x_k − y_k 를 만들면
    시나리오 난이도(어떤 시드는 그냥 어렵다)가 상쇄된다. 독립표본 CI 보다 훨씬 좁아
    적은 시드로 결론이 난다. harness 가 모든 조건에 같은 시드를 주는 이유가 이것이다.

왜 BCa 인가
    포획률은 [0,1] 로 잘려 있고 분포가 치우친다. 백분위 부트스트랩은 이런 경우
    편향된 CI 를 준다. BCa 는 편향(z0)과 가속(a)을 보정한다. (scipy.stats.bootstrap)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def _bootstrap(data, statistic, *, paired=False, n_resamples=9999,
               confidence_level=0.95, seed=0, method="BCa"):
    """scipy.stats.bootstrap 호출 — rng/random_state 인자명 차이를 흡수한다.

    scipy 1.15 에서 `random_state` 가 `rng` 로 바뀌었다(구 이름은 경고와 함께 유지).
    두 이름을 다 시도해 버전에 관계없이 동작하게 한다.
    """
    from scipy.stats import bootstrap as _bs
    kw = dict(statistic=statistic, paired=paired, n_resamples=int(n_resamples),
              confidence_level=float(confidence_level), method=method, vectorized=False)
    try:
        return _bs(data, rng=np.random.default_rng(seed), **kw)
    except TypeError:
        return _bs(data, random_state=np.random.default_rng(seed), **kw)


@dataclass(frozen=True)
class Estimate:
    """점추정 + CI. `n` 은 결측 제거 후 실제 사용된 표본 수다."""
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    @property
    def significant(self) -> bool:
        """CI 가 0 을 포함하지 않는가. 차이 추정에만 의미가 있다."""
        return (self.lo > 0) or (self.hi < 0)


def mean_ci(x, *, confidence_level=0.95, n_resamples=9999, seed=0) -> Estimate:
    """단일 표본 평균의 부트스트랩 CI. NaN 은 제거한다(교전 미성립 에피소드 등)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return Estimate(np.nan, np.nan, np.nan, 0)
    if x.size < 3 or np.allclose(x, x[0]):
        # 표본이 너무 적거나 분산 0 → 부트스트랩이 degenerate. CI 를 만들지 않는다.
        return Estimate(float(x.mean()), np.nan, np.nan, int(x.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _bootstrap((x,), np.mean, confidence_level=confidence_level,
                       n_resamples=n_resamples, seed=seed)
    ci = r.confidence_interval
    return Estimate(float(x.mean()), float(ci.low), float(ci.high), int(x.size))


def paired_diff(x, y, *, confidence_level=0.95, n_resamples=9999, seed=0) -> Estimate:
    """대응표본 평균차 (x − y) 의 부트스트랩 CI.

    x, y 는 **같은 순서의 같은 시드**여야 한다. 둘 중 하나라도 NaN 인 쌍은 통째로 버린다
    (한쪽만 버리면 대응이 깨져 차이가 무의미해진다).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError(f"대응표본 길이 불일치: {x.shape} vs {y.shape} — "
                         f"조건마다 같은 시드 집합을 썼는지 확인하라")
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if d.size == 0:
        return Estimate(np.nan, np.nan, np.nan, 0)
    if d.size < 3 or np.allclose(d, d[0]):
        return Estimate(float(d.mean()), np.nan, np.nan, int(d.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _bootstrap((d,), np.mean, confidence_level=confidence_level,
                       n_resamples=n_resamples, seed=seed)
    ci = r.confidence_interval
    return Estimate(float(d.mean()), float(ci.low), float(ci.high), int(d.size))


def cohens_dz(x, y) -> float:
    """대응표본 효과크기 d_z = mean(d)/sd(d). CI 와 함께 보고하면 '얼마나 큰 차이인가'가 선다.

    관례: |d_z| 0.2 작음 · 0.5 중간 · 0.8 큼.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if d.size < 2:
        return np.nan
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan


def wilcoxon_p(x, y) -> float:
    """대응표본 부호순위 검정 p. 정규성을 가정하지 않는다(포획률은 치우친 분포).

    CI 의 보조 지표로만 쓴다 — p 만 보고하는 표는 효과크기를 숨긴다.
    """
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if d.size < 3 or np.allclose(d, 0):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(d).pvalue)
        except ValueError:
            return np.nan


# ── 조건행렬 집계 ────────────────────────────────────────────────────────────
def coverage_report(df):
    """조건 × 포메이션 커버리지 표. 평가가 중간에 끊겼는지 한눈에 본다."""
    return df.pivot_table(index="condition", columns="formation", values="seed",
                          aggfunc="count").fillna(0).astype(int)


def drop_incomplete(df, *, min_frac: float = 0.9, warn=None):
    """수집이 덜 끝난 조건을 통째로 제외한다.

    ★ 왜 필요한가: `balanced_subset` 은 **모든** 조건의 공통 교집합을 취하므로,
      이제 막 5 시드 들어온 조건이 하나 끼면 나머지 완성된 조건들까지 5 시드로 잘린다.
      (실제로 겪었다 — 포획률이 전부 1.0 으로 보이는 무의미한 그림이 나왔다.)
      가장 많이 덮은 조건의 `min_frac` 미만인 조건은 아직 비교 대상이 아니므로 뺀다.

    수집 중에 중간 결과를 그릴 때를 위한 것이다. 전량 수집이 끝나면 아무것도 지우지 않는다.
    """
    if "condition" not in df.columns or df["condition"].nunique() < 2:
        return df
    # 조건별로 덮은 (formation, seed) 조합 수. groupby.apply 는 FutureWarning 이 나므로
    # 중복 제거 후 크기를 세는 방식으로 바꾼다.
    cov = (df[["condition", "formation", "seed"]].drop_duplicates()
           .groupby("condition").size())
    keep = cov[cov >= cov.max() * float(min_frac)].index
    dropped = sorted(set(cov.index) - set(keep))
    if dropped and warn:
        warn(f"[stats] 수집 미완 조건 제외: {dropped} "
             f"(덮은 조합 {[int(cov[c]) for c in dropped]} < 기준 "
             f"{cov.max() * min_frac:.0f})")
    return df[df["condition"].isin(keep)]


def balanced_subset(df, *, warn=None):
    """모든 조건이 **똑같이 덮은** (포메이션, 시드) 조합만 남긴다.

    ★ 왜 필요한가: 평가가 중간에 끊기면 조건마다 덮은 포메이션이 달라진다.
      그 상태로 평균을 내면 '쉬운 포메이션만 돈 조건'이 유리해져 비교가 거짓이 된다.
      대응표본의 전제(같은 전장끼리 비교)도 깨진다.
      그래서 집계 전에 **공통 교집합**으로 잘라낸다. 자르면 표본이 줄지만 그건 정직한 축소다.
    """
    keys = ["formation", "seed"]
    conds = sorted(df["condition"].unique())
    sets = [set(map(tuple, df.loc[df["condition"] == c, keys].to_numpy())) for c in conds]
    common = set.intersection(*sets) if sets else set()
    full = set(map(tuple, df[keys].to_numpy()))
    if warn and len(common) < len(full):
        cov = coverage_report(df)
        warn(f"[stats] 조건별 커버리지가 불균형이라 공통 교집합만 사용한다: "
             f"{len(common)}/{len(full)} (포메이션, 시드) 조합\n{cov.to_string()}")
    if not common:
        return df.iloc[0:0]
    mask = [tuple(r) in common for r in df[keys].to_numpy()]
    return df[mask]


def pivot_by_seed(df, metric: str, *, formation: str | None = None, balance: bool = True):
    """조건 × 시드 행렬로 편다. 대응표본 비교의 입력이 되는 표.

    반환: DataFrame (index=seed, columns=condition). 같은 행 = 같은 전장.

    `balance=True`(기본) 면 모든 조건이 공통으로 덮은 조합만 쓴다 — 중단된 평가에서
    조건마다 포메이션 구성이 달라 생기는 거짓 비교를 막는다(`balanced_subset` 참조).
    """
    d = df if formation is None else df[df["formation"] == formation]
    if balance and d["condition"].nunique() > 1:
        # 미완 조건을 먼저 걷어낸 뒤 교집합을 취한다. 순서가 반대면 갓 시작한 조건 하나가
        # 완성된 조건 전부를 그 시드 수까지 끌어내린다.
        d = balanced_subset(drop_incomplete(d))
    # 같은 (condition, seed) 가 여러 포메이션에 걸치면 평균을 낸다(formation=None 인 경우).
    return d.pivot_table(index="seed", columns="condition", values=metric, aggfunc="mean")


def summarize(df, metric: str = "capture_rate", *, formations=None,
              baseline: str = "heur_heur", seed: int = 0):
    """조건별 평균 CI + baseline 대비 대응차이 표를 만든다.

    반환 컬럼: formation, condition, mean, lo, hi, n, d_mean, d_lo, d_hi, dz, p, sig
    `formation="ALL"` 행은 포메이션을 합친 전체 요약이다.
    """
    import pandas as pd
    forms = list(formations) if formations is not None else sorted(df["formation"].unique())
    rows = []
    for form in forms + ["ALL"]:
        piv = pivot_by_seed(df, metric, formation=None if form == "ALL" else form)
        if baseline not in piv.columns:
            base = None
        else:
            base = piv[baseline].to_numpy()
        for cond in piv.columns:
            col = piv[cond].to_numpy()
            est = mean_ci(col, seed=seed)
            row = {"formation": form, "condition": cond, "metric": metric,
                   "mean": est.mean, "lo": est.lo, "hi": est.hi, "n": est.n}
            if base is not None and cond != baseline:
                d = paired_diff(col, base, seed=seed)
                row |= {"d_mean": d.mean, "d_lo": d.lo, "d_hi": d.hi,
                        "dz": cohens_dz(col, base), "p": wilcoxon_p(col, base),
                        "sig": d.significant}
            else:
                row |= {"d_mean": np.nan, "d_lo": np.nan, "d_hi": np.nan,
                        "dz": np.nan, "p": np.nan, "sig": False}
            rows.append(row)
    return pd.DataFrame(rows)


#: LLM 계획 품질 지표와 개선 방향. llm_metrics.csv 의 컬럼명이다.
LLM_METRICS = (
    ("latency_s", "lower"),            # 응답 지연 — 비동기 설계의 근거
    ("fallback", "lower"),             # 폴백률 — 낮을수록 신뢰성
    ("coverage", "higher"),            # 활성 클러스터 중 배정된 비율
    ("churn", "lower"),                # 재계획 간 배정 요동 — 높으면 반쯤 깐 그물을 버린다
    ("crossings", "lower"),            # 교차 배정 — 모선 가로지르기·충돌 선행지표
    ("n_hold", "lower"),               # HOLD 척수 — 과도하면 전력 낭비
    ("heuristic_agreement", None),     # ★ 방향 없음: 1.0=LLM 무용, 0.0=통제 불가
)


def compare_llm(llm_df, *, by: str = "backend", metrics=LLM_METRICS,
                confidence_level: float = 0.95, seed: int = 0):
    """지휘관별 계획 품질 비교표 (독립표본 CI).

    에피소드 지표와 달리 **대응(paired)이 아니다** — 계획 호출은 지휘관마다 시점·횟수가
    달라 시드로 짝지을 수 없다. 그래서 각 그룹의 평균 CI 를 따로 내고 겹침으로 판단한다.
    (겹치지 않으면 확실히 다르고, 겹쳐도 다를 수는 있다 — 보수적인 판정이다.)

    `heuristic_agreement` 는 좋고 나쁨의 방향이 없다. 1.0 이면 코드로 대체 가능하다는 뜻이고
    0.0 이면 통제가 안 된다는 뜻이라, **중간값이 의미 있는** 지표다. 표에 방향을 적지 않는다.
    """
    import pandas as pd
    if by not in llm_df.columns:
        raise ValueError(f"llm_metrics 에 '{by}' 컬럼이 없다: {list(llm_df.columns)}")
    rows = []
    for g, sub in llm_df.groupby(by):
        for m, direction in metrics:
            if m not in sub.columns:
                continue
            x = pd.to_numeric(sub[m], errors="coerce").astype(float).to_numpy()
            est = mean_ci(x, confidence_level=confidence_level, seed=seed)
            rows.append({by: g, "metric": m, "better": direction or "n/a",
                         "mean": est.mean, "lo": est.lo, "hi": est.hi, "n": est.n})
    return pd.DataFrame(rows)


def llm_pairwise(llm_df, *, by: str = "backend", metrics=LLM_METRICS, seed: int = 0):
    """지휘관 두 그룹씩 평균차 CI (독립표본 부트스트랩).

    `compare_llm` 이 각 그룹을 따로 보여준다면 이건 **차이 자체**에 CI 를 붙인다.
    "GPT 가 로컬보다 churn 이 낮다"를 주장하려면 이 표가 필요하다.
    """
    import itertools
    import pandas as pd
    groups = sorted(llm_df[by].dropna().unique())
    rows = []
    for a, b in itertools.combinations(groups, 2):
        for m, direction in metrics:
            if m not in llm_df.columns:
                continue
            xa = pd.to_numeric(llm_df.loc[llm_df[by] == a, m], errors="coerce").dropna()
            xb = pd.to_numeric(llm_df.loc[llm_df[by] == b, m], errors="coerce").dropna()
            xa, xb = xa.to_numpy(float), xb.to_numpy(float)
            if xa.size < 3 or xb.size < 3:
                rows.append({"a": a, "b": b, "metric": m, "better": direction or "n/a",
                             "d_mean": (xa.mean() - xb.mean()) if xa.size and xb.size else np.nan,
                             "d_lo": np.nan, "d_hi": np.nan, "sig": False,
                             "n_a": xa.size, "n_b": xb.size})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 독립표본 두 개 → statistic 이 두 표본을 받는 형태로 준다.
                r = _bootstrap((xa, xb), lambda p, q, axis=-1: (np.mean(p, axis=axis)
                                                               - np.mean(q, axis=axis)),
                               paired=False, n_resamples=4999, seed=seed, method="basic")
            ci = r.confidence_interval
            rows.append({"a": a, "b": b, "metric": m, "better": direction or "n/a",
                         "d_mean": float(xa.mean() - xb.mean()),
                         "d_lo": float(ci.low), "d_hi": float(ci.high),
                         "sig": bool(ci.low > 0 or ci.high < 0),
                         "n_a": int(xa.size), "n_b": int(xb.size)})
    return pd.DataFrame(rows)


def interaction(df, metric: str = "capture_rate", *, formation: str | None = None,
                seed: int = 0, backend: str = "") -> dict:
    """2×2 주효과와 상호작용을 대응표본으로 분해한다.

        기동 주효과  = (heur_unet − heur_heur) 와 (llm_unet − llm_heur) 의 평균
        배정 주효과  = (llm_heur  − heur_heur) 와 (llm_unet − heur_unet) 의 평균
        상호작용     = (llm_unet − llm_heur) − (heur_unet − heur_heur)
                       > 0 이면 두 계층이 **서로를 돕는다**(제안의 핵심 주장).
                       ≈ 0 이면 두 기여가 단순 합산이다 — 그것도 정직하게 보고할 것.

    `backend` 를 주면 그 지휘관의 조건(`llm_*@backend`)으로 분해한다 — 백엔드를 여러 개
    돌렸을 때 지휘관마다 따로 결론을 내기 위한 것이다.

    반환값의 각 항목은 Estimate. CI 가 0 을 포함하면 그 효과는 확인되지 않은 것이다.
    """
    piv = pivot_by_seed(df, metric, formation=formation)
    sfx = f"@{backend}" if backend else ""
    lk_h, lk_u = f"llm_heur{sfx}", f"llm_unet{sfx}"
    need = {"heur_heur", "heur_unet", lk_h, lk_u}
    missing = need - set(piv.columns)
    if missing:
        raise ValueError(f"상호작용 분해에 필요한 조건이 없다: {sorted(missing)} "
                         f"(LLM 조건을 건너뛰었다면 이 함수는 쓸 수 없다)")
    hh = piv["heur_heur"].to_numpy(); hu = piv["heur_unet"].to_numpy()
    lh = piv[lk_h].to_numpy();        lu = piv[lk_u].to_numpy()
    return {
        "maneuver_at_heur_assign": paired_diff(hu, hh, seed=seed),
        "maneuver_at_llm_assign":  paired_diff(lu, lh, seed=seed),
        "assign_at_heur_maneuver": paired_diff(lh, hh, seed=seed),
        "assign_at_unet_maneuver": paired_diff(lu, hu, seed=seed),
        "total":       paired_diff(lu, hh, seed=seed),
        "interaction": paired_diff(lu - lh, hu - hh, seed=seed),
    }
