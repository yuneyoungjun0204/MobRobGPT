"""LLM 지휘관 계측 — 배정 계획 자체의 품질을 재는 계층.

포획률은 **시스템 전체**의 지표라 LLM 이 잘했는지 못했는지를 가린다.
(정책이 잘 막아서 포획률이 높은 건지, 배정이 좋아서인지 구분이 안 된다.)
여기서는 계획을 낸 순간에 확인 가능한 것만 잰다 — 시뮬 결과와 독립이므로
"LLM 계층의 기여"를 따로 보고할 수 있다.

재는 것 6가지
    ① 응답 지연        — 비동기 설계의 근거. p50/p95 를 보고한다.
    ② 폴백률           — LLM 이 실패해 휴리스틱으로 떨어진 비율. 신뢰성의 하한.
    ③ 커버리지         — 활성 클러스터 중 배정된 비율. 안 막은 무리가 곧 돌파다.
    ④ churn(요동)      — 재계획 간 배정이 바뀐 배의 비율. 높으면 반쯤 깐 그물을 버린다.
    ⑤ 교차 배정        — 두 배의 요격 경로가 교차하는가. 모선 가로지르기·충돌의 선행지표.
    ⑥ 휴리스틱 일치율  — 코드가 낸 배정과 같은 비율. 100%면 LLM 을 쓸 이유가 없고,
                          0%면 통제가 안 되는 것이다. **중간값이 의미 있다.**

사용:
    rec = LLMRecorder()
    ... run_condition(..., recorder=rec)
    rec.to_frame().to_csv("llm_metrics.csv", index=False)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def _seg_cross(p1, p2, q1, q2) -> bool:
    """선분 p1p2 와 q1q2 가 교차하는가 (외적 부호 판정). 접점/공선은 비교차로 본다."""
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cr(q1, q2, p1), cr(q1, q2, p2)
    d3, d4 = cr(p1, p2, q1), cr(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class LLMRecorder:
    """계획 호출을 하나씩 기록한다. `observe` 는 harness 의 LLM 훅이 부른다."""

    #: 어느 지휘관의 기록인가. 백엔드를 여러 개 돌릴 때 CSV 에서 갈라 보기 위한 라벨.
    backend: str = ""
    model: str = ""
    #: ★ 즉시 기록 경로(JSONL). 주면 관측 1건마다 append 한다 — 평가가 중간에 끊겨도
    #   계측이 남고, 재개 실행분과 자연히 합쳐진다(에피소드 체크포인트와 같은 이유).
    #   CSV 가 아니라 JSONL 인 이유: 컬럼이 늘어도 헤더를 다시 쓸 필요가 없고,
    #   append 도중 죽어도 마지막 한 줄만 버리면 나머지가 온전하다.
    jsonl_path: str | None = None
    #: 실행 중 폴백 감시. 초반 `_min_check` 회 안에서 폴백률이 임계를 넘으면 예외로 세운다.
    #   (사전 프로브를 통과해도 도중에 크레딧·쿼터가 끊길 수 있다.)
    abort_on_fallback: bool = True
    abort_threshold: float = 0.5
    _min_check: int = 12
    rows: list[dict] = field(default_factory=list)
    _prev_assign: dict[int, int] = field(default_factory=dict)   # ally_id → cluster_id
    _t_last: float | None = None

    # ── 지연 측정: plan() 호출을 감싸 시간을 잰다 ──
    def wrap(self, commander):
        """commander.plan 을 감싸 지연을 재는 프록시를 돌려준다.

        원본 객체를 변형하지 않는다(다른 곳에서 같은 지휘관을 쓸 수 있으므로).
        """
        rec = self

        class _Timed:
            def __init__(self, inner):
                self._inner = inner

            def plan(self, state):
                t0 = time.perf_counter()
                p = self._inner.plan(state)
                rec._t_last = time.perf_counter() - t0
                return p

            def __getattr__(self, k):
                return getattr(self._inner, k)

        return _Timed(commander)

    # ── 계획 1건 관측 ──
    def observe(self, env, state, plan) -> dict:
        """계획 하나를 기록. env 는 기하 계산용(요격점 좌표), state/plan 은 스키마 객체."""
        assigned: dict[int, int] = {}
        for d in getattr(plan, "deployments", []) or []:
            for aid in (d.ally_ids or []):
                assigned[int(aid)] = int(d.cluster_id)

        clusters = list(getattr(state, "enemy_clusters", []) or [])
        allies = [a for a in (getattr(state, "allies", []) or []) if getattr(a, "alive", True)]
        n_cl = len(clusters)
        n_al = len(allies)

        # ── ③ 커버리지: 배정을 하나라도 받은 클러스터 비율 ──
        covered = len({c for c in assigned.values()})
        coverage = covered / n_cl if n_cl else np.nan

        # ── ② 폴백: openai/ollama commander 가 rationale 앞에 이유를 붙인다 ──
        rat = str(getattr(plan, "rationale", "") or "")
        fallback = "휴리스틱 방어" in rat

        # ── ④ churn: 직전 계획 대비 배정이 바뀐 배의 비율 ──
        prev = self._prev_assign
        common = [int(a.id) for a in allies if int(a.id) in prev]
        churn = (float(np.mean([assigned.get(i, -1) != prev[i] for i in common]))
                 if common else np.nan)

        # ── ⑤ 교차 배정: 배 위치 → 배정 클러스터 중심 선분끼리 교차 ──
        seg = []
        cl_by_id = {int(c.id): c for c in clusters}
        for a in allies:
            k = assigned.get(int(a.id))
            c = cl_by_id.get(k) if k is not None else None
            if c is None:
                continue
            seg.append(((float(a.pos.x), float(a.pos.y)),
                        (float(c.center.x), float(c.center.y))))
        crossings = sum(1 for i in range(len(seg)) for j in range(i + 1, len(seg))
                        if _seg_cross(*seg[i], *seg[j]))

        # ── ⑥ 휴리스틱 일치율: 같은 state 에 코드가 낸 배정과 배별로 비교 ──
        try:
            from .fallback import heuristic_plan
            hp = heuristic_plan(state)
            hmap: dict[int, int] = {}
            for d in hp.deployments or []:
                for aid in (d.ally_ids or []):
                    hmap[int(aid)] = int(d.cluster_id)
            keys = [int(a.id) for a in allies]
            agree = (float(np.mean([assigned.get(i, -1) == hmap.get(i, -1) for i in keys]))
                     if keys else np.nan)
        except Exception:
            agree = np.nan

        row = {
            "backend": self.backend,
            "model": self.model,
            "t": int(getattr(env, "t", [0])[0]) if env is not None else -1,
            "latency_s": self._t_last if self._t_last is not None else np.nan,
            "fallback": bool(fallback),
            "n_clusters": n_cl,
            "n_allies": n_al,
            "n_deployments": len(getattr(plan, "deployments", []) or []),
            "n_assigned_ships": len(assigned),
            "n_hold": n_al - len(assigned),
            "coverage": coverage,
            "churn": churn,
            "crossings": crossings,
            "heuristic_agreement": agree,
            "rationale_chars": len(rat),
        }
        self.rows.append(row)
        if self.jsonl_path:
            self._flush(row)
        self._prev_assign = assigned
        self._t_last = None
        # ★ 실행 중 폴백 감시: 사전 프로브를 통과해도 도중에 크레딧이 떨어지거나 쿼터가
        #   막히면 이후 전량이 조용히 휴리스틱이 된다. 초반 표본에서 폴백이 지배적이면
        #   수집을 계속하는 것이 낭비이자 위험이므로 즉시 세운다.
        if self.abort_on_fallback and len(self.rows) >= self._min_check:
            rate = sum(bool(r["fallback"]) for r in self.rows) / len(self.rows)
            if rate >= self.abort_threshold:
                raise RuntimeError(
                    f"[llm] 지휘관 '{self.model or self.backend}' 폴백률 {rate:.0%} "
                    f"({len(self.rows)}회 중) — LLM 이 응답하지 않는다. 수집을 중단한다.\n"
                    f"       마지막 사유: {rat[:160]}")
        return row

    def _flush(self, row: dict) -> None:
        """관측 1건을 JSONL 에 append. 기록 실패가 평가를 멈추게 두지는 않는다."""
        import json
        import os
        try:
            os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    @staticmethod
    def load_jsonl(path: str):
        """중단·재개로 여러 번에 걸쳐 쌓인 JSONL 을 DataFrame 으로. 깨진 줄은 건너뛴다."""
        import json
        import os
        import pandas as pd
        if not os.path.exists(path):
            return pd.DataFrame()
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # append 도중 죽어 잘린 마지막 줄
        return pd.DataFrame(rows)

    def reset_episode(self) -> None:
        """에피소드 경계. churn 이 에피소드를 가로질러 계산되지 않게 한다."""
        self._prev_assign = {}

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows)

    def summary(self) -> dict:
        """논문 표에 그대로 넣을 수 있는 요약. 표본이 없으면 전부 NaN."""
        import pandas as pd
        df = self.to_frame()
        if df.empty:
            return {}
        lat = df["latency_s"].dropna()
        return {
            "backend": self.backend,
            "model": self.model,
            "n_calls": int(len(df)),
            "latency_p50_s": float(lat.median()) if len(lat) else np.nan,
            "latency_p95_s": float(lat.quantile(0.95)) if len(lat) else np.nan,
            "fallback_rate": float(df["fallback"].mean()),
            "coverage_mean": float(df["coverage"].mean(skipna=True)),
            "churn_mean": float(df["churn"].mean(skipna=True)),
            "crossings_mean": float(df["crossings"].mean()),
            "heuristic_agreement": float(df["heuristic_agreement"].mean(skipna=True)),
            "hold_mean": float(df["n_hold"].mean()),
        }
