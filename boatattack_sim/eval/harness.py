"""평가 하네스 — 조건행렬 × 시드 × 포메이션을 돌려 에피소드별 지표를 긴 형식(long)으로 낸다.

논문 표/그림의 단일 소스다. 여기서 나온 CSV 만이 plots.py 의 입력이며,
숫자를 손으로 옮겨 적는 경로는 만들지 않는다(전사 오류 차단).

핵심 설계 — **시드 대응표본(paired)**
    모든 조건이 **같은 시드 집합**을 쓴다. seed=k 의 전장은 조건이 달라도 동일하므로
    조건 간 차이에서 시나리오 난이도 노이즈가 상쇄된다. stats.py 의 paired bootstrap 이
    이걸 전제로 하며, 덕분에 독립표본 대비 훨씬 적은 시드로 유의한 CI 가 나온다.
    → 시드를 조건마다 다르게 주면 통계가 통째로 무효가 된다. 절대 그러지 말 것.

조건행렬 (배정 × 기동)
    heur_heur : 휴리스틱 배정 + 휴리스틱 기동   ← baseline
    heur_unet : 휴리스틱 배정 + U-Net 점수맵     ← 기동 계층의 순수 기여
    llm_heur  : LLM 지휘관   + 휴리스틱 기동     ← 전략 계층의 순수 기여
    llm_unet  : LLM 지휘관   + U-Net 점수맵      ← 제안 (상호작용 포함)
    2×2 이므로 두 계층의 **주효과와 상호작용**을 분리할 수 있다(stats.interaction 참조).

실행: run_eval.py 를 쓴다. 이 모듈을 직접 부르는 건 커스텀 실험을 짤 때만.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# ── 지표 계약 ────────────────────────────────────────────────────────────────
#   _ev 의 항목은 세 부류이고 누적 방식이 다르다. 섞으면 조용히 틀린 값이 나온다.
#     SCALAR_ADD : [N]     누적 합    (결정 윈도우마다 리셋되므로 델타를 더한다)
#     SHIP_ADD   : [N,P]   누적 합    (배별 → 팀 합으로 집계)
#     SHIP_MIN   : [N,P]   최소 추적  (윈도우별 min 의 min = 전역 min)
SCALAR_ADD = ("captures", "breaches", "ally_collisions", "obstacle_collisions",
              "nets_used", "path_dist", "net_touches", "land_collisions",
              "cap_dist_sum", "cap_dist_n",
              "cap_t_sum", "net_edist_sum", "net_edist_n")
SHIP_ADD = ("traveled", "turn_sum")
SHIP_MIN = ("mother_dmin", "ally_dmin")


@dataclass(frozen=True)
class Condition:
    """평가 조건 하나. `key` 가 CSV 의 조건 컬럼이 된다."""
    key: str
    assign: str      # "heuristic" | "llm"
    maneuver: str    # "heuristic" | "policy"
    label: str       # 그림에 찍히는 한국어 라벨
    backend: str = ""   # LLM 조건에서 어느 지휘관인가 ("openai" | "ollama"). 휴리스틱이면 ""


def slugify_model(backend: str, model: str) -> str:
    """`ollama` + `qwen2.5:7b` → `qwen2.5-7b`. 조건 키에 넣을 수 있는 짧은 태그.

    모델명을 태그로 쓰는 이유: 같은 백엔드 안에서 **모델만 바꾼 비교**(7b vs 14b)를
    해야 하므로 백엔드 이름만으로는 조건이 충돌한다. `:` `/` 는 키·파일명에 쓰기 어렵다.
    """
    m = (model or backend).strip()
    for ch in (":", "/", " ", "@"):
        m = m.replace(ch, "-")
    return m


def with_backend(cond: Condition, backend: str, model: str = "") -> Condition:
    """LLM 조건을 특정 지휘관(백엔드+모델)에 묶은 사본.

    key 에 `@<모델태그>` 를 붙여 조건을 분리한다 — 같은 배정 계층을 다른 지휘관으로
    돌려 비교하기 위한 것이다(예: `llm_unet@gpt-4o-mini` vs `llm_unet@qwen2.5-7b`).
    휴리스틱 조건은 지휘관과 무관하므로 그대로 돌려준다 — 복제하면 같은 결과가 중복 집계된다.
    """
    if cond.assign != "llm":
        return cond
    tag = slugify_model(backend, model)
    return Condition(f"{cond.key}@{tag}", cond.assign, cond.maneuver,
                     f"{cond.label} ({tag})", tag)


#: 논문 본표의 4개 조건. 순서가 그림의 막대 순서다(baseline → 제안).
MATRIX_2x2: tuple[Condition, ...] = (
    Condition("heur_heur", "heuristic", "heuristic", "휴리스틱 배정 + 휴리스틱 기동"),
    Condition("heur_unet", "heuristic", "policy",    "휴리스틱 배정 + U-Net 점수맵"),
    Condition("llm_heur",  "llm",       "heuristic", "LLM 지휘관 + 휴리스틱 기동"),
    Condition("llm_unet",  "llm",       "policy",    "LLM 지휘관 + U-Net 점수맵"),
)

#: 적 포메이션. 논문은 셋 다 보고한다 — 어느 하나만 쓰면 결론이 포메이션에 종속된다.
FORMATIONS: tuple[str, ...] = ("concentrated", "diversionary", "wave")


# ── 에피소드 러너 ────────────────────────────────────────────────────────────
def _new_acc(env):
    return ({k: 0.0 for k in SCALAR_ADD} | {k: 0.0 for k in SHIP_ADD}
            | {k: np.inf for k in SHIP_MIN})


def run_episode(env, seed: int, *, max_steps: int | None = None,
                on_decision: Callable[[object, int], None] | None = None) -> dict:
    """에피소드 1회를 끝까지 돌리고 지표 dict 를 반환.

    `on_decision(env, step)` 은 **결정 직전**에 불린다 — LLM 조건에서 계획을 갱신하는 자리다.
    (env.step 이 내부에서 decision_period 마다 _rl_decide 를 부르므로, 그 전에 주입해야
     이번 결정에 반영된다.)
    """
    env.reset(seed=int(seed))
    cap = int(max_steps or env.cfg.max_steps) + 1
    period = int(env.cfg.decision_period)

    acc = _new_acc(env)
    prev = {k: 0.0 for k in SCALAR_ADD} | {k: 0.0 for k in SHIP_ADD}
    n_dec = 0
    t0 = time.perf_counter()
    steps = 0
    while steps < cap and not bool(env.done[0]):
        is_dec = (env._micro_ct % period == 0)
        if is_dec:
            if on_decision is not None:
                on_decision(env, steps)
            # _rl_decide 가 _ev 를 새로 만든다 → 델타 기준선을 0 으로 되돌린다.
            for k in prev:
                prev[k] = 0.0
            n_dec += 1
        env.step()
        ev = env._ev
        for k in SCALAR_ADD:
            cur = float(ev[k][0]); acc[k] += cur - prev[k]; prev[k] = cur
        for k in SHIP_ADD:
            cur = float(np.asarray(ev[k][0]).sum()); acc[k] += cur - prev[k]; prev[k] = cur
        for k in SHIP_MIN:
            v = np.asarray(ev[k][0], float)
            v = v[np.isfinite(v)]
            if v.size:
                acc[k] = min(acc[k], float(v.min()))
        steps += 1

    wall = time.perf_counter() - t0
    capn, brn = acc["captures"], acc["breaches"]
    denom = capn + brn
    n_enemy = int(env.M)
    P = int(env.P)              # 아군 척수 — 충돌'율'의 분모
    return {
        "seed": int(seed),
        "steps": steps,
        "decisions": n_dec,
        "wall_s": round(wall, 4),
        # ── 1차 지표 ──
        "captures": capn,
        "breaches": brn,
        # ★ 포획률 = N_cap/(N_cap+N_brc). 분모 0(교전 성립 안 함)은 NaN 으로 둔다 —
        #   0 으로 채우면 "전부 놓쳤다"로 읽혀 평균이 아래로 편향된다.
        "capture_rate": (capn / denom) if denom > 0 else np.nan,
        "survived": int(env.e_alive[0].sum()),
        # ★ 교전 종결률: 스폰된 적 중 실제로 결판난 비율. capture_rate 가 분모를 좁히는 만큼
        #   이걸 같이 봐야 "적게 만나서 높은 포획률"을 걸러낼 수 있다.
        "resolved_frac": denom / n_enemy if n_enemy else np.nan,
        # ── 안전 (충돌은 '건수'와 '율'을 함께 낸다) ──
        #   율의 분모는 아군 척수 P. 즉 "에피소드당 배 한 척이 사고 날 확률"이라
        #   척수가 다른 설정끼리도 비교된다.
        "ally_collisions": acc["ally_collisions"],
        "obstacle_collisions": acc["obstacle_collisions"],
        "land_collisions": acc["land_collisions"],
        "net_touches": acc["net_touches"],
        #: 아군끼리 + 모선 충돌 (사용자 정의의 '충돌'). 그물 접촉·좌초는 성격이 달라 뺀다.
        "collisions": acc["ally_collisions"] + acc["obstacle_collisions"],
        "collision_rate": (acc["ally_collisions"] + acc["obstacle_collisions"]) / max(P, 1),
        #: 모든 손실 원인(그물 접촉·좌초 포함)
        "ally_losses": acc["ally_collisions"] + acc["obstacle_collisions"]
                       + acc["land_collisions"] + acc["net_touches"],
        "loss_rate": (acc["ally_collisions"] + acc["obstacle_collisions"]
                      + acc["land_collisions"] + acc["net_touches"]) / max(P, 1),
        "mother_dmin": acc["mother_dmin"] if np.isfinite(acc["mother_dmin"]) else np.nan,
        "ally_dmin": acc["ally_dmin"] if np.isfinite(acc["ally_dmin"]) else np.nan,
        # ── 자원 효율 (낮을수록 좋음) ──
        "nets_used": acc["nets_used"],
        #: ★ 그물 1장당 포획 수. 자원 효율의 본체 — 그물을 많이 써서 올린 포획률을 걸러낸다.
        "captures_per_net": (capn / acc["nets_used"]) if acc["nets_used"] > 0 else np.nan,
        #: 포획 1척당 소모 그물. 적을수록 효율적(사용자 요청 방향과 부호가 맞는다).
        "nets_per_capture": (acc["nets_used"] / capn) if capn > 0 else np.nan,
        # ── 기동 비용 (낮을수록 좋음) ──
        "traveled_m": acc["traveled"],
        #: 포획 1척을 위해 팀이 움직인 거리. 연료·시간 비용의 대리 지표.
        "traveled_per_capture": (acc["traveled"] / capn) if capn > 0 else np.nan,
        "turn_sum_rad": acc["turn_sum"],
        # ── 교전 품질 ──
        #: 포획 거리: 모선에서 멀리 잡을수록 종심이 깊다(공간축 조기 제압). 포획 0 이면 NaN.
        "cap_dist_mean": (acc["cap_dist_sum"] / acc["cap_dist_n"]
                          if acc["cap_dist_n"] > 0 else np.nan),
        #: ★ 평균 포획 시각(step). **낮을수록 조기 제압**(시간축).
        #   거리축과 짝이다 — 멀리서 잡아도 늦으면 다음 파를 못 막는다.
        "cap_time_mean": (acc["cap_t_sum"] / acc["cap_dist_n"]
                          if acc["cap_dist_n"] > 0 else np.nan),
        #: 에피소드 길이로 정규화한 제압 속도. 0=즉시, 1=끝까지 끌림. 길이가 다른 에피소드끼리 비교용.
        "cap_time_frac": (acc["cap_t_sum"] / acc["cap_dist_n"] / steps
                          if acc["cap_dist_n"] > 0 and steps > 0 else np.nan),
        #: ★ 그물 전개 개시 시 최근접 적까지 거리 (m). **낮을수록 좋다** —
        #   그물은 물에 고정되지 않아 표류하므로 적에 붙여 뿌려야 유효하다.
        "net_deploy_edist": (acc["net_edist_sum"] / acc["net_edist_n"]
                             if acc["net_edist_n"] > 0 else np.nan),
    }


# ── 조건별 환경 구성 ─────────────────────────────────────────────────────────
def make_env(ckpt: str, formation: str, *, nets_per_ship: int = 3,
             device: str = "cpu", avoid_steer: bool = False):
    """평가용 환경. 모든 조건이 **같은 cfg** 를 써야 비교가 성립한다.

    조건별로 바뀌는 건 `assign_source` / `maneuver_source` 두 스위치뿐이다.
    체크포인트를 조건마다 다르게 로드하거나 cfg 를 손대면 그 순간 비교가 무효다.
    """
    from commander.unet_bridge import CommandedCnnEnv
    return CommandedCnnEnv(ckpt, enemy_mode=formation, device=device,
                           avoid_steer=avoid_steer, nets_per_ship=nets_per_ship)


def _llm_hook(commander, replan_every: int, recorder=None):
    """LLM 조건의 on_decision 훅. `replan_every` **결정**마다 계획을 갱신한다.

    ★ 단위 주의 — 여기서 세는 단위는 '결정'(= decision_period 25 step)이지 step 이 아니다.
      실배포 경로(`run_commander_ui.py --replan N`)의 N 은 **step** 단위이고, 권장 사용값
      `--replan 25` 는 곧 매 결정 재계획(= 여기서 replan_every=1)에 해당한다.
      따라서 **replan_every=1 이 배포 시스템과 같은 설정**이다.

      replan_every>1 로 두면 API 호출은 그 배수만큼 줄지만, 배정 계획이 그만큼 낡는다.
      휴리스틱 배정(`_compute_assignment`)은 매 결정 새로 계산되므로, replan_every>1 은
      'LLM 대 휴리스틱'이 아니라 '낡은 LLM 계획 대 최신 휴리스틱'을 비교하게 만든다.
      비용 절감 목적이라면 그 자체를 **재계획 주기 ablation** 으로 따로 보고할 것.
    """
    from commander.rl_bridge import build_battlefield_defense
    state_n = {"i": 0}

    def hook(env, _step):
        if state_n["i"] % max(1, int(replan_every)) == 0:
            st = build_battlefield_defense(env, None)
            plan = commander.plan(st)
            if recorder is not None:
                recorder.observe(env, st, plan)
            env.set_plan(plan, None)
        state_n["i"] += 1

    return hook


def run_condition(cond: Condition, seeds: Sequence[int], formation: str, ckpt: str, *,
                  commander=None, replan_every: int = 1, recorder=None,
                  progress: Callable[[str], None] | None = None,
                  sink: Callable[[dict], None] | None = None,
                  **env_kw) -> list[dict]:
    """조건 하나를 시드 전체에 대해 돌린다. 환경은 한 번만 만들어 재사용(reset 으로 초기화)."""
    if cond.assign == "llm" and commander is None:
        raise ValueError(f"조건 {cond.key} 는 LLM 배정인데 commander 가 없습니다. "
                         f"run_eval.py --backends 로 지정하거나 --skip-llm 을 쓰세요.")
    env = make_env(ckpt, formation, **env_kw)
    env.assign_source = cond.assign
    env.maneuver_source = cond.maneuver
    hook = _llm_hook(commander, replan_every, recorder) if cond.assign == "llm" else None

    rows = []
    for s in seeds:
        if cond.assign == "llm":
            env.set_plan(None, None)          # 시드마다 계획을 비워 이전 에피소드가 새지 않게
        r = run_episode(env, s, on_decision=hook)
        r |= {"condition": cond.key, "assign": cond.assign, "maneuver": cond.maneuver,
              "label": cond.label, "formation": formation,
              # ★ 실험 설정을 원자료에 박아 둔다. 이 값이 다르면 조건이 사실상 다르므로
              #   같은 CSV 에 섞어 놓고 비교하면 안 된다(재계획 주기 = LLM 계획의 신선도).
              "replan_every": int(replan_every) if cond.assign == "llm" else 0}
        r["backend"] = cond.backend
        rows.append(r)
        if sink is not None:
            sink(r)                       # 에피소드 1건마다 즉시 기록 — 끊겨도 이것까지는 남는다
        if recorder is not None:
            recorder.reset_episode()      # churn 이 에피소드 경계를 넘지 않게
        if progress:
            progress(f"  {cond.key:<18} {formation:<13} seed={s:<4} "
                     f"cap={r['captures']:.0f} br={r['breaches']:.0f} "
                     f"rate={r['capture_rate']:.3f}")
    return rows


def run_matrix(ckpt: str, seeds: Sequence[int], *,
               conditions: Iterable[Condition] = MATRIX_2x2,
               formations: Iterable[str] = FORMATIONS,
               commander=None, commanders: dict | None = None,
               replan_every: int = 1, recorder=None, recorders: dict | None = None,
               progress: Callable[[str], None] | None = None,
               checkpoint: str | None = None, resume: bool = True,
               **env_kw):
    """조건행렬 전체 → pandas.DataFrame (에피소드 1행 = long format).

    `commanders` = {태그: commander} 를 주면 조건의 `backend` 태그로 골라 쓴다
    (로컬 vs GPT, 또는 7b vs 14b 동시 비교). 단일 지휘관이면 `commander` 하나만 줘도 된다.
    `recorders` = {태그: LLMRecorder} 도 같은 방식으로 분리 기록한다.

    ★ `checkpoint` 경로를 주면 **(포메이션, 조건) 묶음이 끝날 때마다 즉시 append** 한다.
      LLM 조건은 한 번 돌리는 데 수십 분이 걸리고 API 비용도 든다 — 중간에 끊겼을 때
      처음부터 다시 돌리는 것은 시간과 돈을 함께 버리는 일이다.
      `resume=True` 면 이미 파일에 있는 (조건, 포메이션, 시드) 조합은 건너뛴다.
    """
    import os
    import pandas as pd

    done: set[tuple] = set()
    prior: list = []
    if checkpoint and resume and os.path.exists(checkpoint):
        try:
            old = pd.read_csv(checkpoint)
            # ★ (condition, formation, seed) 중복 제거. 같은 출력 폴더에 두 실행이 겹치면
            #   같은 시드가 두 번 들어가고, 그러면 그 시드가 집계에서 두 배 가중된다.
            #   시뮬은 결정적이라 값은 같지만 가중치가 틀어지므로 반드시 접는다.
            n0 = len(old)
            old = old.drop_duplicates(["condition", "formation", "seed"], keep="first")
            if len(old) < n0 and progress:
                progress(f"[resume] ⚠ 중복 {n0 - len(old)} 행 제거 "
                         f"(같은 출력 폴더에서 두 실행이 겹쳤을 때 생긴다)")
                old.to_csv(checkpoint, index=False, encoding="utf-8-sig")
            prior = [old]
            done = set(map(tuple, old[["condition", "formation", "seed"]].to_numpy()))
            if progress:
                progress(f"[resume] 기존 {len(old)} 에피소드 재사용 → {checkpoint}")
        except Exception as e:      # 손상된 체크포인트로 전체를 막지는 않는다
            if progress:
                progress(f"[resume] 체크포인트 읽기 실패({e}) → 처음부터")

    rows: list[dict] = []
    for formation in formations:
        for cond in conditions:
            todo = [s for s in seeds if (cond.key, formation, int(s)) not in done]
            if not todo:
                if progress:
                    progress(f"[{formation}] {cond.key} — 이미 완료, 건너뜀")
                continue
            cmd = commander
            rec = recorder
            if cond.assign == "llm" and commanders:
                cmd = commanders.get(cond.backend, commander)
                if recorders:
                    rec = recorders.get(cond.backend, recorder)
            if progress:
                progress(f"[{formation}] {cond.key} ({len(todo)} seeds)")
            # ★ 에피소드 1건마다 append. 묶음 단위로 쓰면 중단 시 최대 30 에피소드(LLM 조건은
            #   API 비용까지)를 버린다 — 실제로 반복해서 겪었다. 손실 창을 1건으로 줄인다.
            #
            # ★★ append 는 **반드시 기존 헤더 순서에 맞춰** 쓴다. to_csv(mode="a") 는 값을
            #    자리(positional)로 붙이므로, dict 키 순서가 파일 헤더와 다르면 열이 통째로
            #    밀려 조용히 오염된다 (실제로 replan_every 칸에 모델명이 들어간 사고가 났다).
            #    파일에 없는 새 키가 생기면 그건 스키마 변경이므로 덧붙이지 않고 세운다.
            def _sink(r, _ck=checkpoint, _cols=[]):
                if not _ck:
                    return
                os.makedirs(os.path.dirname(_ck) or ".", exist_ok=True)
                if not os.path.exists(_ck):
                    pd.DataFrame([r]).to_csv(_ck, mode="w", header=True,
                                             index=False, encoding="utf-8-sig")
                    _cols[:] = list(r.keys())
                    return
                if not _cols:
                    _cols[:] = list(pd.read_csv(_ck, nrows=0).columns)
                extra = set(r) - set(_cols)
                if extra:
                    raise ValueError(
                        f"체크포인트 스키마 불일치: 기존 파일에 없는 컬럼 {sorted(extra)}. "
                        f"열이 밀려 데이터가 오염되므로 중단한다. "
                        f"--no-resume 으로 새로 시작하거나 기존 CSV 를 치울 것: {_ck}")
                pd.DataFrame([{c: r.get(c) for c in _cols}]).to_csv(
                    _ck, mode="a", header=False, index=False, encoding="utf-8-sig")

            got = run_condition(cond, todo, formation, ckpt, commander=cmd,
                                replan_every=replan_every, recorder=rec,
                                progress=progress,
                                sink=(_sink if checkpoint else None), **env_kw)
            rows += got
    new = pd.DataFrame(rows) if rows else pd.DataFrame()
    frames = [f for f in (prior + ([new] if len(new) else [])) if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else new
