"""평가 전량 수집 드라이버 — 남은 (지휘관 × 포메이션) 블록을 순서대로 끝까지 돌린다.

왜 별도 드라이버인가
    LLM 조건은 블록(30 시드) 하나에 ~30분이 걸리는데, 대화형 도구에서 띄운 백그라운드
    작업은 그보다 일찍 끊긴다. 그래서 이 스크립트를 **분리 프로세스로 한 번 띄워** 두고
    전량을 순차 처리한다. run_eval.py 가 에피소드 단위로 체크포인트를 남기므로
    중간에 죽어도 다시 띄우면 이어서 간다.

    한 블록이 실패해도 다음 블록으로 넘어간다 — 한 지휘관의 API 문제로 전체가 멈추면 안 된다.

사용:
    python run_eval_all.py                    # 남은 것 전부
    python run_eval_all.py --dry-run          # 무엇이 남았는지만 출력
    python run_eval_all.py --only qwen2.5-7b  # 특정 지휘관만

병렬 실행:
    로컬(GPU)과 클라우드(네트워크)는 자원이 겹치지 않아 동시에 돌려도 서로 느려지지
    않는다. 다만 **출력 디렉터리를 반드시 분리해야 한다** — 같은 episodes.csv 에 두
    프로세스가 append 하면 컬럼이 어긋나고(과거 실제 발생) 그림도 서로 덮어쓴다.
    run_eval.py 의 실행 락도 out 디렉터리 단위라 분리하면 자연히 갈린다.

        python run_eval_all.py --only qwen2.5-14b
        python run_eval_all.py --only gemini-3.5-flash-lite                --out results/eval_gemini --figdir 논문_그래프/_gemini

    끝난 뒤 episodes.csv 를 이어 붙여 하나로 합치고 그림을 다시 생성한다:
        python run_eval.py --from-csv results/eval/episodes.csv --figdir 논문_그래프
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "results", "eval")
CSV = os.path.join(OUT, "episodes.csv")
FIGDIR = os.path.join(REPO, "논문_그래프")
SEEDS = 30
FORMATIONS = ("concentrated", "diversionary", "wave")
#: (백엔드, 모델) — 조건 키의 태그는 harness.slugify_model 이 만든다.
#   비교 축:
#     ① 로컬 모델 스케일   qwen2.5 7b vs 14b   (비용 0, 오프라인 배치 가능성)
#     ② 로컬 vs 클라우드   qwen2.5 14b vs gemini-3.5-flash-lite
#   ※ 클라우드는 예산상 1종만 둔다. 따라서 "클라우드 등급 간" 비교는 못 하고,
#     로컬 사다리(7b→14b)가 클라우드 경량에 닿는지만 본다.
#   ※ openai 는 크레딧 소진으로 제외했다. 충전하면 여기에 다시 넣으면 된다 —
#     실호출 프로브(run_eval._probe_commander)가 통과해야만 수집이 시작된다.
#   ※ 순서가 곧 실행 순서다. 무료(로컬)를 앞에, 과금(클라우드)을 뒤에 둔다 —
#     앞에서 문제가 드러나면 과금 전에 멈출 수 있다.
#   비용 근거(실측): 입력 4,146 tok/호출 × 4,299 호출/모델 = 17.8M tok.
#     (호출/에피소드 23.9 는 qwen2.5-7b 180 에피소드 실측값이다.)
#     시스템 프롬프트가 매 호출 동일하게 3,200 tok 들어가 입력이 전체의 84% 다.
#     → 3.5-flash-lite ~$2 (총 예상 과금)
#   Gemini 세대 접근성(실호출 프로브로 확인, 2026-08-31):
#     2.5-*            → 404 "no longer available to new users"
#     3-flash-preview  → 스키마 위반(ValidationError)
#     3.1-flash-lite   → OK   ·  3.5-flash-lite → OK  ·  3.6-flash → OK
#     3.7-flash, *-latest → ServerError
#   3.1-flash-lite / 3.6-flash 는 예산 조정으로 제외했다(2026-09-01, 사용자 지시).
#     되살리려면 아래 튜플에 다시 넣기만 하면 된다 — 둘 다 프로브를 통과했으므로
#     코드 변경은 그것뿐이다.
COMMANDERS = (
    ("ollama", "qwen2.5:7b"),
    ("ollama", "qwen2.5:14b"),
    ("gemini", "gemini-3.5-flash-lite"),
)


def coverage() -> dict:
    """(condition, formation) → 완료 시드 수."""
    if not os.path.exists(CSV):
        return {}
    import pandas as pd
    d = pd.read_csv(CSV)
    g = d.groupby(["condition", "formation"])["seed"].nunique()
    return {k: int(v) for k, v in g.items()}


def remaining(only: str | None = None) -> list[tuple[str, str, str]]:
    """아직 30 시드를 못 채운 (백엔드, 모델, 포메이션) 목록."""
    from boatattack_sim.eval.harness import slugify_model
    cov = coverage()
    todo = []
    for backend, model in COMMANDERS:
        tag = slugify_model(backend, model)
        if only and only != tag:
            continue
        for form in FORMATIONS:
            need = [f"llm_heur@{tag}", f"llm_unet@{tag}"]
            if any(cov.get((c, form), 0) < SEEDS for c in need):
                todo.append((backend, model, form))
    return todo


def main() -> int:
    sys.path.insert(0, REPO)
    global OUT, CSV, FIGDIR
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    # 출력 분리 — 병렬 실행 시 필수(모듈 주석 참조).
    if "--out" in sys.argv:
        OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
        CSV = os.path.join(OUT, "episodes.csv")
    if "--figdir" in sys.argv:
        FIGDIR = os.path.abspath(sys.argv[sys.argv.index("--figdir") + 1])
    print(f"[all] out={OUT}", flush=True)
    print(f"[all] fig={FIGDIR}", flush=True)

    todo = remaining(only)
    if not todo:
        print("[all] 남은 블록 없음 — 이미 전량 완료")
        return 0
    print(f"[all] 남은 블록 {len(todo)}개:")
    for b, m, f in todo:
        print(f"        {b}:{m}  {f}")
    if "--dry-run" in sys.argv:
        return 0

    t0 = time.perf_counter()
    for i, (backend, model, form) in enumerate(todo, 1):
        print(f"\n[all] ({i}/{len(todo)}) {backend}:{model} · {form}", flush=True)
        cmd = [sys.executable, os.path.join(REPO, "run_eval.py"),
               "--commanders", f"{backend}:{model}",
               "--seeds", str(SEEDS), "--replan-every", "1",
               "--formations", form,
               "--out", OUT, "--figdir", FIGDIR,
               "--quiet", "--force-lock"]
        r = subprocess.run(cmd, cwd=REPO)
        status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        cov = coverage()
        from boatattack_sim.eval.harness import slugify_model
        tag = slugify_model(backend, model)
        got = [cov.get((f"llm_{k}@{tag}", form), 0) for k in ("heur", "unet")]
        print(f"[all] → {status} · 확보 {got} / {SEEDS}", flush=True)

    left = remaining(only)
    print(f"\n[all] 경과 {(time.perf_counter() - t0) / 60:.1f}분 · "
          f"남은 블록 {len(left)}개")
    if left:
        print("[all] 다시 실행하면 이어서 진행한다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
