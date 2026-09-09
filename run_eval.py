"""평가 실행기 — 조건행렬을 돌려 CSV·그림·LaTeX 표를 한 번에 낸다.

논문 숫자의 단일 진입점이다. 여기서 나온 산출물만 논문에 넣는다.

실행 예
    # LLM 없이 기동 계층만 (API 불필요, 가장 빠름)
    python run_eval.py --skip-llm --seeds 30

    # 로컬(Ollama)과 GPT 를 **동시에** 비교 — 지휘관 축이 하나 더 생긴다
    python run_eval.py --backends ollama,openai --seeds 30 --replan-every 4

    # 단일 백엔드
    python run_eval.py --backends openai --model gpt-4o-mini --seeds 30

    # 이미 돌린 CSV 로 그림만 다시
    python run_eval.py --from-csv results/eval/episodes.csv

산출물
    results/eval/episodes.csv     에피소드 1행 = 원자료. 모든 그림·표가 여기서만 나온다
    results/eval/summary.csv      조건별 평균 CI + baseline 대비 대응차이
    results/eval/interaction.txt  2×2 주효과·상호작용 분해 (백엔드별)
    results/eval/table_main.tex   논문 본표 (그대로 \\input)
    results/eval/llm_metrics.csv  LLM 조건이 있을 때
    논문_그래프/                   ★ 그림 전부 (PDF + PNG) — 보기 편하도록 최상위에 모은다

주의 — 모든 조건이 같은 시드를 쓴다(대응표본). 시드를 조건별로 바꾸면 통계가 무효다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

#: 백엔드별 기본 모델. --model 로 덮어쓸 수 있다(단일 백엔드일 때만 의미가 있다).
DEFAULT_MODEL = {"openai": "gpt-4o-mini", "ollama": "qwen2.5:7b",
                 "gemini": "gemini-2.5-flash"}


def load_dotenv(path: str = ".env") -> None:
    """`.env` 의 KEY=VALUE 를 환경변수로 올린다(이미 있으면 덮지 않는다).

    API 키를 소스에 박지 않기 위한 것이다. `.env` 는 gitignore 되어 있다.
    외부 의존(python-dotenv) 없이 최소 파싱만 한다 — 따옴표와 주석만 처리.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def parse_commander_specs(spec: str, fallback_model: str | None) -> list[tuple[str, str]]:
    """`ollama:qwen2.5:7b,openai` → [("ollama","qwen2.5:7b"), ("openai","gpt-4o-mini")].

    백엔드만 적으면 기본 모델을 쓴다. 모델까지 적으면 **같은 백엔드 안에서도 여러 모델**을
    비교할 수 있다(예: `ollama:qwen2.5:7b,ollama:qwen2.5:14b`) — 모델 스케일 비교용.
    백엔드 이름 뒤 첫 번째 `:` 만 구분자로 보고 나머지는 모델명에 그대로 둔다
    (`qwen2.5:7b` 처럼 모델명 자체에 `:` 가 들어가기 때문).
    """
    out: list[tuple[str, str]] = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        if ":" in item:
            b, m = item.split(":", 1)
        else:
            b, m = item, ""
        b = b.strip()
        m = m.strip() or fallback_model or DEFAULT_MODEL.get(b, "")
        if not m:
            raise SystemExit(f"[eval] 백엔드 '{b}' 의 기본 모델을 모른다 — "
                             f"`{b}:<모델명>` 형식으로 지정하라")
        out.append((b, m))
    if not out:
        raise SystemExit("[eval] --commanders 가 비어 있다")
    return out


def _probe_commander(base, ckpt: str, nets: int) -> tuple[bool, str]:
    """지휘관에게 **실제 계획을 한 번 요청**해 LLM 이 정말 응답하는지 본다.

    ★ 왜 초기화 확인만으로 부족한가: 크레딧 소진·모델 권한 없음·스키마 거부는 전부
      **호출 시점**에 터진다. 클라이언트 객체는 멀쩡히 만들어지므로 `client is None`
      검사를 통과한다. 그 상태로 돌리면 매 호출이 조용히 휴리스틱으로 떨어지고,
      결과 CSV 에는 'LLM 조건'이라는 이름만 남는다.
      (실제로 4173 호출이 전량 폴백된 채 수집이 끝난 적이 있다 — 그래서 이 검사를 넣었다.)

    반환 (성공 여부, 사유).
    """
    try:
        from commander.rl_bridge import build_battlefield_defense
        from commander.unet_bridge import CommandedCnnEnv
        env = CommandedCnnEnv(ckpt, enemy_mode="diversionary", nets_per_ship=nets)
        env.assign_source = "heuristic"
        env.reset(seed=0)
        for _ in range(60):          # 클러스터가 생길 만큼만 진행
            env.step()
        plan = base.plan(build_battlefield_defense(env, None))
    except Exception as e:           # 프로브 자체가 깨지면 그것도 알려야 한다
        return False, f"프로브 예외 {type(e).__name__}: {e}"
    rat = str(getattr(plan, "rationale", "") or "")
    if "휴리스틱 방어" in rat:        # commander/*_commander.py::_fallback 이 붙이는 표식
        return False, rat.split("→")[0].strip() or "폴백"
    return True, "ok"


def _build_commanders(specs, log, jsonl_dir: str | None = None,
                      ckpt: str = "", nets: int = 3, probe: bool = True):
    """{태그: (commander, recorder)}. 초기화 **와 실호출** 실패가 여기서 드러나야 한다.

    지휘관이 폴백 전용 모드로 조용히 떨어지면 '휴리스틱을 LLM 이라 부르며' 평가하게 된다.
    """
    from boatattack_sim.eval.harness import slugify_model
    from commander import make_commander
    from commander.llm_metrics import LLMRecorder
    out = {}
    for b, mdl in specs:
        base = make_commander(b, mdl, verbose=False)
        if getattr(base, "client", "?") is None:
            raise SystemExit(
                f"[eval] 지휘관 '{b}:{mdl}' 초기화 실패 — 폴백(휴리스틱) 전용 모드다.\n"
                f"        이 상태로 돌리면 LLM 조건이 사실상 휴리스틱이 되어 결과가 거짓이 된다.\n"
                f"        openai → OPENAI_API_KEY 확인 / ollama → `pip install ollama` 와 서버 확인")
        if probe and ckpt:
            ok, why = _probe_commander(base, ckpt, nets)
            if not ok:
                raise SystemExit(
                    f"[eval] 지휘관 '{b}:{mdl}' 실호출 실패 — 계획이 휴리스틱으로 폴백됐다.\n"
                    f"        사유: {why}\n"
                    f"        이 상태의 수집은 'LLM 조건'이라는 이름만 붙은 휴리스틱 결과다.\n"
                    f"        --no-probe 로 무시할 수 있으나, 그 데이터는 논문에 쓸 수 없다.")
            log(f"[eval] 실호출 확인: {slugify_model(b, mdl)} → LLM 응답 정상")
        tag = slugify_model(b, mdl)
        # 지휘관마다 별도 JSONL — 중단돼도 계측이 남고, 재개분이 자연히 이어 붙는다.
        jp = os.path.join(jsonl_dir, f"llm_calls_{tag}.jsonl") if jsonl_dir else None
        rec = LLMRecorder(backend=b, model=tag, jsonl_path=jp)
        out[tag] = (rec.wrap(base), rec)
        log(f"[eval] 지휘관 준비: {tag}  ({b} / {mdl})")
    return out


def _pid_alive(pid: int) -> bool:
    """그 PID 가 아직 살아 있는가. 판단이 안 되면 **살아 있다고 본다**(보수적).

    잘못 '죽었다'고 보면 진짜 동시 실행을 통과시켜 데이터가 오염된다.
    잘못 '살았다'고 보면 사용자가 --force-lock 을 쓰면 그만이다. 손실이 작은 쪽으로 기운다.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import subprocess
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=10)
            return str(pid) in (out.stdout or "")
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


class _RunLock:
    """출력 폴더당 하나만 돌게 막는 락.

    ★ 왜 필요한가: 같은 `--out` 에 두 실행이 겹치면 두 프로세스가 같은 시드를 각자 돌려
      episodes.csv 에 중복 행을 남긴다. 값은 같아도 그 시드가 집계에서 **두 배 가중**되어
      평균과 CI 가 조용히 틀어진다. (실제로 겪었다 — 앞 실행이 끝나기 전에 다음 실행을 띄웠다.)
      파일 존재만으로 잠그고, 죽은 프로세스가 남긴 락은 --force-lock 으로 지운다.
    """

    def __init__(self, path: str, force: bool = False):
        self.path, self.force = path, force

    def __enter__(self):
        import json
        import time as _t
        if os.path.exists(self.path) and not self.force:
            try:
                with open(self.path, encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}
            # ★ 죽은 프로세스가 남긴 락은 자동 회수한다. 평가가 중간에 killed 되면 __exit__ 이
            #   안 돌아 락이 남는데, 그때마다 손으로 지우게 하면 --force-lock 을 습관적으로
            #   쓰게 되고 결국 진짜 동시 실행까지 통과시킨다. PID 생존만 확인하면 충분하다.
            pid = info.get("pid")
            if pid and not _pid_alive(int(pid)):
                print(f"[eval] 죽은 프로세스(pid={pid})의 락을 회수한다: {self.path}")
            else:
                raise SystemExit(
                    f"[eval] 이미 다른 평가가 이 폴더를 쓰고 있다: {self.path}\n"
                    f"        {info}\n"
                    f"        동시 실행은 같은 시드를 중복 기록해 집계를 틀어지게 한다.\n"
                    f"        정말 끝난 실행이라면 --force-lock 으로 무시하고 진행하라.")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started": _t.strftime("%Y-%m-%d %H:%M:%S"),
                       "argv": " ".join(sys.argv[1:])}, f, ensure_ascii=False)
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="MobRobGPT 방어체계 평가 하네스")
    ap.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    ap.add_argument("--seeds", type=int, default=30,
                    help="시드 수. 논문용은 30 이상 권장")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--formations", default="concentrated,diversionary,wave")
    ap.add_argument("--commanders", "--backends", dest="commanders", default="openai",
                    help="쉼표 구분 `백엔드[:모델]`. 예: ollama:qwen2.5:7b,ollama:qwen2.5:14b,"
                         "openai:gpt-4o-mini → 지휘관마다 조건이 하나씩 생긴다")
    ap.add_argument("--model", default=None,
                    help="모델을 안 적은 백엔드에 쓸 기본 모델")
    ap.add_argument("--skip-llm", action="store_true",
                    help="LLM 조건을 빼고 기동 계층 2조건만 (API 불필요)")
    ap.add_argument("--replan-every", type=int, default=1,
                    help="LLM 재계획 주기(**결정** 단위, step 아님). 기본 1 = 매 결정 재계획 "
                         "= 배포 경로의 `--replan 25` 와 동일. >1 로 올리면 API 호출은 줄지만 "
                         "계획이 낡아 휴리스틱(매 결정 재계산) 대비 불리해진다 — "
                         "그 자체를 보려면 재계획 주기 ablation 으로 따로 돌릴 것")
    ap.add_argument("--nets", type=int, default=3, help="배당 그물 수")
    ap.add_argument("--metric", default="capture_rate")
    ap.add_argument("--out", default="results/eval", help="CSV·표 저장 위치")
    ap.add_argument("--figdir", default="논문_그래프", help="그림 저장 폴더")
    ap.add_argument("--from-csv", default=None,
                    help="시뮬을 건너뛰고 이 CSV 로 그림·표만 다시 만든다")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="기존 episodes.csv 를 이어받지 않고 처음부터 (기존 파일은 .bak 보존)")
    ap.add_argument("--force-lock", action="store_true",
                    help="남아 있는 실행 락을 무시한다(앞 실행이 비정상 종료했을 때만)")
    ap.add_argument("--no-probe", dest="probe", action="store_false",
                    help="지휘관 실호출 사전검증을 건너뛴다(권장하지 않음 — "
                         "폴백 상태로 수집하면 'LLM 조건'이 사실상 휴리스틱이 된다)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    from boatattack_sim.eval import harness as H, plots as P, stats as S

    load_dotenv()          # GEMINI_API_KEY 등 — 소스에 키를 박지 않기 위해
    os.makedirs(args.out, exist_ok=True)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))

    llm_df = None
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        log(f"[eval] CSV 재사용: {args.from_csv} ({len(df)} 에피소드)")
        cand = os.path.join(os.path.dirname(args.from_csv), "llm_metrics.csv")
        if os.path.exists(cand):
            llm_df = pd.read_csv(cand)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        forms = [f.strip() for f in args.formations.split(",") if f.strip()]
        commanders = recorders = None

        if args.skip_llm:
            conds = tuple(c for c in H.MATRIX_2x2 if c.assign != "llm")
            log("[eval] --skip-llm: 휴리스틱 배정 2조건만 돌린다")
        else:
            specs = parse_commander_specs(args.commanders, args.model)
            built = _build_commanders(specs, log, jsonl_dir=args.out,
                                      ckpt=args.ckpt, nets=args.nets,
                                      probe=args.probe)
            commanders = {t: c for t, (c, _) in built.items()}
            recorders = {t: r for t, (_, r) in built.items()}
            # 휴리스틱 조건은 지휘관과 무관하므로 1벌만, LLM 조건은 지휘관마다 복제.
            conds = tuple(c for c in H.MATRIX_2x2 if c.assign != "llm")
            for b, mdl in specs:
                conds += tuple(H.with_backend(c, b, mdl)
                               for c in H.MATRIX_2x2 if c.assign == "llm")

        n_ep = len(seeds) * len(forms) * len(conds)
        log(f"[eval] 조건 {len(conds)} × 포메이션 {len(forms)} × 시드 {len(seeds)} "
            f"= {n_ep} 에피소드")
        ckpt_csv = os.path.join(args.out, "episodes.csv")
        # ★ 체크포인트를 episodes.csv 자체로 쓴다 — 중간에 끊겨도 남고, 다시 돌리면 이어간다.
        #   --no-resume 이면 기존 파일을 치우고 처음부터(이전 결과는 .bak 로 보존).
        if not args.resume and os.path.exists(ckpt_csv):
            bak = ckpt_csv + ".bak"
            os.replace(ckpt_csv, bak)
            log(f"[eval] --no-resume: 기존 결과를 {bak} 로 옮기고 처음부터")
        t0 = time.perf_counter()
        # 락은 시뮬 구간에만 건다 — 그림·표만 다시 만드는 --from-csv 는 동시에 돌아도 안전하다.
        with _RunLock(os.path.join(args.out, ".eval.lock"), force=args.force_lock):
            df = H.run_matrix(args.ckpt, seeds, conditions=conds, formations=forms,
                              commanders=commanders, recorders=recorders,
                              replan_every=args.replan_every, nets_per_ship=args.nets,
                              checkpoint=ckpt_csv, resume=args.resume,
                              progress=(None if args.quiet else log))
        log(f"[eval] 완료 {time.perf_counter() - t0:.1f}s")
        log(f"[eval] 원자료 → {ckpt_csv} ({len(df)} 에피소드)")

        if recorders:
            # ★ 메모리(이번 실행분)가 아니라 **JSONL 전체**를 읽는다 — 이전에 중단된
            #   실행분까지 합쳐야 계측이 에피소드 CSV 와 같은 범위를 덮는다.
            from commander.llm_metrics import LLMRecorder as _LR
            frames = []
            for tag, r in recorders.items():
                got = _LR.load_jsonl(r.jsonl_path) if r.jsonl_path else r.to_frame()
                if len(got):
                    frames.append(got)
            if frames:
                llm_df = pd.concat(frames, ignore_index=True)
                p = os.path.join(args.out, "llm_metrics.csv")
                llm_df.to_csv(p, index=False, encoding="utf-8-sig")
                log(f"[eval] LLM 계측 → {p} ({len(llm_df)} 호출)")
                for b, r in recorders.items():
                    s = r.summary()
                    if s:
                        log(f"        [{b}] " + "  ".join(
                            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in s.items() if k not in ("backend", "model")))
                        if s.get("fallback_rate", 0) > 0.2:
                            log(f"        ⚠ [{b}] 폴백률 {s['fallback_rate']:.0%} — "
                                f"LLM 이 자주 실패해 휴리스틱으로 떨어지고 있다. "
                                f"이 조건의 결과는 'LLM 성능'이 아니다.")

    # ── 커버리지 점검 ──
    #   중단·재개로 조건마다 덮은 포메이션이 다르면 그대로 평균 내는 순간 비교가 거짓이 된다.
    #   집계는 공통 교집합만 쓰지만(stats.pivot_by_seed), 무엇이 빠졌는지는 눈에 보여야 한다.
    cov = S.coverage_report(df)
    p = os.path.join(args.out, "coverage.csv")
    cov.to_csv(p, encoding="utf-8-sig")
    log(f"\n[eval] 조건 × 포메이션 커버리지 → {p}\n{cov.to_string()}")
    if cov.values.min() != cov.values.max():
        log("[eval] ⚠ 커버리지 불균형 — 집계는 모든 조건이 공통으로 덮은 조합만 사용한다.\n"
            "        평가를 마저 돌리면 표본이 늘어난다(같은 명령을 다시 실행하면 이어받는다).")

    # ── 요약 통계 ──
    summ = S.summarize(df, args.metric)
    p = os.path.join(args.out, "summary.csv")
    summ.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"[eval] 요약 → {p}")
    log("\n" + summ[summ["formation"] == "ALL"]
        [["condition", "mean", "lo", "hi", "d_mean", "d_lo", "d_hi", "dz", "sig"]]
        .to_string(index=False))

    # ── 2×2 분해 (백엔드별) ──
    bks = P.backends_in(df) or [""]
    blocks = []
    for b in bks:
        try:
            inter = S.interaction(df, args.metric, backend=b)
            head = f"[{b or 'llm'}]"
            blocks.append(head + "\n" + "\n".join(f"  {k:<26} {v}" for k, v in inter.items()))
        except ValueError as e:
            log(f"[eval] 상호작용 분해 건너뜀({b or 'llm'}): {e}")
    if blocks:
        txt = "\n\n".join(blocks)
        p = os.path.join(args.out, "interaction.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"metric = {args.metric}\n\n{txt}\n")
        log(f"\n[eval] 2×2 분해 → {p}\n{txt}")

    # ── 전 지표 요약 (관점별) ──
    #   포획률 하나로는 "그물을 퍼부어 올린 것 아닌가"에 답할 수 없다. 지표 전부를 같은
    #   방식(대응표본 CI)으로 내고, 개선 방향까지 함께 적어 CSV 만 봐도 읽히게 한다.
    all_metrics = [m for _, ms in P.METRIC_GROUPS for m in ms if m in df.columns]
    parts = []
    for m in all_metrics:
        s = S.summarize(df, m)
        s["better"] = "lower" if m in P.LOWER_BETTER else "higher"
        parts.append(s)
    if parts:
        allsum = pd.concat(parts, ignore_index=True)
        p = os.path.join(args.out, "summary_all_metrics.csv")
        allsum.to_csv(p, index=False, encoding="utf-8-sig")
        log(f"[eval] 전 지표 요약 → {p}")
        log("\n[전 지표 · 전체(ALL) · baseline 대비 대응차이]")
        v = allsum[(allsum["formation"] == "ALL") & (allsum["condition"] != "heur_heur")]
        log(v[["metric", "better", "condition", "mean", "d_mean", "d_lo", "d_hi", "dz", "sig"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ── LLM 계층 비교 (계획 품질) ──
    #   포획률은 시스템 전체의 지표라 "LLM 이 무엇을 했는가"를 가린다.
    #   계획 자체의 품질은 시뮬 결과와 독립이므로 따로 보고할 수 있다.
    if llm_df is not None and len(llm_df):
        by = "model" if "model" in llm_df.columns and llm_df["model"].nunique() > 1 else "backend"
        try:
            q = S.compare_llm(llm_df, by=by)
            p = os.path.join(args.out, "llm_quality.csv")
            q.to_csv(p, index=False, encoding="utf-8-sig")
            log(f"[eval] LLM 계획 품질 → {p}")
            log("\n[지휘관별 계획 품질 (평균 · 95% CI)]")
            log(q.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        except Exception as e:
            log(f"[eval] LLM 품질표 건너뜀: {e}")
        if llm_df[by].nunique() >= 2:
            try:
                pw = S.llm_pairwise(llm_df, by=by)
                p = os.path.join(args.out, "llm_pairwise.csv")
                pw.to_csv(p, index=False, encoding="utf-8-sig")
                log(f"[eval] 지휘관 쌍대비교 → {p}")
                log("\n[지휘관 쌍대비교 (평균차 · 95% CI · sig=CI가 0 배제)]")
                log(pw.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
            except Exception as e:
                log(f"[eval] 쌍대비교 건너뜀: {e}")

    # ── 논문 표 ──
    tex = P.paper_table(df, args.metric)
    p = os.path.join(args.out, "table_main.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    log(f"[eval] 논문 표 → {p}")

    tex2 = P.multimetric_table(df)
    p = os.path.join(args.out, "table_multimetric.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write(tex2 + "\n")
    log(f"[eval] 다지표 표 → {p}")

    # ── 그림 ──
    log(f"[eval] 그림 생성 → {args.figdir}/")
    made = P.save_all(df, args.figdir, llm_df=llm_df, metric=args.metric,
                      verbose=not args.quiet)
    log(f"[eval] 그림 {len(made)}개 (PDF+PNG) → {os.path.abspath(args.figdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
