"""프롬프트 입력창 + 지휘관 이유(rationale) 패널이 있는 시뮬레이터 뷰어.

레이아웃:  [ 왼쪽: 해상 시뮬 씬 ]   [ 오른쪽: 지휘관 판단(이유) 패널 ]
                     [ 하단: 자연어 명령 입력창 ]

하단 입력창에 명령 → Enter → Ollama 지휘관(기본 qwen2.5:7b — 14b 대비 품질 동등·6배 빠름)이 3척 배정 결정 →
시뮬 주입(그물 전개) + 오른쪽 패널에 명령/배정/이유(rationale) 전체 표시.
(Ollama 없으면 위협비례 휴리스틱 폴백.)

배정은 **매 스텝** 현재 전장으로 재매핑(sticky) → 배가 격침되면 남은 배로 자동 재배분.
LLM 전체 재계획은 주기(기본 100 step)마다. 아군끼리 충돌하면 양쪽 다 격침(비활성화).

실행:
    python run_commander_ui.py
    python run_commander_ui.py qwen2.5:7b
    python run_commander_ui.py --enemy wave
조작키: [space] 재생/일시정지  [r] 랜덤 리셋  [q] 종료  [a] 자동 재계획 토글
       [1] 집중  [2] 파상  [3] 양동  (상단 버튼과 동일 — 그 대형으로 리셋·재시작)
옵션: --replan N  (N step 마다 LLM 자동 재계획, 전장 변화 적응. 0=끄기, 기본 100)
     --rl          (경로 기동을 강화학습 잔차 정책으로 — 배정은 여전히 LLM. DefenseVecEnv 백엔드)
     --cell        (경로/그물을 '셀선택' 정책(CellPointerActor)으로 — 배정은 여전히 LLM. --rl 함의.
                    LLM 배정→_assign 주입→셀 정책이 obs+후보셀 pruning 으로 존중. 기본 ckpt=cell_latest.pt)
     --gain K      (--rl 시 RL 잔차 배율, 기본 1. 크게 하면 휴리스틱 이탈 과장. 셀 모델은 무의미)
     --ckpt PATH   (RL 정책 체크포인트. 기본: --cell=cell_latest.pt, 그 외=rl_latest.pt)
"""
import sys
import random
import textwrap
import threading


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


_CELL_SHIP_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#A78BFA"]


def _overlay_cells_cmd(ax, viz):
    """셀 후보 오버레이(잘 보이게): 하늘색=전체후보, 배색=배별유효, 빨강X=그물배제, 흰테두리=선택."""
    cw = viz["world"]
    # 전체 후보 — 크고 또렷한 하늘색 점(어두운 배경 대비)
    ax.scatter(cw[:, 0], cw[:, 1], s=26, c="#8fe0ff", alpha=0.55,
               edgecolors="#0a2a3a", linewidths=0.5, zorder=2.2)
    for p, (val, exc) in enumerate(zip(viz["valid"], viz["excluded"])):
        col = _CELL_SHIP_COLORS[p % len(_CELL_SHIP_COLORS)]
        if len(val):                                          # 배별 유효 후보 — 배색 큰 점
            ax.scatter(cw[val, 0], cw[val, 1], s=70, c=col, alpha=0.55,
                       edgecolors="white", linewidths=0.6, zorder=2.6)
        if len(exc):                                          # 그물 배제 — 굵은 빨강 X
            ax.scatter(cw[exc, 0], cw[exc, 1], s=130, marker="X", c="#FF1744",
                       edgecolors="white", linewidths=1.2, zorder=3.2)
    sel = viz.get("selected")
    if sel is not None:                                       # 선택 셀 — 큰 흰테두리 점 + 링
        for p in range(len(sel)):
            col = _CELL_SHIP_COLORS[p % len(_CELL_SHIP_COLORS)]
            ax.scatter(cw[sel[p], 0], cw[sel[p], 1], s=170, facecolors=col,
                       edgecolors="white", linewidths=2.0, zorder=6)
            ax.scatter(cw[sel[p], 0], cw[sel[p], 1], s=320, facecolors="none",
                       edgecolors=col, linewidths=1.4, alpha=0.8, zorder=5.8)


def main() -> None:
    enemy = _arg("--enemy", "random")
    backend = "openai" if "--openai" in sys.argv else "ollama"
    replan = int(_arg("--replan", "100"))   # LLM 자동 재계획 주기(step). 0=끄기
    rl = "--rl" in sys.argv                  # 경로 기동을 RL 정책으로 (배정은 여전히 LLM)
    cell = "--cell" in sys.argv              # RL 을 '셀선택' 정책(CellPointerActor)으로 (--rl 함의)
    if cell:
        rl = True
    gain = float(_arg("--gain", "1"))        # RL 잔차 배율(시각화용; 셀 모델은 무의미)
    _ckpt_default = "boatattack_sim/models/best_mixed_far.pt" if cell \
        else "boatattack_sim/models/rl_latest.pt"
    ckpt = _arg("--ckpt", _ckpt_default)
    apf = "--apf" in sys.argv                # RL 모드 APF(충돌회피 안전층). 기본 OFF, v 키로 토글
    _flagvals = {sys.argv[i + 1] for i, a in enumerate(sys.argv)
                 if a.startswith("--") and i + 1 < len(sys.argv)}
    model = next((a for a in sys.argv[1:] if not a.startswith("-") and a not in _flagvals), None)

    import warnings
    import matplotlib
    from matplotlib import font_manager as _fm
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, Button
    from matplotlib.animation import FuncAnimation

    # ⚠ renderer 는 import 시점에 rcParams['font.family'] 를 한글폰트 '단일'로 덮어쓴다.
    #    그래서 폰트 설정은 반드시 renderer import '뒤'에 해야 우리 폴백 리스트가 이긴다.
    from boatattack_sim.eval import renderer
    from commander.sim_bridge import CommandedSimulator, build_battlefield, plan_to_assign
    from commander import make_commander

    _names = {f.name for f in _fm.fontManager.ttflist}
    _kfont = next((_f for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic")
                   if _f in _names), None)
    if _kfont:  # 한글 + DejaVu 폴백(렌더러 ✖ 등) → 글리프 누락 경고 방지
        matplotlib.rcParams["font.family"] = [_kfont, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    if cell:  # 셀선택 RL 정책 (배정=LLM, 경로/그물=셀 정책 / DefenseVecEnv 셀 백엔드)
        from commander.cell_bridge import CommandedCellEnv, build_battlefield_defense
        print(f"셀 정책 로딩 중… ({ckpt})")
        sim = CommandedCellEnv(ckpt, enemy_mode=enemy, avoid_steer=apf)   # 기본 APF OFF, --apf 로 ON
        _build_bf = build_battlefield_defense
    elif rl:   # RL 경로 기동 (배정=LLM, 경로=강화학습 잔차 정책 / DefenseVecEnv 백엔드)
        from commander.rl_bridge import CommandedDefenseEnv, build_battlefield_defense
        print(f"RL 정책 로딩 중… ({ckpt}, gain={gain})")
        sim = CommandedDefenseEnv(ckpt, enemy_mode=enemy, gain=gain, avoid_steer=apf)
        _build_bf = build_battlefield_defense
    else:    # 휴리스틱 경로 기동 (기존)
        sim = CommandedSimulator(enemy_mode=enemy)
        _build_bf = build_battlefield
    sim.reset(seed=0)
    sim.running = True

    def _scalar_t():   # sim.t: CommandedSimulator=스칼라, RL(DefenseVecEnv)=배열([N]) → 스칼라화
        t = sim.t
        return int(t.flat[0]) if hasattr(t, "flat") else int(t)

    def _is_done():
        d = sim.done
        return bool(d.flat[0]) if hasattr(d, "flat") else bool(d)

    commander = make_commander(backend, model)
    model = commander.model   # 실제 사용 모델명(라벨용)
    print(f"모델 로딩 중… ({model}) — 로드 후 창이 뜹니다.")
    commander.warmup()        # 창 뜨기 전에 모델 메모리 로드 → 첫 명령 지연 제거

    # 위성사진 배경 (Esri World Imagery). 오프라인/실패 시 해색 배경으로 폴백.
    from commander.satellite import fetch_satellite_bg
    bg_img = bg_extent = None
    try:
        print("위성 배경 로딩 중…")
        res = fetch_satellite_bg(sim.cfg.geo_lat, sim.cfg.geo_lon, sim.cfg.world_size)
        if res:
            bg_img, bg_extent = res
            print("위성 배경 로드 완료")
        else:
            print("위성 배경 실패(오프라인?) → 해색 배경 폴백")
    except Exception as e:
        print(f"위성 배경 오류({e}) → 해색 배경 폴백")

    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_axes((0.03, 0.11, 0.58, 0.80))          # 씬(왼쪽)
    ax_info = fig.add_axes((0.635, 0.11, 0.35, 0.80))    # 이유 패널(오른쪽)
    ax_box = fig.add_axes((0.12, 0.035, 0.76, 0.04))     # 명령 입력창(하단)
    # 적 대형 버튼 (상단): 누르면 그 대형으로 리셋·재시작
    ax_b1 = fig.add_axes((0.03, 0.935, 0.16, 0.045))
    ax_b2 = fig.add_axes((0.205, 0.935, 0.16, 0.045))
    ax_b3 = fig.add_axes((0.38, 0.935, 0.16, 0.045))
    DEFAULT_CMD = "모든 적군 포획"
    text_box = TextBox(ax_box, "명령 ", initial=DEFAULT_CMD)
    btn_conc = Button(ax_b1, "집중")
    btn_wave = Button(ax_b2, "파상")
    btn_div = Button(ax_b3, "양동")

    info = {
        "cmd": "(없음)",
        "assign": "전원 예비(정지) — 명령 대기",
        "rationale": "명령 전에는 아군이 움직이지 않습니다(순수 LLM 제어).\n"
                     "하단 입력창에 명령을 입력하고 Enter 를 누르세요.\n"
                     "예) 정면 밀집 무리를 우선 차단\n"
                     "예) 큰 무리에 2척, 1척은 예비",
        "status": "대기 중 (아군 정지)",
        "last_cmd": DEFAULT_CMD,
        "auto": replan > 0,                       # LLM 자동 재계획 ON/OFF
        "replan_period": replan if replan > 0 else 50,
        "last_replan_t": 0,
    }

    def draw_info():
        ax_info.clear()
        ax_info.axis("off")
        ax_info.set_facecolor("#0d1b2a")
        lines = [
            f"■ 지휘관: {model}",
            f"■ 상태: {info['status']}",
            "",
            "■ 명령(프롬프트)",
            textwrap.fill(info["cmd"], width=32),
            "",
            "■ 투입 배분 (아군→클러스터)",
            info["assign"],
            "",
            "■ 판단 근거 (rationale)",
            textwrap.fill(info["rationale"], width=32),
        ]
        ax_info.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                     fontsize=10.5, family=matplotlib.rcParams["font.family"],
                     transform=ax_info.transAxes)

    def _llm_worker(bf, cmd, gen):
        """백그라운드 스레드: LLM 호출만 수행(시뮬 미접근). 리셋 시 gen 이 바뀌면 결과 폐기."""
        try:
            plan = commander.plan(bf)
            res = ("ok", plan, cmd)
        except Exception as e:
            res = ("err", e, cmd)
        if info.get("gen", 0) == gen:             # 리셋 안 됐을 때만 결과 반영
            info["_pending"] = res

    def _reset_llm():
        """리셋/대형변경 시 진행 중 LLM 호출 무효화(결과 폐기) + busy 해제."""
        info["gen"] = info.get("gen", 0) + 1
        info["busy"] = False
        info.pop("_pending", None)

    def on_submit(cmd: str):
        """LLM 호출을 '논블로킹'으로 시작만 한다 → 호출 동안 시뮬은 계속 진행(WP 추종·적 전진).
        결과는 update() 가 도착 시 _apply_result 로 적용."""
        cmd = (cmd or "").strip()
        if not cmd:
            return
        if info.get("busy"):        # 이전 호출 진행 중 — 새 호출 무시(한 번에 하나)
            return
        info["busy"] = True
        info["last_cmd"] = cmd                    # 자동 재계획이 이 명령을 반복 사용
        info["last_replan_t"] = _scalar_t()       # 재계획 타이머 리셋
        info["cmd"], info["status"] = cmd, "지휘관 호출 중… (시뮬 계속 진행)"
        draw_info(); fig.canvas.draw_idle()
        bf = _build_bf(sim, command=cmd)          # 전장 스냅샷(메인 스레드) → 스레드로 전달
        gen = info.get("gen", 0)
        print(f"\n[on_submit] LLM 호출 시작(논블로킹), 입력='{cmd}'")
        threading.Thread(target=_llm_worker, args=(bf, cmd, gen), daemon=True).start()

    def _apply_result():
        """LLM 결과가 도착했으면 메인 스레드에서 계획 적용(현재 전장 기준 재매핑)."""
        pend = info.pop("_pending", None)
        if pend is None:
            return
        try:
            status, payload, cmd = pend
            if status == "err":
                raise payload
            plan = payload
            bf = _build_bf(sim, command=cmd)               # 결과 도착 시점의 현재 전장으로 매핑
            assign = plan_to_assign(plan, bf)
            sim.set_plan(plan, cmd)                        # 매 스텝 재매핑(죽은 배·위치 적응)
            held = set(getattr(plan, "hold_ships", None) or [])
            committed = int((assign >= 0).sum())
            reserve = sim.cfg.n_allies - committed - len(held & {i for i in range(sim.cfg.n_allies) if assign[i] < 0})
            alloc = "  ".join(f"C{d.cluster_id}:{d.ally_ids or '자동'}" for d in plan.deployments) or "(없음)"

            def _tag(i, a):
                if i in held:
                    return f"#{i}→정지(HOLD)"
                return f"#{i}→C{a}" if a >= 0 else f"#{i}→예비"
            info["assign"] = (f"투입 {committed}척 / 예비 {reserve}척"
                              + (f" / 정지 {len(held)}척" if held else "") + f"\n{alloc}\n"
                              + "  ".join(_tag(i, a) for i, a in enumerate(assign.tolist())))
            info["rationale"] = plan.rationale
            info["status"] = "배정 적용됨"
            print(f"[명령 적용] deployments={[(d.cluster_id, d.ally_ids) for d in plan.deployments]}  "
                  f"hold={sorted(held)}  assign={assign.tolist()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            info["status"] = f"오류: {type(e).__name__}: {e}"
        finally:
            info["busy"] = False
            draw_info()

    text_box.on_submit(on_submit)

    def apply_formation(mode, label):
        print(f"\n[대형] {label}({mode}) 리셋+재시작 (콜백 발화)")
        sim.enemy_mode = mode
        sim.reset(seed=random.randint(0, 2_000_000_000))   # 매번 다른 시드 → 변형
        sim.set_command(None); sim.running = True
        _reset_llm()                                       # 진행 중 LLM 무효화
        info["status"] = f"[{label}] 대형 리셋 — 지휘관 호출 중…"
        draw_info(); fig.canvas.draw_idle()
        on_submit(DEFAULT_CMD)

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == " ":
            sim.running = not sim.running
        elif k == "1":
            apply_formation("concentrated", "집중")
        elif k == "2":
            apply_formation("wave", "파상")
        elif k == "3":
            apply_formation("diversionary", "양동")
        elif k == "v":                              # APF(충돌회피 안전층) 토글 — RL 모드에서 유효
            sim.cfg.avoid_steer = not getattr(sim.cfg, "avoid_steer", False)
            info["status"] = f"APF(충돌회피) {'ON' if sim.cfg.avoid_steer else 'OFF'}"
            draw_info()
        elif k == "c" and hasattr(sim, "resolve_conflicts"):   # 경로 겹침 해소(중복 HOLD) 토글 — RL
            sim.resolve_conflicts = not sim.resolve_conflicts
            info["status"] = f"경로중복 해소 {'ON' if sim.resolve_conflicts else 'OFF'}"
            draw_info()
        elif k == "z" and cell:                     # 후보셀 오버레이 토글 (셀 모드)
            info["show_cells"] = not info.get("show_cells", True)
            info["status"] = f"후보셀 표시 {'ON' if info['show_cells'] else 'OFF'}"
            draw_info()
        elif k == "a":
            info["auto"] = not info.get("auto")
            info["status"] = (f"자동 재계획 {'ON' if info['auto'] else 'OFF'} "
                              f"(주기 {info['replan_period']} step)")
            draw_info()
        elif k == "r":
            sim.reset(seed=random.randint(0, 2_000_000_000)); sim.set_command(None); sim.running = True
            info["busy"] = True                 # set_val 이 트리거하는 submit 차단(표시만 갱신)
            try:
                text_box.set_val(DEFAULT_CMD)   # 입력창 표시를 기본 명령으로 동기화
            except Exception:
                pass
            finally:
                info["busy"] = False
            _reset_llm()                        # 진행 중 LLM 무효화
            on_submit(DEFAULT_CMD)              # 실제 적용 1회
        elif k == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    btn_conc.on_clicked(lambda e: apply_formation("concentrated", "집중"))
    btn_wave.on_clicked(lambda e: apply_formation("wave", "파상"))
    btn_div.on_clicked(lambda e: apply_formation("diversionary", "양동"))

    def update(_):
        if sim.running and not _is_done():
            sim.step()                                  # LLM 추론 중에도 계속 진행(WP 추종·적 전진)
            if info.get("_pending") is not None:        # LLM 결과 도착 → 이번 프레임에 적용
                _apply_result()
            # 유동적 재계획: 주기마다 LLM 재호출(논블로킹). busy면 스킵(한 번에 하나).
            if (info.get("auto") and not info.get("busy")
                    and _scalar_t() - info.get("last_replan_t", 0) >= info["replan_period"]):
                print(f"[auto-replan] t={_scalar_t()}")
                on_submit(info.get("last_cmd", DEFAULT_CMD))
        renderer.draw_scene(ax, sim.get_frame(), bg_img=bg_img, bg_extent=bg_extent)
        if cell and info.get("show_cells", True) and hasattr(sim, "cell_viz"):
            _overlay_cells_cmd(ax, sim.cell_viz())          # 후보셀/그물배제/선택 오버레이 (z 토글)
        return []

    draw_info()
    anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    print(f"뷰어 실행: model={model}, enemy={enemy}. 기본 명령 '{DEFAULT_CMD}' 자동 적용.")
    on_submit(DEFAULT_CMD)     # 시작 시 기본 명령 자동 실행 (모델 이미 로드됨)
    plt.show()
    _ = (anim, text_box, btn_conc, btn_wave, btn_div)   # 참조 유지(GC 방지)


if __name__ == "__main__":
    main()
