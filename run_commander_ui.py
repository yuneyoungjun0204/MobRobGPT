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
    ros2 = "--ros2" in sys.argv              # ROS2 센서 사용 (GPS/IMU → SIM 좌표)
    if cell:
        rl = True
    gain = float(_arg("--gain", "1"))        # RL 잔차 배율(시각화용; 셀 모델은 무의미)
    _ckpt_default = "boatattack_sim/models/best_mixed_far.pt" if cell \
        else "boatattack_sim/models/rl_latest.pt"
    ckpt = _arg("--ckpt", _ckpt_default)
    # --specialized [경로]: 공격양상 기하분류 → 집중/양동/파상 특화 셀 정책 라우팅(--cell 전용)
    specialized_root = (_arg("--specialized", "30_model")
                        if ("--specialized" in sys.argv and cell) else None)
    # --world <m>: 실험장 한 변 크기(m). 주면 모든 길이를 비례축소 → 33m 수조 등에서 그대로 사용.
    #   관측이 길이/길이 비율이라 정규화 입력이 동일 → 재학습 없이 동작(tests/test_scale_e2e.py 검증).
    world_size = float(_arg("--world", "0")) or None
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
        if specialized_root:
            print(f"Loading specialized cell routing... (Concentrated/Diversionary/Wave, {specialized_root})")
        else:
            print(f"Loading cell policy... ({ckpt})")
        if world_size is not None:
            print(f"  * Scale Transform: world_size={world_size}m (proportional scaling, no retraining needed)")
        sim = CommandedCellEnv(ckpt, enemy_mode=enemy, avoid_steer=apf,   # 기본 APF OFF, --apf 로 ON
                               specialized_root=specialized_root,
                               world_size=world_size)
        _build_bf = build_battlefield_defense
    elif rl:   # RL 경로 기동 (배정=LLM, 경로=강화학습 잔차 정책 / DefenseVecEnv 백엔드)
        from commander.rl_bridge import CommandedDefenseEnv, build_battlefield_defense
        print(f"Loading RL policy... ({ckpt}, gain={gain})")
        sim = CommandedDefenseEnv(ckpt, enemy_mode=enemy, gain=gain, avoid_steer=apf)
        _build_bf = build_battlefield_defense
    else:    # 휴리스틱 경로 기동 (기존)
        sim = CommandedSimulator(enemy_mode=enemy)
        _build_bf = build_battlefield
    sim.reset(seed=0)
    sim.running = True

    # ROS2 센서 브릿지 (--ros2 + --cell 전용)
    ros2_bridge = None
    if ros2 and cell:
        from commander.ros2_bridge import ROS2SensorBridge, ROS2_AVAILABLE
        if ROS2_AVAILABLE:
            _ws = world_size if world_size else sim.cfg.world_size
            ros2_bridge = ROS2SensorBridge(world_size=_ws, n_allies=sim.P, n_enemies=sim.M)
            ros2_bridge.start()
            print(f"  * ROS2 Sensor Bridge: world={_ws}m (GPS/IMU -> SIM coords)")
        else:
            print("  ! ROS2 unavailable (rclpy not found) - Simulation mode")

    def _scalar_t():   # sim.t: CommandedSimulator=스칼라, RL(DefenseVecEnv)=배열([N]) → 스칼라화
        t = sim.t
        return int(t.flat[0]) if hasattr(t, "flat") else int(t)

    def _is_done():
        d = sim.done
        return bool(d.flat[0]) if hasattr(d, "flat") else bool(d)

    commander = make_commander(backend, model)
    model = commander.model   # 실제 사용 모델명(라벨용)
    print(f"Loading model... ({model}) — window opens after load.")
    commander.warmup()        # 창 뜨기 전에 모델 메모리 로드 → 첫 명령 지연 제거

    # 위성사진 배경 (Esri World Imagery). 오프라인/실패 시 해색 배경으로 폴백.
    from commander.satellite import fetch_satellite_bg
    bg_img = bg_extent = None
    try:
        print("Loading satellite background...")
        res = fetch_satellite_bg(sim.cfg.geo_lat, sim.cfg.geo_lon, sim.cfg.world_size)
        if res:
            bg_img, bg_extent = res
            print("Satellite background loaded")
        else:
            print("Satellite background failed (offline?) → sea color fallback")
    except Exception as e:
        print(f"Satellite background error ({e}) → sea color fallback")

    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_axes((0.03, 0.11, 0.58, 0.80))          # 씬(왼쪽)
    ax_info = fig.add_axes((0.635, 0.11, 0.35, 0.80))    # 이유 패널(오른쪽)
    ax_box = fig.add_axes((0.12, 0.035, 0.76, 0.04))     # 명령 입력창(하단)
    # 적 대형 버튼 (상단): 누르면 그 대형으로 리셋·재시작
    ax_b1 = fig.add_axes((0.03, 0.935, 0.16, 0.045))
    ax_b2 = fig.add_axes((0.205, 0.935, 0.16, 0.045))
    ax_b3 = fig.add_axes((0.38, 0.935, 0.16, 0.045))
    DEFAULT_CMD = "Capture all enemies"
    text_box = TextBox(ax_box, "Command ", initial=DEFAULT_CMD)
    btn_conc = Button(ax_b1, "Concentrated")
    btn_wave = Button(ax_b2, "Wave")
    btn_div = Button(ax_b3, "Diversionary")

    info = {
        "cmd": "(none)",
        "assign": "All reserve (stopped) — waiting for command",
        "rationale": "Allies do not move before receiving a command (pure LLM control).\n"
                     "Enter a command in the input box below and press Enter.\n"
                     "e.g.) Block the frontal dense swarm first\n"
                     "e.g.) Send 2 ships to the large cluster, 1 in reserve",
        "status": "Standby (allies stopped)",
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
            f"■ Commander: {model}",
            f"■ Status: {info['status']}",
            "",
            "■ Command (prompt)",
            textwrap.fill(info["cmd"], width=32),
            "",
            "■ Deployment (allies→clusters)",
            info["assign"],
            "",
            "■ Rationale",
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
        info["cmd"], info["status"] = cmd, "Calling commander... (simulation continues)"
        draw_info(); fig.canvas.draw_idle()
        bf = _build_bf(sim, command=cmd)          # 전장 스냅샷(메인 스레드) → 스레드로 전달
        gen = info.get("gen", 0)
        print(f"\n[on_submit] LLM call started (non-blocking), input='{cmd}'")
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
            alloc = "  ".join(f"C{d.cluster_id}:{d.ally_ids or 'auto'}" for d in plan.deployments) or "(none)"

            def _tag(i, a):
                if i in held:
                    return f"#{i}→HOLD"
                return f"#{i}→C{a}" if a >= 0 else f"#{i}→reserve"
            info["assign"] = (f"Deployed {committed} / Reserve {reserve}"
                              + (f" / HOLD {len(held)}" if held else "") + f"\n{alloc}\n"
                              + "  ".join(_tag(i, a) for i, a in enumerate(assign.tolist())))
            info["rationale"] = plan.rationale
            info["status"] = "Assignment applied"
            print(f"[Command Applied] deployments={[(d.cluster_id, d.ally_ids) for d in plan.deployments]}  "
                  f"hold={sorted(held)}  assign={assign.tolist()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            info["status"] = f"Error: {type(e).__name__}: {e}"
        finally:
            info["busy"] = False
            draw_info()

    text_box.on_submit(on_submit)

    def apply_formation(mode, label):
        print(f"\n[Formation] {label}({mode}) reset+restart (callback triggered)")
        sim.enemy_mode = mode
        sim.reset(seed=random.randint(0, 2_000_000_000))   # 매번 다른 시드 → 변형
        sim.set_command(None); sim.running = True
        _reset_llm()                                       # 진행 중 LLM 무효화
        info["status"] = f"[{label}] Formation reset — calling commander..."
        draw_info(); fig.canvas.draw_idle()
        on_submit(DEFAULT_CMD)

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == " ":
            sim.running = not sim.running
        elif k == "1":
            apply_formation("concentrated", "Concentrated")
        elif k == "2":
            apply_formation("wave", "Wave")
        elif k == "3":
            apply_formation("diversionary", "Diversionary")
        elif k == "v":                              # APF(충돌회피 안전층) 토글 — RL 모드에서 유효
            sim.cfg.avoid_steer = not getattr(sim.cfg, "avoid_steer", False)
            info["status"] = f"APF (collision avoidance) {'ON' if sim.cfg.avoid_steer else 'OFF'}"
            draw_info()
        elif k == "c" and hasattr(sim, "resolve_conflicts"):   # 경로 겹침 해소(중복 HOLD) 토글 — RL
            sim.resolve_conflicts = not sim.resolve_conflicts
            info["status"] = f"Path conflict resolution {'ON' if sim.resolve_conflicts else 'OFF'}"
            draw_info()
        elif k == "z" and cell:                     # 후보셀 오버레이 토글 (셀 모드)
            info["show_cells"] = not info.get("show_cells", True)
            info["status"] = f"Cell overlay {'ON' if info['show_cells'] else 'OFF'}"
            draw_info()
        elif k == "a":
            info["auto"] = not info.get("auto")
            info["status"] = (f"Auto replan {'ON' if info['auto'] else 'OFF'} "
                              f"(period {info['replan_period']} step)")
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

    btn_conc.on_clicked(lambda e: apply_formation("concentrated", "Concentrated"))
    btn_wave.on_clicked(lambda e: apply_formation("wave", "Wave"))
    btn_div.on_clicked(lambda e: apply_formation("diversionary", "Diversionary"))

    def update(_):
        if sim.running and not _is_done():
            # ROS2 모드: 센서 주입 + 정책 추론만 (시뮬레이션 물리 없음)
            if ros2_bridge is not None:
                ros2_bridge.step_policy_only(sim)
            else:
                sim.step()                              # 시뮬레이션 모드: 물리 포함
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
        if specialized_root and getattr(sim, "_formation", None):   # 특화 라우팅: 현재 대형 표시
            _KMODE = {"concentrated": "Concentrated", "diversionary": "Diversionary", "wave": "Wave"}
            ax.text(0.99, 0.99, f"Attack pattern: {_KMODE.get(sim._formation, sim._formation)}  → Specialized model",
                    transform=ax.transAxes, color="#FFD54F", fontsize=11, va="top", ha="right",
                    weight="bold")
        return []

    draw_info()
    anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    _mode = "ros2" if ros2_bridge else enemy
    print(f"Viewer started: model={model}, mode={_mode}. Default command '{DEFAULT_CMD}' auto-applied.")
    on_submit(DEFAULT_CMD)     # 시작 시 기본 명령 자동 실행 (모델 이미 로드됨)
    try:
        plt.show()
    finally:
        if ros2_bridge:
            ros2_bridge.shutdown()
    _ = (anim, text_box, btn_conc, btn_wave, btn_div)   # 참조 유지(GC 방지)


if __name__ == "__main__":
    main()
