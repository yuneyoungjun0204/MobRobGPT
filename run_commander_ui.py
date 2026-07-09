"""프롬프트 입력창 + 지휘관 이유(rationale) 패널이 있는 시뮬레이터 뷰어.

레이아웃:  [ 왼쪽: 해상 시뮬 씬 ]   [ 오른쪽: 지휘관 판단(이유) 패널 ]
                     [ 하단: 자연어 명령 입력창 ]

하단 입력창에 명령 → Enter → Ollama 지휘관(qwen2.5:14b)이 3척 배정 결정 →
시뮬 주입(그물 전개) + 오른쪽 패널에 명령/배정/이유(rationale) 전체 표시.
(Ollama 없으면 위협비례 휴리스틱 폴백.)

실행:
    python run_commander_ui.py
    python run_commander_ui.py qwen2.5:7b
    python run_commander_ui.py --enemy wave
조작키: [space] 재생/일시정지  [r] 랜덤 리셋  [q] 종료
       [1] 집중  [2] 파상  [3] 양동  (상단 버튼과 동일 — 그 대형으로 리셋·재시작)
"""
import sys
import random
import textwrap


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


def main() -> None:
    enemy = _arg("--enemy", "random")
    backend = "openai" if "--openai" in sys.argv else "ollama"
    model = next((a for a in sys.argv[1:] if not a.startswith("-") and a != enemy), None)

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

    sim = CommandedSimulator(enemy_mode=enemy)
    sim.reset(seed=0)
    sim.running = True
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

    def on_submit(cmd: str):
        cmd = (cmd or "").strip()
        if not cmd:
            return
        if info.get("busy"):        # 재진입 가드: 이전 호출 처리 중이면 무시(중복/스택 방지)
            print("[on_submit] 이전 명령 처리 중 — 무시")
            return
        info["busy"] = True
        print(f"\n[on_submit] 콜백 발화, 입력='{cmd}'")
        try:
            info["cmd"], info["status"] = cmd, "지휘관 호출 중…"
            draw_info(); fig.canvas.draw_idle()   # flush_events() 제거 → 키 콜백 재진입 방지

            bf = build_battlefield(sim, command=cmd)
            plan = commander.plan(bf)                      # LLM 동기 호출 (클러스터별 투입 척수)
            assign = plan_to_assign(plan, bf)              # 척수 → 아군별 담당 클러스터
            sim.set_command(assign)                        # 시뮬 주입(경로는 시뮬이 기하로 생성)

            committed = int((assign >= 0).sum())
            reserve = sim.cfg.n_allies - committed
            alloc = "  ".join(f"C{d.cluster_id}:{d.n_ships}척" for d in plan.deployments) or "(없음)"
            info["assign"] = (f"투입 {committed}척 / 예비 {reserve}척\n{alloc}\n"
                              + "  ".join(f"#{i}→C{a}" if a >= 0 else f"#{i}→예비"
                                         for i, a in enumerate(assign.tolist())))
            info["rationale"] = plan.rationale
            info["status"] = "배정 적용됨"
            print(f"[명령] {cmd}")
            print(f"  deployments={[(d.cluster_id, d.n_ships) for d in plan.deployments]}  assign={assign.tolist()}")
            print(f"  rationale: {plan.rationale}")
            draw_info()
        except Exception as e:
            import traceback
            traceback.print_exc()
            info["status"] = f"오류: {type(e).__name__}: {e}"
            draw_info()
        finally:
            info["busy"] = False

    text_box.on_submit(on_submit)

    def apply_formation(mode, label):
        print(f"\n[대형] {label}({mode}) 리셋+재시작 (콜백 발화)")
        sim.enemy_mode = mode
        sim.reset(seed=random.randint(0, 2_000_000_000))   # 매번 다른 시드 → 변형
        sim.set_command(None); sim.running = True
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
        elif k == "r":
            sim.reset(seed=random.randint(0, 2_000_000_000)); sim.set_command(None); sim.running = True
            info["busy"] = True                 # set_val 이 트리거하는 submit 차단(표시만 갱신)
            try:
                text_box.set_val(DEFAULT_CMD)   # 입력창 표시를 기본 명령으로 동기화
            except Exception:
                pass
            finally:
                info["busy"] = False
            on_submit(DEFAULT_CMD)              # 실제 적용 1회
        elif k == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    btn_conc.on_clicked(lambda e: apply_formation("concentrated", "집중"))
    btn_wave.on_clicked(lambda e: apply_formation("wave", "파상"))
    btn_div.on_clicked(lambda e: apply_formation("diversionary", "양동"))

    def update(_):
        if sim.running and not sim.done:
            sim.step()
        renderer.draw_scene(ax, sim.get_frame(), bg_img=bg_img, bg_extent=bg_extent)
        return []

    draw_info()
    anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    print(f"뷰어 실행: model={model}, enemy={enemy}. 기본 명령 '{DEFAULT_CMD}' 자동 적용.")
    on_submit(DEFAULT_CMD)     # 시작 시 기본 명령 자동 실행 (모델 이미 로드됨)
    plt.show()
    _ = (anim, text_box, btn_conc, btn_wave, btn_div)   # 참조 유지(GC 방지)


if __name__ == "__main__":
    main()
