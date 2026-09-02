"""지휘관 UI 틀 그대로 GIF/영상으로 굽는 헤드리스 익스포터.

왜 별도 스크립트인가
    render_sim.py --gif 는 **씬만** 그린다 — 지휘관 패널도, 확대 캠도 없다.
    run_commander_ui.py 는 그 UI 를 다 갖췄지만 TextBox/Button 위젯과 plt.show()
    에 묶여 있어 헤드리스에서 못 돈다. 그래서 같은 레이아웃을 Agg 백엔드로
    재구성하고, 입력창 대신 **명령 로그**를 둔다.

레이아웃 (run_commander_ui.py 와 같은 좌/우 배치)
    +--------------------------------------+------------------+
    | [브리핑] [적 캠] [아군 캠]  <- 오버레이 |  COMMANDER       |
    |                                      |  * 현재 명령      |
    |            해상 씬                    |  * 배정          |
    |                                      |  * 판단 근거      |
    |                                      |  ---------------  |
    |                                      |  ORDER LOG        |
    |                                      |  t=0   ...        |
    +--------------------------------------+------------------+

    상단 3패널은 renderer.setup_panels / draw_panels 를 그대로 쓴다. 이 함수들은
    예전에 만들어 두고 어디서도 호출하지 않던 것이다 — 새로 그리지 않는다.

실행:
    python tools/make_commander_gif.py --out demo.gif
    python tools/make_commander_gif.py --out demo.gif --enemy diversionary --frames 600
    python tools/make_commander_gif.py --out demo.gif --backend gemini --model gemini-3.5-flash-lite
    python tools/make_commander_gif.py --out demo.mp4 --fps 30        # ffmpeg 있으면 mp4
    python tools/make_commander_gif.py --out demo.gif --no-llm        # 휴리스틱만(오프라인)

지휘관 호출은 **동기**다. run_commander_ui 는 화면이 멈추면 안 되니 스레드로
비동기 호출하지만, 여기서는 프레임을 파일로 굽는 것이라 기다려도 된다 —
오히려 동기라야 "이 프레임에서 이 명령이 나왔다"가 정확히 맞는다.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── 색 (renderer.C 와 같은 계열. 패널 밖 여백까지 다크로 칠하기 위해 여기서도 쓴다) ──
BG_FIG = "#0A121F"
BG_PANEL = "#0d1b2a"
FG = "#ECEFF4"
FG_DIM = "#8A93A0"
ACCENT = "#26C6DA"
OK = "#00E676"
WARN = "#FFB74D"
BAD = "#EF5350"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="지휘관 UI(씬 + 확대 캠 + LLM 명령 패널)를 GIF/MP4 로 굽는다.")
    p.add_argument("--out", default="commander_demo.gif",
                   help="출력 경로. 확장자 .mp4 면 ffmpeg 시도 후 실패 시 GIF 폴백")
    p.add_argument("--frames", type=int, default=400, help="프레임 수 (기본 400)")
    p.add_argument("--fps", type=int, default=25, help="초당 프레임 (기본 25)")
    p.add_argument("--enemy", default="diversionary",
                   help="적 스폰 대형: concentrated|diversionary|wave|random")
    p.add_argument("--seed", type=int, default=0, help="난수 시드")
    p.add_argument("--replan", type=int, default=100,
                   help="지휘관 재계획 주기(step). 0=최초 1회만")
    p.add_argument("--backend", default="ollama", help="ollama|openai|gemini")
    p.add_argument("--model", default=None, help="모델명 (미지정 시 백엔드 기본값)")
    p.add_argument("--command", default="Capture all enemies",
                   help="지휘관에게 줄 자연어 명령")
    p.add_argument("--no-llm", action="store_true",
                   help="LLM 을 부르지 않고 휴리스틱만 쓴다(오프라인/CI)")
    p.add_argument("--dpi", type=int, default=100, help="저장 해상도")
    p.add_argument("--satellite", action="store_true",
                   help="위성 배경을 받아온다(네트워크 필요, 기본 꺼짐)")
    return p.parse_args(argv)


# ── 지휘관 명령 로그 ────────────────────────────────────────────────────────
class OrderLog:
    """지휘관이 낸 계획을 시간순으로 쌓는다.

    폴백(LLM 실패 → 휴리스틱)을 **반드시 구분해서 보관한다**. 구분하지 않으면
    화면에는 명령이 흐르는데 실제로는 LLM 이 한 번도 안 불린 상태를 알 수 없다
    (이 프로젝트에서 gpt-4o-mini 가 폴백률 1.000 으로 전량 폐기된 전례가 있다).
    """

    MAX_SHOWN = 6

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, t: int, plan, assign_txt: str) -> dict:
        rat = str(getattr(plan, "rationale", "") or "")
        row = {
            "t": int(t),
            "fallback": "휴리스틱 방어" in rat,
            "alloc": "  ".join(f"C{d.cluster_id}:{list(d.ally_ids or [])}"
                               for d in (plan.deployments or [])) or "(none)",
            "assign": assign_txt,
            "rationale": rat,
        }
        self.rows.append(row)
        return row

    def recent(self):
        return self.rows[-self.MAX_SHOWN:]

    @property
    def n_fallback(self) -> int:
        return sum(1 for r in self.rows if r["fallback"])


def draw_commander_panel(ax, *, model: str, command: str, log: OrderLog,
                         status: str, flash: bool):
    """오른쪽 지휘관 패널: 현재 명령 / 배정 / 근거 / 명령 로그.

    `flash=True` 면 이번 프레임에 새 계획이 내려온 것 — 테두리를 밝혀 그 순간을 보이게 한다.
    """
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG_PANEL)
    edge = OK if flash else ACCENT
    for s in ax.spines.values():
        s.set_color(edge); s.set_linewidth(2.4 if flash else 1.2)

    last = log.rows[-1] if log.rows else None
    y = 0.975

    def line(txt, *, color=FG, size=8.5, bold=False, dy=0.030, x=0.035):
        nonlocal y
        ax.text(x, y, txt, color=color, fontsize=size, va="top", ha="left",
                fontweight="bold" if bold else "normal", transform=ax.transAxes)
        y -= dy

    line("LLM COMMANDER", color=ACCENT, size=10.5, bold=True, dy=0.040)
    line(f"model   {model}", color=FG_DIM, size=8)
    line(f"status  {status}", color=OK if "applied" in status.lower() else WARN, size=8,
         dy=0.042)

    line("COMMAND", color=ACCENT, size=8.5, bold=True, dy=0.028)
    for ln in textwrap.wrap(command, width=40)[:2]:
        line(ln, size=8)
    y -= 0.014

    line("DEPLOYMENT", color=ACCENT, size=8.5, bold=True, dy=0.028)
    if last:
        for ln in textwrap.wrap(last["alloc"], width=40)[:2]:
            line(ln, size=8, color=OK)
        for ln in textwrap.wrap(last["assign"], width=40)[:2]:
            line(ln, size=8)
    else:
        line("(awaiting first plan)", color=FG_DIM, size=8)
    y -= 0.014

    line("RATIONALE", color=ACCENT, size=8.5, bold=True, dy=0.028)
    if last:
        rat = last["rationale"]
        # 폴백은 rationale 앞에 이유가 붙는다 — 색으로 갈라 LLM 출력과 혼동을 막는다.
        col = WARN if last["fallback"] else FG
        for ln in textwrap.wrap(rat, width=40)[:6]:
            line(ln, size=7.8, color=col, dy=0.026)
    else:
        line("(none)", color=FG_DIM, size=8)

    # ── 명령 로그 (아래쪽 고정) ──
    y = 0.325
    line("ORDER LOG", color=ACCENT, size=8.5, bold=True, dy=0.032)
    if not log.rows:
        line("(empty)", color=FG_DIM, size=8)
    for r in log.recent():
        mark, col = ("FB", WARN) if r["fallback"] else ("OK", OK)
        ax.text(0.035, y, f"[{mark}]", color=col, fontsize=7.5, va="top",
                ha="left", fontweight="bold", transform=ax.transAxes)
        ax.text(0.135, y, f"t={r['t']:<5d} {r['alloc'][:30]}", color=FG,
                fontsize=7.5, va="top", ha="left", transform=ax.transAxes)
        y -= 0.030

    if log.n_fallback:
        ax.text(0.035, 0.022,
                f"! {log.n_fallback}/{len(log.rows)} plans fell back to heuristic",
                color=WARN, fontsize=7.5, va="bottom", ha="left",
                transform=ax.transAxes)


def main(argv=None) -> int:
    args = parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")                    # ★ 창 없이 — DISPLAY 없는 환경에서도 돈다
    import warnings
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm

    # renderer 는 import 시점에 font.family 를 한글폰트 '단일'로 덮어쓴다.
    # 폴백 리스트가 이기려면 반드시 renderer import '뒤'에 폰트를 다시 잡아야 한다.
    from boatattack_sim.eval import renderer
    from commander.fallback import heuristic_plan
    from commander.sim_bridge import CommandedSimulator, build_battlefield, plan_to_assign

    _names = {f.name for f in _fm.fontManager.ttflist}
    _kf = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic")
                if f in _names), None)
    if _kf:
        matplotlib.rcParams["font.family"] = [_kf, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    sim = CommandedSimulator(enemy_mode=args.enemy)
    sim.reset(seed=args.seed)
    sim.running = True

    # ── 지휘관 ──
    commander = None
    model_label = "heuristic (no LLM)"
    if not args.no_llm:
        try:
            from commander import make_commander
            commander = make_commander(args.backend, args.model)
            model_label = getattr(commander, "model", args.backend)
            commander.warmup()
            print(f"[gif] 지휘관 준비: {model_label}")
        except Exception as e:
            print(f"[gif] 지휘관 초기화 실패({type(e).__name__}: {e}) → 휴리스틱으로 진행")
            commander = None
            model_label = "heuristic (LLM unavailable)"

    # ── 위성 배경 (옵션) ──
    bg_img = bg_extent = None
    if args.satellite:
        try:
            from commander.satellite import fetch_satellite_bg
            res = fetch_satellite_bg(sim.cfg.geo_lat, sim.cfg.geo_lon, sim.cfg.world_size)
            if res:
                bg_img, bg_extent = res
                print("[gif] 위성 배경 로드 완료")
        except Exception as e:
            print(f"[gif] 위성 배경 실패({e}) → 해색 배경")

    # ── 레이아웃: run_commander_ui.py 와 같은 좌(씬)/우(패널) 배치 ──
    fig = plt.figure(figsize=(13.5, 9.0), facecolor=BG_FIG)
    ax = fig.add_axes((0.030, 0.035, 0.600, 0.885))       # 씬
    ax_info = fig.add_axes((0.648, 0.035, 0.335, 0.885))  # 지휘관 패널
    # 상단 오버레이 3패널(브리핑 / 적 캠 / 아군 캠) — 씬 axes 안쪽 상단에 얹는다.
    #   renderer.setup_panels 는 run_commander_ui 좌표에 맞춰져 있어 여기서 직접 배치한다.
    #   ★ 오른쪽 끝(0.630)이 ax_info 왼쪽(0.648)보다 작다 → 지휘관 패널 글자를 가리지 않는다.
    panels = {
        "brief":     fig.add_axes((0.042, 0.690, 0.150, 0.215)),
        "cam_enemy": fig.add_axes((0.330, 0.690, 0.145, 0.215)),
        "cam_ally":  fig.add_axes((0.485, 0.690, 0.145, 0.215)),
    }
    for a in panels.values():
        a.set_zorder(20)
        a.set_facecolor(BG_PANEL)

    title = fig.text(0.030, 0.958, "", color=FG, fontsize=12, ha="left",
                     va="center", fontweight="bold")
    subtitle = fig.text(0.630, 0.958, "", color=FG_DIM, fontsize=9, ha="right",
                        va="center")

    log = OrderLog()
    state = {"cmd": args.command, "status": "standby", "flash": 0,
             "assign_txt": "(none)", "last_replan": -10 ** 9}

    def issue_order(t: int) -> None:
        """지휘관 호출 → 계획 주입 → 로그 적재. 동기 호출(파일로 굽는 중이라 기다려도 된다)."""
        bf = build_battlefield(sim, command=state["cmd"])
        if commander is not None:
            plan = commander.plan(bf)
        else:
            # ★ sim.heuristic_plan() 이 아니다 — 그쪽은 '기동' 재계획이라 None 을 돌려준다.
            #   지휘관 수준(클러스터 배정) 휴리스틱은 commander.fallback 쪽이고,
            #   각 어댑터가 LLM 실패 시 쓰는 것과 같은 함수다.
            plan = heuristic_plan(bf)
            plan.rationale = f"LLM 미사용(--no-llm) → 휴리스틱 방어. {plan.rationale}"
        assign = plan_to_assign(plan, bf, mode="llm")
        sim.set_plan(plan, state["cmd"])
        committed = int((assign >= 0).sum())
        state["assign_txt"] = (f"deployed {committed}/{sim.cfg.n_allies}  "
                               + " ".join(f"#{i}->"
                                          + (f"C{a}" if a >= 0 else "res")
                                          for i, a in enumerate(assign.tolist())))
        row = log.add(t, plan, state["assign_txt"])
        state["status"] = "fallback applied" if row["fallback"] else "assignment applied"
        state["flash"] = 8            # 8 프레임 동안 패널 테두리 강조
        state["last_replan"] = t

    def scalar_t() -> int:
        t = sim.t
        return int(t.flat[0]) if hasattr(t, "flat") else int(t)

    def is_done() -> bool:
        d = sim.done
        return bool(d.flat[0]) if hasattr(d, "flat") else bool(d)

    issue_order(scalar_t())           # 첫 명령

    def update(_frame):
        t = scalar_t()
        if sim.running and not is_done():
            sim.step()
            t = scalar_t()
            if args.replan > 0 and t - state["last_replan"] >= args.replan:
                issue_order(t)

        fd = sim.get_frame()
        renderer.draw_scene(ax, fd, bg_img=bg_img, bg_extent=bg_extent)
        renderer.draw_panels(panels, fd, bg_img, bg_extent)

        st = fd.get("stats", {})
        title.set_text(f"USV SWARM DEFENSE   |   t={t}   "
                       f"captured {int(st.get('captures', 0))}   "
                       f"breached {int(st.get('breaches', 0))}   "
                       f"enemies alive {int(fd.get('n_alive', 0))}")
        subtitle.set_text(f"formation: {args.enemy}   seed: {args.seed}   "
                          f"replan every {args.replan} steps")

        flash = state["flash"] > 0
        if flash:
            state["flash"] -= 1
        draw_commander_panel(ax_info, model=model_label, command=state["cmd"],
                             log=log, status=state["status"], flash=flash)
        return []

    # ── 저장 ──
    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, update, frames=args.frames, interval=1000 // max(args.fps, 1),
                         blit=False, cache_frame_data=False)

    out = args.out
    t0 = time.perf_counter()
    if out.lower().endswith(".mp4"):
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(out, writer=FFMpegWriter(fps=args.fps), dpi=args.dpi,
                      savefig_kwargs={"facecolor": BG_FIG})
            print(f"[gif] MP4 저장: {out}")
        except Exception as e:
            out = os.path.splitext(out)[0] + ".gif"
            print(f"[gif] ffmpeg 실패({type(e).__name__}: {e}) → GIF 로 폴백: {out}")
            anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi,
                      savefig_kwargs={"facecolor": BG_FIG})
    else:
        anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi,
                  savefig_kwargs={"facecolor": BG_FIG})

    plt.close(fig)
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"[gif] 완료: {out}  ({size / 1e6:.2f} MB, {args.frames} 프레임, "
          f"{time.perf_counter() - t0:.1f}s)")
    print(f"[gif] 지휘관 호출 {len(log.rows)}회 (폴백 {log.n_fallback}회) · "
          f"최종 stats={sim.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
