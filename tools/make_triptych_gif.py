"""집중·양동·파상 세 대형을 **한 화면에 나란히** 굽는 GIF.

왜 따로 만드는가
    make_commander_gif.py 는 한 전장을 크게 보여준다. 이쪽은 "같은 정책이 세 공격
    양상에서 각각 어떻게 대응하는가"를 한 장면으로 비교시키는 용도다 — 발표에서
    세 영상을 번갈아 트는 것보다 훨씬 빨리 전달된다.

    시뮬 3개를 같은 루프에서 나란히 돌린다(각자 자기 시드·자기 대형).
    지휘관도 각자 따로 부른다 — 전장이 다르니 계획도 달라야 한다.

레이아웃
    +------------------+------------------+------------------+
    |   CONCENTRATED   |   DIVERSIONARY   |      WAVE        |
    |     (씬 + 줌)     |     (씬 + 줌)     |    (씬 + 줌)     |
    |  captured 3/10   |  captured 5/10   |  captured 2/10   |
    +------------------+------------------+------------------+
    |  LLM COMMANDER — 대형별 현재 배정 / 최근 명령            |
    +--------------------------------------------------------+

실행:
    python tools/make_triptych_gif.py                       # 휴리스틱 지휘관
    python tools/make_triptych_gif.py --backend gemini --model gemini-3.5-flash-lite
    python tools/make_triptych_gif.py --seeds 15,1,1 --frames 500
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.make_commander_gif import (  # noqa: E402  (경로 주입 뒤에 와야 한다)
    ACCENT, BG_FIG, BG_PANEL, FG, FG_DIM, OK, WARN,
    OrderLog, _shrink_gif, action_view,
)

FORMATIONS = ("concentrated", "diversionary", "wave")
LABEL = {"concentrated": "CONCENTRATED", "diversionary": "DIVERSIONARY", "wave": "WAVE"}
LABEL_KO = {"concentrated": "집중", "diversionary": "양동", "wave": "파상"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="세 대형을 한 화면에 나란히 굽는다")
    p.add_argument("--out", default=None)
    p.add_argument("--seeds", default="15,1,1",
                   help="집중,양동,파상 순서의 시드 (기본 15,1,1)")
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--step", type=int, default=1, help="프레임당 시뮬 step")
    p.add_argument("--replan", type=int, default=60)
    p.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    p.add_argument("--nets", type=int, default=3)
    p.add_argument("--backend", default=None, help="지정하면 LLM 지휘관")
    p.add_argument("--model", default=None)
    p.add_argument("--command", default="Capture all enemies")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--colors", type=int, default=0, help="0=후처리 없음(화질 우선)")
    p.add_argument("--zoom", dest="zoom", action="store_true", default=True)
    p.add_argument("--no-zoom", dest="zoom", action="store_false")
    p.add_argument("--satellite", dest="satellite", action="store_true", default=True)
    p.add_argument("--no-satellite", dest="satellite", action="store_false")
    p.add_argument("--score-map", dest="score_map", action="store_true", default=True)
    p.add_argument("--no-score-map", dest="score_map", action="store_false")
    a = p.parse_args(argv)
    if a.out is None:
        who = "heuristic" if not a.backend else (a.model or a.backend)
        who = re.sub(r"[^0-9A-Za-z._-]+", "-", str(who))
        a.out = os.path.join(REPO, "논문_그래프", "6_영상",
                             f"triptych_unet_{who}_seeds{a.seeds.replace(',', '-')}.gif")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager as _fm

    from boatattack_sim.eval import renderer
    from commander.fallback import heuristic_plan
    from commander.rl_bridge import build_battlefield_defense
    from commander.sim_bridge import plan_to_assign
    from commander.unet_bridge import CommandedCnnEnv

    _names = {f.name for f in _fm.fontManager.ttflist}
    _kf = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in _names), None)
    if _kf:
        matplotlib.rcParams["font.family"] = [_kf, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != 3:
        raise SystemExit(f"--seeds 는 3개여야 한다 (집중,양동,파상). 받음: {args.seeds}")

    commander = None
    who = "heuristic (no LLM)"
    if args.backend:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(REPO, ".env"))
        except ImportError:
            pass
        from commander import make_commander
        commander = make_commander(args.backend, args.model)
        who = getattr(commander, "model", args.backend)
        commander.warmup()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO, args.ckpt)
    print(f"[trip] U-Net={os.path.basename(ckpt)}  지휘관={who}  시드={seeds}")

    sims, logs, states = [], [], []
    for form, sd in zip(FORMATIONS, seeds):
        env = CommandedCnnEnv(ckpt, enemy_mode=form, nets_per_ship=args.nets)
        env.reset(seed=sd)
        env.running = True
        sims.append(env)
        logs.append(OrderLog())
        states.append({"last": -10 ** 9, "view": None, "flash": 0})

    bg_img = bg_extent = None
    if args.satellite:
        try:
            from commander.satellite import fetch_satellite_bg
            res = fetch_satellite_bg(sims[0].cfg.geo_lat, sims[0].cfg.geo_lon,
                                     sims[0].cfg.world_size)
            if res:
                bg_img, bg_extent = res
                print("[trip] 위성 배경 로드 완료")
        except Exception as e:                    # noqa: BLE001
            print(f"[trip] 위성 배경 실패({e}) → 해색 배경")

    # ── 레이아웃: 씬 3열 + 아래 지휘관 띠 ──
    fig = plt.figure(figsize=(16.5, 7.4), facecolor=BG_FIG)
    W, L, GAP = 0.305, 0.020, 0.012
    axes = [fig.add_axes((L + i * (W + GAP), 0.235, W, 0.665)) for i in range(3)]
    ax_cmd = fig.add_axes((L, 0.022, 3 * W + 2 * GAP, 0.185))

    title = fig.text(L, 0.973, "", color=FG, fontsize=14, ha="left", va="center",
                     fontweight="bold")
    sub = fig.text(L, 0.943, "", color=FG_DIM, fontsize=9, ha="left", va="center")
    heads = [fig.text(L + i * (W + GAP) + W / 2, 0.912, "", color=ACCENT, fontsize=11,
                      ha="center", va="center", fontweight="bold") for i in range(3)]
    stats_txt = [fig.text(L + i * (W + GAP) + W / 2, 0.222, "", color=FG, fontsize=9,
                          ha="center", va="top") for i in range(3)]

    def issue(i: int) -> None:
        env = sims[i]
        bf = build_battlefield_defense(env, command=args.command)
        plan = commander.plan(bf) if commander else heuristic_plan(bf)
        if commander is None:
            plan.rationale = f"휴리스틱 방어. {plan.rationale}"
        assign = plan_to_assign(plan, bf, mode="llm")
        env.set_plan(plan, args.command)
        committed = int((assign >= 0).sum())
        logs[i].add(int(env.t.flat[0]), plan,
                    f"deployed {committed}/{env.P}")
        states[i]["last"] = int(env.t.flat[0])
        states[i]["flash"] = 8

    for i in range(3):
        issue(i)

    def draw_cmd_panel():
        ax_cmd.clear()
        ax_cmd.set_xlim(0, 1); ax_cmd.set_ylim(0, 1)
        ax_cmd.set_xticks([]); ax_cmd.set_yticks([])
        ax_cmd.set_facecolor(BG_PANEL)
        flash_any = any(s["flash"] > 0 for s in states)
        for sp in ax_cmd.spines.values():
            sp.set_color(OK if flash_any else ACCENT)
            sp.set_linewidth(2.2 if flash_any else 1.2)
        ax_cmd.text(0.008, 0.86, f"LLM COMMANDER   ·   {who}", color=ACCENT,
                    fontsize=10.5, va="center", ha="left", fontweight="bold",
                    transform=ax_cmd.transAxes)
        ax_cmd.text(0.008, 0.60, f'COMMAND   "{args.command}"', color=FG_DIM,
                    fontsize=8.5, va="center", ha="left", transform=ax_cmd.transAxes)
        for i, form in enumerate(FORMATIONS):
            x = 0.008 + i * 0.334
            log = logs[i]
            last = log.rows[-1] if log.rows else None
            col = WARN if (last and last["fallback"]) else OK
            ax_cmd.text(x, 0.36, LABEL[form], color=ACCENT, fontsize=8.5,
                        va="center", ha="left", fontweight="bold",
                        transform=ax_cmd.transAxes)
            if last:
                ax_cmd.text(x + 0.105, 0.36,
                            f"t={last['t']:<5d} {last['alloc'][:44]}", color=col,
                            fontsize=8, va="center", ha="left",
                            transform=ax_cmd.transAxes)
                ax_cmd.text(x, 0.14,
                            f"계획 {len(log.rows)}회"
                            + (f" · 폴백 {log.n_fallback}" if log.n_fallback else ""),
                            color=FG_DIM, fontsize=7.5, va="center", ha="left",
                            transform=ax_cmd.transAxes)

    def update(_f):
        for i, env in enumerate(sims):
            for _ in range(max(1, args.step)):
                if not (env.running and not bool(env.done.flat[0])):
                    break
                env.step()
                t = int(env.t.flat[0])
                if args.replan > 0 and t - states[i]["last"] >= args.replan:
                    issue(i)

            fd = env.get_frame()
            ax = axes[i]
            view = None
            if args.zoom:
                view = action_view(fd, smooth=states[i]["view"])
                states[i]["view"] = view
            renderer.draw_scene(ax, fd, bg_img=bg_img, bg_extent=bg_extent,
                                show_help=False, view=view)
            ax.set_title("", loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#26C6DA"); sp.set_linewidth(1.3)
            if args.score_map and hasattr(env, "cnn_viz"):
                from boatattack_sim.eval import cnn_overlay as CO
                viz = env.cnn_viz()
                cfg_v, valid = viz["cfg"], viz["valid"]
                for pi in range(len(valid)):
                    if int(viz["assign"][pi]) < 0:
                        continue
                    CO.draw_valid(ax, cfg_v, valid[pi], pi, alpha=0.07, z=2.0)
                    if viz["prob"] is not None:
                        CO.draw_score(ax, cfg_v, viz["prob"][-1, pi], pi, alpha=0.38, z=2.2)
                    if viz["pix"] is not None:
                        CO.draw_picks(ax, cfg_v, viz["pix"][pi],
                                      None if viz["offset"] is None else viz["offset"][pi],
                                      p=pi, z=7.0)
            for art in (list(ax.texts) + list(ax.lines)
                        + list(ax.patches) + list(ax.collections)):
                art.set_clip_on(True)
                art.set_clip_path(ax.patch)

            st = fd.get("stats", {})
            alive_allies = int(np.asarray(fd["ally_alive"], bool).sum())
            heads[i].set_text(f"{LABEL[FORMATIONS[i]]}  ·  {LABEL_KO[FORMATIONS[i]]}"
                              f"   (seed {seeds[i]})")
            stats_txt[i].set_text(
                f"captured {int(st.get('captures', 0))}/{env.cfg.n_enemies}   "
                f"breached {int(st.get('breaches', 0))}   "
                f"allies {alive_allies}/{env.P}   "
                f"nets {int(st.get('nets_used', 0))}   t={int(env.t.flat[0])}")
            if states[i]["flash"] > 0:
                states[i]["flash"] -= 1

        title.set_text("USV SWARM DEFENSE  —  U-Net score-map policy across three attack patterns")
        sub.set_text(f"maneuver: U-Net ({os.path.basename(ckpt)})   ·   "
                     f"commander: {who}   ·   replan every {args.replan} steps")
        draw_cmd_panel()
        return []

    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, update, frames=args.frames,
                         interval=1000 // max(args.fps, 1),
                         blit=False, cache_frame_data=False)
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    t0 = time.perf_counter()
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi,
              savefig_kwargs={"facecolor": BG_FIG})
    plt.close(fig)
    if args.colors:
        _shrink_gif(out, args.colors)
    print(f"[trip] 완료: {out}  ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{args.frames} 프레임, {time.perf_counter() - t0:.0f}s)")
    for i, form in enumerate(FORMATIONS):
        print(f"[trip]   {form:<14s} seed {seeds[i]:<3d} stats={dict(sims[i].stats)} "
              f"계획 {len(logs[i].rows)}회 폴백 {logs[i].n_fallback}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
