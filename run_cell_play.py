"""학습된 '셀선택' RL 정책(CellPointerActor)으로 방어를 구동·시청하는 독립 뷰어.

One-Way_Towing 에서 학습한 셀 정책(run_20260713-233000)을 그대로 불러와 DefenseVecEnv(1월드)를
구동한다. 잔차(wp_delta) 방식인 run_rl_play.py 와 달리, 이 정책은 **이산 셀 선택** 행동공간을
쓴다: 모선중심 20×20 직교격자를 요격환형[r_min,r_max]으로 거른 정적 후보셀(C개) 중, 배마다
K=cell_nets 개 셀을 pointer-attention 으로 골라 그 셀들을 잇는 연속 그물벽을 친다.

정책 = boatattack_sim/models/cell_latest.pt (run_20260713-233000/best.pt).
체크포인트에 저장된 자체 config(world 12600·cartesian 20·evade/weave 등)를 그대로 사용한다.
결정주기(decision_period)마다 build_cell_obs→정책 greedy_joint(월드병렬·배순차·교차잠금)→셀선택→
경로/그물 재생성. LLM 지휘관은 개입하지 않는다(순수 RL 셀 정책 확인용).

실행:
    python run_cell_play.py                         # 최신 셀 정책, diversionary
    python run_cell_play.py --enemy wave
    python run_cell_play.py --ckpt boatattack_sim/models/cell_latest_final.pt
    python run_cell_play.py --heur                  # 정책 대신 휴리스틱 셀선택(비교용)
    python run_cell_play.py --no-joint              # 배별 독립 greedy(교차잠금 끔)
조작키: [space] 재생/일시정지  [r] 리셋  [c] 셀 오버레이 토글  [v] APF 토글  [q] 종료
"""
import argparse

import numpy as np
import torch


def _ally_paths(env, w):
    """배별 자동조종 경로 WP(+전개중 그물끝) → 렌더용 딕셔너리 리스트."""
    paths = []
    for p in range(env.P):
        wps = [{"x": float(env.route[w, p, k, 0]),
                "y": float(env.route[w, p, k, 1]), "paint": bool(env.net_mask[w, p, k])}
               for k in range(env.Kw)]
        if bool(env.doing_net[w, p]):
            wps.append({"x": float(env.net_end[w, p, 0]),
                        "y": float(env.net_end[w, p, 1]), "paint": True})
        paths.append(wps)
    return paths


# 배별 색 (make_cell_gif 과 동일 팔레트)
SHIP_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#A78BFA", "#FF9F43"]


def _overlay_cells(ax, env, sel, valid, show):
    """후보셀(faint) + 배별 선택셀(강조·벽). sel[P,K]=선택 셀 idx, valid[p]=배 p 유효 후보셀 idx."""
    if not show:
        return
    cw = env.cell_world
    if valid is not None:                       # 배별 유효 후보셀을 배색으로 옅게
        for p, vidx in enumerate(valid):
            if len(vidx) == 0:
                continue
            col = SHIP_COLORS[p % len(SHIP_COLORS)]
            ax.scatter(cw[vidx, 0], cw[vidx, 1], s=14, c=col, alpha=0.22,
                       edgecolors="none", zorder=2.3)
    else:                                        # 후보 전체(faint)
        ax.scatter(cw[:, 0], cw[:, 1], s=5, c="#4DD0E1", alpha=0.15, zorder=2.2)
    if sel is not None:                          # 선택 셀 — 배색 강조 + 잇는 벽
        for p in range(env.P):
            col = SHIP_COLORS[p % len(SHIP_COLORS)]
            pts = cw[sel[p]]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.4, alpha=0.7, zorder=6.5)
            ax.scatter(pts[:, 0], pts[:, 1], s=70, facecolors=col, edgecolors="white",
                       lw=1.2, zorder=7)


def main() -> None:
    ap = argparse.ArgumentParser(description="셀선택 RL 정책 방어 시청")
    ap.add_argument("--ckpt", default="boatattack_sim/models/cell_latest.pt")
    ap.add_argument("--enemy", default="diversionary")
    ap.add_argument("--spf", type=int, default=3, help="프레임당 micro-step 수")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heur", action="store_true", help="정책 대신 휴리스틱 셀선택(비교용)")
    ap.add_argument("--no-joint", action="store_true",
                    help="배별 독립 greedy(교차잠금 끔). 기본=joint(월드병렬·배순차·교차셀 잠금)")
    ap.add_argument("--apf", action="store_true",
                    help="APF(충돌회피) 강제 ON. 미지정 시 체크포인트 config 값 사용. v 키로 토글.")
    args = ap.parse_args()

    import warnings
    import matplotlib
    from matplotlib import font_manager as _fm
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from boatattack_sim.eval import renderer   # rcParams 폰트 덮어씀 → 뒤에서 한글폰트 복구
    from boatattack_sim.env.defense_env import DefenseVecEnv
    from boatattack_sim.model.cell_actor import load_cell_actor, cell_obs_to_torch
    from commander.satellite import fetch_satellite_bg

    _names = {f.name for f in _fm.fontManager.ttflist}
    _kfont = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in _names), None)
    if _kfont:
        matplotlib.rcParams["font.family"] = [_kfont, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    device = "cpu"
    print(f"셀 정책 로딩 중... ({args.ckpt})")
    actor, cfg = load_cell_actor(args.ckpt, device=device)
    actor.eval()
    if not getattr(cfg, "cell_action", False):
        raise SystemExit(f"[오류] {args.ckpt} 는 셀선택(cell_action) 정책이 아닙니다. "
                         f"잔차 정책이면 run_rl_play.py 를 쓰세요.")
    if args.apf:
        cfg.avoid_steer = True
    print(f"  grid={cfg.cell_grid} cart_n={cfg.cell_cart_n} r=[{cfg.cell_r_min},{cfg.cell_r_max}] "
          f"K(cell_nets)={cfg.cell_nets} world={cfg.world_size} dp={cfg.decision_period} "
          f"APF={cfg.avoid_steer}")

    env = DefenseVecEnv(num_worlds=1, cfg=cfg, enemy_mode=args.enemy, seed=args.seed)
    env.reset(seed=args.seed)
    N, P = env.N, env.P
    mask_r = float(env._cell_half())            # joint 교차잠금 반경(격자 반칸)
    print(f"후보셀 C={env.n_cells}  joint 잠금반경={mask_r:.0f}m")

    # 위성 배경
    bg_img = bg_extent = None
    try:
        print("위성 배경 로딩 중...")
        res = fetch_satellite_bg(cfg.geo_lat, cfg.geo_lon, cfg.world_size)
        if res:
            bg_img, bg_extent = res
            print("위성 배경 로드 완료")
    except Exception as e:
        print(f"위성 배경 오류({e}) -> 해색 폴백")

    _SK = ("captures", "breaches", "ally_collisions", "nets_used")
    state = {"micro": 0, "ev": None, "running": True, "sel": None, "valid": None,
             "show_cells": True, "joint": not args.no_joint,
             "stats": {k: 0 for k in _SK}, "prev": {k: 0.0 for k in _SK}}

    def decide():
        obs = env.build_cell_obs()
        state["valid"] = [np.where(~obs["cell_mask"][0, p])[0] for p in range(P)]
        if args.heur:
            cells = env.heuristic_cells()               # [1,P,K]
        else:
            with torch.no_grad():
                p, _ = actor(cell_obs_to_torch(obs, device))
                if state["joint"]:
                    g = actor.greedy_joint(p, N, P, cell_world=env.cell_world, mask_radius=mask_r)
                else:
                    g = actor.greedy(p)
                cells = g["cells"].view(N, P, -1).cpu().numpy()
        state["sel"] = cells[0]                          # [P,K]
        env._apply_actions({"cells": cells})
        state["ev"] = env.fresh_ev()
        state["prev"] = {k: 0.0 for k in state["prev"]}

    def step_one():
        if bool(env.done[0]):
            env._spawn_worlds(np.array([0])); state["micro"] = 0; state["ev"] = None
            state["stats"] = {k: 0 for k in state["stats"]}
        if state["micro"] % cfg.decision_period == 0:
            decide()
        env._micro(state["ev"])
        for k in state["prev"]:
            cur = float(state["ev"][k][0])
            state["stats"][k] += cur - state["prev"][k]; state["prev"][k] = cur
        state["micro"] += 1

    def frame():
        w = 0
        return {
            "world_size": cfg.world_size, "cell_size": cfg.cell_size,
            "t": int(env.t[w]), "done": bool(env.done[w]),
            "mothership": env.center, "mothership_radius": cfg.mothership_radius,
            "moback_size": cfg.moback_size, "moback_heading": cfg.moback_heading,
            "enemy_pos": env.e_pos[w], "enemy_hdg": env.e_hdg[w],
            "enemy_alive": env.e_alive[w], "enemy_size": cfg.enemy_size,
            "ally_pos": env.a_pos[w], "ally_hdg": env.a_hdg[w],
            "ally_paths": _ally_paths(env, w),
            "ally_nets": env.a_nets[w], "ally_painting": env.doing_net[w],
            "ally_alive": env.a_alive[w],
            "assign": env._assign[w], "assignI": env._assignI[w],
            "route": env.route[w], "net_mask": env.net_mask[w],
            "ship_len": cfg.ship_len, "ship_wid": cfg.ship_wid,
            "painted": env.painted[w],
            "selected": -1, "manual": False, "running": state["running"],
            "net_stage": 0, "stats": dict(state["stats"]),
            "n_alive": int(env.e_alive[w].sum()),
            "n_clusters": cfg.n_clusters, "cluster_gap_deg": cfg.cluster_gap_deg,
            "enemy_speed": cfg.enemy_speed, "show_clusters": True,
            "show_residual": False, "wp_adjust_max": cfg.wp_adjust_max,
            "enemy_mode": str(env.world_mode[w]),
        }

    fig, ax = plt.subplots(figsize=(9.5, 9))
    try:
        fig.canvas.manager.set_window_title("셀선택 RL 정책 방어 시청 - cell policy play")
    except Exception:
        pass
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.06)

    def update(_):
        if state["running"]:
            for _ in range(args.spf):
                step_one()
        fd = frame()
        renderer.draw_scene(ax, fd, bg_img=bg_img, bg_extent=bg_extent)
        _overlay_cells(ax, env, state["sel"], state["valid"], state["show_cells"])
        s = state["stats"]
        src = "휴리스틱" if args.heur else ("정책-joint" if state["joint"] else "정책-indep")
        ax.text(0.30, 0.99,
                f"CELL POLICY [{src}]  APF {'ON' if env.cfg.avoid_steer else 'OFF'}  "
                f"대형={args.enemy}  포획 {s['captures']}  돌파 {s['breaches']}  "
                f"충돌 {s['ally_collisions']}  그물 {s['nets_used']}  "
                f"생존적 {int(env.e_alive[0].sum())}",
                transform=ax.transAxes, color="#00E676", fontsize=9,
                va="top", ha="left", weight="bold")
        return []

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == " ":
            state["running"] = not state["running"]
        elif k == "r":
            env._spawn_worlds(np.array([0])); state["micro"] = 0; state["ev"] = None
            state["stats"] = {k2: 0 for k2 in state["stats"]}
        elif k == "c":
            state["show_cells"] = not state["show_cells"]
        elif k == "v":
            env.cfg.avoid_steer = not env.cfg.avoid_steer
        elif k == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    print(f"뷰어 실행: enemy={args.enemy}. space=일시정지 r=리셋 c=셀토글 v=APF q=종료")
    plt.show()
    _ = anim


if __name__ == "__main__":
    main()
