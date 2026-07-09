"""학습된 RL 정책(강화학습)으로 경로를 추론해 방어를 구동·시청하는 뷰어.

One-Way_Towing 에서 가장 최근 학습한 정책(attention 인코더 + LSTM, 잔차 방식)을 그대로
불러와 DefenseVecEnv(1월드)를 구동한다. LLM 지휘관 버전(run_commander_ui.py)과 달리
경로·그물 결정 전체를 RL 정책이 담당한다(휴리스틱 기준경로 위에 ±wp_adjust_max 잔차 보정).

정책 = boatattack_sim/models/rl_latest.pt (run_20260709-161852/best.pt).
결정 주기(decision_period)마다 관측→정책→행동→경로 재계획. LSTM hidden 은 에피소드 내 유지.

실행:
    python run_rl_play.py                       # 최신 정책, greedy
    python run_rl_play.py --enemy wave          # 적 대형
    python run_rl_play.py --ckpt boatattack_sim/models/rl_latest_final.pt
    python run_rl_play.py --sample              # 결정적(greedy) 대신 분포 샘플
    python run_rl_play.py --gain 8              # RL 잔차(휴리스틱 이탈)를 8배 과장해 시각화
조작키: [space] 재생/일시정지  [r] 리셋  [g] greedy/sample 토글
       [ ] 잔차배율 -/+ (RL 보정 과장)  [v] APF(충돌회피) 토글  [q] 종료
"""
import argparse

import numpy as np
import torch


# ── numpy obs ↔ torch (배치 B=N×P), 행동 dict → env (train/grpo.py 와 동일) ──
def obs_to_torch(obs, device):
    N, P = obs["own"].shape[:2]
    B = N * P
    def ten(x, dt):
        return torch.as_tensor(np.ascontiguousarray(x.reshape(B, *x.shape[2:])),
                               dtype=dt, device=device)
    return {"own": ten(obs["own"], torch.float32),
            "enemy": ten(obs["enemy"], torch.float32),
            "enemy_mask": ten(obs["enemy_mask"], torch.bool),
            "ally": ten(obs["ally"], torch.float32),
            "ally_mask": ten(obs["ally_mask"], torch.bool)}


def act_to_env(a, N, P, Kw):
    cont = a["cont"].view(N, P, -1).cpu().numpy()
    out = {"net_go": a["netgo"].view(N, P, Kw).cpu().numpy()}
    if cont.shape[-1] == 7:
        out["fan"] = cont
    else:
        out["wp"] = cont[..., :Kw * 2].reshape(N, P, Kw, 2)
    if "rot" in a:
        out["rot"] = a["rot"].view(N, P).cpu().numpy()
    return out


def greedy(p):
    g = {"cont": p["cont_mean"], "netgo": (p["netgo"] > 0).float()}
    if "rot_mean" in p:
        g["rot"] = p["rot_mean"]
    return g


def _ally_paths(env, w):
    paths = []
    for p in range(env.P):
        wps = [{"x": float(env.route[w, p, k, 0]),
                "y": float(env.route[w, p, k, 1]), "paint": False}
               for k in range(env.Kw)]
        if bool(env.doing_net[w, p]):
            wps.append({"x": float(env.net_end[w, p, 0]),
                        "y": float(env.net_end[w, p, 1]), "paint": True})
        paths.append(wps)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="boatattack_sim/models/rl_latest.pt")
    ap.add_argument("--enemy", default="rotate")
    ap.add_argument("--sample", action="store_true", help="greedy 대신 분포 샘플")
    ap.add_argument("--spf", type=int, default=2, help="프레임당 micro-step 수")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="잔차(RL 보정) 배율. 1=충실, >1=휴리스틱 이탈을 과장해 RL 기여를 시각화 "
                         "(학습분포 이탈 → 부정확·불안정 가능). 실행 중 [ ] 키로 조절.")
    ap.add_argument("--apf", action="store_true",
                    help="APF(충돌회피 안전층) ON. 기본 OFF=순수 RL 경로. 실행 중 v 키 토글.")
    args = ap.parse_args()

    import warnings
    import matplotlib
    from matplotlib import font_manager as _fm
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from boatattack_sim.eval import renderer  # rcParams 폰트 덮어씀 → 뒤에서 한글폰트 복구
    from boatattack_sim.model.actor import load_actor
    from boatattack_sim.env.defense_env import DefenseVecEnv
    from commander.satellite import fetch_satellite_bg

    _names = {f.name for f in _fm.fontManager.ttflist}
    _kfont = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in _names), None)
    if _kfont:
        matplotlib.rcParams["font.family"] = [_kfont, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    device = "cpu"
    print(f"정책 로딩 중… ({args.ckpt})")
    actor, cfg = load_actor(args.ckpt, device=device)
    actor.eval()
    print(f"  backbone={getattr(cfg,'backbone',None)} attn={cfg.attn_backbone} recurrent={actor.recurrent} "
          f"Kw={cfg.transit_wp} wp_adjust_max={cfg.wp_adjust_max}")

    cfg.avoid_steer = bool(args.apf)   # 기본 OFF: 순수 RL 경로(APF 안전층 없음)
    cfg.enemy_wave_near = 2600.0       # 파상: 웨이브 간 텀 확실히 (gap↑, near↓ → 3단이 맵 안)
    cfg.enemy_wave_gap = 1800.0
    cfg.spawn_phase_lo = 1.0           # 스폰 랜덤 당김 끄기 → 웨이브 텀 설계대로
    cfg.free_current_wp = True         # 추종 중인 현재 WP도 매 결정 변동(고정 해제)
    # 재배정 시 새 경로를 WP1(처음)부터 추종 (preserve_ptr_on_reeng 는 끔=기본)
    env = DefenseVecEnv(num_worlds=1, cfg=cfg, enemy_mode=args.enemy)
    P, Kw = env.P, actor.Kw

    # 위성 배경 (정책 config 의 지오 앵커; 실패 시 해색 폴백)
    bg_img = bg_extent = None
    try:
        print("위성 배경 로딩 중…")
        res = fetch_satellite_bg(cfg.geo_lat, cfg.geo_lon, cfg.world_size)
        if res:
            bg_img, bg_extent = res
            print("위성 배경 로드 완료")
    except Exception as e:
        print(f"위성 배경 오류({e}) → 해색 폴백")

    _SK = ("captures", "breaches", "ally_collisions", "nets_used")
    state = {"h": actor.init_hidden(P, device), "micro": 0, "ev": None,
             "running": True, "greedy": not args.sample, "gain": float(args.gain),
             "stats": {k: 0 for k in _SK + ("survived",)},
             "prev": {k: 0.0 for k in _SK}}

    def decide():
        ot = obs_to_torch(env.build_obs(), device)
        with torch.no_grad():
            p, state["h"] = actor(ot, state["h"])
            act = greedy(p) if state["greedy"] else actor.sample(p)
        if state["gain"] != 1.0:                    # 잔차(휴리스틱 이탈) 배율 → RL 보정 과장
            act = dict(act); act["cont"] = act["cont"] * state["gain"]
        env._apply_actions(act_to_env(act, 1, P, Kw))
        state["ev"] = env.fresh_ev()
        state["prev"] = {k: 0.0 for k in state["prev"]}

    def step_one():
        if bool(env.done[0]):
            env._spawn_worlds(np.array([0])); state["micro"] = 0; state["ev"] = None
            state["stats"] = {k: 0 for k in state["stats"]}
            state["h"] = actor.init_hidden(P, device)
        if state["micro"] % cfg.decision_period == 0:
            decide()
        env._micro(state["ev"])
        for k in state["prev"]:
            cur = float(state["ev"][k][0])
            state["stats"][k] += cur - state["prev"][k]; state["prev"][k] = cur
        state["micro"] += 1

    def frame():
        w = 0
        fd = {
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
        return fd

    fig, ax = plt.subplots(figsize=(9.5, 9))
    try:
        fig.canvas.manager.set_window_title("RL 정책 방어 시청 — policy play")
    except Exception:
        pass
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.06)

    def update(_):
        if state["running"] and not bool(env.done[0]):
            for _ in range(args.spf):
                step_one()
        elif state["running"] and bool(env.done[0]):
            step_one()   # done 처리(리셋) 1회
        fd = frame()
        renderer.draw_scene(ax, fd, bg_img=bg_img, bg_extent=bg_extent)
        s = state["stats"]
        mode = "greedy" if state["greedy"] else "sample"
        ax.text(0.30, 0.99,
                f"RL POLICY [{mode}]  잔차배율 x{state['gain']:.1f}  "
                f"APF {'ON' if env.cfg.avoid_steer else 'OFF'}  대형={args.enemy}  "
                f"포획 {s['captures']}  돌파 {s['breaches']}  그물 {s['nets_used']}  "
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
            state["h"] = actor.init_hidden(P, device)
        elif k == "g":
            state["greedy"] = not state["greedy"]
        elif k == "]":
            state["gain"] = round(state["gain"] + 2.0, 1)
        elif k == "[":
            state["gain"] = max(0.0, round(state["gain"] - 2.0, 1))
        elif k == "v":
            env.cfg.avoid_steer = not env.cfg.avoid_steer
        elif k == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    print(f"뷰어 실행: enemy={args.enemy}, {'greedy' if state['greedy'] else 'sample'}. "
          f"space=일시정지 r=리셋 g=greedy토글 q=종료")
    plt.show()
    _ = anim


if __name__ == "__main__":
    main()
