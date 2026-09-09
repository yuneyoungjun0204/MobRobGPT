"""로스백(실제 적선) + 가상(시뮬) 아군 혼합 정책 추론 — U-Net(CNN 점수맵) / 셀선택 겸용.

이전엔 --unet --ros2 로 ROS2 실시간 센서(아군까지)를 받았지만, 이 스크립트는 **녹화된
로스백의 적선만 실데이터**로 재생하고 아군은 없으므로 시뮬 물리로 가상 기동시킨다
(commander/replay_cnn_env.py::ReplayCnnEnv 또는 commander/replay_cell_env.py::ReplayCellEnv).
--ckpt 하나만 바꾸면 체크포인트의 config(`cell_action`/`cnn_action`)를 읽어 어느 정책 계열인지
자동 판별해 알맞은 환경을 쓴다 — 사용자가 정책 종류를 따로 지정할 필요 없다.

맵 실제 크기는 --span 으로 지정한다 — 8 m 짜리 로스백이든 33 m 짜리든 같은 명령으로 그대로
동작한다(SimScale, 문서 docs/unet_model_deploy.md §10~11 참고).

실행:
    python run_replay_infer.py \\
        --bag /home/yune/Downloads/ros_data/S03-gcs --span 8 \\
        --ckpt boatattack_sim/models/u-net_map.pt --llm ollama

    python run_replay_infer.py --bag <path> --span 33 --llm heuristic \\
        --max-decisions 5 --out out.json     # LLM 없이 휴리스틱 배정만으로 빠른 확인

    python run_replay_infer.py --bag <path> --span 8 --llm heuristic --viz
        # matplotlib 창으로 실시간 시각화(대도·경로·그물·점수맵 히트맵)
        # 조작: [space] 재생/일시정지  [q] 종료

    python run_replay_infer.py --bag <path> --span 10 \\
        --ckpt specialized_cos6k_model/concentrated/best.pt \\
        --specialized specialized_cos6k_model --llm heuristic --viz
        # 셀선택 정책(자동 감지) + 대형(집중/양동/파상) 특화 라우팅
"""
from __future__ import annotations

import argparse
import json
import sys


def _detect_policy_kind(ckpt: str) -> str:
    """체크포인트 config 의 `cell_action`/`cnn_action` 플래그로 정책 계열을 판별한다.

    두 모델 계열(U-Net CNN 점수맵 vs 셀선택)은 관측·행동공간·환경 클래스가 전혀 달라
    (docs/unet_model_deploy.md vs commander/cell_bridge.py) 사용자가 매번 지정하게 하면
    실수하기 쉽다 — 체크포인트 자체가 이미 이 플래그를 갖고 있으므로 그걸 신뢰한다.
    """
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    if cfg is None:
        raise ValueError(f"{ckpt}: 체크포인트에 'config' 가 없습니다.")
    get = cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d)
    if bool(get("cell_action", False)):
        return "cell"
    if bool(get("cnn_action", False)):
        return "cnn"
    raise ValueError(f"{ckpt}: cell_action/cnn_action 플래그가 모두 없어 정책 종류를 판별할 수 없습니다.")


def _make_commander(backend: str, model: str | None):
    """backend='heuristic' 이면 LLM 없이 위협비례 휴리스틱만 쓰는 어댑터를 반환."""
    if backend == "heuristic":
        from commander.fallback import heuristic_plan

        class _Heuristic:
            def plan(self, state):
                return heuristic_plan(state)

        return _Heuristic()
    from commander import make_commander
    return make_commander(backend, model)


def _decode_route(env, p: int, w: int = 0) -> dict:
    """route[p] → 실 좌표(m) 웨이포인트 전체 + 정지규칙(§6.3).

    두 정책 계열 다 route[0]은 항상 이동 전용(transit) leg다 — CNN은 픽셀 2점 규약이
    그렇게 정의하고(§6.3), 셀선택도 `cell_action.build_routes_from_cells`가
    `net_mask[1:L]=True`(leg 0 제외)로 만든다. 그래서 route[1]=그물벽 끝점,
    net_mask[1]=그 leg 여부는 두 계열 모두에서 참이다. `waypoints_m`/`net_mask`(Kw 길이
    전체)를 기본으로 반환하고, 그 첫 두 점을 `move_wp_m`/`net_wp_m`/`net_leg`로도 채운다.
    """
    stop = bool(env._assign[w, p] < 0) or bool(env.a_nets[w, p] <= 0) or not bool(env.a_alive[w, p])
    if stop:
        return {"stop": True, "move_wp_m": None, "net_wp_m": None, "net_leg": False,
                "waypoints_m": None, "net_mask": None}
    wps = [[round(float(c), 4) for c in env.scale.sim_to_enu(env.route[w, p, k])]
           for k in range(env.Kw)]
    net_mask = [bool(x) for x in env.net_mask[w, p]]
    return {
        "stop": False,
        "move_wp_m": wps[0],
        "net_wp_m": wps[1] if len(wps) > 1 else None,
        "net_leg": net_mask[1] if len(net_mask) > 1 else False,
        "waypoints_m": wps,
        "net_mask": net_mask,
    }


def _overlay_unet(ax, viz: dict) -> None:
    """U-Net 점수맵 오버레이 — `run_commander_ui.py::_overlay_score_cmd` 와 동일한 그림 방식.

    배별 유효마스크(옅게) + 점수맵 heatmap(자기회귀 마지막 단계) + 선택 픽셀 2점(이동 WP·
    그물 벽 끝점). 미배정 배는 점수맵이 균등분포라 의미 없으므로 그리지 않는다(§3.3).
    """
    from boatattack_sim.eval import cnn_overlay as CO
    cfg = viz["cfg"]
    valid, prob, pix, off = viz["valid"], viz["prob"], viz["pix"], viz["offset"]
    for p in range(len(valid)):
        if int(viz["assign"][p]) < 0:
            continue
        CO.draw_valid(ax, cfg, valid[p], p, alpha=0.14, z=2.0)
        if prob is not None:
            CO.draw_score(ax, cfg, prob[-1, p], p, alpha=0.55, z=2.2)
        if pix is not None:
            CO.draw_picks(ax, cfg, pix[p], None if off is None else off[p], p=p, z=7.0)


_CELL_SHIP_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#A78BFA"]


def _overlay_cells(ax, viz: dict) -> None:
    """셀 후보 오버레이 — `run_commander_ui.py::_overlay_cells_cmd` 와 동일한 그림 방식.

    하늘색=전체후보, 배색=배별유효, 빨강X=그물배제(이미 설치된 그물 근처), 흰테두리=선택 셀.
    """
    cw = viz["world"]
    ax.scatter(cw[:, 0], cw[:, 1], s=26, c="#8fe0ff", alpha=0.55,
               edgecolors="#0a2a3a", linewidths=0.5, zorder=2.2)
    for p, (val, exc) in enumerate(zip(viz["valid"], viz["excluded"])):
        col = _CELL_SHIP_COLORS[p % len(_CELL_SHIP_COLORS)]
        if len(val):
            ax.scatter(cw[val, 0], cw[val, 1], s=70, c=col, alpha=0.55,
                       edgecolors="white", linewidths=0.6, zorder=2.6)
        if len(exc):
            ax.scatter(cw[exc, 0], cw[exc, 1], s=130, marker="X", c="#FF1744",
                       edgecolors="white", linewidths=1.2, zorder=3.2)
    sel = viz.get("selected")
    if sel is not None:
        for p in range(len(sel)):
            col = _CELL_SHIP_COLORS[p % len(_CELL_SHIP_COLORS)]
            ax.scatter(cw[sel[p], 0], cw[sel[p], 1], s=170, facecolors=col,
                       edgecolors="white", linewidths=2.0, zorder=6)
            ax.scatter(cw[sel[p], 0], cw[sel[p], 1], s=320, facecolors="none",
                       edgecolors=col, linewidths=1.4, alpha=0.8, zorder=5.8)


def _run_viz(env, advance_one_micro, max_reached, spf: int, log: list,
             bg_img=None, bg_extent=None, hide_cells: bool = False) -> None:
    """matplotlib 실시간 창. `run_cell_play.py` 와 동일한 렌더 파이프라인을 재사용한다."""
    import warnings
    import matplotlib
    from matplotlib import font_manager as _fm
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from boatattack_sim.eval import renderer   # import 시 rcParams['font.family'] 를 덮어씀 → 뒤에서 복구

    # 리눅스 CI/서버 이미지엔 Malgun Gothic/AppleGothic/NanumGothic이 없는 경우가 많다 --
    # 대신 fontconfig가 아는 Noto CJK(.ttc)를 matplotlib에 직접 등록해본다. 이 파일은
    # 지역별로 이름이 갈려도(JP/KR/...) 글리프 자체는 통합 CJK+한글을 다 담고 있어서,
    # 어떤 지역 이름으로 등록되든 한글 렌더링에는 문제가 없다.
    try:
        import subprocess
        for _ttc in ("Noto Sans CJK KR", "Noto Sans CJK JP"):
            _out = subprocess.run(["fc-match", "-f", "%{file}", _ttc],
                                  capture_output=True, text=True, timeout=2).stdout.strip()
            if _out:
                _fm.fontManager.addfont(_out)
                break
    except Exception:
        pass
    _names = {f.name for f in _fm.fontManager.ttflist}
    _kfont = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic")
                    if f in _names), None) or next((f for f in _names if "CJK" in f), None)
    if _kfont:
        matplotlib.rcParams["font.family"] = [_kfont, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    # 그래도 못 찾은 글리프가 있을 수 있으니 경고는 항상 죽여둔다 -- matplotlib 버전마다
    # 메시지 문구가 "missing from font"/"missing from current font"로 갈려서 정규식
    # 하나로는 둘 다 못 잡는다(구버전 필터가 실제로 안 먹었던 원인). "font"만 걸어 포괄한다.
    warnings.filterwarnings("ignore", message="Glyph .* missing from .*font")

    ui = {"running": True, "done": False}

    policy_label = "셀선택" if hasattr(env, "cell_viz") else "CNN 점수맵(U-Net)"
    fig, ax = plt.subplots(figsize=(9.5, 9))
    try:
        fig.canvas.manager.set_window_title(f"로스백 리플레이 — {policy_label} 정책 (실제 적 + 가상 아군)")
    except Exception:
        pass
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.06)

    def update(_):
        if ui["running"] and not ui["done"]:
            for _ in range(spf):
                if max_reached() or not advance_one_micro():
                    ui["done"] = True
                    break
        fd = env.get_frame()
        renderer.draw_scene(ax, fd, bg_img=bg_img, bg_extent=bg_extent)
        if hasattr(env, "cnn_viz"):
            _overlay_unet(ax, env.cnn_viz())
        elif hasattr(env, "cell_viz") and not hide_cells:
            _overlay_cells(ax, env.cell_viz())
        status = "종료(로스백 소진 또는 --max-decisions 도달)" if ui["done"] else ("재생" if ui["running"] else "일시정지")
        formation = f"  대형={getattr(env, '_formation', None)}" if getattr(env, "_formation", None) else ""
        ax.text(0.02, 0.99,
                f"[{status}]  실제 적(로스백) + 가상 아군(시뮬)  t={env.bag_time_real:6.2f}s  "
                f"결정={len(log)}  적생존={int(env.e_alive[0].sum())}{formation}",
                transform=ax.transAxes, color="#00E676", fontsize=9,
                va="top", ha="left", weight="bold")
        return []

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == " ":
            ui["running"] = not ui["running"]
        elif k == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    print("뷰어 실행: space=재생/일시정지  q=종료")
    plt.show()
    _ = anim


def main() -> None:
    ap = argparse.ArgumentParser(
        description="로스백 실제 적선 + 가상 아군 혼합 정책 추론 (CNN 점수맵 U-Net / 셀선택 자동 감지)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--bag", required=True, help="rosbag2 디렉터리 경로 (metadata.yaml 포함)")
    ap.add_argument("--span", type=float, required=True,
                    help="실제 운용 박스 한 변[m] — 이번 8, 다음번 33 등 로스백마다 다르게 지정")
    ap.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    ap.add_argument("--n-tracks", type=int, default=5, help="로스백 안 적선 척수")
    ap.add_argument("--first-index", type=int, default=1, help="적선 토픽 번호 시작값 (usv{N})")
    ap.add_argument("--pose-topic-fmt", default="/usv/usv{i}/pose")
    ap.add_argument("--heading-topic-fmt", default="/usv/usv{i}/heading")
    ap.add_argument("--ally-speed-real", type=float, default=0.3,
                    help="가상 아군의 실제 순항속도[m/s] — SimScale 시계 스케일 산출용")
    ap.add_argument("--nets", type=int, default=3, help="배당 그물 장수(--ckpt 가 CNN 점수맵 모델일 때만 적용)")
    ap.add_argument("--specialized", default=None, metavar="ROOT_DIR",
                    help="--ckpt 가 셀선택 모델일 때: 대형(집중/양동/파상) 특화 라우팅 루트 디렉터리 "
                         "(예: specialized_cos6k_model — {ROOT}/{concentrated,diversionary,wave}/best.pt)")
    ap.add_argument("--net-reload-period", type=float, default=None,
                    help="그물 소진 후 재보급 주기[실제 초], 0보다 커야 함 "
                         "(기본: nets_per_ship×결정주기×6 에서 자동 산출). "
                         "너무 짧으면 소진 자체가 무의미해지고, 너무 길면 재보급까지 오래 정지한다")
    ap.add_argument("--enemy-mode", default="wave", help="초기 스폰용(매 tick 실데이터로 덮어씀)")
    ap.add_argument("--llm", default="ollama", choices=["ollama", "openai", "heuristic"],
                    help="배정 지휘관 백엔드. LLM 실패 시 각 어댑터가 자동으로 휴리스틱 폴백.")
    ap.add_argument("--model", default=None, help="LLM 모델명 (미지정 시 백엔드 기본값)")
    ap.add_argument("--command", default=None, help="지휘관에게 줄 자연어 지시(선택)")
    ap.add_argument("--replan-period", type=int, default=8,
                    help="지휘관을 다시 부르는 결정 간격(결정=decision_period micro-step)")
    ap.add_argument("--max-decisions", type=int, default=None, help="처리할 결정 횟수 상한(기본: 로스백 끝까지)")
    ap.add_argument("--geo", type=float, nargs=2, default=None, metavar=("LAT", "LON"),
                    help="육지 마스크용 실 앵커 위경도(선택, 미지정시 체크포인트 기본값)")
    ap.add_argument("--enu-origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="맵 중앙(모선)에 대응할 실좌표[m](선택, 미지정시 로스백 적선 전체 중심)")
    ap.add_argument("--out", default=None, help="결정별 명령 로그를 저장할 JSON 경로(선택)")
    ap.add_argument("--viz", action="store_true",
                    help="matplotlib 창으로 실시간 시각화(대도·경로·그물·정책별 오버레이: "
                         "CNN=점수맵 히트맵, 셀선택=후보셀)")
    ap.add_argument("--spf", type=int, default=3, help="--viz 전용: 프레임당 micro-step 수(재생 속도)")
    ap.add_argument("--satellite", action="store_true",
                    help="--viz 전용: 체크포인트 지오 앵커(cfg.geo_lat/geo_lon) 주변 위성 배경(인터넷 필요, "
                         "실패 시 기존 해색 배경으로 자동 폴백). 실 로스백 좌표는 이 앵커와 무관한 로컬 "
                         "ENU 이므로 배경은 실제 위치가 아니라 체크포인트 학습 해역의 이미지다")
    ap.add_argument("--hide-cells", action="store_true",
                    help="--viz 전용, 셀선택 모델에서만 적용: 후보셀/유효셀/선택셀 오버레이를 숨기고 "
                         "대도·경로·그물만 표시(깔끔한 운용 화면)")
    args = ap.parse_args()

    if args.replan_period < 1:
        raise SystemExit("--replan-period 는 1 이상이어야 합니다.")

    from commander.bag_replay import BagEnemyReplay
    from commander.rl_bridge import build_battlefield_defense

    print(f"[replay] 로스백 로딩: {args.bag} (적 {args.n_tracks}척)")
    bag = BagEnemyReplay(
        args.bag, n_tracks=args.n_tracks, first_index=args.first_index,
        pose_topic_fmt=args.pose_topic_fmt, heading_topic_fmt=args.heading_topic_fmt,
    )
    print(f"[replay] 로스백 길이 {bag.duration_sec:.1f} s")

    kind = _detect_policy_kind(args.ckpt)
    print(f"[replay] 정책 로딩: {args.ckpt} (감지: {'셀선택' if kind == 'cell' else 'CNN 점수맵(U-Net)'})")
    common = dict(
        span_real=args.span, ally_speed_real=args.ally_speed_real, enemy_mode=args.enemy_mode,
        enu_origin=tuple(args.enu_origin) if args.enu_origin else None,
        net_reload_period_real=args.net_reload_period,
    )
    if kind == "cell":
        from commander.replay_cell_env import ReplayCellEnv
        env = ReplayCellEnv(args.ckpt, bag, specialized_root=args.specialized, **common)
    else:
        from commander.replay_cnn_env import ReplayCnnEnv
        if args.specialized:
            print("[replay] ⚠ --specialized 는 셀선택 모델 전용입니다 — CNN 점수맵 모델에서는 무시합니다.")
        env = ReplayCnnEnv(args.ckpt, bag, nets_per_ship=args.nets,
                            geo=tuple(args.geo) if args.geo else None, **common)
    print(f"[replay] enu_origin(맵 중앙 대응 실좌표) = {tuple(round(c, 3) for c in env.scale.enu_origin)}")
    print(f"[replay] 그물 재보급 주기 = {env.net_reload_period_real:.3g} 실초")
    if bag.n_tracks > env.M:
        print(f"[replay] ⚠ 로스백 척수({bag.n_tracks}) > 정책 적슬롯(M={env.M}) — 앞 {env.M}개만 사용합니다.")
    v_enemy_real = bag.mean_speed_mps()   # 실측 적 평균속력 — report() 의 ve/va 대조를 실제로 만든다
    print(env.scale.report(v_enemy_real=v_enemy_real))
    print(f"[replay] 실측 적 평균속력 ≈ {v_enemy_real:.4g} m/s (학습 가정 ve/va={env.cfg.enemy_speed_mult:g})")
    print(f"[replay] 결정주기 = {env.scale.period_real:.3g} 실초 (아군 {env.P}척, 적슬롯 {env.M})")

    bg_img = bg_extent = None
    if args.viz and args.satellite:
        from commander.satellite import fetch_satellite_bg
        print("[replay] 위성 배경 로딩 중...")
        res = fetch_satellite_bg(env.cfg.geo_lat, env.cfg.geo_lon, env.cfg.world_size)
        if res:
            bg_img, bg_extent = res
            print("[replay] 위성 배경 로드 완료")
        else:
            print("[replay] ⚠ 위성 배경 로드 실패(오프라인 등) — 기본 해색 배경 사용")

    commander = _make_commander(args.llm, args.model)

    log: list = []
    state = {"decision_idx": 0, "fleet_lost_warned": False}

    def replan_if_due() -> None:
        if state["decision_idx"] % args.replan_period == 0:
            bf = build_battlefield_defense(env, args.command)
            plan = commander.plan(bf)
            env.set_plan(plan, args.command)
            print(f"[t={env.bag_time_real:6.2f}s] 재배정: {plan.rationale}")

    def log_decision() -> None:
        allies = [_decode_route(env, p) for p in range(env.P)]
        rec = {
            "decision": state["decision_idx"],
            "t_bag_sec": round(env.bag_time_real, 3),
            "assign": env._assign[0].tolist(),
            "enemies_alive": int(env.e_alive[0].sum()),
            "allies": allies,
        }
        log.append(rec)
        summary = "  ".join(
            f"USV{p}:{'STOP' if a['stop'] else a['move_wp_m']}" for p, a in enumerate(allies)
        )
        print(f"[t={rec['t_bag_sec']:6.2f}s] decision={state['decision_idx']:03d} "
              f"적생존={rec['enemies_alive']}  {summary}")
        # 전멸은 "예비 대기(STOP)"와 겉보기로 구분이 안 된다 — 그대로 두면 실패한 재생이
        # 조용히 끝까지(exit 0) 돌아 로그만 봐서는 정상 종료와 구별되지 않는다.
        if not bool(env.a_alive[0].any()) and not state["fleet_lost_warned"]:
            print(f"[replay] ⚠ t={rec['t_bag_sec']:.2f}s 시점 아군 전멸(a_alive 전원 False) — "
                  f"남은 재생은 죽은 함대로 진행됩니다.")
            state["fleet_lost_warned"] = True
        state["decision_idx"] += 1

    def advance_one_micro() -> bool:
        """micro-step 1회 진행 — 결정 경계(§decision_period)마다 재배정 호출·결과 로그.

        `env._micro_ct % decision_period == 0` 는 두 자리에서 확인한다: env.step() 호출
        *전*이면 "이번 스텝에서 새 결정이 시작된다"(재배정을 결정 시작 전에 주입해야 한다),
        호출 *후*면 "직전 결정의 batch 가 방금 끝났다"(로그를 여기서 남긴다) 는 뜻이다.
        """
        if env.bag_exhausted():
            return False
        if env._micro_ct % env.cfg.decision_period == 0:
            replan_if_due()
        env.step()
        if env._micro_ct % env.cfg.decision_period == 0:
            log_decision()
        return True

    def max_reached() -> bool:
        return args.max_decisions is not None and state["decision_idx"] >= args.max_decisions

    if args.viz:
        _run_viz(env, advance_one_micro, max_reached, args.spf, log,
                 bg_img=bg_img, bg_extent=bg_extent, hide_cells=args.hide_cells)
    else:
        while not max_reached():
            if not advance_one_micro():
                break

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[replay] 결정 로그 저장: {args.out} ({len(log)}개 결정)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
