#!/usr/bin/env python3
"""run_sim_bridge.py — Unreal 시뮬레이터(ROS2 Odometry, 로컬 m 프레임)에서 u-net_map.pt 추론.

입력은 `/usv/{friendly_0X,enemy_0X,mothership}/odometry`(x/y m, 모선=(0,0)) + `/usv/sim_state`,
출력은 `/usv/waypoints`(x_m/y_m 경유점 JSON) + `/usv/net_trigger`(구간당 1회). 섬은 없다
(land 채널 전부 0). 기본은 dry-run — 명령을 발행하지 않고 콘솔/로그에만 남긴다.

오프라인(로스백 재생) — 아군도 녹화 궤적 그대로 움직이는 '그림자 추론':
    ros2 bag play /home/yune/rosbag/usv_20261004_194054
    python3.10 run_sim_bridge.py --ckpt boatattack_sim/models/u-net_map.pt --llm heuristic --viz

라이브 시뮬레이터 — 실제로 명령 발행(기존 /usv/waypoints 컨트롤러는 꺼둘 것):
    python3.10 run_sim_bridge.py --ckpt boatattack_sim/models/u-net_map.pt --llm heuristic --publish --viz

--span auto(기본): 첫 웨이브 적의 모선 거리 최대값을 학습 스폰반경(enemy_spawn_radius)에
맞춘다(span = R·world_size/enemy_spawn_radius). 숫자를 주면 그 실미터를 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    ap.add_argument("--span", default="auto",
                    help="모델 맵 전체 폭에 대응하는 실미터. 'auto'(기본) = 적 출현거리에 맞춤")
    ap.add_argument("--ally-ids", default="friendly_01,friendly_02,friendly_03")
    ap.add_argument("--enemy-ids", default=",".join(f"enemy_{i:02d}" for i in range(1, 8)))
    ap.add_argument("--mothership-id", default="mothership",
                    help="이 배의 Odometry 를 맵 중앙(enu_origin)으로 쓴다. 빈 문자열이면 (0,0)")
    ap.add_argument("--topic-fmt", default="/usv/{id}/odometry")
    ap.add_argument("--publish", action="store_true",
                    help="/usv/waypoints, /usv/net_trigger 를 실제로 발행(기본: dry-run)")
    ap.add_argument("--cruise-kn", type=float, default=40.0)
    ap.add_argument("--accept-radius-m", type=float, default=4.0)
    ap.add_argument("--ally-speed-real", type=float, default=18.0,
                    help="아군 실속력 m/s(로스백 실측 ≈18) — 결정주기 실초 환산에 쓰인다")
    ap.add_argument("--nets", type=int, default=3, help="배당 그물 수(시뮬레이터 = 3)")
    ap.add_argument("--net-reload-period", type=float, default=1e9,
                    help="그물 재보급 주기(실초). 기본 = 사실상 재보급 없음(시뮬레이터 규칙)")
    ap.add_argument("--publish-hz", type=float, default=2.0)
    ap.add_argument("--stale-timeout", type=float, default=1.0)
    ap.add_argument("--wait-timeout", type=float, default=300.0,
                    help="아군/적 첫 수신 대기 한도(초)")
    ap.add_argument("--quorum-timeout", type=float, default=5.0)
    ap.add_argument("--llm", default="heuristic", choices=["ollama", "openai", "gemini", "heuristic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--command", default=None)
    ap.add_argument("--replan-period", type=int, default=8)
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--out", default=None, help="결정 로그 JSON 저장 경로")
    ap.add_argument("--cmd-log", default=None, help="발행(또는 dry-run) 명령 JSONL 경로")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="micro-step 을 실시간 dt 로 페이싱하지 않는다(테스트용)")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--spf", type=int, default=3)
    ap.add_argument("--pause-start", action="store_true")
    args = ap.parse_args()

    from commander.rl_bridge import build_battlefield_defense
    from commander.sim_odom_ros2 import SimOdomLink, SimWaypointSink, SimNetTriggerSink
    from commander.sim_odom_env import SimOdomCnnEnv, auto_span
    from boatattack_sim.model.cnn_actor import load_cnn_actor
    from run_replay_infer import _make_commander

    ally_ids = [v.strip() for v in args.ally_ids.split(",") if v.strip()]
    enemy_ids = [v.strip() for v in args.enemy_ids.split(",") if v.strip()]
    link = SimOdomLink(ally_ids, enemy_ids, mothership_id=args.mothership_id or None,
                       topic_fmt=args.topic_fmt, enable_publish=args.publish,
                       stale_timeout=args.stale_timeout)
    print(f"[sim_bridge] 구독 시작: 아군 {ally_ids}, 적 {enemy_ids}, 모선 {args.mothership_id or '(없음)'}"
          f" — {'발행 모드(--publish)' if args.publish else 'DRY-RUN(발행 안 함)'}")

    try:
        print("[sim_bridge] 아군 Odometry 대기 중...")
        if not link.wait_allies(args.wait_timeout):
            missing = [v for v, a in zip(ally_ids, link.ally_view.ally_snapshot().alive) if not a]
            raise SystemExit(f"아군 Odometry 미수신: {missing} — 토픽/재생 상태를 확인하세요.")
        origin = (0.0, 0.0)
        if args.mothership_id:
            ms = link.mothership_xy()
            if ms is None:
                print(f"[sim_bridge] ⚠ {args.mothership_id} Odometry 미수신 — enu_origin=(0,0) 사용")
            else:
                origin = ms
        print(f"[sim_bridge] enu_origin(맵 중앙=모선) = ({origin[0]:.2f}, {origin[1]:.2f}) m")

        if str(args.span).lower() == "auto":
            _, cfg = load_cnn_actor(args.ckpt)
            print("[sim_bridge] --span auto: 적 출현 대기 중...")
            r = link.wait_enemy_range(origin, args.wait_timeout)
            if r <= 0:
                raise SystemExit("적 Odometry 가 대기 한도 안에 안 보였습니다 — --span 을 직접 지정하세요.")
            span = auto_span(cfg, r)
            print(f"[sim_bridge] span auto = {span:.0f} m  (적 최대거리 {r:.0f} m × world_size "
                  f"{cfg.world_size:g} / enemy_spawn_radius {cfg.enemy_spawn_radius:g})")
        else:
            span = float(args.span)

        sim_id_fn = lambda: link.state.sim_id
        wp_sink = SimWaypointSink(link.publish_waypoints_fn, cruise_kn=args.cruise_kn,
                                  accept_radius_m=args.accept_radius_m, log_path=args.cmd_log)
        net_sink = SimNetTriggerSink(link.publish_net_fn, sim_id_fn=sim_id_fn, log_path=args.cmd_log)
        env = SimOdomCnnEnv(
            args.ckpt, span, link.ally_view, link.enemy_view,
            waypoint_sink=wp_sink, net_sink=net_sink, sim_id_fn=sim_id_fn,
            publish_hz=args.publish_hz, ally_speed_real=args.ally_speed_real,
            nets_per_ship=args.nets, enu_origin=origin,
            net_reload_period_real=args.net_reload_period, quorum_timeout=args.quorum_timeout,
        )
        S = env.scale.S
        print(f"[sim_bridge] S = {S:.3g} sim-m/m, 결정주기 = {env.scale.period_real:.3g} s, "
              f"micro-step = {env.scale.dt_real:.3g} s, land = {int(env.land_map.sum())} 셀(섬 없음)")
        print(f"[sim_bridge] 실미터 환산: arrive_radius {env.cfg.arrive_radius / S:.0f} m, "
              f"그물 최대길이 {env.cfg.net_max_len / S:.0f} m, 요격환형 "
              f"{env.cfg.cell_r_min / S:.0f}~{env.cfg.cell_r_max / S:.0f} m")
        print(env.scale.report())

        commander = _make_commander(args.llm, args.model)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="replan")
        pending: dict = {"future": None}
        log: list = []
        state = {"decision_idx": 0, "seen_ready": False}
        info: dict = {"model": getattr(commander, "model", args.llm), "status": "아군 수신 대기",
                      "cmd": args.command or "(none)", "assign": "(아직 없음)",
                      "rationale": "첫 재배정 대기 중..."}

        def replan_if_due(force: bool = False) -> None:
            if pending["future"] is not None:
                return
            if force or state["decision_idx"] % args.replan_period == 0:
                pending["future"] = executor.submit(commander.plan,
                                                    build_battlefield_defense(env, args.command))

        def apply_replan_if_ready() -> None:
            fut = pending["future"]
            if fut is None or not fut.done():
                return
            pending["future"] = None
            try:
                plan = fut.result()
            except Exception as exc:
                print(f"[sim_bridge] 재배정 실패, 이전 배정 유지: {exc}")
                return
            env.set_plan(plan, args.command)
            info["status"] = f"재배정 적용 (t={env.bag_time_real:.1f}s, phase={link.state.phase})"
            info["assign"] = "  ".join(f"C{d.cluster_id}:{d.ally_ids or 'none'}"
                                       for d in plan.deployments) or "(none)"
            info["rationale"] = plan.rationale
            print(f"[t={env.bag_time_real:6.2f}s] 재배정: {plan.rationale}")

        def log_decision() -> None:
            rec = {
                "decision": state["decision_idx"],
                "t_sec": round(env.bag_time_real, 3),
                "phase": link.state.phase,
                "assign": env._assign[0].tolist(),
                "enemies_alive": int(env.e_alive[0].sum()),
                "nets_left": env.a_nets[0].tolist(),
                "waypoints_m": {vid: [list(map(lambda c: round(c, 2), q)) for q in env.waypoints_xy(p)]
                                for p, vid in enumerate(ally_ids)},
                "external_assignment": link.state.last_assignment,
            }
            log.append(rec)
            wps = "  ".join(f"{vid}:{rec['waypoints_m'][vid] or 'STOP'}" for vid in ally_ids)
            print(f"[t={rec['t_sec']:6.2f}s] decision={rec['decision']:03d} phase={rec['phase']} "
                  f"적생존={rec['enemies_alive']} 배정={rec['assign']}  {wps}")
            if rec["external_assignment"]:
                print(f"           (외부 /usv/assignment: {rec['external_assignment']})")
            state["decision_idx"] += 1

        def advance_one_micro() -> bool:
            t0 = time.monotonic()
            apply_replan_if_ready()
            before = env._micro_ct
            env.step()
            ticked = env._micro_ct != before
            if env.ready and ticked:
                if not state["seen_ready"]:
                    state["seen_ready"] = True
                    replan_if_due(force=True)
                elif env._micro_ct % env.cfg.decision_period == 0:
                    log_decision()
                    replan_if_due()
            if args.realtime:
                rem = env.scale.dt_real - (time.monotonic() - t0)
                if rem > 0:
                    time.sleep(rem)
            return True

        def max_reached() -> bool:
            return args.max_decisions is not None and state["decision_idx"] >= args.max_decisions

        try:
            if args.viz:
                from run_replay_infer import _run_viz
                _run_viz(env, advance_one_micro, max_reached, args.spf, log,
                         info_provider=lambda: info, start_paused=args.pause_start,
                         hide_unassigned_wps=True)
            else:
                while not max_reached():
                    advance_one_micro()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)
                print(f"[sim_bridge] 결정 로그 저장: {args.out} ({len(log)}개 결정)")
    finally:
        link.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
