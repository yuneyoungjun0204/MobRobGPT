"""적선(로스백 재생/라이브 ROS2 토픽/GCS `/api/state`) + GCS 실텔레메트리(실제 아군)로
구동하는 CNN 점수맵(U-Net) 정책 — 아군 경유점을 실제 GCS(`/home/yune/gcs`)의 명령 계약으로
내보내는 운용 진입점.

`run_replay_infer.py`의 자매 스크립트다 -- 그 파일은 적(로스백)만 실데이터이고 아군은
시뮬 물리로 가상 기동시키는 **오프라인 분석/시각화 도구**라 여기서는 건드리지 않는다
(동시에 다른 세션이 그 파일을 활발히 고치고 있기도 하다). 이 스크립트는 반대로 아군도
실제(GCS가 보는 실보트)로 만들고, 시각화는 선택 사항이다.

적선 데이터 출처 3가지 (`--enemy-source`, 아래 인자 설명 참고):
  bag  로스백(.db3)을 이 스크립트가 독립적으로 재생 (`commander/bag_replay.py`)
  live raw ROS2 토픽 `/usv/usv{i}/pose`를 직접 구독 (`commander/live_enemy_ros2.py`) --
       그 토픽이 GCS datum 기준 프레임이 아니면 아군과 좌표계가 어긋난다(2026-09-10
       세션 실측 확인, `commander/gcs_cnn_env.py` 모듈독스트링 참고)
  gcs  GCS `/api/state`에서 role=target 배들의 위치를 아군과 같은 방식으로 읽는다
       (`commander/gcs_cnn_env.py::GcsLiveCnnEnv`) -- GCS가 이미 재투영해 둔 값이라
       좌표계 문제가 구조적으로 없다. 재투영이 필요한 데모(예: bag_enemy_relay.py
       --source pose)에는 이쪽을 쓸 것.

GCS 쪽 요구사항 (docs/contracts.md, server/api.py):
  - GCS 서버가 `--ally-ids`에 준 각 vehicle_id로 이미 vehicles.yaml에 등록돼 있고,
    role이 "defender"여야 한다(command/authority.py) -- 아니면 매 명령이 role_mismatch로
    거부된다(이 스크립트는 거부를 그대로 로그만 남기고 계속 돈다).
  - `--enemy-source gcs`는 추가로 `--target-ids`의 각 vehicle_id가 role="target"으로
    등록돼 있어야 한다(등록만 돼 있으면 됨 -- 이 스크립트는 이 배들에게 명령을 내리지
    않는다).
  - GCS 서버가 `--gcs-url`(기본 http://127.0.0.1:8080)에서 HTTP API를 서빙 중이어야 한다.
  - "그물 뿌리기"는 GCS로 전혀 전송되지 않는다 -- `--net-log`로 지정한 파일(또는 stdout)
    로만 신호가 나간다. 실제 그물 전개는 그 신호를 구독하는 별도 시스템의 몫이다.

실행 예:
    python run_gcs_bridge.py \\
        --bag /home/yune/Downloads/ros_data/S03-gcs --span 8 \\
        --ckpt boatattack_sim/models/u-net_map.pt --llm openai \\
        --gcs-url http://127.0.0.1:8080 --ally-ids usv1,usv2,usv3 --viz

    python run_gcs_bridge.py --enemy-source gcs --target-ids usv4,usv5 --span 80 \\
        --gcs-url http://127.0.0.1:8091 --ally-ids usv1,usv2,usv3 --llm heuristic
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time

import numpy as np


def _load_ckpt_config(ckpt: str):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    if cfg is None:
        raise ValueError(f"{ckpt}: 체크포인트에 'config' 가 없습니다.")
    get = cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d)
    return get


def _detect_policy_kind(ckpt: str) -> str:
    get = _load_ckpt_config(ckpt)
    if bool(get("cnn_action", False)):
        return "cnn"
    raise ValueError(
        f"{ckpt}: cnn_action 플래그가 없습니다. run_gcs_bridge.py는 CNN 점수맵(U-Net) "
        f"정책만 지원합니다(셀선택 정책은 run_replay_infer.py를 쓰세요).")


def _policy_n_allies(ckpt: str) -> int:
    return int(_load_ckpt_config(ckpt)("n_allies"))


def _policy_n_enemies(ckpt: str) -> int:
    return int(_load_ckpt_config(ckpt)("n_enemies"))


def _make_commander(backend: str, model: str | None):
    if backend == "heuristic":
        from commander.fallback import heuristic_plan

        class _Heuristic:
            def plan(self, state):
                return heuristic_plan(state)

        return _Heuristic()
    from commander import make_commander
    return make_commander(backend, model)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="실제 적선(로스백/라이브 ROS2/GCS) + GCS 실텔레메트리 아군으로 CNN 점수맵 정책을 운용",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--bag", default=None, help="rosbag2 디렉터리 경로 (metadata.yaml 포함, "
                    "--enemy-source bag 일 때 필수)")
    ap.add_argument("--enemy-source", default="bag", choices=["bag", "live", "gcs"],
                    help="적선 데이터 출처. bag=--bag 의 .db3 를 이 스크립트가 독립적으로 "
                         "재생(오프라인, 매 실행마다 t=0부터). live=commander/live_enemy_ros2.py로 "
                         "--pose-topic-fmt/--heading-topic-fmt 를 실시간 구독(예: tests/simulation/"
                         "mock/run_mixed_demo.sh 처럼 GCS 지도에 이미 ros2 bag play -l 로 같은 "
                         "로스백이 라이브 중계되고 있을 때). gcs=--target-ids 로 준 GCS "
                         "vehicle_id(role=target)들의 위치를 아군과 똑같이 GCS `/api/state`에서 "
                         "읽는다(rclpy 불필요). ⚠ live 는 raw ROS2 토픽(로스백 자기 자신의 녹화 "
                         "로컬 프레임)을 그대로 쓰므로, 그 프레임이 GCS datum 기준과 다르면(예: "
                         "bag_enemy_relay.py --source pose 로 재투영해 중계하는 데모) 아군과 "
                         "다른 좌표계가 섞여 적이 지도 밖으로 클리핑된다(2026-09-10 세션 실측 "
                         "확인) -- 그 경우 live 대신 gcs 를 쓸 것. gcs 는 GCS 가 이미 재투영해 "
                         "둔 값을 그대로 읽으므로 이 문제가 없다.")
    ap.add_argument("--span", type=float, required=True,
                    help="실제 운용 박스 한 변[m]")
    ap.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt")
    ap.add_argument("--n-tracks", type=int, default=5, help="로스백 안 적선 척수")
    ap.add_argument("--first-index", type=int, default=1, help="적선 토픽 번호 시작값 (usv{N})")
    ap.add_argument("--pose-topic-fmt", default="/usv/usv{i}/pose")
    ap.add_argument("--heading-topic-fmt", default="/usv/usv{i}/heading")
    ap.add_argument("--ally-speed-real", type=float, default=0.3,
                    help="아군의 실제 순항속도[m/s] -- SimScale 시계 스케일 산출용")
    ap.add_argument("--nets", type=int, default=3, help="배당 그물 장수")
    ap.add_argument("--net-reload-period", type=float, default=None,
                    help="그물 소진 후 재보급 주기[실제 초] (기본: 자동 산출)")
    ap.add_argument("--enemy-mode", default="wave",
                    help="초기 스폰용(bag/live 는 매 tick 로스백/ROS2 로 덮어씀, gcs 는 "
                         "GCS `/api/state` 로 덮어씀)")
    ap.add_argument("--llm", default="ollama", choices=["ollama", "openai", "heuristic"])
    ap.add_argument("--model", default=None, help="LLM 모델명 (미지정 시 백엔드 기본값)")
    ap.add_argument("--command", default=None, help="지휘관에게 줄 자연어 지시(선택)")
    ap.add_argument("--replan-period", type=int, default=8,
                    help="지휘관을 다시 부르는 결정 간격")
    ap.add_argument("--max-decisions", type=int, default=None, help="처리할 결정 횟수 상한")
    ap.add_argument("--geo", type=float, nargs=2, default=None, metavar=("LAT", "LON"))
    ap.add_argument("--enu-origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="맵 중앙(=모선)에 대응하는 실좌표(datum 기준 동/북 m). 미지정시 GCS가 "
                         "지금 보고하는 --ally-ids 의 실위치 평균을 자동으로 쓴다 -- gcs.datum "
                         "자체를 (0,0)으로 가정하지 않는다: '배치모드'(demo_relocate_service.py) "
                         "등으로 함대를 재배치해도 registry의 datum은 절대 움직이지 않으므로, "
                         "datum=(0,0) 가정은 재배치 후 함대가 datum에서 수 km 떨어져 있을 때 "
                         "그 오프셋만큼 아군·적을 전부 지도 밖으로 클리핑시킨다(2026-09-10 세션 "
                         "실측 확인).")
    ap.add_argument("--out", default=None, help="결정별 명령 로그를 저장할 JSON 경로(선택)")

    ap.add_argument("--gcs-url", default="http://127.0.0.1:8080",
                    help="GCS 서버 base URL")
    ap.add_argument("--ally-ids", required=True,
                    help="쉼표구분 GCS vehicle_id 목록, 체크포인트의 P와 개수가 같아야 함 "
                         "(예: usv1,usv2,usv3). vehicles.yaml에 role=defender로 등록돼 있어야 함")
    ap.add_argument("--target-ids", default=None,
                    help="--enemy-source gcs 일 때 필수. 쉼표구분 GCS vehicle_id 목록(예: "
                         "usv4,usv5) -- 체크포인트의 M(적 슬롯 수)보다 많으면 안 됨, 적으면 "
                         "나머지 슬롯은 계속 e_alive=False. vehicles.yaml에 role=target으로 "
                         "등록돼 있어야 함(등록만 돼 있으면 됨 -- 이 스크립트는 이 배들에게 "
                         "명령을 내리지 않는다, 위치만 읽는다).")
    ap.add_argument("--publish-hz", type=float, default=2.0,
                    help="GCS로 경유점을 재전송하는 빈도(docs/contracts.md §4: 0.5~2Hz 권장)")
    ap.add_argument("--net-log", default=None,
                    help="그물 전개 시작/종료 신호를 JSON-lines로 남길 파일 경로(선택)")
    ap.add_argument("--net-ros2-namespace", default="mobrobgpt",
                    help="그물 전개 신호를 /<ns>/<vehicle_id>/net_deploy(std_msgs/Int32, "
                         "0=대기/1=전개중)로 발행할 네임스페이스. rclpy 없으면 자동으로 생략됨.")
    ap.add_argument("--realtime", dest="realtime", action="store_true", default=True,
                    help="micro-step 사이를 실제 SimScale.dt_real 만큼 대기(기본 켜짐 -- 실보트 "
                         "연동이므로 run_replay_infer.py와 달리 CPU 속도로 폭주하면 안 된다)")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="대기 없이 최대 속도로 진행(스텁 GCS로 테스트할 때만 사용). "
                         "publish 스로틀은 항상 실제 시계(wall clock) 기준이다 -- 이 모드에서도 "
                         "일부러 바꾸지 않는다(실제 네트워크 엔드포인트와 docs/contracts.md §4의 "
                         "max_age 관계를 지키는 스로틀이라, 모드별로 다르게 굴면 이 플래그가 "
                         "정작 실운용에서 쓰는 발행 경로를 검증하지 못하게 된다). 그 결과 짧게 "
                         "돌리면 실제 몇 초가 지나야 나갈 goto 가 안 나가 배가 못 움직인 것처럼 "
                         "보일 수 있다 -- 스모크 테스트는 `--no-realtime --publish-hz 1000` "
                         "조합으로 돌릴 것(commander/test_gcs_bridge.py 의 환경 테스트가 쓰는 "
                         "방식과 동일).")

    ap.add_argument("--viz", action="store_true", help="matplotlib 실시간 시각화(run_replay_infer.py 재사용)")
    ap.add_argument("--spf", type=int, default=3, help="--viz 전용: 프레임당 micro-step 수")
    ap.add_argument("--satellite", action="store_true",
                    help="--viz 전용: 실제 지도 위치의 위성 배경(인터넷 필요, 실패 시 기본 배경으로 "
                         "자동 폴백). run_replay_infer.py의 --satellite와 달리 체크포인트의 학습 "
                         "지오 앵커가 아니라 --geo(지정 시) 또는 GCS가 지금 보고하는 --ally-ids의 "
                         "실제 lat/lon 평균을 앵커로 쓴다 -- 이 맵 중앙(enu_origin)이 실제로 어디인지 "
                         "GCS 쪽 값으로 구하므로, 함대가 어디로 재배치돼 있든 그 실제 지점의 위성사진이 "
                         "뜬다.")
    args = ap.parse_args()

    if args.replan_period < 1:
        raise SystemExit("--replan-period 는 1 이상이어야 합니다.")
    ally_ids = [v.strip() for v in args.ally_ids.split(",") if v.strip()]
    if not ally_ids:
        raise SystemExit("--ally-ids 는 최소 1개 이상의 vehicle_id 를 포함해야 합니다.")

    from commander.rl_bridge import build_battlefield_defense
    from commander.gcs_bridge import GcsClient, GcsAllyLink, NetDeploySink

    _detect_policy_kind(args.ckpt)          # raises if this isn't a CNN-score-map checkpoint
    print(f"[gcs_bridge] 정책 로딩: {args.ckpt} (감지: CNN 점수맵(U-Net))")
    n_allies = _policy_n_allies(args.ckpt)
    if len(ally_ids) != n_allies:
        raise SystemExit(
            f"--ally-ids 개수({len(ally_ids)}: {ally_ids})가 체크포인트의 아군 수"
            f"(n_allies={n_allies})와 다릅니다.")

    client = GcsClient(args.gcs_url)
    # source is always "rl" -- this script IS the RL autonomy module, never the operator.
    # A configurable value here would let it request "manual"'s higher command priority
    # (command/authority.py), which is an authority boundary gcs otherwise keeps sharp.
    ally_link = GcsAllyLink(client, ally_ids, source="rl")

    if args.enu_origin is not None:
        enu_origin = tuple(args.enu_origin)
    else:
        # ★ 2026-09-10 세션 실측으로 뒤집힌 가정: "gcs.datum=(0,0)이 곧 모선 위치"는
        # `demo_relocate_service.py`("배치모드")가 "registry의 datum 자체는 절대 옮기지
        # 않는다"고 명시적으로 설계한 것과 정면으로 어긋난다 -- 배치모드로 함대를 재배치하면
        # datum은 그대로인데 함대만 수 km 떨어진 곳으로 옮겨가고, 그 상태에서 enu_origin=
        # (0,0)을 쓰면 그 오프셋 전체가 좌표 오차가 되어 --span(수십~수백 m) 박스 밖으로
        # 아군·적이 전부 클리핑된다(실측: /api/state 의 usv1 ned=(3415,4900) m인데 가정은
        # (0,0)). datum은 "GCS 좌표계 정의"일 뿐 "함대 현재 위치"가 아니므로, 대신 GCS가
        # 지금 보고하는 아군 실위치의 평균을 쓴다 -- ReplayCnnEnv가 원래 하던
        # bag.centroid() 근사와 같은 발상이고("모선 위치를 모르면 아는 것들의 평균으로
        # 근사"), GCS 연동에서는 로스백 적선보다 아군 쪽이 훨씬 신뢰할 수 있는 소스다.
        try:
            origin_snap = ally_link.ally_snapshot()
        except Exception as exc:
            raise SystemExit(
                f"--enu-origin 자동산출 실패 -- GCS({args.gcs_url})에서 아군 위치를 못 "
                f"읽었습니다: {exc}. GCS가 떠 있는지 확인하거나 --enu-origin X Y 를 "
                f"직접 지정하세요.")
        if not origin_snap.alive.any():
            raise SystemExit(
                f"--enu-origin 자동산출 실패 -- --ally-ids {ally_ids} 중 GCS가 지금 "
                f"살아있다고 보고하는 배가 없습니다. GCS가 이 배들을 이미 보고 있는지 "
                f"확인하거나 --enu-origin X Y 를 직접 지정하세요.")
        centroid = origin_snap.pos[origin_snap.alive].mean(axis=0)
        enu_origin = (float(centroid[0]), float(centroid[1]))
        offset_from_datum = float(np.hypot(*centroid))
        print(f"[gcs_bridge] --enu-origin 미지정 -- 아군 {int(origin_snap.alive.sum())}/"
              f"{len(ally_ids)}척의 현재 GCS 위치 평균 "
              f"{tuple(round(c, 3) for c in enu_origin)}(datum 기준 동/북 m, "
              f"datum에서 {offset_from_datum:.1f} m)을 맵 중앙(=모선)으로 사용합니다.")
        if offset_from_datum > 50.0:
            print(f"[gcs_bridge] ⚠ 함대가 gcs.datum에서 {offset_from_datum:.0f} m 떨어져 "
                  f"있습니다 -- '배치모드' 등으로 재배치됐을 가능성이 높습니다(datum 자체는 "
                  f"안 움직이는 게 정상 -- demo_relocate_service.py 설계). 위 자동산출 "
                  f"원점을 그대로 쓰면 됩니다, 이 경고는 참고용입니다.")

    bag = None
    enemy_link = None
    if args.enemy_source == "gcs":
        if not args.target_ids:
            raise SystemExit("--enemy-source gcs 는 --target-ids 가 필요합니다.")
        target_ids = [v.strip() for v in args.target_ids.split(",") if v.strip()]
        if not target_ids:
            raise SystemExit("--target-ids 는 최소 1개 이상의 vehicle_id 를 포함해야 합니다.")
        overlap = set(target_ids) & set(ally_ids)
        if overlap:
            raise SystemExit(
                f"--target-ids 와 --ally-ids 가 겹칩니다({sorted(overlap)}) -- 같은 배를 "
                f"아군으로 명령하면서 동시에 적으로 읽으면 관측이 자기 자신을 적으로 "
                f"본다.")
        n_enemies = _policy_n_enemies(args.ckpt)
        if len(target_ids) > n_enemies:
            raise SystemExit(
                f"--target-ids 개수({len(target_ids)}: {target_ids})가 체크포인트의 적 "
                f"슬롯 수(n_enemies={n_enemies})보다 많습니다.")
        # source="rl" 로 만들지만 submit_goto 는 절대 호출하지 않는다 -- 읽기 전용
        # (GcsLiveCnnEnv._ingest_gcs 는 ally_snapshot() 만 부른다).
        enemy_link = GcsAllyLink(client, target_ids, source="rl")
        print(f"[gcs_bridge] GCS `/api/state` 적선 읽기: {target_ids}")
    elif args.enemy_source == "live":
        from commander.live_enemy_ros2 import LiveEnemyReplay
        print("[gcs_bridge] ⚠ --enemy-source live 는 raw ROS2 토픽을 GCS 재투영 없이 "
              "그대로 씁니다 -- 그 토픽이 GCS datum 기준 프레임이 아니면(예: "
              "bag_enemy_relay.py --source pose 로 재투영해 중계하는 데모) 적이 아군과 "
              "다른 좌표계로 섞여 지도 밖에 클리핑됩니다. 그런 데모에서는 --enemy-source "
              "gcs 를 쓰세요.")
        print(f"[gcs_bridge] 라이브 ROS2 적선 구독: {args.n_tracks}척 "
              f"(pose={args.pose_topic_fmt}, heading={args.heading_topic_fmt})")
        bag = LiveEnemyReplay(
            n_tracks=args.n_tracks, first_index=args.first_index,
            pose_topic_fmt=args.pose_topic_fmt, heading_topic_fmt=args.heading_topic_fmt,
        )
        print(f"[gcs_bridge] 전 트랙({bag.n_tracks}척) pose+heading 첫 수신 완료")
    else:
        if not args.bag:
            raise SystemExit("--enemy-source bag 는 --bag 이 필요합니다.")
        from commander.bag_replay import BagEnemyReplay
        print(f"[gcs_bridge] 로스백 로딩: {args.bag} (적 {args.n_tracks}척)")
        bag = BagEnemyReplay(
            args.bag, n_tracks=args.n_tracks, first_index=args.first_index,
            pose_topic_fmt=args.pose_topic_fmt, heading_topic_fmt=args.heading_topic_fmt,
        )
        print(f"[gcs_bridge] 로스백 길이 {bag.duration_sec:.1f} s")

    net_sink = NetDeploySink(log_path=args.net_log, ros2_vehicle_ids=ally_ids,
                              ros2_namespace=args.net_ros2_namespace)

    if args.enemy_source == "gcs":
        from commander.gcs_cnn_env import GcsLiveCnnEnv
        env = GcsLiveCnnEnv(
            args.ckpt, args.span, ally_link, enemy_link,
            net_sink=net_sink, publish_hz=args.publish_hz,
            ally_speed_real=args.ally_speed_real, enemy_mode=args.enemy_mode,
            nets_per_ship=args.nets,
            geo=tuple(args.geo) if args.geo else None,
            enu_origin=enu_origin,
            net_reload_period_real=args.net_reload_period,
        )
    else:
        from commander.gcs_cnn_env import GcsBagCnnEnv
        env = GcsBagCnnEnv(
            args.ckpt, bag, args.span, ally_link,
            net_sink=net_sink, publish_hz=args.publish_hz,
            ally_speed_real=args.ally_speed_real, enemy_mode=args.enemy_mode,
            nets_per_ship=args.nets,
            geo=tuple(args.geo) if args.geo else None,
            enu_origin=enu_origin,
            net_reload_period_real=args.net_reload_period,
        )
    print(f"[gcs_bridge] GCS = {args.gcs_url}  ally_ids = {ally_ids}")
    print(f"[gcs_bridge] enu_origin(맵 중앙 대응 실좌표) = "
          f"{tuple(round(c, 3) for c in env.scale.enu_origin)}")
    print(f"[gcs_bridge] 결정주기 = {env.scale.period_real:.3g} 실초, "
          f"micro-step = {env.scale.dt_real:.3g} 실초")
    print(env.scale.report(v_enemy_real=bag.mean_speed_mps() if bag is not None else None))
    arrive_radius_real = env.cfg.arrive_radius / env.scale.S
    net_max_len_real = env.cfg.net_max_len / env.scale.S
    print(f"[gcs_bridge] arrive_radius = {arrive_radius_real:.3g} m, "
          f"net_max_len = {net_max_len_real:.3g} m (실좌표, 이 --span 기준)")
    if arrive_radius_real < 1.0:
        print(f"[gcs_bridge] ⚠ arrive_radius가 {arrive_radius_real:.3g} m로 실보트 GPS "
              f"정밀도보다 작을 수 있습니다 -- 영원히 '도착' 판정을 못 받아 ptr이 전진하지 "
              f"않고 그물도 시작하지 않을 위험이 있습니다. --span을 키우세요.")

    commander = _make_commander(args.llm, args.model)
    # LLM 호출(수백 ms~수 초)을 tick 스레드 밖으로 뺀다 -- 안 그러면 재배정마다 그 시간
    # 동안 env.step()이 멈춰 GCS로 아무 goto도 나가지 않고, gcs의 hold(3s)/fade(10s)를
    # 매번 잠식한다(docs/contracts.md §4, 아키텍트 검토 B3). 대기 중엔 직전 배정으로
    # 계속 전진·발행하다가, 응답이 오면 다음 tick에 반영한다.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-replan")
    pending: dict = {"future": None}
    log: list = []
    state = {"decision_idx": 0, "last_wait_print": 0.0, "seen_ready": False}
    # --viz 정보패널용(run_commander_ui.py와 같은 키 구조) -- --viz 없이 돌 때도 그냥 dict
    # 갱신일 뿐이라 추가 비용이 없다(그리기는 _run_viz의 info_provider 콜백이 호출될 때만).
    info: dict = {
        "model": getattr(commander, "model", args.llm),
        "status": "GCS 텔레메트리 대기 중",
        "cmd": args.command or "(none)",
        "assign": "(아직 없음)",
        "rationale": "첫 재배정 대기 중...",
    }

    def replan_if_due(force: bool = False) -> None:
        if pending["future"] is not None:
            return                      # 이전 재배정이 아직 응답 대기 중 -- 쌓아 보내지 않는다
        if force or state["decision_idx"] % args.replan_period == 0:
            bf = build_battlefield_defense(env, args.command)
            pending["future"] = executor.submit(commander.plan, bf)

    def apply_replan_if_ready() -> None:
        fut = pending["future"]
        if fut is None or not fut.done():
            return
        pending["future"] = None
        try:
            plan = fut.result()
        except Exception as exc:
            print(f"[gcs_bridge] LLM 재배정 실패, 이전 배정 유지: {exc}")
            info["status"] = f"Error: {type(exc).__name__}: {exc}"
            return
        env.set_plan(plan, args.command)
        held = sorted(getattr(plan, "hold_ships", None) or [])
        alloc = "  ".join(f"C{d.cluster_id}:{d.ally_ids or 'none'}"
                          for d in plan.deployments) or "(none)"
        info["status"] = f"재배정 적용됨 (t={env.bag_time_real:.1f}s)"
        info["cmd"] = args.command or "(none)"
        info["assign"] = alloc + (f"\nHold: {held}" if held else "")
        info["rationale"] = plan.rationale
        print(f"[t={env.bag_time_real:6.2f}s] 재배정: {plan.rationale}")

    def log_decision() -> None:
        rec = {"decision": state["decision_idx"], "t_bag_sec": round(env.bag_time_real, 3),
               "assign": env._assign[0].tolist(), "enemies_alive": int(env.e_alive[0].sum())}
        log.append(rec)
        state["decision_idx"] += 1

    def advance_one_micro() -> bool:
        if env.bag_exhausted():
            return False
        t_before = time.monotonic()
        apply_replan_if_ready()
        before_ct = env._micro_ct
        env.step()
        # `ticked`, not `env.ready`: `_micro_ct` does not advance on ANY tick where
        # `step()` early-returns without a decision -- both during the initial quorum
        # wait and during a later mid-run telemetry outage (`env.ready` stays True
        # sticky through the latter, so gating on `ready` alone still spins the exact
        # same way if an outage happens to start on a decision boundary -- architect
        # review, round-3 residual finding). Gating on whether this call's `step()`
        # actually advanced covers both cases with one check, and moving the replan
        # submission to run after `step()` (rather than the old pre-step check) costs
        # nothing now that submission is async -- `_rl_decide()` inside `step()` never
        # waited on it anyway, only `apply_replan_if_ready()`'s next poll does.
        ticked = env._micro_ct != before_ct
        if not env.ready:
            now = time.monotonic()
            if now - state["last_wait_print"] > 2.0:
                print(f"[gcs_bridge] GCS 텔레메트리 대기 중 -- 아직 안 보이는 배: "
                      f"{env.missing_ally_ids()}")
                state["last_wait_print"] = now
        elif ticked:
            if not state["seen_ready"]:
                # This is the tick where `_have_gcs` just latched inside `step()` --
                # no earlier call could have submitted a replan for it, so that
                # decision already ran with `_plan=None` (all-idle, safe). Force one
                # now rather than waiting up to a full `replan_period` for the modulo
                # to line up, or the fleet sits idle at startup for no reason.
                state["seen_ready"] = True
                replan_if_due(force=True)
            elif env._micro_ct % env.cfg.decision_period == 0:
                # log first, then check the cadence -- log_decision() is what advances
                # decision_idx, and checking it beforehand would still see decision 0's
                # own count, submitting a second (redundant, costly for a real LLM
                # backend) replan right on top of the forced one above.
                log_decision()
                replan_if_due()
        if args.realtime:
            remaining = env.scale.dt_real - (time.monotonic() - t_before)
            if remaining > 0:
                time.sleep(remaining)
        return True

    def max_reached() -> bool:
        return args.max_decisions is not None and state["decision_idx"] >= args.max_decisions

    if args.viz:
        try:
            from run_replay_infer import _run_viz
        except Exception as exc:
            print(f"[gcs_bridge] ⚠ --viz 재사용 실패({exc}) -- headless 로 계속합니다.")
            args.viz = False

    bg_img = bg_extent = None
    if args.viz and args.satellite:
        if args.geo:
            lat0, lon0 = float(args.geo[0]), float(args.geo[1])
        else:
            # --geo 미지정 -- enu_origin(맵 중앙)에 대응하는 실제 lat/lon을 GCS가 지금
            # 보고하는 --ally-ids 자신의 lat/lon 평균으로 구한다(별도 위경도 변환 불필요 --
            # GCS가 이미 datum 기준으로 재투영해 둔 값이다). enu_origin 자체도 이 배들의
            # ENU 위치 평균이라 같은 앵커를 가리킨다.
            try:
                # ★ 변수명 `state`는 쓰지 않는다 -- main() 스코프에서 `state = {...}`로
                # 재바인딩하면 아래 정의된 advance_one_micro() 가 클로저로 참조하는
                # "결정 진행상태" state 딕셔너리(`seen_ready` 등)를 덮어써서
                # KeyError('seen_ready')로 애니메이션이 깨진다(2026-09-11 세션 실측).
                gcs_state = client.get_json("/api/state")
                pts = [(gcs_state["vehicles"][v]["lat"], gcs_state["vehicles"][v]["lon"])
                       for v in ally_ids
                       if gcs_state["vehicles"].get(v) and gcs_state["vehicles"][v].get("lat") is not None]
            except Exception as exc:
                pts = []
                print(f"[gcs_bridge] ⚠ --satellite 앵커용 lat/lon 조회 실패({exc})")
            if pts:
                lat0 = sum(p[0] for p in pts) / len(pts)
                lon0 = sum(p[1] for p in pts) / len(pts)
            else:
                lat0 = lon0 = None
        if lat0 is None:
            print("[gcs_bridge] ⚠ --satellite 앵커(lat/lon)를 구하지 못했습니다 -- 기본 배경 사용. "
                  "--geo LAT LON 을 직접 지정해보세요.")
        else:
            from commander.satellite import fetch_satellite_bg
            print(f"[gcs_bridge] 위성 배경 로딩 중... (앵커 {lat0:.6f},{lon0:.6f}, "
                  f"--span {args.span:g} m)")
            res = fetch_satellite_bg(lat0, lon0, args.span)   # world_size 인자 = 실제 미터(--span)
            if res:
                img, ext_real = res   # ext_real: [xmin,xmax,ymin,ymax], 0~args.span 실미터 프레임
                # fetch_satellite_bg는 "world_size(여기선 --span 실미터) = 렌더 프레임 전체 폭"을
                # 가정한다(run_replay_infer.py의 기존 호출처럼 world_size==sim 프레임일 때만
                # 맞는 가정) -- 여기서는 sim 프레임이 cfg.world_size(항상 12600)로 고정이고
                # --span 은 SimScale.S 로 별도 변환되므로, 실미터 프레임(중심=--span/2)을
                # SimScale과 같은 배율로 sim 프레임(중심=cfg.world_size/2)에 맞춰 재투영해야
                # renderer.draw_scene(bg_extent=...)가 아군/적 궤적과 같은 좌표계에 놓인다.
                half_real = args.span / 2.0
                half_sim = env.cfg.world_size / 2.0
                bg_extent = [(v - half_real) * env.scale.S + half_sim for v in ext_real]
                bg_img = img
                print("[gcs_bridge] 위성 배경 로드 완료")
            else:
                print("[gcs_bridge] ⚠ 위성 배경 로드 실패(오프라인 등) -- 기본 배경 사용")

    try:
        if args.viz:
            _run_viz(env, advance_one_micro, max_reached, args.spf, log,
                     bg_img=bg_img, bg_extent=bg_extent, info_provider=lambda: info)
        else:
            while not max_reached():
                if not advance_one_micro():
                    break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(bag, "shutdown"):
            bag.shutdown()
        net_sink.shutdown()

    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[gcs_bridge] 결정 로그 저장: {args.out} ({len(log)}개 결정)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
