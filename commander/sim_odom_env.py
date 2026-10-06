"""commander/sim_odom_env.py — 시뮬레이터(ROS2 Odometry, 로컬 m 프레임)용 CNN 점수맵 정책 환경.

`GcsLiveCnnEnv`(적·아군 모두 실측, 물리 미실행)의 결정 루프·쿼럼·WP 진행·그물 판정·
재장전을 그대로 상속하고, 원천과 출력만 바꾼다:

  · 원천   GCS `/api/state`  →  `SimOdomLink` 의 `ally_view`/`enemy_view`(Odometry 최신값)
  · 출력   GCS goto(위경도)   →  `/usv/waypoints` JSON(x_m/y_m 남은 경로 전체) + `/usv/net_trigger`
  · 지형   체크포인트 해역    →  `land_source="none"`(섬 없음, land 채널은 유지하되 전부 0)

좌표: 시뮬레이터 map 프레임의 x/y(m)는 그대로 ENU(동/북 m)로 취급한다 — `SimScale` 의
`enu_origin`(보통 모선 Odometry = (0,0))이 sim 맵 중앙(world_size/2)에 대응한다.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .gcs_cnn_env import GcsLiveCnnEnv
from .sim_odom_ros2 import SimNetTriggerSink, SimShipView, SimWaypointSink


def auto_span(cfg, enemy_range_m: float, margin: float = 1.0) -> float:
    """적 출현거리를 학습 시 적 스폰반경(`enemy_spawn_radius`)에 맞추는 실미터 span.

    sim 에서 모선(맵 중앙)→적 스폰 = enemy_spawn_radius, 맵 반폭 = world_size/2 이므로
    span_real = 2·R_real·(world_size/2)/enemy_spawn_radius = R_real·world_size/enemy_spawn_radius.
    """
    if enemy_range_m <= 0:
        raise ValueError("enemy_range_m 은 0보다 커야 합니다(적이 아직 안 보임).")
    return float(enemy_range_m) * float(cfg.world_size) / float(cfg.enemy_spawn_radius) * margin


class SimOdomCnnEnv(GcsLiveCnnEnv):
    """시뮬레이터 Odometry 로 아군·적을 모두 읽고 x/y 경유점을 내보내는 CNN 점수맵 환경."""

    def __init__(
        self,
        ckpt: str,
        span_real: float,
        ally_view: SimShipView,
        enemy_view: SimShipView,
        *,
        waypoint_sink: Optional[SimWaypointSink] = None,
        net_sink: Optional[SimNetTriggerSink] = None,
        sim_id_fn: Callable[[], Optional[str]] = lambda: None,
        publish_hz: float = 2.0,
        ally_speed_real: float = 18.0,
        enemy_mode: str = "wave",
        device: str = "cpu",
        nets_per_ship: int = 3,
        enu_origin: tuple[float, float] | None = None,
        net_reload_period_real: float | None = None,
        quorum_timeout: float = 5.0,
        land_source: str | None = "none",
    ):
        super().__init__(
            ckpt, span_real, ally_view, enemy_view,
            net_sink=net_sink or SimNetTriggerSink(sim_id_fn=sim_id_fn),
            waypoint_sink=None, publish_hz=publish_hz, ally_speed_real=ally_speed_real,
            enemy_mode=enemy_mode, device=device, nets_per_ship=nets_per_ship,
            enu_origin=enu_origin, net_reload_period_real=net_reload_period_real,
            quorum_timeout=quorum_timeout, land_source=land_source,
        )
        # 부모는 GotoCmdSink 를 기본으로 만든다 — 여기선 쓰지 않으므로 시뮬레이터 sink 로 교체.
        self.waypoint_sink = waypoint_sink or SimWaypointSink()
        self.sim_id_fn = sim_id_fn
        if land_source == "none" and bool(np.any(self.land_map)):
            raise RuntimeError("land_source='none' 인데 land_map 이 비어있지 않습니다.")

    # ── 원천: GCS HTTP 대신 Odometry 스냅샷(뷰가 직접 읽으므로 공유 상태 불필요) ──
    def _poll_state(self):
        return None

    # ── 출력: 배별 '남은 경로' 전체를 x/y(m)로 ──────────────────────────
    def waypoints_xy(self, p: int) -> list[tuple[float, float]]:
        """배 p 에게 보낼 남은 경유점(실미터). 정지 대상이면 빈 목록.

        정지 규칙(`run_replay_infer._decode_route` 와 동일): 미배정(_assign<0)·그물 소진 배는
        제자리 — 단, 이미 그물을 치고 있는 배(doing_net)는 그 구간을 끝까지 간다.
        route[k>1] 은 마지막 점의 반복이라 연속 중복점은 버린다.
        """
        stop = (bool(self._assign[0, p] < 0) or bool(self.a_nets[0, p] <= 0)) \
            and not bool(self.doing_net[0, p])
        if stop:
            return []
        ptr = int(np.clip(self.ptr[0, p], 0, self.Kw - 1))
        pts: list[tuple[float, float]] = []
        for k in range(ptr, self.Kw):
            x, y = (float(c) for c in self.scale.sim_to_enu(self.route[0, p, k]))
            if pts and abs(pts[-1][0] - x) < 1e-6 and abs(pts[-1][1] - y) < 1e-6:
                continue
            pts.append((x, y))
        return pts

    def _publish_to_gcs(self) -> None:
        now = time.monotonic()
        if now < self._next_publish_wall:
            return
        self._next_publish_wall = now + self._publish_period_real
        sim_id = self.sim_id_fn()
        for p in range(self.P):
            if not bool(self.a_alive[0, p]):
                continue                          # 위치를 모르는 배는 명령하지 않는다
            self.waypoint_sink.publish(self.ally_link.vehicle_ids[p], self.waypoints_xy(p), sim_id)


__all__ = ["SimOdomCnnEnv", "auto_span"]
