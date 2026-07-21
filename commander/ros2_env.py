"""
commander/ros2_env.py — ROS2 실시간 센서 기반 환경

--ros2 옵션 사용 시, 시뮬레이션 대신 실제 ROS2 센서 데이터를 사용하여
UI를 표시하고 waypoint를 발행한다.

주요 기능:
  - /enemy_X/fix 토픽에서 적 위치 수신
  - /ally_X/fix, /ally_X/imu 토픽에서 아군 위치/방위 수신
  - GPS → 시뮬 좌표 변환 후 클러스터링
  - LLM 지휘관 명령 → 경로 생성 → /ally_X/waypoints 발행
"""

import numpy as np
from typing import Optional

from .ros2_sensor_bridge import create_ros2_bridge, ROS2SensorBridge, ROS2_AVAILABLE
from .sim_bridge import plan_to_assign
from .schema import (
    BattlefieldState, Mothership, EnemyCluster, AllyShip, Constraints, Point,
)


class ROS2CommanderEnv:
    """ROS2 센서 기반 지휘관 환경."""

    def __init__(
        self,
        n_allies: int = 3,
        n_enemies: int = 10,
        world_size: float = 12600.0,
        mothership_radius: float = 260.0,
    ):
        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self.world_size = world_size
        self.mothership_radius = mothership_radius

        # 설정 객체 (시뮬레이터 호환용)
        self.cfg = type("Cfg", (), {
            "world_size": world_size,
            "mothership_radius": mothership_radius,
            "n_allies": n_allies,
            "n_enemies": n_enemies,
            "n_clusters": 3,
            "cluster_gap_deg": 11.97,
            "enemy_speed": 9.0,
            "ally_speed": 6.0,
            "ship_len": 230.0,
            "ship_wid": 76.0,
            "enemy_size": 125.0,
            "moback_size": 380.0,
            "moback_heading": 0.0,
            "geo_lat": 34.625,
            "geo_lon": 128.52,
            "nets_per_ship": 5,
            "transit_wp": 6,
        })()

        # ROS2 브릿지
        self._bridge: Optional[ROS2SensorBridge] = None
        self._state_updated = False

        # 내부 상태
        self.P = n_allies
        self.M = n_enemies
        self.Kw = 6  # waypoint 개수

        # 아군/적 상태 (시뮬 좌표)
        self.a_pos = np.zeros((self.P, 2))
        self.a_hdg = np.zeros(self.P)
        self.a_alive = np.ones(self.P, dtype=bool)
        self.a_nets = np.full(self.P, 5)

        self.e_pos = np.zeros((self.M, 2))
        self.e_hdg = np.zeros(self.M)
        self.e_alive = np.ones(self.M, dtype=bool)

        # 경로/그물
        self.route = np.zeros((self.P, self.Kw, 2))
        self.net_mask = np.zeros((self.P, self.Kw), dtype=bool)
        self.doing_net = np.zeros(self.P, dtype=bool)

        # 배정
        self._assign = np.full(self.P, -1, dtype=np.int64)
        self._assignI = np.zeros((self.P, 2))

        # 모선
        self.center = np.array([world_size / 2, world_size / 2])

        # 계획
        self._plan = None
        self._plan_command = None

        # 상태
        self.running = True
        self.done = False  # 에피소드 종료 여부
        self.t = 0
        self.stats = {
            "captures": 0,
            "breaches": 0,
            "ally_collisions": 0,
            "nets_used": 0,
            "survived": 0,
        }

    def start_ros2(self):
        """ROS2 브릿지 시작."""
        def on_update():
            self._state_updated = True

        self._bridge = create_ros2_bridge(
            n_allies=self.n_allies,
            n_enemies=self.n_enemies,
            world_size=self.world_size,
            on_state_update=on_update,
        )
        return self._bridge

    def update_from_ros2(self):
        """ROS2 센서 데이터로 상태 업데이트."""
        if self._bridge is None:
            return False

        sim_state = self._bridge.get_sim_state()
        if sim_state is None:
            return False

        # 상태 업데이트
        self.a_pos = sim_state["ally_pos"]
        self.a_hdg = sim_state["ally_hdg"]
        self.a_alive = sim_state["ally_alive"]

        self.e_pos = sim_state["enemy_pos"]
        self.e_hdg = sim_state["enemy_hdg"]
        self.e_alive = sim_state["enemy_alive"]

        self.center = sim_state["center"]

        self._state_updated = False
        return True

    def set_plan(self, plan, command: str = ""):
        """LLM 계획 설정."""
        self._plan = plan
        self._plan_command = command

        if plan is None:
            return

        # 현재 전장 상태 구축 (plan_to_assign에 필요)
        state = build_battlefield_ros2(self, command)

        # 배정 계산 (BattlefieldState 전달)
        assign = plan_to_assign(plan, state)
        self._assign = np.array(assign, dtype=np.int64)

        # 요격점 계산: 클러스터 중심 → 모선 방향의 요격 지점
        for p in range(self.P):
            cid = int(self._assign[p])
            if cid >= 0 and cid < len(state.enemy_clusters):
                cl = state.enemy_clusters[cid]
                # 모선 방향으로 요격점 설정
                cx, cy = cl.center.x, cl.center.y
                mx, my = self.center[0], self.center[1]
                dx, dy = mx - cx, my - cy
                d = np.hypot(dx, dy)
                if d > 1.0:
                    # 적보다 먼저 도달 가능한 요격 반경
                    v_a = self.cfg.ally_speed
                    v_e = self.cfg.enemy_speed
                    r = v_a * d / (v_a + v_e)
                    r = min(r, self.world_size / 3.0)  # 최대 요격 반경
                    self._assignI[p] = np.array([mx - dx/d * r, my - dy/d * r])
                else:
                    self._assignI[p] = self.center.copy()
            else:
                self._assignI[p] = self.a_pos[p].copy()

        # 경로 생성 (요격점으로 직진)
        for p in range(self.P):
            if self._assign[p] >= 0:
                target = self._assignI[p]
                for k in range(self.Kw):
                    frac = (k + 1) / self.Kw
                    self.route[p, k] = self.a_pos[p] + frac * (target - self.a_pos[p])
                self.net_mask[p, :] = True
            else:
                self.route[p, :, :] = self.a_pos[p]
                self.net_mask[p, :] = False

        # 경로 발행
        self.publish_waypoints()

    def publish_waypoints(self):
        """경로를 ROS2로 발행."""
        if self._bridge is not None:
            self._bridge.publish_waypoints(self.route, self.net_mask)

    def step(self):
        """한 스텝 진행 (ROS2 상태 업데이트)."""
        self.update_from_ros2()
        self.t += 1

        # 통계 업데이트 (돌파, 포획 등)
        for i in range(self.M):
            if self.e_alive[i]:
                dist = np.linalg.norm(self.e_pos[i] - self.center)
                if dist < self.mothership_radius:
                    self.e_alive[i] = False
                    self.stats["breaches"] += 1

        self.stats["survived"] = int(self.e_alive.sum())

    def reset(self, seed: int = None):
        """리셋 (ROS2에서는 상태만 초기화). seed는 호환성을 위해 무시됨."""
        self.t = 0
        self._plan = None
        self.stats = {k: 0 for k in self.stats}
        self._assign[:] = -1
        self.net_mask[:] = False
        self.doing_net[:] = False

    def get_frame(self) -> dict:
        """UI 렌더링용 프레임 데이터."""
        return {
            "world_size": self.world_size,
            "cell_size": self.world_size / 200,
            "t": self.t,
            "done": False,
            "mothership": self.center,
            "mothership_radius": self.mothership_radius,
            "moback_size": self.cfg.moback_size,
            "moback_heading": self.cfg.moback_heading,
            "enemy_pos": self.e_pos,
            "enemy_hdg": self.e_hdg,
            "enemy_alive": self.e_alive,
            "enemy_size": self.cfg.enemy_size,
            "ally_pos": self.a_pos,
            "ally_hdg": self.a_hdg,
            "ally_paths": self._get_ally_paths(),
            "ally_nets": self.a_nets,
            "ally_painting": self.doing_net,
            "ally_alive": self.a_alive,
            "assign": self._assign,
            "assignI": self._assignI,
            "route": self.route,
            "net_mask": self.net_mask,
            "ship_len": self.cfg.ship_len,
            "ship_wid": self.cfg.ship_wid,
            "painted": np.zeros((200, 200), dtype=bool),
            "selected": -1,
            "manual": False,
            "running": self.running,
            "net_stage": 0,
            "stats": dict(self.stats),
            "n_alive": int(self.e_alive.sum()),
            "n_clusters": self.cfg.n_clusters,
            "cluster_gap_deg": self.cfg.cluster_gap_deg,
            "enemy_speed": self.cfg.enemy_speed,
            "show_clusters": True,
            "show_residual": False,
            "wp_adjust_max": 500.0,
            "enemy_mode": "ros2",
            "ros2_mode": True,  # ROS2 모드 플래그
        }

    def _get_ally_paths(self):
        """경로 데이터 생성."""
        paths = []
        for p in range(self.P):
            wps = []
            for k in range(self.Kw):
                wps.append({
                    "x": float(self.route[p, k, 0]),
                    "y": float(self.route[p, k, 1]),
                    "paint": bool(self.net_mask[p, k]),
                })
            paths.append(wps)
        return paths


def build_battlefield_ros2(env: ROS2CommanderEnv, command: str = "") -> BattlefieldState:
    """ROS2 환경에서 전장 상태 구축 (LLM 입력용).

    BattlefieldState 객체를 반환하여 ollama_commander 등과 호환.
    """
    c = env.center
    cfg = env.cfg

    # 간단한 클러스터링 (실제로는 더 정교해야 함)
    clusters = []
    alive_enemies = np.where(env.e_alive)[0]
    if len(alive_enemies) > 0:
        # 위치 기반 간단 클러스터링
        positions = env.e_pos[alive_enemies]
        centroid = positions.mean(axis=0)

        # 방위각 계산 (모선 기준)
        bearing = float(np.degrees(np.arctan2(centroid[0] - c[0], centroid[1] - c[1])) % 360.0)

        clusters.append(EnemyCluster(
            id=0,
            center=Point(x=float(centroid[0]), y=float(centroid[1])),
            bearing=bearing,
            spread=30.0,  # 기본 스프레드
            count=len(alive_enemies),
            approach_speed=float(cfg.enemy_speed),
            net_covered=False,
        ))

    # 위협 수준 계산
    if env.e_alive.any():
        d = np.hypot(env.e_pos[env.e_alive, 0] - c[0], env.e_pos[env.e_alive, 1] - c[1]).min()
        threat = float(np.clip(1.0 - d / (env.world_size / 2.0), 0.0, 1.0))
    else:
        threat = 0.0

    # 아군 상태
    allies = []
    for p in range(env.P):
        allies.append(AllyShip(
            id=p,
            pos=Point(x=float(env.a_pos[p, 0]), y=float(env.a_pos[p, 1])),
            heading=float(env.a_hdg[p]),
            nets_remaining=int(env.a_nets[p]),
            alive=bool(env.a_alive[p]),
            assigned_cluster=int(env._assign[p]) if int(env._assign[p]) >= 0 else None,
            route=[Point(x=float(env.route[p, k, 0]), y=float(env.route[p, k, 1]))
                   for k in range(env.Kw) if env.net_mask[p, k]],
            deploying=bool(env.doing_net[p]),
            route_hits_net=False,
            cluster_covered_by_teammate=False,
        ))

    return BattlefieldState(
        mothership=Mothership(
            pos=Point(x=float(c[0]), y=float(c[1])),
            radius=float(env.mothership_radius),
            threat_level=threat,
        ),
        enemy_clusters=clusters,
        allies=allies,
        constraints=Constraints(
            net_max_len=float(getattr(cfg, "net_max_len", 800.0)),
            ally_speed=float(cfg.ally_speed),
            enemy_speed=float(cfg.enemy_speed),
            world_size=float(env.world_size),
            max_intercept_radius=float(env.world_size / 3.0),
        ),
        command=command if command else None,
    )
