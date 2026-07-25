"""
mqtt_ros2_bridge/usv_bridge.py — USV 방어 시뮬레이터 전용 브릿지

방어 모드 시뮬레이터와 oneway_ros2 RL 추론 시스템 간의 통합 브릿지.
"""

import json
import time
import threading
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# 공통 설정 로드 (MobRobGPT/config/defense_config.json)
def _load_defense_config():
    """defense_config.json에서 설정 로드."""
    paths = [
        os.environ.get("DEFENSE_CONFIG"),
        Path(__file__).parents[4] / "config" / "defense_config.json",
        Path.home() / "민철_UI" / "MobRobGPT" / "config" / "defense_config.json",
    ]
    for p in paths:
        if p and Path(p).exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

_CFG = _load_defense_config()
_CFG_WORLD = float(_CFG["world"]["size"]) if _CFG else 33.0
_CFG_LAT = float(_CFG["gps_origin"]["lat"]) if _CFG else 34.625
_CFG_LON = float(_CFG["gps_origin"]["lon"]) if _CFG else 128.52

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import NavSatFix, Imu
    from geometry_msgs.msg import Quaternion
    from nav_msgs.msg import Path
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from .config import BridgeConfig
from .mqtt_client import MqttClient
from .converters import (
    euler_to_quaternion,
    heading_nav_to_enu,
    path_to_waypoints,
    waypoints_to_mqtt_string,
)

# 클러스터 색상 (MobRobGPT 스타일)
CLUSTER_COLORS = ["#FF8A65", "#BA68C8", "#4FC3F7", "#FFD54F", "#81C784", "#F06292"]


@dataclass
class DefenseVessel:
    """방어 시뮬레이터 선박 상태."""
    id: int = 0
    x: float = 0.0  # 시뮬레이터 좌표
    z: float = 0.0
    lat: float = 0.0  # GPS 좌표
    lon: float = 0.0
    heading: float = 0.0  # nav convention (0=North, CW+)
    speed: float = 0.0
    alive: bool = True
    nets_remaining: int = 3
    painting: bool = False


@dataclass
class ClusterInfo:
    """클러스터 정보."""
    id: int = 0
    centroid_x: float = 0.0
    centroid_z: float = 0.0
    enemy_ids: List[int] = field(default_factory=list)
    threat: float = 0.0
    bearing: float = 0.0  # 모선 기준 방위각

@dataclass
class Assignment:
    """아군 → 클러스터 할당."""
    ally_id: int = 0
    cluster_id: int = -1  # -1 = 미할당
    status: str = "reserve"  # active, reserve, stopped

@dataclass
class CommanderState:
    """지휘관 판단 상태 (MobRobGPT 스타일)."""
    model: str = "oneway_ros2 (RL)"
    status: str = "ready"  # ready, calling, error
    command: str = "모든 적군 포획"
    clusters: List[ClusterInfo] = field(default_factory=list)
    assignments: List[Assignment] = field(default_factory=list)
    rationale: str = ""
    last_update: int = 0

@dataclass
class DefenseState:
    """방어 시뮬레이터 전체 상태."""
    allies: List[DefenseVessel] = field(default_factory=list)
    enemies: List[DefenseVessel] = field(default_factory=list)
    # GPS 원점: config/defense_config.json에서 로드
    mothership_lat: float = field(default_factory=lambda: _CFG_LAT)
    mothership_lon: float = field(default_factory=lambda: _CFG_LON)
    # 모선 위치: 월드 중심 (world_size / 2)
    mothership_x: float = field(default_factory=lambda: _CFG_WORLD / 2)
    mothership_z: float = field(default_factory=lambda: _CFG_WORLD / 2)
    # 월드 크기: config/defense_config.json에서 로드
    world_size: float = field(default_factory=lambda: _CFG_WORLD)
    step: int = 0
    running: bool = False
    commander: CommanderState = field(default_factory=CommanderState)


class DefenseBridge:
    """방어 시뮬레이터 ↔ ROS2 브릿지 (non-ROS2 테스트용)."""

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.state = DefenseState()
        self._lock = threading.Lock()

        # 시뮬레이터 좌표 → GPS 변환을 위한 기준점
        # 주의: sim_center는 동적으로 mothership 위치에서 가져옴
        self.meters_per_deg_lat = 111320.0
        self.meters_per_deg_lon = 111320.0 * 0.85  # 대략 위도 35도

    def sim_to_gps(self, x: float, z: float) -> tuple:
        """시뮬레이터 좌표 → GPS (lat, lon)."""
        # x: East (+), z: South (+)
        # mothership 위치를 GPS 기준점으로 사용 (스케일 독립적)
        dx = x - self.state.mothership_x  # East offset from mothership
        dz = z - self.state.mothership_z  # South offset from mothership

        lat = self.state.mothership_lat - (dz / self.meters_per_deg_lat)
        lon = self.state.mothership_lon + (dx / self.meters_per_deg_lon)

        return lat, lon

    def gps_to_sim(self, lat: float, lon: float) -> tuple:
        """GPS (lat, lon) → 시뮬레이터 좌표."""
        dlat = lat - self.state.mothership_lat
        dlon = lon - self.state.mothership_lon

        x = self.state.mothership_x + (dlon * self.meters_per_deg_lon)
        z = self.state.mothership_z - (dlat * self.meters_per_deg_lat)

        return x, z

    def update_from_mqtt(self, data: Dict[str, Any]):
        """MQTT 데이터로 상태 업데이트."""
        with self._lock:
            if "allies" in data:
                self.state.allies = []
                for a in data["allies"]:
                    vessel = DefenseVessel(
                        id=a.get("id", 0),
                        x=a.get("x", 0),
                        z=a.get("z", 0),
                        heading=a.get("heading", 0),
                        speed=a.get("speed", 0),
                        alive=a.get("alive", True),
                        nets_remaining=a.get("netsRemaining", 3),
                        painting=a.get("painting", False),
                    )
                    vessel.lat, vessel.lon = self.sim_to_gps(vessel.x, vessel.z)
                    self.state.allies.append(vessel)

            if "enemies" in data:
                self.state.enemies = []
                for e in data["enemies"]:
                    vessel = DefenseVessel(
                        id=e.get("id", 0),
                        x=e.get("x", 0),
                        z=e.get("z", 0),
                        heading=e.get("heading", 0),
                        speed=e.get("speed", 0),
                        alive=e.get("alive", True),
                    )
                    vessel.lat, vessel.lon = self.sim_to_gps(vessel.x, vessel.z)
                    self.state.enemies.append(vessel)

            if "mothership" in data:
                ms = data["mothership"]
                self.state.mothership_x = ms.get("x", 6300)
                self.state.mothership_z = ms.get("z", 6300)
                self.state.mothership_lat, self.state.mothership_lon = \
                    self.sim_to_gps(self.state.mothership_x, self.state.mothership_z)

            if "step" in data:
                self.state.step = data["step"]
            if "running" in data:
                self.state.running = data["running"]

    def get_state(self) -> DefenseState:
        """현재 상태 스냅샷."""
        with self._lock:
            return DefenseState(
                allies=list(self.state.allies),
                enemies=list(self.state.enemies),
                mothership_lat=self.state.mothership_lat,
                mothership_lon=self.state.mothership_lon,
                mothership_x=self.state.mothership_x,
                mothership_z=self.state.mothership_z,
                step=self.state.step,
                running=self.state.running,
                commander=self.state.commander,
            )

    def cluster_enemies(self, cluster_distance: float = 800.0) -> List[ClusterInfo]:
        """적군을 클러스터링 (DBSCAN-style greedy)."""
        alive_enemies = [e for e in self.state.enemies if e.alive]
        if not alive_enemies:
            return []

        clusters = []
        assigned = set()

        for enemy in alive_enemies:
            if enemy.id in assigned:
                continue

            # 새 클러스터 시작
            cluster_enemies = [enemy]
            assigned.add(enemy.id)

            # 가까운 적 추가
            for other in alive_enemies:
                if other.id in assigned:
                    continue
                dist = ((enemy.x - other.x) ** 2 + (enemy.z - other.z) ** 2) ** 0.5
                if dist < cluster_distance:
                    cluster_enemies.append(other)
                    assigned.add(other.id)

            # 클러스터 정보 계산
            cx = sum(e.x for e in cluster_enemies) / len(cluster_enemies)
            cz = sum(e.z for e in cluster_enemies) / len(cluster_enemies)

            # 모선 기준 방위각
            dx = cx - self.state.mothership_x
            dz = cz - self.state.mothership_z
            import math
            bearing = (math.degrees(math.atan2(dx, -dz)) + 360) % 360

            # 위협도 (적 수 + 모선 거리 역비례)
            dist_to_mother = (dx ** 2 + dz ** 2) ** 0.5
            threat = len(cluster_enemies) * 0.3 + (1.0 - min(dist_to_mother / 5000.0, 1.0)) * 0.7

            clusters.append(ClusterInfo(
                id=len(clusters),
                centroid_x=cx,
                centroid_z=cz,
                enemy_ids=[e.id for e in cluster_enemies],
                threat=min(threat, 1.0),
                bearing=bearing,
            ))

        return clusters

    def update_commander_state(self, waypoints_received: Dict[int, List] = None):
        """지휘관 상태 업데이트 (클러스터, 할당, 근거)."""
        with self._lock:
            # 클러스터링
            clusters = self.cluster_enemies()
            self.state.commander.clusters = clusters

            # 아군 할당 계산
            alive_allies = [a for a in self.state.allies if a.alive]
            sorted_clusters = sorted(clusters, key=lambda c: c.threat, reverse=True)

            assignments = []
            for i, ally in enumerate(alive_allies):
                if i < len(sorted_clusters):
                    assignments.append(Assignment(
                        ally_id=ally.id,
                        cluster_id=sorted_clusters[i].id,
                        status="active",
                    ))
                else:
                    assignments.append(Assignment(
                        ally_id=ally.id,
                        cluster_id=-1,
                        status="reserve",
                    ))

            # 죽은 아군
            for ally in self.state.allies:
                if not ally.alive:
                    assignments.append(Assignment(
                        ally_id=ally.id,
                        cluster_id=-1,
                        status="stopped",
                    ))

            self.state.commander.assignments = assignments

            # 판단 근거 생성
            self.state.commander.rationale = self._generate_rationale(clusters, assignments)
            self.state.commander.last_update = self.state.step

    def _generate_rationale(self, clusters: List[ClusterInfo], assignments: List[Assignment]) -> str:
        """판단 근거 텍스트 생성."""
        alive_enemies = len([e for e in self.state.enemies if e.alive])
        active_assigns = [a for a in assignments if a.status == "active"]

        if not clusters:
            return "적군 클러스터가 탐지되지 않음. 대기 상태 유지."

        lines = []

        # 클러스터 분석
        primary = max(clusters, key=lambda c: c.threat)
        lines.append(f"총 {alive_enemies}척의 적군이 {len(clusters)}개 클러스터로 분산됨.")
        lines.append(f"최대 위협 클러스터: C{primary.id} (방위 {primary.bearing:.0f}°, {len(primary.enemy_ids)}척)")

        # 할당 분석
        if active_assigns:
            lines.append(f"아군 {len(active_assigns)}척을 위협도 순으로 클러스터에 배치.")

        # TF 겹침 분석
        if len(clusters) >= 2:
            bearing_diff = abs(clusters[0].bearing - clusters[1].bearing)
            if bearing_diff < 30:
                lines.append(f"클러스터 간 방위각 차이 {bearing_diff:.0f}°로 TF 겹침 영역 존재.")

        return " ".join(lines)


if ROS2_AVAILABLE:
    class DefenseBridgeNode(Node):
        """방어 시뮬레이터 ROS2 브릿지 노드."""

        def __init__(self, config: Optional[BridgeConfig] = None):
            super().__init__("usv_defense_bridge")

            self.config = config or BridgeConfig.from_env()
            self.bridge = DefenseBridge(self.config)
            self.cb_group = ReentrantCallbackGroup()

            # MQTT 클라이언트
            self.mqtt = MqttClient(
                self.config,
                on_message=self._on_mqtt_message,
                logger=self.get_logger(),
            )

            # ROS2 발행자
            self.ally_fix_pubs = []
            self.ally_imu_pubs = []
            for i in range(self.config.n_allies):
                self.ally_fix_pubs.append(
                    self.create_publisher(NavSatFix, f"/ally_{i}/fix", 10)
                )
                self.ally_imu_pubs.append(
                    self.create_publisher(Imu, f"/ally_{i}/imu", 10)
                )

            self.enemy_fix_pubs = []
            for i in range(self.config.n_enemies):
                self.enemy_fix_pubs.append(
                    self.create_publisher(NavSatFix, f"/enemy_{i}/fix", 10)
                )

            # 모선 발행자
            self.mothership_fix_pub = self.create_publisher(NavSatFix, "/mothership/fix", 10)

            # ★ 시뮬레이터 이벤트 발행자 (리셋, 아군 상태)
            self.reset_event_pub = self.create_publisher(String, "/defense/reset", 10)
            self.allies_status_pub = self.create_publisher(String, "/defense/allies_status", 10)

            # ROS2 구독자 (oneway_ros2에서 웨이포인트 수신)
            self.wp_subs = []
            for i in range(self.config.n_allies):
                self.wp_subs.append(
                    self.create_subscription(
                        Path,
                        f"/ally_{i}/waypoints",
                        lambda msg, idx=i: self._on_waypoints(msg, idx),
                        10,
                        callback_group=self.cb_group
                    )
                )

            # ★ 외부 시뮬레이터 ROS2 토픽 구독 (ROS2 → MQTT)
            self.sim_state_sub = self.create_subscription(
                String,
                "/sim/defense/state",
                self._on_sim_state,
                10,
                callback_group=self.cb_group
            )

            # 발행 타이머
            self.publish_timer = self.create_timer(
                1.0 / self.config.telemetry_rate,
                self._publish_to_ros2,
                callback_group=self.cb_group
            )

            # MQTT 연결 및 구독
            if self.mqtt.connect():
                self.mqtt.subscribe("usv/defense/state")
                # 와일드카드로 개별 선박 토픽 구독
                self.mqtt.subscribe("usv/ally/+/telemetry")
                self.mqtt.subscribe("usv/enemy/+/telemetry")
                self.mqtt.subscribe("usv/mothership/state")
                # ★ 시뮬레이터 이벤트 구독 (리셋, 아군 상태)
                self.mqtt.subscribe("usv/system/events")
                self.mqtt.subscribe("usv/allies/status")
                self.get_logger().info("방어 시뮬레이터 브릿지 시작")
            else:
                self.get_logger().error("MQTT 연결 실패")

        def _on_mqtt_message(self, topic: str, payload: Dict[str, Any]):
            """MQTT 메시지 수신."""
            # ★ 시뮬레이터 이벤트 처리 (리셋, 아군 상태)
            if topic == "usv/system/events":
                # 리셋 이벤트 → ROS2로 발행
                msg = String()
                msg.data = json.dumps(payload)
                self.reset_event_pub.publish(msg)
                self.get_logger().info(f"★ 리셋 이벤트 ROS2 발행: {payload.get('event', 'unknown')}")
                return

            if topic == "usv/allies/status":
                # 아군 상태 이벤트 → ROS2로 발행
                msg = String()
                msg.data = json.dumps(payload)
                self.allies_status_pub.publish(msg)
                killed_ids = payload.get("killedIds", [])
                self.get_logger().info(f"★ 아군 상태 이벤트 ROS2 발행: killed={killed_ids}")
                return

            # 토픽 파싱: usv/ally/{id}/telemetry 또는 usv/enemy/{id}/telemetry
            parts = topic.split("/")
            if len(parts) >= 4 and parts[0] == "usv":
                vessel_type = parts[1]  # ally, enemy, mothership

                if vessel_type == "ally" and parts[3] == "telemetry":
                    self._update_ally(payload)
                elif vessel_type == "enemy" and parts[3] == "telemetry":
                    self._update_enemy(payload)
                elif vessel_type == "mothership":
                    self._update_mothership(payload)
                elif vessel_type == "defense":
                    self.bridge.update_from_mqtt(payload)
            else:
                self.bridge.update_from_mqtt(payload)


        def _update_ally(self, data: Dict[str, Any]):
            """개별 아군 텔레메트리 업데이트."""
            ally_id = data.get("id", 0)
            with self.bridge._lock:
                # 기존 아군 찾기 또는 생성
                state = self.bridge.state
                ally = None
                for a in state.allies:
                    if a.id == ally_id:
                        ally = a
                        break

                if ally is None:
                    ally = DefenseVessel(id=ally_id)
                    state.allies.append(ally)

                # 업데이트
                ally.x = data.get("x", ally.x)
                ally.z = data.get("z", ally.z)
                ally.heading = data.get("heading", ally.heading)
                ally.speed = data.get("speed", ally.speed)
                ally.alive = data.get("alive", ally.alive)
                ally.nets_remaining = data.get("netsRemaining", ally.nets_remaining)
                ally.painting = data.get("painting", ally.painting)

                # GPS 좌표 처리
                raw_lat = data.get("lat")
                raw_lon = data.get("lon")
                if raw_lat is not None and raw_lon is not None:
                    # GPS 범위 밖이면 시뮬레이션 좌표로 간주
                    if raw_lat < -90 or raw_lat > 90 or raw_lon < -180 or raw_lon > 180:
                        ally.lat, ally.lon = self.bridge.sim_to_gps(raw_lat, raw_lon)
                    else:
                        ally.lat, ally.lon = raw_lat, raw_lon
                else:
                    # lat/lon이 없으면 x/z에서 변환
                    ally.lat, ally.lon = self.bridge.sim_to_gps(ally.x, ally.z)

        def _update_enemy(self, data: Dict[str, Any]):
            """개별 적군 텔레메트리 업데이트."""
            enemy_id = data.get("id", 0)
            with self.bridge._lock:
                state = self.bridge.state
                enemy = None
                for e in state.enemies:
                    if e.id == enemy_id:
                        enemy = e
                        break

                if enemy is None:
                    enemy = DefenseVessel(id=enemy_id)
                    state.enemies.append(enemy)

                enemy.x = data.get("x", enemy.x)
                enemy.z = data.get("z", enemy.z)
                enemy.heading = data.get("heading", enemy.heading)
                enemy.speed = data.get("speed", enemy.speed)
                enemy.alive = data.get("alive", enemy.alive)

                # GPS 좌표 처리
                raw_lat = data.get("lat")
                raw_lon = data.get("lon")
                if raw_lat is not None and raw_lon is not None:
                    # GPS 범위 밖이면 시뮬레이션 좌표로 간주
                    if raw_lat < -90 or raw_lat > 90 or raw_lon < -180 or raw_lon > 180:
                        enemy.lat, enemy.lon = self.bridge.sim_to_gps(raw_lat, raw_lon)
                    else:
                        enemy.lat, enemy.lon = raw_lat, raw_lon
                else:
                    # lat/lon이 없으면 x/z에서 변환
                    enemy.lat, enemy.lon = self.bridge.sim_to_gps(enemy.x, enemy.z)

        def _update_mothership(self, data: Dict[str, Any]):
            """모선 상태 업데이트."""
            with self.bridge._lock:
                state = self.bridge.state
                state.mothership_x = data.get("x", state.mothership_x)
                state.mothership_z = data.get("z", state.mothership_z)

                # GPS 좌표 처리
                raw_lat = data.get("lat")
                raw_lon = data.get("lon")
                if raw_lat is not None and raw_lon is not None:
                    # GPS 범위 밖이면 시뮬레이션 좌표로 간주
                    if raw_lat < -90 or raw_lat > 90 or raw_lon < -180 or raw_lon > 180:
                        state.mothership_lat, state.mothership_lon = self.bridge.sim_to_gps(raw_lat, raw_lon)
                    else:
                        state.mothership_lat, state.mothership_lon = raw_lat, raw_lon
                else:
                    # lat/lon이 없으면 x/z에서 변환
                    state.mothership_lat, state.mothership_lon = self.bridge.sim_to_gps(
                        state.mothership_x, state.mothership_z
                    )

        def _publish_to_ros2(self):
            """ROS2로 센서 데이터 발행."""
            state = self.bridge.get_state()
            now = self.get_clock().now().to_msg()

            # 모선 발행
            mothership_fix = NavSatFix()
            mothership_fix.header.stamp = now
            mothership_fix.header.frame_id = "mothership"
            mothership_fix.latitude = state.mothership_lat
            mothership_fix.longitude = state.mothership_lon
            self.mothership_fix_pub.publish(mothership_fix)

            # 아군 발행
            for ally in state.allies:
                if ally.id >= len(self.ally_fix_pubs):
                    continue
                if not ally.alive:
                    continue

                # GPS
                fix = NavSatFix()
                fix.header.stamp = now
                fix.header.frame_id = f"ally_{ally.id}"
                fix.latitude = ally.lat
                fix.longitude = ally.lon
                self.ally_fix_pubs[ally.id].publish(fix)

                # IMU
                imu = Imu()
                imu.header.stamp = now
                imu.header.frame_id = f"ally_{ally.id}"
                yaw_enu = heading_nav_to_enu(ally.heading)
                x, y, z, w = euler_to_quaternion(0, 0, yaw_enu * 3.14159 / 180)
                imu.orientation = Quaternion(x=x, y=y, z=z, w=w)
                self.ally_imu_pubs[ally.id].publish(imu)

            # 적군 발행
            for enemy in state.enemies:
                if enemy.id >= len(self.enemy_fix_pubs):
                    continue
                if not enemy.alive:
                    continue

                fix = NavSatFix()
                fix.header.stamp = now
                fix.header.frame_id = f"enemy_{enemy.id}"
                fix.latitude = enemy.lat
                fix.longitude = enemy.lon
                self.enemy_fix_pubs[enemy.id].publish(fix)

        def _on_sim_state(self, msg: String):
            """외부 시뮬레이터에서 상태 수신 → MQTT로 전송."""
            try:
                data = json.loads(msg.data)
                # MQTT로 발행
                self.mqtt.publish("usv/defense/state", data)
                self.get_logger().debug(f"외부 시뮬레이터 상태 전달: step={data.get('step', 0)}")
            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON 파싱 오류: {e}")
            except Exception as e:
                self.get_logger().error(f"외부 시뮬레이터 상태 처리 오류: {e}")

        def _on_waypoints(self, msg: Path, ally_idx: int):
            """oneway_ros2에서 웨이포인트 수신 → 시뮬레이터로 전송."""
            import math

            frame_id = msg.header.frame_id.lower()

            sim_waypoints = []
            target_world_size = self.bridge.state.world_size  # ≈ 46.24m (시뮬레이터)
            world_size_original = 12600.0  # MobRobGPT 원본 맵 크기

            # 첫 번째 좌표로 좌표계 유형 추정
            if not msg.poses:
                return

            first_x = msg.poses[0].pose.position.x
            first_y = msg.poses[0].pose.position.y

            # 모든 좌표의 최대값으로 소스 월드 크기 추정
            max_coord = max(
                max(abs(pose.pose.position.x) for pose in msg.poses),
                max(abs(pose.pose.position.y) for pose in msg.poses)
            )

            # 좌표 범위로 타입 추정
            # Web Mercator (EPSG:3857): x,y in millions (위성지도 투영)
            # GPS (WGS84): lat in [-90, 90], lon in [-180, 180]
            # 원본 월드: 0~12600
            # 스케일드 월드: 0~46
            is_web_mercator = max_coord > 100000  # 100km 이상이면 Web Mercator
            is_gps = (-90 <= first_y <= 90) and (-180 <= first_x <= 180) and not is_web_mercator
            is_original_world = 100 < max_coord <= 20000  # 100m~20km면 원본 월드

            # Web Mercator → WGS84 변환 함수
            def webmercator_to_wgs84(x, y):
                """EPSG:3857 → EPSG:4326 (WGS84)"""
                EARTH_RADIUS = 6378137.0  # 미터
                lon = (x / EARTH_RADIUS) * (180 / math.pi)
                lat = (2 * math.atan(math.exp(y / EARTH_RADIUS)) - math.pi / 2) * (180 / math.pi)
                return lat, lon

            self.get_logger().info(
                f"Ally {ally_idx}: 좌표 분석 - frame={frame_id}, "
                f"first=({first_x:.4f}, {first_y:.4f}), max={max_coord:.2f}, "
                f"webmercator={is_web_mercator}, gps={is_gps}, original={is_original_world}"
            )

            net_mask = []  # 그물 전개 구간

            # frame_id='world'는 스케일드 월드 좌표로 직접 처리 (GPS 감지보다 우선)
            if frame_id == "world":
                # 스케일드 월드 좌표 (0~33m) → y→z 변환만
                for pose in msg.poses:
                    x = pose.pose.position.x
                    y = pose.pose.position.y
                    z = target_world_size - y  # y(North) → z(South)
                    paint = pose.pose.position.z > 0.5
                    sim_waypoints.append({"x": x, "z": z, "paint": paint})
                    net_mask.append(paint)
                coord_type = f"world→SIM({target_world_size:.1f}m)"

            elif is_web_mercator:
                # Web Mercator (EPSG:3857) → WGS84 → 시뮬레이터 좌표
                for i, pose in enumerate(msg.poses):
                    merc_x = pose.pose.position.x
                    merc_y = pose.pose.position.y
                    lat, lon = webmercator_to_wgs84(merc_x, merc_y)
                    x, z = self.bridge.gps_to_sim(lat, lon)
                    paint = pose.pose.position.z > 0.5
                    sim_waypoints.append({"x": x, "z": z, "paint": paint})
                    net_mask.append(paint)
                coord_type = "WebMercator→GPS→SIM"

            elif frame_id in ("wgs84", "gps", "earth") or is_gps:
                # WGS84/GPS 좌표 (lat, lon) → 시뮬레이터 좌표로 변환
                waypoints = path_to_waypoints(msg)
                for i, (lat, lon) in enumerate(waypoints):
                    x, z = self.bridge.gps_to_sim(lat, lon)
                    paint = msg.poses[i].pose.position.z > 0.5 if i < len(msg.poses) else False
                    sim_waypoints.append({"x": x, "z": z, "paint": paint})
                    net_mask.append(paint)
                coord_type = "GPS→SIM"

            elif is_original_world:
                # MobRobGPT 원본 좌표 (0~12600m) → 스케일 + y→z 변환
                scale = target_world_size / world_size_original
                for pose in msg.poses:
                    x_orig = pose.pose.position.x
                    y_orig = pose.pose.position.y
                    x = x_orig * scale
                    z = target_world_size - (y_orig * scale)
                    paint = pose.pose.position.z > 0.5
                    sim_waypoints.append({"x": x, "z": z, "paint": paint})
                    net_mask.append(paint)
                coord_type = f"원본월드(12600m)→{target_world_size:.1f}m"

            else:
                # 이미 스케일드 월드 좌표 (0~46m) → y→z 변환만
                for pose in msg.poses:
                    x = pose.pose.position.x
                    y = pose.pose.position.y
                    z = target_world_size - y
                    paint = pose.pose.position.z > 0.5
                    sim_waypoints.append({"x": x, "z": z, "paint": paint})
                    net_mask.append(paint)
                coord_type = f"스케일드월드({target_world_size:.1f}m)"

            # 디버그: 첫 번째와 마지막 WP 좌표 출력
            if sim_waypoints:
                first_wp = sim_waypoints[0]
                last_wp = sim_waypoints[-1]
                self.get_logger().info(
                    f"Ally {ally_idx}: {coord_type} WP {len(sim_waypoints)}개 → "
                    f"first=({first_wp['x']:.2f}, {first_wp['z']:.2f}) "
                    f"last=({last_wp['x']:.2f}, {last_wp['z']:.2f})"
                )

            if not sim_waypoints:
                return

            # MQTT로 전송 (net_mask 포함)
            paint_count = sum(1 for p in net_mask if p)
            self.mqtt.publish(f"usv/ally/{ally_idx}/route", {
                "waypoints": sim_waypoints,
                "net_mask": net_mask,
                "source": "oneway_ros2",
            })
            self.get_logger().info(
                f"Ally {ally_idx}: MQTT 전송 완료 (그물 {paint_count}/{len(net_mask)} 구간)"
            )

            # 지휘관 상태 업데이트 및 발행
            self.bridge.update_commander_state()
            self._publish_commander_state()

        def _publish_commander_state(self):
            """지휘관 상태 MQTT 발행 (MobRobGPT 스타일)."""
            state = self.bridge.get_state()
            cmd = state.commander

            # 클러스터 정보
            clusters = []
            for c in cmd.clusters:
                clusters.append({
                    "id": c.id,
                    "centroidX": c.centroid_x,
                    "centroidZ": c.centroid_z,
                    "enemyIds": c.enemy_ids,
                    "enemyCount": len(c.enemy_ids),
                    "threat": c.threat,
                    "bearing": c.bearing,
                    "color": CLUSTER_COLORS[c.id % len(CLUSTER_COLORS)],
                })

            # 할당 정보
            assignments = []
            for a in cmd.assignments:
                assignments.append({
                    "allyId": a.ally_id,
                    "clusterId": a.cluster_id,
                    "status": a.status,
                })

            commander_msg = {
                "model": cmd.model,
                "status": cmd.status,
                "command": cmd.command,
                "clusters": clusters,
                "assignments": assignments,
                "rationale": cmd.rationale,
                "lastUpdate": cmd.last_update,
            }

            self.mqtt.publish("usv/commander/state", commander_msg)
            self.get_logger().debug(f"Commander state published: {len(clusters)} clusters")

        def destroy_node(self):
            self.mqtt.disconnect()
            super().destroy_node()


def main():
    """방어 브릿지 노드 실행."""
    if not ROS2_AVAILABLE:
        print("ROS2 not available")
        return

    rclpy.init()

    config = BridgeConfig.from_env()
    config.n_allies = 3
    config.n_enemies = 10

    node = DefenseBridgeNode(config)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
