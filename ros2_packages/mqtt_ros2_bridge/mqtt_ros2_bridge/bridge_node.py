"""
mqtt_ros2_bridge/bridge_node.py — MQTT ↔ ROS2 브릿지 노드

USV 시뮬레이터(MQTT)와 ROS2 시스템 간의 양방향 브릿지.

시뮬레이터에서 시뮬레이션 좌표(x, z meters)를 lat/lon 필드로 보내는 경우
실제 GPS 좌표로 변환하여 발행.
"""

import json
import math
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import NavSatFix, Imu
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    from std_msgs.msg import Float32, Bool, String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[bridge_node] ROS2 not available")

from .config import BridgeConfig
from .mqtt_client import MqttClient
from .converters import (
    telemetry_to_navsatfix,
    telemetry_to_imu,
    navsatfix_to_telemetry,
    imu_to_heading_nav,
    path_to_waypoints,
    waypoints_to_mqtt_string,
    mqtt_string_to_waypoints,
)


# ============================================================
# 시뮬레이션 좌표 → GPS 변환 설정
# ============================================================
# 기준 GPS 좌표 (시뮬레이션 월드 중심 = 모선 위치)
GPS_BASE_LAT = 34.625   # 남해 매물도 근해
GPS_BASE_LON = 128.52

# 시뮬레이션 월드 크기 (meters)
SIM_WORLD_SIZE = 12600.0  # Unity 시뮬레이터 기본값

# 좌표 변환 상수
DEG_TO_M_LAT = 111320.0  # 위도 1도 ≈ 111.32km


def is_sim_coordinate(lat: float, lon: float) -> bool:
    """시뮬레이션 좌표인지 판별 (GPS 범위 밖이면 시뮬레이션 좌표)."""
    if not math.isfinite(lat) or not math.isfinite(lon):
        return True
    if lat < -90 or lat > 90:
        return True
    if lon < -180 or lon > 180:
        return True
    return False


def sim_to_gps(sim_x: float, sim_z: float, world_size: float = SIM_WORLD_SIZE) -> tuple:
    """시뮬레이션 좌표 (x, z) → GPS 좌표 (lat, lon).

    시뮬레이션 좌표계:
        x: East (+), 원점 = world_size / 2
        z: South (+) 또는 North (+) - 설정에 따라 다름

    여기서는 x=East, z=North로 가정 (일반적인 ENU 좌표계).
    """
    center = world_size / 2

    # 중심 기준 오프셋 (meters)
    dx = sim_x - center  # East offset
    dz = sim_z - center  # North offset (z=North+)

    # GPS 변환
    lat = GPS_BASE_LAT + (dz / DEG_TO_M_LAT)
    lon = GPS_BASE_LON + (dx / (DEG_TO_M_LAT * math.cos(math.radians(GPS_BASE_LAT))))

    return lat, lon


@dataclass
class VesselState:
    """선박 상태 캐시."""
    lat: float = 0.0
    lon: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    battery: float = 100.0
    autopilot: bool = False
    last_update: float = 0.0


class MqttRos2Bridge:
    """MQTT ↔ ROS2 브릿지 (non-ROS2 테스트 지원)."""

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.vessels: Dict[str, VesselState] = {}
        self._lock = threading.Lock()

    def update_vessel(self, vessel_id: str, data: Dict[str, Any]):
        """선박 상태 업데이트."""
        with self._lock:
            if vessel_id not in self.vessels:
                self.vessels[vessel_id] = VesselState()
            v = self.vessels[vessel_id]
            if "lat" in data:
                v.lat = data["lat"]
            if "lon" in data:
                v.lon = data["lon"]
            if "heading" in data:
                v.heading = data["heading"]
            if "speed_ms" in data:
                v.speed = data["speed_ms"]
            if "battery" in data:
                v.battery = data["battery"]
            if "autopilot" in data:
                v.autopilot = bool(data["autopilot"])

    def get_vessel(self, vessel_id: str) -> Optional[VesselState]:
        """선박 상태 조회."""
        with self._lock:
            return self.vessels.get(vessel_id)


if ROS2_AVAILABLE:
    class MqttRos2BridgeNode(Node):
        """ROS2 브릿지 노드."""

        def __init__(self, config: Optional[BridgeConfig] = None):
            super().__init__("mqtt_ros2_bridge")

            self.config = config or BridgeConfig.from_env()
            self.cb_group = ReentrantCallbackGroup()

            # MQTT 클라이언트
            self.mqtt = MqttClient(
                self.config,
                on_message=self._on_mqtt_message,
                logger=self.get_logger(),
            )

            # 선박 상태 캐시
            self.ally_states: Dict[int, VesselState] = {}
            self.enemy_states: Dict[int, VesselState] = {}

            # QoS 설정
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            # === ROS2 발행자 (MQTT → ROS2) ===
            self.ally_fix_pubs = []
            self.ally_imu_pubs = []
            for i in range(self.config.n_allies):
                fix_pub = self.create_publisher(
                    NavSatFix,
                    f"{self.config.ros2_ally_prefix}{i}/fix",
                    10
                )
                imu_pub = self.create_publisher(
                    Imu,
                    f"{self.config.ros2_ally_prefix}{i}/imu",
                    10
                )
                self.ally_fix_pubs.append(fix_pub)
                self.ally_imu_pubs.append(imu_pub)

            self.enemy_fix_pubs = []
            for i in range(self.config.n_enemies):
                fix_pub = self.create_publisher(
                    NavSatFix,
                    f"{self.config.ros2_enemy_prefix}{i}/fix",
                    10
                )
                self.enemy_fix_pubs.append(fix_pub)

            # 모선 발행자
            self.mothership_fix_pub = self.create_publisher(
                NavSatFix, "/mothership/fix", 10
            )
            self.mothership_state = VesselState(lat=GPS_BASE_LAT, lon=GPS_BASE_LON)

            # === ROS2 구독자 (ROS2 → MQTT) ===
            self.wp_subs = []
            for i in range(self.config.n_allies):
                sub = self.create_subscription(
                    Path,
                    f"{self.config.ros2_ally_prefix}{i}/waypoints",
                    lambda msg, idx=i: self._on_waypoints(msg, idx),
                    10,
                    callback_group=self.cb_group
                )
                self.wp_subs.append(sub)

            # 타이머
            self.publish_timer = self.create_timer(
                1.0 / self.config.telemetry_rate,
                self._publish_to_ros2,
                callback_group=self.cb_group
            )

            # MQTT 연결
            if self.mqtt.connect():
                self._subscribe_mqtt_topics()
                self.get_logger().info("MQTT-ROS2 브릿지 시작")
            else:
                self.get_logger().error("MQTT 연결 실패")

        def _subscribe_mqtt_topics(self):
            """MQTT 토픽 구독."""
            # 기존 단일 선박 모드
            telemetry_topic = self.config.mqtt_telemetry_topic.format(
                token=self.config.device_token
            )
            self.mqtt.subscribe(telemetry_topic)

            # 다중 선박 모드
            for i in range(self.config.n_allies):
                self.mqtt.subscribe(f"usv/ally/{i}/telemetry")
            for i in range(self.config.n_enemies):
                self.mqtt.subscribe(f"usv/enemy/{i}/telemetry")

            self.mqtt.subscribe("usv/ally/all/telemetry")
            self.mqtt.subscribe("usv/enemy/all/telemetry")
            self.mqtt.subscribe("usv/mothership/state")
            self.mqtt.subscribe("usv/mothership/telemetry")

        def _on_mqtt_message(self, topic: str, payload: Dict[str, Any]):
            """MQTT 메시지 수신 콜백."""
            try:
                # 토픽 파싱
                if topic.startswith("usv/ally/"):
                    parts = topic.split("/")
                    if parts[2] == "all":
                        # 전체 아군 일괄
                        self._handle_all_allies(payload)
                    else:
                        idx = int(parts[2])
                        self._handle_ally_telemetry(idx, payload)

                elif topic.startswith("usv/enemy/"):
                    parts = topic.split("/")
                    if parts[2] == "all":
                        self._handle_all_enemies(payload)
                    else:
                        idx = int(parts[2])
                        self._handle_enemy_telemetry(idx, payload)

                elif topic.startswith("usv/mothership"):
                    self._handle_mothership_telemetry(payload)

                elif "telemetry" in topic:
                    # 단일 선박 모드 (기존)
                    self._handle_single_telemetry(payload)

            except Exception as e:
                self.get_logger().error(f"MQTT 메시지 처리 오류: {e}")

        def _handle_ally_telemetry(self, idx: int, data: Dict[str, Any]):
            """아군 텔레메트리 처리."""
            if idx not in self.ally_states:
                self.ally_states[idx] = VesselState()
            state = self.ally_states[idx]

            # 센서 데이터 업데이트
            if "sensor" in data:
                # 개별 센서 형식
                sensor = data["sensor"]
                value = data["value"]
                if sensor == "lat":
                    state.lat = value
                elif sensor == "lon":
                    state.lon = value
                elif sensor == "heading":
                    state.heading = value
                elif sensor == "speed_ms":
                    state.speed = value
                elif sensor == "battery":
                    state.battery = value
            else:
                # 전체 상태 형식
                raw_lat = data.get("lat", state.lat)
                raw_lon = data.get("lon", state.lon)

                # 시뮬레이션 좌표인 경우 GPS로 변환
                if is_sim_coordinate(raw_lat, raw_lon):
                    # (lat, lon) 필드에 (x, z) 시뮬레이션 좌표가 들어온 경우
                    # raw_lat = x (East), raw_lon = z (North)
                    state.lat, state.lon = sim_to_gps(raw_lat, raw_lon)
                else:
                    state.lat = raw_lat
                    state.lon = raw_lon

                if "heading" in data:
                    state.heading = data["heading"]
                if "speed_ms" in data:
                    state.speed = data["speed_ms"]

        def _handle_enemy_telemetry(self, idx: int, data: Dict[str, Any]):
            """적군 텔레메트리 처리."""
            if idx not in self.enemy_states:
                self.enemy_states[idx] = VesselState()
            state = self.enemy_states[idx]

            raw_lat = data.get("lat", state.lat)
            raw_lon = data.get("lon", state.lon)

            # 시뮬레이션 좌표인 경우 GPS로 변환
            if is_sim_coordinate(raw_lat, raw_lon):
                state.lat, state.lon = sim_to_gps(raw_lat, raw_lon)
            else:
                state.lat = raw_lat
                state.lon = raw_lon

            if "heading" in data:
                state.heading = data["heading"]

        def _handle_all_allies(self, data: Dict[str, Any]):
            """전체 아군 일괄 처리."""
            if "allies" in data:
                for ally in data["allies"]:
                    idx = ally.get("id", ally.get("idx", 0))
                    self._handle_ally_telemetry(idx, ally)

        def _handle_all_enemies(self, data: Dict[str, Any]):
            """전체 적군 일괄 처리."""
            if "enemies" in data:
                for enemy in data["enemies"]:
                    idx = enemy.get("id", enemy.get("idx", 0))
                    self._handle_enemy_telemetry(idx, enemy)

        def _handle_mothership_telemetry(self, data: Dict[str, Any]):
            """모선 텔레메트리 처리."""
            raw_lat = data.get("lat", data.get("x", self.mothership_state.lat))
            raw_lon = data.get("lon", data.get("z", self.mothership_state.lon))

            # 시뮬레이션 좌표인 경우 GPS로 변환
            if is_sim_coordinate(raw_lat, raw_lon):
                self.mothership_state.lat, self.mothership_state.lon = sim_to_gps(raw_lat, raw_lon)
            else:
                self.mothership_state.lat = raw_lat
                self.mothership_state.lon = raw_lon

        def _handle_single_telemetry(self, data: Dict[str, Any]):
            """단일 선박 텔레메트리 (기존 모드)."""
            # ally_0으로 처리
            self._handle_ally_telemetry(0, data)

        def _publish_to_ros2(self):
            """ROS2로 텔레메트리 발행."""
            now = self.get_clock().now().to_msg()

            # 아군 발행
            for idx, state in self.ally_states.items():
                if idx < len(self.ally_fix_pubs):
                    # GPS
                    fix_msg = telemetry_to_navsatfix(
                        state.lat, state.lon,
                        stamp=now,
                        frame_id=f"ally_{idx}"
                    )
                    self.ally_fix_pubs[idx].publish(fix_msg)

                    # IMU
                    imu_msg = telemetry_to_imu(
                        state.heading,
                        stamp=now,
                        frame_id=f"ally_{idx}"
                    )
                    self.ally_imu_pubs[idx].publish(imu_msg)

            # 적군 발행
            for idx, state in self.enemy_states.items():
                if idx < len(self.enemy_fix_pubs):
                    fix_msg = telemetry_to_navsatfix(
                        state.lat, state.lon,
                        stamp=now,
                        frame_id=f"enemy_{idx}"
                    )
                    self.enemy_fix_pubs[idx].publish(fix_msg)

            # 모선 발행
            mothership_fix = telemetry_to_navsatfix(
                self.mothership_state.lat, self.mothership_state.lon,
                stamp=now,
                frame_id="mothership"
            )
            self.mothership_fix_pub.publish(mothership_fix)

        def _on_waypoints(self, msg: Path, ally_idx: int):
            """ROS2 웨이포인트 → MQTT 명령."""
            waypoints = path_to_waypoints(msg)
            wp_str = waypoints_to_mqtt_string(waypoints)

            # MQTT로 웨이포인트 명령 전송
            command_topic = f"usv/ally/{ally_idx}/commands"
            self.mqtt.publish(command_topic, {
                "channel": "waypoints",
                "value": wp_str,
            })

            self.get_logger().info(
                f"Ally {ally_idx}: {len(waypoints)}개 웨이포인트 전송"
            )

        def destroy_node(self):
            """노드 정리."""
            self.mqtt.disconnect()
            super().destroy_node()


def main():
    """브릿지 노드 실행."""
    if not ROS2_AVAILABLE:
        print("ROS2 not available")
        return

    rclpy.init()

    config = BridgeConfig.from_env()
    node = MqttRos2BridgeNode(config)

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
