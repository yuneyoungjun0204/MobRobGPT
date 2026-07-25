"""
mqtt_ros2_bridge/config.py — 브릿지 설정
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class BridgeConfig:
    """MQTT-ROS2 브릿지 설정."""

    # MQTT 브로커
    mqtt_host: str = "localhost"
    mqtt_port: int = 9001
    mqtt_transport: str = "websockets"  # "tcp" or "websockets"
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None

    # 디바이스 설정
    device_token: str = "sim-usv-1"
    device_id: str = "1"

    # 다중 선박 설정
    n_allies: int = 3
    n_enemies: int = 10

    # 브릿지 동작
    telemetry_rate: float = 10.0  # Hz (ROS2 → MQTT)
    command_rate: float = 10.0    # Hz (MQTT → ROS2)

    # 토픽 접두사
    ros2_ally_prefix: str = "/ally_"
    ros2_enemy_prefix: str = "/enemy_"
    mqtt_telemetry_topic: str = "devices/{token}/telemetry"
    mqtt_command_topic: str = "devices/{id}/commands"

    # 좌표 변환
    heading_convention: str = "nav"  # "nav" (0=North, CW+) or "enu" (0=East, CCW+)

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        """환경변수에서 설정 로드."""
        return cls(
            mqtt_host=os.getenv("MQTT_HOST", "localhost"),
            mqtt_port=int(os.getenv("MQTT_PORT", "9001")),
            mqtt_transport=os.getenv("MQTT_TRANSPORT", "websockets"),
            mqtt_username=os.getenv("MQTT_USERNAME"),
            mqtt_password=os.getenv("MQTT_PASSWORD"),
            device_token=os.getenv("DEVICE_TOKEN", "sim-usv-1"),
            device_id=os.getenv("DEVICE_ID", "1"),
            n_allies=int(os.getenv("N_ALLIES", "3")),
            n_enemies=int(os.getenv("N_ENEMIES", "10")),
            telemetry_rate=float(os.getenv("TELEMETRY_RATE", "10.0")),
        )


# 센서 이름 → ROS2 메시지 타입 매핑
SENSOR_MAPPING = {
    "lat": {"type": "NavSatFix", "field": "latitude"},
    "lon": {"type": "NavSatFix", "field": "longitude"},
    "heading": {"type": "Imu", "field": "orientation"},
    "speed_ms": {"type": "TwistStamped", "field": "twist.linear.x"},
    "sog": {"type": "TwistStamped", "field": "twist.linear.x"},  # knots → m/s
    "battery": {"type": "BatteryState", "field": "percentage"},
    "throttle": {"type": "Float32", "field": "data"},
    "steer": {"type": "Float32", "field": "data"},
    "autopilot": {"type": "Bool", "field": "data"},
    "mode": {"type": "UInt8", "field": "data"},
}

# 명령 채널 → ROS2 토픽 매핑
COMMAND_MAPPING = {
    "throttle": "/cmd_throttle",
    "steer": "/cmd_steer",
    "waypoints": "/waypoints",
    "waypoint": "/waypoint",
    "autopilot": "/autopilot",
    "stop": "/emergency_stop",
    "homing": "/return_home",
}
