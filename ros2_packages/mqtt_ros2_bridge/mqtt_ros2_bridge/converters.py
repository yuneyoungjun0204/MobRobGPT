"""
mqtt_ros2_bridge/converters.py — 메시지 변환 유틸리티
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

# ROS2 메시지 (조건부 import)
try:
    from sensor_msgs.msg import NavSatFix, Imu, BatteryState
    from geometry_msgs.msg import Quaternion, TwistStamped, PoseStamped
    from nav_msgs.msg import Path
    from std_msgs.msg import Header, Float32, Bool, UInt8
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def heading_nav_to_enu(heading_nav: float) -> float:
    """Navigation heading (0=North, CW+) → ENU yaw (0=East, CCW+).

    Args:
        heading_nav: 항법 방위각 (도, 0=북, 시계방향+)

    Returns:
        ENU yaw (도, 0=동, 반시계+)
    """
    return (90.0 - heading_nav) % 360.0


def heading_enu_to_nav(yaw_enu: float) -> float:
    """ENU yaw (0=East, CCW+) → Navigation heading (0=North, CW+).

    Args:
        yaw_enu: ENU yaw (도, 0=동, 반시계+)

    Returns:
        항법 방위각 (도, 0=북, 시계방향+)
    """
    return (90.0 - yaw_enu) % 360.0


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Euler angles (rad) → Quaternion (x, y, z, w)."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Quaternion → Euler angles (roll, pitch, yaw) in radians."""
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def knots_to_mps(knots: float) -> float:
    """Knots → m/s."""
    return knots * 0.514444


def mps_to_knots(mps: float) -> float:
    """m/s → knots."""
    return mps * 1.943844


if ROS2_AVAILABLE:
    def telemetry_to_navsatfix(
        lat: float,
        lon: float,
        stamp=None,
        frame_id: str = "gps"
    ) -> NavSatFix:
        """텔레메트리 → NavSatFix 메시지."""
        msg = NavSatFix()
        if stamp:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0
        return msg

    def telemetry_to_imu(
        heading_nav: float,
        stamp=None,
        frame_id: str = "imu"
    ) -> Imu:
        """텔레메트리 heading → Imu 메시지 (orientation만)."""
        msg = Imu()
        if stamp:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        # nav heading → ENU yaw → quaternion
        yaw_enu = heading_nav_to_enu(heading_nav)
        yaw_rad = math.radians(yaw_enu)
        x, y, z, w = euler_to_quaternion(0, 0, yaw_rad)

        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w

        return msg

    def navsatfix_to_telemetry(msg: NavSatFix) -> Dict[str, float]:
        """NavSatFix → 텔레메트리 dict."""
        return {
            "lat": msg.latitude,
            "lon": msg.longitude,
        }

    def imu_to_heading_nav(msg: Imu) -> float:
        """Imu orientation → navigation heading (도)."""
        q = msg.orientation
        _, _, yaw_rad = quaternion_to_euler(q.x, q.y, q.z, q.w)
        yaw_enu = math.degrees(yaw_rad)
        return heading_enu_to_nav(yaw_enu)

    def waypoints_to_path(
        waypoints: list,  # [(lat, lon), ...]
        stamp=None,
        frame_id: str = "wgs84"
    ) -> Path:
        """웨이포인트 리스트 → Path 메시지."""
        path = Path()
        if stamp:
            path.header.stamp = stamp
        path.header.frame_id = frame_id

        for lat, lon in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = lon
            pose.pose.position.y = lat
            pose.pose.position.z = 0.0
            path.poses.append(pose)

        return path

    def path_to_waypoints(msg: Path) -> list:
        """Path 메시지 → 웨이포인트 리스트 [(lat, lon), ...]."""
        waypoints = []
        for pose in msg.poses:
            lat = pose.pose.position.y
            lon = pose.pose.position.x
            waypoints.append((lat, lon))
        return waypoints

    def waypoints_to_mqtt_string(waypoints: list) -> str:
        """웨이포인트 리스트 → MQTT 문자열 "lat,lon;lat,lon;..."."""
        return ";".join(f"{lat},{lon}" for lat, lon in waypoints)

    def mqtt_string_to_waypoints(s: str) -> list:
        """MQTT 문자열 → 웨이포인트 리스트."""
        if not s:
            return []
        pairs = s.split(";")
        waypoints = []
        for pair in pairs:
            parts = pair.split(",")
            if len(parts) == 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    waypoints.append((lat, lon))
                except ValueError:
                    continue
        return waypoints


# Non-ROS2 버전 (테스트용)
def telemetry_dict_to_ros2_data(telemetry: Dict[str, float]) -> Dict[str, Any]:
    """텔레메트리 dict → ROS2 호환 데이터 dict."""
    data = {}

    if "lat" in telemetry and "lon" in telemetry:
        data["gps"] = {
            "latitude": telemetry["lat"],
            "longitude": telemetry["lon"],
        }

    if "heading" in telemetry:
        yaw_enu = heading_nav_to_enu(telemetry["heading"])
        data["heading_enu_deg"] = yaw_enu

    if "speed_ms" in telemetry:
        data["speed_mps"] = telemetry["speed_ms"]
    elif "sog" in telemetry:
        data["speed_mps"] = knots_to_mps(telemetry["sog"])

    if "battery" in telemetry:
        data["battery_pct"] = telemetry["battery"]

    if "autopilot" in telemetry:
        data["autopilot"] = bool(telemetry["autopilot"])

    if "mode" in telemetry:
        data["mode"] = int(telemetry["mode"])

    return data
