"""
commander/ros2_sensor_bridge.py — ROS2 센서 데이터 브릿지

/enemy_X/fix, /ally_X/fix, /ally_X/imu 토픽을 구독하여
실시간 센서 데이터를 수집하고, /ally_X/waypoints로 명령을 발행한다.

토픽 구조:
  구독:
    /enemy_0/fix ~ /enemy_9/fix: sensor_msgs/NavSatFix (적 GPS)
    /ally_0/fix ~ /ally_2/fix: sensor_msgs/NavSatFix (아군 GPS)
    /ally_0/imu ~ /ally_2/imu: sensor_msgs/Imu (아군 방위)
  발행:
    /ally_0/waypoints ~ /ally_2/waypoints: nav_msgs/Path (경유점)
"""

import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable

from .geo_bridge import GeoBridge

# ROS2 임포트 (없으면 스텁 모드)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import NavSatFix, Imu
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[ros2_sensor_bridge] ROS2 not available")


def quat_to_euler(x, y, z, w):
    """Quaternion → Euler (roll, pitch, yaw)."""
    import math
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


@dataclass
class SensorState:
    """센서 상태 저장소."""
    n_allies: int = 3
    n_enemies: int = 10

    # 아군 상태
    ally_geo: np.ndarray = field(default=None)  # [P, 2] (lat, lon)
    ally_hdg_enu: np.ndarray = field(default=None)  # [P] yaw in degrees
    ally_alive: np.ndarray = field(default=None)  # [P] bool
    ally_timestamps: np.ndarray = field(default=None)  # [P] 최근 업데이트 시간

    # 적 상태
    enemy_geo: np.ndarray = field(default=None)  # [M, 2] (lat, lon)
    enemy_hdg_enu: np.ndarray = field(default=None)  # [M] yaw (추정)
    enemy_alive: np.ndarray = field(default=None)  # [M] bool
    enemy_timestamps: np.ndarray = field(default=None)  # [M] 최근 업데이트 시간

    # 모선 (아군 무게중심 또는 고정)
    mothership_geo: Optional[Tuple[float, float]] = None

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        P, M = self.n_allies, self.n_enemies
        self.ally_geo = np.zeros((P, 2))
        self.ally_hdg_enu = np.zeros(P)
        self.ally_alive = np.zeros(P, dtype=bool)
        self.ally_timestamps = np.zeros(P)

        self.enemy_geo = np.zeros((M, 2))
        self.enemy_hdg_enu = np.zeros(M)
        self.enemy_alive = np.zeros(M, dtype=bool)
        self.enemy_timestamps = np.zeros(M)

    def update_ally(self, idx: int, lat: float, lon: float,
                    yaw_enu_deg: Optional[float] = None):
        """아군 위치/방위 업데이트."""
        with self._lock:
            self.ally_geo[idx] = [lat, lon]
            if yaw_enu_deg is not None:
                self.ally_hdg_enu[idx] = yaw_enu_deg
            self.ally_alive[idx] = True
            self.ally_timestamps[idx] = time.time()

    def update_enemy(self, idx: int, lat: float, lon: float,
                     yaw_enu_deg: float = 0.0):
        """적 위치 업데이트."""
        with self._lock:
            # 이전 위치로 방위 추정
            if self.enemy_alive[idx]:
                old = self.enemy_geo[idx]
                dx = (lon - old[1]) * 111320.0 * np.cos(np.deg2rad(lat))
                dy = (lat - old[0]) * 111320.0
                if np.hypot(dx, dy) > 1.0:  # 1m 이상 이동시
                    yaw_enu_deg = np.degrees(np.arctan2(dy, dx))
                    self.enemy_hdg_enu[idx] = yaw_enu_deg

            self.enemy_geo[idx] = [lat, lon]
            self.enemy_alive[idx] = True
            self.enemy_timestamps[idx] = time.time()

    def update_mothership(self, lat: float, lon: float):
        """모선 위치 설정."""
        with self._lock:
            self.mothership_geo = (lat, lon)

    def check_stale(self, timeout: float = 5.0) -> dict:
        """만료된 센서 확인."""
        now = time.time()
        with self._lock:
            stale_allies = np.where(
                self.ally_alive & ((now - self.ally_timestamps) > timeout)
            )[0].tolist()
            stale_enemies = np.where(
                self.enemy_alive & ((now - self.enemy_timestamps) > timeout)
            )[0].tolist()
        return {"allies": stale_allies, "enemies": stale_enemies}

    def snapshot(self) -> dict:
        """현재 상태 스냅샷."""
        with self._lock:
            return {
                "ally_geo": self.ally_geo.copy(),
                "ally_hdg_enu": self.ally_hdg_enu.copy(),
                "ally_alive": self.ally_alive.copy(),
                "enemy_geo": self.enemy_geo.copy(),
                "enemy_hdg_enu": self.enemy_hdg_enu.copy(),
                "enemy_alive": self.enemy_alive.copy(),
                "mothership_geo": self.mothership_geo,
            }


class ROS2SensorBridge:
    """ROS2 센서 브릿지 (non-ROS2 테스트 지원)."""

    def __init__(
        self,
        n_allies: int = 3,
        n_enemies: int = 10,
        world_size: float = 12600.0,
        on_state_update: Optional[Callable] = None,
    ):
        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self.world_size = world_size
        self.on_state_update = on_state_update

        # 센서 상태
        self.state = SensorState(n_allies=n_allies, n_enemies=n_enemies)

        # 좌표 변환
        self.geo_bridge = GeoBridge(
            world_size=world_size,
            target_sim_radius=5450.0
        )

        self._fitted = False
        self._node = None
        self._executor = None
        self._spin_thread = None

    def fit_geo_bridge(self) -> bool:
        """좌표 변환 초기화 (센서 데이터로)."""
        snap = self.state.snapshot()

        # 유효한 데이터 확인
        valid_allies = snap["ally_alive"]
        valid_enemies = snap["enemy_alive"]

        if not valid_allies.any():
            print("[ROS2Bridge] 아군 데이터 없음, fit 대기")
            return False

        allies_geo = snap["ally_geo"][valid_allies]
        enemies_geo = snap["enemy_geo"][valid_enemies] if valid_enemies.any() else np.zeros((0, 2))

        mothership = snap["mothership_geo"]
        if mothership is None:
            # 아군 무게중심을 모선으로
            mothership = (allies_geo[:, 0].mean(), allies_geo[:, 1].mean())

        self.geo_bridge.fit(allies_geo, enemies_geo, mothership)
        self._fitted = True
        print(f"[ROS2Bridge] geo_bridge fit: scale={self.geo_bridge.scale:.4f}")
        return True

    def get_sim_state(self) -> Optional[dict]:
        """센서 데이터 → 시뮬 좌표 변환."""
        if not self._fitted:
            if not self.fit_geo_bridge():
                return None

        snap = self.state.snapshot()

        # GPS → 시뮬 좌표
        a_sim = self.geo_bridge.to_sim(snap["ally_geo"][:, 0], snap["ally_geo"][:, 1])
        a_hdg = self.geo_bridge.hdg_to_sim(snap["ally_hdg_enu"])

        e_sim = self.geo_bridge.to_sim(snap["enemy_geo"][:, 0], snap["enemy_geo"][:, 1])
        e_hdg = self.geo_bridge.hdg_to_sim(snap["enemy_hdg_enu"])

        return {
            "ally_pos": a_sim,
            "ally_hdg": a_hdg,
            "ally_alive": snap["ally_alive"],
            "enemy_pos": e_sim,
            "enemy_hdg": e_hdg,
            "enemy_alive": snap["enemy_alive"],
            "center": self.geo_bridge.sim_center,
        }

    def publish_waypoints(self, routes_sim: np.ndarray, net_mask: np.ndarray = None):
        """시뮬 좌표 경로 → GPS 변환 후 발행."""
        if not self._fitted:
            return

        # 시뮬 → GPS 변환
        routes_geo = []
        for p in range(min(self.n_allies, routes_sim.shape[0])):
            wp_geo = []
            for k in range(routes_sim.shape[1]):
                lat, lon = self.geo_bridge.to_geo(routes_sim[p, k])
                is_net = bool(net_mask[p, k]) if net_mask is not None else False
                wp_geo.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "deploy_net": is_net,
                })
            routes_geo.append(wp_geo)

        # ROS2 발행 (노드가 있을 때만)
        if self._node is not None and hasattr(self._node, "publish_waypoints"):
            self._node.publish_waypoints(routes_geo)

        return routes_geo


if ROS2_AVAILABLE:
    class ROS2SensorNode(Node):
        """ROS2 센서 노드."""

        def __init__(self, bridge: ROS2SensorBridge):
            super().__init__("mobrob_sensor_bridge")
            self.bridge = bridge

            # 콜백 그룹
            self.cb_group = ReentrantCallbackGroup()

            # QoS (센서용)
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            # 구독: 아군 GPS/IMU
            self.ally_fix_subs = []
            self.ally_imu_subs = []
            for i in range(bridge.n_allies):
                fix_sub = self.create_subscription(
                    NavSatFix,
                    f"/ally_{i}/fix",
                    lambda msg, idx=i: self._ally_fix_cb(msg, idx),
                    sensor_qos,
                    callback_group=self.cb_group
                )
                self.ally_fix_subs.append(fix_sub)

                imu_sub = self.create_subscription(
                    Imu,
                    f"/ally_{i}/imu",
                    lambda msg, idx=i: self._ally_imu_cb(msg, idx),
                    sensor_qos,
                    callback_group=self.cb_group
                )
                self.ally_imu_subs.append(imu_sub)

            # 구독: 적 GPS
            self.enemy_fix_subs = []
            for i in range(bridge.n_enemies):
                fix_sub = self.create_subscription(
                    NavSatFix,
                    f"/enemy_{i}/fix",
                    lambda msg, idx=i: self._enemy_fix_cb(msg, idx),
                    sensor_qos,
                    callback_group=self.cb_group
                )
                self.enemy_fix_subs.append(fix_sub)

            # 발행: 아군 WP
            self.wp_pubs = []
            for i in range(bridge.n_allies):
                pub = self.create_publisher(Path, f"/ally_{i}/waypoints", 10)
                self.wp_pubs.append(pub)

            self.get_logger().info(
                f"[ROS2SensorNode] Started: {bridge.n_allies} allies, {bridge.n_enemies} enemies"
            )

        def _ally_fix_cb(self, msg: NavSatFix, idx: int):
            self.bridge.state.update_ally(idx, lat=msg.latitude, lon=msg.longitude)
            if self.bridge.on_state_update:
                self.bridge.on_state_update()

        def _ally_imu_cb(self, msg: Imu, idx: int):
            q = msg.orientation
            _, _, yaw = quat_to_euler(q.x, q.y, q.z, q.w)
            yaw_deg = np.degrees(yaw)

            # 위치 유지, 방위만 업데이트
            snap = self.bridge.state.snapshot()
            if self.bridge.state.ally_alive[idx]:
                lat, lon = snap["ally_geo"][idx]
                self.bridge.state.update_ally(idx, lat=lat, lon=lon, yaw_enu_deg=yaw_deg)

        def _enemy_fix_cb(self, msg: NavSatFix, idx: int):
            self.bridge.state.update_enemy(idx, lat=msg.latitude, lon=msg.longitude)
            if self.bridge.on_state_update:
                self.bridge.on_state_update()

        def publish_waypoints(self, routes_geo: List[List[dict]]):
            """GPS 경로 발행."""
            for i, wps in enumerate(routes_geo):
                if i >= len(self.wp_pubs):
                    break

                path = Path()
                path.header = Header()
                path.header.stamp = self.get_clock().now().to_msg()
                path.header.frame_id = "wgs84"

                for wp in wps:
                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position.x = wp["lon"]
                    pose.pose.position.y = wp["lat"]
                    pose.pose.position.z = 1.0 if wp.get("deploy_net", False) else 0.0
                    path.poses.append(pose)

                self.wp_pubs[i].publish(path)


def create_ros2_bridge(
    n_allies: int = 3,
    n_enemies: int = 10,
    world_size: float = 12600.0,
    on_state_update: Optional[Callable] = None,
) -> ROS2SensorBridge:
    """ROS2 브릿지 생성 및 시작."""
    bridge = ROS2SensorBridge(
        n_allies=n_allies,
        n_enemies=n_enemies,
        world_size=world_size,
        on_state_update=on_state_update,
    )

    if ROS2_AVAILABLE:
        # ROS2 초기화
        if not rclpy.ok():
            rclpy.init()

        node = ROS2SensorNode(bridge)
        bridge._node = node

        # 별도 스레드에서 spin
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        bridge._executor = executor

        def spin_thread():
            try:
                executor.spin()
            except Exception as e:
                print(f"[ROS2Bridge] Spin error: {e}")

        bridge._spin_thread = threading.Thread(target=spin_thread, daemon=True)
        bridge._spin_thread.start()
        print("[ROS2Bridge] ROS2 node started in background thread")
    else:
        print("[ROS2Bridge] Running without ROS2 (test mode)")

    return bridge


def shutdown_ros2_bridge(bridge: ROS2SensorBridge):
    """ROS2 브릿지 종료."""
    if bridge._executor is not None:
        bridge._executor.shutdown()
    if bridge._node is not None:
        bridge._node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
