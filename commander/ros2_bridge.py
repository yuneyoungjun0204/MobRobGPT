"""ROS2 센서 브릿지 — GPS/IMU → SIM 좌표 변환 + 웨이포인트 발행.

기존 CommandedCellEnv를 전혀 수정하지 않고, 센서 데이터만 주입/추출.
"""
from __future__ import annotations

import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional, Callable

# ROS2 imports (optional)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import NavSatFix, Imu
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Quaternion
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object


def euler_from_quaternion(q):
    """Quaternion (x, y, z, w) → Euler angles (roll, pitch, yaw)."""
    x, y, z, w = q
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

# GPS → meters 변환 상수
DEG_TO_M_LAT = 111320.0  # 위도 1도 ≈ 111.32km


@dataclass
class SensorState:
    """센서 상태."""
    ally_pos: np.ndarray      # [P, 2] SIM 좌표 (meters)
    ally_hdg: np.ndarray      # [P] 방위각 (degrees, 0=North, CW+)
    ally_alive: np.ndarray    # [P] bool
    enemy_pos: np.ndarray     # [M, 2]
    enemy_hdg: np.ndarray     # [M]
    enemy_alive: np.ndarray   # [M] bool
    center: np.ndarray        # [2] 모선/원점


class ROS2SensorBridge:
    """ROS2 센서 → SIM 좌표 변환 브릿지.

    GPS (lat, lon) → ENU (east, north) → SIM (x=east, y=north)
    IMU quaternion → yaw → 항법 방위각 (0=North, CW+)
    """

    def __init__(
        self,
        world_size: float = 33.0,
        origin_lat: float = 34.625,
        origin_lon: float = 128.52,
        n_allies: int = 3,
        n_enemies: int = 10,
        on_update: Optional[Callable] = None,
        src_world_size: float = 6000.0,  # 발행자 좌표계의 world 크기 (스케일 변환용)
    ):
        self.world_size = world_size
        self.src_world_size = src_world_size
        self.scale = world_size / src_world_size  # 좌표 스케일 팩터
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self._on_update = on_update

        # 상태 배열
        self._ally_pos = np.zeros((n_allies, 2))
        self._ally_hdg = np.zeros(n_allies)
        self._ally_alive = np.zeros(n_allies, dtype=bool)
        self._ally_gps_valid = np.zeros(n_allies, dtype=bool)

        self._enemy_pos = np.zeros((n_enemies, 2))
        self._enemy_hdg = np.zeros(n_enemies)
        self._enemy_alive = np.zeros(n_enemies, dtype=bool)

        self._center = np.array([world_size / 2, world_size / 2])
        self._mother_valid = False

        # 원점 자동 보정 (첫 수신 모선 GPS로 원점 설정)
        self._origin_calibrated = False

        # ROS2
        self._node: Optional[Node] = None
        self._wp_pubs = []
        self._running = False
        self._lock = threading.Lock()

        # 웨이포인트 변경 감지용 (매 프레임 발행 방지)
        self._last_routes: Optional[np.ndarray] = None
        self._last_masks: Optional[np.ndarray] = None

    def _gps_to_sim(self, lat: float, lon: float) -> np.ndarray:
        """GPS (lat, lon) → SIM 좌표 (meters, 원점 기준)."""
        d_lat = lat - self.origin_lat
        d_lon = lon - self.origin_lon
        # ENU: east = d_lon * cos(lat), north = d_lat
        north = d_lat * DEG_TO_M_LAT
        east = d_lon * DEG_TO_M_LAT * np.cos(np.radians(self.origin_lat))
        # SIM 좌표: 원점을 world 중심으로 이동
        half = self.world_size / 2
        return np.array([east + half, north + half])

    def _sim_to_gps(self, x: float, y: float) -> tuple:
        """SIM 좌표 (meters) → GPS (lat, lon)."""
        half = self.world_size / 2
        east = x - half   # world 중심 기준 동쪽 오프셋 (meters)
        north = y - half  # world 중심 기준 북쪽 오프셋 (meters)
        d_lat = north / DEG_TO_M_LAT
        d_lon = east / (DEG_TO_M_LAT * np.cos(np.radians(self.origin_lat)))
        lat = self.origin_lat + d_lat
        lon = self.origin_lon + d_lon
        return lat, lon

    def _imu_to_heading(self, qx, qy, qz, qw) -> float:
        """IMU quaternion → 항법 방위각 (0=North, CW+)."""
        _, _, yaw = euler_from_quaternion([qx, qy, qz, qw])
        yaw_deg = np.degrees(yaw)
        # ENU (0=East, CCW+) → NAV (0=North, CW+)
        hdg = (90.0 - yaw_deg) % 360.0
        return hdg

    # ─────────────────────────────────────────────────────────────
    # ROS2 콜백
    # ─────────────────────────────────────────────────────────────

    def _is_valid_gps(self, lat: float, lon: float) -> bool:
        """GPS 좌표 유효성 검사 (NaN, 무한대, 범위 이탈 거부)."""
        if not np.isfinite(lat) or not np.isfinite(lon):
            return False
        if lat < -90 or lat > 90:
            return False
        if lon < -180 or lon > 180:
            return False
        return True

    def _is_pos_in_world(self, pos: np.ndarray) -> bool:
        """변환된 SIM 좌표가 world 범위 내인지 검사.

        world 밖의 좌표는 무효한 GPS(멀리 떨어진 위치)로 간주.
        약간의 여유(-10% ~ 110%)를 두어 경계 근처 유효 데이터 허용.
        """
        margin = self.world_size * 0.1
        lo = -margin
        hi = self.world_size + margin
        return (lo <= pos[0] <= hi) and (lo <= pos[1] <= hi)

    def _auto_calibrate_origin(self, lat: float, lon: float):
        """첫 번째 수신 GPS로 원점 자동 보정.

        실제 GPS 위치를 world 중심으로 설정.
        """
        if self._origin_calibrated:
            return
        self.origin_lat = lat
        self.origin_lon = lon
        self._origin_calibrated = True
        print(f"[ROS2Bridge] GPS origin auto-calibrated: lat={lat:.6f}, lon={lon:.6f}")

    def _on_ally_gps(self, idx: int, msg: 'NavSatFix'):
        # GPS lat/lon → 로컬 미터 변환
        lat, lon = msg.latitude, msg.longitude  # 정상 순서
        if not self._is_valid_gps(lat, lon):
            print(f"[ROS2] ally_{idx} REJECTED: lat={lat:.2f}, lon={lon:.2f} (invalid range)")
            return
        if not self._origin_calibrated:
            print(f"[ROS2] ally_{idx} waiting for origin calibration...")
            return

        # 원점(모선) 기준 로컬 미터 변환
        pos = self._gps_to_sim(lat, lon)
        if not self._is_pos_in_world(pos):
            return
        with self._lock:
            if not self._ally_gps_valid[idx]:  # 첫 수신만 로그
                print(f"[ROS2] ally_{idx} GPS→m: lat={lat:.6f}, lon={lon:.6f} → ({pos[0]:.2f}, {pos[1]:.2f})")
            self._ally_pos[idx] = pos
            self._ally_gps_valid[idx] = True
            self._ally_alive[idx] = True
        if self._on_update:
            self._on_update()

    def _on_ally_imu(self, idx: int, msg: 'Imu'):
        q = msg.orientation
        hdg = self._imu_to_heading(q.x, q.y, q.z, q.w)
        with self._lock:
            self._ally_hdg[idx] = hdg

    def _on_enemy_gps(self, idx: int, msg: 'NavSatFix'):
        # GPS lat/lon → 로컬 미터 변환
        lat, lon = msg.latitude, msg.longitude  # 정상 순서
        if not self._is_valid_gps(lat, lon):
            return
        if not self._origin_calibrated:
            return

        # 원점(모선) 기준 로컬 미터 변환
        pos = self._gps_to_sim(lat, lon)
        if not self._is_pos_in_world(pos):
            return
        with self._lock:
            if not self._enemy_alive[idx]:  # 첫 수신만 로그
                print(f"[ROS2] enemy_{idx} GPS→m: lat={lat:.6f}, lon={lon:.6f} → ({pos[0]:.2f}, {pos[1]:.2f})")
            self._enemy_pos[idx] = pos
            self._enemy_alive[idx] = True
            # 적 방위: 모선 방향으로 추정
            dx = self._center[0] - pos[0]
            dy = self._center[1] - pos[1]
            self._enemy_hdg[idx] = (np.degrees(np.arctan2(dx, dy))) % 360

    def _on_mother_gps(self, msg: 'NavSatFix'):
        # GPS lat/lon으로 원점 보정
        lat, lon = msg.latitude, msg.longitude  # 정상 순서
        if not self._is_valid_gps(lat, lon):
            print(f"[ROS2] mothership REJECTED: lat={lat:.2f} (valid: -90~90), lon={lon:.2f} (valid: -180~180)")
            return

        # 모선 GPS로 원점 보정 (모선 = world 중심)
        if not self._origin_calibrated:
            self.origin_lat = lat
            self.origin_lon = lon
            self._origin_calibrated = True
            print(f"[ROS2Bridge] Origin calibrated: lat={lat:.6f}, lon={lon:.6f}")

        # 모선은 항상 world 중심에 위치
        with self._lock:
            self._center = np.array([self.world_size / 2, self.world_size / 2])
            self._mother_valid = True

    # ─────────────────────────────────────────────────────────────
    # 시작/종료
    # ─────────────────────────────────────────────────────────────

    def start(self) -> 'ROS2SensorBridge':
        """ROS2 노드 시작."""
        if not ROS2_AVAILABLE:
            print("[ROS2Bridge] rclpy not available")
            return self

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node('sensor_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 아군 GPS/IMU 구독
        for i in range(self.n_allies):
            self._node.create_subscription(
                NavSatFix, f'/ally_{i}/fix',
                lambda msg, idx=i: self._on_ally_gps(idx, msg), qos)
            self._node.create_subscription(
                Imu, f'/ally_{i}/imu',
                lambda msg, idx=i: self._on_ally_imu(idx, msg), qos)

        # 적군 GPS 구독
        for i in range(self.n_enemies):
            self._node.create_subscription(
                NavSatFix, f'/enemy_{i}/fix',
                lambda msg, idx=i: self._on_enemy_gps(idx, msg), qos)

        # 모선 GPS 구독 (optional)
        self._node.create_subscription(
            NavSatFix, '/mothership/fix',
            self._on_mother_gps, qos)

        # 웨이포인트 발행자
        self._wp_pubs = [
            self._node.create_publisher(Path, f'/ally_{i}/waypoints', 10)
            for i in range(self.n_allies)
        ]

        # 선박 위치 발행자
        self._ally_pub = self._node.create_publisher(PoseArray, '/ships/allies', 10)
        self._enemy_pub = self._node.create_publisher(PoseArray, '/ships/enemies', 10)
        self._mother_pub = self._node.create_publisher(PoseStamped, '/ships/mothership', 10)

        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        print(f"[ROS2Bridge] Started: allies={self.n_allies}, enemies={self.n_enemies}")
        return self

    def shutdown(self):
        """종료."""
        self._running = False
        if self._node:
            self._node.destroy_node()
            self._node = None
        print("[ROS2Bridge] Shutdown")

    def _spin_loop(self):
        try:
            while self._running and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.01)
        except Exception as e:
            print(f"[ROS2Bridge] Spin error: {e}")

    # ─────────────────────────────────────────────────────────────
    # 상태 조회/발행
    # ─────────────────────────────────────────────────────────────

    def get_state(self) -> SensorState:
        """현재 센서 상태 반환."""
        with self._lock:
            return SensorState(
                ally_pos=self._ally_pos.copy(),
                ally_hdg=self._ally_hdg.copy(),
                ally_alive=self._ally_alive.copy(),
                enemy_pos=self._enemy_pos.copy(),
                enemy_hdg=self._enemy_hdg.copy(),
                enemy_alive=self._enemy_alive.copy(),
                center=self._center.copy(),
            )

    def publish_waypoints(self, routes: np.ndarray, net_mask: np.ndarray):
        """웨이포인트 발행 (world 좌표). routes: [P, K, 2], net_mask: [P, K].

        변경된 경우에만 발행하여 UI 깜빡임 방지.
        GPS 변환 없이 world 좌표 직접 전송 (usv_bridge에서 y→z 변환).
        """
        if not self._node or not self._running:
            return

        # 변경 감지: 경로가 같으면 발행하지 않음
        if (self._last_routes is not None and
            self._last_masks is not None and
            np.allclose(routes, self._last_routes, atol=0.1) and
            np.array_equal(net_mask, self._last_masks)):
            return  # 변경 없음 - 발행 스킵

        # 변경됨 - 저장 후 발행
        self._last_routes = routes.copy()
        self._last_masks = net_mask.copy()

        print(f"[ROS2] 웨이포인트 발행: {routes.shape[1]}개 WP")

        for i, pub in enumerate(self._wp_pubs):
            path = Path()
            path.header.frame_id = 'world'  # world 좌표계 (GPS 변환 없음)
            path.header.stamp = self._node.get_clock().now().to_msg()

            for k in range(routes.shape[1]):
                pose = PoseStamped()
                pose.header = path.header
                # world 좌표 직접 전송 (x=East, y=North)
                pose.pose.position.x = float(routes[i, k, 0])
                pose.pose.position.y = float(routes[i, k, 1])
                pose.pose.position.z = 1.0 if net_mask[i, k] else 0.0  # z=1: 그물 전개
                path.poses.append(pose)

            pub.publish(path)
            if i == 0:  # 첫 번째 아군만 로그
                print(f"  Ally {i}: ({routes[i, 0, 0]:.2f}, {routes[i, 0, 1]:.2f})")

    def inject_to_env(self, env):
        """센서 데이터를 환경에 주입.

        env: CommandedCellEnv (DefenseVecEnv 상속)
        """
        state = self.get_state()
        P = min(self.n_allies, env.P)
        M = min(self.n_enemies, env.M)

        # 아군 위치/방위 주입 (world 0)
        env.a_pos[0, :P] = state.ally_pos[:P]
        env.a_hdg[0, :P] = state.ally_hdg[:P]
        env.a_alive[0, :P] = state.ally_alive[:P]

        # 적군 위치/방위 주입
        env.e_pos[0, :M] = state.enemy_pos[:M]
        env.e_hdg[0, :M] = state.enemy_hdg[:M]
        env.e_alive[0, :M] = state.enemy_alive[:M]

    def has_sensor_data(self) -> bool:
        """센서 데이터가 수신되었는지 확인."""
        with self._lock:
            return self._ally_gps_valid.any() or self._enemy_alive.any()

    def step_policy_only(self, env):
        """센서 기반 정책 추론 (자율 시뮬레이션 없음).

        ROS2 모드에서 env.step() 대신 사용.
        센서 데이터가 있을 때만 위치 주입 + 정책 추론 + 발행.
        센서 데이터가 없으면 대기 (배들 정지).
        """
        # 디버그: 100 스텝마다 상태 출력
        t = int(env.t[0]) if hasattr(env.t, '__getitem__') else int(env.t)
        if t % 100 == 0:
            with self._lock:
                print(f"[ROS2] t={t} ally_valid={self._ally_gps_valid.tolist()} "
                      f"enemy_alive={self._enemy_alive.sum()}")

        # 센서 데이터가 없으면 대기 (자율 시뮬레이션 안 함)
        if not self.has_sensor_data():
            # 선박 위치만 발행 (현재 상태 유지)
            self.publish_ship_states(env)
            return env.get_frame()

        # 1. 센서 데이터 주입
        self.inject_to_env(env)

        # 2. 정책 결정 (decision_period 마다)
        micro_ct = getattr(env, '_micro_ct', 0)
        if micro_ct % env.cfg.decision_period == 0:
            if hasattr(env, '_rl_decide'):
                env._rl_decide()
            # 결정 시점에만 웨이포인트 발행 (매 프레임 X)
            routes, masks = self.extract_waypoints(env)
            self.publish_waypoints(routes, masks)

        # 3. 시간 증가
        env.t[0] += 1
        env._micro_ct = micro_ct + 1

        # 4. 선박 위치 발행
        self.publish_ship_states(env)

        return env.get_frame()

    def extract_waypoints(self, env) -> tuple:
        """환경에서 웨이포인트 추출.

        Returns: (routes, net_mask)
        """
        return env.route[0].copy(), env.net_mask[0].copy()

    def _heading_to_quaternion(self, hdg_deg: float) -> 'Quaternion':
        """NAV heading (0=North, CW+) → ROS2 Quaternion.

        Args:
            hdg_deg: 항법 방위각 (0=North, 시계방향+)

        Returns:
            Quaternion for ENU frame
        """
        # NAV (0=North, CW+) → ENU (0=East, CCW+)
        yaw = np.radians(90.0 - hdg_deg)
        return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))

    def publish_ship_states(self, env):
        """선박 위치/상태를 ROS2 토픽으로 발행.

        토픽:
            /ships/allies: 아군 전체 위치/방위 (PoseArray)
            /ships/enemies: 적군 전체 위치/방위 (PoseArray)
            /ships/mothership: 모선 위치 (PoseStamped)

        좌표 체계:
            x: East (동쪽, meters)
            y: North (북쪽, meters)
            z: 0 (alive), -1 (dead)
            orientation: heading을 quaternion으로 변환
        """
        if not self._node or not self._running:
            return

        stamp = self._node.get_clock().now().to_msg()

        # --- 아군 PoseArray ---
        ally_msg = PoseArray()
        ally_msg.header.frame_id = 'world'
        ally_msg.header.stamp = stamp

        for i in range(env.P):
            pose = Pose()
            pose.position.x = float(env.a_pos[0, i, 0])
            pose.position.y = float(env.a_pos[0, i, 1])
            pose.position.z = 0.0 if env.a_alive[0, i] else -1.0
            pose.orientation = self._heading_to_quaternion(float(env.a_hdg[0, i]))
            ally_msg.poses.append(pose)

        self._ally_pub.publish(ally_msg)

        # --- 적군 PoseArray ---
        enemy_msg = PoseArray()
        enemy_msg.header.frame_id = 'world'
        enemy_msg.header.stamp = stamp

        for i in range(env.M):
            pose = Pose()
            pose.position.x = float(env.e_pos[0, i, 0])
            pose.position.y = float(env.e_pos[0, i, 1])
            pose.position.z = 0.0 if env.e_alive[0, i] else -1.0
            pose.orientation = self._heading_to_quaternion(float(env.e_hdg[0, i]))
            enemy_msg.poses.append(pose)

        self._enemy_pub.publish(enemy_msg)

        # --- 모선 PoseStamped ---
        mother_msg = PoseStamped()
        mother_msg.header.frame_id = 'world'
        mother_msg.header.stamp = stamp
        mother_msg.pose.position.x = float(env.center[0])
        mother_msg.pose.position.y = float(env.center[1])
        mother_msg.pose.position.z = 0.0
        # 모선은 heading 없음 → identity quaternion
        mother_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self._mother_pub.publish(mother_msg)


__all__ = ["ROS2SensorBridge", "SensorState", "ROS2_AVAILABLE"]
