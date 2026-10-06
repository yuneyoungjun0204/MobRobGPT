"""commander/sim_odom_ros2.py — Unreal 시뮬레이터(ROS2) 입출력 어댑터.

시뮬레이터는 위경도가 아니라 로컬 미터 프레임(`map`, 모선=(0,0), x/y = 동/북 m)으로
모든 선박을 `nav_msgs/Odometry`로 내보내고, 명령도 같은 x/y(m)로 받는다
(2026-10-04 `usv_20261004_194054` 로스백으로 확인):

  입력  /usv/{friendly_0X,enemy_0X,mothership}/odometry   nav_msgs/Odometry (BEST_EFFORT)
        /usv/sim_state                                     std_msgs/String JSON
              {"sim_id","phase","ships":[{"ship","nets_left"|"captured"}],"nets":[...]}
  출력  /usv/waypoints     std_msgs/String JSON
              {"sim_id","ship","points":[{"x_m","y_m"}],"cruise_kn","stop_at_final","accept_radius_m"}
        /usv/net_trigger   std_msgs/String JSON  {"sim_id","ship","lay":true}  (1회 = 그물 1장)

`SimOdomLink.ally_view` / `.enemy_view` 는 `GcsAllyLink`와 덕타이핑으로 맞춰져 있어
(`vehicle_ids`/`n_allies`/`ally_snapshot()` → `AllySnapshot`) `GcsLiveCnnEnv` 계열의
결정·WP 진행·그물 판정 로직을 그대로 재사용한다(`commander/sim_odom_env.py`).

상태 보관(`SimOdomState`)과 메시지 생성(`SimWaypointSink`/`SimNetTriggerSink`)은 rclpy
없이도 동작한다 — ROS 노드는 얇은 콜백 래퍼일 뿐이라 단위 테스트가 ROS 없이 돈다.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Callable, Optional

import numpy as np

from .gcs_bridge import AllySnapshot

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                           DurabilityPolicy, qos_profile_sensor_data)
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


# 적 출현거리를 잴 수 있는 sim_state phase (failure 등 종료 phase 제외)
ACTIVE_PHASES = ("ready", "running")


def yaw_enu_from_quat(x: float, y: float, z: float, w: float) -> float:
    """쿼터니언 → ENU yaw(deg, 0=+x(동), 반시계+)."""
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_enu_to_nav(yaw_enu_deg: float) -> float:
    """ENU yaw → nav heading(deg, 0=북, 시계+) — 시뮬 env 의 a_hdg/e_hdg 규약."""
    return (90.0 - float(yaw_enu_deg)) % 360.0


class SimOdomState:
    """선박별 최신 Odometry + sim_state 를 스레드 안전하게 보관(ROS 무관)."""

    def __init__(self, stale_timeout: float = 1.0, clock: Callable[[], float] = time.monotonic):
        self.stale_timeout = float(stale_timeout)
        self._clock = clock
        self._lock = threading.Lock()
        self._odom: dict[str, tuple[float, float, float, float]] = {}   # id -> (x, y, hdg_nav, t)
        self.sim_id: Optional[str] = None
        self.phase: Optional[str] = None
        self.captured: set[str] = set()
        self.nets_left: dict[str, int] = {}
        self.last_assignment: Optional[dict] = None   # 외부(기존 컨트롤러) /usv/assignment — 비교용

    def on_odom(self, ship: str, x: float, y: float, yaw_enu_deg: float,
                t: Optional[float] = None) -> None:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw_enu_deg)):
            return
        with self._lock:
            self._odom[ship] = (float(x), float(y), yaw_enu_to_nav(yaw_enu_deg),
                                self._clock() if t is None else float(t))

    def on_sim_state(self, d: dict) -> None:
        with self._lock:
            self.sim_id = d.get("sim_id", self.sim_id)
            self.phase = d.get("phase", self.phase)
            ships = d.get("ships") or []
            self.captured = {s["ship"] for s in ships if s.get("captured")}
            self.nets_left = {s["ship"]: int(s["nets_left"]) for s in ships
                              if s.get("nets_left") is not None}

    def on_assignment(self, d: dict) -> None:
        with self._lock:
            self.last_assignment = d.get("by_ship", d)

    def latest(self, ship: str) -> Optional[tuple[float, float, float, float]]:
        with self._lock:
            return self._odom.get(ship)

    def snapshot(self, ids: list[str], *, drop_captured: bool) -> AllySnapshot:
        now = self._clock()
        P = len(ids)
        pos = np.zeros((P, 2), np.float64)
        hdg = np.zeros(P, np.float64)
        alive = np.zeros(P, bool)
        with self._lock:
            for i, sid in enumerate(ids):
                rec = self._odom.get(sid)
                if rec is None:
                    continue
                pos[i] = rec[:2]
                hdg[i] = rec[2]
                fresh = (now - rec[3]) <= self.stale_timeout
                alive[i] = fresh and not (drop_captured and sid in self.captured)
        return AllySnapshot(pos=pos, hdg=hdg, alive=alive)


class SimShipView:
    """`GcsAllyLink` 덕타입(읽기 전용) — `GcsLiveCnnEnv`의 ally_link/enemy_link 자리."""

    def __init__(self, state: SimOdomState, vehicle_ids: list[str], *, drop_captured: bool):
        if not vehicle_ids:
            raise ValueError("vehicle_ids must be non-empty")
        self.state = state
        self.vehicle_ids = list(vehicle_ids)
        self.drop_captured = bool(drop_captured)

    @property
    def n_allies(self) -> int:
        return len(self.vehicle_ids)

    def ally_snapshot(self, state=None) -> AllySnapshot:   # state 인자는 GcsAllyLink 호환용
        return self.state.snapshot(self.vehicle_ids, drop_captured=self.drop_captured)


class SimWaypointSink:
    """`/usv/waypoints` JSON 발행. `publish_fn=None` 이면 dry-run(발행 없이 기록만).

    같은 배에 같은 포인트 목록(0.1 m 반올림 기준)은 다시 보내지 않는다 — 시뮬레이터가
    새 목록을 받을 때마다 leg 를 처음부터 다시 시작할 수 있기 때문이다.
    """

    def __init__(self, publish_fn: Optional[Callable[[str], None]] = None, *,
                 cruise_kn: float = 40.0, accept_radius_m: float = 4.0,
                 stop_at_final: bool = False, log_path: Optional[str] = None,
                 verbose: bool = True):
        self.publish_fn = publish_fn
        self.cruise_kn = float(cruise_kn)
        self.accept_radius_m = float(accept_radius_m)
        self.stop_at_final = bool(stop_at_final)
        self.log_path = log_path
        self.verbose = verbose
        self._last: dict[str, tuple] = {}
        self.sent: list[dict] = []     # 실제로 보낸(또는 dry-run 으로 기록한) 메시지

    def build(self, ship: str, points_xy, sim_id: Optional[str]) -> dict:
        return {
            "sim_id": sim_id,
            "ship": ship,
            "points": [{"x_m": round(float(x), 3), "y_m": round(float(y), 3)} for x, y in points_xy],
            "cruise_kn": self.cruise_kn,
            "stop_at_final": self.stop_at_final,
            "accept_radius_m": self.accept_radius_m,
        }

    def publish(self, ship: str, points_xy, sim_id: Optional[str]) -> bool:
        key = tuple((round(float(x), 1), round(float(y), 1)) for x, y in points_xy)
        if self._last.get(ship) == key:
            return False
        self._last[ship] = key
        _emit(self.build(ship, points_xy, sim_id), self.publish_fn, self.log_path, self.sent,
              tag="waypoints", verbose=self.verbose)
        return True


class SimNetTriggerSink:
    """`/usv/net_trigger` JSON 발행 — `NetDeploySink.set(vid, deploying)` 덕타입.

    믹스인 `_advance_and_paint` 가 그물 구간 시작 시 `set(vid, True)`, 끝날 때 `set(vid, False)`
    를 부른다. 시뮬레이터는 lay 1회 = 그물 1장(약 40 m)이므로 True 에서만 1회 발행한다
    (구간당 1장).
    """

    def __init__(self, publish_fn: Optional[Callable[[str], None]] = None, *,
                 sim_id_fn: Callable[[], Optional[str]] = lambda: None,
                 log_path: Optional[str] = None, verbose: bool = True):
        self.publish_fn = publish_fn
        self.sim_id_fn = sim_id_fn
        self.log_path = log_path
        self.verbose = verbose
        self.sent: list[dict] = []

    def set(self, vehicle_id: str, deploying: bool) -> None:
        if not deploying:
            return
        msg = {"sim_id": self.sim_id_fn(), "ship": vehicle_id, "lay": True}
        _emit(msg, self.publish_fn, self.log_path, self.sent,
              tag="net_trigger", verbose=self.verbose)

    def shutdown(self) -> None:      # NetDeploySink 호환
        pass


def _emit(msg: dict, publish_fn, log_path, sent: list, *, tag: str, verbose: bool) -> None:
    data = json.dumps(msg, ensure_ascii=False)
    sent.append(msg)
    if publish_fn is not None:
        try:
            publish_fn(data)
        except Exception as exc:              # 발행 실패가 결정 루프를 죽이면 안 된다
            print(f"[sim_bridge] /usv/{tag} 발행 실패: {exc}")
    if verbose:
        mode = "PUB" if publish_fn is not None else "DRY"
        print(f"[sim_bridge] {mode} /usv/{tag} {data}")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t_wall": time.time(), "topic": f"/usv/{tag}",
                                    "published": publish_fn is not None, "msg": msg},
                                   ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[sim_bridge] 로그 기록 실패({log_path}): {exc}")


class SimOdomLink:
    """rclpy 노드 1개를 백그라운드 스레드로 돌려 시뮬레이터 토픽을 구독(및 선택적으로 발행)한다."""

    def __init__(self, ally_ids: list[str], enemy_ids: list[str], *,
                 mothership_id: Optional[str] = "mothership",
                 topic_fmt: str = "/usv/{id}/odometry",
                 sim_state_topic: str = "/usv/sim_state",
                 assignment_topic: Optional[str] = "/usv/assignment",
                 waypoints_topic: str = "/usv/waypoints",
                 net_trigger_topic: str = "/usv/net_trigger",
                 enable_publish: bool = False,
                 stale_timeout: float = 1.0):
        if not ROS2_AVAILABLE:
            raise ImportError(
                "rclpy/nav_msgs/std_msgs 를 불러올 수 없습니다. ROS2 환경을 먼저 source 하세요: "
                "`source /opt/ros/humble/setup.bash` (python3.10 필요).")
        overlap = set(ally_ids) & set(enemy_ids)
        if overlap:
            raise ValueError(f"아군/적 id 가 겹칩니다: {sorted(overlap)}")
        self.state = SimOdomState(stale_timeout=stale_timeout)
        self.ally_view = SimShipView(self.state, ally_ids, drop_captured=False)
        self.enemy_view = SimShipView(self.state, enemy_ids, drop_captured=True)
        self.mothership_id = mothership_id
        if not rclpy.ok():
            rclpy.init()
        ids = list(ally_ids) + list(enemy_ids) + ([mothership_id] if mothership_id else [])
        self._node = _SimOdomNode(self.state, ids, topic_fmt, sim_state_topic, assignment_topic,
                                  waypoints_topic if enable_publish else None,
                                  net_trigger_topic if enable_publish else None)
        self.publish_waypoints_fn = self._node.publish_waypoints if enable_publish else None
        self.publish_net_fn = self._node.publish_net if enable_publish else None
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

    # ── 준비/진단 헬퍼 ───────────────────────────────────────────────
    def wait_allies(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ally_view.ally_snapshot().alive.all():
                return True
            time.sleep(0.1)
        return False

    def wait_enemy_range(self, origin: tuple[float, float], timeout: float,
                         settle: float = 1.0, no_state_grace: float = 3.0) -> float:
        """적이 보이기 시작하면 `settle`초 동안 관측한 모선-적 최대거리를 돌려준다(못 보면 0).

        첫 1척만 보고 바로 재면 같은 웨이브의 나머지(더 먼 적)를 놓치고, 반대로 settle 이후
        한 번만 재면 그 사이 끊긴 트랙(로스백 끝 등) 때문에 0이 나올 수 있다 — 창 안의 최대값을 쓴다.
        """
        deadline = time.monotonic() + timeout
        t_start = time.monotonic()
        while time.monotonic() < deadline:
            # 종료 phase(failure 등)에선 남은 적이 이미 모선 근처라 출현거리를 잴 수 없고, sim_state 를
            # 받기 전엔 captured 를 몰라 포획된(계속 Odometry 를 내는) 적을 셀 수 있다 — 활성 phase 에서만 잰다.
            # sim_state 를 내지 않는 원천이면 `no_state_grace`초 뒤부터는 phase 없이 진행한다.
            phase = self.state.phase
            if phase is None and time.monotonic() - t_start < no_state_grace:
                time.sleep(0.1)
                continue
            if phase is not None and phase not in ACTIVE_PHASES:
                time.sleep(0.1)
                continue
            r = self.max_enemy_range(origin)
            if r > 0:
                end = time.monotonic() + settle
                while time.monotonic() < end:
                    time.sleep(0.05)
                    r = max(r, self.max_enemy_range(origin))
                return r
            time.sleep(0.1)
        return 0.0

    def mothership_xy(self) -> Optional[tuple[float, float]]:
        if not self.mothership_id:
            return None
        rec = self.state.latest(self.mothership_id)
        return None if rec is None else (rec[0], rec[1])

    def max_enemy_range(self, origin: tuple[float, float]) -> float:
        snap = self.enemy_view.ally_snapshot()
        if not snap.alive.any():
            return 0.0
        d = np.hypot(*(snap.pos[snap.alive] - np.asarray(origin, np.float64)).T)
        return float(d.max())

    def shutdown(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self._spin_thread.join(timeout=2.0)


if ROS2_AVAILABLE:
    class _SimOdomNode(Node):
        def __init__(self, state: SimOdomState, ids: list[str], topic_fmt: str,
                     sim_state_topic: str, assignment_topic: Optional[str],
                     waypoints_topic: Optional[str], net_trigger_topic: Optional[str]):
            super().__init__("mobrobgpt_sim_odom_bridge")
            self._state = state
            self._subs = []
            # Odometry 는 BEST_EFFORT 로 발행된다(로스백 QoS 확인) — sensor_data 프로파일로 받는다.
            for sid in ids:
                self._subs.append(self.create_subscription(
                    Odometry, topic_fmt.format(id=sid),
                    lambda m, s=sid: self._on_odom(s, m), qos_profile_sensor_data))
            # JSON 상태/명령 토픽은 RELIABLE/VOLATILE 로 녹화돼 있다 — 같은 QoS 로 맞춘다.
            rel = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            self._subs.append(self.create_subscription(
                String, sim_state_topic, self._on_sim_state, rel))
            if assignment_topic:
                self._subs.append(self.create_subscription(
                    String, assignment_topic, self._on_assignment, rel))
            self._pub_wp = (self.create_publisher(String, waypoints_topic, rel)
                            if waypoints_topic else None)
            self._pub_net = (self.create_publisher(String, net_trigger_topic, rel)
                             if net_trigger_topic else None)

        def _on_odom(self, ship: str, m) -> None:
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            self._state.on_odom(ship, p.x, p.y, yaw_enu_from_quat(q.x, q.y, q.z, q.w))

        def _on_sim_state(self, m) -> None:
            try:
                self._state.on_sim_state(json.loads(m.data))
            except (ValueError, KeyError, TypeError):
                pass

        def _on_assignment(self, m) -> None:
            try:
                self._state.on_assignment(json.loads(m.data))
            except (ValueError, TypeError, AttributeError):
                pass

        def publish_waypoints(self, data: str) -> None:
            self._pub_wp.publish(String(data=data))

        def publish_net(self, data: str) -> None:
            self._pub_net.publish(String(data=data))


__all__ = ["SimOdomLink", "SimOdomState", "SimShipView", "SimWaypointSink",
           "SimNetTriggerSink", "yaw_enu_from_quat", "yaw_enu_to_nav", "ROS2_AVAILABLE"]
