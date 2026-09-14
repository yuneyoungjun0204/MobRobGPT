"""commander/gcs_bridge.py — GCS(지상국) HTTP 연동: 아군 텔레메트리 수신 + 경유점 명령 송신.

`/home/yune/gcs` 저장소(docs/contracts.md)가 이미 갖춘 명령/관측 계약을 그대로 쓴다.
ROS2(rclpy)는 필요 없다 — GCS의 `server/api.py`가 `source="rl"`을 명시적으로 허용하는
평범한 HTTP API를 제공하므로, 여기서는 표준 라이브러리 `urllib`만으로 붙는다.

계약 (gcs 저장소 쪽, 변경하지 않음 — 이 파일은 gcs를 import 하지 않는다):
    GET  {base_url}/api/state
        -> {"t": <float>, "vehicles": {"<id>": {"ned": {"x": north_m, "y": east_m},
                                                 "heading": deg (0=N, CW+),
                                                 "connected": bool,
                                                 "stale": {"position": bool, ...}} | null}}
    POST {base_url}/api/command/{vehicle_id}/goto
        body = {"east": m, "north": m, "source": "rl", "speed": m/s|omit, "stamp": unix_s}
        -> {"accepted": bool, "reason": str, "detail": str}   (verdict, 항상 200)

    대상 vehicle_id는 gcs 쪽 `vehicles.yaml`에 `role: defender`로 등록돼 있어야 한다
    (command/authority.py의 ROLE_SOURCES — 그 외 role이면 매 명령이 role_mismatch로
    거부된다. 이건 설정 문제이지 이 브릿지의 버그가 아니다).

절대 하지 않는 것 — "그물 뿌리기"는 여기서도, gcs 쪽에서도 액추에이터 명령이 된 적이
없다(boatattack_sim/env/defense_env.py의 net_mask/doing_net은 순수 로직 플래그).
gcs의 command/gate.py는 이 저장소가 보낼 수 있는 유일한 MAVLink 동작 명령이고
tests/basic/test_coverage.py가 서보/릴레이/액추에이터 토큰을 전 파일에서 금지한다.
그래서 `NetDeploySink`는 GCS에 아무것도 보내지 않는다 — 그물을 실제로 전개하는 쪽
(별도 액추에이터 컨트롤러/시뮬레이터)이 이 신호를 구독해야 한다.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class GcsRequestError(RuntimeError):
    """GCS HTTP 요청이 실패했다(연결 거부, 타임아웃, non-2xx 등)."""


class GcsClient:
    """`base_url`의 GCS HTTP API에 대한 아주 얇은 클라이언트. 표준 라이브러리만 쓴다."""

    def __init__(self, base_url: str, timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def get_json(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GcsRequestError(f"GET {url} failed: {exc}") from exc

    def post_json(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # gcs의 goto 엔드포인트는 거부도 200으로 돌려준다(server/api.py) — 여기 걸리는
            # 건 malformed body(400) 정도다. 그래도 verdict 형태로 통일해 호출부를 단순하게.
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                raise GcsRequestError(f"POST {url} failed: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GcsRequestError(f"POST {url} failed: {exc}") from exc


@dataclass
class AllySnapshot:
    """한 tick의 아군 텔레메트리. `pos`/`hdg`/`alive`는 `GcsBagCnnEnv`가 기대하는
    [P,2]/[P]/[P] 배열과 같은 순서(= 생성 시 준 vehicle_ids 순서)다."""
    pos: np.ndarray     # [P,2] local ENU metres (x=east, y=north) -- map frame
    hdg: np.ndarray     # [P] deg, 0=North, CW+  (gcs `heading` 필드와 동일 규약)
    alive: np.ndarray   # [P] bool


class GcsAllyLink:
    """gcs `/api/state`에서 아군(P척) 상태를 읽고, `/api/command/{id}/goto`로 명령한다."""

    def __init__(self, client: GcsClient, vehicle_ids: list[str], *,
                 source: str = "rl", default_speed: Optional[float] = None):
        if not vehicle_ids:
            raise ValueError("vehicle_ids must be non-empty")
        self.client = client
        self.vehicle_ids = list(vehicle_ids)
        self.source = source
        self.default_speed = default_speed

    @property
    def n_allies(self) -> int:
        return len(self.vehicle_ids)

    def ally_snapshot(self, state: Optional[dict] = None) -> AllySnapshot:
        """`/api/state`를 (필요하면) 읽어 아군 P척의 위치/방위/생존을 뽑는다.

        `state`를 넘기면 그 dict를 그대로 쓴다(테스트, 또는 한 번 받은 스냅샷을
        아군/적 양쪽에서 재사용할 때). 한 대라도 없으면(vehicle_id 미등록) 예외 대신
        그 배만 alive=False로 표시한다 — 등록 누락이 전체 tick을 멈추면 안 된다.

        위치가 신선해도 자세(attitude)만 stale 할 수 있다(docs/contracts.md §3 -- 서로
        다른 MAVLink 메시지에서 온다). 그 경우 `heading`을 신뢰하면 "정북을 향한 배"를
        지어내는 꼴이라 -- 대신 그 배 전체를 이번 tick 미수신으로 표시한다(하나 걸러도
        위치가 마지막 값으로 유지되는 건 안전하지만, 없는 헤딩을 만들어 정책에 먹이는
        건 안전하지 않다). 값 자체도 finite 여야 한다.
        """
        if state is None:
            state = self.client.get_json("/api/state")
        vehicles = state.get("vehicles", {})
        P = self.n_allies
        pos = np.zeros((P, 2), dtype=np.float64)
        hdg = np.zeros(P, dtype=np.float64)
        alive = np.zeros(P, dtype=bool)
        for i, vid in enumerate(self.vehicle_ids):
            v = vehicles.get(vid)
            if not v:
                continue
            ned = v.get("ned")
            stale = v.get("stale") or {}
            if (ned is None or not v.get("connected")
                    or stale.get("position") or stale.get("attitude")):
                continue
            east, north = float(ned["y"]), float(ned["x"])   # ned = {x: north, y: east}
            heading = v.get("heading")
            if heading is None or not (np.isfinite(east) and np.isfinite(north)
                                        and np.isfinite(heading)):
                continue
            pos[i] = (east, north)
            hdg[i] = float(heading)
            alive[i] = True
        return AllySnapshot(pos=pos, hdg=hdg, alive=alive)

    def submit_goto(self, vehicle_id: str, east: float, north: float, *,
                     speed: Optional[float] = None, stamp: Optional[float] = None) -> dict:
        body = {
            "east": float(east), "north": float(north),
            "source": self.source,
            "stamp": float(stamp) if stamp is not None else time.time(),
        }
        eff_speed = self.default_speed if speed is None else speed
        if eff_speed is not None:
            body["speed"] = float(eff_speed)
        return self.client.post_json(f"/api/command/{vehicle_id}/goto", body)


class NetDeploySink:
    """그물 전개 시작/종료 신호를 GCS 밖으로 내보낸다. GCS는 이 클래스가 존재하는지도 모른다.

    기본은 로그 출력뿐이다. `log_path`를 주면 JSON-lines로도 남겨, 실제 액추에이터를 쥔
    별도 프로세스가 tail 해서 쓸 수 있게 한다. `ros2_vehicle_ids`를 주면 그 각 배마다
    `/{ros2_namespace}/{vehicle_id}/net_deploy`(std_msgs/Int32, 0=대기/1=전개중)로도
    발행한다 -- 첫 경유점(그물 시작 leg) 도착 시 0->1, 두번째 경유점(그물 완성) 도착 시
    다시 1->0(`commander/gcs_cnn_env.py::_advance_and_paint`의 start/finish가 그 두 순간에
    이 클래스의 `set()`을 부른다). 구독만 하고 발행은 안 하므로 `rclpy.spin()`이 필요 없다
    -- `commander/live_enemy_ros2.py`처럼 백그라운드 스레드를 따로 안 둬도 된다.

    이 클래스는 어떤 MAVLink/서보/릴레이 호출도 하지 않는다 -- `tests/basic/test_coverage.py`
    (gcs)의 FORBIDDEN 토큰 감사 대상 밖이면서도 같은 원칙을 스스로 지킨다. ROS2 발행도
    "신호를 알린다"일 뿐 액추에이터를 직접 움직이지 않는다 -- 그물을 실제로 펴는 쪽은
    이 토픽을 구독하는 별도 프로세스의 몫이다.
    """

    def __init__(self, log_fn: Callable[[str], None] = print,
                 log_path: Optional[str] = None,
                 ros2_vehicle_ids: Optional[list] = None,
                 ros2_namespace: str = "mobrobgpt"):
        self.log_fn = log_fn
        self.log_path = log_path
        self._ros2_node = None
        self._ros2_pubs: dict = {}
        if ros2_vehicle_ids:
            self._init_ros2(list(ros2_vehicle_ids), ros2_namespace)

    def _init_ros2(self, vehicle_ids: list, namespace: str) -> None:
        try:
            import rclpy
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
            from std_msgs.msg import Int32
        except ImportError as exc:
            self.log_fn(f"[net-deploy] ⚠ rclpy 없음 -- ROS2 발행 생략(파일/로그만 동작): {exc}")
            return
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("mobrobgpt_net_deploy_pub")
        self._ros2_node = self._node
        # TRANSIENT_LOCAL: 늦게 붙는 구독자(액추에이터 컨트롤러가 나중에 뜨는 경우)도
        # 마지막 상태를 즉시 받는다 -- 이 신호는 "바뀔 때만" 발행되므로(폴링 아님) 래치가
        # 없으면 중간에 붙은 구독자는 다음 전이가 올 때까지 현재 상태를 알 방법이 없다.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        self._Int32 = Int32
        for vid in vehicle_ids:
            pub = self._node.create_publisher(Int32, f"/{namespace}/{vid}/net_deploy", qos)
            self._ros2_pubs[vid] = pub
            msg = Int32(); msg.data = 0
            pub.publish(msg)   # 초기 상태(대기) -- 첫 전이 전에 구독해도 0을 본다

    def set(self, vehicle_id: str, active: bool, *, stamp: Optional[float] = None) -> None:
        rec = {
            "vehicle_id": vehicle_id,
            "net_deploy": bool(active),
            "stamp": float(stamp) if stamp is not None else time.time(),
        }
        line = json.dumps(rec, ensure_ascii=False)
        self.log_fn(f"[net-deploy] {line}")
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        pub = self._ros2_pubs.get(vehicle_id)
        if pub is not None:
            msg = self._Int32(); msg.data = 1 if active else 0
            pub.publish(msg)

    def shutdown(self) -> None:
        if self._ros2_node is not None:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                pass


# WGS84 곡률반경 기반 ENU(east,north 미터, datum 기준) -> 위경도. gcs 저장소
# common/geo.py::Datum.to_geodetic와 동일한 공식(이 저장소는 gcs를 import 하지 않으므로
# 복제) -- 그쪽에서 독립 검증(2026-09-09 세션, cm 단위 오차)된 바로 그 공식이다.
_WGS84_A = 6378137.0                       # semi-major axis, m
_WGS84_F = 1.0 / 298.257223563             # flattening
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)    # first eccentricity squared


def enu_to_latlon(datum_lat: float, datum_lon: float,
                   east: float, north: float) -> tuple[float, float]:
    """datum(lat,lon) 기준 로컬 ENU (east,north)[m] -> (lat,lon)[deg]."""
    import math
    sin_lat = math.sin(math.radians(datum_lat))
    w = 1.0 - _WGS84_E2 * sin_lat * sin_lat
    m = _WGS84_A * (1.0 - _WGS84_E2) / (w ** 1.5)   # meridian radius (동/북의 '북' 방향)
    n = _WGS84_A / math.sqrt(w)                      # prime-vertical radius ('동' 방향)
    lat = datum_lat + math.degrees(north / m)
    lon = datum_lon + math.degrees(east / (n * math.cos(math.radians(datum_lat))))
    return lat, lon


class WaypointSink:
    """배당 배정된 경유점(wp1+wp2)을 위경도로 묶어 ROS2로 내보낸다. GCS는 모른다.

    `commander/gcs_cnn_env.py::_GcsAllyTelemetryMixin._publish_to_gcs()`가 매 tick,
    goto(HTTP, "현재 활성 경유점" 하나만)와는 별도로 이 클래스의 `publish()`를 불러
    그 배의 route 전체(현재 Kw=2, 즉 wp1+wp2)를 한 메시지로 묶어 발행한다.

    `datum`(GCS `gcs.datum`의 lat/lon) 없이는 ENU->위경도 변환이 불가능하므로,
    `datum is None`이면 ROS2 설정 자체를 건너뛰고 `publish()`가 조용히 no-op한다
    (`NetDeploySink`가 rclpy 없을 때 그러는 것과 같은 원칙 -- 이 신호가 없어도 goto는
    독립적으로 계속 나간다, 이 클래스가 스크립트 전체를 막으면 안 된다).

    커스텀 .msg를 만들지 않는다(이 저장소 전체 관례) -- `std_msgs/String`에 JSON을 담는다.
    `bridge/run_ros2_bridge.py`(gcs 저장소)의 `/gcs/cmd/target/intent`가 이미 같은
    관례(String+JSON)를 쓴다.

    `state_path`를 주면 최신 경유점을 vehicle_id별로 모아 그 경로에 JSON으로도 쓴다 --
    브라우저는 ROS2 토픽을 직접 구독할 수 없으므로, gcs 웹 UI(index.html)의 시각화는
    이 파일을 HTTP로 중계하는 별도 사이드카(tools/gcs_bridge_control.py)를 통해 읽는다.
    rclpy 설치 여부와 무관하게 동작한다(`datum`만 있으면 됨) -- NetDeploySink의
    `log_path`가 ROS2 pub 유무와 독립적으로 항상 쓰이는 것과 같은 이유다.
    """

    def __init__(self, datum: Optional[tuple] = None,
                 ros2_vehicle_ids: Optional[list] = None,
                 ros2_namespace: str = "mobrobgpt",
                 log_fn: Callable[[str], None] = print,
                 state_path: Optional[Path] = None):
        self.datum = tuple(datum) if datum is not None else None
        self.log_fn = log_fn
        self.state_path = Path(state_path) if state_path is not None else None
        self._latest: dict = {}
        self._ros2_node = None
        self._ros2_pubs: dict = {}
        if self.datum is not None and ros2_vehicle_ids:
            self._init_ros2(list(ros2_vehicle_ids), ros2_namespace)
        elif ros2_vehicle_ids and self.datum is None:
            self.log_fn("[waypoints] ⚠ datum 없음 -- 경유점 위경도 ROS2 발행 생략")

    def _init_ros2(self, vehicle_ids: list, namespace: str) -> None:
        try:
            import rclpy
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
            from std_msgs.msg import String
        except ImportError as exc:
            self.log_fn(f"[waypoints] ⚠ rclpy 없음 -- ROS2 발행 생략: {exc}")
            return
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("mobrobgpt_waypoints_pub")
        self._ros2_node = self._node
        # TRANSIENT_LOCAL: net_deploy와 같은 이유 -- 매 tick 발행이긴 하지만, 늦게 붙는
        # 구독자가 첫 publish_hz 주기(최대 2s)를 기다리지 않고 바로 마지막 값을 본다.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        self._String = String
        for vid in vehicle_ids:
            self._ros2_pubs[vid] = self._node.create_publisher(
                String, f"/{namespace}/{vid}/waypoints", qos)

    def publish(self, vehicle_id: str, waypoints_enu: list,
                active_index: int, *, stamp: Optional[float] = None) -> None:
        """`waypoints_enu`: [(east,north), ...] (route 순서 그대로, 현재 길이 2).
        `active_index`는 `waypoints_enu`(=발행되는 `waypoints` 배열)를 가리키는
        인덱스여야 한다 -- 호출부가 route 슬롯 전체(패딩 포함) 기준 ptr을 그대로 넘기면
        `waypoints[active_index]`가 범위를 벗어난다, 반드시 `min(ptr, len(waypoints_enu)-1)`
        로 clamp해서 넘길 것.

        datum이 없으면 완전히 no-op(ENU->위경도 변환 자체가 불가능). datum이 있으면
        ROS2 publisher 유무와 무관하게 `_latest`/`state_path`는 항상 갱신된다 -- 이
        vehicle_id로 ROS2 publisher가 없을 때(rclpy 미설치, 또는 이 배가
        ros2_vehicle_ids에 없었을 때)만 `pub.publish()`를 건너뛴다."""
        if self.datum is None:
            return
        rec = {
            "vehicle_id": vehicle_id,
            "active_index": int(active_index),
            "waypoints": [
                {"lat": lat, "lon": lon}
                for lat, lon in (enu_to_latlon(self.datum[0], self.datum[1], e, n)
                                 for e, n in waypoints_enu)
            ],
            "stamp": float(stamp) if stamp is not None else time.time(),
        }
        self._latest[vehicle_id] = rec
        if self.state_path is not None:
            self._write_state()
        pub = self._ros2_pubs.get(vehicle_id)
        if pub is not None:
            msg = self._String()
            msg.data = json.dumps(rec, ensure_ascii=False)
            pub.publish(msg)

    def _write_state(self) -> None:
        """temp+rename: 사이드카(tools/gcs_bridge_control.py)가 같은 파일을 동시에
        읽어도 절반만 쓰인 JSON을 보는 일이 없게 한다."""
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"vehicles": self._latest}, ensure_ascii=False))
            tmp.replace(self.state_path)
        except OSError as exc:
            self.log_fn(f"[waypoints] ⚠ 상태 파일 기록 실패({self.state_path}): {exc}")

    def shutdown(self) -> None:
        if self._ros2_node is not None:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                pass


__all__ = ["GcsClient", "GcsAllyLink", "AllySnapshot", "NetDeploySink", "WaypointSink",
           "enu_to_latlon", "GcsRequestError"]
