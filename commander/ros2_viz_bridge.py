"""
commander/ros2_viz_bridge.py — ROS2 + usv-simulator 3D 시각화 연동 브리지

ROS2에서 실제 센서 데이터를 받거나, 시뮬레이션 데이터를 usv-simulator로 전송하여
3D 시각화를 수행합니다.

아키텍처:
    [MobRobGPT commander] ──MQTT──> [usv-simulator 3D 뷰어]
           │
           └── ROS2 (선택: 실제 센서)

토픽 구조 (usv-simulator MQTT):
    발행: devices/<token>/telemetry   (아군/적/모선 위치)
    구독: devices/<id>/commands       (명령 수신, 선택)

사용:
    bridge = Ros2VizBridge(mqtt_host="localhost", mqtt_port=9001)
    bridge.publish_state(allies, enemies, mothership, nets)
"""

import json
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[ros2_viz_bridge] paho-mqtt not installed. Run: pip install paho-mqtt")


@dataclass
class ShipVizState:
    """시각화용 선박 상태."""
    id: int
    lat: float
    lon: float
    heading: float  # deg, 0=North, CW+
    speed: float    # m/s
    alive: bool = True
    team: str = "ally"  # "ally" | "enemy" | "mothership"


@dataclass
class NetVizState:
    """시각화용 그물 상태."""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    ally_id: int
    installed: bool = True


class Ros2VizBridge:
    """ROS2 + usv-simulator 3D 시각화 브리지.

    MobRobGPT 시뮬레이션/ROS2 데이터를 usv-simulator로 MQTT 전송하여
    브라우저에서 3D 시각화를 수행합니다.
    """

    def __init__(
        self,
        mqtt_host: str = "localhost",
        mqtt_port: int = 9001,  # WebSocket 포트
        mqtt_protocol: str = "websockets",
        device_token: str = "defense-sim",
        publish_rate: float = 10.0,  # Hz
        geo_origin: Tuple[float, float] = (34.625, 128.52),  # 남해 매물도
        world_size: float = 12600.0,
    ):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_protocol = mqtt_protocol
        self.device_token = device_token
        self.publish_rate = publish_rate
        self.geo_origin = geo_origin  # (lat, lon)
        self.world_size = world_size

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._lock = threading.Lock()

        # 상태 캐시
        self._allies: List[ShipVizState] = []
        self._enemies: List[ShipVizState] = []
        self._mothership: Optional[ShipVizState] = None
        self._nets: List[NetVizState] = []
        self._stats: Dict[str, Any] = {}

        # 발행 스레드
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """MQTT 브로커에 연결."""
        if not MQTT_AVAILABLE:
            print("[ros2_viz_bridge] paho-mqtt not available")
            return False

        try:
            self._client = mqtt.Client(
                client_id=f"mobrobgpt-{int(time.time())}",
                transport=self.mqtt_protocol
            )
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            print(f"[ros2_viz_bridge] Connecting to {self.mqtt_host}:{self.mqtt_port}...")
            self._client.connect(self.mqtt_host, self.mqtt_port, 60)
            self._client.loop_start()

            # 연결 대기 (최대 5초)
            for _ in range(50):
                if self._connected:
                    return True
                time.sleep(0.1)

            print("[ros2_viz_bridge] Connection timeout")
            return False

        except Exception as e:
            print(f"[ros2_viz_bridge] Connection error: {e}")
            return False

    def disconnect(self):
        """MQTT 연결 해제."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[ros2_viz_bridge] Connected to MQTT broker")
            self._connected = True
        else:
            print(f"[ros2_viz_bridge] Connection failed: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        print("[ros2_viz_bridge] Disconnected from MQTT broker")
        self._connected = False

    def _sim_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """시뮬레이션 좌표 → GPS 좌표 변환.

        시뮬: x=East, y=North, 중심=(world_size/2, world_size/2)
        GPS: lat=North, lon=East
        """
        center = self.world_size / 2
        dx_m = x - center  # East offset (m)
        dy_m = y - center  # North offset (m)

        # 미터 → 도 변환 (등거리 근사)
        lat = self.geo_origin[0] + dy_m / 111320.0
        lon = self.geo_origin[1] + dx_m / (111320.0 * np.cos(np.radians(self.geo_origin[0])))

        return lat, lon

    def update_state(
        self,
        allies: List[Dict[str, Any]],
        enemies: List[Dict[str, Any]],
        mothership: Dict[str, Any],
        nets: List[Dict[str, Any]] = None,
        stats: Dict[str, Any] = None,
    ):
        """시뮬레이션 상태 업데이트 (시뮬 좌표계).

        Args:
            allies: [{"x", "y", "heading", "speed", "alive"}, ...]
            enemies: [{"x", "y", "heading", "speed", "alive"}, ...]
            mothership: {"x", "y", "heading"}
            nets: [{"start_x", "start_y", "end_x", "end_y", "ally_id"}, ...]
            stats: {"captures", "breaches", ...}
        """
        with self._lock:
            # 아군 변환
            self._allies = []
            for i, a in enumerate(allies):
                lat, lon = self._sim_to_geo(a["x"], a["y"])
                self._allies.append(ShipVizState(
                    id=i, lat=lat, lon=lon,
                    heading=a.get("heading", 0),
                    speed=a.get("speed", 0),
                    alive=a.get("alive", True),
                    team="ally"
                ))

            # 적군 변환
            self._enemies = []
            for i, e in enumerate(enemies):
                lat, lon = self._sim_to_geo(e["x"], e["y"])
                self._enemies.append(ShipVizState(
                    id=i, lat=lat, lon=lon,
                    heading=e.get("heading", 0),
                    speed=e.get("speed", 0),
                    alive=e.get("alive", True),
                    team="enemy"
                ))

            # 모선 변환
            lat, lon = self._sim_to_geo(mothership["x"], mothership["y"])
            self._mothership = ShipVizState(
                id=0, lat=lat, lon=lon,
                heading=mothership.get("heading", 0),
                speed=0,
                alive=True,
                team="mothership"
            )

            # 그물 변환
            self._nets = []
            if nets:
                for n in nets:
                    s_lat, s_lon = self._sim_to_geo(n["start_x"], n["start_y"])
                    e_lat, e_lon = self._sim_to_geo(n["end_x"], n["end_y"])
                    self._nets.append(NetVizState(
                        start_lat=s_lat, start_lon=s_lon,
                        end_lat=e_lat, end_lon=e_lon,
                        ally_id=n.get("ally_id", 0),
                        installed=n.get("installed", True)
                    ))

            # 통계
            if stats:
                self._stats = stats

    def _publish_telemetry(self):
        """현재 상태를 MQTT로 발행."""
        if not self._connected or not self._client:
            return

        topic = f"devices/{self.device_token}/telemetry"
        ts = int(time.time())

        with self._lock:
            readings = []

            # 아군 위치
            for i, ally in enumerate(self._allies):
                if ally.alive:
                    readings.extend([
                        {"sensor": f"ally{i}_lat", "value": ally.lat, "ts": ts},
                        {"sensor": f"ally{i}_lon", "value": ally.lon, "ts": ts},
                        {"sensor": f"ally{i}_heading", "value": ally.heading, "ts": ts},
                        {"sensor": f"ally{i}_speed", "value": ally.speed, "ts": ts},
                    ])

            # 적군 위치
            for i, enemy in enumerate(self._enemies):
                if enemy.alive:
                    readings.extend([
                        {"sensor": f"enemy{i}_lat", "value": enemy.lat, "ts": ts},
                        {"sensor": f"enemy{i}_lon", "value": enemy.lon, "ts": ts},
                        {"sensor": f"enemy{i}_heading", "value": enemy.heading, "ts": ts},
                    ])

            # 모선 위치
            if self._mothership:
                readings.extend([
                    {"sensor": "mothership_lat", "value": self._mothership.lat, "ts": ts},
                    {"sensor": "mothership_lon", "value": self._mothership.lon, "ts": ts},
                ])

            # 그물 (JSON 배열로)
            if self._nets:
                nets_data = [
                    {"s": [n.start_lat, n.start_lon], "e": [n.end_lat, n.end_lon], "a": n.ally_id}
                    for n in self._nets if n.installed
                ]
                readings.append({"sensor": "nets", "value": json.dumps(nets_data), "ts": ts})

            # 통계
            for key, val in self._stats.items():
                readings.append({"sensor": f"stat_{key}", "value": val, "ts": ts})

        # 단건 발행
        for r in readings:
            try:
                self._client.publish(topic, json.dumps(r), qos=1)
            except Exception as e:
                print(f"[ros2_viz_bridge] Publish error: {e}")
                break

    def start_publishing(self):
        """백그라운드 발행 스레드 시작."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        print(f"[ros2_viz_bridge] Publishing started ({self.publish_rate} Hz)")

    def stop_publishing(self):
        """발행 스레드 중지."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[ros2_viz_bridge] Publishing stopped")

    def _publish_loop(self):
        """발행 루프."""
        interval = 1.0 / self.publish_rate
        while self._running:
            self._publish_telemetry()
            time.sleep(interval)


class CommandedSimWithViz:
    """시뮬레이터 + 3D 시각화 래퍼.

    기존 CommandedSimulator/CommandedCellEnv를 감싸고,
    매 스텝마다 usv-simulator로 상태를 전송합니다.
    """

    def __init__(
        self,
        sim,  # CommandedSimulator | CommandedCellEnv
        mqtt_host: str = "localhost",
        mqtt_port: int = 9001,
        auto_connect: bool = True,
    ):
        self.sim = sim
        self.bridge = Ros2VizBridge(
            mqtt_host=mqtt_host,
            mqtt_port=mqtt_port,
            geo_origin=(sim.cfg.geo_lat, sim.cfg.geo_lon),
            world_size=sim.cfg.world_size,
        )

        if auto_connect:
            if self.bridge.connect():
                self.bridge.start_publishing()

    def __getattr__(self, name):
        """시뮬레이터 속성/메서드 위임."""
        return getattr(self.sim, name)

    def step(self):
        """한 스텝 진행 + 시각화 업데이트."""
        result = self.sim.step()
        self._update_viz()
        return result

    def reset(self, **kwargs):
        """리셋 + 시각화 업데이트."""
        result = self.sim.reset(**kwargs)
        self._update_viz()
        return result

    def _update_viz(self):
        """현재 상태를 시각화 브리지로 전송."""
        frame = self.sim.get_frame()

        # 아군
        allies = []
        for i in range(len(frame.get("ally_x", []))):
            allies.append({
                "x": frame["ally_x"][i],
                "y": frame["ally_y"][i],
                "heading": frame.get("ally_hdg", [0]*10)[i],
                "speed": frame.get("ally_speed", [0]*10)[i] if "ally_speed" in frame else 0,
                "alive": frame.get("ally_alive", [True]*10)[i] if "ally_alive" in frame else True,
            })

        # 적군
        enemies = []
        for i in range(len(frame.get("enemy_x", []))):
            enemies.append({
                "x": frame["enemy_x"][i],
                "y": frame["enemy_y"][i],
                "heading": frame.get("enemy_hdg", [0]*10)[i],
                "speed": self.sim.cfg.enemy_speed if hasattr(self.sim, "cfg") else 9,
                "alive": frame.get("enemy_alive", [True]*10)[i] if "enemy_alive" in frame else True,
            })

        # 모선
        center = frame.get("center", (self.sim.cfg.world_size/2, self.sim.cfg.world_size/2))
        mothership = {
            "x": center[0],
            "y": center[1],
            "heading": 0,
        }

        # 그물 (설치된 것만)
        nets = []
        if "net_segments" in frame:
            for seg in frame["net_segments"]:
                nets.append({
                    "start_x": seg[0], "start_y": seg[1],
                    "end_x": seg[2], "end_y": seg[3],
                    "ally_id": seg[4] if len(seg) > 4 else 0,
                    "installed": True,
                })

        # 통계
        stats = frame.get("stats", {})

        self.bridge.update_state(allies, enemies, mothership, nets, stats)

    def close(self):
        """정리."""
        self.bridge.stop_publishing()
        self.bridge.disconnect()


def create_viz_sim(sim, mqtt_host="localhost", mqtt_port=9001):
    """시뮬레이터에 시각화 브리지를 추가하는 팩토리 함수."""
    return CommandedSimWithViz(sim, mqtt_host=mqtt_host, mqtt_port=mqtt_port)
