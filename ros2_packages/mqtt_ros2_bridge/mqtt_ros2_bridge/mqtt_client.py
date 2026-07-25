"""
mqtt_ros2_bridge/mqtt_client.py — MQTT 클라이언트 래퍼
"""

import json
import threading
from typing import Callable, Dict, Any, Optional
import paho.mqtt.client as mqtt

from .config import BridgeConfig


class MqttClient:
    """MQTT 클라이언트 래퍼."""

    def __init__(
        self,
        config: BridgeConfig,
        on_message: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        logger=None,
    ):
        self.config = config
        self.on_message_callback = on_message
        self.logger = logger
        self._connected = False
        self._lock = threading.Lock()
        self._subscriptions: list[tuple[str, int]] = []  # 구독 목록 저장

        # MQTT 클라이언트 생성 (고유 ID 사용)
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        client_id = f"ros2_bridge_{config.device_id}_{unique_suffix}"
        self.client = mqtt.Client(
            client_id=client_id,
            transport=config.mqtt_transport,
        )

        # 인증 설정
        if config.mqtt_username:
            self.client.username_pw_set(
                config.mqtt_username,
                config.mqtt_password
            )

        # 콜백 설정
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def connect(self) -> bool:
        """MQTT 브로커에 연결."""
        try:
            self.client.connect(
                self.config.mqtt_host,
                self.config.mqtt_port,
                keepalive=60
            )
            self.client.loop_start()
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"MQTT 연결 실패: {e}")
            return False

    def disconnect(self):
        """연결 해제."""
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False

    def subscribe(self, topic: str, qos: int = 1):
        """토픽 구독."""
        self.client.subscribe(topic, qos)
        # 재연결 시 다시 구독하기 위해 저장
        if (topic, qos) not in self._subscriptions:
            self._subscriptions.append((topic, qos))
        if self.logger:
            self.logger.info(f"MQTT 구독: {topic}")

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 1):
        """메시지 발행."""
        with self._lock:
            msg = json.dumps(payload)
            self.client.publish(topic, msg, qos=qos)

    def is_connected(self) -> bool:
        """연결 상태 확인."""
        return self._connected

    def _on_connect(self, client, userdata, flags, rc):
        """연결 콜백."""
        if rc == 0:
            self._connected = True
            if self.logger:
                self.logger.info("MQTT 연결 성공")

            # 명령 토픽 구독
            command_topic = self.config.mqtt_command_topic.format(
                id=self.config.device_id
            )
            self.subscribe(command_topic)

            # 기존 구독 복원 (재연결 시)
            for topic, qos in self._subscriptions:
                self.client.subscribe(topic, qos)
                if self.logger:
                    self.logger.debug(f"MQTT 재구독: {topic}")
        else:
            if self.logger:
                self.logger.error(f"MQTT 연결 실패: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        """연결 해제 콜백."""
        self._connected = False
        if self.logger:
            self.logger.warning(f"MQTT 연결 해제: rc={rc}")

    def _on_message(self, client, userdata, msg):
        """메시지 수신 콜백."""
        try:
            payload = json.loads(msg.payload.decode())
            if self.on_message_callback:
                self.on_message_callback(msg.topic, payload)
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(f"MQTT JSON 파싱 오류: {e}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"MQTT 메시지 처리 오류: {e}")


class MultiVesselMqttClient(MqttClient):
    """다중 선박용 MQTT 클라이언트."""

    def __init__(
        self,
        config: BridgeConfig,
        on_ally_telemetry: Optional[Callable[[int, Dict], None]] = None,
        on_enemy_telemetry: Optional[Callable[[int, Dict], None]] = None,
        on_command: Optional[Callable[[int, str, Any], None]] = None,
        logger=None,
    ):
        super().__init__(config, logger=logger)
        self.on_ally_telemetry = on_ally_telemetry
        self.on_enemy_telemetry = on_enemy_telemetry
        self.on_command = on_command

    def subscribe_all(self):
        """모든 관련 토픽 구독."""
        # 아군 텔레메트리
        for i in range(self.config.n_allies):
            self.subscribe(f"usv/ally/{i}/telemetry")
            self.subscribe(f"usv/ally/{i}/commands")

        # 적군 텔레메트리
        for i in range(self.config.n_enemies):
            self.subscribe(f"usv/enemy/{i}/telemetry")

        # 전체 일괄 토픽
        self.subscribe("usv/ally/all/telemetry")
        self.subscribe("usv/enemy/all/telemetry")

        # 시스템
        self.subscribe("usv/system/#")

    def publish_ally_telemetry(self, idx: int, data: Dict[str, Any]):
        """아군 텔레메트리 발행."""
        self.publish(f"usv/ally/{idx}/telemetry", data)

    def publish_enemy_telemetry(self, idx: int, data: Dict[str, Any]):
        """적군 텔레메트리 발행."""
        self.publish(f"usv/enemy/{idx}/telemetry", data)

    def publish_command(self, idx: int, channel: str, value: Any):
        """아군 명령 발행."""
        self.publish(f"usv/ally/{idx}/commands", {
            "channel": channel,
            "value": value,
        })
