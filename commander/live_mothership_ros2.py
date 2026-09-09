"""commander/live_mothership_ros2.py — GCS가 새로 발행하는 datum(모선 기준점) 토픽 확인.

`/home/yune/gcs`(server/run_ros2_bridge.py, 2026-09-09 추가)는 이제
`/<namespace>/datum/gps/fix`(sensor_msgs/NavSatFix, frame_id="datum")로 `gcs.datum`을
발행한다 — vehicles.yaml을 직접 읽지 않고도 "이 GCS의 맵 프레임 원점이 실제로 어디인지"를
구독자가 알 수 있게 하기 위해서다.

이 값을 다시 (동/북) 오프셋으로 변환하는 건 의미가 없다 — datum 자기 자신 기준
ENU 변환은 언제나 (0,0)이다(정의상). 이 모듈의 실질적 쓸모는 **그 존재 여부 확인**이다:
이 토픽이 살아있다는 건 "GCS가 보고하는 아군 ned/east/north 값들이 실제로 이 datum
기준으로 잡혀 있다"는 걸 뜻하므로, `run_gcs_bridge.py`가 `--enu-origin`을 (0,0)으로
기본값 삼는 게 안전하다는 근거가 된다(로스백 적선 평균 위치를 원점으로 잘못 삼던
이전 기본값과 달리). 토픽이 없으면(구버전 GCS 서버) 이 가정이 검증 안 됐다는 뜻이므로
호출부가 경고를 낼 수 있게 `None`을 돌려준다.
"""
from __future__ import annotations

import time
from typing import Optional

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import NavSatFix
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def fetch_gcs_datum(namespace: str = "gcs_fleet", timeout: float = 3.0) -> Optional[dict]:
    """`/<namespace>/datum/gps/fix`를 짧게 구독해 1개 메시지만 받고 반환한다.

    datum은 고정값(움직이지 않음)이라 지속 구독이 필요 없다 — 한 번 받으면 그걸로 끝.
    토픽이 없거나(구버전 GCS 서버), `timeout`초 안에 못 받으면 `None`(호출부가 "이
    확인이 안 됐다"고 경고할 수 있게).
    """
    if not ROS2_AVAILABLE:
        return None
    own_context = not rclpy.ok()
    if own_context:
        rclpy.init()
    node = rclpy.create_node("mobrobgpt_datum_probe")
    got: dict = {}

    def _cb(msg: "NavSatFix") -> None:
        got["lat"] = msg.latitude
        got["lon"] = msg.longitude
        got["alt"] = msg.altitude

    # BEST_EFFORT: gcs의 다른 텔레메트리 퍼블리셔들과 같은 QoS -- RELIABLE로 구독하면
    # BEST_EFFORT 퍼블리셔와 호환이 깨져 메시지를 하나도 못 받는다(실측 확인된 오류:
    # "New publisher discovered ... offering incompatible QoS ... RELIABILITY").
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=5)
    sub = node.create_subscription(NavSatFix, f"/{namespace}/datum/gps/fix", _cb, qos)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and not got:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        if own_context:
            rclpy.shutdown()
    return got or None


__all__ = ["fetch_gcs_datum"]
