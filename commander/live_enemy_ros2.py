"""commander/live_enemy_ros2.py — 라이브 ROS2 토픽 구독 기반 적선 리더.

`commander/bag_replay.py::BagEnemyReplay`와 똑같은 duck-type 인터페이스
(`n_tracks`/`duration_sec`/`first_common_time_sec`/`sample()`/`centroid()`/
`max_radius_from()`/`mean_speed_mps()`)를 제공하지만, .db3 파일을 오프라인으로
한 번에 읽는 대신 `/usv/usv{i}/pose`(PoseStamped, ENU m)와
`/usv/usv{i}/heading`(Float64, deg, nav 규약)를 **실시간으로 구독**해 최신값을
들고 있는다. `commander/gcs_bridge.py`/`commander/replay_cnn_env.py`는 이 두
클래스를 구분하지 않고 그대로 받는다(둘 다 `bag` 자리에 duck-typing으로 꽂힌다).

왜 필요한가 — `BagEnemyReplay`는 스크립트가 실행되는 순간을 자기만의 t=0으로 잡아
.db3 파일을 독립적으로 재생한다. 그런데 `tests/simulation/mock/run_mixed_demo.sh`
같은 실제 운용 환경은 같은 로스백을 `ros2 bag play -l`로 이미 무한 루프 재생 중이고,
GCS 지도(usv4-8)는 `bag_enemy_relay.py`가 그 라이브 토픽을 그대로 중계한 것이다.
두 재생이 서로 다른 시계로 따로 돌면 우리 정책이 겨냥하는 위치와 지도에 실제로
보이는 위치가 어긋난다(2026-09-09 세션, 아키텍트 검토로 확인된 실결함 — "좌표계가
안 맞는" 것처럼 보였던 원인). 이 클래스는 오프라인 재생 대신 그 라이브 피드를 직접
구독해 이 시간 축 불일치를 원천적으로 없앤다.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class LiveEnemyReplay:
    """`/usv/usv{i}/pose`+`/usv/usv{i}/heading`를 라이브 구독하는 적선 M척 리더.

    `BagEnemyReplay`와 인터페이스가 같지만 의미가 다른 지점 두 곳:
      · `sample(t_real_sec)` — `t_real_sec` 인자는 무시한다(라이브 데이터엔 "그 시각"이
        없다, 항상 "지금 가장 최근 값"을 돌려준다). 호출부(`ReplayCnnEnv`)는 자기
        시뮬 시계를 그대로 넘기지만 여기선 그냥 버려진다 — duck-typing 호환용 자리.
      · `duration_sec = inf` — 라이브 피드는 절대 "소진"되지 않는다. 종료는
        `--max-decisions`나 Ctrl+C로 한다(`bag_exhausted()`가 항상 False).
    """

    def __init__(
        self,
        n_tracks: int = 5,
        first_index: int = 1,
        pose_topic_fmt: str = "/usv/usv{i}/pose",
        heading_topic_fmt: str = "/usv/usv{i}/heading",
        stale_timeout: float = 1.0,
        ready_timeout: float = 15.0,
    ):
        if not ROS2_AVAILABLE:
            raise ImportError(
                "rclpy/geometry_msgs/std_msgs 를 불러올 수 없습니다. ROS2 환경을 먼저 "
                "source 하세요: `source /opt/ros/humble/setup.bash` (python3.10 필요).")

        self.indices = list(range(first_index, first_index + n_tracks))
        self.pose_topic_fmt = pose_topic_fmt
        self.heading_topic_fmt = heading_topic_fmt
        self.stale_timeout = float(stale_timeout)
        # BagEnemyReplay와 이름을 맞춘 duck-type 필드 — 값의 의미는 클래스 docstring 참고.
        self.duration_sec = float("inf")
        self.first_common_time_sec = 0.0

        self._lock = threading.Lock()
        self._pos: dict[int, tuple[float, float] | None] = {i: None for i in self.indices}
        self._pos_t: dict[int, float] = {i: 0.0 for i in self.indices}
        self._hdg: dict[int, float] = {i: 0.0 for i in self.indices}
        self._hdg_seen: dict[int, bool] = {i: False for i in self.indices}
        self._speed_hist: dict[int, list[float]] = {i: [] for i in self.indices}

        if not rclpy.ok():
            rclpy.init()
        self._node = _LiveEnemyNode(self)
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

        self._wait_ready(ready_timeout)

    # ── 준비 대기 ────────────────────────────────────────────────────
    def _wait_ready(self, timeout: float) -> None:
        """전 트랙이 pose+heading 을 최소 1회씩 받을 때까지 블로킹 대기.

        `ReplayCnnEnv.__init__`은 생성 직후 바로 `_inject_replay_enemies()`를 부르므로,
        여기서 미리 채워두지 않으면 첫 결정이 빈 관측(e_alive 전부 False)으로 굳어진다
        (`BagEnemyReplay.first_common_time_sec`가 오프라인 쪽에서 하던 것과 같은 보장).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if all(self._pos[i] is not None and self._hdg_seen[i]
                       for i in self.indices):
                    return
            time.sleep(0.1)
        with self._lock:
            missing = [i for i in self.indices
                       if self._pos[i] is None or not self._hdg_seen[i]]
        raise TimeoutError(
            f"{timeout:g}초 안에 pose/heading 을 못 받은 트랙: {missing} — 토픽 이름"
            f"(pose_topic_fmt/heading_topic_fmt)과 발행 쪽(예: `ros2 bag play -l`)이 "
            f"돌고 있는지 확인하세요.")

    # ── 콜백에서 호출 (락으로 보호) ──────────────────────────────────
    def _on_pose(self, idx: int, x: float, y: float, t_mono: float) -> None:
        with self._lock:
            prev = self._pos[idx]
            prev_t = self._pos_t[idx]
            self._pos[idx] = (x, y)
            self._pos_t[idx] = t_mono
            if prev is not None and t_mono > prev_t:
                dt = t_mono - prev_t
                if dt > 1e-3:
                    d = math.hypot(x - prev[0], y - prev[1])
                    hist = self._speed_hist[idx]
                    hist.append(d / dt)
                    if len(hist) > 50:      # 최근 표본만 유지 — mean_speed_mps()용
                        del hist[0]

    def _on_heading(self, idx: int, deg: float, t_mono: float) -> None:
        with self._lock:
            self._hdg[idx] = deg
            self._hdg_seen[idx] = True

    # ── BagEnemyReplay 호환 인터페이스 ───────────────────────────────
    @property
    def n_tracks(self) -> int:
        return len(self.indices)

    def sample(self, t_real_sec: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`t_real_sec`는 duck-typing 호환용으로만 받고 쓰지 않는다 — 항상 최신값을 반환."""
        now = time.monotonic()
        M = self.n_tracks
        xy = np.zeros((M, 2), dtype=np.float64)
        hdg = np.zeros(M, dtype=np.float64)
        alive = np.zeros(M, dtype=bool)
        with self._lock:
            for k, i in enumerate(self.indices):
                p = self._pos[i]
                if p is None or (now - self._pos_t[i]) > self.stale_timeout:
                    continue                # 신호 유실 — BagEnemyReplay의 _max_gap_ns와 동격
                xy[k] = p
                alive[k] = True
                hdg[k] = self._hdg[i]
        return xy, hdg, alive

    def centroid(self) -> tuple[float, float]:
        xy, _, alive = self.sample()
        if not alive.any():
            return (0.0, 0.0)
        pts = xy[alive]
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def max_radius_from(self, origin: tuple[float, float]) -> float:
        """현재 스냅샷 기준 반경. 오프라인 `BagEnemyReplay`와 달리 전체 궤적을 모르므로
        (라이브라 미래를 알 수 없다) --span 적정성의 하한 참고치일 뿐이다."""
        xy, _, alive = self.sample()
        if not alive.any():
            return 0.0
        o = np.asarray(origin, dtype=np.float64)
        pts = xy[alive]
        return float(np.max(np.hypot(*(pts - o).T)))

    def mean_speed_mps(self, window_sec: float = 1.0) -> float:
        with self._lock:
            speeds = [float(np.mean(h)) for h in self._speed_hist.values() if h]
        return float(np.mean(speeds)) if speeds else 0.0

    def shutdown(self) -> None:
        """`rclpy.spin()`은 컨텍스트가 살아있는 동안 블로킹하므로, 노드부터 destroy 하면
        스핀 스레드가 죽은 노드를 계속 들고 있다가 인터프리터 종료 시 native abort
        (core dump)로 이어진다 — 반드시 `rclpy.shutdown()`을 먼저 불러 스핀을 빠져나오게
        하고, 스레드가 실제로 끝난 뒤에 노드를 destroy 하는 순서를 지킨다."""
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self._spin_thread.join(timeout=2.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass


if ROS2_AVAILABLE:
    class _LiveEnemyNode(Node):
        """구독 전용 노드. `LiveEnemyReplay`가 락으로 상태를 관리하므로 콜백은 얇게 둔다."""

        def __init__(self, owner: LiveEnemyReplay):
            super().__init__("mobrobgpt_live_enemy_reader")
            # BEST_EFFORT: `ros2 bag play`가 원본 QoS로 재발행하는데, 발행측이 RELIABLE여도
            # BEST_EFFORT 구독은 항상 호환된다(반대는 아니다) — 녹화 당시 QoS를 몰라도 안전.
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
            self._subs = []
            for i in owner.indices:
                pose_topic = owner.pose_topic_fmt.format(i=i)
                hdg_topic = owner.heading_topic_fmt.format(i=i)
                self._subs.append(self.create_subscription(
                    PoseStamped, pose_topic,
                    lambda msg, idx=i: owner._on_pose(
                        idx, msg.pose.position.x, msg.pose.position.y, time.monotonic()),
                    qos))
                self._subs.append(self.create_subscription(
                    Float64, hdg_topic,
                    lambda msg, idx=i: owner._on_heading(idx, float(msg.data), time.monotonic()),
                    qos))


__all__ = ["LiveEnemyReplay"]
