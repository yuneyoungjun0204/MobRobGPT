"""commander/bag_replay.py — 실제 로스백 리플레이(적선 트랙) → sim 상태 주입용 샘플러.

`/usv/usv{i}/pose`(geometry_msgs/PoseStamped, `map` 프레임 미터) +
`/usv/usv{i}/heading`(std_msgs/Float64, deg)를 읽어 임의 시각의 위치/헤딩을 최근접
샘플로 반환한다.

heading 토픽은 실측으로 이미 nav 규약(0°=North, CW+)과 일치함을 확인했다 —
usv1 샘플에서 attitude 쿼터니언으로 계산한 ENU yaw(≈101.8°)를
`geo_bridge.hdg_to_sim`(= (90 - yaw_enu) % 360)으로 변환하면 348.2°가 되어
heading 토픽 값(347.98°)과 일치한다. 따라서 추가 변환 없이 heading 을 그대로
sim heading 으로 쓴다.

`pose.position`은 이미 로컬 미터 좌표(모선 근방 원점)라 GPS 변환도 필요 없다 —
`SimScale.enu_to_sim`에 바로 넣을 수 있는 "실 ENU" 값으로 취급한다.

다른 로스백/다른 척수/다른 토픽 이름에도 재사용하도록 전부 인자화한다.
"""
from __future__ import annotations

import numpy as np


class BagEnemyReplay:
    """rosbag2(.db3) 안의 실 적선 N척 pose+heading 시계열 → t초 시점 샘플."""

    def __init__(
        self,
        bag_path: str,
        n_tracks: int = 5,
        first_index: int = 1,
        pose_topic_fmt: str = "/usv/usv{i}/pose",
        heading_topic_fmt: str = "/usv/usv{i}/heading",
        storage_id: str = "sqlite3",
    ):
        self.bag_path = str(bag_path)
        self.indices = list(range(first_index, first_index + n_tracks))
        self.pose_topic_fmt = pose_topic_fmt
        self.heading_topic_fmt = heading_topic_fmt
        self.storage_id = storage_id
        self.t0_ns = 0
        self.duration_sec = 0.0
        self._pos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._hdg: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._load()

    @property
    def n_tracks(self) -> int:
        return len(self.indices)

    def _load(self) -> None:
        try:
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        except ImportError as e:
            raise ImportError(
                "rosbag2_py/rclpy 를 불러올 수 없습니다. ROS2 환경을 먼저 source 하세요: "
                "`source /opt/ros/humble/setup.bash` (python3.10 필요)."
            ) from e

        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self.bag_path, storage_id=self.storage_id),
            ConverterOptions("cdr", "cdr"),
        )
        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

        pose_topics = {i: self.pose_topic_fmt.format(i=i) for i in self.indices}
        hdg_topics = {i: self.heading_topic_fmt.format(i=i) for i in self.indices}
        missing = [t for t in list(pose_topics.values()) + list(hdg_topics.values())
                   if t not in type_map]
        if missing:
            raise ValueError(
                f"로스백 '{self.bag_path}' 에 없는 토픽: {missing} "
                f"(사용 가능: {sorted(type_map)})"
            )

        pose_topic_to_i = {v: k for k, v in pose_topics.items()}
        hdg_topic_to_i = {v: k for k, v in hdg_topics.items()}

        pos_t: dict[int, list] = {i: [] for i in self.indices}
        pos_xy: dict[int, list] = {i: [] for i in self.indices}
        hdg_t: dict[int, list] = {i: [] for i in self.indices}
        hdg_v: dict[int, list] = {i: [] for i in self.indices}

        msg_cache = {}
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            i = pose_topic_to_i.get(topic)
            if i is not None:
                msg_type = msg_cache.setdefault(topic, get_message(type_map[topic]))
                msg = deserialize_message(data, msg_type)
                pos_t[i].append(t_ns)
                pos_xy[i].append((msg.pose.position.x, msg.pose.position.y))
                continue
            i = hdg_topic_to_i.get(topic)
            if i is not None:
                msg_type = msg_cache.setdefault(topic, get_message(type_map[topic]))
                msg = deserialize_message(data, msg_type)
                hdg_t[i].append(t_ns)
                hdg_v[i].append(float(msg.data))

        starts = [pos_t[i][0] for i in self.indices if pos_t[i]]
        if not starts:
            raise ValueError(f"로스백 '{self.bag_path}' 에서 pose 메시지를 하나도 못 읽었습니다.")
        self.t0_ns = min(starts)
        ends = [pos_t[i][-1] for i in self.indices if pos_t[i]]
        self.duration_sec = (max(ends) - self.t0_ns) / 1e9

        self._max_gap_ns: dict[int, int] = {}
        for i in self.indices:
            ta = np.asarray(pos_t[i], dtype=np.int64)
            self._pos[i] = (ta, np.asarray(pos_xy[i], dtype=np.float64).reshape(-1, 2))
            self._hdg[i] = (np.asarray(hdg_t[i], dtype=np.int64),
                             np.asarray(hdg_v[i], dtype=np.float64))
            # 트랙 중단(탐지 유실) 감지용 — 정상 간격의 3배 또는 1초 중 큰 쪽보다 벌어지면
            # "그 시각에 가장 가까운 표본이 있다"가 아니라 "신호가 끊겼다"로 본다.
            if ta.size >= 2:
                self._max_gap_ns[i] = max(int(np.median(np.diff(ta))) * 3, int(1e9))
            else:
                self._max_gap_ns[i] = int(1e9)

    @staticmethod
    def _nearest(ts: np.ndarray, t_query: int, clamp: bool = False) -> int | None:
        """`ts`에서 `t_query`에 가장 가까운 인덱스. `clamp=False`면 범위 밖은 None(=미탐지).

        `clamp=True`면 범위 밖에서도 가장 가까운 끝 인덱스를 돌려준다(마지막 known 값 유지) —
        heading 처럼 "이 트랙이 살아있다는 건 pose 로 이미 확인했고, 그 순간 heading 표본이
        마침 없을 뿐"인 경우에 쓴다. 여기서 None/0.0 을 반환하면 실제로는 방위가 있는 배가
        (거짓으로) 정북(0°)을 향한 것처럼 보여 클러스터링·배정이 흔들린다.
        """
        if ts.size == 0:
            return None
        if t_query < ts[0]:
            return 0 if clamp else None
        if t_query > ts[-1]:
            return ts.size - 1 if clamp else None
        j = int(np.searchsorted(ts, t_query))
        j = min(max(j, 0), ts.size - 1)
        if j > 0 and abs(int(ts[j - 1]) - t_query) < abs(int(ts[j]) - t_query):
            j -= 1
        return j

    @property
    def first_common_time_sec(self) -> float:
        """전 트랙이 전부 살아있게 되는 최초 시각(로스백 시작 기준 초).

        척마다 pose 발행 시작 시각이 몇 ms 씩 어긋난다(실측: 0~2.6 ms) — `sample(0.0)`을
        그대로 쓰면 아직 안 켜진 트랙이 몇 개 빠진 상태로 첫 배정이 결정된다. 리플레이는
        이 시각부터 시작해 첫 결정부터 전 척을 본다.
        """
        starts = [ts[0] for ts, _ in self._pos.values() if len(ts)]
        if not starts:
            return 0.0
        return (max(starts) - self.t0_ns) / 1e9

    def max_radius_from(self, origin: tuple[float, float]) -> float:
        """전 트랙·전 구간 pose 중 `origin`에서 가장 먼 거리[m]. `--span` 적정성 점검용."""
        pts = [vals for ts, vals in self._pos.values() if len(ts)]
        if not pts:
            return 0.0
        allp = np.concatenate(pts, axis=0)
        o = np.asarray(origin, dtype=np.float64)
        return float(np.max(np.hypot(*(allp - o).T)))

    def mean_speed_mps(self, window_sec: float = 1.0) -> float:
        """트랙별 평균 속력[m/s]의 트랙 평균(`SimScale.report()` 대조용 실측치).

        연속 표본 간 순간속력을 그대로 평균하면 위치 양자화 잡음(이 로스백은 ~1 cm 단위로
        반올림돼 표본의 82%가 이전과 완전히 같은 값이다)이 누적돼 실제 이동보다 몇 배
        부풀려진다. `window_sec` 간격으로 재표본해 그 구간의 순변위/시간으로 계산하면
        잡음이 상쇄된다.
        """
        speeds = []
        for ts, xy in self._pos.values():
            if ts.size < 2:
                continue
            t0, t1 = int(ts[0]), int(ts[-1])
            span = (t1 - t0) / 1e9
            if span < window_sec:
                d = float(np.hypot(*(xy[-1] - xy[0])))
                if span > 1e-6:
                    speeds.append(d / span)
                continue
            n_win = max(1, int(span / window_sec))
            grid = np.linspace(t0, t1, n_win + 1).astype(np.int64)
            idx = [self._nearest(ts, g) for g in grid]
            idx = [i for i in idx if i is not None]
            if len(idx) < 2:
                continue
            pts = xy[idx]
            real_dt = np.diff(ts[idx]).astype(np.float64) / 1e9
            d = np.hypot(*np.diff(pts, axis=0).T)
            ok = real_dt > 1e-6
            if ok.any():
                speeds.append(float(np.mean(d[ok] / real_dt[ok])))
        return float(np.mean(speeds)) if speeds else 0.0

    def centroid(self) -> tuple[float, float]:
        """전 트랙·전 구간 pose 평균 좌표(local-m). 모선 위치를 모를 때의 기본 원점 후보.

        운용 해역(모선/방어 목표)이 로스백 원점(0,0)과 멀리 떨어져 있을 수 있으므로 —
        SimScale 의 `enu_origin` 을 이걸로 잡으면 실제 적선 전체가 sim 격자 중심 부근에
        오도록 재배치된다.
        """
        pts = [vals for ts, vals in self._pos.values() if len(ts)]
        if not pts:
            return (0.0, 0.0)
        allp = np.concatenate(pts, axis=0)
        return (float(allp[:, 0].mean()), float(allp[:, 1].mean()))

    def sample(self, t_real_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """t_real_sec(로스백 시작 기준 실경과초) 시점의 (xy[M,2] local-m, hdg_deg[M], alive[M]).

        `alive`는 "이 트랙의 전체 시간범위 안"이 아니라 "그 순간 근처에 실제 표본이 있었다"를
        뜻한다 — 범위 안이라도 표본 간격(§`_max_gap_ns`)보다 멀면 신호 유실로 보고 False.
        """
        t_query = self.t0_ns + int(round(t_real_sec * 1e9))
        M = self.n_tracks
        xy = np.zeros((M, 2), dtype=np.float64)
        hdg = np.zeros(M, dtype=np.float64)
        alive = np.zeros(M, dtype=bool)
        for k, i in enumerate(self.indices):
            ts, vals = self._pos[i]
            j = self._nearest(ts, t_query)
            if j is None or abs(int(ts[j]) - t_query) > self._max_gap_ns[i]:
                continue
            xy[k] = vals[j]
            alive[k] = True
            hts, hvals = self._hdg[i]
            hj = self._nearest(hts, t_query, clamp=True)   # 트랙 생존 확정 → heading 은 hold
            if hj is not None:
                hdg[k] = hvals[hj]
        return xy, hdg, alive


__all__ = ["BagEnemyReplay"]
