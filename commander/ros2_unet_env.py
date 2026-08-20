"""ROS2 실센서 + 'CNN 점수맵' 정책(CnnScoreActor / U-Net lite) 통합 환경.

`run_commander_ui.py --unet --ros2` 에서 사용.

설계 — 왜 ros2_cell_env.py 와 다른가
────────────────────────────────────
`ros2_cell_env.ROS2CellEnv` 는 배정·유효마스크·관측을 duck-type 객체 위에 **손으로 다시 구현**한다.
CNN 점수맵 정책에서는 그 길을 쓰지 않는다. `docs/unet_model_deploy.md §7-A` 가 명시하듯,
배정(`_compute_assignment`)·유효마스크(`_cnn_valid_mask`)의 폴백 사다리 순서 하나만 어긋나도
모델은 **에러 없이 조용히 엉뚱한 곳을 찍는다.**

그래서 시뮬용 `CommandedCnnEnv` 를 **그대로 상속**하고, `step()` 만 갈아끼운다:
  · 시뮬 물리(`_micro`)를 돌리지 않는다 — 상태는 실센서가 준다.
  · 매 tick 센서 값을 env 의 상태 배열에 써 넣는다(a_pos/a_hdg/e_pos/e_hdg/…).
  · 결정주기마다 상속받은 `_rl_decide()` 를 그대로 호출 → 시뮬 경로와 **완전히 동일한 추론**.
  · 나온 route/net_mask 를 ROS2 `/ally_X/waypoints` + MQTT 로 발행.

데이터 흐름:
  /enemy_X/fix, /ally_X/fix, /ally_X/imu, /mothership/fix
    → ROS2SensorBridge(+GeoBridge affine) → sim 좌표
    → env 상태 배열 주입 → build_cnn_obs → CnnScoreActor.greedy → 픽셀 2점
    → pix_to_routes → route/net_mask → ROS2SensorBridge → /ally_X/waypoints

센서가 주지 않아 직접 관리하는 상태 3개 (§10.2):
  a_nets(남은 그물) · doing_net(부설 중) · net_installed[200,200](이미 깐 그물).
  마지막 것을 빼먹으면 정책이 이미 친 벽을 못 보고 겹쳐 친다.
"""
from __future__ import annotations

import numpy as np

from .ros2_sensor_bridge import create_ros2_bridge, ROS2SensorBridge
from .unet_bridge import CommandedCnnEnv


class ROS2CnnEnv(CommandedCnnEnv):
    """실센서로 상태를 채우고 CNN 점수맵 정책으로 WP 를 내는 운용 환경(1월드)."""

    def __init__(self, ckpt: str = "boatattack_sim/models/u-net_map.pt",
                 enemy_mode: str = "wave", device: str = "cpu",
                 apf: bool = False, imu_frame: str = "NED",
                 nets_per_ship: int = 3):
        super().__init__(ckpt, enemy_mode=enemy_mode, device=device, avoid_steer=apf,
                         nets_per_ship=nets_per_ship)
        self.reset(seed=0)                 # 배열 할당 + 육지 캐시 로드. 이후 물리는 안 돌린다
        self._bridge: ROS2SensorBridge | None = None
        self._imu_frame = imu_frame
        self._tick = 0
        self._have_sensors = False
        # 내부 관리 상태 — 센서가 주지 않는다.
        #   a_nets 는 배가 싣고 나온 그물 장수. 한 장 완성할 때마다 1 줄고, 0 이 되면
        #   정지 규칙(assign<0 과 동급)에 걸려 그 배는 더 이상 WP 를 받지 않는다.
        self.a_nets[0] = self.cfg.nets_per_ship
        self.doing_net[0] = False
        self.net_installed[0] = False

    # ── ROS2 ─────────────────────────────────────────────────────────
    def start_ros2(self) -> ROS2SensorBridge | None:
        self._bridge = create_ros2_bridge(
            n_allies=self.P, n_enemies=self.M, world_size=float(self.cfg.world_size),
            on_state_update=lambda: None,          # 상태 반영은 step() 에서 일괄
            imu_frame=self._imu_frame)
        print(f"[ROS2CnnEnv] ROS2 브릿지 시작 (IMU: {self._imu_frame})")
        return self._bridge

    def _ingest_sensors(self) -> bool:
        """센서 스냅샷 → env 상태 배열. 학습 계약대로 적 슬롯 M개를 고정한다."""
        if self._bridge is None:
            return False
        st = self._bridge.get_sim_state()
        if st is None:
            return False
        self.a_pos[0] = st["ally_pos"]
        self.a_hdg[0] = st["ally_hdg"]
        self.a_alive[0] = st["ally_alive"]
        # ★ 적 슬롯 M개 고정 — 넘치면 모선 근접순 상위 M, 모자라면 e_alive=False.
        #   래스터 관측이라 순서는 무관하다(순열 불변).
        xy = np.asarray(st["enemy_pos"], np.float64)
        hdg = np.asarray(st["enemy_hdg"], np.float64)
        ok = np.asarray(st["enemy_alive"], bool)
        c = np.asarray(self.center, np.float64)
        idx = np.where(ok)[0]
        if len(idx) > self.M:
            d = np.hypot(*(xy[idx] - c).T)
            idx = idx[np.argsort(d)[:self.M]]
        self.e_pos[0] = 0.0; self.e_hdg[0] = 0.0; self.e_alive[0] = False
        n = len(idx)
        self.e_pos[0, :n] = xy[idx]
        self.e_hdg[0, :n] = hdg[idx]
        self.e_alive[0, :n] = True
        # 모선 위치가 토픽으로 오면 반영 (없으면 브릿지가 sim 중앙을 준다)
        self.center = np.asarray(st["center"], np.float64)
        return True

    # ── WP 진행 + 그물 상태 (센서가 주지 않는다) ──────────────────────
    def _advance_and_paint(self):
        """WP 도착 판정(ptr 전진) + 그물 부설 진행. `_micro` 의 해당 블록을 실기용으로 옮긴 것.

        시뮬에서는 `DefenseVecEnv._micro` 가 기동·도착·부설을 한 번에 처리하지만, 실기에서는
        기동을 실제 USV 가 하므로 **관측(센서 위치)으로부터 같은 상태 전이를 재구성**해야 한다.
        순서·조건은 `_micro` 와 1:1로 맞춘다 — 어긋나면 정책이 보는 net_installed/leg_netted 가
        학습 때와 달라진다.

        그물 진행량(paint_dist)은 실기에 릴 엔코더가 없으므로 **항행 거리로 근사**한다.
        실제 계측값이 있으면 이 줄을 그 값으로 대체할 것.
        """
        cfg = self.cfg
        moved = np.hypot(*(self.a_pos[0] - self._prev_pos).T)          # [P] 이번 tick 이동거리
        self._prev_pos = self.a_pos[0].copy()

        ptr_c = np.clip(self.ptr, 0, self.Kw - 1)
        route_t = np.take_along_axis(self.route, ptr_c[:, :, None, None], axis=2)[:, :, 0, :]
        self.net_end = route_t.copy()
        arrived = (np.hypot(self.a_pos[..., 0] - route_t[..., 0],
                            self.a_pos[..., 1] - route_t[..., 1]) <= cfg.arrive_radius)

        # ① 그물 시작 — 현재 leg 의 net_mask 가 켜졌고 아직 안 깐 배 (_micro 와 동일 조건)
        cur_net = np.take_along_axis(self.net_mask, ptr_c[..., None], axis=2)[..., 0]   # [N,P]
        start = (cur_net & (~self.leg_netted) & (~self.doing_net)
                 & self.a_alive & (self.a_nets > 0))
        if start.any():
            self.doing_net |= start
            self.leg_netted |= start
            self.paint_dist[start] = 0.0
            self.net_start[start] = self.a_pos[start]
            self.a_nets -= start.astype(np.int64)
            self.stats["nets_used"] += int(start.sum())

        # ② 부설 진행 — 이동거리 누적 + 궤적 도색
        painting = self.doing_net & (self.paint_dist < cfg.net_max_len) & self.a_alive
        self.paint_dist += np.where(painting, moved[None, :], 0.0)
        self._paint(painting)

        # ③ 완성 — 길이한계 또는 목표 WP 도착 → net_installed 에 띠로 등록
        finish = self.doing_net & ((self.paint_dist >= cfg.net_max_len) | arrived)
        if finish.any():
            self.doing_net &= ~finish
            self._rasterize_net(self.net_start, self.net_end, finish)
            self.prev_on_inst |= finish

        # ④ WP 도착 → 다음 WP (일방통행: Kw-1 에서 멈춤)
        arr_route = (~self.doing_net) & arrived
        advance = arr_route & (self.ptr < self.Kw - 1)
        if arr_route.any():
            ai, aj = np.where(arr_route)
            self.wp_reached[ai, aj, ptr_c[ai, aj]] = True
        self.ptr = np.where(advance, self.ptr + 1, self.ptr)
        self.leg_netted = np.where(advance, False, self.leg_netted)

    # ── 운용 루프 ────────────────────────────────────────────────────
    def publish_waypoints(self):
        if self._bridge is not None:
            self._bridge.publish_waypoints(self.route[0], self.net_mask[0])

    def step(self):
        """UI 애니메이션 루프가 부르는 tick. 시뮬 물리(_micro)는 **돌리지 않는다.**"""
        if not self._ingest_sensors():
            return self.get_frame()                # 센서 미수신 — 마지막 상태 유지
        if not self._have_sensors:                 # 첫 수신 시점을 t=0 으로
            self._have_sensors = True
            self._prev_pos = self.a_pos[0].copy()
            self._tick = 0
        self._advance_and_paint()
        if self._tick % self.cfg.decision_period == 0 and self._plan is not None:
            self._rl_decide()                      # 시뮬 경로와 완전히 동일한 추론
            self.publish_waypoints()
        self._tick += 1
        self.t[0] = self._tick
        self.stats["survived"] = int(self.e_alive[0].sum())
        return self.get_frame()

    def reset(self, seed=None):
        super().reset(seed)
        self._tick = 0
        self._have_sensors = False
        self._prev_pos = self.a_pos[0].copy()
        self.a_nets[0] = self.cfg.nets_per_ship
        self.doing_net[0] = False
        self.net_installed[0] = False


__all__ = ["ROS2CnnEnv"]
