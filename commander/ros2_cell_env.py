"""
commander/ros2_cell_env.py — ROS2 센서 + 셀 정책 통합 환경

--cell --ros2 모드에서 사용:
  - ROS2 센서 데이터로 전장 상태 수신
  - 셀 정책(CellPointerActor)으로 경로/그물 결정
  - ROS2 waypoints 토픽으로 명령 발행

데이터 흐름:
  /enemy_X/fix, /ally_X/fix, /ally_X/imu → ROS2SensorBridge → 시뮬 좌표
    → 셀 정책 관측 생성 → CellPointerActor 추론 → 경로/그물 결정
    → ROS2SensorBridge → /ally_X/waypoints 발행
"""

import numpy as np
import torch
from typing import Optional, Callable

from .ros2_sensor_bridge import create_ros2_bridge, ROS2SensorBridge, ROS2_AVAILABLE
from .ros2_env import build_battlefield_ros2
from .sim_bridge import plan_to_assign
from .schema import BattlefieldState

# 셀 정책 로드
try:
    from boatattack_sim.model.cell_actor import load_cell_actor, cell_obs_to_torch
    CELL_AVAILABLE = True
except ImportError:
    CELL_AVAILABLE = False
    print("[ros2_cell_env] cell_actor not available")


class ROS2CellEnv:
    """ROS2 센서 + 셀 정책 통합 환경.

    ROS2 센서 데이터를 받아서 셀 정책으로 경로/그물 결정 후
    ROS2 waypoints로 발행한다.
    """

    def __init__(
        self,
        ckpt: str = "boatattack_sim/models/cell_latest.pt",
        n_allies: int = 3,
        n_enemies: int = 10,
        world_size: float = 12600.0,
        device: str = "cpu",
        imu_frame: str = "NED",
    ):
        if not CELL_AVAILABLE:
            raise ImportError("boatattack_sim.model.cell_actor not available")

        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self.world_size = world_size
        self.device = device

        # 셀 정책 로드
        print(f"[ROS2CellEnv] 셀 정책 로딩: {ckpt}")
        self._actor, self._cfg = load_cell_actor(ckpt, device=device)
        self._actor.eval()

        # 설정 객체 (시뮬레이터 호환용)
        self.cfg = type("Cfg", (), {
            "world_size": world_size,
            "mothership_radius": 260.0,
            "n_allies": n_allies,
            "n_enemies": n_enemies,
            "n_clusters": 3,
            "cluster_gap_deg": 11.97,
            "enemy_speed": 9.0,
            "ally_speed": 6.0,
            "ship_len": 230.0,
            "ship_wid": 76.0,
            "enemy_size": 125.0,
            "moback_size": 380.0,
            "moback_heading": 0.0,
            "geo_lat": 34.625,
            "geo_lon": 128.52,
            "nets_per_ship": 5,
            "transit_wp": 6,
            "cell_spacing": getattr(self._cfg, "cell_spacing", 473.0),
            "cell_action": True,
        })()

        # ROS2 브릿지
        self._bridge: Optional[ROS2SensorBridge] = None
        self._imu_frame = imu_frame

        # 내부 상태
        self.P = n_allies
        self.M = n_enemies
        self.Kw = 6  # waypoint 개수

        # 아군/적 상태 (시뮬 좌표)
        self.a_pos = np.zeros((self.P, 2))
        self.a_hdg = np.zeros(self.P)
        self.a_alive = np.ones(self.P, dtype=bool)
        self.a_nets = np.full(self.P, 5)

        self.e_pos = np.zeros((self.M, 2))
        self.e_hdg = np.zeros(self.M)
        self.e_alive = np.ones(self.M, dtype=bool)

        # 경로/그물
        self.route = np.zeros((self.P, self.Kw, 2))
        self.net_mask = np.zeros((self.P, self.Kw), dtype=bool)
        self.doing_net = np.zeros(self.P, dtype=bool)

        # 배정
        self._assign = np.full(self.P, -1, dtype=np.int64)
        self._assignI = np.zeros((self.P, 2))

        # 모선
        self.center = np.array([world_size / 2, world_size / 2])
        self.mothership_radius = 260.0

        # 계획
        self._plan = None
        self._plan_command = None

        # 상태
        self.running = True
        self.done = False
        self.t = 0
        self.stats = {
            "captures": 0,
            "breaches": 0,
            "ally_collisions": 0,
            "nets_used": 0,
            "survived": 0,
        }

        # 셀 정책 상태
        self._h = self._actor.init_hidden(self.P, device)
        self._last_cells = None
        self._decision_period = int(getattr(self._cfg, "decision_period", 25))
        self._micro_ct = 0

    def start_ros2(self) -> Optional[ROS2SensorBridge]:
        """ROS2 브릿지 시작."""
        def on_update():
            pass  # 상태 업데이트는 step()에서 처리

        self._bridge = create_ros2_bridge(
            n_allies=self.n_allies,
            n_enemies=self.n_enemies,
            world_size=self.world_size,
            on_state_update=on_update,
            imu_frame=self._imu_frame,
        )
        print(f"[ROS2CellEnv] ROS2 브릿지 시작 (IMU: {self._imu_frame})")
        return self._bridge

    def update_from_ros2(self) -> bool:
        """ROS2 센서 데이터로 상태 업데이트."""
        if self._bridge is None:
            return False

        sim_state = self._bridge.get_sim_state()
        if sim_state is None:
            return False

        # 상태 업데이트
        self.a_pos = sim_state["ally_pos"]
        self.a_hdg = sim_state["ally_hdg"]
        self.a_alive = sim_state["ally_alive"]

        self.e_pos = sim_state["enemy_pos"]
        self.e_hdg = sim_state["enemy_hdg"]
        self.e_alive = sim_state["enemy_alive"]

        self.center = sim_state["center"]

        return True

    def set_plan(self, plan, command: str = ""):
        """LLM 계획 설정 → 배정 계산."""
        self._plan = plan
        self._plan_command = command

        if plan is None:
            self._assign[:] = -1
            return

        # 현재 전장 상태 구축
        state = build_battlefield_ros2(self, command)

        # 배정 계산
        assign = plan_to_assign(plan, state)
        self._assign = np.array(assign, dtype=np.int64)

        # 요격점 계산
        for p in range(self.P):
            cid = int(self._assign[p])
            if cid >= 0 and cid < len(state.enemy_clusters):
                cl = state.enemy_clusters[cid]
                cx, cy = cl.center.x, cl.center.y
                mx, my = self.center[0], self.center[1]
                dx, dy = mx - cx, my - cy
                d = np.hypot(dx, dy)
                if d > 1.0:
                    v_a = self.cfg.ally_speed
                    v_e = self.cfg.enemy_speed
                    r = v_a * d / (v_a + v_e)
                    r = min(r, self.world_size / 3.0)
                    self._assignI[p] = np.array([mx - dx/d * r, my - dy/d * r])
                else:
                    self._assignI[p] = self.center.copy()
            else:
                self._assignI[p] = self.a_pos[p].copy()

        # 셀 정책으로 경로 생성
        self._run_cell_policy()

    def _run_cell_policy(self):
        """셀 정책 실행하여 경로/그물 결정."""
        if self._plan is None:
            # 계획 없으면 제자리 정지
            for p in range(self.P):
                self.route[p, :, :] = self.a_pos[p]
                self.net_mask[p, :] = False
            return

        # 셀 관측 생성
        obs = self._build_simple_obs()

        # 셀 정책 추론
        # CellPointerActor 사용법:
        #   1. forward(obs, hidden) → p (encoded), new_hidden
        #   2. sample(p) 또는 greedy(p) → {"cells": [B, K]}
        with torch.no_grad():
            obs_t = cell_obs_to_torch(obs, self.device)
            p, self._h = self._actor(obs_t, self._h)  # forward
            actions = self._actor.greedy(p)  # 결정적 선택
            cells = actions["cells"].cpu().numpy()  # [B, K] where B = N*P = 1*P = P

        # [P, K] 형태로 reshape (B=P 이므로 그대로)
        K = cells.shape[1] if len(cells.shape) > 1 else 1
        self._last_cells = cells.reshape(self.P, K)

        # 셀 좌표 → 경로 변환
        cell_world = self._get_cell_world()
        for p in range(self.P):
            if self._assign[p] < 0:
                # 예비: 제자리 정지
                self.route[p, :, :] = self.a_pos[p]
                self.net_mask[p, :] = False
            else:
                # 선택된 셀들로 경로 구성
                for k in range(min(self.Kw, K)):
                    cell_idx = int(self._last_cells[p, k])
                    cell_idx = min(max(cell_idx, 0), len(cell_world) - 1)
                    self.route[p, k] = cell_world[cell_idx]
                    self.net_mask[p, k] = True
                # 나머지 WP는 마지막 셀로
                for k in range(K, self.Kw):
                    self.route[p, k] = self.route[p, K - 1]
                    self.net_mask[p, k] = False

        # ROS2로 경로 발행
        self.publish_waypoints()

    def _build_simple_obs(self) -> dict:
        """셀 정책용 관측 생성.

        cell_obs_to_torch가 요구하는 형식:
          own: [N, P, 9] - pos2·head2·nets1·doing1 + 배정요격점2·할당플래그1
          ally: [N, P, P-1, 6]
          ally_mask: [N, P, P-1] bool
          enemy: [N, P, M, 6]
          enemy_mask: [N, P, M] bool
          cell: [N, P, n_cells, 5]
          cell_mask: [N, P, n_cells] bool
        """
        N = 1  # 단일 환경
        cell_world = self._get_cell_world()
        n_cells = len(cell_world)
        W = self.world_size
        cx, cy = self.center

        # own: [N, P, 9]
        own = np.zeros((N, self.P, 9), dtype=np.float32)
        # ally: [N, P, P-1, 6]
        ally = np.zeros((N, self.P, self.P - 1, 6), dtype=np.float32)
        ally_mask = np.ones((N, self.P, self.P - 1), dtype=bool)
        # enemy: [N, P, M, 6]
        enemy = np.zeros((N, self.P, self.M, 6), dtype=np.float32)
        enemy_mask = np.ones((N, self.P, self.M), dtype=bool)
        # cell: [N, P, n_cells, 5]
        cell = np.zeros((N, self.P, n_cells, 5), dtype=np.float32)
        cell_mask = np.ones((N, self.P, n_cells), dtype=bool)

        for p in range(self.P):
            px, py = self.a_pos[p]
            hdg_rad = np.radians(self.a_hdg[p])

            # own[9]: pos2·head2·nets1·doing1·배정요격점2·할당플래그1
            own[0, p, 0] = (px - cx) / W  # 모선 기준 상대 위치
            own[0, p, 1] = (py - cy) / W
            own[0, p, 2] = np.sin(hdg_rad)
            own[0, p, 3] = np.cos(hdg_rad)
            own[0, p, 4] = self.a_nets[p] / 5.0
            own[0, p, 5] = float(self.doing_net[p])
            # 배정 요격점 (상대 좌표)
            if self._assign[p] >= 0:
                own[0, p, 6] = (self._assignI[p, 0] - cx) / W
                own[0, p, 7] = (self._assignI[p, 1] - cy) / W
                own[0, p, 8] = 1.0  # 할당됨
            else:
                own[0, p, 6] = 0.0
                own[0, p, 7] = 0.0
                own[0, p, 8] = 0.0  # 미할당

            # ally[6]: 다른 아군 상태 (자신 제외)
            ai = 0
            for q in range(self.P):
                if q == p:
                    continue
                qx, qy = self.a_pos[q]
                q_hdg_rad = np.radians(self.a_hdg[q])
                ally[0, p, ai, 0] = (qx - cx) / W
                ally[0, p, ai, 1] = (qy - cy) / W
                ally[0, p, ai, 2] = np.sin(q_hdg_rad)
                ally[0, p, ai, 3] = np.cos(q_hdg_rad)
                ally[0, p, ai, 4] = self.a_nets[q] / 5.0
                ally[0, p, ai, 5] = float(self.a_alive[q])
                ally_mask[0, p, ai] = self.a_alive[q]
                ai += 1

            # enemy[6]: 적 상태
            for e in range(self.M):
                ex, ey = self.e_pos[e]
                e_hdg_rad = np.radians(self.e_hdg[e])
                enemy[0, p, e, 0] = (ex - cx) / W
                enemy[0, p, e, 1] = (ey - cy) / W
                enemy[0, p, e, 2] = np.sin(e_hdg_rad)
                enemy[0, p, e, 3] = np.cos(e_hdg_rad)
                # 적 → 모선 방향 (접근 속도 프록시)
                dx, dy = cx - ex, cy - ey
                d = np.hypot(dx, dy)
                if d > 1.0:
                    enemy[0, p, e, 4] = dx / d
                    enemy[0, p, e, 5] = dy / d
                enemy_mask[0, p, e] = self.e_alive[e]

            # cell[5]: 셀 상태
            for c in range(n_cells):
                cell_x, cell_y = cell_world[c]
                cell[0, p, c, 0] = (cell_x - cx) / W
                cell[0, p, c, 1] = (cell_y - cy) / W
                # 셀 → 모선 거리/방향
                dc = np.hypot(cell_x - cx, cell_y - cy)
                cell[0, p, c, 2] = dc / W
                # 셀 → 배 거리
                dp = np.hypot(cell_x - px, cell_y - py)
                cell[0, p, c, 3] = dp / W
                # 셀 점유 여부 (그물 설치됨)
                cell[0, p, c, 4] = 0.0  # 간소화: 항상 비점유

        return {
            "own": own,
            "ally": ally,
            "ally_mask": ally_mask,
            "enemy": enemy,
            "enemy_mask": enemy_mask,
            "cell": cell,
            "cell_mask": cell_mask,
        }

    def _get_cell_world(self) -> np.ndarray:
        """셀 격자 좌표 반환."""
        spacing = getattr(self.cfg, "cell_spacing", 473.0)
        n_per_axis = int(self.world_size / spacing)
        coords = []
        cx, cy = self.center
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                # 모선 주변 annular 필터링
                d = np.hypot(x - cx, y - cy)
                if 500 < d < 5500:  # 대략적인 범위
                    coords.append([x, y])
        return np.array(coords) if coords else np.array([[cx, cy]])

    def publish_waypoints(self):
        """경로를 ROS2로 발행."""
        if self._bridge is not None:
            self._bridge.publish_waypoints(self.route, self.net_mask)

    def step(self):
        """한 스텝 진행."""
        self.update_from_ros2()
        self.t += 1
        self._micro_ct += 1

        # 결정 주기마다 셀 정책 재실행
        if self._micro_ct >= self._decision_period:
            self._micro_ct = 0
            if self._plan is not None:
                self._run_cell_policy()

        # 통계 업데이트
        for i in range(self.M):
            if self.e_alive[i]:
                dist = np.linalg.norm(self.e_pos[i] - self.center)
                if dist < self.mothership_radius:
                    self.e_alive[i] = False
                    self.stats["breaches"] += 1

        self.stats["survived"] = int(self.e_alive.sum())

    def reset(self, seed: int = None):
        """리셋."""
        self.t = 0
        self._micro_ct = 0
        self._plan = None
        self.stats = {k: 0 for k in self.stats}
        self._assign[:] = -1
        self.net_mask[:] = False
        self.doing_net[:] = False
        self._h = self._actor.init_hidden(self.P, self.device)

    def get_frame(self) -> dict:
        """UI 렌더링용 프레임 데이터."""
        return {
            "world_size": self.world_size,
            "cell_size": self.world_size / 200,
            "t": self.t,
            "done": False,
            "mothership": self.center,
            "mothership_radius": self.mothership_radius,
            "moback_size": self.cfg.moback_size,
            "moback_heading": self.cfg.moback_heading,
            "enemy_pos": self.e_pos,
            "enemy_hdg": self.e_hdg,
            "enemy_alive": self.e_alive,
            "enemy_size": self.cfg.enemy_size,
            "ally_pos": self.a_pos,
            "ally_hdg": self.a_hdg,
            "ally_paths": self._get_ally_paths(),
            "ally_nets": self.a_nets,
            "ally_painting": self.doing_net,
            "ally_alive": self.a_alive,
            "assign": self._assign,
            "assignI": self._assignI,
            "route": self.route,
            "net_mask": self.net_mask,
            "ship_len": self.cfg.ship_len,
            "ship_wid": self.cfg.ship_wid,
            "painted": np.zeros((200, 200), dtype=bool),
            "selected": -1,
            "manual": False,
            "running": self.running,
            "net_stage": 0,
            "stats": dict(self.stats),
            "n_alive": int(self.e_alive.sum()),
            "n_clusters": self.cfg.n_clusters,
            "cluster_gap_deg": self.cfg.cluster_gap_deg,
            "enemy_speed": self.cfg.enemy_speed,
            "show_clusters": True,
            "show_residual": False,
            "wp_adjust_max": 500.0,
            "enemy_mode": "ros2",
            "ros2_mode": True,
            "cell_mode": True,
        }

    def _get_ally_paths(self):
        """경로 데이터 생성."""
        paths = []
        for p in range(self.P):
            wps = []
            for k in range(self.Kw):
                wps.append({
                    "x": float(self.route[p, k, 0]),
                    "y": float(self.route[p, k, 1]),
                    "paint": bool(self.net_mask[p, k]),
                })
            paths.append(wps)
        return paths

    def cell_viz(self) -> dict:
        """셀 시각화 데이터."""
        cell_world = self._get_cell_world()
        return {
            "world": cell_world,
            "valid": [np.arange(len(cell_world)) for _ in range(self.P)],
            "excluded": [np.array([], dtype=int) for _ in range(self.P)],
            "selected": self._last_cells,
        }

    def set_command(self, assign_array):
        """하위호환용 직접 배정."""
        if assign_array is None:
            self._plan = None
            self._assign[:] = -1
        else:
            self._assign = np.asarray(assign_array, np.int64)
