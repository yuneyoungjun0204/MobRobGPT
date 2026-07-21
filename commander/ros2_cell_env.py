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
        ckpt: str = "boatattack_sim/models/best_mixed_far.pt",
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
            "nets_per_ship": 3,
            "transit_wp": 6,
            "cell_spacing": getattr(self._cfg, "cell_spacing", 473.0),
            "cell_cart_n": getattr(self._cfg, "cell_cart_n", 20),  # 20×20 격자
            "cell_r_min": getattr(self._cfg, "cell_r_min", 800.0),  # 환형 필터 최소 반경
            "cell_r_max": getattr(self._cfg, "cell_r_max", 4500.0),  # 환형 필터 최대 반경
            "cell_nets": getattr(self._cfg, "cell_nets", 3),  # 셀당 최소 유효 개수
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
        self.a_nets = np.full(self.P, 3)  # nets_per_ship = 3 (시뮬레이션과 동일)

        self.e_pos = np.zeros((self.M, 2))
        self.e_hdg = np.zeros(self.M)
        self.e_alive = np.ones(self.M, dtype=bool)

        # 경로/그물
        self.route = np.zeros((self.P, self.Kw, 2))
        self.net_mask = np.zeros((self.P, self.Kw), dtype=bool)
        self.doing_net = np.zeros(self.P, dtype=bool)

        # 그물 설치 추적 (시뮬레이션과 동일)
        # 200×200 격자 (world_size / 200 = 63m 해상도)
        self._net_grid_size = 200
        self.net_installed = np.zeros((self._net_grid_size, self._net_grid_size), dtype=bool)
        self._cell_half_r = getattr(self._cfg, "cell_spacing", 473.0) / 2.0

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

        # 셀 좌표 → 경로 변환 + 그물 설치 위치 마킹
        cell_world = self._get_cell_world()
        for p in range(self.P):
            if self._assign[p] < 0:
                # 예비: 제자리 정지
                self.route[p, :, :] = self.a_pos[p]
                self.net_mask[p, :] = False
            else:
                # 선택된 셀들로 경로 구성
                route_pts = []
                for k in range(min(self.Kw, K)):
                    cell_idx = int(self._last_cells[p, k])
                    cell_idx = min(max(cell_idx, 0), len(cell_world) - 1)
                    self.route[p, k] = cell_world[cell_idx]
                    self.net_mask[p, k] = True
                    route_pts.append(cell_world[cell_idx])
                # 나머지 WP는 마지막 셀로
                for k in range(K, self.Kw):
                    self.route[p, k] = self.route[p, K - 1]
                    self.net_mask[p, k] = False

                # 선택된 셀들을 그물 설치 격자에 마킹 (중복 배치 방지)
                if route_pts:
                    self._mark_cells_netted(np.array(route_pts))

        # ROS2로 경로 발행
        self.publish_waypoints()

    def _cell_valid_mask(self) -> np.ndarray:
        """셀 유효성 마스크 반환 (시뮬레이션과 동일 방식).

        True = 무효(마스킹됨), False = 유효.
        이미 그물이 설치된 위치의 셀은 무효화하여 중복 배치 방지.

        Returns:
            [N, P, C] bool 배열
        """
        cell_world = self._get_cell_world()
        C = len(cell_world)

        # 베이스 마스크: 모든 셀 유효 (False = valid)
        base = np.zeros((1, self.P, C), dtype=bool)

        # 그물 설치 추적 확인
        if not self.net_installed.any():
            return base

        # 그물 설치된 격자셀의 중심 좌표 계산
        G = self._net_grid_size
        cell_size = self.world_size / G
        ii, jj = np.where(self.net_installed)
        if len(ii) == 0:
            return base

        # 설치된 그물의 격자셀 중심 좌표
        netxy = np.stack([(ii + 0.5) * cell_size, (jj + 0.5) * cell_size], axis=1)  # [M, 2]
        cw = cell_world  # [C, 2]

        # 그물 바로 위 후보셀만 배제 (R = 격자 반칸 또는 250m 중 큰 값)
        R = max(float(self._cell_half_r), 250.0)

        # 각 후보셀 → 가장 가까운 그물셀 거리 < R 이면 무효
        d2 = ((cw[:, None, 0] - netxy[None, :, 0]) ** 2
              + (cw[:, None, 1] - netxy[None, :, 1]) ** 2)  # [C, M]
        occ = d2.min(1) < (R * R)  # [C]

        # 그 후보셀은 모든 배에게 무효
        mask = base | occ[None, None, :]

        # 유효셀 부족한 배 → 원복 (굶음 방지)
        cell_nets = int(getattr(self.cfg, "cell_nets", 3))
        short = (~mask).sum(2) < cell_nets
        if short.any():
            mask = np.where(short[..., None], base, mask)

        return mask

    def _mark_cells_netted(self, route_pts: np.ndarray):
        """경로 좌표들을 그물 설치 격자에 마킹.

        Args:
            route_pts: [K, 2] 좌표 배열
        """
        G = self._net_grid_size
        cell_size = self.world_size / G
        for pt in route_pts:
            x, y = pt
            i = int(x / cell_size)
            j = int(y / cell_size)
            if 0 <= i < G and 0 <= j < G:
                self.net_installed[i, j] = True

    def _build_simple_obs(self) -> dict:
        """셀 정책용 관측 생성 (simulation 모드와 동일한 형식).

        정규화: 학습 환경과 동일하게 action_grid_half (6000) 사용.
        - 위치: 모선 중심 기준 (pos - center) / half → [-1, 1]
        - 상대 위치/거리: / half

        cell_bridge.py의 build_cell_obs()와 동일한 형식:
          own: [N, P, 9] - pos2(모선중심기준)·head2·nets1·doing1·to_intercept2(상대)·flag1
          ally: [N, P, A, 6] - rel_pos2·nets1·doing1·dist1·alive1
          ally_mask: [N, P, A] bool
          enemy: [N, P, Kc, 6] - 클러스터 centroid 기반
          enemy_mask: [N, P, Kc] bool
          cell: [N, P, C, 5] - 배 기준 상대 위치
          cell_mask: [N, P, C] bool
        """
        from boatattack_sim.env import clustering

        N = 1  # 단일 환경
        P = self.P
        cfg = self.cfg
        # ★ 정규화 분모: action_grid_half (학습 설정과 동일, 기본 6000)
        half = float(getattr(cfg, 'action_grid_half', 6000.0))
        c = np.array(self.center)  # 모선 중심 (정규화 원점)
        Kc = cfg.n_clusters  # 클러스터 수 = 3

        cell_world = self._get_cell_world()
        C = len(cell_world)
        cw = np.array(cell_world)  # [C, 2]

        # ── own: [N, P, 9] ──
        # pos2(모선 중심 기준), head2, nets1, doing1, to_intercept2(상대), assign_flag1
        fwd_sin = np.sin(np.radians(self.a_hdg))  # [P]
        fwd_cos = np.cos(np.radians(self.a_hdg))  # [P]

        assigned_f = (self._assign >= 0).astype(np.float64)  # [P]
        to_I = self._assignI - self.a_pos  # [P, 2] 배→요격점 상대 벡터

        own = np.zeros((N, P, 9), dtype=np.float64)
        # ★ 위치: 모선 중심 기준 상대좌표 / half
        own[0, :, 0] = (self.a_pos[:, 0] - c[0]) / half
        own[0, :, 1] = (self.a_pos[:, 1] - c[1]) / half
        own[0, :, 2] = fwd_sin
        own[0, :, 3] = fwd_cos
        own[0, :, 4] = self.a_nets / cfg.nets_per_ship
        own[0, :, 5] = self.doing_net.astype(np.float64)
        own[0, :, 6] = to_I[:, 0] / half * assigned_f
        own[0, :, 7] = to_I[:, 1] / half * assigned_f
        own[0, :, 8] = assigned_f

        # ── ally: [N, P, A, 6] ──
        A = max(P - 1, 1)
        ally = np.zeros((N, P, A, 6), dtype=np.float64)
        ally_mask = np.ones((N, P, A), dtype=bool)

        for p in range(P):
            others = [q for q in range(P) if q != p]
            for slot, q in enumerate(others[:A]):
                rel = self.a_pos[q] - self.a_pos[p]  # 상대 위치
                d = np.hypot(rel[0], rel[1])
                ally[0, p, slot, 0] = rel[0] / half
                ally[0, p, slot, 1] = rel[1] / half
                ally[0, p, slot, 2] = self.a_nets[q] / cfg.nets_per_ship
                ally[0, p, slot, 3] = float(self.doing_net[q])
                ally[0, p, slot, 4] = d / half
                ally[0, p, slot, 5] = float(self.a_alive[q])
                ally_mask[0, p, slot] = not self.a_alive[q]

        # ── enemy: [N, P, Kc, 6] - 클러스터 기반 ──
        # clustering.cluster_by_gaps_vec 사용 (simulation 모드와 동일)
        e_pos_batch = self.e_pos[None, :, :]  # [1, M, 2]
        e_alive_batch = self.e_alive[None, :]  # [1, M]
        e_hdg_batch = self.e_hdg[None, :]  # [1, M]

        cl = clustering.cluster_by_gaps_vec(
            e_pos_batch, e_alive_batch, e_hdg_batch,
            c, cfg.enemy_speed, Kc, cfg.cluster_gap_deg
        )
        centroid = cl["centroid"]  # [N, Kc, 2]

        enemy = np.zeros((N, P, Kc, 6), dtype=np.float64)
        enemy_mask = np.ones((N, P, Kc), dtype=bool)

        for p in range(P):
            rel = centroid[0] - self.a_pos[p]  # [Kc, 2]
            d = np.hypot(rel[:, 0], rel[:, 1])
            enemy[0, p, :, 0] = rel[:, 0] / half
            enemy[0, p, :, 1] = rel[:, 1] / half
            enemy[0, p, :, 2] = d / half
            enemy[0, p, :, 3] = cl["count"][0] / max(self.M, 1)
            enemy[0, p, :, 4] = cl["spread_deg"][0] / 180.0
            enemy[0, p, :, 5] = cl["active"][0].astype(np.float64)
            enemy_mask[0, p, :] = ~cl["active"][0]

        # ── cell: [N, P, C, 5] ──
        cell = np.zeros((N, P, C, 5), dtype=np.float64)
        cell_mask = self._cell_valid_mask()  # [N, P, C] True=invalid

        # 셀→모선 거리 (모든 배에 동일)
        cell_to_m = np.hypot(cw[:, 0] - c[0], cw[:, 1] - c[1])  # [C]

        for p in range(P):
            rel = cw - self.a_pos[p]  # [C, 2] 배 기준 상대 위치
            d = np.hypot(rel[:, 0], rel[:, 1])
            cell[0, p, :, 0] = rel[:, 0] / half
            cell[0, p, :, 1] = rel[:, 1] / half
            cell[0, p, :, 2] = d / half
            cell[0, p, :, 3] = cell_to_m / half
            cell[0, p, :, 4] = (~cell_mask[0, p, :]).astype(np.float64)

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
        """셀 격자 좌표 반환 (시뮬레이션과 동일한 방식).

        cell_bridge.py의 cell_world와 동일:
        - 고정 20×20 카르테시안 격자 (모선 중심 기준)
        - 환형 필터: r_min=800, r_max=4500
        """
        if hasattr(self, "_cell_world_cache"):
            return self._cell_world_cache

        n = getattr(self.cfg, "cell_cart_n", 20)  # 20×20 격자
        spacing = getattr(self.cfg, "cell_spacing", 473.0)
        half = self.world_size / 2
        cx, cy = self.center

        # 격자 중심(모선 위치) 기준으로 n×n 셀 배치
        x = np.linspace(half - (n-1)/2 * spacing, half + (n-1)/2 * spacing, n)
        y = np.linspace(half - (n-1)/2 * spacing, half + (n-1)/2 * spacing, n)
        xx, yy = np.meshgrid(x, y)
        all_cells = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # 환형 필터 (r_min ~ r_max 범위 내 셀만) - 시뮬레이션과 동일
        r_min = getattr(self.cfg, "cell_r_min", 800.0)
        r_max = getattr(self.cfg, "cell_r_max", 4500.0)
        dist = np.hypot(all_cells[:, 0] - cx, all_cells[:, 1] - cy)
        valid = (dist >= r_min) & (dist <= r_max)

        self._cell_world_cache = all_cells[valid]
        return self._cell_world_cache

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
        # 그물 설치 추적 초기화
        self.net_installed[:] = False
        # 셀 격자 캐시 초기화 (모선 위치 변경 가능성)
        if hasattr(self, "_cell_world_cache"):
            del self._cell_world_cache

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
