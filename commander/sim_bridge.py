"""시뮬레이터 ↔ 지휘관 브릿지 (배정 방식).

- LLM 은 '어느 클러스터에 몇 척'만 결정(CommanderPlan.deployments).
- CommandedSimulator 는 AUTO 모드 유지 → 시뮬의 heuristic_plan + _build_cluster_path 가
  배정(self.assign)으로부터 실제 그물벽 경로를 기하로 생성(요격 링 위 수직벽). 모델이 약해도 확실히 막힘.
- LLM 배정은 _compute_assignment 에서 주입. 명령 전엔 전원 예비(정지).
- 40m 비율 축소(scaled_config) · 위성 앵커(geo) · 아군/모선 충돌 회피(_separate) 유지.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from boatattack_sim.env.simulator import Simulator
from boatattack_sim.env.config import DEFAULT_CONFIG, DEFAULT_REWARD
from boatattack_sim.env import clustering

from .schema import (
    BattlefieldState, Mothership, EnemyCluster, AllyShip, Constraints, Point,
)

DEFAULT_WORLD_SIZE = 12600.0   # One-Way 원본 스케일. (40.0 으로 주면 비율 축소판)

# 위성 배경 앵커: 삼성중공업 대덕연구센터(대전 유성구 문지로 217) — 정확한 Square 33 좌표로 교체 권장.
GEO_LAT_DEFAULT = 36.4107
GEO_LON_DEFAULT = 127.4017


def scaled_config(world_size: float = DEFAULT_WORLD_SIZE, enemy_speed_mult: float = 1.5,
                  geo_lat: float = GEO_LAT_DEFAULT, geo_lon: float = GEO_LON_DEFAULT):
    """12600m 기본설정을 target world_size 로 '비율 유지' 축소 → 동역학 보존. 적=아군×mult."""
    base = DEFAULT_CONFIG
    s = world_size / base.world_size
    return replace(
        base,
        world_size=world_size,
        mothership_radius=base.mothership_radius * s,
        enemy_spawn_margin=base.enemy_spawn_margin * s,
        enemy_spawn_radius=base.enemy_spawn_radius * s,
        enemy_wave_gap=base.enemy_wave_gap * s,
        arrive_radius=base.arrive_radius * s,
        ally_row_gap=base.ally_row_gap * s,
        ally_side_spacing=base.ally_side_spacing * s,
        net_max_len=base.net_max_len * s,
        ally_collision_radius=base.ally_collision_radius * s,
        ally_speed=base.ally_speed * s,
        ship_len=base.ship_len * s,
        ship_wid=base.ship_wid * s,
        enemy_size=base.enemy_size * s,
        moback_size=base.moback_size * s,
        enemy_speed_mult=enemy_speed_mult,
        geo_lat=geo_lat, geo_lon=geo_lon,
    )


class CommandedSimulator(Simulator):
    """LLM 배정을 존중하는 시뮬레이터 (AUTO 모드 — 경로는 시뮬이 기하로 생성)."""

    def __init__(self, enemy_mode: str = "random", cfg=None,
                 world_size: float = DEFAULT_WORLD_SIZE, enemy_speed_mult: float = 1.5,
                 geo_lat: float = GEO_LAT_DEFAULT, geo_lon: float = GEO_LON_DEFAULT):
        # super().__init__ 가 reset→_compute_assignment 를 호출하므로 속성을 미리 만든다.
        self._commanded_assign = None
        self._plan = None                 # LLM 계획(매 스텝 현재 상태로 재매핑)
        self._plan_command = None
        if cfg is None:
            cfg = scaled_config(world_size, enemy_speed_mult, geo_lat, geo_lon)
        super().__init__(cfg=cfg, enemy_mode=enemy_mode)
        # manual=False(AUTO) 유지 → step() 이 heuristic_plan() 으로 assign 기반 경로 생성

    def set_plan(self, plan, command: str | None = None) -> None:
        """LLM 계획 저장 → **매 스텝** 현재 전장으로 재매핑(죽은 배·위치변화 즉시 적응).

        척수(deployments)·정지(hold_ships)는 다음 LLM 재계획까지 유지되며, '어느 배가 어느
        클러스터'는 매 스텝 sticky 규칙으로 다시 계산된다 → 배가 격침되면 남은 배로 자동 재배분.
        """
        self._plan = plan
        self._plan_command = command
        self._commanded_assign = None

    def set_command(self, assign_array) -> None:
        """(하위호환) 고정 배정 직접 주입 — 매 스텝 재매핑 안 함. None 이면 전원 예비(정지)."""
        self._plan = None
        self._commanded_assign = (None if assign_array is None
                                  else np.asarray(assign_array, np.int64))

    def _keepout(self) -> float:
        return self.cfg.mothership_radius * 1.3

    def _inject_assign(self, arr) -> None:
        """배정 배열[P] → self.assign/assignI. 격침된 배는 항상 -1(예비)."""
        c = np.array(self.cfg.center, np.float64)
        t = DEFAULT_REWARD.assign_intercept_t
        ncl = len(self._cl_cent)
        for i in range(min(self.cfg.n_allies, len(arr))):
            k = int(arr[i])
            self.assign[i] = k
            if 0 <= k < ncl:
                cent = self._cl_cent[k]
                self.assignI[i] = cent + t * (c - cent)
        self.assign[~self.a_alive] = -1

    def _compute_assignment(self):
        # 1) 원본: 클러스터링(_cl_cent) + 기본 그리디 배정
        prev = self.assign.copy() if getattr(self, "assign", None) is not None else None
        super()._compute_assignment()
        # 2) LLM 계획이 있으면 매 스텝 현재 상태로 재매핑(연속성 유지 → 스래싱 방지).
        if self._plan is not None:
            if prev is not None:
                self.assign[:] = prev            # 연속성 힌트로 직전 배정 사용
            state = build_battlefield(self, self._plan_command)
            self._inject_assign(plan_to_assign(self._plan, state))
            return
        # 3) 고정 배정(하위호환). 없으면 전원 예비(정지).
        if self._commanded_assign is None:
            self.assign[:] = -1
            return
        self._inject_assign(self._commanded_assign)

    def _resolve_ally_collisions(self):
        """아군끼리 충돌(충돌반경 이내)하면 양쪽 모두 격침(비활성화). 그물 전개도 중단."""
        P = self.cfg.n_allies
        r = self.cfg.ally_collision_radius
        for a in range(P):
            if not self.a_alive[a]:
                continue
            for b in range(a + 1, P):
                if not self.a_alive[b]:
                    continue
                dd = float(np.hypot(*(self.a_pos[a] - self.a_pos[b])))
                if dd < r:
                    self.a_alive[a] = self.a_alive[b] = False
                    self.a_painting[a] = self.a_painting[b] = False
                    self.stats["ally_collisions"] += 1

    def step(self):
        super().step()
        self._separate()          # 모선 keep-out (아군 상호 밀어내기는 제거 — 충돌=격침)
        return self.get_frame()

    def _separate(self) -> None:
        """모선 keep-out(모선 위로 못 올라감) + 월드 경계 클립만. 아군-아군 충돌은
        밀어내지 않고 _resolve_ally_collisions 에서 격침 처리(부딪히면 비활성화)."""
        cfg = self.cfg
        P = cfg.n_allies
        c = np.array(cfg.center, np.float64)
        keep = self._keepout()
        W = cfg.world_size
        for i in range(P):
            if not self.a_alive[i]:
                continue
            v = self.a_pos[i] - c
            d = float(np.hypot(v[0], v[1]))
            if d < keep:
                if d < 1e-6:
                    ang = 2.0 * np.pi * i / float(P)
                    self.a_pos[i] = c + np.array([np.cos(ang), np.sin(ang)]) * keep
                else:
                    self.a_pos[i] = c + v / d * keep
        np.clip(self.a_pos, 0.0, W, out=self.a_pos)


def build_battlefield(sim: Simulator, command: str | None = None) -> BattlefieldState:
    """시뮬 현재 상태 → BattlefieldState. 클러스터 id = 시뮬 클러스터 인덱스(1:1)."""
    cfg = sim.cfg
    c = np.array(cfg.center, np.float64)

    cl = clustering.cluster_by_gaps_vec(
        sim.e_pos[None], sim.e_alive[None], sim.e_hdg[None], c,
        cfg.enemy_speed, cfg.n_clusters, cfg.cluster_gap_deg,
    )
    cent = cl["centroid"][0]
    cnt = cl["count"][0]
    active = cl["active"][0]
    spread = cl["spread_deg"][0]

    # 설치된 그물 셀의 (방위, 반경) 미리 계산 → 클러스터별 net_covered 판정용
    ni, nj = np.where(sim.net_installed) if hasattr(sim, "net_installed") else (np.array([]), np.array([]))
    if len(ni):
        cellsz = sim.grid.cell
        nx = (ni + 0.5) * cellsz; ny = (nj + 0.5) * cellsz
        net_brg = np.degrees(np.arctan2(nx - c[0], ny - c[1])) % 360.0
        net_dist = np.hypot(nx - c[0], ny - c[1])
    else:
        net_brg = net_dist = None
    NET_TOL_DEG = 6.0            # 클러스터 접근 방위 ±허용각 안에 그물이 있으면 커버로 간주

    clusters = []
    for k in range(cfg.n_clusters):
        if not bool(active[k]) or int(cnt[k]) == 0:
            continue
        bearing = float(np.degrees(np.arctan2(cent[k, 0] - c[0], cent[k, 1] - c[1])) % 360.0)
        cdist = float(np.hypot(cent[k, 0] - c[0], cent[k, 1] - c[1]))
        # 접근 방위(±TOL)에 & 모선~클러스터 사이 반경에 설치 그물이 있으면 포획 예상.
        net_covered = False
        if net_brg is not None:
            dbrg = np.abs(((net_brg - bearing + 180.0) % 360.0) - 180.0)
            net_covered = bool(np.any((dbrg <= NET_TOL_DEG) & (net_dist < cdist) & (net_dist > 1.0)))
        clusters.append(EnemyCluster(
            id=k,
            center=Point(x=float(cent[k, 0]), y=float(cent[k, 1])),
            bearing=bearing, spread=float(spread[k]),
            count=int(cnt[k]), approach_speed=float(cfg.enemy_speed),
            net_covered=net_covered,
        ))

    allies = [
        AllyShip(id=i,
                 pos=Point(x=float(sim.a_pos[i, 0]), y=float(sim.a_pos[i, 1])),
                 heading=float(sim.a_hdg[i]),
                 nets_remaining=int(sim.a_nets[i]),
                 alive=bool(sim.a_alive[i]),
                 assigned_cluster=int(sim.assign[i]) if int(sim.assign[i]) >= 0 else None,
                 # 현재 자동조종 경로(WP) 와 전개 상태 → LLM 경로 중복/충돌 판단용 (죽은 배는 빈 경로)
                 route=([Point(x=float(w["x"]), y=float(w["y"])) for w in sim.a_paths[i]]
                        if bool(sim.a_alive[i]) else []),
                 deploying=bool(sim.a_painting[i]))
        for i in range(cfg.n_allies)
    ]

    if sim.e_alive.any():
        d = np.hypot(sim.e_pos[sim.e_alive, 0] - c[0], sim.e_pos[sim.e_alive, 1] - c[1]).min()
        threat = float(np.clip(1.0 - d / (cfg.world_size / 2.0), 0.0, 1.0))
    else:
        threat = 0.0

    return BattlefieldState(
        mothership=Mothership(pos=Point(x=float(c[0]), y=float(c[1])),
                              radius=float(cfg.mothership_radius), threat_level=threat),
        enemy_clusters=clusters,
        allies=allies,
        constraints=Constraints(net_max_len=float(cfg.net_max_len),
                                ally_speed=float(getattr(cfg, "ally_speed", 6.0)),
                                enemy_speed=float(cfg.enemy_speed),
                                world_size=float(cfg.world_size),
                                max_intercept_radius=float(getattr(sim, "_R_FEAS",
                                                                    cfg.world_size / 3.0))),
        command=command,
    )


def plan_to_assign(plan, state: BattlefieldState) -> np.ndarray:
    """CommanderPlan(클러스터별 투입 척수) → sim.assign 배열[P] (아군별 담당 클러스터 idx, -1=예비).

    LLM 은 '몇 척'만 정하고, '어느 배'는 여기서 기계적으로 결정한다. 재계획 스래싱(배가
    매번 담당 클러스터를 갈아타 경로 재생성 → WP1 도 못 감)을 막기 위해 **연속성(sticky)**
    우선: 이미 그 클러스터를 담당 중이고 여전히 필요한 배는 그대로 유지하고, 남는 슬롯만
    가장 가까운 미배정 배로 채운다. 아군 부족 시 위협(척수) 큰 클러스터부터 커버.

    HOLD: plan.hold_ships 의 아군은 배정과 무관하게 assign=-1(제자리 정지)로 덮어쓴다.
    """
    P = len(state.allies)
    assign = np.full(P, -1, np.int64)
    centers = {c.id: (c.center.x, c.center.y) for c in state.enemy_clusters}
    threat = {c.id: c.count for c in state.enemy_clusters}
    ally_pos = {a.id: (a.pos.x, a.pos.y) for a in state.allies}
    current = {a.id: a.assigned_cluster for a in state.allies}   # 직전 배정(연속성 힌트)

    # 클러스터별 필요 척수 집계(유효 클러스터만)
    need: dict[int, int] = {}
    for dep in plan.deployments:
        if dep.cluster_id in centers and dep.n_ships >= 1:
            need[dep.cluster_id] = need.get(dep.cluster_id, 0) + dep.n_ships

    available = {a.id for a in state.allies if a.alive}          # 격침된 배는 배정 제외

    # 1) 연속성: 직전과 동일 클러스터를 계속 담당(경로 유지 → 스래싱 방지)
    for i in list(available):
        k = current.get(i)
        if k is not None and need.get(k, 0) > 0:
            assign[i] = k
            need[k] -= 1
            available.discard(i)

    # 2) 남은 슬롯: 위협 큰 클러스터부터 가장 가까운 미배정 배로 채움
    for cid in sorted(need, key=lambda k: -threat.get(k, 0)):
        cx, cy = centers[cid]
        while need[cid] > 0 and available:
            nearest = min(available,
                          key=lambda i: (ally_pos[i][0] - cx) ** 2 + (ally_pos[i][1] - cy) ** 2)
            assign[nearest] = cid
            available.discard(nearest)
            need[cid] -= 1

    # 3) HOLD: 지정 아군은 제자리 정지(assign=-1). 전개중이면 그물은 마저 설치됨.
    for i in getattr(plan, "hold_ships", None) or []:
        if 0 <= int(i) < P:
            assign[int(i)] = -1
    return assign


def apply_plan(sim: CommandedSimulator, plan, state: BattlefieldState) -> None:
    """CommanderPlan → 시뮬에 계획 주입(매 스텝 재매핑)."""
    sim.set_plan(plan, state.command)


__all__ = ["CommandedSimulator", "build_battlefield", "plan_to_assign", "apply_plan",
           "scaled_config"]
