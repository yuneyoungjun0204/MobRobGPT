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
        if cfg is None:
            cfg = scaled_config(world_size, enemy_speed_mult, geo_lat, geo_lon)
        super().__init__(cfg=cfg, enemy_mode=enemy_mode)
        # manual=False(AUTO) 유지 → step() 이 heuristic_plan() 으로 assign 기반 경로 생성

    def set_command(self, assign_array) -> None:
        """지휘관 배정 주입(아군별 담당 클러스터 idx, -1=예비). None 이면 전원 예비(정지)."""
        self._commanded_assign = (None if assign_array is None
                                  else np.asarray(assign_array, np.int64))

    def _keepout(self) -> float:
        return self.cfg.mothership_radius * 1.3

    def _compute_assignment(self):
        # 1) 원본: 클러스터링(_cl_cent) + 기본 그리디 배정
        super()._compute_assignment()
        # 2) 명령 없으면 전원 예비(정지). 있으면 그것으로 덮어씀.
        if self._commanded_assign is None:
            self.assign[:] = -1
            return
        c = np.array(self.cfg.center, np.float64)
        t = DEFAULT_REWARD.assign_intercept_t
        ncl = len(self._cl_cent)
        for i in range(min(self.cfg.n_allies, len(self._commanded_assign))):
            k = int(self._commanded_assign[i])
            self.assign[i] = k
            if 0 <= k < ncl:
                cent = self._cl_cent[k]
                self.assignI[i] = cent + t * (c - cent)

    def step(self):
        super().step()
        self._separate()          # 아군 상호 분리 + 모선 keep-out (위치 보정)
        return self.get_frame()

    def _separate(self) -> None:
        cfg = self.cfg
        P = cfg.n_allies
        c = np.array(cfg.center, np.float64)
        keep = self._keepout()
        minsep = cfg.ally_collision_radius
        W = cfg.world_size
        for _ in range(12):
            for a in range(P):
                if not self.a_alive[a]:
                    continue
                for b in range(a + 1, P):
                    if not self.a_alive[b]:
                        continue
                    diff = self.a_pos[a] - self.a_pos[b]
                    d = float(np.hypot(diff[0], diff[1]))
                    if d >= minsep:
                        continue
                    if d < 1e-6:
                        ang = 2.0 * np.pi * (a * P + b) / float(P * P)
                        u = np.array([np.cos(ang), np.sin(ang)]); push = minsep / 2.0
                    else:
                        u = diff / d; push = (minsep - d) / 2.0
                    self.a_pos[a] += u * push
                    self.a_pos[b] -= u * push
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

    clusters = []
    for k in range(cfg.n_clusters):
        if not bool(active[k]) or int(cnt[k]) == 0:
            continue
        bearing = float(np.degrees(np.arctan2(cent[k, 0] - c[0], cent[k, 1] - c[1])) % 360.0)
        clusters.append(EnemyCluster(
            id=k,
            center=Point(x=float(cent[k, 0]), y=float(cent[k, 1])),
            bearing=bearing, spread=float(spread[k]),
            count=int(cnt[k]), approach_speed=float(cfg.enemy_speed),
        ))

    allies = [
        AllyShip(id=i,
                 pos=Point(x=float(sim.a_pos[i, 0]), y=float(sim.a_pos[i, 1])),
                 heading=float(sim.a_hdg[i]),
                 nets_remaining=int(sim.a_nets[i]),
                 assigned_cluster=int(sim.assign[i]) if int(sim.assign[i]) >= 0 else None)
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

    LLM 은 '몇 척'만 정하고, '어느 배'는 여기서 기계적으로: 투입 많은(위협 큰) 클러스터부터
    가장 가까운 미배정 아군을 n_ships 만큼 확보. 남는 배는 예비(-1). 합계 초과 시 자동 클램프.
    """
    P = len(state.allies)
    assign = np.full(P, -1, np.int64)
    centers = {c.id: (c.center.x, c.center.y) for c in state.enemy_clusters}
    ally_pos = {a.id: (a.pos.x, a.pos.y) for a in state.allies}
    available = [a.id for a in state.allies]

    for dep in sorted(plan.deployments, key=lambda d: -d.n_ships):
        if dep.cluster_id not in centers or dep.n_ships < 1 or not available:
            continue
        cx, cy = centers[dep.cluster_id]
        nearest = sorted(available, key=lambda i: (ally_pos[i][0] - cx) ** 2 + (ally_pos[i][1] - cy) ** 2)
        for i in nearest[:dep.n_ships]:
            if 0 <= i < P:
                assign[i] = dep.cluster_id
            available.remove(i)
    return assign


def apply_plan(sim: CommandedSimulator, plan, state: BattlefieldState) -> None:
    """CommanderPlan → 시뮬에 배정 주입."""
    sim.set_command(plan_to_assign(plan, state))


__all__ = ["CommandedSimulator", "build_battlefield", "plan_to_assign", "apply_plan",
           "scaled_config"]
