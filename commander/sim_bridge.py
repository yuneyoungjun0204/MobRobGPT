"""시뮬레이터 ↔ 지휘관 브릿지 (LLM 이 WP 좌표를 직접 생성하는 방식).

- CommandedSimulator: Simulator 서브클래스. **manual 모드**로 돌려 시뮬 자동 경로생성
  (_build_cluster_path/heuristic_plan)을 끄고, LLM 이 준 6-WP 경로를 a_paths 에 직접 주입.
  명령 전에는 전원 경로 없음 → 정지.
- build_battlefield: 시뮬 현재 상태 → 지휘관 입력 BattlefieldState (클러스터 id 1:1).
- apply_plan: CommanderPlan.routes → sim.set_routes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from boatattack_sim.env.simulator import Simulator
from boatattack_sim.env.config import DEFAULT_CONFIG
from boatattack_sim.env import clustering

from .schema import (
    BattlefieldState, Mothership, EnemyCluster, AllyShip, Constraints, Point,
)

DEFAULT_WORLD_SIZE = 40.0   # 맵 한 변 [m] (기존 12600 → 40 축소). 모든 길이 상수는 비율 유지 축소.

# 위성 배경 앵커: 삼성중공업 대덕연구센터 (대전 유성구 문지로 217) 'Square 33'
# ※ 아래는 연구센터 대략 좌표. 정확한 Square 33 중심 위경도로 교체 권장.
GEO_LAT_DEFAULT = 36.4107
GEO_LON_DEFAULT = 127.4017


def scaled_config(world_size: float = DEFAULT_WORLD_SIZE, enemy_speed_mult: float = 2.0,
                  geo_lat: float = GEO_LAT_DEFAULT, geo_lon: float = GEO_LON_DEFAULT):
    """기존 12600m 설정을 target world_size 로 '비율 유지' 축소 → 동역학 동일 보존.
    길이(반경·그물·간격·속도)는 s=world/12600 배, 각도·개수·비율은 그대로."""
    base = DEFAULT_CONFIG
    s = world_size / base.world_size
    return replace(
        base,
        world_size=world_size,
        mothership_radius=base.mothership_radius * s,
        enemy_spawn_margin=base.enemy_spawn_margin * s,
        enemy_spawn_radius=base.enemy_spawn_radius * s,
        enemy_wave_gap=base.enemy_wave_gap * s,       # wave 모드 단 간격

        arrive_radius=base.arrive_radius * s,
        ally_row_gap=base.ally_row_gap * s,           # 모선 뒤(아래) 거리
        ally_side_spacing=base.ally_side_spacing * s, # ★ 아군 옆 간격(누락 → 맵 밖으로 튕김)
        net_max_len=base.net_max_len * s,
        ally_collision_radius=base.ally_collision_radius * s,
        ally_speed=base.ally_speed * s,          # 속도도 축소(같은 step 수로 횡단 → 동역학 보존)
        geo_lat=geo_lat, geo_lon=geo_lon,         # 위성 배경 앵커(삼성 대덕)
        # 렌더 표시 크기(선박/모선)도 축소 — 안 하면 맵보다 커서 화면이 안 보임
        ship_len=base.ship_len * s,
        ship_wid=base.ship_wid * s,      # ★ 아군 선박 폭 (누락돼 있어 크게 보였음)
        enemy_size=base.enemy_size * s,
        moback_size=base.moback_size * s,
        enemy_speed_mult=enemy_speed_mult,
    )


class CommandedSimulator(Simulator):
    """LLM 이 준 WP 경로를 그대로 따르는 시뮬레이터 (manual 모드)."""

    def __init__(self, enemy_mode: str = "random", cfg=None,
                 world_size: float = DEFAULT_WORLD_SIZE, enemy_speed_mult: float = 2.0,
                 geo_lat: float = GEO_LAT_DEFAULT, geo_lon: float = GEO_LON_DEFAULT):
        if cfg is None:
            cfg = scaled_config(world_size, enemy_speed_mult, geo_lat, geo_lon)
        super().__init__(cfg=cfg, enemy_mode=enemy_mode)
        self.manual = True          # 자동 휴리스틱 경로생성 끔 → LLM WP 만 따름
        self.clear_routes()         # 명령 전엔 전원 정지

    def clear_routes(self) -> None:
        for i in range(self.cfg.n_allies):
            self.a_paths[i] = []

    def _keepout(self) -> float:
        return self.cfg.mothership_radius * 1.3   # 반경 비례 keep-out (맵 스케일 무관)

    def _clamp_radius(self, x: float, y: float) -> tuple[float, float]:
        """WP 반경을 [모선 keep-out, 도달가능 반경 R_FEAS] 로 강제.
        모선 안이면 밖으로, 도달 불가한 먼 곳이면 안쪽 요격 링으로 당김."""
        cx, cy = self.cfg.center
        vx, vy = x - cx, y - cy
        d = (vx * vx + vy * vy) ** 0.5
        keep = self._keepout()
        rmax = float(getattr(self, "_R_FEAS", self.cfg.world_size / 3.0))
        rmax = max(rmax, keep + 1.0)
        if d < 1e-6:
            return cx, cy - keep
        r = min(max(d, keep), rmax)
        return cx + vx / d * r, cy + vy / d * r

    def set_routes(self, routes) -> None:
        """LLM ShipRoute 리스트 → a_paths 직접 주입. 미포함 아군은 예비(정지).

        각 WP dict = {x, y, paint, started, active} (시뮬 _step_allies 계약).
        좌표는 [0, world_size] 클램프 + 모선 keep-out 밖으로 밀어냄, 최대 6개.
        """
        W = float(self.cfg.world_size)
        cx, cy = self.cfg.center
        rmax = float(getattr(self, "_R_FEAS", self.cfg.world_size / 3.0))
        net_min_r = 0.4 * rmax     # 이 반경보다 안쪽(모선 근처)에선 그물 금지 → 낭비/시작부터 깔림 방지
        self.clear_routes()
        for r in routes:
            i = int(r.ally_id)
            if not (0 <= i < self.cfg.n_allies):
                continue
            pts = []
            for k, w in enumerate(r.waypoints[:6]):
                x = float(min(max(w.x, 0.0), W))
                y = float(min(max(w.y, 0.0), W))
                x, y = self._clamp_radius(x, y)        # 모선 keep-out + 도달가능 반경 클램프
                rr = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                # 첫 WP(k==0)는 무조건 transit(출발→링 이동), 그물은 바깥 요격 링에서만
                paint = bool(w.deploy_net) and k >= 1 and rr >= net_min_r
                pts.append({"x": x, "y": y, "paint": paint,
                            "started": False, "active": False})
            self.a_paths[i] = pts

    def step(self):
        frame = super().step()
        self._separate()          # 아군 상호 분리 + 모선 keep-out (실제 위치 보정)
        return self.get_frame()

    def _separate(self) -> None:
        """매 스텝 위치 보정: 아군끼리 겹치지 않게 밀어내고, 모선 안으로 못 들어가게."""
        cfg = self.cfg
        P = cfg.n_allies
        c = np.array(cfg.center, np.float64)
        keep = self._keepout()
        minsep = cfg.ally_collision_radius
        W = cfg.world_size
        for _ in range(12):       # 반복 완화(수렴). 실주행은 몇 회면 충분, 극단케이스 대비 여유.
            # 아군 상호 분리
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
                    if d < 1e-6:      # 완전히 겹침 → 쌍별 결정적 방향으로 분리
                        ang = 2.0 * np.pi * (a * P + b) / float(P * P)
                        u = np.array([np.cos(ang), np.sin(ang)])
                        push = minsep / 2.0
                    else:
                        u = diff / d
                        push = (minsep - d) / 2.0
                    self.a_pos[a] += u * push
                    self.a_pos[b] -= u * push
            # 모선 keep-out (분리 뒤에 두어 모선 제약이 항상 마지막에 보장)
            for i in range(P):
                if not self.a_alive[i]:
                    continue
                v = self.a_pos[i] - c
                d = float(np.hypot(v[0], v[1]))
                if d < keep:
                    if d < 1e-6:   # 정확히 모선 중심 → 인덱스별 방향으로 밀어냄
                        ang = 2.0 * np.pi * i / float(P)
                        self.a_pos[i] = c + np.array([np.cos(ang), np.sin(ang)]) * keep
                    else:
                        self.a_pos[i] = c + v / d * keep
            np.clip(self.a_pos, 0.0, W, out=self.a_pos)

    def _compute_assignment(self):
        # 경로를 LLM WP 로 직접 주입하므로 클러스터 자동배정은 미사용(표시상 전원 예비).
        super()._compute_assignment()   # _cl_cent 등 갱신(무해)
        self.assign[:] = -1


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
                 assigned_cluster=None)
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


def apply_plan(sim: CommandedSimulator, plan) -> None:
    """CommanderPlan → 시뮬에 경로 주입."""
    sim.set_routes(plan.routes)


__all__ = ["CommandedSimulator", "build_battlefield", "apply_plan"]
