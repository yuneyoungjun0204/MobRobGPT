"""시뮬레이터 ↔ 지휘관 브릿지 (배정 방식).

- LLM 은 '어느 클러스터에 어느 USV(ally_ids)'를 결정(CommanderPlan.deployments).
- CommandedSimulator 는 AUTO 모드 유지 → 시뮬의 heuristic_plan + _build_cluster_path 가
  배정(self.assign)으로부터 실제 그물벽 경로를 기하로 생성(요격 링 위 수직벽). 모델이 약해도 확실히 막힘.
- LLM 배정은 _compute_assignment 에서 주입. 명령 전엔 전원 예비(정지).
- 40m 비율 축소(scaled_config) · 위성 앵커(geo) · 아군/모선 충돌 회피(_separate) 유지.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from boatattack_sim.env.simulator import Simulator
from boatattack_sim.env.config import (DEFAULT_CONFIG, DEFAULT_REWARD,
                                       SimConfig, RewardCfg)
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
    """12600m 기본설정을 target world_size 로 '비율 유지' 축소 → 동역학 보존. 적=아군×mult.

    ★ 스케일 규칙은 SimConfig.at_scale 하나로 통일(단일 소스).
      예전엔 여기서 14개 필드만 손으로 곱했는데, action_grid_half·cell_r_min/max·
      cell_spacing·enemy_wave_near·ally_mother_radius·wp_repel_*·wp_adjust_max·
      route_step·norm_k_* 등이 빠져 **정규화 기준(action_grid_half)이 안 줄어드는**
      치명적 구멍이 있었다. 이제 _LEN_SIM 전체가 자동으로 스케일된다.
    """
    return SimConfig.at_scale(
        world_size=world_size,
        enemy_speed_mult=enemy_speed_mult,     # 무차원 — 스케일 후 적용
        geo_lat=geo_lat, geo_lon=geo_lon,
        n_clusters=3,                          # 클러스터 최대 3개
    )


def scaled_reward(world_size: float = DEFAULT_WORLD_SIZE) -> RewardCfg:
    """world_size 에 맞춰 길이 차원 보상 파라미터(영향반경·배정 cost[m])를 축소.

    ★ 예전 scaled_config 는 RewardCfg 를 전혀 안 건드렸다 →
      assign_sticky_bonus(6000m)·w_assign_bias(2000m)·avoid_r(600m) 가 base 스케일로
      남아, 33m 맵에선 맵 전체보다 큰 상수가 되어 배정/회피가 완전히 고착된다.
    """
    r = RewardCfg()
    r.apply_scale(world_size / DEFAULT_CONFIG.world_size)
    return r


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
        P = self.cfg.n_allies
        self._deploy_net = np.ones(P, bool)    # 배별 '지금 그물 투척?' (LLM deploy_net, 100스텝마다)
        self._last_deploy = np.ones(P, bool)   # 직전 투척여부(변경 시 경로 재생성 트리거)
        self._net_legs = [None] * P            # 배별 그물 깔 WP 인덱스(LLM net_legs; None=자동)
        self._last_legs = [None] * P           # 직전 net_legs (변경 시 경로 재생성 트리거)

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
            # 배별 그물 투척 여부·레그 = 담당 클러스터의 deploy_net/net_legs (LLM이 결정)
            deploy_by_cluster = {d.cluster_id: bool(getattr(d, "deploy_net", True)) for d in self._plan.deployments}
            legs_by_cluster = {d.cluster_id: getattr(d, "net_legs", None) for d in self._plan.deployments}
            for i in range(self.cfg.n_allies):
                k = int(self.assign[i])
                self._deploy_net[i] = deploy_by_cluster.get(k, True) if k >= 0 else True
                self._net_legs[i] = legs_by_cluster.get(k, None) if k >= 0 else None
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

    def heuristic_plan(self):
        """One-Way 식 WP 순차 추종: decision_period 강제 재계획을 없애 배가 배정된 WP를
        끝까지 따라가게 한다(도착 WP는 _step_allies 가 pop → 계속 다음 WP로 진행).
        경로 재생성은 ①경로 소진 ②담당 클러스터 변경 ③투척여부(deploy_net) 변경 시에만."""
        cfg = self.cfg
        for i in range(cfg.n_allies):
            if not self.a_alive[i]:
                self.a_paths[i] = []; self._plan_cluster[i] = -2
                continue
            k = int(self.assign[i])
            if k < 0 or self.a_nets[i] <= 0:          # 예비/그물소진 → 정지
                if not self.a_painting[i]:
                    self.a_paths[i] = []
                self._plan_cluster[i] = k
                continue
            if self.a_painting[i]:                    # 전개중 = 현재 그물 유지(떨림 방지)
                continue
            dep_changed = (bool(self._deploy_net[i]) != bool(self._last_deploy[i])
                           or self._net_legs[i] != self._last_legs[i])
            if self._plan_cluster[i] == k and self.a_paths[i] and not dep_changed:
                continue                              # 담당 동일·경로 잔존·투척여부·레그 유지 → 그대로
            self.a_paths[i] = self._build_cluster_path(i, k)
            self._plan_cluster[i] = k
            self._plan_t[i] = self.t
            self._last_deploy[i] = bool(self._deploy_net[i])
            self._last_legs[i] = self._net_legs[i]

    def _build_cluster_path(self, i, k):
        """그물 투척 여부·레그를 LLM 이 결정: deploy_net=False → 요격 진입점까지만(미전개).
        net_legs=None → 휴리스틱 기본(요격 링 구간 전개). []=미전개. [i,j]=그 WP 인덱스만 전개."""
        path = super()._build_cluster_path(i, k)
        legs = self._net_legs[i]
        # 투척 안 함(deploy_net False 또는 net_legs=[]) → 진입 WP 하나만, 페인트 끔
        if path and (not self._deploy_net[i] or (legs is not None and len(legs) == 0)):
            wp = dict(path[0]); wp["paint"] = False
            return [wp]
        # net_legs 지정 → 그 WP 인덱스에만 그물(나머지 transit)
        if path and legs is not None:
            legset = {int(x) for x in legs}
            for idx, wp in enumerate(path):
                wp["paint"] = idx in legset
        return path


def route_crosses_net(pts, net_installed, world_size) -> bool:
    """경로 WP 목록이 이미 설치된 그물 셀(±1셀) 근방을 지나면 True (그물 접촉 격침 위험)."""
    if net_installed is None or not net_installed.any() or not pts:
        return False
    G = net_installed.shape[0]; cell = world_size / G
    for x, y in pts:
        ci = int(x / cell); cj = int(y / cell)
        i0, i1 = max(0, ci - 1), min(G, ci + 2)
        j0, j1 = max(0, cj - 1), min(G, cj + 2)
        if net_installed[i0:i1, j0:j1].any():
            return True
    return False


def covered_by_teammate(assignI, assign, alive, center, net_max_len):
    """각 아군의 담당 클러스터 접근로가 '다른 클러스터에 배정된' 다른 아군의 그물벽에 덮이는지 [P] bool.

    그물벽은 요격점(assignI)에 수직으로 net_max_len 길이 → 모선 기준 각폭 half=atan(L/2 / r).
    아군 i 의 접근 방위가 다른 아군 j(다른 클러스터)의 각폭 안에 들면 i 는 중복(덮임).
    """
    P = len(assign)
    c = np.asarray(center, np.float64)
    out = [False] * P
    brg = []
    rad = []
    half = []
    for p in range(P):
        v = np.asarray(assignI[p], np.float64) - c
        r = float(np.hypot(v[0], v[1]))
        rad.append(r)
        brg.append(float(np.degrees(np.arctan2(v[0], v[1])) % 360.0) if r > 1.0 else None)
        half.append(float(np.degrees(np.arctan2(net_max_len / 2.0, max(r, 1.0)))))
    for i in range(P):
        if not alive[i] or int(assign[i]) < 0 or brg[i] is None:
            continue
        for j in range(P):
            if j == i or not alive[j] or int(assign[j]) < 0 or int(assign[j]) == int(assign[i]):
                continue
            if brg[j] is None:
                continue
            d = abs(((brg[i] - brg[j] + 180.0) % 360.0) - 180.0)
            if d <= half[j]:                        # i 접근방위가 j 그물벽 각폭 안 → 덮임
                out[i] = True
                break
    return out


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

    ninst = getattr(sim, "net_installed", None)
    cov = covered_by_teammate(sim.assignI, sim.assign, sim.a_alive, c, cfg.net_max_len)
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
                 deploying=bool(sim.a_painting[i]),
                 route_hits_net=(bool(sim.a_alive[i]) and not bool(sim.a_painting[i])
                                 and route_crosses_net([(w["x"], w["y"]) for w in sim.a_paths[i]],
                                                       ninst, cfg.world_size)),
                 cluster_covered_by_teammate=bool(cov[i]))
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


def _intercept_point(cx, cy, mx, my, v_a, v_e, r_cap):
    """모선-클러스터 선상의 요격 지점. 반경 = v_a·d/(v_a+v_e), max_intercept 로 캡."""
    dx, dy = cx - mx, cy - my
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (mx, my)
    r = min(v_a * d / (v_a + v_e), r_cap)
    return (mx + dx / d * r, my + dy / d * r)


def _segments_cross(a, b, c, d):
    """선분 ab, cd 가 교차하면 True (두 배의 직선 경로 충돌 위험 판정)."""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    return (ccw(a, b, c) * ccw(a, b, d) < 0) and (ccw(c, d, a) * ccw(c, d, b) < 0)


def _ship_cost(a, ipt, assigned_pairs, W):
    """아군 a 를 요격점 ipt 로 보낼 때의 '효율+안전' 비용(작을수록 좋음).

    효율: 요격점까지 이동거리(≈도착시간) + 선회량 + 그물 보유.
    안전: 이미 배정된 배들의 경로와 교차하면(충돌 위험) 큰 페널티.
    """
    px, py = a.pos.x, a.pos.y
    ix, iy = ipt
    travel = math.hypot(ix - px, iy - py)                       # 효율: 이동거리
    desired = math.degrees(math.atan2(ix - px, iy - py)) % 360.0
    turn = abs(((desired - a.heading + 180.0) % 360.0) - 180.0)  # 효율: 선회량[deg]
    turn_pen = (turn / 180.0) * (W * 0.06)
    nets_pen = 0.0 if a.nets_remaining > 0 else (W * 2.0)         # 그물 없으면 사실상 배제
    cross_pen = sum(W * 0.7 for (p, q) in assigned_pairs
                    if _segments_cross((px, py), (ix, iy), p, q))  # 안전: 경로 교차
    return travel + turn_pen + nets_pen + cross_pen


def plan_to_assign(plan, state: BattlefieldState) -> np.ndarray:
    """CommanderPlan → sim.assign 배열[P] (아군별 담당 클러스터 idx, -1=예비).

    배정 주체는 LLM: 각 deployment 의 ally_ids(어느 USV) 를 그대로 존중한다. LLM 이 비워
    두거나 지정한 배가 죽어 담당이 0인 클러스터만, 코드가 '효율(거리·선회·그물)+안전(경로
    교차 회피)' 복합점수(_ship_cost)로 대신 골라 채운다(폴백).

    HOLD: plan.hold_ships 의 아군은 배정과 무관하게 assign=-1(제자리 정지)로 덮어쓴다.
    """
    P = len(state.allies)
    assign = np.full(P, -1, np.int64)
    clusters = {c.id: c for c in state.enemy_clusters}
    threat = {c.id: c.count for c in state.enemy_clusters}
    allies = {a.id: a for a in state.allies}
    mx, my = state.mothership.pos.x, state.mothership.pos.y
    con = state.constraints
    v_a = max(con.ally_speed, 1e-6)
    r_cap = con.max_intercept_radius
    W = con.world_size
    icept = {c.id: _intercept_point(c.center.x, c.center.y, mx, my, v_a,
                                    max(c.approach_speed, 1e-6), r_cap)
             for c in state.enemy_clusters}

    available = {a.id for a in state.allies if a.alive}          # 격침된 배는 배정 제외
    assigned_pairs: list = []                                    # 교차검사용 (배pos, 요격점)

    def commit(aid, cid):
        assign[aid] = cid
        available.discard(aid)
        assigned_pairs.append(((allies[aid].pos.x, allies[aid].pos.y), icept[cid]))

    # 위협 큰 클러스터 먼저 (아군 부족 시 우선 커버 + 교차검사 순서 안정)
    deps = sorted((d for d in plan.deployments if d.cluster_id in clusters),
                  key=lambda d: -threat.get(d.cluster_id, 0))

    # 1) LLM 이 지정한 ally_ids 존중 (배정 주체 = LLM)
    for d in deps:
        for aid in d.ally_ids:
            if aid in available:
                commit(aid, d.cluster_id)

    # 2) 담당 배가 0인 클러스터 → 전역 최소비용 매칭(헝가리안)으로 효율 최적 배정.
    #    탐욕(클러스터별 최근접)은 한 배가 먼저 가져가면 다른 배가 더 나은 매칭을 놓쳐 총
    #    이동거리·선회가 커짐. 전 아군×미담당클러스터 비용행렬을 한 번에 최소화 → 전역 최적.
    uncovered = [d.cluster_id for d in deps
                 if not any(assign[j] == d.cluster_id for j in range(P))]
    avail = sorted(available)
    if uncovered and avail:
        slots = uncovered[:len(avail)]          # 아군 부족 시 위협 큰 클러스터 우선(deps=threat desc)
        # ★ 연속성(sticky): 현재 담당 클러스터면 비용 차감 → 타겟이 매 스텝 뒤바뀌는 것 억제
        #   (이만큼 더 싸야 전환). 담당 클러스터가 사라지면 매칭 안 돼 자연히 재배정됨.
        STICKY = W * 0.45   # 연속성 최우선 — 현재 담당을 크게 선호(확실히 더 나을 때만 전환)
        cost = np.array([[_ship_cost(allies[aid], icept[cid], assigned_pairs, W)
                          - (STICKY if allies[aid].assigned_cluster == cid else 0.0)
                          for cid in slots] for aid in avail], dtype=float)
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)          # 전역 최소비용(헝가리안)
        except Exception:                                     # scipy 없으면 탐욕 폴백
            rows, cols = [], []
            order = sorted(range(cost.size), key=lambda f: cost.flat[f])
            for f in order:
                ri, ci = divmod(f, len(slots))
                if ri not in rows and ci not in cols:
                    rows.append(ri); cols.append(ci)
        for ri, ci in zip(list(rows), list(cols)):
            commit(avail[ri], slots[ci])

    # 2.5) 효율 보정(2-opt): 배정된 배 쌍의 담당 클러스터를 맞바꿔 총 (이동거리+선회+경로교차)
    #   이 뚜렷이 줄면 스왑한다. LLM 이 낸 '왼쪽 클러스터를 오른쪽 배가' 식 교차/비효율 배정을
    #   제거(같은 배·클러스터 집합에서 '짝'만 최적화 → 최근접·최소선회·비교차). 근소차는 연속성
    #   위해 유지(임계 W*0.1) → 교차(큰 페널티)나 명백한 비효율만 고침.
    idxs = [aid for aid in range(P) if assign[aid] >= 0]
    EPS = W * 0.1
    improved = True
    guard = 0
    while improved and guard < 20:
        improved = False
        guard += 1
        for xi in range(len(idxs)):
            for yi in range(xi + 1, len(idxs)):
                a1, a2 = idxs[xi], idxs[yi]
                c1, c2 = int(assign[a1]), int(assign[a2])
                if c1 == c2 or c1 not in icept or c2 not in icept:
                    continue
                s1 = (allies[a1].pos.x, allies[a1].pos.y)
                s2 = (allies[a2].pos.x, allies[a2].pos.y)
                cur = (_ship_cost(allies[a1], icept[c1], [(s2, icept[c2])], W)
                       + _ship_cost(allies[a2], icept[c2], [(s1, icept[c1])], W))
                swp = (_ship_cost(allies[a1], icept[c2], [(s2, icept[c1])], W)
                       + _ship_cost(allies[a2], icept[c1], [(s1, icept[c2])], W))
                if swp < cur - EPS:
                    assign[a1], assign[a2] = c2, c1
                    improved = True

    # 3) HOLD: 지정 아군은 제자리 정지(assign=-1). 전개중이면 그물은 마저 설치됨.
    for i in getattr(plan, "hold_ships", None) or []:
        if 0 <= int(i) < P:
            assign[int(i)] = -1

    # 4) ★ 최소 1대 활성 보장(전원 HOLD/예비 금지): 살아있는 배가 전부 assign<0 이면,
    #    가장 위협 큰 활성 클러스터에 가장 싸게 갈 수 있는 1대를 강제 배정(HOLD 해제).
    alive_ids = [a.id for a in state.allies if a.alive]
    active_cl = [c for c in state.enemy_clusters]
    if alive_ids and active_cl and not any(assign[i] >= 0 for i in alive_ids):
        tgt = max(active_cl, key=lambda c: threat.get(c.id, 0))     # 가장 위협 큰 클러스터
        best = min(alive_ids, key=lambda i: _ship_cost(allies[i], icept[tgt.id], [], W))
        assign[best] = tgt.id                                       # 이 배는 활성(HOLD 해제)
    return assign


def apply_plan(sim: CommandedSimulator, plan, state: BattlefieldState) -> None:
    """CommanderPlan → 시뮬에 계획 주입(매 스텝 재매핑)."""
    sim.set_plan(plan, state.command)


__all__ = ["CommandedSimulator", "build_battlefield", "plan_to_assign", "apply_plan",
           "scaled_config", "scaled_reward"]
