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
from boatattack_sim.env.config import DEFAULT_CONFIG, DEFAULT_REWARD
from boatattack_sim.env import clustering

from . import geometry as GEO
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
        n_clusters=3,                          # 클러스터 최대 3개
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


# ── 배정 기하는 commander/geometry.py 단일 소스 (fallback 지휘관과 공유) ──
#    폴백이 boatattack_sim 을 끌어오지 않도록 순수 기하만 그쪽에 뒀다. 아래는 하위호환 별칭.
_intercept_point = GEO.intercept_point
_segments_cross = GEO.segments_cross
_ship_cost = GEO.ship_cost


def plan_to_assign(plan, state: BattlefieldState, mode: str = "llm") -> np.ndarray:
    """CommanderPlan → sim.assign 배열[P] (아군별 담당 클러스터 idx, -1=예비).

    배정 주체는 LLM: 각 deployment 의 ally_ids(어느 USV) 를 그대로 존중한다. LLM 이 비워
    두거나 지정한 배가 죽어 담당이 0인 클러스터만, 코드가 '효율(거리·선회·그물)+안전(경로
    교차 회피)' 복합점수(_ship_cost)로 대신 골라 채운다(폴백).

    HOLD: plan.hold_ships 의 아군은 배정과 무관하게 assign=-1(제자리 정지)로 덮어쓴다.

    mode — ★배정 권한 (기본 "llm")
    ──────────────────────────────
    "llm"    : **순수 LLM 배정.** LLM 이 `ally_ids` 로 명시한 배만 배정된다. 코드는 아무것도
               보완하지 않는다 — 헝가리안 자동배정도, 2-opt 효율보정도, 전원HOLD 방지도 없다.
               LLM 이 비워 둔 클러스터는 담당 없이 남고, 지목 안 된 배는 예비(정지)로 남는다.
               비효율·교차·모선 관통 배정도 그대로 나간다. 결과의 책임이 전부 지휘관에게 있다.
    "hybrid" : LLM 명시분은 잠그되(2-opt 면제), 빈 클러스터는 헝가리안으로 채우고 자동배정된
               배들끼리만 2-opt. 전원HOLD 방지도 동작. LLM 판단 + 코드 보완의 절충.
    "code"   : 종래 동작 — 모든 배정 배가 2-opt 대상. 효율은 최적이지만 LLM 판단이 뒤집힌다.

    ※ 왜 이 스위치가 생겼나 (2026-08-20 실측):
      "code" 시절, 실제 전장 76개 배정안 중 **64.5%가 코드에 의해 변경**됐고(배 단위 53.3%),
      14개 상황 중 **57%는 LLM 이 무엇을 내든 최종 배정이 동일**했다. 2-opt 는 임계 W*0.1 로
      최대 20회 반복하는 사실상 완전탐색이라, 초기값(LLM 배정)이 무엇이든 같은 국소최적으로
      수렴한다. 즉 LLM 호출이 결과에 영향을 주지 못했다. 화면의 rationale 만 LLM 것이라
      '지휘관이 결정한다'는 착시가 있었다.

      ※ 참고: 모선 관통 경로는 코드 보완이 막아주던 것이 아니다. heuristic_plan 으로 잰
        배정 감사 59회에서 관통 8건이 나왔다(경로 교차는 0건) — 휴리스틱도 똑같이 낸다.
    """
    mode = str(mode).lower()
    if mode not in ("llm", "hybrid", "code"):
        raise ValueError(f"알 수 없는 배정 mode: {mode!r} (llm|hybrid|code)")
    pure = (mode == "llm")          # 코드 보완 일절 없음
    respect_llm = (mode != "code")  # LLM 명시분 2-opt 면제
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
    llm_locked: set = set()          # ★ LLM 이 ally_ids 로 직접 지목한 배 (2-opt 면제 대상)

    def commit(aid, cid, by_llm=False):
        assign[aid] = cid
        available.discard(aid)
        assigned_pairs.append(((allies[aid].pos.x, allies[aid].pos.y), icept[cid]))
        if by_llm:
            llm_locked.add(aid)

    # 위협 큰 클러스터 먼저 (아군 부족 시 우선 커버 + 교차검사 순서 안정)
    deps = sorted((d for d in plan.deployments if d.cluster_id in clusters),
                  key=lambda d: -threat.get(d.cluster_id, 0))

    # 1) LLM 이 지정한 ally_ids 존중 (배정 주체 = LLM)
    #    ★ by_llm=True → respect_llm 이면 아래 2-opt 가 이 배들을 건드리지 않는다.
    for d in deps:
        for aid in d.ally_ids:
            if aid in available:
                commit(aid, d.cluster_id, by_llm=True)

    # 2) 담당 배가 0인 클러스터 → 전역 최소비용 매칭(헝가리안)으로 효율 최적 배정.
    #    탐욕(클러스터별 최근접)은 한 배가 먼저 가져가면 다른 배가 더 나은 매칭을 놓쳐 총
    #    이동거리·선회가 커짐. 전 아군×미담당클러스터 비용행렬을 한 번에 최소화 → 전역 최적.
    #    ★ mode="llm" 이면 통째로 건너뛴다 — 빈 클러스터는 담당 없이 남는다(LLM 뜻 그대로).
    uncovered = [] if pure else [d.cluster_id for d in deps
                                 if not any(assign[j] == d.cluster_id for j in range(P))]
    avail = sorted(available)
    if uncovered and avail:
        slots = uncovered[:len(avail)]          # 아군 부족 시 위협 큰 클러스터 우선(deps=threat desc)
        # ★ 연속성(sticky): 현재 담당 클러스터면 비용 차감 → 타겟이 매 스텝 뒤바뀌는 것 억제
        #   (이만큼 더 싸야 전환). 담당 클러스터가 사라지면 매칭 안 돼 자연히 재배정됨.
        #   ★ id 뿐 아니라 방위로도 같은 무리를 인정한다 — 클러스터 id 는 매 결정 재부여되므로
        #     id 만 보면 무리가 그대로인데도 연속성이 끊긴 것으로 오판한다(GEO.sticky_bonus).
        cost = np.array([[_ship_cost(allies[aid], icept[cid], assigned_pairs, W)
                          - GEO.sticky_bonus(allies[aid], clusters[cid], W)
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
    #   이 뚜렷이 줄면 스왑한다. 같은 배·클러스터 집합에서 '짝'만 최적화(최근접·최소선회·비교차).
    #   근소차는 연속성 위해 유지(임계 W*0.1) → 교차(큰 페널티)나 명백한 비효율만 고침.
    #
    #   ★ respect_llm=True (기본): LLM 이 직접 지목한 배(llm_locked)는 **대상에서 제외**한다.
    #     시스템이 자동으로 채운 배들끼리만 스왑한다. 이 잠금이 없으면 2-opt 가 사실상
    #     완전탐색이라 LLM 배정을 전부 같은 국소최적으로 되돌려 버린다(위 docstring 실측 참조).
    #     대신 LLM 이 비효율·교차 배정을 내면 그대로 나간다 — 그게 '권한을 준다'는 뜻이다.
    #     mode="llm" 이면 대상이 비어 2-opt 자체가 돌지 않는다.
    idxs = [] if pure else [aid for aid in range(P)
                            if assign[aid] >= 0 and not (respect_llm and aid in llm_locked)]
    if len(idxs) > 1:
        # ★ fallback._pick 과 **같은 함수**를 쓴다(geometry.refine_pairs, 배 id 순 정규화).
        #   예전엔 여기와 폴백이 각자 2-opt 를 돌려 순회 순서가 달라 서로 다른 국소최적에
        #   수렴했다(diversionary 결정 8%가 갈렸다).
        pool = [aid for aid in idxs if int(assign[aid]) in icept]
        if len(pool) > 1:
            for a, c in GEO.refine_pairs([(allies[aid], int(assign[aid])) for aid in pool],
                                         icept, W, clusters=clusters):
                assign[a.id] = c

    # 3) HOLD: 지정 아군은 제자리 정지(assign=-1). 전개중이면 그물은 마저 설치됨.
    for i in getattr(plan, "hold_ships", None) or []:
        if 0 <= int(i) < P:
            assign[int(i)] = -1

    # 4) ★ 최소 1대 활성 보장(전원 HOLD/예비 금지): 살아있는 배가 전부 assign<0 이면,
    #    가장 위협 큰 활성 클러스터에 가장 싸게 갈 수 있는 1대를 강제 배정(HOLD 해제).
    #    ★ mode="llm" 이면 이 보정도 하지 않는다 — 전원 정지도 LLM 의 결정으로 존중한다.
    alive_ids = [] if pure else [a.id for a in state.allies if a.alive]
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
           "scaled_config"]
