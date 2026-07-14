"""LLM 지휘관 ↔ 강화학습(RL) 실행 브릿지.

run_commander_ui.py --rl 에서 사용. simulator.py 휴리스틱 대신 **학습된 RL 정책**이 경로를
기동하되, '어느 USV 가 어느 클러스터를 맡을지'는 여전히 LLM 지휘관이 결정한다:

  LLM(deployments: 클러스터→ally_ids, deploy_net, hold)
        → plan_to_assign → CommandedDefenseEnv._assign 주입(매 결정)
        → DefenseVecEnv 의 휴리스틱 baseline + RL 잔차 경로가 그 배정을 따름.

즉 LLM=전략(배정), RL=전술(경로). 정책은 학습 환경(DefenseVecEnv)에서 그대로 구동되어
관측/행동 충실도가 보장된다(잔차 방식: route = 휴리스틱 + action·wp_adjust_max·assigned).
"""
from __future__ import annotations

import numpy as np
import torch

from boatattack_sim.env.defense_env import DefenseVecEnv
from boatattack_sim.env import clustering
from boatattack_sim.model.actor import load_actor

from .schema import (
    BattlefieldState, Mothership, EnemyCluster, AllyShip, Constraints, Point,
)
from .sim_bridge import plan_to_assign, route_crosses_net, covered_by_teammate


# ── numpy obs ↔ torch, 행동 dict → env (train/grpo.py 와 동일) ──
def _obs_to_torch(obs, device):
    N, P = obs["own"].shape[:2]
    B = N * P
    def ten(x, dt):
        return torch.as_tensor(np.ascontiguousarray(x.reshape(B, *x.shape[2:])),
                               dtype=dt, device=device)
    return {"own": ten(obs["own"], torch.float32),
            "enemy": ten(obs["enemy"], torch.float32),
            "enemy_mask": ten(obs["enemy_mask"], torch.bool),
            "ally": ten(obs["ally"], torch.float32),
            "ally_mask": ten(obs["ally_mask"], torch.bool)}


def _act_to_env(a, N, P, Kw):
    cont = a["cont"].view(N, P, -1).cpu().numpy()
    out = {"net_go": a["netgo"].view(N, P, Kw).cpu().numpy()}
    if cont.shape[-1] == 7:
        out["fan"] = cont
    else:
        out["wp"] = cont[..., :Kw * 2].reshape(N, P, Kw, 2)
    if "rot" in a:
        out["rot"] = a["rot"].view(N, P).cpu().numpy()
    return out


def _greedy(p):
    g = {"cont": p["cont_mean"], "netgo": (p["netgo"] > 0).float()}
    if "rot_mean" in p:
        g["rot"] = p["rot_mean"]
    return g


def build_battlefield_defense(env: "DefenseVecEnv", command: str | None = None) -> BattlefieldState:
    """DefenseVecEnv(월드0) 상태 → BattlefieldState (LLM 지휘관 입력). build_battlefield 의 VecEnv 판."""
    w = 0
    cfg = env.cfg
    c = np.array(env.center, np.float64)
    cl = clustering.cluster_by_gaps_vec(env.e_pos, env.e_alive, env.e_hdg, c,
                                        cfg.enemy_speed, cfg.n_clusters, cfg.cluster_gap_deg)
    cent = cl["centroid"][w]; cnt = cl["count"][w]
    active = cl["active"][w]; spread = cl["spread_deg"][w]

    # 설치된 그물 셀의 (방위, 반경) → 클러스터별 net_covered (이미 그물 깔린 접근로)
    ni = env.net_installed[w]
    if ni.any():
        G = ni.shape[0]; cell = cfg.world_size / G
        ii, jj = np.where(ni)
        nx = (ii + 0.5) * cell; ny = (jj + 0.5) * cell
        net_brg = np.degrees(np.arctan2(nx - c[0], ny - c[1])) % 360.0
        net_dist = np.hypot(nx - c[0], ny - c[1])
    else:
        net_brg = net_dist = None
    NET_TOL_DEG = 6.0

    clusters = []
    for k in range(cfg.n_clusters):
        if not bool(active[k]) or int(cnt[k]) == 0:
            continue
        bearing = float(np.degrees(np.arctan2(cent[k, 0] - c[0], cent[k, 1] - c[1])) % 360.0)
        cdist = float(np.hypot(cent[k, 0] - c[0], cent[k, 1] - c[1]))
        net_covered = False
        if net_brg is not None:
            dbrg = np.abs(((net_brg - bearing + 180.0) % 360.0) - 180.0)
            net_covered = bool(np.any((dbrg <= NET_TOL_DEG) & (net_dist < cdist) & (net_dist > 1.0)))
        clusters.append(EnemyCluster(
            id=k, center=Point(x=float(cent[k, 0]), y=float(cent[k, 1])),
            bearing=bearing, spread=float(spread[k]), net_covered=net_covered,
            count=int(cnt[k]), approach_speed=float(cfg.enemy_speed)))

    Kw = env.Kw
    ninst = env.net_installed[w] if hasattr(env, "net_installed") else None
    cov = covered_by_teammate(env._assignI[w], env._assign[w], env.a_alive[w], c, cfg.net_max_len)
    allies = []
    for i in range(env.P):
        alive = bool(env.a_alive[w, i])
        pts = [(float(env.route[w, i, k, 0]), float(env.route[w, i, k, 1])) for k in range(Kw)]
        hits_net = (alive and not bool(env.doing_net[w, i])
                    and route_crosses_net(pts, ninst, cfg.world_size))
        allies.append(AllyShip(
            id=i, pos=Point(x=float(env.a_pos[w, i, 0]), y=float(env.a_pos[w, i, 1])),
            heading=float(env.a_hdg[w, i]), nets_remaining=int(env.a_nets[w, i]),
            alive=alive,
            assigned_cluster=int(env._assign[w, i]) if int(env._assign[w, i]) >= 0 else None,
            route=([Point(x=x, y=y) for x, y in pts] if alive else []),
            deploying=bool(env.doing_net[w, i]),
            route_hits_net=hits_net,
            cluster_covered_by_teammate=bool(cov[i])))

    if env.e_alive[w].any():
        d = np.hypot(env.e_pos[w, env.e_alive[w], 0] - c[0],
                     env.e_pos[w, env.e_alive[w], 1] - c[1]).min()
        threat = float(np.clip(1.0 - d / (cfg.world_size / 2.0), 0.0, 1.0))
    else:
        threat = 0.0

    # 적 포메이션(집중/양동/파상)을 LLM 에 알려 포메이션별 플레이북을 쓰게 한다.
    mode = str(env.world_mode[w]) if getattr(env, "world_mode", None) is not None else ""
    tag = f"[ENEMY FORMATION: {mode}]" if mode else ""
    cmd_full = (f"{tag} {command}".strip() if command else tag) or None

    return BattlefieldState(
        mothership=Mothership(pos=Point(x=float(c[0]), y=float(c[1])),
                              radius=float(cfg.mothership_radius), threat_level=threat),
        enemy_clusters=clusters, allies=allies,
        constraints=Constraints(net_max_len=float(cfg.net_max_len),
                                ally_speed=float(getattr(cfg, "ally_speed", 6.0)),
                                enemy_speed=float(cfg.enemy_speed),
                                world_size=float(cfg.world_size),
                                max_intercept_radius=float(cfg.world_size / 3.0)),
        command=cmd_full)


class CommandedDefenseEnv(DefenseVecEnv):
    """LLM 배정을 존중하는 RL 실행 환경(1월드). 경로는 RL 정책이 기동."""

    def __init__(self, ckpt: str, enemy_mode: str = "rotate", device: str = "cpu",
                 gain: float = 1.0, avoid_steer: bool = False):
        actor, cfg = load_actor(ckpt, device=device)
        cfg.avoid_steer = bool(avoid_steer)   # 기본 OFF: 순수 RL 경로(APF 안전층 없음). 학습분포 이탈 주의.
        cfg.n_clusters = 3                     # 클러스터 최대 3개 (LLM 이 3개 그룹으로 다룸)
        # 파상(wave): 웨이브 간 텀을 확실히 → gap↑(1000→1800), near↓(4000→2600)로 3단이 맵 안(≤6300)에.
        cfg.enemy_wave_near = 2600.0
        cfg.enemy_wave_gap = 1800.0
        cfg.spawn_phase_lo = 1.0     # 스폰 랜덤 당김(학습용 비동기화) 끄기 → 웨이브 텀이 설계대로
        cfg.free_current_wp = True   # 추종 중인 현재 WP도 매 결정 RL 잔차로 변동(고정 해제)
        # 재배정 시엔 새 경로를 WP1(처음)부터 추종 → preserve_ptr_on_reeng 는 끔(기본 False).
        # super().__init__ 가 _compute_assignment 를 부를 수 있으므로 속성 선주입.
        self._actor = actor
        self._device = device
        self.gain = float(gain)
        self._plan = None
        self._plan_command = None
        super().__init__(num_worlds=1, cfg=cfg, enemy_mode=enemy_mode)
        self._actor.eval()
        self._h = self._actor.init_hidden(self.P, device)
        self._micro_ct = 0
        self._ev = None
        self.running = True
        self._SK = ("captures", "breaches", "ally_collisions", "nets_used")
        self.stats = {k: 0 for k in self._SK + ("survived",)}
        self._sprev = {k: 0.0 for k in self._SK}
        self.resolve_conflicts = False  # 기본 OFF: 겹침/중복 판단은 LLM 이 담당(프롬프트). c 키로 코드 강제 ON
        self._cmd_deploy = np.ones(self.P, bool)   # 배별 그물 투척 여부(LLM deploy_net)
        self._cmd_net_legs = [None] * self.P       # 배별 그물 레그 WP 인덱스(LLM net_legs; None=자동)

    # ── LLM 계획 주입 (매 결정 현재 상태로 재매핑) ──
    def set_plan(self, plan, command: str | None = None) -> None:
        self._plan = plan
        self._plan_command = command

    def set_command(self, _ignored=None) -> None:
        """호환용: None 이면 계획 해제(전원 예비)."""
        self._plan = None

    def _compute_assignment(self, assign_pref=None):
        prev = self._assign[0].copy() if getattr(self, "_assign", None) is not None else None
        super()._compute_assignment(assign_pref)             # 기본 배정 + 교점/중심(_assign 덮어씀)
        if self._plan is None:
            self._assign[0] = -1                             # 명령 전엔 전원 예비(정지)
            self._assignI[0] = 0.0
            self._assign_cent[0] = np.array(self.center)[None, :]
            return
        if prev is not None:
            self._assign[0] = prev   # 연속성 기준 = 직전 배정(build_battlefield_defense 가 읽어 sticky)
        state = build_battlefield_defense(self, self._plan_command)
        self._inject(plan_to_assign(self._plan, state))
        # 배별 그물 투척여부·레그·거리배율 (스키마 최소화 후엔 기본값; 구버전 필드 있으면 존중)
        deploy_by = {d.cluster_id: bool(getattr(d, "deploy_net", True)) for d in self._plan.deployments}
        legs_by = {d.cluster_id: getattr(d, "net_legs", None) for d in self._plan.deployments}
        rad_by = {d.cluster_id: float(getattr(d, "radius_adjust", 1.0) or 1.0)
                  for d in self._plan.deployments}
        c = np.array(self.center, np.float64)
        for p in range(self.P):
            k = int(self._assign[0, p])
            self._cmd_deploy[p] = deploy_by.get(k, True) if k >= 0 else True
            self._cmd_net_legs[p] = legs_by.get(k, None) if k >= 0 else None
            # 거리 밀기/당기기: 요격점·중심을 모선 기준으로 radius_adjust 배 → 그물벽 반경 이동
            if k >= 0:
                r = max(0.5, min(1.6, rad_by.get(k, 1.0)))
                if r != 1.0:
                    self._assign_cent[0, p] = c + (self._assign_cent[0, p] - c) * r
                    self._assignI[0, p] = c + (self._assignI[0, p] - c) * r

    def _inject(self, a) -> None:
        K = self.cfg.n_clusters
        c = np.array(self.center, np.float64)
        t = self.rcfg.assign_intercept_t
        cl = clustering.cluster_by_gaps_vec(self.e_pos, self.e_alive, self.e_hdg, c,
                                            self.cfg.enemy_speed, K, self.cfg.cluster_gap_deg)
        cent = cl["centroid"][0]                             # [K,2]
        a = np.asarray(a, np.int64).copy()
        a[~self.a_alive[0]] = -1
        self._assign[0] = a
        ci = np.clip(a, 0, K - 1)
        I = cent + t * (c[None, :] - cent)                   # [K,2] 교점
        self._assignI[0] = np.where((a >= 0)[:, None], I[ci], 0.0)
        self._assign_cent[0] = np.where((a >= 0)[:, None], cent[ci], c[None, :])

    # ── RL 결정 + micro-step (run_rl_play 결정루프의 클래스판) ──
    def _rl_decide(self):
        ot = _obs_to_torch(self.build_obs(), self._device)   # build_obs → _compute_assignment(주입)
        with torch.no_grad():
            p, self._h = self._actor(ot, self._h)
            act = _greedy(p)
        if self.gain != 1.0:
            act = dict(act); act["cont"] = act["cont"] * self.gain
        self._apply_actions(_act_to_env(act, 1, self.P, self._actor.Kw))
        self._apply_net_decision()                           # LLM 의 그물 투척여부·레그로 net_mask 덮어씀
        if self.resolve_conflicts:
            self._resolve_route_conflicts()                  # 경로 겹침 → 무리 중 1대만 남기고 HOLD
        self._ev = self.fresh_ev()

    def _apply_net_decision(self):
        """휴리스틱이 정한 net_mask 를 LLM 의 deploy_net/net_legs 로 덮어쓴다(배정된 배만).
        deploy_net=False → 그물 0(위치만). net_legs=[i,j] → 그 레그에만. None → 휴리스틱 유지."""
        Kw = self.Kw
        for p in range(self.P):
            if int(self._assign[0, p]) < 0:
                continue
            if not self._cmd_deploy[p]:
                self.net_mask[0, p, :] = False; self.doing_net[0, p] = False
            elif self._cmd_net_legs[p] is not None:
                m = np.zeros(Kw, bool)
                for idx in self._cmd_net_legs[p]:
                    if 0 <= int(idx) < Kw:
                        m[int(idx)] = True
                self.net_mask[0, p] = m
                if not m.any():
                    self.doing_net[0, p] = False

    def _resolve_route_conflicts(self):
        """계획 경로가 '진짜 중복(같은 커버리지)'인 배만 HOLD(제자리 정지). 서로 다른 클러스터로
        가는(교차하더라도) 경로는 건드리지 않는다 — 과잉 HOLD 로 방어 구멍이 나기 때문. 중복 판정:
          (1) 같은 클러스터에 2척+ → 요격점 근접/전개중 한 대만 남기고 나머지 HOLD.
          (2) 계획 경로가 이미 설치된 그물을 지남 → 그 커버는 이미 됨(중복) → HOLD.
        전개중(doing_net)인 배는 그물을 마저 깔도록 HOLD 대상에서 제외한다.
        (경로 교차=충돌 위험은 APF/충돌집계가 담당 — 여기서 HOLD 하면 wave 에서 과잉 HOLD.)"""
        w = 0; P = self.P
        active = [p for p in range(P)
                  if int(self._assign[w, p]) >= 0 and bool(self.a_alive[w, p])]
        if len(active) < 1:
            return set()
        R = {p: self.route[w, p] for p in active}
        dI = {p: float(np.hypot(*(self.a_pos[w, p] - self._assignI[w, p]))) for p in active}
        painting = {p: bool(self.doing_net[w, p]) for p in active}
        hold = set()

        # (1) 같은 클러스터 = 중복 → 한 대만 유지(전개중>요격근접), 나머지 HOLD
        by_cl: dict[int, list] = {}
        for p in active:
            by_cl.setdefault(int(self._assign[w, p]), []).append(p)
        for ps in by_cl.values():
            if len(ps) > 1:
                keeper = min([q for q in ps if painting[q]] or ps, key=lambda q: dI[q])
                hold |= {q for q in ps if q != keeper and not painting[q]}

        # (2) 이미 설치된 그물과 겹치는 경로 → 중복 → HOLD
        for p in active:
            if not painting[p] and self._route_hits_net(R[p]):
                hold.add(p)

        for p in hold:                                        # HOLD 적용: 이번 결정 제자리 정지
            self._assign[w, p] = -1
            self.route[w, p, :, :] = self.a_pos[w, p]         # 경로 접어 정지 + 깔끔한 렌더
        return hold

    def _route_hits_net(self, route_p) -> bool:
        """계획 경로가 이미 설치된 그물 셀 근방(±1셀)을 지나면 True (그물 중복 전개 방지)."""
        ni = self.net_installed[0]
        if not ni.any():
            return False
        G = ni.shape[0]; cell = self.cfg.world_size / G
        for x, y in route_p:
            ci = int(x / cell); cj = int(y / cell)
            i0, i1 = max(0, ci - 1), min(G, ci + 2)
            j0, j1 = max(0, cj - 1), min(G, cj + 2)
            if ni[i0:i1, j0:j1].any():
                return True
        return False

    def step(self):
        if bool(self.done[0]):
            self._spawn_worlds(np.array([0])); self._micro_ct = 0; self._ev = None
            self._h = self._actor.init_hidden(self.P, self._device)
            self.stats = {k: 0 for k in self._SK + ("survived",)}
            self._sprev = {k: 0.0 for k in self._SK}
        if self._micro_ct % self.cfg.decision_period == 0:
            self._rl_decide()
            self._sprev = {k: 0.0 for k in self._SK}     # ev 는 결정마다 새로 누적됨
        self._micro(self._ev)
        for k in self._SK:                               # 결정 구간 내 델타 누적
            cur = float(self._ev[k][0]); self.stats[k] += cur - self._sprev[k]; self._sprev[k] = cur
        self._micro_ct += 1
        return self.get_frame()

    def reset(self, seed=None):
        super().reset(seed)                    # 배열 할당 + 스폰 (base)
        self._micro_ct = 0; self._ev = None
        if getattr(self, "_actor", None) is not None:
            self._h = self._actor.init_hidden(self.P, self._device)

    # ── 렌더러 프레임 (draw_scene 계약) ──
    def get_frame(self) -> dict:
        cfg = self.cfg; w = 0
        paths = []
        for i in range(self.P):
            wps = [{"x": float(self.route[w, i, k, 0]),
                    "y": float(self.route[w, i, k, 1]), "paint": False}
                   for k in range(self.Kw)]
            if bool(self.doing_net[w, i]):
                wps.append({"x": float(self.net_end[w, i, 0]),
                            "y": float(self.net_end[w, i, 1]), "paint": True})
            paths.append(wps)
        return {
            "world_size": cfg.world_size, "cell_size": cfg.cell_size,
            "t": int(self.t[w]), "done": bool(self.done[w]),
            "mothership": self.center, "mothership_radius": cfg.mothership_radius,
            "moback_size": cfg.moback_size, "moback_heading": cfg.moback_heading,
            "enemy_pos": self.e_pos[w], "enemy_hdg": self.e_hdg[w],
            "enemy_alive": self.e_alive[w], "enemy_size": cfg.enemy_size,
            "ally_pos": self.a_pos[w], "ally_hdg": self.a_hdg[w], "ally_paths": paths,
            "ally_nets": self.a_nets[w], "ally_painting": self.doing_net[w],
            "ally_alive": self.a_alive[w],
            "assign": self._assign[w], "assignI": self._assignI[w],
            "route": self.route[w], "net_mask": self.net_mask[w],
            "ship_len": cfg.ship_len, "ship_wid": cfg.ship_wid, "painted": self.painted[w],
            "selected": -1, "manual": False, "running": self.running, "net_stage": 0,
            "stats": dict(self.stats),
            "n_alive": int(self.e_alive[w].sum()),
            "n_clusters": cfg.n_clusters, "cluster_gap_deg": cfg.cluster_gap_deg,
            "enemy_speed": cfg.enemy_speed, "show_clusters": True,
            "show_residual": False, "wp_adjust_max": cfg.wp_adjust_max,
            "enemy_mode": str(self.world_mode[w]),
        }


__all__ = ["CommandedDefenseEnv", "build_battlefield_defense"]
