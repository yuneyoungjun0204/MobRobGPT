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
from .sim_bridge import plan_to_assign


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

    clusters = []
    for k in range(cfg.n_clusters):
        if not bool(active[k]) or int(cnt[k]) == 0:
            continue
        bearing = float(np.degrees(np.arctan2(cent[k, 0] - c[0], cent[k, 1] - c[1])) % 360.0)
        clusters.append(EnemyCluster(
            id=k, center=Point(x=float(cent[k, 0]), y=float(cent[k, 1])),
            bearing=bearing, spread=float(spread[k]),
            count=int(cnt[k]), approach_speed=float(cfg.enemy_speed)))

    Kw = env.Kw
    allies = []
    for i in range(env.P):
        alive = bool(env.a_alive[w, i])
        allies.append(AllyShip(
            id=i, pos=Point(x=float(env.a_pos[w, i, 0]), y=float(env.a_pos[w, i, 1])),
            heading=float(env.a_hdg[w, i]), nets_remaining=int(env.a_nets[w, i]),
            alive=alive,
            assigned_cluster=int(env._assign[w, i]) if int(env._assign[w, i]) >= 0 else None,
            route=([Point(x=float(env.route[w, i, k, 0]), y=float(env.route[w, i, k, 1]))
                    for k in range(Kw)] if alive else []),
            deploying=bool(env.doing_net[w, i])))

    if env.e_alive[w].any():
        d = np.hypot(env.e_pos[w, env.e_alive[w], 0] - c[0],
                     env.e_pos[w, env.e_alive[w], 1] - c[1]).min()
        threat = float(np.clip(1.0 - d / (cfg.world_size / 2.0), 0.0, 1.0))
    else:
        threat = 0.0

    return BattlefieldState(
        mothership=Mothership(pos=Point(x=float(c[0]), y=float(c[1])),
                              radius=float(cfg.mothership_radius), threat_level=threat),
        enemy_clusters=clusters, allies=allies,
        constraints=Constraints(net_max_len=float(cfg.net_max_len),
                                ally_speed=float(getattr(cfg, "ally_speed", 6.0)),
                                enemy_speed=float(cfg.enemy_speed),
                                world_size=float(cfg.world_size),
                                max_intercept_radius=float(cfg.world_size / 3.0)),
        command=command)


class CommandedDefenseEnv(DefenseVecEnv):
    """LLM 배정을 존중하는 RL 실행 환경(1월드). 경로는 RL 정책이 기동."""

    def __init__(self, ckpt: str, enemy_mode: str = "rotate", device: str = "cpu",
                 gain: float = 1.0, avoid_steer: bool = False):
        actor, cfg = load_actor(ckpt, device=device)
        cfg.avoid_steer = bool(avoid_steer)   # 기본 OFF: 순수 RL 경로(APF 안전층 없음). 학습분포 이탈 주의.
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

    # ── LLM 계획 주입 (매 결정 현재 상태로 재매핑) ──
    def set_plan(self, plan, command: str | None = None) -> None:
        self._plan = plan
        self._plan_command = command

    def set_command(self, _ignored=None) -> None:
        """호환용: None 이면 계획 해제(전원 예비)."""
        self._plan = None

    def _compute_assignment(self, assign_pref=None):
        super()._compute_assignment(assign_pref)             # 기본 배정 + 교점/중심
        if self._plan is None:
            self._assign[0] = -1                             # 명령 전엔 전원 예비(정지)
            self._assignI[0] = 0.0
            self._assign_cent[0] = np.array(self.center)[None, :]
            return
        state = build_battlefield_defense(self, self._plan_command)
        self._inject(plan_to_assign(self._plan, state))

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
        self._ev = self.fresh_ev()

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
