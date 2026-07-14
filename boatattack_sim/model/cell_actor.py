"""
boatattack_sim/model/cell_actor.py — 셀선택 Pointer Actor (어텐션)

관측 토큰(모선0,0 전역 프레임): own·ally·enemy(원본) → self-attention → own 컨텍스트.
후보셀 → cell 임베딩(keys). Pointer: own 컨텍스트 query 로 **cell_nets개 셀을 순차 선택**
  (이전 선택을 query 에 더하고 마스킹 → 중복 방지). 그물 = 선택에 내재(net_go 붕괴 없음).

행동/분포:
  sample(p) → {"cells":[B,K]} (K=cfg.cell_nets)
  logp_entropy(p,act) = Σ_k log Cat(cell_k) , Σ_k H(Cat_k)   (GRPO/MAPPO logp 인터페이스 호환)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..env.config import SimConfig, DEFAULT_CONFIG
from .actor import mlp, _SABlock

# 관측 피처 차원 (defense_env.build_cell_obs 와 1:1)
#   own 9 = pos2·head2·nets1·doing1 + 배정요격점2·할당플래그1 (배별 WP 구분)
OWN_F, ALLY_F, ENEMY_F, CELL_F = 9, 6, 6, 5


def cell_obs_dims(cfg):
    """관측 피처 차원 (own, ally, enemy, cell). build_cell_obs 와 1:1.
    cfg.cell_obs_slim=True → 셀기준 대폭축소 관측."""
    if getattr(cfg, "cell_obs_slim", False):
        return 5, 2, 3, 5      # own[셀pos2·셀요격2·배정1] ally[셀pos2] enemy[셀중심2·차지셀1] cell[5]
    return OWN_F, ALLY_F, ENEMY_F, CELL_F   # full(기존)


def cell_obs_to_torch(obs, device):
    """build_cell_obs [N,P,...] → 토치 [B=N*P,...] (own/ally/enemy/cell + 마스크)."""
    N, P = obs["own"].shape[:2]; B = N * P
    def t(x, dt): return torch.as_tensor(np.ascontiguousarray(x.reshape(B, *x.shape[2:])),
                                         dtype=dt, device=device)
    return {"own": t(obs["own"], torch.float32),
            "ally": t(obs["ally"], torch.float32), "ally_mask": t(obs["ally_mask"], torch.bool),
            "enemy": t(obs["enemy"], torch.float32), "enemy_mask": t(obs["enemy_mask"], torch.bool),
            "cell": t(obs["cell"], torch.float32), "cell_mask": t(obs["cell_mask"], torch.bool)}


class CellPointerActor(nn.Module):
    def __init__(self, cfg: SimConfig = DEFAULT_CONFIG, d: int = 128,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        self.cfg = cfg; self.d = d
        self.K = int(cfg.cell_nets)                  # 선택할 셀 수(=그물 수)
        self.Kw = cfg.transit_wp                     # 트레이너 호환
        self.recurrent = bool(getattr(cfg, "cell_recurrent", False))   # ★ LSTM 시간기억
        own_f, ally_f, enemy_f, cell_f = cell_obs_dims(cfg)   # ★ full/slim 자동
        self.own_in = nn.Linear(own_f, d)
        self.ally_in = nn.Linear(ally_f, d)
        self.enemy_in = nn.Linear(enemy_f, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))     # own/ally/enemy
        self.blocks = nn.ModuleList([_SABlock(d, heads) for _ in range(layers)])
        self.cell_in = mlp([cell_f, d, d])                  # 셀 → keys
        if self.recurrent:
            self.lstm = nn.LSTM(d, d, batch_first=True)     # ★ own 컨텍스트 시간기억(배별)
        self.q_proj = nn.Linear(d, d)                       # query 투영
        self.log_std = nn.Parameter(torch.zeros(1))         # 로깅 호환용 더미(손실 무관)
        # ── ★ 하이브리드: 선택셀 내 미세 연속 오프셋(coarse-to-fine) ──
        self.hybrid = bool(getattr(cfg, "cell_hybrid", False))
        if self.hybrid:
            self.off_mu = mlp([2 * d, d, 2])                # [ctx ⊕ 선택셀key] → μ(x,y) ∈ raw
            self.off_logstd = nn.Parameter(torch.full((2,), -1.2))   # std≈0.3 (초기 미세탐색)
            nn.init.zeros_(self.off_mu[-1].weight); nn.init.zeros_(self.off_mu[-1].bias)  # μ≈0 시작(=순수셀)

    def init_hidden(self, B, device="cpu"):
        if not self.recurrent:
            return None
        z = torch.zeros(1, B, self.d, device=device)
        return (z, z.clone())                               # (h0, c0) [1,B,d]

    def _encode(self, obs):
        own = self.own_in(obs["own"]).unsqueeze(1) + self.type_emb[0]     # [B,1,d]
        ally = self.ally_in(obs["ally"]) + self.type_emb[1]              # [B,A,d]
        enemy = self.enemy_in(obs["enemy"]) + self.type_emb[2]          # [B,M,d]
        x = torch.cat([own, ally, enemy], 1)
        B = own.shape[0]
        own_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_pad = torch.cat([own_pad, obs["ally_mask"], obs["enemy_mask"]], 1)
        for blk in self.blocks:
            x = blk(x, key_pad)
        ctx = x[:, 0]                                                    # own 토큰 [B,d]
        cell_keys = self.cell_in(obs["cell"])                           # [B,C,d]
        cell_valid = ~obs["cell_mask"]                                  # [B,C] True=유효
        return ctx, cell_keys, cell_valid

    def forward(self, obs, hidden=None):
        ctx, cell_keys, cell_valid = self._encode(obs)
        h_out = None
        if self.recurrent:                                  # ★ own 컨텍스트 → LSTM(시간기억)
            if hidden is None:
                hidden = self.init_hidden(ctx.shape[0], ctx.device)
            out, h_out = self.lstm(ctx.unsqueeze(1), hidden)   # [B,1,d]
            ctx = out[:, 0]
        return {"ctx": ctx, "cell_keys": cell_keys, "cell_valid": cell_valid}, h_out

    def _logits(self, q, cell_keys, avail):
        """q[B,d], cell_keys[B,C,d], avail[B,C] → 셀 logits[B,C] (무효/기선택=-inf).
        ★ 전부 무효인 행은 uniform 으로 폴백(Categorical(all -inf) 크래시 방지)."""
        qk = self.q_proj(q)                                             # [B,d]
        score = (qk.unsqueeze(1) * cell_keys).sum(-1) / (self.d ** 0.5)  # [B,C]
        avail = avail | (~avail.any(-1, keepdim=True))                  # 전부무효 행 → 전부 유효로(폴백)
        return score.masked_fill(~avail, float("-inf"))

    def _off_mu(self, ctx, key_k):
        """선택셀 key + 상태 컨텍스트 → 오프셋 평균 μ[B,2] (raw, env서 clip·스케일)."""
        return self.off_mu(torch.cat([ctx, key_k], -1))

    def _decode(self, p, cells=None, offsets=None):
        """순차 K선택. cells=None → 샘플, 아니면 주어진 cells 의 logp/entropy 재생.
        하이브리드: 각 선택셀에 연속 오프셋(Gaussian) 동시 처리.
          offset logp 포함 조건 = hybrid AND (샘플링 | offsets 제공). (BC 는 cells만 → 셀 logp만)
        반환 (picks[B,K], offs[B,K,2]|None, logp[B], entropy[B])."""
        ctx = p["ctx"]; cell_keys = p["cell_keys"]; avail = p["cell_valid"].clone()
        B = ctx.shape[0]; ar = torch.arange(B, device=ctx.device)
        q = ctx; picks = []; offs = []
        lp = torch.zeros(B, device=ctx.device); ent = torch.zeros(B, device=ctx.device)
        use_off = self.hybrid and (cells is None or offsets is not None)
        for k in range(self.K):
            logits = self._logits(q, cell_keys, avail)
            dist = Categorical(logits=logits)
            pick = dist.sample() if cells is None else cells[:, k]
            lp = lp + dist.log_prob(pick)
            ent = ent + dist.entropy()
            picks.append(pick)
            key_k = cell_keys[ar, pick]                                  # [B,d] 선택셀 key
            if use_off:
                mu = self._off_mu(ctx, key_k); std = self.off_logstd.exp()
                odist = torch.distributions.Normal(mu, std)
                o = odist.sample() if offsets is None else offsets[:, k]
                lp = lp + odist.log_prob(o).sum(-1)                      # 연속 오프셋 logp
                ent = ent + odist.entropy().sum(-1)
                offs.append(o)
            avail = avail.clone(); avail[ar, pick] = False               # 중복 방지
            q = q + key_k                                                # 이전 선택 반영
        off_out = torch.stack(offs, 1) if offs else None
        return torch.stack(picks, 1), off_out, lp, ent

    @torch.no_grad()
    def sample(self, p):
        picks, offs, _, _ = self._decode(p, cells=None)
        out = {"cells": picks}                                          # [B,K]
        if offs is not None: out["offset"] = offs                       # [B,K,2] (하이브리드)
        return out

    @torch.no_grad()
    def greedy(self, p):
        """평가용 결정적 선택: 매 스텝 argmax (중복 마스킹 유지). 하이브리드=오프셋 μ(결정적)."""
        ctx = p["ctx"]; cell_keys = p["cell_keys"]; avail = p["cell_valid"].clone()
        B = ctx.shape[0]; ar = torch.arange(B, device=ctx.device); q = ctx; picks = []; offs = []
        for k in range(self.K):
            pick = self._logits(q, cell_keys, avail).argmax(-1)
            picks.append(pick)
            key_k = cell_keys[ar, pick]
            if self.hybrid: offs.append(self._off_mu(ctx, key_k))       # 결정적=μ
            avail = avail.clone(); avail[ar, pick] = False
            q = q + key_k
        out = {"cells": torch.stack(picks, 1)}
        if offs: out["offset"] = torch.stack(offs, 1)
        return out

    def logp_entropy(self, p, act):
        off = act.get("offset") if isinstance(act, dict) else None
        _, _, lp, ent = self._decode(p, cells=act["cells"].long(), offsets=off)
        return lp, ent

    @torch.no_grad()
    def greedy_joint(self, p, N, P, cell_world=None, mask_radius=0.0):
        """★ 추론 Joint 디코딩(N월드 벡터화): 배들을 순차 greedy, 앞 배가 고른 셀(+반경)을 뒷 배서 제외.
        → WP 겹침·몰림 방지 (재학습 불필요). 월드는 병렬, 배(P)만 순차."""
        ctx = p["ctx"]; keys = p["cell_keys"]; valid = p["cell_valid"]
        B, C = valid.shape; d = self.d; dev = ctx.device
        picks = torch.zeros(B, self.K, dtype=torch.long, device=dev)
        ar = torch.arange(N, device=dev)
        cw = None; D2 = None
        if cell_world is not None and mask_radius > 0:
            cw = torch.as_tensor(np.asarray(cell_world), dtype=torch.float32, device=dev)   # [C,2]
            D2 = torch.cdist(cw, cw) < mask_radius                                          # [C,C] 근방 여부(사전계산)
        used = torch.zeros(N, C, dtype=torch.bool, device=dev)         # 월드별 이미 쓴 셀
        offs = torch.zeros(B, self.K, 2, device=dev) if self.hybrid else None
        for pp in range(P):
            bi = ar * P + pp                                          # [N] 배 pp의 배치 인덱스
            q = ctx[bi]; ctx0 = ctx[bi]                               # [N,d]
            kpp = keys[bi]                                            # [N,C,d]
            vpp = valid[bi]                                           # [N,C]
            avail = vpp & (~used)                                     # [N,C]
            empty = ~avail.any(1)                                     # 다 막힌 월드 → 원복(크래시 방지)
            avail[empty] = vpp[empty]
            for k in range(self.K):
                score = (self.q_proj(q).unsqueeze(1) * kpp).sum(-1) / (d ** 0.5)   # [N,C]
                score = score.masked_fill(~avail, float("-inf"))
                pick = score.argmax(1)                               # [N]
                picks[bi, k] = pick
                key_k = kpp[ar, pick]
                if self.hybrid:
                    offs[bi, k] = self._off_mu(ctx0, key_k)          # 결정적=μ
                avail = avail.clone(); avail[ar, pick] = False
                q = q + key_k                                        # 이전 선택 반영
                used[ar, pick] = True                               # cross-ship 잠금
                if D2 is not None:
                    used = used | D2[pick]                          # 근방셀도 잠금 [N,C]
        out = {"cells": picks}
        if offs is not None: out["offset"] = offs
        return out


def build_cell_actor(cfg: SimConfig = DEFAULT_CONFIG, d: int = 128) -> CellPointerActor:
    return CellPointerActor(cfg, d=d)


def save_cell_actor(actor, path, cfg, d=None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": actor.state_dict(), "config": cfg.to_dict(),
                "d": getattr(actor, "d", d or 128), "cell_actor": True}, path)


def load_cell_actor(path, cfg=None, device="cpu"):
    ck = torch.load(path, map_location=device)
    c = cfg or SimConfig.from_dict(ck["config"])
    actor = build_cell_actor(c, d=ck.get("d", 128)).to(device)
    actor.load_state_dict(ck["model"], strict=False)
    actor.eval()
    return actor, c
