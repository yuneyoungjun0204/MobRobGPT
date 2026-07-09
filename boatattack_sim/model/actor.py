"""
boatattack_sim/model/actor.py — Deep Sets 기반 소형 feedforward Actor

★ 임베디드 정합 설계: 어텐션 대신 **순열불변 풀링(Deep Sets)**, **GRU 없음**(near-Markov),
   **critic 없음**(GRPO). own MLP + DeepSet(enemy 클러스터) + DeepSet(ally) → trunk → 정책 헤드.

출력(잔차 보정 액션, spec.py 와 1:1):
  연속: cont_mean[Kw*2] = wp_delta(Kw,2),  tanh + 전역 log_std
  이산: net_go[2]   (env 가 지속 경로를 델타로 미세 보정 + net_go 시 구간 leg 를 그물로 도색)
파라미터 공유: 모든 아군이 같은 Actor (배치 B = N×P).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Categorical

from ..env.config import SimConfig, DEFAULT_CONFIG
from ..env import spec as SPEC


def mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class DeepSetEncoder(nn.Module):
    """순열불변 set 인코더: per-entity MLP → masked (mean ⊕ max) 풀링. 어텐션 대신.
    전부 패딩이면 0 반환(NaN-safe)."""
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.phi = mlp([in_dim, d, d])

    def forward(self, x, mask):                       # x[B,M,in], mask[B,M] True=pad
        h = self.phi(x)                               # [B,M,d]
        valid = (~mask).float().unsqueeze(-1)         # [B,M,1]
        cnt = valid.sum(1).clamp(min=1.0)             # [B,1]
        mean = (h * valid).sum(1) / cnt               # [B,d]
        h_max = h.masked_fill(mask.unsqueeze(-1), float("-inf"))
        mx = h_max.max(1).values                      # [B,d]
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)   # all-pad → 0
        return torch.cat([mean, mx], -1)              # [B,2d]


class _SABlock(nn.Module):
    """Pre-LN self-attention 블록 + FFN. ReZero 게이트(α init 0) → 초기엔 통과(near-identity)."""
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = mlp([d, 2 * d, d])
        self.a1 = nn.Parameter(torch.zeros(1))        # ReZero: 학습 초기 잔차 기여 0
        self.a2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, key_pad):                    # x[B,S,d], key_pad[B,S] True=무시
        h = self.ln1(x)
        att, _ = self.attn(h, h, h, key_padding_mask=key_pad, need_weights=False)
        x = x + self.a1 * att
        x = x + self.a2 * self.ff(self.ln2(x))
        return x


class SelfAttnEncoder(nn.Module):
    """[own, 아군들, 적클러스터들] 토큰 통합 self-attention → **own 토큰 출력**(관계반영 ego 임베딩).
    어텐션은 키 순서에 불변(순열불변), 패딩은 key_padding_mask 로 제외. own 토큰은 절대 마스크 안 함
    → 어떤 query 행도 키 전부가 마스크되지 않음(NaN 안전)."""
    def __init__(self, own_dim, Fe, Fa, d, heads=4, layers=2):
        super().__init__()
        self.own_in = nn.Linear(own_dim, d)
        self.ally_in = nn.Linear(Fa, d)
        self.enemy_in = nn.Linear(Fe, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))   # own / ally / enemy 타입 임베딩
        self.blocks = nn.ModuleList([_SABlock(d, heads) for _ in range(layers)])

    def forward(self, own, enemy, enemy_mask, ally, ally_mask):
        B = own.shape[0]
        ot = self.own_in(own).unsqueeze(1) + self.type_emb[0]          # [B,1,d]
        at = self.ally_in(ally) + self.type_emb[1]                     # [B,A,d]
        et = self.enemy_in(enemy) + self.type_emb[2]                   # [B,Kc,d]
        x = torch.cat([ot, at, et], 1)                                 # [B,1+A+Kc,d]
        own_pad = torch.zeros(B, 1, dtype=torch.bool, device=own.device)
        key_pad = torch.cat([own_pad, ally_mask, enemy_mask], 1)       # True=pad (own 항상 False)
        for blk in self.blocks:
            x = blk(x, key_pad)
        return x[:, 0]                                                 # own 토큰 [B,d]


class Actor(nn.Module):
    def __init__(self, cfg: SimConfig = DEFAULT_CONFIG, d: int = 128,
                 init_log_std: float = -1.0, init_netgo_bias: float = 1.0, dims=None):
        super().__init__()
        # dims override(로드 시 체크포인트 weight에서 추론한 차원) 우선; 없으면 현재 spec.
        dims = dims or SPEC.obs_dims(cfg)
        own_dim = dims["own"]
        Kc, Fe = dims["enemy"]
        A, Fa = dims["ally"]
        self.cfg = cfg
        self.Kw = cfg.transit_wp
        self.d = d                                    # 체크포인트 저장/복원용
        self.own_dim, self.Fe, self.Fa = own_dim, Fe, Fa   # 기대 입력 차원(obs 자동정합용)
        # heuristic_baseline 모드에선 그물 배치를 env(hmask)가 결정 → 정책 net_go 는 무효.
        #   → GRPO 목적(logπ·entropy)에서 제외해 dead-dim 표류·낭비 차단. 순수정책 모드만 학습.
        self.learn_netgo = not getattr(cfg, "heuristic_baseline", True)
        # ── 집합 인코딩: attn_backbone(관계추론 self-attention) / DeepSet(mean⊕max 풀링) ──
        #   둘 다 obs 토큰(own/enemy/ally)을 받아 [B,d] feature 로 → 아래 LSTM 시간메모리에 연결.
        self.attn_backbone = bool(getattr(cfg, "attn_backbone", False))
        if self.attn_backbone:                        # ★ 통합 self-attention(공간 관계·협조)
            self.encoder = SelfAttnEncoder(
                own_dim, Fe, Fa, d,
                heads=int(getattr(cfg, "attn_heads", 4)),
                layers=int(getattr(cfg, "attn_layers", 2)))
            self.trunk = mlp([d, d, d])               # own 토큰(d) → trunk
        else:                                         # 기존 Deep Sets(mean⊕max 풀링)
            self.own = mlp([own_dim, d, d])
            self.enemy = DeepSetEncoder(Fe, d)
            self.ally = DeepSetEncoder(Fa, d)
            self.trunk = mlp([d + 2 * d + 2 * d, d, d])
        # ── 백본: deepset(stateless) / lstm(결정 간 시간메모리) ──
        self.backbone = getattr(cfg, "backbone", "deepset")
        self.recurrent = (self.backbone == "lstm")
        if self.recurrent:
            self.lstm = nn.LSTMCell(d, d)             # trunk feature → 시간메모리(h,c)[B,d]
        # 연속: 잔차 wp_delta(Kw*2) 또는 부채꼴 fan(7). ★ structured_action 시 7파라미터.
        #   (그물 방향=다음 WP 방향이라 net_dir 액션 불요 — 경로 leg 가 곧 그물)
        self.n_cont = 7 if getattr(cfg, "structured_action", False) else self.Kw * 2
        self.cont_head = nn.Linear(d, self.n_cont)
        self.log_std = nn.Parameter(torch.full((self.n_cont,), float(init_log_std)))
        # ★ net_go = **레그별** 0/1 (Kw개 독립 Bernoulli): 어느 구간(leg)을 그물로 깔지.
        #   (transit leg 는 끄고, 차단 leg 만 켤 수 있어 simulator 휴리스틱과 동형.)
        self.netgo_head = nn.Linear(d, self.Kw)
        # 전개 쪽 초기 바이어스(모든 leg) → 초기 탐색이 전개를 충분히 시도(net 붕괴 완화).
        with torch.no_grad():
            self.netgo_head.bias.fill_(float(init_netgo_bias))
        # ★ 배정 학습(b: soft-bias): 배별 클러스터 선호 logits(K개) → _compute_assignment cost 를 조정.
        #   휴리스틱 배정을 base 로 두고 정책이 선호만 얹음(붕괴 회피). learn_assign=False 면 목적서 제외.
        self.learn_assign = bool(getattr(cfg, "learn_assign", False))
        self.assign_head = nn.Linear(d, int(getattr(cfg, "n_clusters", 4)))
        # ★ WP 순회방향(정/역) — 배별 단일 Bernoulli. 가법 head(옛 체크포인트 warm-start 호환).
        #   역방향이면 env 가 route/net leg 를 거꾸로 순회(첫 결정 확정·동결). learn_wpdir=False 면 목적 제외.
        self.learn_wpdir = bool(getattr(cfg, "learn_wpdir", False))
        self.wpdir_head = nn.Linear(d, 1)
        # ★ route rotate — 연속 스칼라(완성 route 전체 회전각). 가법 head(warm-start 호환).
        #   tanh→[-1,1]→env 가 ±rot_max_deg 로 스케일. learn_rot=False 면 목적 제외.
        self.learn_rot = bool(getattr(cfg, "learn_rot", False))
        self.rot_head = nn.Linear(d, 1)
        self.rot_log_std = nn.Parameter(torch.full((1,), float(init_log_std)))

    def init_hidden(self, B, device="cpu"):
        """순환 백본의 초기 hidden (h,c)[B,d]. deepset 은 None."""
        if not self.recurrent:
            return None
        z = torch.zeros(B, self.d, device=device)
        return (z, z.clone())

    @staticmethod
    def _fit(x, target):
        """obs 마지막 피처차원을 모델 기대치(target)에 맞게 자동 패딩(0)/절단.
        → 관측 스키마가 바뀌어도(approach 제거·net_radar 변경 등) 구/신 모델 모두 추론 가능."""
        cur = x.shape[-1]
        if cur == target:
            return x
        if cur > target:
            return x[..., :target]
        pad = torch.zeros(*x.shape[:-1], target - cur, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def forward(self, obs, hidden=None):              # obs: torch tensors [B,...]
        if self.attn_backbone:                        # 통합 self-attention → own 토큰
            z0 = self.encoder(self._fit(obs["own"], self.own_dim),
                              self._fit(obs["enemy"], self.Fe), obs["enemy_mask"],
                              self._fit(obs["ally"], self.Fa), obs["ally_mask"])
            z = self.trunk(z0)                        # [B,d]
        else:
            o = self.own(self._fit(obs["own"], self.own_dim))
            e = self.enemy(self._fit(obs["enemy"], self.Fe), obs["enemy_mask"])
            a = self.ally(self._fit(obs["ally"], self.Fa), obs["ally_mask"])
            z = self.trunk(torch.cat([o, e, a], -1))  # [B,d] 집합인코딩 feature
        if self.recurrent:                            # 2단: feature → LSTMCell 시간메모리
            if hidden is None:
                hidden = self.init_hidden(z.shape[0], z.device)
            h, c = self.lstm(z, hidden)
            feat = h; hidden_out = (h, c)
        else:
            feat = z; hidden_out = None
        p = {"cont_mean": torch.tanh(self.cont_head(feat)),   # [-1,1] (델타 스케일은 env)
             "netgo": self.netgo_head(feat),
             "assign": self.assign_head(feat),                # [B,K] 클러스터 선호 logits
             "wpdir": self.wpdir_head(feat),                  # [B,1] WP 순회방향(정/역) logit
             "rot_mean": torch.tanh(self.rot_head(feat))}     # [B,1] route 회전각 [-1,1]
        return p, hidden_out

    def _dists(self, p):
        std = self.log_std.exp().clamp(0.1, 2.0)        # 하한 0.1 → 연속 탐색 유지
        rstd = self.rot_log_std.exp().clamp(0.1, 2.0)   # rotate 탐색 std
        return (Normal(p["cont_mean"], std), Bernoulli(logits=p["netgo"]),
                Categorical(logits=p["assign"]),        # 연속 잔차 / 레그별 net / 클러스터 선호
                Bernoulli(logits=p["wpdir"]),           # WP 순회방향(정/역)
                Normal(p["rot_mean"], rstd))            # route 회전각

    @torch.no_grad()
    def sample(self, p):
        nrm, cg, ca, cw, cr = self._dists(p)
        return {"cont": nrm.sample().clamp(-1.0, 1.0), "netgo": cg.sample(),
                "assign": ca.sample(),                  # [B] 선호 클러스터 idx
                "wpdir": cw.sample(),                   # [B,1] 0=정방향 1=역방향
                "rot": cr.sample().clamp(-1.0, 1.0)}    # [B,1] route 회전각 [-1,1]

    def logp_entropy(self, p, act):
        """샘플(detach) 액션의 logπ·entropy (정책 mean/logits 에 gradient).
        learn_netgo / learn_assign / learn_wpdir / learn_rot=False 면 해당 항은 목적에서 제외(무효 dim)."""
        nrm, cg, ca, cw, cr = self._dists(p)
        lp = nrm.log_prob(act["cont"]).sum(-1)
        ent = nrm.entropy().sum(-1)
        if self.learn_netgo:
            lp = lp + cg.log_prob(act["netgo"]).sum(-1)
            ent = ent + cg.entropy().sum(-1)
        if self.learn_assign and "assign" in act:
            lp = lp + ca.log_prob(act["assign"])
            ent = ent + ca.entropy()
        if self.learn_wpdir and "wpdir" in act:
            lp = lp + cw.log_prob(act["wpdir"]).sum(-1)
            ent = ent + cw.entropy().sum(-1)
        if self.learn_rot and "rot" in act:
            lp = lp + cr.log_prob(act["rot"]).sum(-1)
            ent = ent + cr.entropy().sum(-1)
        return lp, ent


def build_actor(cfg: SimConfig = DEFAULT_CONFIG, d: int = 128,
                init_log_std: float = -1.0, init_netgo_bias: float = 1.0, dims=None) -> Actor:
    return Actor(cfg, d=d, init_log_std=init_log_std, init_netgo_bias=init_netgo_bias, dims=dims)


def load_actor(path: str, cfg: SimConfig = None, device="cpu"):
    ck = torch.load(path, map_location=device)
    c = cfg or SimConfig.from_dict(ck["config"])
    if cfg is None and "backbone" not in ck["config"]:   # 하위호환: 옛 체크포인트=deepset
        c.backbone = "deepset"
    sd = ck["model"]
    # ★ 백본 자동 감지: state_dict 키로 attention('encoder.') vs Deep Sets 구분 → 구조 일치 보장.
    #   (구버전 체크포인트는 config 에 attn_backbone 키가 없어 기본값과 어긋날 수 있음)
    c.attn_backbone = any(k.startswith("encoder.") for k in sd)
    # ★ 추론 시 obs 차원을 체크포인트 weight 에서 자동 인식 → 관측 스키마가 바뀌어도
    #   (approach 제거·net_radar 변경 등) 옛/새 체크포인트 모두 로드 가능.
    #   (입력 Linear 의 in_features = obs 피처차원.) 백본별로 키가 다르다.
    if c.attn_backbone:
        dims = {
            "own":   sd["encoder.own_in.weight"].shape[1],
            "enemy": [c.n_clusters, sd["encoder.enemy_in.weight"].shape[1]],
            "ally":  [max(c.max_pairs - 1, 1), sd["encoder.ally_in.weight"].shape[1]],
        }
    else:
        dims = {
            "own":   sd["own.0.weight"].shape[1],
            "enemy": [c.n_clusters, sd["enemy.phi.0.weight"].shape[1]],
            "ally":  [max(c.max_pairs - 1, 1), sd["ally.phi.0.weight"].shape[1]],
        }
    actor = build_actor(c, d=ck.get("d", 128), dims=dims).to(device)
    # ★ 액션 차원 하위호환: 옛 체크포인트는 n_follow(마지막 cont 차원) 이전 모델이라
    #   cont_head/log_std 가 Kw*2(=12) 로 현재 n_cont(=Kw*2+1=13)보다 1 작다.
    #   → 빠진 n_follow 차원을 'tanh≈+1 → n_follow=Kw(전부 따라감, 옛 동작)' 으로 패딩해 로드.
    #   (반대로 더 큰 옛 체크포인트는 절단.) 새 모델은 차원이 같아 그대로 통과.
    got = sd["cont_head.weight"].shape[0]
    exp = actor.n_cont
    if got != exp:
        din = sd["cont_head.weight"].shape[1]
        dt, dv = sd["cont_head.weight"].dtype, sd["cont_head.weight"].device
        if got < exp:                                   # 패딩(옛=n_follow 없음)
            n = exp - got
            sd["cont_head.weight"] = torch.cat(
                [sd["cont_head.weight"], torch.zeros(n, din, dtype=dt, device=dv)], 0)
            sd["cont_head.bias"] = torch.cat(            # bias 큰 + → tanh≈+1 → 전부 따라감
                [sd["cont_head.bias"], torch.full((n,), 3.0, dtype=dt, device=dv)], 0)
            sd["log_std"] = torch.cat(                  # 추가 차원은 사실상 결정적
                [sd["log_std"], torch.full((n,), -3.0, dtype=sd["log_std"].dtype, device=dv)], 0)
        else:                                           # 절단(미래·확장 차원 무시)
            sd["cont_head.weight"] = sd["cont_head.weight"][:exp]
            sd["cont_head.bias"] = sd["cont_head.bias"][:exp]
            sd["log_std"] = sd["log_std"][:exp]
    actor.load_state_dict(sd, strict=False)   # 신규 head(assign_head 등)는 random init 유지(warm-start 호환)
    actor.eval()
    return actor, c
