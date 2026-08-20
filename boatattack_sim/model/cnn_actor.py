"""
boatattack_sim/model/cnn_actor.py — CNN 점수맵 Actor (spatial softmax + 자기회귀)

멀티채널 래스터 관측 → U-Net lite(+CoordConv) → 점수맵 feature [B,d,H,W].
own 스칼라 → query q. score = <W_q q, feat>/√d 로 **맵 전체에 점수**를 그리고,
기하 마스크(환형∩요격점게이트∩Voronoi) 밖은 -inf → spatial softmax 로 픽셀 K개를 순차 선택한다.

자기회귀는 **CNN 재-forward 없이** feat 재사용 + query 갱신으로 한다(현 CellPointerActor._decode 와 동형).
→ CNN forward 는 결정당 1회. GRPO 의 K후보 재샘플/logp 재계산이 전부 이 feat 위에서 돈다.

인터페이스는 CellPointerActor 와 동일(sample/greedy/greedy_joint/logp_entropy) —
train/grpo_cell.py 의 _joint/_counterfactual/_adv/_mix_agent 를 그대로 쓰기 위한 하드 제약이다.

행동/분포:
  sample(p) → {"pix":[B,K] flat idx, "offset":[B,K,2]}  (offset = ±반픽셀, cnn_subpixel)
  logp_entropy(p, act) = Σ_k log Cat(pix_k) [+ Σ_k log N(off_k)] , Σ_k H
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..env.config import SimConfig, DEFAULT_CONFIG
from ..env import cnn_map as CM
from ..env.rasterizer import obs_channels, OWN_F
from .actor import mlp


def cnn_obs_to_torch(obs, device):
    """build_cnn_obs → 토치 텐서 dict (P 축 복제는 forward 에서 expand 로 무복사)."""
    def t(x, dt):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)
    return {"gmap": t(obs["gmap"], torch.float32),
            "smap": t(obs["smap"], torch.float32),
            "own": t(obs["own"], torch.float32),
            "valid": t(obs["valid"], torch.bool)}


def _cbr(cin, cout, stride=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout), nn.SiLU())


class UNetLite(nn.Module):
    """H→H/2→H/4→H/2→H 인코더/디코더 + skip (30² 면 30→15→8→15→30. 홀수 크기도
    F.interpolate(size=skip.shape) 로 복원해 안전). 점수맵 해상도 = 입력 해상도 유지.
    bottleneck GAP → FiLM 으로 전역 문맥(전체 전황)을 디코더에 주입."""

    def __init__(self, cin, w=32, d=32):
        super().__init__()
        self.stem = nn.Sequential(_cbr(cin, w), _cbr(w, w))          # [w,H,W]
        self.enc1 = nn.Sequential(_cbr(w, w, 2), _cbr(w, w))         # H/2
        self.enc2 = nn.Sequential(_cbr(w, 2 * w, 2), _cbr(2 * w, 2 * w))   # H/4
        self.bottle = nn.Sequential(_cbr(2 * w, 2 * w), _cbr(2 * w, 2 * w))
        self.film = nn.Linear(2 * w, 4 * w)                          # GAP → (γ, β)
        self.dec2 = nn.Sequential(_cbr(3 * w, w), _cbr(w, w))        # up + skip(enc1)
        self.dec1 = nn.Sequential(_cbr(2 * w, w), _cbr(w, w))        # up + skip(stem)
        self.head = nn.Conv2d(w, d, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        b = self.bottle(self.enc2(s1))
        g = self.film(b.mean((2, 3)))                                # [B,4w]
        gamma, beta = g.chunk(2, dim=1)
        b = b * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        u = F.interpolate(b, size=s1.shape[-2:], mode="nearest")
        u = self.dec2(torch.cat([u, s1], 1))
        u = F.interpolate(u, size=s0.shape[-2:], mode="nearest")
        u = self.dec1(torch.cat([u, s0], 1))
        return self.head(u)                                          # [B,d,H,W]


class CnnScoreActor(nn.Module):
    def __init__(self, cfg: SimConfig = DEFAULT_CONFIG, d: int = None, width: int = None):
        super().__init__()
        self.cfg = cfg
        self.n = CM.grid_n(cfg)
        self.d = int(d or getattr(cfg, "cnn_d", 32))
        w = int(width or getattr(cfg, "cnn_width", 32))
        self.K = int(getattr(cfg, "cnn_nets", 2))          # 선택 픽셀 수
        self.Kw = cfg.transit_wp                            # 트레이너 호환
        self.recurrent = bool(getattr(cfg, "cnn_recurrent", False))
        self.subpixel = bool(getattr(cfg, "cnn_subpixel", True))
        self.dup_r_px = float(getattr(cfg, "cnn_dup_r", 0.0)) / CM.px_size(cfg)
        # ★ 도달성 상한(px). 0=끔. dup_r 이 '너무 가까움'을 막듯 이건 '너무 멂'을 막는다 —
        #   그물은 net_max_len 에서 끊기므로 그보다 먼 짝은 벽이 목표에 닿지 못한다.
        self.reach_px = (float(cfg.net_max_len) * float(getattr(cfg, "cnn_reach_slack", 1.0))
                         / CM.px_size(cfg)) if bool(getattr(cfg, "cnn_reach_mask", False)) else 0.0
        Cg, Cs, Cc = obs_channels(cfg)
        self.Cg, self.Cs, self.Cc = Cg, Cs, Cc
        self.net = UNetLite(Cg + Cs + Cc, w=w, d=self.d)
        self.own_mlp = mlp([OWN_F, 128, self.d])
        self.q_proj = nn.Linear(self.d, self.d)
        self.log_std = nn.Parameter(torch.zeros(1))         # 로깅 호환용 더미(손실 무관)
        if self.recurrent:
            self.lstm = nn.LSTM(self.d, self.d, batch_first=True)
        if self.subpixel:
            self.off_mu = mlp([2 * self.d, self.d, 2])
            self.off_logstd = nn.Parameter(torch.full((2,), -1.2))   # std≈0.30 (초기 미세탐색)
            nn.init.zeros_(self.off_mu[-1].weight); nn.init.zeros_(self.off_mu[-1].bias)
        if Cc:
            # config 에서 완전히 파생되는 정적 채널 → 체크포인트에 넣지 않는다(persistent=False).
            # 저장하면 coord 구성이 바뀔 때 stem 과 별개로 shape 충돌을 일으킨다.
            self.register_buffer("coord", torch.as_tensor(CM.coord_channels(cfg)),
                                 persistent=False)
        self.register_buffer("_ar", torch.arange(self.n).float(), persistent=False)

    # ── 인코딩 ────────────────────────────────────────────────────────
    def init_hidden(self, B, device="cpu"):
        if not self.recurrent:
            return None
        z = torch.zeros(1, B, self.d, device=device)
        return (z, z.clone())

    def forward(self, obs, hidden=None):
        g = obs["gmap"]; s = obs["smap"]
        N, P = s.shape[:2]; B = N * P
        H, W = g.shape[-2:]
        x = [g.unsqueeze(1).expand(N, P, self.Cg, H, W).reshape(B, self.Cg, H, W),
             s.reshape(B, self.Cs, H, W)]
        if self.Cc:
            x.append(self.coord.unsqueeze(0).expand(B, self.Cc, H, W))
        feat = self.net(torch.cat(x, 1))                            # [B,d,H,W]
        q = self.own_mlp(obs["own"].reshape(B, -1))                 # [B,d]
        h_out = None
        if self.recurrent:
            if hidden is None:
                hidden = self.init_hidden(B, q.device)
            out, h_out = self.lstm(q.unsqueeze(1), hidden)
            q = out[:, 0]
        return {"feat": feat, "q": q, "valid": obs["valid"].reshape(B, H, W)}, h_out

    # ── 점수맵 / 마스크 ───────────────────────────────────────────────
    def _score(self, q, feat):
        """q[B,d], feat[B,d,H,W] → 점수맵 [B,H,W]. einsum 으로 [B,d,H,W] 중간텐서 회피."""
        return torch.einsum("bd,bdhw->bhw", self.q_proj(q), feat) / (self.d ** 0.5)

    @staticmethod
    def _fallback(avail):
        """전 픽셀 무효인 행 → 전부 유효로 폴백 (Categorical(all -inf) 크래시 방지)."""
        empty = ~avail.flatten(1).any(1)
        return avail | empty[:, None, None]

    def _logits(self, q, feat, avail):
        sc = self._score(q, feat)
        return sc.masked_fill(~self._fallback(avail), float("-inf")).flatten(1)   # [B,H*W]

    def _exclude(self, avail, pick, radius_px=None, reach=True):
        """선택 픽셀 주변 원판을 마스크에서 제거 (중복/근접 방지) + 도달성 상한.

        reach=True 이고 self.reach_px>0 이면 **너무 먼** 픽셀도 함께 제거한다
        (그물벽이 net_max_len 에서 끊겨 목표에 닿지 못하는 짝을 행동공간에서 배제).
        reach=False 는 cross-ship 잠금처럼 '남의 자리 제거' 용도 — 그건 도달성과 무관하다."""
        r = self.dup_r_px if radius_px is None else radius_px
        rk = self.reach_px if reach else 0.0
        B, H, W = avail.shape
        bi = (pick // W).float(); bj = (pick % W).float()
        if r <= 0 and rk <= 0:
            out = avail.clone()
            out[torch.arange(B, device=avail.device), pick // W, pick % W] = False
            return out
        ii = self._ar[:H]; jj = self._ar[:W]
        d2 = ((ii.view(1, H, 1) - bi.view(B, 1, 1)) ** 2
              + (jj.view(1, 1, W) - bj.view(B, 1, 1)) ** 2)
        out = avail & (d2 > r * r) if r > 0 else avail.clone()
        if r <= 0:
            out[torch.arange(B, device=avail.device), pick // W, pick % W] = False
        if rk > 0:
            near = out & (d2 <= rk * rk)
            # 상한을 걸어 전부 막히면(코리도가 좁은 극단) 원복 — 크래시보다 낫다
            keep = near.flatten(1).any(1)
            out = torch.where(keep[:, None, None], near, out)
        return out

    @staticmethod
    def _gather_feat(feat, pick):
        """feat[B,d,H,W] 에서 flat 픽셀 pick[B] 의 특징 [B,d]."""
        B, d = feat.shape[:2]
        return feat.flatten(2).gather(2, pick.view(B, 1, 1).expand(B, d, 1)).squeeze(2)

    # ── 자기회귀 디코딩 ───────────────────────────────────────────────
    def _decode(self, p, pix=None, offsets=None):
        """순차 K선택. pix=None → 샘플, 아니면 주어진 pix 의 logp/entropy 재생.
        반환 (picks[B,K], offs[B,K,2]|None, logp[B], entropy[B])."""
        feat = p["feat"]; q = p["q"]
        avail = p["valid"].clone()
        B = q.shape[0]; dev = q.device
        lp = torch.zeros(B, device=dev); ent = torch.zeros(B, device=dev)
        picks = []; offs = []
        use_off = self.subpixel and (pix is None or offsets is not None)
        q0 = q
        for k in range(self.K):
            dist = Categorical(logits=self._logits(q, feat, avail))
            pick = dist.sample() if pix is None else pix[:, k].long()
            lp = lp + dist.log_prob(pick)
            ent = ent + dist.entropy()
            picks.append(pick)
            fk = self._gather_feat(feat, pick)                       # [B,d]
            if use_off:
                mu = self.off_mu(torch.cat([q0, fk], -1))
                od = Normal(mu, self.off_logstd.exp())
                o = od.sample() if offsets is None else offsets[:, k]
                lp = lp + od.log_prob(o).sum(-1)
                ent = ent + od.entropy().sum(-1)
                offs.append(o)
            avail = self._exclude(avail, pick)
            q = q + fk                                               # 이전 선택 반영
        return (torch.stack(picks, 1), (torch.stack(offs, 1) if offs else None), lp, ent)

    @torch.no_grad()
    def sample(self, p):
        picks, offs, _, _ = self._decode(p, pix=None)
        out = {"pix": picks}
        if offs is not None:
            out["offset"] = offs
        return out

    @torch.no_grad()
    def greedy(self, p):
        """평가용 결정적 선택: 매 스텝 argmax. 서브픽셀은 μ(결정적)."""
        feat = p["feat"]; q = p["q"]; q0 = q
        avail = p["valid"].clone()
        picks = []; offs = []
        for k in range(self.K):
            pick = self._logits(q, feat, avail).argmax(-1)
            picks.append(pick)
            fk = self._gather_feat(feat, pick)
            if self.subpixel:
                offs.append(self.off_mu(torch.cat([q0, fk], -1)))
            avail = self._exclude(avail, pick)
            q = q + fk
        out = {"pix": torch.stack(picks, 1)}
        if offs:
            out["offset"] = torch.stack(offs, 1)
        return out

    @torch.no_grad()
    def greedy_joint(self, p, N, P, mask_radius=None):
        """★ 추론 Joint 디코딩: 배들을 순차 greedy, 앞 배가 고른 픽셀(+반경)을 뒷 배서 제외.
        → WP 겹침·몰림 방지 (재학습 불필요). 월드는 병렬, 배(P)만 순차.

        ★ 겹침 레짐 정합성: `cnn_gate_disjoint` 가 꺼져 있으면(=학습이 겹침을 허용했으면)
          cross-ship 잠금도 **자동으로 끈다**. 안 그러면 학습에서 허용한 다층 벽(종심방어)을
          추론에서 도로 떼어놓아 레짐 실험이 무효가 된다. mask_radius 를 명시하면 그 값이 우선."""
        feat = p["feat"]; q_all = p["q"]; valid = p["valid"]
        B, H, W = valid.shape
        dev = q_all.device
        if mask_radius is None:
            r_m = (float(getattr(self.cfg, "cnn_joint_mask_r", 0.0))
                   if getattr(self.cfg, "cnn_gate_disjoint", True) else 0.0)
        else:
            r_m = float(mask_radius)
        r = r_m / CM.px_size(self.cfg)
        picks = torch.zeros(B, self.K, dtype=torch.long, device=dev)
        offs = torch.zeros(B, self.K, 2, device=dev) if self.subpixel else None
        used = torch.zeros(N, H, W, dtype=torch.bool, device=dev)     # 월드별 이미 쓴 픽셀
        ar = torch.arange(N, device=dev)
        for pp in range(P):
            bi = ar * P + pp
            fp = feat[bi]; q = q_all[bi]; q0 = q
            av = valid[bi] & (~used)
            empty = ~av.flatten(1).any(1)
            av[empty] = valid[bi][empty]                              # 다 막힌 월드 원복
            for k in range(self.K):
                pick = self._logits(q, fp, av).argmax(-1)
                picks[bi, k] = pick
                fk = self._gather_feat(fp, pick)
                if self.subpixel:
                    offs[bi, k] = self.off_mu(torch.cat([q0, fk], -1))
                av = self._exclude(av, pick)
                used = used | (~self._exclude(torch.ones_like(av), pick, r, reach=False))
                q = q + fk
        out = {"pix": picks}
        if offs is not None:
            out["offset"] = offs
        return out

    def logp_entropy(self, p, act):
        off = act.get("offset") if isinstance(act, dict) else None
        _, _, lp, ent = self._decode(p, pix=act["pix"].long(), offsets=off)
        return lp, ent

    @torch.no_grad()
    def score_stages(self, p, picks=None):
        """★ 시각화용: 자기회귀 단계별 점수맵(확률)과 선택을 그대로 뽑아낸다.
        반환 {"prob": [K,B,H,W] 마스크된 softmax, "pix": [B,K], "offset": [B,K,2]|None}.
        picks 를 주면 그 선택으로 조건화(정책이 실제 고른 경로의 맵을 재현)."""
        feat = p["feat"]; q = p["q"]; q0 = q
        avail = p["valid"].clone()
        B, H, W = avail.shape
        probs = []; out_pix = []; offs = []
        for k in range(self.K):
            logits = self._logits(q, feat, avail)                     # [B,H*W]
            probs.append(torch.softmax(logits, dim=1).view(B, H, W))
            pick = logits.argmax(-1) if picks is None else picks[:, k].long()
            out_pix.append(pick)
            fk = self._gather_feat(feat, pick)
            if self.subpixel:
                offs.append(self.off_mu(torch.cat([q0, fk], -1)))
            avail = self._exclude(avail, pick)
            q = q + fk
        return {"prob": torch.stack(probs, 0),
                "pix": torch.stack(out_pix, 1),
                "offset": (torch.stack(offs, 1) if offs else None)}

    # ── BC (휴리스틱 픽셀 모방) ───────────────────────────────────────
    def bc_loss(self, p, tgt_pix, alive=None, sigma_px=None):
        """휴리스틱 픽셀 타깃의 교차엔트로피 (teacher forcing).
        ★ dense 100² 맵에서 원핫 CE 는 그래디언트가 지나치게 희소하다 → 타깃 주변
          σ=cnn_bc_sigma(px) 가우시안으로 라벨 스무딩해 초기 학습이 실제로 붙게 한다."""
        feat = p["feat"]; q = p["q"]
        avail = p["valid"].clone()
        B, H, W = avail.shape
        sig = float(self.cfg.cnn_bc_sigma if sigma_px is None else sigma_px)
        ii = self._ar[:H]; jj = self._ar[:W]
        loss = torch.zeros(B, device=q.device)
        for k in range(self.K):
            logits = self._logits(q, feat, avail)                     # [B,H*W]
            tgt = tgt_pix[:, k].long()
            if sig > 0:
                ti = (tgt // W).float(); tj = (tgt % W).float()
                d2 = ((ii.view(1, H, 1) - ti.view(B, 1, 1)) ** 2
                      + (jj.view(1, 1, W) - tj.view(B, 1, 1)) ** 2)
                wgt = torch.exp(-d2 / (2.0 * sig * sig)) * self._fallback(avail).float()
                wgt = wgt.flatten(1)
                wgt = wgt / wgt.sum(1, keepdim=True).clamp(min=1e-8)
                # 마스크 픽셀은 log_softmax=-inf, wgt=0 → 0·(-inf)=nan 이 되므로 0 으로 치환
                lsm = F.log_softmax(logits, dim=1)
                lsm = torch.where(wgt > 0, lsm, torch.zeros_like(lsm))
                loss = loss - (wgt * lsm).sum(1)
            else:
                loss = loss - F.log_softmax(logits, dim=1).gather(1, tgt[:, None])[:, 0]
            fk = self._gather_feat(feat, tgt)
            avail = self._exclude(avail, tgt)
            q = q + fk
        if alive is not None:
            return (loss * alive).sum() / alive.sum().clamp(min=1.0)
        return loss.mean()


def build_cnn_actor(cfg: SimConfig = DEFAULT_CONFIG, d: int = None, width: int = None):
    return CnnScoreActor(cfg, d=d, width=width)


def save_cnn_actor(actor, path, cfg, d=None):
    import os
    from ..env.rasterizer import global_channel_names, ship_channel_names
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # ★ 채널 '이름' 을 함께 남긴다 — 개수만으로는 스키마 변경을 못 잡는다.
    #   (구 14ch = 8 global + 3 ship + x,y,r  vs  신 14ch = 9 global + 3 ship + x,y
    #    → 개수 검사 통과, 의미는 전부 어긋남. 실제로 한 번 당했다.)
    torch.save({"model": actor.state_dict(), "config": cfg.to_dict(),
                "d": getattr(actor, "d", d or 32),
                "width": getattr(actor.net.head, "in_channels", 32),
                "obs_schema": list(global_channel_names(cfg)) + list(ship_channel_names(cfg)),
                "cnn_actor": True}, path)


def _ckpt_in_channels(sd):
    """체크포인트 stem 첫 conv 의 in_channels. 없으면 None."""
    w = sd.get("net.stem.0.0.weight")
    return int(w.shape[1]) if w is not None and w.dim() == 4 else None


def _drop_coord_r_stem(sd, want):
    """구 체크포인트(coord_r 포함)의 stem 첫 conv 에서 **마지막 입력채널**(=coord_r)을 잘라낸다.
    채널 순서가 [전역 … 배별 … coord_x, coord_y, coord_r] 라 마지막 슬라이스가 정확히 r 이다.
    → 14ch 챔피언에서 13ch 학습으로 warm-start 가능 (나머지 가중치는 그대로 재사용)."""
    k = "net.stem.0.0.weight"
    sd = dict(sd)
    sd[k] = sd[k][:, :want].clone()
    sd.pop("coord", None)                 # 구 체크포인트의 정적 coord 버퍼(3ch) 잔재 제거
    return sd


def _pad_coord_r_stem(sd, want):
    """반대 방향: coord_r 없이 학습된 체크포인트를 coord_r 켜진 설정으로 warm-start.
    stem 첫 conv 뒤에 **0 가중치 입력채널**을 하나 붙인다 — 0 이라 초기 함수가 정확히 보존되고
    (새 채널 기여 0), 학습이 그 채널을 쓸지 스스로 정한다.
    coord_r 을 껐다 되돌린 창(2026-08-13~14) 에 학습된 체크포인트를 살리기 위한 경로다."""
    k = "net.stem.0.0.weight"
    sd = dict(sd)
    w = sd[k]                                              # [out, got, 3, 3]
    pad = torch.zeros(w.shape[0], want - w.shape[1], *w.shape[2:],
                      dtype=w.dtype, device=w.device)
    sd[k] = torch.cat([w, pad], dim=1).contiguous()
    sd.pop("coord", None)                 # 정적 coord 버퍼는 config 에서 재생성(persistent=False)
    return sd


def _coord_block_only(c, got):
    """채널 수 차이가 **coord 블록에서만** 났는지 확인. 전역/배별 채널이 어긋난 상태에서
    끝에 슬라이스/패딩을 하면 채널 의미가 통째로 밀린다 — 그 사고를 막는 가드다."""
    from ..env.rasterizer import global_channel_names, ship_channel_names
    n_fixed = len(global_channel_names(c)) + len(ship_channel_names(c))
    return got - n_fixed in (2, 3)        # 체크포인트의 coord 블록이 x,y 이거나 x,y,r


def load_cnn_actor(path, cfg=None, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    explicit = cfg is not None
    c = cfg or SimConfig.from_dict(ck["config"])
    sd = ck["model"]
    got = _ckpt_in_channels(sd)                       # 체크포인트가 학습된 입력 채널 수
    if got is not None:
        want = sum(obs_channels(c))                   # 현 config 가 만드는 입력 채널 수
        if got != want:
            if not explicit and getattr(c, "cnn_coord", True) and got == want + 1:
                # cfg 를 체크포인트에서 복원하는 경우 — cnn_coord_r 키가 없던 구세대다.
                # 그 모델을 원형 그대로 되살리는 게 맞다.
                c.cnn_coord_r = True
            elif not explicit and getattr(c, "cnn_coord", True) and got == want - 1:
                # 반대 방향: coord_r 을 껐던 창(2026-08-13~14)에 학습된 체크포인트.
                # 저장된 config 에 cnn_coord_r=False 가 있으면 여기 안 온다(from_dict 가 복원).
                # 키가 없는데 1채널 모자란다면 그 세대다 — 원형(coord_r 없음)대로 되살린다.
                c.cnn_coord_r = False
            elif explicit and got == want + 1 and _coord_block_only(c, got):
                # 호출자가 config 를 명시했다(예: grpo_cnn --init). 호출자 의도가 우선 —
                # config 를 몰래 되돌리지 말고, 남는 coord_r 입력채널만 떼어 warm-start 한다.
                sd = _drop_coord_r_stem(sd, want)
                print(f"[load_cnn_actor] {got}ch 체크포인트 -> {want}ch 설정: "
                      f"stem 의 coord_r 입력채널을 잘라 warm-start")
            elif explicit and got == want - 1 and _coord_block_only(c, got):
                # coord_r 없이 학습된 체크포인트 → coord_r 켜진 설정으로 warm-start.
                # 0 채널을 붙이므로 초기 함수는 정확히 보존된다.
                sd = _pad_coord_r_stem(sd, want)
                print(f"[load_cnn_actor] {got}ch 체크포인트 -> {want}ch 설정: "
                      f"stem 에 coord_r 입력채널을 0 으로 붙여 warm-start")
            else:
                raise RuntimeError(
                    f"체크포인트 입력채널 {got} != 설정 {want} — 관측 스키마가 다르다. "
                    f"(cnn_coord={getattr(c, 'cnn_coord', True)}, "
                    f"cnn_coord_r={getattr(c, 'cnn_coord_r', False)})")
    # ★ 개수가 맞아도 **구성**이 다를 수 있다(구 14ch coord_r 판 vs 신 14ch land 판).
    #   저장된 스키마가 있으면 이름으로 대조한다. 없으면(구 체크포인트) 경고만 — _LEGACY_ABSENT
    #   가 land 를 OFF 로 되돌리므로 정상 경로에선 어긋나지 않는다.
    from ..env.rasterizer import global_channel_names, ship_channel_names
    saved = ck.get("obs_schema")
    if saved is not None:
        now = list(global_channel_names(c)) + list(ship_channel_names(c))
        if list(saved) != now:
            raise RuntimeError(
                f"관측 채널 구성이 다르다 — 채널 수는 같아도 의미가 어긋난다.\n"
                f"  체크포인트: {list(saved)}\n  현재 설정  : {now}")
    actor = build_cnn_actor(c, d=ck.get("d", 32), width=ck.get("width", 32)).to(device)
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    return actor, c
