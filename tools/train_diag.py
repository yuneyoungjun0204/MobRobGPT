"""
tools/train_diag.py — 채택 런(run_20260819-203248)과 같은 설정으로 GRPO 학습을 다시 돌리되,
**수렴 진단 지표를 매 업데이트마다** 기록한다.

채택 런의 metrics.csv 는 20 업데이트마다 R/best/valid/loss/cap/br 여섯 개만 남겼다.
논문 학습 곡선에 흔히 쓰는 수렴 근거(정책 엔트로피, 업데이트 간 근사 KL, 그래디언트 노름,
파라미터 이동 거리, 휴리스틱 대비 승률, 조밀한 greedy 평가)는 그 파일로는 만들 수 없다.
그래서 학습 루프를 그대로 복제하고 진단만 얹는다. 학습 코드(One-Way_Towing/CNN)는 건드리지
않는다 --- 다른 worktree 의 파일이라 수정하지 않고 sys.path 로 가져다 쓴다.

기록:
  <out>/metrics.csv  매 업데이트 --- lr, R, best, R_heur, win_heur, gain_heur, valid,
                     loss, pg, ent_term, ent_total, ent_pix1, top1, kl, grad_norm,
                     dtheta, dtheta_rel, cap, br
  <out>/evals.csv    --eval-every 마다 greedy 평가(3 시드 x 120 결정) + 기준선
  <out>/bc.csv       BC 워밍업 손실

사용 (저장소 루트, 전역 python 3.10 --- .venv 에는 torch 가 없다):
  PYTHONIOENCODING=utf-8 python tools/train_diag.py --seed 0 --out results/train_diag/seed0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace

CNN_ROOT = os.environ.get("CNN_ROOT", r"C:\Users\ANSL\orca\workspaces\One-Way_Towing\CNN")
sys.path.insert(0, CNN_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from boatattack_sim.env.config import GRPOCfg, RewardCfg  # noqa: E402
from boatattack_sim.env.defense_env import DefenseVecEnv  # noqa: E402
from boatattack_sim.model.cnn_actor import (build_cnn_actor, cnn_obs_to_torch,  # noqa: E402
                                            save_cnn_actor)
from boatattack_sim.train import grpo_cnn as G  # noqa: E402
from boatattack_sim.train.grpo import _adv  # noqa: E402


# ── 채택 런의 CLI 기본값 (grpo_cnn.py __main__ 의 argparse 기본값을 그대로 옮김) ──
def adopted_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        grid_n=50, nets_k=2, no_subpixel=False, gate_r=3000.0, gate_angle=60.0,
        disjoint=False, no_disjoint=False, no_annulus=False, width=32, d_map=32,
        no_coord=False, no_coord_r=False, bc_sigma=0.8, marker_r=0, min_valid=150,
        reach_mask=False, reach_slack=1.0, pair_bias=False, off_polar=False, nets=1,
        no_land=False, land_source="auto", land_sites=None, land_threshold=0.30,
        no_land_avoid=False, land_gain=1.6, domain_rand=False,
        # 채택 런 로그 헤더: worlds=24 k=8 eval_period=180 updates=800 adv=heur_rel bc=300
        # dmin 근접 페널티 ON — 모선 w=3.0 · 아군 w=3.0
        enemy="rotate", worlds=24, k=8, updates=800, eval_period=180, adv_mode="heur_rel",
        ma_mode="joint", bc_warmup=300, w_mother_dmin=3.0, w_ally_dmin=3.0,
        eval_seeds=3, eval_worlds=None, baseline_decisions=120, seed=seed,
    )


def _write_row(path: str, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _flat_params(actor) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in actor.parameters()])


@torch.no_grad()
def _pix_entropy_top1(actor, p):
    """1단계 픽셀 분포의 엔트로피(nat)와 top-1 확률 --- 정책이 얼마나 첨예해졌는가."""
    st = actor.score_stages(p)
    pr = st["prob"][0].flatten(1)                          # [B,H*W]
    ent = -(pr * torch.log(pr.clamp_min(1e-12))).sum(1)    # 마스크 밖은 p=0 → 기여 0
    return float(ent.mean()), float(pr.max(1).values.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=0, help="torch/numpy/env 시드 (채택 런은 env 0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--updates", type=int, default=None, help="스모크용 축소")
    ap.add_argument("--bc", type=int, default=None, help="스모크용 BC 스텝 축소")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    args = adopted_args(a.seed)
    if a.updates:
        args.updates = a.updates
    if a.bc is not None:
        args.bc_warmup = a.bc

    cfg = G.make_cfg(args)
    gcfg = GRPOCfg(num_worlds=args.worlds, k_samples=args.k, eval_period=args.eval_period,
                   updates=args.updates, lr=1e-4, ma_mode=args.ma_mode, curriculum=False,
                   heuristic_candidate=True, adv_mode=args.adv_mode)
    gcfg.heuristic_candidate = True
    rcfg = RewardCfg(w_mother_dmin=args.w_mother_dmin, w_ally_dmin=args.w_ally_dmin)
    cfg.dmin_track = True
    device = "cpu"

    env = DefenseVecEnv(num_worlds=gcfg.num_worlds, cfg=cfg, rcfg=rcfg,
                        enemy_mode=args.enemy, seed=args.seed)
    actor = build_cnn_actor(cfg)
    N, P = env.N, env.P
    n_par = sum(p.numel() for p in actor.parameters())
    print(f"[diag] seed={a.seed} params={n_par:,} worlds={N} P={P} k={gcfg.k_samples} "
          f"updates={gcfg.updates} ma={args.ma_mode} out={a.out}", flush=True)

    # 기준선 + 평가 환경 (grpo_cnn.main 과 같은 프로토콜)
    ev_seeds = G.eval_seed_list(args.eval_seeds)
    n_ev = int(args.eval_worlds or min(32, N))
    bl_env = DefenseVecEnv(num_worlds=n_ev, cfg=cfg, enemy_mode=args.enemy, seed=99991)
    baseline = G.eval_heuristic_pix(bl_env, decisions=args.baseline_decisions,
                                    seeds=ev_seeds)["cap_rate"]
    eval_env = DefenseVecEnv(num_worlds=n_ev, cfg=cfg, rcfg=rcfg, enemy_mode=args.enemy,
                             seed=99991)
    print(f"[diag] baseline heur→pix cap_rate={baseline:.4f}", flush=True)

    # ── BC 워밍업 (손실을 파일로) ──
    opt_bc = torch.optim.Adam(actor.parameters(), lr=3e-4)
    for it in range(args.bc_warmup):
        obs = env.build_cnn_obs()
        tgt = torch.as_tensor(env.heuristic_pix().reshape(N * P, -1), dtype=torch.long)
        p, _ = actor(cnn_obs_to_torch(obs, device))
        alive = torch.as_tensor(env.a_alive.reshape(N * P), dtype=torch.float32)
        loss = actor.bc_loss(p, tgt, alive=alive)
        opt_bc.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_bc.step()
        env.step({"pix": env.heuristic_pix()})
        _write_row(os.path.join(a.out, "bc.csv"), dict(it=it, loss=float(loss.detach())))
        if it % 50 == 0 or it == args.bc_warmup - 1:
            print(f"[BC] it{it:4d} loss={float(loss):.3f}", flush=True)
    env.reset(seed=args.seed)
    theta0 = _flat_params(actor).clone()
    theta0_norm = float(theta0.norm())

    def do_eval(upd):
        ev = G.eval_cnn(eval_env, actor, decisions=120, device=device, seeds=ev_seeds)
        _write_row(os.path.join(a.out, "evals.csv"),
                   dict(upd=upd, eval_cap=ev["cap_rate"], cap=ev["cap"], br=ev["br"],
                        baseline=round(baseline, 6), gap=round(ev["cap_rate"] - baseline, 6)))
        print(f"  [eval] @upd{upd} cap={ev['cap_rate']:.4f} (baseline {baseline:.3f} "
              f"{ev['cap_rate'] - baseline:+.4f})", flush=True)
        return ev["cap_rate"]

    # ── GRPO (grpo_cnn.train 과 동형 + 진단) ──
    opt = torch.optim.Adam(actor.parameters(), lr=gcfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(gcfg.updates, 1),
                                                       eta_min=gcfg.lr * 0.05)
    best_cap = do_eval(0)
    save_cnn_actor(actor, os.path.join(a.out, "best.pt"), cfg, d=actor.d)
    t0 = time.time()
    for upd in range(gcfg.updates):
        ent_c = gcfg.ent_coef * (1.0 - (1.0 - gcfg.ent_coef_end_ratio) * upd / max(gcfg.updates, 1))
        lr_now = opt.param_groups[0]["lr"]
        obs_t = cnn_obs_to_torch(env.build_cnn_obs(), device)
        p, _ = actor(obs_t)

        # _joint 를 그대로 쓰되 후보·보상 행렬을 진단에 쓰기 위해 여기서 다시 뽑는다
        # (grpo_cnn._joint 는 R 을 돌려주지 않는다). 논리는 _joint 와 동일.
        Kk = gcfg.k_samples
        cands = [actor.sample(p) for _ in range(Kk)]
        cands[0] = G._heur_cnn_cand(env, device)
        ff = np.ones((N, P), bool)
        R = np.stack([env.rollout_eval(G.cnn_act_to_env(c, N, P), period=gcfg.eval_period,
                                       force_fresh=ff) for c in cands], axis=1)      # [N,K]
        A, valid = _adv(R, Kk, gcfg.adv_mode)
        At = torch.as_tensor(A, dtype=torch.float32)
        w = torch.as_tensor((valid & (R.std(1) > gcfg.var_eps)).astype(np.float32)).view(N, 1)
        pg = torch.zeros(()); ent_acc = torch.zeros(())
        lp_old = []
        for k in range(Kk):
            lp, ent = actor.logp_entropy(p, cands[k])
            pg = pg + (-(At[:, k:k + 1].detach() * lp.view(N, P)) * w).sum()
            ent_acc = ent_acc + ent.sum()
            lp_old.append(lp.detach())
        denom = float(max(w.sum().item() / max(P, 1), 1)) * Kk * P
        pg_term = pg / max(denom, 1.0)
        ent_term = -ent_c * ent_acc / (N * P)
        loss = pg_term + ent_term
        ent_pix1, top1 = _pix_entropy_top1(actor, p)

        opt.zero_grad(); loss.backward()
        gnorm = float(nn.utils.clip_grad_norm_(actor.parameters(), gcfg.grad_clip))
        opt.step(); sched.step()

        # 근사 KL(old‖new): 같은 관측·같은 표본 후보로 새 정책 logp 재계산 (k3 추정량)
        with torch.no_grad():
            p_new, _ = actor(obs_t)
            kls = []
            for k in range(1, Kk):                      # 후보0(휴리스틱)은 표본이 아니다
                lp_new, _ = actor.logp_entropy(p_new, cands[k])
                lr_ = lp_new - lp_old[k]
                kls.append(((lr_.exp() - 1.0) - lr_).mean())
            kl = float(torch.stack(kls).mean())
            th = _flat_params(actor)
            dtheta = float((th - theta0).norm())
            a0 = actor.sample(p_new)
        _, _, _, _, info = env.step(G.cnn_act_to_env(a0, N, P))
        cap = float(info["captures"].sum()); br = float(info["breaches"].sum())

        r_heur = float(R[:, 0].mean())
        row = dict(upd=upd, lr=lr_now, R=float(R.mean()), best=float(R.max(1).mean()),
                   R_heur=r_heur, win_heur=float((R[:, 1:] > R[:, :1]).mean()),
                   gain_heur=float((R[:, 1:] - R[:, :1]).mean()),
                   valid=float((R.std(1) > gcfg.var_eps).mean()),
                   loss=float(loss.detach()), pg=float(pg_term.detach()),
                   ent_term=float(ent_term.detach()),
                   ent_total=float(ent_acc.detach()) / (N * P * Kk),
                   ent_pix1=ent_pix1, top1=top1, kl=kl, grad_norm=gnorm,
                   dtheta=dtheta, dtheta_rel=dtheta / theta0_norm, cap=cap, br=br)
        _write_row(os.path.join(a.out, "metrics.csv"), row)
        if upd % 20 == 0 or upd == gcfg.updates - 1:
            el = (time.time() - t0) / 60
            print(f"upd{upd:4d} lr={lr_now:.2e} R={row['R']:+.2f} best={row['best']:+.2f} "
                  f"win={row['win_heur']:.2f} ent1={ent_pix1:.2f} top1={top1:.2f} "
                  f"kl={kl:.2e} gn={gnorm:.2f} dθ={row['dtheta_rel']:.3f} "
                  f"valid={row['valid']:.2f} [{el:.1f} min]", flush=True)
        if (upd + 1) % a.eval_every == 0:
            c = do_eval(upd + 1)
            save_cnn_actor(actor, os.path.join(a.out, "policy.pt"), cfg, d=actor.d)
            if c > best_cap:
                best_cap = c
                save_cnn_actor(actor, os.path.join(a.out, "best.pt"), cfg, d=actor.d)
    print(f"[diag] done. best greedy cap_rate={best_cap:.4f} ({(time.time() - t0) / 60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
