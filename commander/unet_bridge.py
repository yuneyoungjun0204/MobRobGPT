"""LLM 지휘관 ↔ 'CNN 점수맵' 정책(CnnScoreActor / U-Net lite) 실행 브릿지.

`run_commander_ui.py --unet` 에서 사용. 셀선택 정책을 쓰는 CommandedCellEnv 와 동일 인터페이스
(set_plan/step/get_frame)지만, 경로 기동을 **50×50 래스터 점수맵 위의 픽셀 선택**이 담당한다.

핵심 합성(셀 모델과 완전히 동일한 주입 지점):
  LLM(deployments: 클러스터→ally_ids, deploy_net, radius_adjust, hold)
    → plan_to_assign → _assign / _assignI 주입(_compute_assignment; CommandedDefenseEnv 재사용)
    → 점수맵 정책이 이 배정을 그대로 관측·존중:
        · smap 채널 `self_intercept` = 배정 요격점 1픽셀 스탬프
        · own[6:8] = 정규화 배정 요격점
        · `_cnn_valid_mask` = 환형 ∩ ¬육지 ∩ 요격점 반경게이트(≤3000m) ∩ 방위게이트(±60°)
          → 각 배는 '자기 클러스터로 뻗는 코리도' 픽셀만 고른다 (하드 게이팅)
        · radius_adjust 로 _assignI 반경 이동 → 게이트 중심 이동(더 멀리/가까이 요격)
  정책은 이 '배정-조건부 마스킹'으로 학습됐으므로 LLM 배정 주입은 학습분포 안(충실).

출력 규약 (docs/unet_model_deploy.md §6.3):
  픽셀 K=2개 → 배에서 가까운 순 정렬 → route[0]=이동 WP, route[1]=그물 벽 끝점.
  leg route[0]→route[1] 이 그물(net_mask[1]=True). 미배정·그물소진 배는 제자리 정지.

셀 모델과 다른 점:
  · **cfg 를 손대지 않는다.** 셀 브릿지가 하던 속도 2배·decision_period 반감·nets_per_ship=3
    보정은 이 모델의 학습분포 밖이다. 체크포인트 config 를 원형 그대로 쓴다.
  · greedy_joint 를 쓰지 않는다 — 이 모델은 `cnn_gate_disjoint=False`(겹침 허용) 레짐에서
    학습됐고, 추론에서 joint 잠금을 켜면 학습 레짐을 되돌려 성능이 떨어진다.
  · net_legs 는 무시(픽셀 2점이 곧 벽 하나).
"""
from __future__ import annotations

import numpy as np
import torch

from boatattack_sim.env.defense_env import DefenseVecEnv
from boatattack_sim.env import cnn_map as CM
from boatattack_sim.model.cnn_actor import load_cnn_actor, cnn_obs_to_torch

from .rl_bridge import CommandedDefenseEnv, build_battlefield_defense


class CommandedCnnEnv(CommandedDefenseEnv):
    """LLM 배정을 존중하는 CNN 점수맵 정책 실행 환경(1월드). 경로/그물을 픽셀 2점이 결정."""

    def __init__(self, ckpt: str, enemy_mode: str = "diversionary", device: str = "cpu",
                 avoid_steer: bool | None = None,
                 geo: tuple[float, float] | None = None,
                 nets_per_ship: int = 3):
        # ★ cfg 는 반드시 체크포인트에서 복원한다(직접 만들면 stem in_channels 가 어긋난다).
        actor, cfg = load_cnn_actor(ckpt, device=device)
        if not getattr(cfg, "cnn_action", False):
            raise ValueError(f"{ckpt} 는 CNN 점수맵(cnn_action) 정책이 아닙니다. "
                             f"셀 정책이면 CommandedCellEnv 를 쓰세요.")
        # ── LLM 정합을 위한 최소 오버라이드만 (학습분포 유지) ──
        cfg.n_clusters = 3          # LLM 이 최대 3그룹으로 다룸
        cfg.spawn_phase_lo = 1.0    # 웨이브 텀 설계대로(스폰 랜덤 당김 끄기)
        # ★ 배당 그물 수. 학습 config 는 1이라 한 장 깔면 a_nets=0 → 정지 규칙에 걸려
        #   배가 그대로 얼어붙는다. 운용에서는 배마다 여러 장을 싣고 재전개해야 한다.
        #   a_nets>0 이 유지되면 매 결정 fresh route(_apply_cnn_actions) 가 ptr·leg_netted·
        #   paint_dist 를 리셋하므로, 완성 후 다음 결정에서 새 픽셀 2점으로 다시 전개한다.
        #   ※ 관측 왜곡을 막는 처리는 아래 build_cnn_obs 오버라이드를 볼 것.
        cfg.nets_per_ship = max(1, int(nets_per_ship))
        # ── 운용 해역 고정 (docs/unet_model_deploy.md §2.4) ──
        #   land_sites='korea' 그대로 두면 해역이 매 에피소드 무작위로 바뀌어
        #   화면·마스크·물리가 서로 다른 섬을 가리킨다.
        if geo is not None:
            cfg.geo_lat, cfg.geo_lon = float(geo[0]), float(geo[1])
        cfg.land_sites = ""
        if avoid_steer is not None:
            cfg.avoid_steer = bool(avoid_steer)
        cfg.mother_keepout = True   # APF 를 꺼도 모선-전용 회피는 항상(모선 충돌 방지)

        # super().__init__ 가 _compute_assignment 를 부를 수 있으므로 속성 선주입.
        self._actor = actor
        self._device = device
        self.gain = 1.0                        # 점수맵 모델은 잔차배율 무의미(호환용)
        self._plan = None
        self._plan_command = None
        DefenseVecEnv.__init__(self, num_worlds=1, cfg=cfg, enemy_mode=enemy_mode)
        self._actor.eval()
        self._h = self._actor.init_hidden(self.P, device)   # cnn_recurrent=False 면 None
        self._micro_ct = 0
        self._ev = None
        self.running = True
        self._SK = ("captures", "breaches", "ally_collisions", "nets_used")
        self.stats = {k: 0 for k in self._SK + ("survived",)}
        self._sprev = {k: 0.0 for k in self._SK}
        self.resolve_conflicts = False         # 겹침은 학습 레짐이 허용(joint 잠금 안 씀)
        self._cmd_deploy = np.ones(self.P, bool)
        self._cmd_net_legs = [None] * self.P   # 점수맵 모델 미사용(호환용)
        self._last_pix = None                  # 시각화용 최근 선택 픽셀 [P,K]
        self._last_off = None                  # 시각화용 최근 서브픽셀 오프셋 [P,K,2]
        self._last_prob = None                 # 시각화용 최근 점수맵 [K,P,H,W]
        self._formation = None                 # UI 라벨 호환(특화 라우팅 미사용)

    # ── 관측: own[4] 를 학습분포 안으로 유지 ──
    def build_cnn_obs(self):
        """`own[4] = a_nets / nets_per_ship` 을 **0/1 이진**으로 되돌린 관측.

        이 모델은 `nets_per_ship=1` 로 학습돼 own[4] 가 0 아니면 1이었다
        (docs/unet_model_deploy.md §3.3 이 "0 또는 1"로 명시). 그 자리에서 own[4] 는 사실상
        **'지금 깔 그물이 있는가'** 플래그로 학습됐다 — §3.3 도 own[4]=0·own[5]=0 인 배는
        "행동이 세계를 바꾸지 못한다"고 설명한다.

        그런데 nets_per_ship 을 3으로 올리면 이 값이 1/3·2/3 같은 **학습 때 본 적 없는 중간값**이
        된다. 그래서 정규화 분모만 되돌려 의미를 보존한다: 남은 그물이 1장이든 3장이든 own[4]=1,
        다 쓰면 0. 잔여 장수는 정책이 알 필요가 없다(다음 결정에 어차피 새로 계획한다).
        """
        obs = super().build_cnn_obs()
        obs["own"][..., 4] = np.minimum(self.a_nets, 1.0).astype(obs["own"].dtype)
        return obs

    # ── 결정: 관측 → 정책 → 픽셀 → route/net_mask ──
    def _rl_decide(self):
        """CommandedDefenseEnv.step 이 decision_period 마다 부른다(이름 계약 유지)."""
        obs = self.build_cnn_obs()               # 내부에서 _compute_assignment(LLM 배정 주입)
        with torch.no_grad():
            p, _ = self._actor(cnn_obs_to_torch(obs, self._device))
            a = self._actor.greedy(p)            # ★ 운용은 greedy (sample·greedy_joint 아님)
            K = self._actor.K
            pix = a["pix"].view(1, self.P, K).cpu().numpy()
            off = (a["offset"].view(1, self.P, K, 2).cpu().numpy()
                   if "offset" in a else None)
            self._last_prob = self._actor.score_stages(p, picks=a["pix"])["prob"] \
                                  .view(K, self.P, *obs["valid"].shape[-2:]).cpu().numpy()
        self._last_pix = pix[0]
        self._last_off = off[0] if off is not None else None
        # 정지 규칙(미배정·그물소진 → 제자리)은 _apply_cnn_actions 안에 들어 있다.
        self._apply_cnn_actions(pix, offset=off)
        self._apply_net_decision()               # LLM 의 deploy_net 반영(net_legs 는 미사용)
        self._ev = self.fresh_ev()

    # ── UI 오버레이용 ──
    def cnn_viz(self) -> dict:
        """점수맵 오버레이 데이터. 배별 유효마스크·선택픽셀·확률맵을 world 좌표와 함께."""
        valid = self._cnn_valid_mask()[0]                    # [P,H,W] True=유효
        return {
            "cfg": self.cfg,                                 # eval.cnn_overlay 가 축 규약에 사용
            "px": CM.px_size(self.cfg),
            "valid": valid,                                  # [P,H,W]
            "pix": self._last_pix,                           # [P,K] flat idx | None
            "offset": self._last_off,                        # [P,K,2] 서브픽셀 | None
            "prob": self._last_prob,                         # [K,P,H,W] | None
            "route": self.route[0], "net_mask": self.net_mask[0],
            "assign": self._assign[0],
        }


__all__ = ["CommandedCnnEnv", "build_battlefield_defense"]
