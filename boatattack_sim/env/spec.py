"""
boatattack_sim/env/spec.py — 관측/액션 스키마 (전이 계약 단일 소스)

obs/action 의 차원·정규화·디코드 규칙을 한곳에 정의하고 JSON 으로 export 한다.
학습(Actor) / 시뮬(VecEnv) / Unity 가 동일 스키마를 공유해 불일치를 차단한다.

좌표/정규화 규약은 encoding.py 와 1:1 (egocentric 상대 디코드, [-1,1] 클램프).
"""
import json

from .config import SimConfig, DEFAULT_CONFIG


# ── 관측 feature 정의 (per-agent, 로컬) — BoatAttack 정규화 채용 ──────
#   거리=norm_range/norm_close(유리함수), 방위=signed_bearing(sqrt+부호),
#   헤딩차=heading_cossin(2채널), 스프레드=spread_norm. 적=각도 클러스터 set.
OWN_FEATURES = [
    "pos_x_norm", "pos_y_norm",          # 선박 위치 / world_size (경계 인지)
    "mother_dist_close",                 # 모선 거리 norm_close(가까울수록 1)
    "mother_bearing",                    # 모선 방위 signed_bearing [-1,1]
    "nets_remaining_norm",               # 잔여 그물 / nets_per_ship
    "deploying_flag",                    # 그물 전개중 0/1
    # (제거: progress — 급박함은 적 dist·approach 로 인코딩, 1차원 슬림화 ablation)
    "assigned_flag",                     # 이 배에 클러스터가 배정됐는가 0/1 (0=예비)
    "assigned_dist_range",               # 배정 클러스터 교점 거리 norm_range (미배정=0)
    "assigned_bearing",                  # 배정 클러스터 교점 방위 signed_bearing (미배정=0)
    # + net_probe_<k> (k=0..net_probe_dirs-1): 헤딩기준 k번째 레이 방향의 설치그물 근접도
    #   norm_close (가까울수록 1, 탐지범위 내 그물 없으면 0). 개수=cfg.net_probe_dirs.
]
ENEMY_FEATURES = [                       # ★ per-cluster (각도 클러스터)
    "dist_range",                        # 아군→클러스터 거리 norm_range
    "bearing",                           # 아군 기준 방위 signed_bearing
    "spread_norm",                       # 클러스터 각도 스프레드 (그물 폭 신호)
    "count_norm",                        # 멤버수 / n_enemies
    "approach_norm",                     # 평균 접근속도 / enemy_speed (champion 조건 복원)
]
ALLY_FEATURES = [
    "dist_range",                        # 타 아군 거리 norm_range
    "bearing",                           # 방위 signed_bearing
    "hdg_cos", "hdg_sin",                # 헤딩차 heading_cossin (2채널)
    "plan_net_dist_range",               # ★협조: 그들 계획 그물끝점 거리 (겹침 방지 핵심)
    "plan_net_bearing",                  # ★협조: 그들 계획 그물끝점 방위
    # (제거: nets_remaining_norm/deploying_flag/wp_dir_flag — 협조 미검증·차원 슬림화 ablation)
]


def obs_dims(cfg: SimConfig = DEFAULT_CONFIG) -> dict:
    return {
        # own = 명명 피처 + net radar D칸(net_probe_dirs, 기본0). build_obs 와 순서 일치.
        #   (wp_dir_flag 끝칸 제거 — 순회방향은 액션이라 관측 중복, 슬림화 ablation)
        "own":   len(OWN_FEATURES) + int(getattr(cfg, "net_probe_dirs", 0)),
        "enemy": [cfg.n_clusters, len(ENEMY_FEATURES)],          # [K, F_e] (+mask)
        "ally":  [max(cfg.max_pairs - 1, 1), len(ALLY_FEATURES)],  # [N_a, F_a] (+mask, wp_dir 포함)
    }


def action_spec(cfg: SimConfig = DEFAULT_CONFIG) -> dict:
    """per-agent 액션 스키마 — **잔차 보정**: 지속 풀 경로를 매 결정마다 작은 델타로만 수정."""
    K = cfg.transit_wp
    return {
        "continuous": {
            "wp_delta": {"shape": [K, 2], "low": -1.0, "high": 1.0,
                         "decode": "첫 결정=절대 풀배치(앵커), 이후 route=anchor+wp_delta*wp_adjust_max "
                                   "(일방통행 start→WP0→…→WP_{K-1}, 도달 WP 동결)"},
        },
        "discrete": {
            "net_go": {"shape": [K], "n": 2,
                       "desc": "**레그별** 그물 도색 0/1 [Kw] (잔여 0이면 무시). "
                               "leg k = WP_{k-1}→WP_k 세그먼트. transit leg 는 0 으로 끌 수 있음."},
        },
        "constants": {
            "route_wp": K, "wp_adjust_max": cfg.wp_adjust_max,
            "net_max_len": cfg.net_max_len,
            "world_size": cfg.world_size, "nets_per_ship": cfg.nets_per_ship,
            "decision_period": cfg.decision_period,
        },
    }


def full_spec(cfg: SimConfig = DEFAULT_CONFIG) -> dict:
    return {
        "coord": "nav (x=East, y=North, hdg 0=North CW+); unit=meter; 1 sim = 1 m",
        "obs": {
            "dims": obs_dims(cfg),
            "own_features": OWN_FEATURES,
            "enemy_features": ENEMY_FEATURES,
            "ally_features": ALLY_FEATURES,
            "normalization": {
                "dist_far": "norm_range(d,k)=d/(|d|+k)  (멀수록1)",
                "dist_near": "norm_close(d,k)=k/(d+k)  (가까울수록1)",
                "bearing": "signed_bearing: sqrt(ang/180), 부호=cross(fwd,to)  [-1,1]",
                "heading_diff": "heading_cossin=(cos δ, sin δ)",
                "spread": "spread_norm=clip(spread_deg/90,0,1)",
                "k": {"enemy": cfg.norm_k_enemy, "mother": cfg.norm_k_mother,
                      "ally": cfg.norm_k_ally},
                "n_clusters": cfg.n_clusters,
            },
            "note": "적=각도 클러스터 set. set 항목 mask(True=패딩) 동반. leading dim=N(월드).",
        },
        "action": action_spec(cfg),
        "agents": {
            "param_shared": True, "max_pairs": cfg.max_pairs,
            "decision_period": cfg.decision_period, "team_reward": True,
        },
    }


def export_json(path: str, cfg: SimConfig = DEFAULT_CONFIG) -> str:
    s = full_spec(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "export", "spec.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    export_json(out)
    print("[spec] exported ->", os.path.abspath(out))
    print(json.dumps(full_spec(), ensure_ascii=False, indent=2)[:800])
