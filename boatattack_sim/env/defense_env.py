"""
boatattack_sim/env/defense_env.py — N월드 벡터화 방어 환경 (GRPO 인프라)

모든 텐서 leading dim = N(월드). decision 1회 = 액션 디코드→경로→decision_period PD 롤아웃
(painting/충돌/포획 집계)→보상[N]→autoreset.

GRPO 반사실 K-롤아웃 지원(필수):
  snapshot() / restore(snap) : [N] 월드 전체 상태 저장·복원 (RNG 포함)
  rollout_eval(actions, period) : snapshot 상태에서 굴려 보상[N]만 반환(상태 불변)

상태(state)는 클래스, 운동학/격자/보상/디코드는 순수 모듈(kinematics/grid/reward/encoding).
대화형 단일 월드 Simulator 와 별개의 '학습 최적화' 벡터 구현 (배열 기반, python 리스트 없음).
"""
import numpy as np

from .config import SimConfig, RewardCfg, DEFAULT_CONFIG, DEFAULT_REWARD
from . import formations as F
from . import kinematics as K
from . import encoding as E
from . import reward as RW
from . import spec as SPEC
from . import clustering


class DefenseVecEnv:
    def __init__(self, num_worlds: int = 512, cfg: SimConfig = DEFAULT_CONFIG,
                 rcfg: RewardCfg = DEFAULT_REWARD, enemy_mode: str = "random",
                 seed: int = 0):
        self.N = num_worlds
        self.cfg = cfg
        self.rcfg = rcfg
        self.enemy_mode = enemy_mode
        self.M = cfg.n_enemies
        self.P = cfg.n_allies
        self.Kw = cfg.transit_wp          # 최대 경유 WP
        self.G = cfg.grid_size
        self.cell = cfg.cell_size
        self.center = np.array(cfg.center, dtype=np.float64)
        self.world_half = cfg.world_size / 2.0
        self.nets_total = cfg.nets_per_ship * cfg.n_allies
        # ★ 셀선택 행동공간: 극좌표 후보셀(정적, 에피소드 불변) — pointer 어텐션 keys
        from .cell_action import make_cells
        self.cell_world, self.cell_polar = make_cells(cfg, self.center)   # [C,2],[C,2]
        self.n_cells = int(self.cell_world.shape[0])
        # 도달가능 반경(휴리스틱 후보 경로용): (r-s0)/va=(H-r)/ve → r=(va·H+ve·s0)/(va+ve)
        self._R_FEAS = (cfg.ally_speed * self.world_half
                        + cfg.enemy_speed * cfg.ally_row_gap) / (cfg.ally_speed + cfg.enemy_speed)
        self.rng = np.random.default_rng(seed)
        self._rot_i = 0                    # "rotate" 모드 순환 카운터
        self.world_mode = np.array(["random"] * self.N, dtype=object)  # 월드별 현재 스폰 모드
        self.reset()

    # 로테이션 순서: 집중 → 양동 → 파상 (한 번씩 반복)
    _ROTATION = ("concentrated", "diversionary", "wave")
    # ★ 'mixed' = 모든 포메이션을 매 스폰마다 무작위 (일반화 학습용)
    #   가중치: 어려운(흩어진) wave·random 을 2배 더 자주 → cap 약점 보강.
    _ALL_MODES = ("concentrated", "diversionary", "wave", "grouped", "random")
    _MIX_WEIGHTS = np.array([1.0, 1.0, 2.0, 1.0, 2.0])   # _ALL_MODES 와 정렬

    def _next_mode(self) -> str:
        """enemy_mode='rotate' 면 집중/양동/파상을 매 스폰마다 한 칸씩 순환.
        'mixed' 면 모든 포메이션을 가중 무작위(wave·random 2배) 선택(전 포메이션 일반화 + 약점 보강)."""
        if self.enemy_mode == "rotate":
            m = self._ROTATION[self._rot_i % len(self._ROTATION)]
            self._rot_i += 1
            return m
        if self.enemy_mode == "mixed":
            p = self._MIX_WEIGHTS / self._MIX_WEIGHTS.sum()
            return str(self.rng.choice(self._ALL_MODES, p=p))
        return self.enemy_mode

    # ── 스폰 ──────────────────────────────────────────────────────────
    def _spawn_worlds(self, idx):
        """idx 월드들의 적/아군 초기화."""
        lo = getattr(self.cfg, "spawn_phase_lo", 1.0)
        for n in idx:
            mode = self._next_mode()
            self.world_mode[n] = mode
            ep, eh, eph = F.spawn_enemies(self.cfg, self.rng, mode)
            # 월드 비동기화: 월드별 시작 거리를 랜덤 축소 → 에피소드 길이 분산 → 리셋 분산
            #   → 매 update 에 '배치중 월드'가 섞여 counterfactual 신호(valid) 지속.
            if lo < 1.0:
                u = self.rng.uniform(lo, 1.0)
                ep = self.center + (ep - self.center) * u
            self.e_pos[n] = ep; self.e_hdg[n] = eh; self.e_phase[n] = eph
            self.e_alive[n] = True
            ap, ah = F.spawn_allies(self.cfg, self.rng)
            self.a_pos[n] = ap; self.a_hdg[n] = ah
            self.a_nets[n] = self.cfg.nets_per_ship
            self.a_alive[n] = True
            # ★ 가변 아군 수: 월드별 활성 아군 수를 ally_choices 중 균등 추출, 나머지는 비활성(부재).
            if getattr(self.cfg, "vary_allies", False) and self.P > 1:
                k = max(1, min(int(self.rng.choice(self.cfg.ally_choices)), self.P))
                self.n_active[n] = k
                self.a_alive[n, k:] = False           # 부재 아군 = a_alive False (할당/이동/obs/충돌 제외)
            else:
                self.n_active[n] = self.P
        # 격자·플랜·타이머 리셋
        self.painted[idx] = False
        self.net_installed[idx] = False
        self.prev_on_inst[idx] = False
        self.t[idx] = 0
        self.done[idx] = False
        if hasattr(self, "fan_anchor"):
            self.fan_anchor[idx] = 0.0                # 새 에피소드 = 부채꼴 앵커 리셋(첫 결정서 재설정)
        self._reset_plans(idx)
        self._prev_route[idx] = self.route[idx]       # 스폰 직후 일관성 기준=현 경로(첫 결정 페널티 0)

    def _reset_plans(self, idx):
        # 기본 경로: 모선→아군 '바깥' 방향으로 route_step 간격의 Kw개 WP (이후 잔차로만 보정)
        cfg = self.cfg; c = self.center
        out = self.a_pos[idx] - c
        d = out / (np.hypot(out[..., 0], out[..., 1])[..., None] + 1e-6)   # 바깥 단위방향
        ks = (np.arange(self.Kw) + 1) * cfg.route_step                     # [Kw]
        route = (self.a_pos[idx][:, :, None, :]
                 + d[:, :, None, :] * ks[None, None, :, None])
        self.route[idx] = np.clip(route, 0.0, cfg.world_size)   # 첫 결정 전 기본값(폴백)
        self.route_anchor[idx] = self.route[idx]                # 앵커도 폴백으로 초기화
        self.anchor_route[idx] = self.route[idx]                # ★ moving-anchor 초기화(폴백 경로)
        self.wp_reached[idx] = False                            # 새 에피소드 = 도달기록 리셋
        self.route_init[idx] = False                            # 다음 결정 = 절대 풀 배치
        self.net_mask[idx] = False
        self.leg_netted[idx] = False
        self.net_end[idx] = self.a_pos[idx]
        self.net_start[idx] = self.a_pos[idx]
        self.ptr[idx] = 0
        self.wp_dir[idx] = False                                # ★ 새 에피소드 = 방향 미확정(다음 결정에 확정)
        self.wp_dir_set[idx] = False
        self._prev_assign[idx] = -1
        self._last_apply_ptr[idx] = 0
        self.doing_net[idx] = False
        self.paint_dist[idx] = 0.0
        if self._visited is not None:
            self._visited[idx] = False       # ★ 순차: 방문셀 리셋(에피소드 시작)

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        N, M, P, Kw, G = self.N, self.M, self.P, self.Kw, self.G
        # 적
        self.e_pos = np.zeros((N, M, 2)); self.e_hdg = np.zeros((N, M))
        self.e_phase = np.zeros((N, M)); self.e_alive = np.ones((N, M), bool)
        # 아군
        self.a_pos = np.zeros((N, P, 2)); self.a_hdg = np.zeros((N, P))
        self.a_nets = np.zeros((N, P), np.int64)
        self.a_alive = np.ones((N, P), bool)          # 충돌 시 비활성화
        self.n_active = np.full(N, P, np.int64)       # 월드별 활성 아군 수(vary_allies 시 1..P)
        # 지속 경로(route = Kw WP, 일방통행 start→WP0→…→WP_{Kw-1}) + 그물 상태
        self.route = np.zeros((N, P, Kw, 2))          # 풀 경로 (world)
        self.anchor_route = np.zeros((N, P, Kw, 2))   # ★ A) moving-anchor: 정책 경로 EMA(base 블렌드용)
        self._prev_route = np.zeros((N, P, Kw, 2))    # 직전 결정 경로(시간 일관성 보상용)
        self.route_anchor = np.zeros((N, P, Kw, 2))   # 첫 결정 절대배치 앵커(잔차 기준점, 불변)
        self.wp_reached = np.zeros((N, P, Kw), bool)  # 이미 도달한 WP(=동결, 더 보정 안 함)
        self.route_init = np.zeros((N, P), bool)      # 첫 배치(절대) 완료 여부
        self.net_mask = np.zeros((N, P, Kw), bool)    # ★ 레그별 그물 깔지(net_go[Kw])
        self.leg_netted = np.zeros((N, P), bool)      # 현재 구간(ptr-leg)에 이미 그물 시작했는지
        self.net_end = np.zeros((N, P, 2))            # 현재 구간 그물 끝점 = 목표 WP (obs/렌더용)
        self.net_start = np.zeros((N, P, 2))          # 현재 구간 그물 시작점
        self.ptr = np.zeros((N, P), np.int64)         # 현재 향하는 WP 인덱스 (일방통행, Kw-1 에서 멈춤)
        self.wp_dir = np.zeros((N, P), bool)          # ★ WP 순회방향 (True=역방향) — 첫 결정 확정·동결
        self.wp_dir_set = np.zeros((N, P), bool)      # ★ wp_dir 확정 여부(래칭): 한번 정하면 안 바뀜
        self._prev_assign = np.full((N, P), -1, np.int64)  # 직전 결정 배정(재engagement 감지=경로커밋)
        self._last_apply_ptr = np.zeros((N, P), np.int64)  # 직전 결정 ptr(도착 트리거 재계획)
        self.doing_net = np.zeros((N, P), bool)       # 현재 구간 도색중
        self.paint_dist = np.zeros((N, P))
        self.fan_anchor = np.zeros((N, P, 7))         # ★ 부채꼴 파라미터 하드앵커(첫결정/재engagement 고정)
        self._fan_change = np.zeros((N, P))           # |fan_eff - anchor| 평균(로깅/소프트페널티용)
        self.prev_on_inst = np.zeros((N, P), bool)    # 직전 step 설치된 그물 위에 있었나(진입 edge 검출)
        # ★ 순차 부설: 배별 방문(선택완료) 셀 — 재선택 제외용
        self._visited = np.zeros((N, P, self.n_cells), bool) if self.cfg.cell_action else None
        # 격자·시간
        self.painted = np.zeros((N, G, G), bool)
        self.net_installed = np.zeros((N, G, G), bool)  # 완성(설치)된 그물 셀 — 아군 접촉 패널티용
        self.t = np.zeros(N, np.int64)
        self.done = np.zeros(N, bool)
        self._wt = 0
        self._spawn_worlds(np.arange(N))
        return self.build_obs(), {}

    # ── 액션 적용 (★첫 결정=절대 풀배치, 이후=잔차 미세보정) ─────────
    def _apply_actions(self, actions, force_fresh=None):
        cfg = self.cfg
        # ── ★ 셀선택 모드: 선택 셀 → 절대 route/net_mask (매 결정 재계획, locked 그물 보호) ──
        if cfg.cell_action and actions.get("cells") is not None:
            self._compute_assignment(actions.get("assign_pref"))   # obs/렌더 일관용(배정선)
            self._apply_cell_actions(actions["cells"], force_fresh, actions.get("offset"))
            return
        self._compute_assignment(actions.get("assign_pref"))  # 결정 시작 배정(+정책 선호 soft-bias)
        net_go = np.asarray(actions["net_go"], np.int64).astype(bool)     # [N,P,Kw] 레그별 (공통)
        if cfg.structured_action:                                         # 부채꼴: wp 없음 → _fan_change 사용
            wp = None; self._resid_mag = np.zeros((self.N, self.P))
        else:
            wp = np.clip(np.asarray(actions["wp"], np.float64), -1, 1)    # [N,P,Kw,2]
            self._resid_mag = (wp ** 2).mean(axis=(2, 3))                 # [N,P] 잔차 크기(앵커 페널티용)

        alive = self.a_alive
        locked = self.doing_net.copy()                       # 전개중 그물 보호(에이전트 단위)
        keep = locked | (~alive)                             # net 결정·paint 보호용

        # ★ WP 순회방향 래칭: 첫 결정(미확정 살아있는 배)에 정/역 확정·동결. eval(force_fresh)에선
        #   후보가 방향을 바꿔볼 수 있게 적용(상태는 rollout_eval restore 로 되돌아감 → 실제는 동결 유지).
        wpd = actions.get("wp_dir")
        if wpd is not None:
            wpd = np.asarray(wpd).astype(bool).reshape(self.N, self.P)
            ffb = (np.asarray(force_fresh, bool) if force_fresh is not None
                   else np.zeros((self.N, self.P), bool))
            set_now = ((~self.wp_dir_set) & alive) | ffb     # 첫 확정 | eval 후보
            self.wp_dir = np.where(set_now, wpd, self.wp_dir)
            self.wp_dir_set = self.wp_dir_set | ((~self.wp_dir_set) & alive)   # 실제는 한번만 래칭

        # ── 휴리스틱 baseline + 정책 잔차(refine) + 경로 커밋(일관성) ──────────
        if cfg.heuristic_baseline:
            assigned = (self._assign >= 0)                   # [N,P]
            if cfg.structured_action:                        # ★ 부채꼴 7파라미터(하드앵커 → fan_eff → 경로)
                fan = np.clip(np.asarray(actions["fan"], np.float64), -1.0, 1.0)
                re_eng = ((self._assign != self._prev_assign) | (~self.route_init)) & alive  # 첫/재engagement
                ffb = (np.asarray(force_fresh, bool) if force_fresh is not None
                       else np.zeros((self.N, self.P), bool))
                set_anchor = re_eng | ffb                    # 앵커 재설정(첫·재engagement·eval 강제)
                self.fan_anchor = np.where(set_anchor[..., None], fan, self.fan_anchor)
                dadj = np.asarray(cfg.fan_adjust_max, np.float64)[None, None, :]  # [1,1,7] 파라미터별 캡
                fan_eff = np.clip(self.fan_anchor + np.clip(fan - self.fan_anchor, -dadj, dadj), -1.0, 1.0)
                self._fan_change = np.abs(fan_eff - self.fan_anchor).mean(-1)     # [N,P] 실효 편차(로깅)
                base, hmask = self.heuristic_route_netgo(fan_eff)   # 부채꼴이 곧 경로(잔차 없음)
                new_route = np.clip(base, 0.0, cfg.world_size)
            else:                                            # ── 기존 잔차(보존) ──
                base, hmask = self.heuristic_route_netgo()   # 강한 부채꼴 baseline [N,P,Kw,2]
                if self.wp_dir.any():                        # WP 역방향(잔차 전용)
                    rev = self.wp_dir[..., None, None]
                    base = np.where(rev, base[:, :, ::-1, :], base)
                    hmask = np.where(self.wp_dir[..., None], hmask[:, :, ::-1], hmask)
                if cfg.moving_anchor:                         # moving-anchor 블렌드
                    base = np.clip((1.0 - cfg.anchor_weight) * base
                                   + cfg.anchor_weight * self.anchor_route, 0.0, cfg.world_size)
                if cfg.wp_residual_mode == "rigid":
                    wp_delta = np.broadcast_to(wp.mean(axis=2, keepdims=True), wp.shape)
                elif cfg.wp_residual_mode == "cumulative":
                    wp_delta = np.cumsum(wp, axis=2)
                else:
                    wp_delta = wp
                new_route = np.clip(base + wp_delta * cfg.wp_adjust_max * assigned[..., None, None],
                                    0.0, cfg.world_size)     # 미배정=잔차0 → new_route=base=정지
            hmask = hmask & assigned[..., None]              # 미배정=그물 0 (명시)
            # ★ GIF 시점 행동공간 재편: n_follow 제거 — 모든 WP를 따라감(휴리스틱과 동일, 배회억제는 보상에 위임).
            if cfg.wp_repulsion:                             # ★ 계획 WP끼리 척력 → plan 벌림
                new_route = self._repel_wps(new_route)
            # ★ route rotate: 완성 route 를 시작점(route[0]) 기준 ±rot_max_deg 회전(그물 벽 동반 회전).
            rot = actions.get("rot")
            if rot is not None and cfg.rot_max_deg != 0.0:
                rotv = np.clip(np.asarray(rot, np.float64).reshape(self.N, self.P), -1.0, 1.0)
                theta = np.deg2rad(cfg.rot_max_deg) * rotv * assigned        # [N,P] 미배정=0
                cs = np.cos(theta)[..., None]; sn = np.sin(theta)[..., None]  # [N,P,1]
                pivot = new_route[:, :, 0:1, :]                              # [N,P,1,2] route 시작점
                rel = new_route - pivot                                      # [N,P,Kw,2]
                rx = rel[..., 0] * cs - rel[..., 1] * sn                     # [N,P,Kw]
                ry = rel[..., 0] * sn + rel[..., 1] * cs
                new_route = np.clip(pivot + np.stack([rx, ry], axis=-1), 0.0, cfg.world_size)
            # ★ 경로 커밋(중간 균형): '첫 결정/재engagement(배정변경)' 때만 전체 (재)배치하고,
            #   그 외엔 **지나온 WP + 현재 향하는 WP(=ptr)는 고정**(따라가는 경로 안정), 단
            #   **아직 안 간 미래 WP(>ptr)는 매 결정 baseline+잔차로 적응** → 적 움직임 추종.
            #   너무 정적(앞쪽도 안 변함)/너무 jumpy(전부 재배치) 사이의 절충. force_fresh=GRPO 평가용.
            kar = np.arange(self.Kw)[None, None, :]
            reeng = (self._assign != self._prev_assign)      # [N,P] 배정변경=재engagement
            self._prev_assign = self._assign.copy()
            ff = force_fresh if force_fresh is not None else False
            # 설치중(locked)·dead 는 route 완전동결(keep) — 그물벽 직선 보장(최고모델 0.936 세팅).
            #   (설치중 비동결/부분동결은 벽 품질 저하로 capture 퇴화 확인 → 복원함)
            fresh = (~keep) & (reeng | ~self.route_init | ff)        # [N,P] 전체 (재)배치
            committed = (~keep) & (~fresh) & self.route_init         # 커밋된 배: 미래 WP만 적응
            # ★ free_current_wp: 현재 향하는 WP(=ptr)도 잔차 반영(>=). 기본 False=현재 WP 고정(>).
            future = ((kar >= self.ptr[..., None]) if cfg.free_current_wp
                      else (kar > self.ptr[..., None]))      # 현재목표(포함)보다 앞 WP
            write = fresh[..., None] | (committed[..., None] & future)
            self.route = np.where(write[..., None], new_route, self.route)
            self.route_anchor = self.route.copy()
            self.net_mask = np.where(write, hmask.astype(bool), self.net_mask)
            self.wp_reached = np.where(fresh[..., None], False, self.wp_reached)
            self.ptr = np.where(fresh, 0, self.ptr)          # fresh 때만 ptr 리셋, 그 외 전진 유지
            self.leg_netted = np.where(fresh, False, self.leg_netted)
            self.route_init = self.route_init | fresh
            self.paint_dist = np.where(fresh, 0.0, self.paint_dist)
            self.doing_net = locked.copy()
            self._last_apply_ptr = self.ptr.copy()           # 다음 결정 도착감지 기준
            # ★ A) moving-anchor EMA: anchor_route 를 정책 실제 경로로 점진 추종(살아있는 배만).
            if cfg.moving_anchor:
                a = cfg.anchor_alpha
                upd = (1.0 - a) * self.anchor_route + a * self.route
                self.anchor_route = np.where(alive[..., None, None], upd, self.anchor_route)
            return

        if cfg.absolute_replan:                              # 매 결정 절대 재계획(잔차 OFF)
            first = alive & (~locked)
        else:
            first = (~self.route_init) & alive & (~locked)   # 에피소드 첫 결정만 절대 배치
        f3 = first[..., None, None]

        # 절대 풀 배치 (egocentric chaining, wp → 0..wp_max_len) — 가장 중요한 결정
        abs_route, _ns, _ne = E.decode_plan(
            wp, self.a_pos, self.a_hdg, np.zeros((self.N, self.P), np.int64),
            np.zeros((self.N, self.P, 2)),
            cfg.wp_max_len, cfg.net_max_len, cfg.world_size)
        # 앵커 고정: 첫 결정의 절대배치를 기준점으로 저장(이후 불변).
        self.route_anchor = np.where(f3, abs_route, self.route_anchor)
        self.wp_reached = np.where(first[..., None], False, self.wp_reached)  # 첫 결정=도달기록 리셋

        # ── WP별 동결 마스크 (★신호 핵심) ──────────────────────────────
        #   경로 전체를 얼리지 않는다: 도색중이라도 '미래 WP'는 자유 조정 → 다음 leg 배치가
        #   action 에 반응 → counterfactual 분산(valid) 회복. 동결 = 다음 중 하나:
        #     비활성 | 이미 도달(wp_reached) | (도색중 & 현재 목표 WP=ptr  ← 진행 net 보호)
        kar = np.arange(self.Kw)[None, None, :]
        lock_cur = self.doing_net[..., None] & (kar == self.ptr[..., None])   # [N,P,Kw]
        freeze_wp = (~alive)[..., None] | self.wp_reached | lock_cur          # [N,P,Kw]
        # 잔차 미세 보정: 앵커 기준 ±wp_adjust_max (동결 WP 는 현 route 유지).
        res_route = np.clip(self.route_anchor + wp * cfg.wp_adjust_max, 0.0, cfg.world_size)
        res_route = np.where(freeze_wp[..., None], self.route, res_route)
        self.route = np.where(f3, abs_route, res_route)
        # ★ 미배정(예비) 배 = 완전 정지: 경로를 현 위치로 고정(경로생성 안 함), 그물 0.
        unassigned = (self._assign < 0)                      # [N,P]
        hold = np.broadcast_to(self.a_pos[:, :, None, :], self.route.shape)
        self.route = np.where(unassigned[..., None, None], hold, self.route)
        net_go = net_go & (~unassigned)[..., None]

        self.ptr = np.where(first, 0, self.ptr)              # 첫 배치 시 ptr 리셋
        self.leg_netted = np.where(first, False, self.leg_netted)
        self.route_init = self.route_init | first
        # net_go: 레그별 그물 깔지 마스크. 도색중이면 보호(keep). 자원 가드는 start_net 에서.
        self.net_mask = np.where(keep[..., None], self.net_mask, net_go)
        self.paint_dist = np.where(keep, self.paint_dist, 0.0)
        self.doing_net = locked.copy()
        # ★ 직전 배정 갱신(순수 정책 경로) — 다음 결정의 assign_sticky_bonus 가 실제로 적용되도록.
        #   (heuristic_baseline 분기는 reeng 감지 후 별도 갱신; 이 경로는 reeng 미사용이라 여기서 갱신.)
        self._prev_assign = self._assign.copy()

    # ── 격자 painting (벡터 scatter) ─────────────────────────────────
    def _paint(self, painting_mask):
        if not painting_mask.any():
            return
        h = max(0, (self.cfg.net_width - 1) // 2)
        ci = np.clip((self.a_pos[..., 0] // self.cell).astype(np.int64), 0, self.G - 1)
        cj = np.clip((self.a_pos[..., 1] // self.cell).astype(np.int64), 0, self.G - 1)
        nidx = np.broadcast_to(np.arange(self.N)[:, None], (self.N, self.P))
        sel = painting_mask
        ns = nidx[sel]
        for di in range(-h, h + 1):
            ii = np.clip(ci[sel] + di, 0, self.G - 1)
            for dj in range(-h, h + 1):
                jj = np.clip(cj[sel] + dj, 0, self.G - 1)
                self.painted[ns, ii, jj] = True

    def _rasterize_net(self, A, B, mask):
        """완성된 그물 세그먼트 A→B 를 net_installed 격자에 띠로 등록 (접촉 패널티용).
        A,B [N,P,2], mask [N,P]. 셀 해상도로 샘플링."""
        if not mask.any():
            return
        h = max(0, (self.cfg.net_width - 1) // 2)
        ns, ps = np.where(mask)
        a = A[ns, ps]; b = B[ns, ps]                        # [K,2]
        length = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        nstep = int(max(2, np.ceil(length.max() / self.cell) + 1))
        for s in np.linspace(0.0, 1.0, nstep):
            p = a + s * (b - a)                             # [K,2]
            ci = np.clip((p[:, 0] // self.cell).astype(np.int64), 0, self.G - 1)
            cj = np.clip((p[:, 1] // self.cell).astype(np.int64), 0, self.G - 1)
            for di in range(-h, h + 1):
                ii = np.clip(ci + di, 0, self.G - 1)
                for dj in range(-h, h + 1):
                    jj = np.clip(cj + dj, 0, self.G - 1)
                    self.net_installed[ns, ii, jj] = True

    # ── 휴리스틱 충돌회피 안전층 (reactive APF; 정책과 독립) ──────────
    def _avoid_steer(self, target):
        """WP 추종 target[N,P,2] 에 아군/모선 척력 오프셋을 더한 '조향 target' 반환.
        척력 = away-방향 단위벡터 × (R−d) (m, 영향반경 안에서만 선형). 합산 후 크기 상한.
        가까운 아군/모선에서 멀어지게 target 을 밀어 pd_follow 가 회피 경로를 타게 한다."""
        cfg = self.cfg; rc = self.rcfg; P = self.P
        p = self.a_pos; av = self.a_alive; c = self.center
        # 아군-아군: 쌍거리 < avoid_r 인 타선에서 (R−d) 만큼 밀어냄.
        dx = p[:, :, None, 0] - p[:, None, :, 0]
        dy = p[:, :, None, 1] - p[:, None, :, 1]
        dd = np.hypot(dx, dy)                                       # [N,P,P]
        pair = av[:, :, None] & av[:, None, :] & ~np.eye(P, dtype=bool)[None]
        push = np.where(pair & (dd < rc.avoid_r), rc.avoid_r - dd, 0.0)
        inv = 1.0 / np.maximum(dd, 1e-6)
        ox = (push * dx * inv).sum(2); oy = (push * dy * inv).sum(2)   # [N,P]
        # 아군-모선: death-disk(ally_mother_radius) 가장자리 밖 거리 기준, 더 세게.
        mx = p[..., 0] - c[0]; my = p[..., 1] - c[1]
        md = np.hypot(mx, my); dm = md - cfg.ally_mother_radius
        mpush = np.where((dm < rc.mother_avoid_r) & av,
                         rc.mother_avoid_w * (rc.mother_avoid_r - dm), 0.0)
        minv = 1.0 / np.maximum(md, 1e-6)
        ox = ox + mpush * mx * minv; oy = oy + mpush * my * minv
        off = np.stack([ox, oy], -1) * cfg.avoid_steer_gain           # [N,P,2]
        mag = np.hypot(off[..., 0], off[..., 1])                      # 크기 상한
        scale = np.minimum(1.0, cfg.avoid_steer_cap / np.maximum(mag, 1e-6))
        off = off * scale[..., None]
        out = np.where(av[..., None], target + off, target)           # 죽은 배 제외
        return np.clip(out, 0.0, cfg.world_size)

    def _mother_keepout(self, target):
        """★ 모선-전용 회피(APF와 별개, 항상 켜짐). 추종 target 을 모선 죽음원반에서 바깥으로만 민다.
        아군-아군 척력은 없음(순수 RL 경로 유지) → 셀로 전이할 때 경로가 모선 중심을 관통해
        obstacle_collision 으로 격침되는 것을 방지(특히 wave). 세기는 avoid_steer 대비 강하게."""
        cfg = self.cfg; rc = self.rcfg; av = self.a_alive; c = self.center
        p = self.a_pos
        mx = p[..., 0] - c[0]; my = p[..., 1] - c[1]
        md = np.hypot(mx, my); dm = md - cfg.ally_mother_radius           # 죽음원반 가장자리까지
        R = float(getattr(rc, "mother_avoid_r", 500.0 * getattr(self.cfg, "scale", 1.0)))                  # APF 와 동일 세기(과밀어내기 방지)
        W = float(getattr(rc, "mother_avoid_w", 1.0))
        mpush = np.where((dm < R) & av, W * (R - dm), 0.0)
        minv = 1.0 / np.maximum(md, 1e-6)
        off = np.stack([mpush * mx * minv, mpush * my * minv], -1)       # 바깥 방향
        off = off * float(cfg.avoid_steer_gain)
        mag = np.hypot(off[..., 0], off[..., 1])
        scale = np.minimum(1.0, float(cfg.avoid_steer_cap) / np.maximum(mag, 1e-6))
        off = off * scale[..., None]
        out = np.where(av[..., None], target + off, target)
        return np.clip(out, 0.0, cfg.world_size)

    # ── WP-레벨 척력: 계획된 WP들끼리 밀어내 plan 을 벌린다(몸체 무관) ──
    def _repel_wps(self, route):
        """route[N,P,Kw,2] 의 WP 들을 **다른 배의 WP** 에서 (R−d) 척력으로 밀어 displace.
        같은 배 내부 WP·죽은 배는 제외. 결정 1회용(계획층) — 몸체는 이후 순수 PD 로 추종.
        반복(force-directed)으로 안정화, world 경계 클립."""
        cfg = self.cfg; N, P, Kw = self.N, self.P, self.Kw
        R = cfg.wp_repel_r; eta = cfg.wp_repel_gain
        M = P * Kw
        ship_id = np.repeat(np.arange(P), Kw)                    # [M] WP→배
        cross = (ship_id[:, None] != ship_id[None, :])           # [M,M] 다른 배 쌍
        wp_alive = np.repeat(self.a_alive, Kw, axis=1)           # [N,M]
        c = self.center; Rm = cfg.wp_repel_mother_r
        out = route.reshape(N, M, 2).copy()
        for _ in range(int(cfg.wp_repel_iters)):
            dx = out[:, :, None, 0] - out[:, None, :, 0]
            dy = out[:, :, None, 1] - out[:, None, :, 1]
            dd = np.hypot(dx, dy)                                # [N,M,M]
            valid = (cross[None] & (dd < R) & (dd > 1e-6)
                     & wp_alive[:, :, None] & wp_alive[:, None, :])
            push = np.where(valid, R - dd, 0.0)
            inv = 1.0 / np.maximum(dd, 1e-6)
            ox = (push * dx * inv).sum(2); oy = (push * dy * inv).sum(2)   # [N,M] 아군 WP 척력
            # ── ★ 모선 척력: 반경 Rm 안 WP 를 모선 밖으로 밀어냄(경로가 모선 안 지나게) ──
            mx = out[:, :, 0] - c[0]; my = out[:, :, 1] - c[1]
            md = np.hypot(mx, my)
            mpush = np.where((md < Rm) & (md > 1e-6) & wp_alive, Rm - md, 0.0)
            minv = 1.0 / np.maximum(md, 1e-6)
            ox = ox + mpush * mx * minv; oy = oy + mpush * my * minv
            out[:, :, 0] = np.clip(out[:, :, 0] + eta * ox, 0.0, cfg.world_size)
            out[:, :, 1] = np.clip(out[:, :, 1] + eta * oy, 0.0, cfg.world_size)
        return out.reshape(N, P, Kw, 2)

    # ── RRT 경로를 route 에 직접 주입(시각화/추종용; 단일월드 루프) ──────
    def apply_rrt_routes(self, rrt_res, w=0, lay_nets=False):
        """rrt_planner.plan_env_world 결과를 월드 w 의 self.route 에 심어 배가 **그 경유점만**
        pd_follow 로 추종하게 한다(척력/인력 없음 — avoid_steer 는 cfg 로 OFF). 결정마다 호출.
        RRT short 경로(가변 길이 L)를 route 슬롯(Kw)에 채우고 남으면 마지막 점 반복(정지).
        lay_nets=False(기본): 그물 안 깔고 순수 추종만. True: 실 leg(1..nnet,<L)에 그물 살포."""
        Kw = self.Kw; cfg = self.cfg
        legs = np.arange(Kw)
        nnet = min(cfg.nets_per_ship, Kw - 1)
        for p in range(self.P):
            if p in rrt_res:
                sp = np.asarray(rrt_res[p]["short"], float)          # [L,2] world
                L = int(min(len(sp), Kw))
                r = np.empty((Kw, 2))
                r[:L] = sp[:L]
                if L < Kw:
                    r[L:] = sp[L - 1]                                 # 마지막 점 반복(도달 후 정지)
                self.route[w, p] = np.clip(r, 0.0, cfg.world_size)
                if lay_nets:
                    nw = rrt_res[p].get("net_wp")               # 플래너 지정 그물 leg(sweep)
                    nm = np.zeros(Kw, bool)
                    if nw is not None:
                        nm[:L] = np.asarray(nw, bool)[:L]
                    else:
                        nm = (legs >= 1) & (legs <= nnet) & (legs < L)
                    self.net_mask[w, p] = nm
                else:
                    self.net_mask[w, p] = np.zeros(Kw, bool)
            else:                                                    # 미배정/사망 = 정지
                self.route[w, p] = self.a_pos[w, p]
                self.net_mask[w, p] = False
            self.route_anchor[w, p] = self.route[w, p]
            self.wp_reached[w, p] = False
            self.ptr[w, p] = 0
            self.leg_netted[w, p] = False
            self.route_init[w, p] = True
            self.doing_net[w, p] = False
            self.paint_dist[w, p] = 0.0
        self.net_end[w] = self.route[w, :, 0]
        self.net_start[w] = self.a_pos[w]

    # ── 1 micro-step ─────────────────────────────────────────────────
    def _micro(self, ev):
        cfg = self.cfg
        N, P, M, Kw = self.N, self.P, self.M, self.Kw

        # 목표 = 현재 향하는 WP (일방통행). 그물도 이 WP 방향으로 깔린다(net=구간 leg).
        ptr_c = np.clip(self.ptr, 0, Kw - 1)
        route_t = np.take_along_axis(self.route, ptr_c[:, :, None, None], axis=2)[:, :, 0, :]
        target = route_t
        # 그물 끝점 = 목표 WP (구간 끝). obs/렌더/배치 shaping 에서 사용.
        self.net_end = route_t.copy()

        # ── ★ 휴리스틱 충돌회피 안전층: 추종 target 을 아군/모선 척력으로 밀어 물리적 회피 ──
        #   (도착 판정은 아래에서 '진짜 WP(route_t)' 기준으로 재계산 → 경로/그물 로직 불변)
        if cfg.avoid_steer:
            steer_target = self._avoid_steer(route_t)                 # 전체 APF(아군+모선)
        elif getattr(cfg, "mother_keepout", False):
            steer_target = self._mother_keepout(route_t)              # 모선-전용(APF 꺼짐)
        else:
            steer_target = target

        # 그물 시작 트리거: **현재 leg(ptr) 의 net_mask 가 켜짐** · 미시작 · 도색안함 · 잔여>0 · 활성.
        #   → transit leg(mask=0)는 도색 안 하고, 차단 leg(mask=1)만 그물로 깐다(simulator 동형).
        cur_net = np.take_along_axis(self.net_mask, ptr_c[..., None], axis=2)[..., 0]   # [N,P]
        start_net = (cur_net & (~self.leg_netted) & (~self.doing_net)
                     & self.a_alive & (self.a_nets > 0))
        if start_net.any():
            self.doing_net |= start_net
            self.leg_netted |= start_net
            self.paint_dist[start_net] = 0.0
            self.net_start[start_net] = self.a_pos[start_net]   # 전개 시작점
            self.a_nets -= start_net.astype(np.int64)
            ev["nets_used"] += start_net.sum(axis=1)
            ev["deployed"] |= start_net

        moving = self.a_alive & (self._assign >= 0)      # 비활성/미배정(예비) 아군은 완전 정지
        painting = self.doing_net & (self.paint_dist < cfg.net_max_len) & self.a_alive

        # ★ 그물 전개중(painting) 이동비용: 감속·선회제한 (per-ship 배열). '쉬기'에 이득 부여.
        dsm = getattr(cfg, "deploy_speed_mult", 1.0)
        dtm = getattr(cfg, "deploy_turn_mult", 1.0)
        spd = (cfg.ally_speed * np.where(painting, dsm, 1.0)).reshape(-1)        # [N*P]
        mturn = (cfg.ally_max_turn * np.where(painting, dtm, 1.0)).reshape(-1)   # [N*P]
        # PD 이동 (flatten). 조향은 회피-오프셋 target, 도착은 진짜 WP(route_t) 기준.
        pos_old = self.a_pos.reshape(-1, 2)
        pos_new, hdg_new, _ = K.pd_follow(
            pos_old, self.a_hdg.reshape(-1), steer_target.reshape(-1, 2),
            spd, mturn, cfg.dt,
            turn_gain=cfg.ally_turn_gain, slow_min=cfg.ally_slow_min,
            arrive_radius=cfg.arrive_radius)
        pos_new = pos_new.reshape(N, P, 2); hdg_new = hdg_new.reshape(N, P)
        arrived = (np.hypot(pos_new[..., 0] - route_t[..., 0],
                            pos_new[..., 1] - route_t[..., 1]) <= cfg.arrive_radius)
        mv = moving
        moved = np.where(mv, np.hypot(pos_new[..., 0] - self.a_pos[..., 0],
                                      pos_new[..., 1] - self.a_pos[..., 1]), 0.0)
        turn = np.abs(K.wrap180(hdg_new - self.a_hdg))   # [N,P] 이번 step 선회량(deg)
        ev["turn_sum"] += np.where(mv, turn, 0.0)         # 부드러움 품질용 누적
        self.a_pos = np.where(mv[..., None], pos_new, self.a_pos)
        self.a_hdg = np.where(mv, hdg_new, self.a_hdg)

        # painting
        self._paint(painting)
        self.paint_dist += np.where(painting, moved, 0.0)
        ev["path_dist"] += moved.sum(axis=1)
        ev["traveled"] += moved                          # 배별 이동거리 (효율용)

        # ── 그물 배치 shaping 추적: interception 정렬 ─────────────────────
        #   '적 근처'(게임가능 프록시)가 아니라 '적→모선 경로를 막는가'를 보상.
        #   포획 메커니즘(적이 그물 띠를 가로질러야 잡힘)과 일치. 그물 중점이 적의 진입
        #   코리도(적과 모선 사이, 0<t<1) 위에 있고 경로선에 가까울수록(perp↓) ↑.
        if painting.any():
            Mn = 0.5 * (self.net_start + self.net_end)            # 그물 중점 [N,P,2]
            C = self.center                                       # [2]
            E = self.e_pos                                        # [N,M,2]
            v = C[None, None, :] - E                              # 적→모선 [N,M,2]
            L = np.hypot(v[..., 0], v[..., 1]) + 1e-6            # [N,M]
            u = v / L[..., None]                                  # 단위 [N,M,2]
            rel = Mn[:, :, None, :] - E[:, None, :, :]            # 중점-적 [N,P,M,2]
            ub = u[:, None, :, :]                                 # [N,1,M,2]
            dot = (rel * ub).sum(-1)                              # 경로축 투영길이 [N,P,M]
            tnorm = dot / L[:, None, :]                           # 진행도(0=적,1=모선) [N,P,M]
            perp = np.linalg.norm(rel - dot[..., None] * ub, axis=-1)   # 경로선 수직거리 [N,P,M]
            gate = (np.clip((tnorm - 0.05) / 0.15, 0.0, 1.0)
                    * np.clip((1.0 - tnorm) / 0.15, 0.0, 1.0))    # 코리도(0<t<1) 게이트
            score = gate * np.exp(-perp / self.rcfg.place_scale)  # [N,P,M]
            score = np.where(self.e_alive[:, None, :], score, 0.0)
            # ★ 분업: 배는 '자기 배정 클러스터'의 적 경로만 막은 것으로 카운트
            member = ((self._ebin[:, None, :] == self._assign[:, :, None])
                      & (self._assign[:, :, None] >= 0))          # [N,P,M]
            score = np.where(member, score, 0.0)
            sscore = score.sum(2)                                 # 막은 적 경로 합 [N,P]
            ev["place_score"] = np.where(painting,
                                         np.maximum(ev["place_score"], sscore),
                                         ev["place_score"])

        # 도착/완성 처리: 목표WP 도착 OR 길이한계 → 그물 완성 → 그 구간을 '설치(installed)' 등록
        #   ★ 순차 부설: 도착으론 완성 안 함(다음 셀로 계속) → net_max_len 소진 때만 완성
        seq = getattr(cfg, "cell_sequential", False)
        finish = self.doing_net & ((self.paint_dist >= cfg.net_max_len) | (arrived & (not seq)))
        if finish.any():
            self.doing_net &= ~finish
            self._rasterize_net(self.net_start, self.net_end, finish)  # 설치 그물 셀 등록
            # 방금 완성한 아군은 자기 그물 끝점 위에 있음 → 이번 step 접촉 패널티 면제
            self.prev_on_inst |= finish
        # WP 도착 → 도달기록(동결) + 다음 WP (일방통행: Kw-1 에서 멈춤, 순환 X)
        arr_route = (~self.doing_net) & arrived
        advance = arr_route & (self.ptr < Kw - 1)
        if arr_route.any():
            ai, aj = np.where(arr_route)
            self.wp_reached[ai, aj, ptr_c[ai, aj]] = True   # 도달한 WP 동결
        self.ptr = np.where(advance, self.ptr + 1, self.ptr)
        self.leg_netted = np.where(advance, False, self.leg_netted)   # 새 구간 = 그물 재시작 가능

        # 적 전진 (alive 만 반영) + ★ 적응형 그물 회피(evade)
        ep = self.e_pos.reshape(-1, 2); eh = self.e_hdg.reshape(-1)
        eph = self.e_phase.reshape(-1)
        evade = self._enemy_evade().reshape(-1)
        tgt = np.broadcast_to(self.center, (N * M, 2))
        ep2, eh2 = K.enemy_step(ep, eh, tgt, cfg.enemy_speed, cfg.enemy_max_turn,
                                self._wt, weave_amp=cfg.enemy_weave_amp,
                                weave_period=cfg.enemy_weave_period, phase=eph, dt=cfg.dt,
                                evade=evade)
        ep2 = ep2.reshape(N, M, 2); eh2 = eh2.reshape(N, M)
        al = self.e_alive
        self.e_pos = np.where(al[..., None], ep2, self.e_pos)
        self.e_hdg = np.where(al, eh2, self.e_hdg)
        self._wt += 1

        # 포획 (painted cell 진입)
        eci = np.clip((self.e_pos[..., 0] // self.cell).astype(np.int64), 0, self.G - 1)
        ecj = np.clip((self.e_pos[..., 1] // self.cell).astype(np.int64), 0, self.G - 1)
        on_paint = self.painted[np.arange(N)[:, None], eci, ecj]
        cap = on_paint & self.e_alive
        if cap.any():
            self.e_alive &= ~cap
            ev["captures"] += cap.sum(axis=1)

        # breach (모선 반경)
        ed = np.hypot(self.e_pos[..., 0] - self.center[0], self.e_pos[..., 1] - self.center[1])
        br = self.e_alive & (ed <= cfg.mothership_radius)
        if br.any():
            self.e_alive &= ~br
            ev["breaches"] += br.sum(axis=1)

        # ── 충돌 → 큰 패널티 + 비활성화 ──
        newly_dead = np.zeros((N, P), bool)
        if P >= 2:                                       # 아군-아군 (둘 다 비활성화)
            r = cfg.ally_collision_radius
            for a in range(P):
                for b in range(a + 1, P):
                    dd = np.hypot(self.a_pos[:, a, 0] - self.a_pos[:, b, 0],
                                  self.a_pos[:, a, 1] - self.a_pos[:, b, 1])
                    hit = (dd < r) & self.a_alive[:, a] & self.a_alive[:, b]
                    newly_dead[:, a] |= hit; newly_dead[:, b] |= hit
                    ev["ally_collisions"] += hit.astype(np.float64)
        # 아군-모선(항공모함) 충돌
        md = np.hypot(self.a_pos[..., 0] - self.center[0],
                      self.a_pos[..., 1] - self.center[1])
        hitm = (md < cfg.ally_mother_radius) & self.a_alive
        ev["obstacle_collisions"] += hitm.sum(axis=1)
        newly_dead |= hitm

        # ── 설치된 그물 접촉 → 비활+패널티 (자기/타 아군 그물 무관) ──
        #   진입 edge 검출: 활성 아군이 '직전엔 안 닿았는데 지금 닿음'.
        #   ★ 전개중(doing_net)이어도 면제하지 않는다 → '다른 아군이 깐 그물'에도 걸린다.
        #     자기 그물도 무관: 옛 자기 그물에 닿으면 죽는다(면제 없음). 단 '지금 깔고 있는'
        #     현재 그물만 예외 — 완성 전까지 net_installed 에 없고, 완성 step 은 prev_on_inst|=finish
        #     로 면제되어 완성 즉시 자살하지 않게 한다.
        aci = np.clip((self.a_pos[..., 0] // self.cell).astype(np.int64), 0, self.G - 1)
        acj = np.clip((self.a_pos[..., 1] // self.cell).astype(np.int64), 0, self.G - 1)
        on_inst = self.net_installed[np.arange(N)[:, None], aci, acj] & self.a_alive
        net_touch = on_inst & (~self.prev_on_inst)
        ev["net_touches"] += net_touch.sum(axis=1)
        newly_dead |= net_touch
        self.prev_on_inst = on_inst                      # 다음 step 진입검출용

        if newly_dead.any():                             # 비활성화 (정지·그물중단)
            self.a_alive &= ~newly_dead
            self.doing_net &= ~newly_dead

        # 시간 / 종료
        self.t += 1
        self.done |= (~self.e_alive.any(axis=1)) | (self.t >= cfg.max_steps)

    def fresh_ev(self):
        """결정 윈도우 1회용 이벤트 누적 dict (micro-step 단위 구동/시각화에서도 재사용)."""
        ev = {k: np.zeros(self.N) for k in
              ("captures", "breaches", "ally_collisions", "obstacle_collisions",
               "nets_used", "path_dist", "net_touches")}
        ev["traveled"] = np.zeros((self.N, self.P))
        ev["turn_sum"] = np.zeros((self.N, self.P))      # 윈도우 누적 선회량 Σ|Δhdg| (부드러움 품질)
        ev["place_score"] = np.zeros((self.N, self.P))   # interception 정렬 배치 점수(최대 누적)
        ev["deployed"] = np.zeros((self.N, self.P), bool)
        ev["pos_start"] = self.a_pos.copy()
        return ev

    def _roll(self, period):
        ev = self.fresh_ev()
        for _ in range(period):
            self._micro(ev)
        ev["pos_end"] = self.a_pos.copy()
        return ev

    def _phi(self):
        return RW.threat_potential_vec(self.e_pos, self.e_alive, self.center,
                                       self.world_half, self.rcfg)

    def _enemy_evade(self):
        """적 적응형 그물 회피 [N,M]: 헤딩 전방(look)에 설치그물 있으면 트인 쪽으로 측면 조향(deg).
        정적 부채꼴을 뚫리게 만들어 RL의 반응·예측 우위 무대를 만든다. 전방 막힘 없으면 0."""
        cfg = self.cfg
        if not (cfg.enemy_evade and self.net_installed.any() and self.e_alive.any()):
            return np.zeros((self.N, self.M))
        L = cfg.enemy_evade_look; rows = np.arange(self.N)[:, None]
        h = np.deg2rad(self.e_hdg)                                    # [N,M]

        def blocked(off_deg):
            a = h + np.deg2rad(off_deg)
            px = self.e_pos[..., 0] + np.sin(a) * L
            py = self.e_pos[..., 1] + np.cos(a) * L
            ci = np.clip((px / self.cell).astype(np.int64), 0, self.G - 1)
            cj = np.clip((py / self.cell).astype(np.int64), 0, self.G - 1)
            return self.net_installed[rows, ci, cj]                   # [N,M]

        ahead = blocked(0.0); left = blocked(-cfg.enemy_evade_deg); right = blocked(cfg.enemy_evade_deg)
        evade = np.zeros((self.N, self.M))
        sl = ahead & (~left)                                         # 좌측 트임 → 좌로
        sr = ahead & (~right) & (~sl)                                # 우측 트임 → 우로
        evade = np.where(sl, -cfg.enemy_evade_deg, evade)
        evade = np.where(sr, cfg.enemy_evade_deg, evade)
        return evade * self.e_alive                                   # 죽은 적 0

    def _coverage_vec(self):
        """레이캐스트 coverage [N]∈[0,1]: 적→모선 ray 가 painted 그물에 막힌 alive 적 비율.
        0=전부 뚫림(방어 실패), 1=전부 차단(잡힐 예정). 포획의 선행지표(dense 보상)."""
        if not self.painted.any() or not self.e_alive.any():
            return np.zeros(self.N)
        N = self.N; c = np.asarray(self.center); e = self.e_pos
        T = max(2, int(self.rcfg.coverage_rays)); ts = np.linspace(0.05, 0.95, T)
        pts = e[:, :, None, :] + (c[None, None, None, :] - e[:, :, None, :]) * ts[None, None, :, None]
        rci = np.clip((pts[..., 0] / self.cell).astype(np.int64), 0, self.G - 1)
        rcj = np.clip((pts[..., 1] / self.cell).astype(np.int64), 0, self.G - 1)
        blocked = self.painted[np.arange(N)[:, None, None], rci, rcj].any(2)   # [N,M] ray 막힘?
        al = self.e_alive
        return (blocked & al).sum(1) / np.maximum(al.sum(1), 1)               # [N] 막힌 적 비율

    def _raycast_cover_mat(self):
        """단일커버 레이캐스트 원시행렬: 적→모선 ray vs 아군 **계획경로(route) leg** 선분교차.
        반환: cross_p[N,M,P](배 p가 적 m의 ray를 막나), net_cross_p[N,M,P](그물 leg로 막나), al[N,M].
        (보상·시각화 공용 — env 와 렌더가 동일 기하를 쓰도록.)"""
        N, P = self.N, self.P
        e = self.e_pos                                  # [N,M,2]
        al = self.e_alive                               # [N,M]
        c = np.asarray(self.center)                     # [2]
        r0 = self.route[:, :, :-1, :]                   # [N,P,L,2] leg 시작 WP
        r1 = self.route[:, :, 1:, :]                    # [N,P,L,2] leg 끝 WP
        leg_net = self.net_mask[:, :, 1:]               # [N,P,L] leg(k→k+1)=그물? (도착 WP 기준)
        p1 = e[:, :, None, None, :]                     # [N,M,1,1,2] ray 시작(적)
        p2 = c[None, None, None, None, :]               # [1,1,1,1,2] ray 끝(모선)
        p3 = r0[:, None, :, :, :]                       # [N,1,P,L,2]
        p4 = r1[:, None, :, :, :]                       # [N,1,P,L,2]
        cr = lambda ax, ay, bx, by: ax * by - ay * bx  # 2D 외적
        gx = p4[..., 0] - p3[..., 0]; gy = p4[..., 1] - p3[..., 1]            # leg 방향
        d1 = cr(gx, gy, p1[..., 0] - p3[..., 0], p1[..., 1] - p3[..., 1])
        d2 = cr(gx, gy, p2[..., 0] - p3[..., 0], p2[..., 1] - p3[..., 1])
        hx = p2[..., 0] - p1[..., 0]; hy = p2[..., 1] - p1[..., 1]            # ray 방향
        d3 = cr(hx, hy, p3[..., 0] - p1[..., 0], p3[..., 1] - p1[..., 1])
        d4 = cr(hx, hy, p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1])
        inter = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))               # [N,M,P,L] proper 교차
        inter = inter & self.a_alive[:, None, :, None]                        # 죽은 배 제외
        cross_p = inter.any(3)                                                # [N,M,P]
        net_cross_p = (inter & leg_net[:, None, :, :]).any(3)                 # [N,M,P]
        return cross_p, net_cross_p, al

    def _ray_net_blocked(self):
        """적→모선 ray 가 **실제 설치/도색된 그물(painted)** 에 막혔나 [N,M] bool.
        (계획 route 가 아니라 이미 깔린 그물 — 실제 방어 성공의 source of truth.)"""
        N = self.N; e = self.e_pos
        if not self.painted.any():
            return np.zeros((N, e.shape[1]), bool)
        c = np.asarray(self.center)
        T = max(2, int(self.rcfg.coverage_rays)); ts = np.linspace(0.05, 0.95, T)
        pts = e[:, :, None, :] + (c[None, None, None, :] - e[:, :, None, :]) * ts[None, None, :, None]
        rci = np.clip((pts[..., 0] / self.cell).astype(np.int64), 0, self.G - 1)
        rcj = np.clip((pts[..., 1] / self.cell).astype(np.int64), 0, self.G - 1)
        return self.painted[np.arange(N)[:, None, None], rci, rcj].any(2)     # [N,M]

    def _raycast_cover_detail(self):
        """단일커버 판정 상세(보상·시각화 공용).
        적→모선 ray 를 아군 **그물 레그(net wall)** 가 가로지르는지 + 그 교차점을 구해,
        한 ray 를 막은 그물벽이 **충돌거리(raycov_collide_m)만큼 가까운 2대**면 redundant(무효).
        반환: nb[N,M](막은 배 수), collide[N,M](충돌근접 중복?), al[N,M].
        (다층 방어=서로 먼 그물벽은 redundant 아님 — wave 정상 방어 보호.)"""
        N, P = self.N, self.P
        e = self.e_pos; al = self.e_alive                  # [N,M,2],[N,M]
        c = np.asarray(self.center)
        r0 = self.route[:, :, :-1, :]; r1 = self.route[:, :, 1:, :]   # [N,P,L,2]
        leg_net = self.net_mask[:, :, 1:]                  # [N,P,L] 그물 레그
        p1 = e[:, :, None, None, :]                        # [N,M,1,1,2] ray 시작(적)
        p2 = c[None, None, None, None, :]                  # 모선
        p3 = r0[:, None, :, :, :]; p4 = r1[:, None, :, :, :]          # [N,1,P,L,2]
        cr = lambda ax, ay, bx, by: ax * by - ay * bx
        d1x = p2[..., 0] - p1[..., 0]; d1y = p2[..., 1] - p1[..., 1]  # ray 방향
        d2x = p4[..., 0] - p3[..., 0]; d2y = p4[..., 1] - p3[..., 1]  # leg 방향
        denom = cr(d1x, d1y, d2x, d2y)                     # [N,M,P,L]
        d1 = cr(d2x, d2y, p1[..., 0] - p3[..., 0], p1[..., 1] - p3[..., 1])
        d2 = cr(d2x, d2y, p2[..., 0] - p3[..., 0], p2[..., 1] - p3[..., 1])
        d3 = cr(d1x, d1y, p3[..., 0] - p1[..., 0], p3[..., 1] - p1[..., 1])
        d4 = cr(d1x, d1y, p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1])
        inter = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))       # proper 교차
        inter = inter & leg_net[:, None, :, :] & self.a_alive[:, None, :, None]   # 그물 레그 + 살아있는 배
        # ray 위 교차 파라미터 t (0=적,1=모선) → 교차점
        t = np.where(np.abs(denom) > 1e-9, d3 / np.where(denom == 0, 1.0, denom), np.inf)
        t = np.where(inter, t, np.inf)                     # 비교차=inf
        tmin = t.min(3)                                    # [N,M,P] 배별 최근접 교차 t
        blk = np.isfinite(tmin)                            # [N,M,P] 배 p 가 그물로 막음?
        px = e[..., 0][..., None] + np.where(blk, tmin, 0.0) * (c[0] - e[..., 0][..., None])
        py = e[..., 1][..., None] + np.where(blk, tmin, 0.0) * (c[1] - e[..., 1][..., None])   # [N,M,P]
        nb = blk.sum(2)                                    # [N,M] 막은 배 수
        # 충돌근접: 두 막은 배의 교차점이 raycov_collide_m 이내
        ddx = px[..., :, None] - px[..., None, :]; ddy = py[..., :, None] - py[..., None, :]
        dist = np.hypot(ddx, ddy)                          # [N,M,P,P]
        pair = blk[..., :, None] & blk[..., None, :] & ~np.eye(P, dtype=bool)[None, None]
        collide = (pair & (dist < self.rcfg.raycov_collide_m)).any((2, 3))     # [N,M]
        return nb, collide, al

    def _raycast_single_cover_vec(self):
        """★ 단일커버 레이캐스트 보상 [N].
        적→모선 ray 를 아군 **그물벽**이 막으면 커버(raycov_net_bonus 우대). 단, 두 그물벽이
        **충돌거리만큼 가까이** 같은 ray 를 막으면 redundant→무효(0): 경로 겹침=충돌 예정.
        (다층 방어=서로 먼 그물벽은 정상 커버로 인정 → wave 방어 보호.)"""
        if not self.e_alive.any():
            return np.zeros(self.N)
        nb, collide, al = self._raycast_cover_detail()     # 계획 net-leg(배별·충돌판정)
        net_painted = self._ray_net_blocked()              # 실제 설치 그물(persistent)
        covered = ((nb >= 1) | net_painted) & (~collide) & al   # 합집합 커버, 충돌근접만 무효
        val = np.where(covered, self.rcfg.raycov_net_bonus, 0.0)
        return val.sum(1) / np.maximum(al.sum(1), 1.0)     # [N] 살아있는 적당 평균

    def _reward(self, ev, phi0, phi1, period):
        rc = self.rcfg
        denom = max(period * self.cfg.ally_speed * self.P, 1.0)
        pdn = ev["path_dist"] / denom
        n_rem = self.e_alive.sum(axis=1)
        base = RW.window_reward_vec(
            ev["captures"], ev["breaches"], ev["ally_collisions"], ev["nets_used"],
            pdn, phi0, phi1, self.done, n_rem, rc,
            steps=period, nets_total=self.nets_total,
            obstacle_collisions=ev["obstacle_collisions"])
        # ── dense 그물 배치 shaping: 적 진입경로를 막을수록 ↑ (interception 정렬) ──
        place = np.where(ev["deployed"], ev["place_score"], 0.0).sum(axis=1)  # [N]
        # ── 경로 효율: 직선변위/이동거리 (1=직진, ↓=배회) ──
        disp = np.hypot(ev["pos_end"][..., 0] - ev["pos_start"][..., 0],
                        ev["pos_end"][..., 1] - ev["pos_start"][..., 1])
        eff = np.clip(disp / (ev["traveled"] + 1e-6), 0.0, 1.0).mean(axis=1)             # [N]
        # ── 설치된 그물 접촉 패널티 (설치 그물 쪽으로 가지 않게) ──
        net_touch = rc.r_net_touch * ev["net_touches"]                                   # [N]
        # ── 배정 기반 dense: (A)경로가 배정 클러스터로 향함 + (C)미배정 배 정지(예비) ──
        assign = self._assign                                  # [N,P] 클러스터 idx(-1=예비)
        assigned = assign >= 0                                 # [N,P]
        maxmove = max(period * self.cfg.ally_speed, 1.0)
        I = self._assignI                                      # [N,P,2] 배정 교점
        ps = ev["pos_start"]; pe = ev["pos_end"]
        d0 = np.hypot(ps[..., 0] - I[..., 0], ps[..., 1] - I[..., 1])     # 시작 거리
        d1 = np.hypot(pe[..., 0] - I[..., 0], pe[..., 1] - I[..., 1])     # 끝 거리
        approach = np.where(assigned, np.clip((d0 - d1) / maxmove, -1.0, 1.0), 0.0).sum(1)
        idle = np.where(~assigned, 1.0 - np.clip(ev["traveled"] / maxmove, 0.0, 1.0), 0.0)
        idle = (idle * self.a_alive).sum(1)                    # 살아있는 미배정 배만
        # ── ★ 충돌 회피 dense 척력(APF): 가까워질수록 페널티 ↑ → 충돌 '전에' 떨어짐 ──
        av = self.a_alive; c = self.center
        # 아군-아군: 끝 위치 쌍거리. ½·η·(R/d−1)² (d<R), 자기·죽은배 제외.
        dx = pe[:, :, None, 0] - pe[:, None, :, 0]
        dy = pe[:, :, None, 1] - pe[:, None, :, 1]
        dd = np.hypot(dx, dy)                                   # [N,P,P]
        both = av[:, :, None] & av[:, None, :] & ~np.eye(self.P, dtype=bool)[None]
        ddc = np.maximum(dd, rc.avoid_dmin)
        arep = np.where(both & (dd < rc.avoid_r),
                        0.5 * rc.avoid_eta * (rc.avoid_r / ddc - 1.0) ** 2, 0.0)
        ally_rep = (arep.sum(2) * av).sum(1)                   # [N] 각 배의 타선 척력 합
        # 아군-모선: death-disk 가장자리 밖 거리로 같은 척력.
        dm = np.hypot(pe[..., 0] - c[0], pe[..., 1] - c[1]) - self.cfg.ally_mother_radius
        dmc = np.maximum(dm, rc.avoid_dmin)
        mrep = np.where((dm < rc.mother_avoid_r) & av,
                        0.5 * rc.avoid_eta * (rc.mother_avoid_r / dmc - 1.0) ** 2, 0.0)
        mother_rep = (mrep * av).sum(1)                        # [N]
        obstacle = ally_rep + rc.mother_avoid_w * mother_rep   # [N]
        # ── ★ 궤적 품질(정책 niche): 부드러움 + 그물 연속마진 ──
        # 부드러움: 윈도우 누적 선회량 / (최대 가능 선회) ∈[0,1] → 페널티 (지그재그↓)
        smooth = (ev["turn_sum"] / max(period * self.cfg.ally_max_turn, 1.0)).mean(axis=1)  # [N]
        # 그물 연속마진: 끝 위치가 설치 그물에 가까울수록 페널티(충돌반경 밖 buffer; 작은 충돌 사전방지)
        net_clear = np.zeros(self.N)
        if rc.w_clear_net > 0.0 and self.net_installed.any():
            G = self.G; cell = self.cell; Rc = rc.clear_net_r
            wr = int(np.ceil(Rc / cell)); offs = np.arange(-wr, wr + 1)
            ci = np.clip((pe[..., 0] // cell).astype(np.int64), 0, G - 1)
            cj = np.clip((pe[..., 1] // cell).astype(np.int64), 0, G - 1)
            ii = np.clip(ci[..., None, None] + offs[None, None, :, None], 0, G - 1)   # [N,P,W,1]
            jj = np.clip(cj[..., None, None] + offs[None, None, None, :], 0, G - 1)   # [N,P,1,W]
            nidx = np.arange(self.N)[:, None, None, None]
            occ = self.net_installed[nidx, ii, jj]                                   # [N,P,W,W]
            dxg = pe[..., 0][..., None, None] - (ii + 0.5) * cell
            dyg = pe[..., 1][..., None, None] - (jj + 0.5) * cell
            dg = np.where(occ, np.hypot(dxg, dyg), np.inf)
            dmin = dg.min((2, 3))                                                     # [N,P] 최근접 그물거리
            prox = np.clip((Rc - dmin) / Rc, 0.0, 1.0)                                # 1=그물 위, 0=≥Rc
            net_clear = (prox * av).sum(1) / np.maximum(av.sum(1), 1.0)               # [N] 살아있는 배 평균 (champion 조건)
        # 시간 일관성: 이번 결정 route 가 직전 결정 route 와 얼마나 다른가 (작을수록 일관)
        dprev = np.hypot(self.route[..., 0] - self._prev_route[..., 0],
                         self.route[..., 1] - self._prev_route[..., 1])               # [N,P,Kw]
        consist = ((dprev.mean(2) * av).sum(1) / np.maximum(av.sum(1), 1.0)
                   / max(self.cfg.wp_max_len, 1.0))                                   # [N] 정규화
        # ── ★ 레이캐스트 coverage: 적→모선 ray가 설치/도색 그물에 막힌 적 비율 (방어 성공 직접 척도) ──
        coverage = self._coverage_vec() if rc.w_coverage > 0.0 else np.zeros(self.N)  # [N]∈[0,1]
        # ── ★ 단일커버 레이캐스트: 적 ray를 '정확히 1대' 아군경로/그물이 막을 때만 보상(중복=무효) ──
        raycov = self._raycast_single_cover_vec() if rc.w_raycov > 0.0 else np.zeros(self.N)  # [N]
        # ── ★ 잔차 앵커: wp 잔차 크기 페널티 → 휴리스틱(잔차0)을 바닥으로(drift 방지) ──
        rmag = getattr(self, "_resid_mag", np.zeros((self.N, self.P)))
        anchor = (rmag * av).sum(1) / np.maximum(av.sum(1), 1.0)          # [N]∈[0,1] 살아있는 배 평균
        total = (base + rc.w_place * place + rc.w_eff * eff + net_touch
                 + rc.w_assign_path * approach + rc.w_idle * idle
                 + rc.w_coverage * coverage                    # ★ 레이캐스트 coverage(방어 성공)
                 + rc.w_raycov * raycov                        # ★ 단일커버 레이캐스트(비중복 협조)
                 - rc.w_anchor * anchor                        # ★ 잔차 앵커(휴리스틱 trust-region)
                 - rc.w_obstacle * obstacle                    # 충돌 회피 척력
                 - rc.w_smooth * smooth - rc.w_clear_net * net_clear   # ★ 품질: 부드러움·그물마진
                 - rc.w_consist * consist)                     # ★ 시간 일관성(경로 튐 억제)
        # ── 진단: 항목별 기여(월드 평균). 'shaping이 event를 압도하나' 확인용 ──
        self._rwd_breakdown = {
            "capture": float((rc.r_capture * ev["captures"]).mean()),
            "breach":  float((rc.r_breach * ev["breaches"]).mean()),
            "coll":    float((rc.r_ally_collision * ev["ally_collisions"]
                              + rc.r_obstacle * ev["obstacle_collisions"]).mean()),
            "net_touch": float(net_touch.mean()),
            "phi":     float((rc.gamma * phi1 - phi0).mean()),
            "place":   float((rc.w_place * place).mean()),
            "approach": float((rc.w_assign_path * approach).mean()),
            "idle":    float((rc.w_idle * idle).mean()),
            "eff":     float((rc.w_eff * eff).mean()),
            "smooth":  float((-rc.w_smooth * smooth).mean()),
            "clear_net": float((-rc.w_clear_net * net_clear).mean()),
            "consist": float((-rc.w_consist * consist).mean()),
            "coverage": float((rc.w_coverage * coverage).mean()),
            "raycov": float((rc.w_raycov * raycov).mean()),
            "anchor": float((-rc.w_anchor * anchor).mean()),
            "netcost": float((-rc.w_net * (ev["nets_used"] / max(self.nets_total, 1))
                              - rc.w_path * pdn).mean()),
            "obstacle": float((-rc.w_obstacle * obstacle).mean()),
            "total":   float(total.mean()),
        }
        return total

    # ── 클러스터→배 배정 (위협 큰 클러스터를 가장 가까운 배에 1:1) ──────
    def _compute_assignment(self, assign_pref=None):
        """매 결정 시작 상태에서 위협 큰 클러스터 top-min(#active,P)개를 골라
        '가장 가까운(효율적) 살아있는 배'에 1:1 그리디 배정한다.
        ★ assign_pref[N,P](정책 선호 클러스터 idx) 주면 그 cost 를 w_assign_bias 만큼 차감(b: soft-bias).
        결과: self._assign[N,P](클러스터 idx, -1=예비), self._assignI[N,P,2](교점),
              self._ebin[N,M](적 시작 클러스터 멤버십, block 보상 마스킹용).
        결정 시작 상태(배·적 위치)만의 함수라 모든 후보 롤아웃에서 동일."""
        cfg = self.cfg; N, P, K = self.N, self.P, cfg.n_clusters
        c = self.center
        cl = clustering.cluster_by_gaps_vec(self.e_pos, self.e_alive, self.e_hdg,
                                            c, cfg.enemy_speed, K, cfg.cluster_gap_deg)
        cent = cl["centroid"]; cnt = cl["count"]; active = cl["active"]   # [N,K,2],[N,K]
        t = self.rcfg.assign_intercept_t
        I = cent + t * (c[None, None, :] - cent)                          # [N,K,2] 교점
        md = np.hypot(cent[..., 0] - c[0], cent[..., 1] - c[1])           # 클러스터-모선 거리
        close = np.clip(1.0 - md / self.world_half, 0.05, 1.0)            # 가까울수록 ↑
        threat = np.where(active, cnt * close, -1.0)                      # [N,K] 위협도
        # 커버 대상 = 위협 상위 P개(활성만) — 1:1(한 클러스터당 한 배)
        topP = np.argsort(-threat, axis=1)[:, :P]                         # [N,P] 클러스터 idx
        target = np.zeros((N, K), bool)
        target[np.arange(N)[:, None], topP] = True
        target &= active
        # 비용 = 배→교점 거리 (낮을수록 효율적). 비대상 클러스터·비활성 배 = 제외.
        cost = np.hypot(self.a_pos[:, :, None, 0] - I[:, None, :, 0],
                        self.a_pos[:, :, None, 1] - I[:, None, :, 1])     # [N,P,K]
        BIG = 1e18
        cost = np.where(target[:, None, :], cost, BIG)
        cost = np.where(self.a_alive[:, :, None], cost, BIG)
        # ── ★ 배정 안정화: 직전 클러스터가 아직 타겟이면 그 쌍 cost 를 낮춰 유지(churn↓) ──
        if self.rcfg.assign_sticky_bonus > 0.0:
            pa = self._prev_assign                                        # [N,P] 직전 클러스터(-1=없음)
            pac = np.clip(pa, 0, K - 1)
            prev_is_target = np.take_along_axis(target, pac, axis=1)      # [N,P] 직전 클러스터가 타겟?
            keep = (pa >= 0) & prev_is_target & self.a_alive
            rn, rp = np.where(keep)
            if len(rn):
                cur = cost[rn, rp, pac[rn, rp]]
                cost[rn, rp, pac[rn, rp]] = np.where(
                    cur < BIG / 2, cur - self.rcfg.assign_sticky_bonus, cur)
        # ── ★ 배정 학습(b): 정책 선호 클러스터의 cost 차감 (휴리스틱 배정 base 위 soft-bias) ──
        wab = getattr(self.rcfg, "w_assign_bias", 0.0)
        if assign_pref is not None and wab > 0.0:
            apf = np.clip(np.asarray(assign_pref, np.int64), 0, K - 1)        # [N,P]
            rn3, rp3 = np.where(self.a_alive)
            kk3 = apf[rn3, rp3]
            cur3 = cost[rn3, rp3, kk3]
            cost[rn3, rp3, kk3] = np.where(cur3 < BIG / 2, cur3 - wab, cur3)  # 선호 클러스터 우대
        assign = np.full((N, P), -1, np.int64)
        assignI = np.zeros((N, P, 2))
        work = cost.copy()
        rows = np.arange(N)
        for _ in range(min(P, K)):                                       # 그리디 최소비용 매칭(1:1)
            flat = work.reshape(N, P * K)
            j = np.argmin(flat, axis=1)
            v = flat[rows, j]
            pi = j // K; ki = j % K
            ok = v < BIG / 2
            if not ok.any():
                break
            sel = np.where(ok)[0]
            assign[sel, pi[sel]] = ki[sel]
            assignI[sel, pi[sel]] = I[sel, ki[sel]]
            work[sel, pi[sel], :] = BIG                                  # 쓴 배 행 제거
            work[sel, :, ki[sel]] = BIG                                  # 쓴 클러스터 열 제거(1:1)
        # 적 시작 클러스터 멤버십 (block 보상에서 자기 클러스터 적만 카운트)
        dx = self.e_pos[..., 0] - c[0]; dy = self.e_pos[..., 1] - c[1]
        brg = np.degrees(np.arctan2(dx, dy)) % 360.0
        self._ebin = (np.floor(brg / (360.0 / K)).astype(np.int64)) % K   # [N,M]
        self._assign = assign
        self._assignI = assignI
        # 배정 클러스터 centroid (휴리스틱 후보 경로용; 미배정=모선)
        ci = np.clip(assign, 0, K - 1)
        gathered = np.take_along_axis(cent, ci[..., None], axis=1)        # [N,P,2]
        self._assign_cent = np.where((assign >= 0)[..., None], gathered, c[None, None, :])

    # ── 휴리스틱을 actor 액션으로 (GRPO 후보 주입용 다리) ──────────────
    def heuristic_route_netgo(self, fan=None):
        """현재 상태 → 휴리스틱 부채꼴 경로(월드좌표 [N,P,Kw,2]) + **레그별** net_mask[N,P,Kw].
        배정 배: WP0=부채꼴 시작(r_near, transit) → 이후 leg 들이 near→far 그물(측면 스윕).
          ★ leg0(배→WP0)=transit(mask=0), leg1..nnet=그물(mask=1) → 첫 net 낭비 방지(1-b).
        미배정 배: 정지(현 위치), mask 전부 0.
        ★ fan[N,P,7]=(bearing,r_near,r_far,spread,aux_radial,aux_lateral,curve)∈[-1,1] (structured_action):
          부채꼴을 fan 파라미터로 직접 변형(보조WP0 추가). fan=0 → 기본 휴리스틱(보조=시작점)."""
        cfg = self.cfg; N, P, Kw = self.N, self.P, self.Kw
        c = self.center
        assign = self._assign; assigned = assign >= 0          # [N,P]
        rel = self._assign_cent - c[None, None, :]             # [N,P,2]
        D = np.hypot(rel[..., 0], rel[..., 1])                 # [N,P]
        rad = rel / np.maximum(D, 1e-6)[..., None]             # 방위 단위
        if fan is not None and cfg.structured_action:          # ★ 부채꼴 7파라미터 디코드(별도 구조)
            return self._fan_route_netgo(fan, assigned, rad, D)
        perp = np.stack([-rad[..., 1], rad[..., 0]], -1)       # 측면축
        Rcap = self._R_FEAS * cfg.net_deploy_reach
        r_far = np.clip(cfg.net_deploy_frac * D, cfg.mothership_radius + cfg.net_standoff_far, Rcap)
        r_near = np.clip(cfg.net_deploy_near * D, cfg.mothership_radius + cfg.net_standoff_near, r_far)
        # 그물 leg 수 = min(자원, Kw-1). WP0..WP_nnet 가 부채꼴(near→far), 이후 WP 는 끝점 반복.
        nnet = int(min(cfg.nets_per_ship, Kw - 1))
        kk = np.minimum(np.arange(Kw), nnet).astype(np.float64)            # [Kw] 0..nnet,flat
        fr = (kk / max(nnet, 1))[None, None, :]                            # [1,1,Kw] 0..1
        radii = r_near[..., None] + (r_far - r_near)[..., None] * fr        # [N,P,Kw]
        edge = (kk - nnet / 2.0)[None, None, :] * (cfg.net_max_len * 0.7)   # 측면 스윕
        ship_lat = ((self.a_pos - c[None, None, :]) * perp).sum(-1)         # [N,P]
        edge = np.where((ship_lat > 0)[..., None], -edge, edge)            # 가까운 wing부터
        wp = (c[None, None, None, :] + rad[:, :, None, :] * radii[..., None]
              + perp[:, :, None, :] * edge[..., None])         # [N,P,Kw,2]
        wp = np.clip(wp, 0.0, cfg.world_size)
        hold = np.broadcast_to(self.a_pos[:, :, None, :], (N, P, Kw, 2))    # 예비=정지
        route = np.where(assigned[:, :, None, None], wp, hold)
        # 레그별 net 마스크: leg k 가 그물 ⇔ 1<=k<=nnet (leg0=transit). 배정 배만.
        leg_is_net = (np.arange(Kw) >= 1) & (np.arange(Kw) <= nnet)        # [Kw]
        net_mask = assigned[:, :, None] & leg_is_net[None, None, :]        # [N,P,Kw]
        return route, net_mask.astype(np.int64)

    def _fan_route_netgo(self, fan, assigned, rad, D):
        """부채꼴 7파라미터 fan[N,P,7] → route[N,P,Kw,2] + net_mask[N,P,Kw].
        구조(Kw): WP0=보조 transit(approach) + WP1=부채꼴시작 → WP2..=그물(near→far 측면스윕).
          그물 leg = 2..nnet+1. fan=0 → 기본 휴리스틱(보조=시작점). (structured-fan-action 로직 이식.)"""
        cfg = self.cfg; N, P, Kw = self.N, self.P, self.Kw; c = self.center
        fan = np.clip(np.asarray(fan, np.float64), -1.0, 1.0)
        a_b, a_rn, a_rf, a_sp = fan[..., 0], fan[..., 1], fan[..., 2], fan[..., 3]
        a_ar, a_al, a_cv = fan[..., 4], fan[..., 5], fan[..., 6]   # 보조WP 반경/측면, 곡률 [N,P]
        # [0] 방위 회전 (far≈2km 서 작은각=큰 요격점 이동, 고레버리지)
        brg = np.arctan2(rad[..., 0], rad[..., 1]) + np.deg2rad(a_b * cfg.fan_bearing_max)
        rad = np.stack([np.sin(brg), np.cos(brg)], -1)
        perp = np.stack([-rad[..., 1], rad[..., 0]], -1)
        # [1][2] near/far 반경 (독립 스케일)
        Rcap = self._R_FEAS * cfg.net_deploy_reach
        r_far = np.clip(cfg.net_deploy_frac * D * (1 + a_rf * cfg.fan_rfar_amp),
                        cfg.mothership_radius + cfg.net_standoff_far, Rcap)
        r_near = np.clip(cfg.net_deploy_near * D * (1 + a_rn * cfg.fan_rnear_amp),
                         cfg.mothership_radius + cfg.net_standoff_near_fan, r_far)
        Kf = Kw - 1
        nnet = int(min(cfg.nets_per_ship, Kf - 1))
        kk = np.minimum(np.arange(Kf), nnet).astype(np.float64)
        fr = (kk / max(nnet, 1))[None, None, :]                            # [1,1,Kf]
        r_mid = 0.5 * (r_near + r_far)                                     # [N,P]
        radii = (r_mid[..., None] + (r_near[..., None]
                 + (r_far - r_near)[..., None] * fr - r_mid[..., None]) * cfg.net_radial_frac)  # [N,P,Kf]
        # [3] 측면 스윕폭
        ew = cfg.net_max_len * 0.7
        edge = (kk - nnet / 2.0)[None, None, :] * ew * (1 + a_sp * cfg.fan_spread_amp)[..., None]
        ship_lat = ((self.a_pos - c[None, None, :]) * perp).sum(-1)        # [N,P]
        edge = np.where((ship_lat > 0)[..., None], -edge, edge)            # 가까운 wing 부터
        # [6] 곡률 (중심 peak 포물선)
        em = max((nnet / 2.0) * ew, 1.0)
        en = edge / em
        radii = radii + (a_cv[..., None] * cfg.fan_curve_max) * (1 - en * en)
        fan_xy = (c[None, None, None, :] + rad[:, :, None, :] * radii[..., None]
                  + perp[:, :, None, :] * edge[..., None])                 # [N,P,Kf,2] = WP1..WP_{Kw-1}
        # [4][5] 보조WP0 = 부채꼴 시작점 + 반경/측면 이동 (기본 0 = 시작점, 회귀 없음)
        aux = (fan_xy[:, :, 0, :] + rad * (a_ar * cfg.aux_radial_max)[..., None]
               + perp * (a_al * cfg.aux_lateral_max)[..., None])           # [N,P,2]
        wp = np.concatenate([aux[:, :, None, :], fan_xy], axis=2)          # [N,P,Kw,2]
        wp = np.clip(wp, 0.0, cfg.world_size)
        hold = np.broadcast_to(self.a_pos[:, :, None, :], (N, P, Kw, 2))   # 예비=정지
        route = np.where(assigned[:, :, None, None], wp, hold)
        leg_is_net = (np.arange(Kw) >= 2) & (np.arange(Kw) <= nnet + 1)    # leg 2..nnet+1 = 그물
        net_mask = assigned[:, :, None] & leg_is_net[None, None, :]        # [N,P,Kw]
        return route, net_mask.astype(np.int64)

    def routes_from_cont(self, cont, w=0):
        """★ 행동공간 시각화용: 후보 cont 잔차 [B,P,Kw,2](월드 w 기준)를 실제 route [B,P,Kw,2] 로
        디코드 — 상태 불변(self.route 안 건드림). heuristic_baseline 잔차 모드(independent/cumulative/
        rigid) 재현. 정책이 '생각할 수 있는 경로들'을 그려보려고 K샘플을 한 번에 변환."""
        cfg = self.cfg; Kw = self.Kw
        self._compute_assignment()
        base, _ = self.heuristic_route_netgo()                 # [N,P,Kw,2]
        if getattr(cfg, "moving_anchor", False):               # A) base 에 anchor(정책 EMA) 반영
            base = np.clip((1.0 - cfg.anchor_weight) * base
                           + cfg.anchor_weight * self.anchor_route, 0.0, cfg.world_size)
        base_w = base[w]                                       # [P,Kw,2]
        assigned = (self._assign[w] >= 0)                      # [P]
        cont = np.asarray(cont, np.float64)                    # [B,P,Kw,2]
        if cfg.wp_residual_mode == "rigid":
            delta = np.broadcast_to(cont.mean(axis=2, keepdims=True), cont.shape)
        elif cfg.wp_residual_mode == "cumulative":
            delta = np.cumsum(cont, axis=2)
        else:                                                  # independent
            delta = cont
        route = base_w[None] + delta * cfg.wp_adjust_max * assigned[None, :, None, None]
        return np.clip(route, 0.0, cfg.world_size), base_w, assigned

    def heuristic_action(self):
        """휴리스틱 경로 → actor 액션(cont [N,P,Kw,2], netgo[N,P,Kw] 레그별).
        route_init 상태별: 첫결정=egocentric 체인 역산, 이후=앵커 잔차. 단위원 클램프."""
        cfg = self.cfg; N, P, Kw = self.N, self.P, self.Kw
        fan0 = np.zeros((N, P, 7)) if cfg.structured_action else None
        route, netgo = self.heuristic_route_netgo(fan0)
        if cfg.heuristic_baseline:
            # baseline 모드: _apply_actions 가 base(=휴리스틱) 위에 잔차를 더함 → 휴리스틱 후보/BC =
            #   **0 잔차/fan**(=base 재현). fan=0 또는 잔차 0 → route=base.
            if cfg.structured_action:
                return np.zeros((N, P, 7)), netgo          # 부채꼴: fan=0 = 휴리스틱
            return np.zeros((N, P, Kw, 2)), netgo
        if cfg.absolute_replan:                                # 절대 모드: 항상 체인 역산
            first = np.ones((N, P, 1, 1), bool)
        else:
            first = (~self.route_init)[:, :, None, None]       # [N,P,1,1]
        prev = self.a_pos.copy()
        cont_first = np.zeros((N, P, Kw, 2))
        for k in range(Kw):                                    # 체인 역산: leg/wp_max_len
            cont_first[:, :, k, :] = (route[:, :, k, :] - prev) / cfg.wp_max_len
            prev = route[:, :, k, :]
        cont_res = (route - self.route_anchor) / cfg.wp_adjust_max   # 잔차 역산
        cont = np.where(first, cont_first, cont_res)
        nrm = np.linalg.norm(cont, axis=-1, keepdims=True)
        cont = cont / np.maximum(nrm, 1.0)                     # |cont_k|<=1
        return cont, netgo

    # ── step (실제 진행 + autoreset) ─────────────────────────────────
    def step(self, actions):
        period = self.cfg.decision_period
        self._apply_actions(actions)                     # 배정은 _apply_actions 내부서 계산
        phi0 = self._phi()
        ev = self._roll(period)
        phi1 = self._phi()
        reward = self._reward(ev, phi0, phi1, period)
        term = self.done.copy()
        trunc = (self.t >= self.cfg.max_steps) & ~(~self.e_alive.any(axis=1))
        info = {"captures": ev["captures"], "breaches": ev["breaches"],
                "ally_collisions": ev["ally_collisions"], "nets_used": ev["nets_used"],
                "obstacle_collisions": ev["obstacle_collisions"],
                "net_touches": ev["net_touches"], "n_alive": self.e_alive.sum(axis=1),
                "turn": ev["turn_sum"].sum(axis=1)}      # [N] 윈도우 총 선회량(deg, 부드러움 지표↓=매끈)
        self._prev_route = self.route.copy()             # 다음 결정 일관성 비교 기준(실제 step 만 갱신)
        # autoreset
        d = np.where(self.done)[0]
        if len(d):
            self._spawn_worlds(d)
        return self.build_obs(), reward, term, trunc, info

    # ── GRPO 반사실 K-롤아웃 (상태 불변) ─────────────────────────────
    def snapshot(self):
        keys = ["e_pos", "e_hdg", "e_phase", "e_alive", "a_pos", "a_hdg", "a_nets",
                "a_alive", "route", "route_anchor", "anchor_route", "wp_reached", "route_init",
                "net_mask", "leg_netted",
                "net_end", "net_start", "ptr", "wp_dir", "wp_dir_set",
                "_prev_assign", "_last_apply_ptr",
                "doing_net", "paint_dist", "fan_anchor",
                "prev_on_inst", "painted", "net_installed", "t", "done"]
        snap = {k: getattr(self, k).copy() for k in keys}
        snap["_wt"] = self._wt
        snap["rng"] = self.rng.bit_generator.state
        return snap

    def restore(self, snap):
        for k, v in snap.items():
            if k in ("_wt", "rng"):
                continue
            setattr(self, k, v.copy())
        self._wt = snap["_wt"]
        self.rng.bit_generator.state = snap["rng"]

    def rollout_eval(self, actions, period=None, force_fresh=None):
        """snapshot 상태에서 actions 로 period 굴려 보상[N]만 반환(restore 후 상태 불변).
        force_fresh[N,P]: 후보 계획 강제 커밋(경로커밋下 GRPO 신호 회복). 상태는 restore 로 불변."""
        period = period or self.cfg.decision_period
        snap = self.snapshot()
        self._apply_actions(actions, force_fresh=force_fresh)   # 배정은 내부서 계산
        phi0 = self._phi()
        ev = self._roll(period)
        phi1 = self._phi()
        reward = self._reward(ev, phi0, phi1, period)
        self.restore(snap)
        return reward

    # ── 관측 구성 (per-agent 로컬 set, BoatAttack 정규화) ────────────
    def _net_probe(self, fwd):
        """net radar: 각 배 헤딩기준 D방향 레이로 '설치그물(net_installed)까지 근접도' [N,P,D].
        레이를 cell 간격으로 표본해 첫(최단) 그물 셀 거리를 찾고 norm_close 로 정규화
        (가까울수록 1, 탐지범위 내 그물 없으면 0 → inf→0). 정책이 깐 그물을 보고 우회하도록.
        레이0=정면, 이후 360°/D 간격(헤딩 회전). net 없으면 즉시 0 (조기 step 비용 0)."""
        cfg = self.cfg; N, P, D = self.N, self.P, cfg.net_probe_dirs
        if D <= 0 or not self.net_installed.any():
            return np.zeros((N, P, max(D, 0)))
        rng = cfg.net_probe_range
        S = int(np.ceil(rng / self.cell)) + 1
        ds = np.linspace(self.cell, rng, S)                    # [S] 표본거리(간격≤cell→띠 관통 방지)
        th = (2.0 * np.pi / D) * np.arange(D)                  # [D] 헤딩기준 회전각
        cs, sn = np.cos(th), np.sin(th)                        # [D]
        fx, fy = fwd[..., 0], fwd[..., 1]                      # [N,P] 진행벡터(=heading_vec)
        dx = fx[..., None] * cs - fy[..., None] * sn           # [N,P,D] fwd 를 θ_k 회전
        dy = fx[..., None] * sn + fy[..., None] * cs
        px = self.a_pos[..., None, None, 0] + dx[..., None] * ds   # [N,P,D,S]
        py = self.a_pos[..., None, None, 1] + dy[..., None] * ds
        ci = np.clip((px // self.cell).astype(np.int64), 0, self.G - 1)
        cj = np.clip((py // self.cell).astype(np.int64), 0, self.G - 1)
        nidx = np.arange(N)[:, None, None, None]
        occ = self.net_installed[nidx, ci, cj]                 # [N,P,D,S] bool
        hit_d = np.where(occ, ds, np.inf).min(axis=-1)         # [N,P,D] 최단 그물거리(없으면 inf)
        return E.norm_close(hit_d, rng * 0.5)                  # inf→0, 0m→1 (가까울수록 위험)

    # ── ★ 셀선택 행동공간 (pointer) ─────────────────────────────────────
    def _cell_half(self):
        """반셀 크기(m) = 격자 간격/2. 하이브리드 오프셋 스케일. cartesian: r_max/(n-1)."""
        cfg = self.cfg
        if getattr(cfg, "cell_grid", "polar") == "cartesian":
            half = float(cfg.cell_r_max) / max(int(cfg.cell_cart_n) - 1, 1)
        else:
            half = float(getattr(cfg, "cell_spacing", 473.0 * getattr(cfg, "scale", 1.0))) * 0.5
        return half * float(getattr(cfg, "cell_off_scale", 1.0))

    def cells_to_routes(self, cells, offset=None):
        """선택 셀 idx [N,P,K] (+ 하이브리드 offset[N,P,K,2] raw) → route[N,P,Kw,2] + net_mask."""
        from .cell_action import build_routes_from_cells
        idx = np.clip(np.asarray(cells, np.int64), 0, self.n_cells - 1)   # [N,P,K]
        # ★ 순차 부설: route=[현위치 → 선택셀], 그 leg를 net leg로 (배가 이동하며 도색)
        if getattr(self.cfg, "cell_sequential", False):
            N, P, Kw = self.N, self.P, self.Kw
            tgt = self.cell_world[idx[:, :, 0]].astype(np.float64)        # [N,P,2] 선택 1셀
            route = np.broadcast_to(tgt[:, :, None, :], (N, P, Kw, 2)).copy()
            route[:, :, 0, :] = self.a_pos                                # route[0]=현위치, route[1:]=셀
            route = np.clip(route, 0.0, self.cfg.world_size)
            net_mask = np.zeros((N, P, Kw), bool); net_mask[:, :, 1] = True   # leg 현위치→셀 = 그물
            al = self.a_alive[..., None, None]
            hold = np.broadcast_to(self.a_pos[:, :, None, :], (N, P, Kw, 2))
            route = np.where(al, route, hold)
            net_mask = net_mask & self.a_alive[..., None]
            return route, net_mask.astype(np.int64)
        cell_pts = self.cell_world[idx].astype(np.float64)                # [N,P,K,2]
        if offset is not None and getattr(self.cfg, "cell_hybrid", False):
            off = np.clip(np.asarray(offset, np.float64), -1.0, 1.0) * self._cell_half()
            cell_pts = cell_pts + off                                     # 셀중심 + ±반셀 미세이동
        return build_routes_from_cells(cell_pts, self.a_pos, self.a_alive,
                                       self.center, self.cfg, self.Kw)

    def _apply_cell_actions(self, cells, force_fresh=None, offset=None):
        """선택 셀 → route/net_mask 절대 세팅 (매 결정 재계획, 설치중 그물 보호)."""
        N, P, Kw = self.N, self.P, self.Kw
        route, net_mask = self.cells_to_routes(cells, offset)
        # ★ 미할당 배(assign<0)는 정지 — 배정된 배만 WP 생성/그물
        done = (self.a_nets <= 0) | (self._assign < 0)
        if done.any():
            hold = np.broadcast_to(self.a_pos[:, :, None, :], (N, P, Kw, 2))
            route = np.where(done[..., None, None], hold, route)
            net_mask = np.where(done[..., None], 0, net_mask)
        alive = self.a_alive; locked = self.doing_net.copy()
        fresh = alive & (~locked)                          # 매 결정 재계획(설치중=동결)
        w = fresh[..., None]
        self.route = np.where(w[..., None], route, self.route)
        self.route_anchor = self.route.copy()
        self.net_mask = np.where(w, net_mask.astype(bool), self.net_mask)
        self.wp_reached = np.where(w, False, self.wp_reached)
        self.ptr = np.where(fresh, 0, self.ptr)
        self.leg_netted = np.where(fresh, False, self.leg_netted)
        self.route_init = self.route_init | fresh
        if not getattr(self.cfg, "cell_sequential", False):
            self.paint_dist = np.where(fresh, 0.0, self.paint_dist)   # 순차는 누적(리셋 X)
        else:                                                          # ★ 선택셀 방문표시(재선택 제외)
            sel = np.clip(np.asarray(cells, np.int64), 0, self.n_cells - 1)[:, :, 0]  # [N,P]
            mk = fresh & (self._assign >= 0)                          # 이번 결정에 실제 선택한 배만
            rn, rp = np.where(mk)
            if len(rn):
                self._visited[rn, rp, sel[rn, rp]] = True
        self.doing_net = locked.copy()
        self._prev_assign = self._assign.copy()
        self._last_apply_ptr = self.ptr.copy()
        self._resid_mag = np.zeros((N, P))
        ptr_c = np.clip(self.ptr, 0, Kw - 1)
        self.net_end = np.take_along_axis(self.route, ptr_c[:, :, None, None], axis=2)[:, :, 0, :].copy()
        self.net_start = self.a_pos.copy()

    def _cell_valid_mask(self):
        """배별 후보셀 유효마스크 [N,P,C] (True=무효). 배정요격점 최근접 k + 각도게이트.
        build_cell_obs·heuristic_cells 공유 → 휴리스틱 타깃이 항상 유효셀 안(BC -inf 방지)."""
        cfg = self.cfg; N, P = self.N, self.P; Cn = self.n_cells; c = self.center
        if not getattr(cfg, "cell_prune", True):
            return np.zeros((N, P, Cn), bool)
        cw = self.cell_world
        I = self._assignI
        dIc = np.hypot(I[:, :, None, 0] - cw[None, None, :, 0],
                       I[:, :, None, 1] - cw[None, None, :, 1])            # [N,P,C]
        amax = float(getattr(cfg, "cell_prune_angle", 180.0))
        if amax < 180.0:
            Ib = np.degrees(np.arctan2(I[..., 0] - c[0], I[..., 1] - c[1]))
            Cb = np.degrees(np.arctan2(cw[:, 0] - c[0], cw[:, 1] - c[1]))
            db = np.abs(((Cb[None, None, :] - Ib[..., None] + 180.0) % 360.0) - 180.0)
            dIc = np.where(db <= amax, dIc, np.inf)
        dIc_gate = dIc.copy()                                        # Voronoi 전(게이트만) — fallback용
        # ★ Voronoi 분할: 각 셀을 최근접 '배정된' 배에만 배정 → 배별 세트 disjoint → WP 겹침 불가
        disjoint = getattr(cfg, "cell_prune_disjoint", False)
        if disjoint:
            assigned = (self._assign >= 0)                            # [N,P]
            comp = np.where(assigned[..., None], dIc, np.inf)         # 미배정 배는 경쟁 제외
            best = np.argmin(comp, axis=1)                            # [N,C] 셀별 최근접 배정배
            own = (best[:, None, :] == np.arange(P)[None, :, None])   # [N,P,C]
            dIc = np.where(assigned[..., None], np.where(own, dIc, np.inf), dIc)
        k = min(int(cfg.cell_prune_k), Cn)

        def _mask_from(d):
            keep = np.argpartition(d, k - 1, axis=2)[:, :, :k]
            m = np.ones((N, P, Cn), bool)
            np.put_along_axis(m, keep, False, axis=2)
            return m | ~np.isfinite(d)

        cell_mask = _mask_from(dIc)
        # ★ fallback: Voronoi로 유효셀<cell_nets 된 배는 게이트 k-nearest 복원(0셀 크래시 방지)
        if disjoint:
            short = (~cell_mask).sum(2) < int(cfg.cell_nets)         # [N,P]
            if short.any():
                cell_mask = np.where(short[..., None], _mask_from(dIc_gate), cell_mask)
        # ★ 순차 부설: 방문(선택완료)한 셀 제외 (남은 유효셀 있을 때만 — 0셀 방지)
        if getattr(cfg, "cell_sequential", False) and self._visited is not None:
            newm = cell_mask | self._visited
            still = (~newm).any(2)                                    # [N,P] 남은 유효셀 있나
            cell_mask = np.where(still[..., None], newm, cell_mask)
        return cell_mask

    def heuristic_cells(self):
        """휴리스틱 배정 요격점서 코리도 가로 K셀 → **유효셀 내** 최근접 [N,P,K] (BC seed·GRPO 후보0)."""
        cfg = self.cfg; N, P = self.N, self.P; K = cfg.cell_nets
        self._compute_assignment()
        c = self.center
        I = self._assignI
        rel = I - c[None, None, :]
        D = np.hypot(rel[..., 0], rel[..., 1])
        rad = rel / np.maximum(D, 1e-6)[..., None]
        perp = np.stack([-rad[..., 1], rad[..., 0]], axis=-1)
        cw = self.cell_world
        valid = ~self._cell_valid_mask()                                   # [N,P,C] True=유효
        cells = np.zeros((N, P, K), np.int64)
        used = np.zeros((N, P, self.n_cells), bool)
        for k in range(K):
            off = (k - (K - 1) / 2.0) * cfg.net_max_len
            pt = I + perp * off
            d = np.hypot(pt[..., None, 0] - cw[None, None, :, 0],
                         pt[..., None, 1] - cw[None, None, :, 1])          # [N,P,C]
            d = np.where(valid & (~used), d, np.inf)                       # ★유효·미사용 셀만
            pick = np.argmin(d, axis=-1)
            cells[:, :, k] = pick
            np.put_along_axis(used, pick[..., None], True, axis=2)
        return cells

    def _cell_tokens(self):
        """후보셀 토큰 [N,P,C,5] + pruning 마스크 [N,P,C] (slim 전용).
        ★ 격자기준 정규화: 분모=cell_r_max(격자 extent) → 축끝 셀이 ±1.0 (격자를 [-1,1] 꽉 채움)."""
        cfg = self.cfg; N, P, M = self.N, self.P, self.M
        c = self.center; half = float(cfg.cell_r_max)     # ★ 6000→r_max: 격자좌표 [-1,1]
        cw = self.cell_world; Cn = self.n_cells
        cwn = (cw - c) / half
        rnorm = self.cell_polar[:, 0] / half
        dens_r = (cfg.cell_r_max - cfg.cell_r_min) / max(cfg.cell_bands, 1) * 1.2
        dce = np.hypot(self.e_pos[:, :, None, 0] - cw[None, None, :, 0],
                       self.e_pos[:, :, None, 1] - cw[None, None, :, 1])
        dens = ((dce < dens_r) & self.e_alive[:, :, None]).sum(1).astype(np.float64) / max(M, 1)
        cell1 = np.concatenate([
            np.broadcast_to(cwn[None, :, :], (N, Cn, 2)),
            np.broadcast_to(rnorm[None, :, None], (N, Cn, 1)),
            dens[:, :, None], np.zeros((N, Cn, 1))], axis=-1)            # [N,C,5]
        cell = np.broadcast_to(cell1[:, None, :, :], (N, P, Cn, 5)).copy()
        return cell, self._cell_valid_mask()

    def _nearest_cell_n(self, pts):
        """world 점들 [...,2] → '밟는 셀'(최근접)의 격자기준 정규화 위치 [...,2].
        ★ 분모=cell_r_max → 셀좌표 [-1,1] (격자 밖 개체도 경계셀로 스냅돼 범위 안)."""
        cw = self.cell_world; c = self.center; norm = float(self.cfg.cell_r_max)
        d = ((pts[..., None, :] - cw) ** 2).sum(-1)                      # [...,C]
        idx = d.argmin(-1)                                              # [...]
        return (cw[idx] - c) / norm

    def _build_cell_obs_slim(self):
        """★ 셀기준 슬림 관측 (cell_obs_slim=True). 모든 개체를 '밟는 셀' 위치로.
          own[5]=셀pos2·셀요격2·배정1 | ally[2]=셀pos | enemy[3]=셀중심2·차지셀수1 | cell[5]."""
        cfg = self.cfg; N, P, M = self.N, self.P, self.M
        Cn = self.n_cells; cw = self.cell_world
        self._compute_assignment()
        # ── own [5] ──
        apos_c = self._nearest_cell_n(self.a_pos)                       # [N,P,2] 밟는 셀
        assigned_f = (self._assign >= 0).astype(np.float64)             # [N,P]
        Ic = self._nearest_cell_n(self._assignI) * assigned_f[..., None]   # 미배정=0
        own = np.concatenate([apos_c, Ic, assigned_f[..., None]], axis=-1)  # [N,P,5]
        # ── ally [2] ──
        A = max(P - 1, 1)
        ally = np.zeros((N, P, A, 2)); ally_mask = np.ones((N, P, A), bool)
        for p in range(P):
            others = [q for q in range(P) if q != p][:A]
            for slot, q in enumerate(others):
                ally[:, p, slot, :] = self._nearest_cell_n(self.a_pos[:, q, :])
                ally_mask[:, p, slot] = ~self.a_alive[:, q]
        # ── enemy [3] = 클러스터 셀중심 + 차지 셀수 ──
        cl = clustering.cluster_by_gaps_vec(self.e_pos, self.e_alive, self.e_hdg,
                                            self.center, cfg.enemy_speed, cfg.n_clusters, cfg.cluster_gap_deg)
        cent = cl["centroid"]; labels = cl["labels"]; active = cl["active"]   # [N,K,2],[N,M],[N,K]
        Kc = cfg.n_clusters
        cent_c = self._nearest_cell_n(cent)                            # [N,K,2] 무리중심 셀
        eidx = (((self.e_pos[..., None, :] - cw) ** 2).sum(-1)).argmin(-1)   # [N,M] 적별 밟는셀 idx
        ncells = np.zeros((N, Kc))
        for k in range(Kc):
            mk = (labels == k) & self.e_alive                          # [N,M]
            occ = np.zeros((N, Cn), bool)
            nn, mm = np.where(mk)
            if len(nn):
                occ[nn, eidx[nn, mm]] = True
            ncells[:, k] = occ.sum(1)                                  # 무리가 차지한 distinct 셀 수
        ncells = ncells / max(M, 1)                                    # [0,1] 정규화 footprint
        enemy1 = np.concatenate([cent_c, ncells[..., None]], axis=-1)  # [N,K,3]
        enemy = np.broadcast_to(enemy1[:, None, :, :], (N, P, Kc, 3)).copy()
        enemy_mask = np.broadcast_to((~active)[:, None, :], (N, P, Kc)).copy()
        # ── cell [5] (그대로) ──
        cell, cell_mask = self._cell_tokens()
        return {"own": own, "ally": ally, "ally_mask": ally_mask,
                "enemy": enemy, "enemy_mask": enemy_mask,
                "cell": cell, "cell_mask": cell_mask}

    def build_cell_obs(self):
        """셀선택 pointer 관측 — 모선(0,0) 전역 프레임 토큰 (클러스터 없음, 적 원본).
          own[N,P,6] · ally[N,P,A,6]+mask · enemy[N,P,M,6]+mask · cell[N,P,C,5]+mask.
        정규화: pos = (world-모선)/action_grid_half ∈[-1,1]."""
        if getattr(self.cfg, "cell_obs_slim", False):
            return self._build_cell_obs_slim()
        cfg = self.cfg; N, P, M = self.N, self.P, self.M
        c = self.center; half = cfg.action_grid_half
        self._compute_assignment()                                        # 렌더/배정선 일관
        # ── own (+ 배정 요격점·할당플래그 → 배별 WP 구분) ──
        apos_n = (self.a_pos - c) / half                                  # [N,P,2]
        ah = K.heading_vec(self.a_hdg)                                    # [N,P,2](sin,cos)
        assigned_f = (self._assign >= 0).astype(np.float64)               # [N,P] 1=배정, 0=예비
        In = (self._assignI - c[None, None, :]) / half                    # [N,P,2] 배정 요격점(전역, 정규화)
        In = In * assigned_f[..., None]                                   # 미배정=0
        own = np.stack([apos_n[..., 0], apos_n[..., 1], ah[..., 0], ah[..., 1],
                        self.a_nets / max(cfg.nets_per_ship, 1),
                        self.doing_net.astype(np.float64),
                        In[..., 0], In[..., 1], assigned_f], axis=-1)     # [N,P,9]
        # ── ally (타 아군, 전역 프레임) ──
        A = max(P - 1, 1)
        ally = np.zeros((N, P, A, 6)); ally_mask = np.ones((N, P, A), bool)
        for p in range(P):
            others = [q for q in range(P) if q != p][:A]
            for slot, q in enumerate(others):
                qn = (self.a_pos[:, q, :] - c) / half
                qh = K.heading_vec(self.a_hdg[:, q])
                ally[:, p, slot, :] = np.stack([
                    qn[:, 0], qn[:, 1], qh[:, 0], qh[:, 1],
                    self.a_nets[:, q] / max(cfg.nets_per_ship, 1),
                    self.doing_net[:, q].astype(np.float64)], axis=-1)
                ally_mask[:, p, slot] = ~self.a_alive[:, q]
        # ── enemy: 클러스터 토큰(휴리스틱 정렬) 또는 raw 원본M (전역 프레임; P축 tile) ──
        if getattr(cfg, "cell_cluster_obs", True):
            cl = clustering.cluster_by_gaps_vec(self.e_pos, self.e_alive, self.e_hdg,
                                                c, cfg.enemy_speed, cfg.n_clusters, cfg.cluster_gap_deg)
            cent = cl["centroid"]                                          # [N,K,2]
            cn = (cent - c[None, None, :]) / half                          # [N,K,2] 중심(정규화)
            cdm = np.hypot(cent[..., 0] - c[0], cent[..., 1] - c[1])       # [N,K] 모선거리
            enemy1 = np.stack([
                cn[..., 0], cn[..., 1],
                cl["count"] / max(M, 1),                                   # 무리 크기
                E.spread_norm(cl["spread_deg"]),                          # 퍼짐
                cl["approach"] / cfg.enemy_speed,                         # 접근속도
                cdm / half], axis=-1)                                     # [N,K,6]
            Kc = cfg.n_clusters
            enemy = np.broadcast_to(enemy1[:, None, :, :], (N, P, Kc, 6)).copy()
            enemy_mask = np.broadcast_to((~cl["active"])[:, None, :], (N, P, Kc)).copy()
        else:
            en = (self.e_pos - c) / half                                  # [N,M,2]
            eh = K.heading_vec(self.e_hdg)
            edm = np.hypot(self.e_pos[..., 0] - c[0], self.e_pos[..., 1] - c[1])
            ttb = np.clip(edm / (cfg.enemy_speed * cfg.max_steps + 1e-6), 0, 1)
            enemy1 = np.stack([en[..., 0], en[..., 1], eh[..., 0], eh[..., 1],
                               ttb, edm / half], axis=-1)                 # [N,M,6]
            enemy = np.broadcast_to(enemy1[:, None, :, :], (N, P, M, 6)).copy()
            enemy_mask = np.broadcast_to((~self.e_alive)[:, None, :], (N, P, M)).copy()
        # ── cell (후보셀 C, 전역 프레임; P축 tile) ──
        cw = self.cell_world                                              # [C,2]
        Cn = self.n_cells
        cwn = (cw - c) / half                                             # [C,2]
        rnorm = self.cell_polar[:, 0] / half                             # [C] 반경 정규화
        # 적 밀도: 각 셀 근처(≤dens_r) 살아있는 적 수
        dens_r = (cfg.cell_r_max - cfg.cell_r_min) / max(cfg.cell_bands, 1) * 1.2
        dce = np.hypot(self.e_pos[:, :, None, 0] - cw[None, None, :, 0],
                       self.e_pos[:, :, None, 1] - cw[None, None, :, 1])   # [N,M,C]
        dens = ((dce < dens_r) & self.e_alive[:, :, None]).sum(1).astype(np.float64)  # [N,C]
        dens = dens / max(M, 1)
        cell1 = np.concatenate([
            np.broadcast_to(cwn[None, :, :], (N, Cn, 2)),
            np.broadcast_to(rnorm[None, :, None], (N, Cn, 1)),
            dens[:, :, None],
            np.zeros((N, Cn, 1))], axis=-1)                              # [N,C,5] (net_present=0 v1)
        cell = np.broadcast_to(cell1[:, None, :, :], (N, P, Cn, 5)).copy()
        # ── ★ 배별 후보셀 pruning (요격점 최근접 k + 각도게이트) — heuristic_cells 와 공유 마스크 ──
        cell_mask = self._cell_valid_mask()                            # [N,P,C]
        return {"own": own, "ally": ally, "ally_mask": ally_mask,
                "enemy": enemy, "enemy_mask": enemy_mask,
                "cell": cell, "cell_mask": cell_mask}

    def build_obs(self):
        cfg = self.cfg; N, P = self.N, self.P
        W = cfg.world_size; A = max(cfg.max_pairs - 1, 1)
        c = self.center
        fwd = K.heading_vec(self.a_hdg)                        # [N,P,2] (sin,cos)
        self._compute_assignment()                            # 결정 시작 상태 기준 배정

        # ── own ──
        to_m = np.stack([c[0] - self.a_pos[..., 0], c[1] - self.a_pos[..., 1]], -1)
        mdist = np.hypot(to_m[..., 0], to_m[..., 1])
        # 배정 클러스터 교점 상대(미배정 배는 0으로 마스킹 → '예비' 신호는 flag 로)
        assigned_f = (self._assign >= 0).astype(np.float64)               # [N,P]
        to_I = self._assignI - self.a_pos                                 # [N,P,2]
        Idist = np.hypot(to_I[..., 0], to_I[..., 1])
        own = np.stack([
            self.a_pos[..., 0] / W, self.a_pos[..., 1] / W,
            E.norm_close(mdist, cfg.norm_k_mother),
            E.signed_bearing(fwd, to_m),
            self.a_nets / cfg.nets_per_ship,
            self.doing_net.astype(np.float64),
            assigned_f,
            E.norm_range(Idist, cfg.norm_k_enemy) * assigned_f,
            E.signed_bearing(fwd, to_I) * assigned_f,
        ], axis=-1)                                            # [N,P,10]
        if cfg.net_probe_dirs > 0:                             # + net radar (설치그물 근접도 D칸, 기본 0=off)
            own = np.concatenate([own, self._net_probe(fwd)], axis=-1)   # [N,P,10+D]

        # ── enemy: 각도 클러스터 set (per agent 상대) ──
        cl = clustering.cluster_by_gaps_vec(self.e_pos, self.e_alive, self.e_hdg,
                                            c, cfg.enemy_speed, cfg.n_clusters, cfg.cluster_gap_deg)
        Kc = cfg.n_clusters
        rel = cl["centroid"][:, None, :, :] - self.a_pos[:, :, None, :]   # [N,P,Kc,2]
        edist = np.hypot(rel[..., 0], rel[..., 1])
        fwdK = np.broadcast_to(fwd[:, :, None, :], (N, P, Kc, 2))
        ebrg = E.signed_bearing(fwdK, rel)                               # [N,P,Kc]
        spr = E.spread_norm(cl["spread_deg"])[:, None, :]               # [N,1,Kc]
        cntn = (cl["count"] / max(self.M, 1))[:, None, :]
        appn = (cl["approach"] / cfg.enemy_speed)[:, None, :]
        enemy = np.stack([                                     # champion 조건: approach 포함
            E.norm_range(edist, cfg.norm_k_enemy), ebrg,
            np.broadcast_to(spr, (N, P, Kc)),
            np.broadcast_to(cntn, (N, P, Kc)),
            np.broadcast_to(appn, (N, P, Kc)),
        ], axis=-1)                                            # [N,P,Kc,5]
        enemy_mask = ~np.broadcast_to(cl["active"][:, None, :], (N, P, Kc))

        # ── ally set (per agent, 타 아군 + 계획 그물끝점; 극좌표) ──
        ally = np.zeros((N, P, A, len(SPEC.ALLY_FEATURES)))
        ally_mask = np.ones((N, P, A), bool)
        for p in range(P):
            fwdp = fwd[:, p, :]                                # [N,2]
            others = [q for q in range(P) if q != p]
            for slot, q in enumerate(others):
                if slot >= A:
                    break
                rel_q = self.a_pos[:, q, :] - self.a_pos[:, p, :]
                d_q = np.hypot(rel_q[:, 0], rel_q[:, 1])
                hc, hs = E.heading_cossin(self.a_hdg[:, p], self.a_hdg[:, q])
                pn = self.net_end[:, q, :] - self.a_pos[:, p, :]
                pnd = np.hypot(pn[:, 0], pn[:, 1])
                ally[:, p, slot, :] = np.stack([
                    E.norm_range(d_q, cfg.norm_k_ally),
                    E.signed_bearing(fwdp, rel_q), hc, hs,
                    E.norm_range(pnd, cfg.norm_k_ally),       # ★협조: 계획 그물끝점 거리
                    E.signed_bearing(fwdp, pn),               # ★협조: 계획 그물끝점 방위
                ], axis=-1)                                   # (제거: nets/deploying/wp_dir — 슬림화)
                ally_mask[:, p, slot] = ~self.a_alive[:, q]   # 비활성 아군은 패딩(마스크)

        return {"own": own, "enemy": enemy, "enemy_mask": enemy_mask,
                "ally": ally, "ally_mask": ally_mask}

    # ── 랜덤 액션 (테스트/베이스라인용) ──────────────────────────────
    def sample_actions(self):
        N, P, Kw = self.N, self.P, self.Kw
        out = {"net_go": self.rng.integers(0, 2, (N, P, Kw))}   # 레그별 0/1 (공통)
        if self.cfg.structured_action:
            out["fan"] = self.rng.uniform(-1, 1, (N, P, 7))     # 부채꼴 7파라미터
        else:
            out["wp"] = self.rng.uniform(-1, 1, (N, P, Kw, 2))  # WP 잔차 델타
        return out


if __name__ == "__main__":
    import time
    env = DefenseVecEnv(num_worlds=256)
    obs, _ = env.reset(seed=0)
    print("[env] obs:", {k: v.shape for k, v in obs.items()})
    t0 = time.time(); steps = 40
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(env.sample_actions())
    dt = time.time() - t0
    dec_per_s = steps / dt
    print(f"[env] {env.N} worlds × {steps} decisions in {dt:.2f}s "
          f"→ {dec_per_s:.1f} dec/s, {dec_per_s*env.N*env.cfg.decision_period:,.0f} env-steps/s")
    print(f"[env] reward mean={r.mean():.3f}  n_alive mean={info['n_alive'].mean():.2f}")
