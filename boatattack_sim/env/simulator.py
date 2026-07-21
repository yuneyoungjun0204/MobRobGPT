"""
boatattack_sim/env/simulator.py — 단일 월드 시뮬레이터 (운동학 코어)

아군은 경로(WP 리스트)를 PD 추종하고, 그물 WP 구간에서 격자를 한 칸씩 칠하며 전개한다.
적 10대는 맵 가장자리에서 모선(중앙)으로 아군 2배 속도로 전진한다.
포획 = painted cell 진입 적. breach = 모선 반경 진입 적.

설계: 상태=클래스(이 파일), 기하/격자/운동학=순수 모듈(grid/kinematics/formations).
get_frame() 으로 렌더러가 소비하는 frame_dict 공급 (항해사모사 DataSource 계약 차용).
이 단일 월드는 그대로 벡터화(VecEnv)·정책 구동으로 확장 가능하도록 배열 기반 상태 유지.

수동 조작 API (인터랙티브 뷰어가 호출):
  toggle_running / toggle_manual / cycle_selected(d) / select(i)
  add_click(x,y) / arm_net() / clear_selected() / reset()
"""
import numpy as np

from .config import SimConfig, DEFAULT_CONFIG, DEFAULT_REWARD
from .grid import Grid
from . import formations as F
from . import kinematics as K
from . import clustering


class Simulator:
    """단일 월드 BoatAttack 방어 시뮬레이터."""

    def __init__(self, cfg: SimConfig = DEFAULT_CONFIG, enemy_mode: str = "random"):
        self.cfg = cfg
        self.enemy_mode = enemy_mode
        self.grid = Grid(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        # UI/제어 상태
        self.running = True       # 시간 진행 on/off
        self.manual = False       # 수동 WP 편집 모드
        self.selected = 0         # 선택된 아군 인덱스
        self.show_clusters = True # 클러스터/배정 오버레이 표시 (k 토글)
        self.show_residual = True  # WP 잔차(±wp_adjust_max) 허용범위 원 표시 (j 토글, 기본 ON)
        # 그물 배치 단계: 0=대기, 1=다음클릭=그물 시작, 2=다음클릭=그물 끝
        self._net_stage = 0
        self.reset()

    def toggle_clusters(self): self.show_clusters = not self.show_clusters
    def toggle_residual(self): self.show_residual = not self.show_residual

    # ── 리셋 ──────────────────────────────────────────────────────────

    def reset(self, seed: int = None):
        cfg = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.grid.reset()
        self.t = 0

        # 적 상태 (벡터)
        self.e_pos, self.e_hdg, self.e_phase = F.spawn_enemies(cfg, self.rng, self.enemy_mode)
        self.e_alive = np.ones(cfg.n_enemies, dtype=bool)

        # 아군 상태 (벡터 + 파이썬 경로 리스트)
        self.a_pos, self.a_hdg = F.spawn_allies(cfg, self.rng)
        P = cfg.n_allies
        self.a_paths = [[] for _ in range(P)]    # 각 원소: {x,y,paint,started,active}
        self.a_nets = np.full(P, cfg.nets_per_ship, dtype=np.int64)
        self.a_painting = np.zeros(P, dtype=bool)
        self.a_paint_dist = np.zeros(P, dtype=np.float64)   # 현재 그물 구간 칠한 거리
        self.a_alive = np.ones(P, dtype=bool)               # 그물 접촉 시 비활성화(격침)
        self.a_net_start = self.a_pos.copy()                # 현재 그물 구간 시작점(설치 등록용)
        self.net_installed = np.zeros((self.grid.G, self.grid.G), dtype=bool)  # 완성 그물 셀
        self.prev_on_inst = np.zeros(P, dtype=bool)         # 직전 step 설치그물 위였나(진입 edge)

        # 클러스터 배정 (위협 큰 클러스터 → 가장 가까운 배 1:1, -1=예비)
        self.assign = np.full(P, -1, dtype=np.int64)
        self.assignI = np.zeros((P, 2))
        self._cl_cent = np.tile(np.array(cfg.center, np.float64), (cfg.n_clusters, 1))
        self._cl_spread = np.zeros(cfg.n_clusters)
        self._plan_cluster = np.full(P, -2, dtype=np.int64)   # 배별 마지막 휴리스틱 계획 클러스터
        self._plan_t = np.zeros(P, dtype=np.int64)            # 배별 마지막 재계획 시각(주기 미세조정)
        # 도달 가능 반경: 모선 반경 r 까지 아군이 적보다 먼저 도착 가능한 한계.
        #   (r-s0)/va = (H-r)/ve  →  r = (va·H + ve·s0)/(va+ve).  바깥 그물의 상한.
        H = cfg.world_size / 2.0; s0 = cfg.ally_row_gap
        self._R_FEAS = (cfg.ally_speed * H + cfg.enemy_speed * s0) / (
            cfg.ally_speed + cfg.enemy_speed)

        # 통계
        self.stats = dict(captures=0, breaches=0, ally_collisions=0,
                          nets_used=0, net_touches=0, survived=0)
        self.done = False
        self._net_stage = 0       # 진행 중이던 그물 배치 취소
        self._compute_assignment()
        return self.get_frame()

    # ── 수동 조작 API ─────────────────────────────────────────────────

    def toggle_running(self): self.running = not self.running
    def toggle_manual(self):  self.manual = not self.manual

    def arm_net(self):
        """그물 배치 토글: 대기→'방향 찍기'(1). 진행 중이면 취소(0).
        한 번의 클릭이 방향이 되어 그 방향으로 net_max_len 만큼 그물이 바로 생성된다."""
        self._net_stage = 1 if self._net_stage == 0 else 0

    def select(self, i: int):
        if 0 <= i < self.cfg.n_allies:
            self.selected = i

    def cycle_selected(self, d: int = 1):
        self.selected = (self.selected + d) % self.cfg.n_allies

    def clear_selected(self):
        self.a_paths[self.selected] = []
        self.a_painting[self.selected] = False
        self._net_stage = 0

    def add_click(self, x: float, y: float):
        """수동 모드: 클릭 좌표를 선택된 아군의 경로 WP 로 추가.
        그물 배치(arm_net 후): **한 번의 클릭이 '방향'** → 현재 경로 끝(없으면 선박 위치)에서
        그 방향으로 net_max_len 만큼 그물(paint) WP 를 즉시 생성. (멀리/가깝게 찍어도 고정 길이.)"""
        if not self.manual:
            return
        x = float(np.clip(x, 0, self.cfg.world_size))
        y = float(np.clip(y, 0, self.cfg.world_size))
        path = self.a_paths[self.selected]
        if self._net_stage == 1:        # 그물: 한 번 클릭 = 방향 → 그 방향으로 즉시 생성
            if path:
                sx, sy = path[-1]["x"], path[-1]["y"]          # 경로 끝에서 출발
            else:
                sx = float(self.a_pos[self.selected, 0])       # 없으면 선박 현재 위치
                sy = float(self.a_pos[self.selected, 1])
            dx, dy = x - sx, y - sy
            d = float(np.hypot(dx, dy))
            if d < 1e-6:                # 방향 없음 → 선박 heading 방향 사용
                hr = np.deg2rad(self.a_hdg[self.selected])
                dx, dy, d = float(np.sin(hr)), float(np.cos(hr)), 1.0
            L = self.cfg.net_max_len
            ex = float(np.clip(sx + dx / d * L, 0, self.cfg.world_size))
            ey = float(np.clip(sy + dy / d * L, 0, self.cfg.world_size))
            path.append({"x": ex, "y": ey, "paint": True,
                         "started": False, "active": False})
            self._net_stage = 0
        else:                           # 일반 경유 WP — 간격을 wp_max_len 으로 제한
            n_transit = sum(1 for w in path if not w["paint"])
            if n_transit >= self.cfg.transit_wp:    # 최대 경유 WP 수 상한
                return
            if path:
                px, py = path[-1]["x"], path[-1]["y"]
            else:
                px = float(self.a_pos[self.selected, 0])
                py = float(self.a_pos[self.selected, 1])
            dx, dy = x - px, y - py
            d = float(np.hypot(dx, dy))
            Lmax = self.cfg.wp_max_len
            if d > Lmax:                # 너무 멀면 방향 유지하고 Lmax 로 클램프
                x = float(np.clip(px + dx / d * Lmax, 0, self.cfg.world_size))
                y = float(np.clip(py + dy / d * Lmax, 0, self.cfg.world_size))
            path.append({"x": x, "y": y, "paint": False,
                         "started": False, "active": False})

    # ── 1 스텝 진행 ───────────────────────────────────────────────────

    def step(self):
        """결정/시간 1스텝. 아군 PD 추종(+painting), 적 전진, 포획/breach/충돌 집계."""
        if self.done:
            return self.get_frame()
        cfg = self.cfg

        if not self.manual:           # AUTO = 클러스터 차단 휴리스틱 자동 배치
            self.heuristic_plan()
        self._step_allies()
        self._step_enemies()
        self._resolve_captures()
        self._resolve_breaches()
        self._resolve_ally_collisions()
        self._resolve_net_touches()
        self._compute_assignment()

        self.t += 1
        # 종료 판정
        if not self.e_alive.any():
            self.done = True
        elif self.t >= cfg.max_steps:
            self.stats["survived"] = int(self.e_alive.sum())
            self.done = True
        return self.get_frame()

    # ── 아군 갱신 (경로 추종 + 그물 painting) ─────────────────────────

    def _step_allies(self):
        cfg = self.cfg
        for i in range(cfg.n_allies):
            if not self.a_alive[i]:              # 격침된 아군은 정지(그물도 중단)
                self.a_painting[i] = False
                continue
            path = self.a_paths[i]
            if not path:
                self.a_painting[i] = False
                continue
            wp = path[0]

            # 그물 구간 시작 시 자원 소비 (1회)
            if wp["paint"] and not wp["started"]:
                wp["started"] = True
                if self.a_nets[i] > 0:
                    self.a_nets[i] -= 1
                    self.stats["nets_used"] += 1
                    wp["active"] = True          # 실제로 칠하는 구간
                    self.a_paint_dist[i] = 0.0   # 새 그물 구간 거리 리셋
                    self.a_net_start[i] = self.a_pos[i].copy()   # 설치 등록용 시작점
                else:
                    wp["active"] = False         # 자원 없음 → transit 취급
            painting = bool(wp.get("active", False))
            # 길이 한계 도달 시 더 이상 칠하지 않음 (그 방향으로 net_max_len 만큼만)
            if painting and self.a_paint_dist[i] >= cfg.net_max_len:
                painting = False
            self.a_painting[i] = painting

            old = self.a_pos[i].copy()
            pos_new, hdg_new, arrived = K.pd_follow(
                self.a_pos[i], self.a_hdg[i], (wp["x"], wp["y"]),
                cfg.ally_speed, cfg.ally_max_turn, cfg.dt,
                turn_gain=cfg.ally_turn_gain, slow_min=cfg.ally_slow_min,
                arrive_radius=cfg.arrive_radius)
            self.a_pos[i] = pos_new[0]; self.a_hdg[i] = float(hdg_new[0])
            moved = float(np.hypot(self.a_pos[i, 0] - old[0],
                                   self.a_pos[i, 1] - old[1]))

            # 그물 전개: 매 스텝 현재 위치 주변 띠 마킹 (길이 한계까지)
            if painting:
                self.grid.paint_at(self.a_pos[i, 0], self.a_pos[i, 1])
                self.a_paint_dist[i] += moved

            if bool(arrived[0]):
                # 그물 구간 완성 → 그 세그먼트를 '설치(installed)' 등록 (아군 접촉 판정용).
                #   방금 완성한 배는 자기 그물 끝점 위 → 이번 step 접촉 면제(prev_on_inst).
                if wp["paint"] and wp.get("active", False):
                    self._install_net(self.a_net_start[i],
                                      np.array([wp["x"], wp["y"]]))
                    self.prev_on_inst[i] = True
                path.pop(0)
                self.a_painting[i] = False

    # ── 적 갱신 ───────────────────────────────────────────────────────

    def _step_enemies(self):
        cfg = self.cfg
        if not self.e_alive.any():
            return
        idx = np.where(self.e_alive)[0]
        pos_new, hdg_new = K.enemy_step(
            self.e_pos[idx], self.e_hdg[idx], np.array(cfg.center),
            cfg.enemy_speed, cfg.enemy_max_turn, self.t,
            weave_amp=cfg.enemy_weave_amp, weave_period=cfg.enemy_weave_period,
            phase=self.e_phase[idx], dt=cfg.dt)
        self.e_pos[idx] = pos_new
        self.e_hdg[idx] = hdg_new

    # ── 판정 ──────────────────────────────────────────────────────────

    def _resolve_captures(self):
        cap = self.grid.captured_mask(self.e_pos, self.e_alive)
        n = int(cap.sum())
        if n:
            self.e_alive[cap] = False
            self.stats["captures"] += n

    def _resolve_breaches(self):
        if not self.e_alive.any():
            return
        c = np.array(self.cfg.center)
        d = np.hypot(self.e_pos[:, 0] - c[0], self.e_pos[:, 1] - c[1])
        breach = self.e_alive & (d <= self.cfg.mothership_radius)
        n = int(breach.sum())
        if n:
            self.e_alive[breach] = False
            self.stats["breaches"] += n

    def _resolve_ally_collisions(self):
        P = self.cfg.n_allies
        if P < 2:
            return
        r = self.cfg.ally_collision_radius
        cnt = 0
        for a in range(P):
            for b in range(a + 1, P):
                dd = np.hypot(*(self.a_pos[a] - self.a_pos[b]))
                if dd < r:
                    cnt += 1
        self.stats["ally_collisions"] += cnt

    # ── 설치 그물 등록 / 아군 그물 접촉 판정 ──────────────────────────
    def _install_net(self, a, b):
        """완성된 그물 세그먼트 A→B 를 net_installed 격자에 띠로 등록 (defense_env 패리티)."""
        G = self.grid.G; cell = self.grid.cell
        h = max(0, (self.cfg.net_width - 1) // 2)
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        nstep = int(max(2, np.ceil(length / cell) + 1))
        for s in np.linspace(0.0, 1.0, nstep):
            p = a + s * (b - a)
            ci = int(np.clip(p[0] // cell, 0, G - 1))
            cj = int(np.clip(p[1] // cell, 0, G - 1))
            i0, i1 = max(0, ci - h), min(G, ci + h + 1)
            j0, j1 = max(0, cj - h), min(G, cj + h + 1)
            self.net_installed[i0:i1, j0:j1] = True

    def _resolve_net_touches(self):
        """설치된 그물(자기/다른 아군 무관)에 새로 진입한 활성 아군 → 비활성화+집계.
        진입 edge(직전 미접촉) 로 검출. 전개중이어도 면제 안 함(다른 아군 그물에도 걸림)."""
        cell = self.grid.cell; G = self.grid.G
        ci = np.clip((self.a_pos[:, 0] // cell).astype(np.int64), 0, G - 1)
        cj = np.clip((self.a_pos[:, 1] // cell).astype(np.int64), 0, G - 1)
        on_inst = self.net_installed[ci, cj] & self.a_alive
        touch = on_inst & (~self.prev_on_inst)
        n = int(touch.sum())
        if n:
            self.a_alive[touch] = False
            self.a_painting[touch] = False
            self.stats["net_touches"] += n
        self.prev_on_inst = on_inst

    # ── 클러스터→배 배정 (위협 큰 클러스터를 가장 가까운 활성 배에 1:1) ──
    def _compute_assignment(self):
        """현재 상태에서 위협 큰 클러스터 top-min(#active,P)개를 골라 가장 가까운
        살아있는 배에 1:1 그리디 배정. assign[P](클러스터 idx,-1=예비), assignI[P,2](교점).
        (defense_env._compute_assignment 단일 월드판 — 시각화용)."""
        cfg = self.cfg; P = cfg.n_allies; Kc = cfg.n_clusters
        c = np.array(cfg.center, np.float64)
        cl = clustering.cluster_by_gaps_vec(self.e_pos[None], self.e_alive[None],
                                            self.e_hdg[None], c, cfg.enemy_speed, Kc,
                                            cfg.cluster_gap_deg)
        cent = cl["centroid"][0]; cnt = cl["count"][0]; active = cl["active"][0]
        self._cl_cent = cent                                # [Kc,2] 휴리스틱 경로 생성에 재사용
        self._cl_spread = cl["spread_deg"][0]               # [Kc] 클러스터 각도 폭(커버 후보 범위)
        t = DEFAULT_REWARD.assign_intercept_t
        I = cent + t * (c[None, :] - cent)                       # [Kc,2] 교점
        md = np.hypot(cent[:, 0] - c[0], cent[:, 1] - c[1])
        close = np.clip(1.0 - md / (cfg.world_size / 2.0), 0.05, 1.0)
        threat = np.where(active, cnt * close, -1.0)
        topP = np.argsort(-threat)[:P]
        target = np.zeros(Kc, bool); target[topP] = True; target &= active
        cost = np.hypot(self.a_pos[:, None, 0] - I[None, :, 0],
                        self.a_pos[:, None, 1] - I[None, :, 1])   # [P,Kc]
        BIG = 1e18
        cost = np.where(target[None, :], cost, BIG)
        cost = np.where(self.a_alive[:, None], cost, BIG)
        assign = np.full(P, -1, np.int64); assignI = np.zeros((P, 2))
        work = cost.copy()
        for _ in range(min(P, Kc)):
            j = int(np.argmin(work)); v = work.flat[j]
            if v >= BIG / 2:
                break
            pi, ki = j // Kc, j % Kc
            assign[pi] = ki; assignI[pi] = I[ki]
            work[pi, :] = BIG; work[:, ki] = BIG
        self.assign = assign; self.assignI = assignI

    # ── 휴리스틱 자동 컨트롤러 (AUTO 모드 기본 동작) ───────────────────
    def heuristic_plan(self):
        """배정된 배는 담당 클러스터의 코리도를 **여러 그물로 층층이** 가로막는다.
        가까운(도달 가능) 곳을 먼저 깔고 바깥/날개로 이동하며 못 막은 부분을 추가 차단.
        RL 결정주기처럼 decision_period 마다(또는 경로 소진 시) 현재 상태로 재계획 → 미세 조정.
        예비(미배정)·그물 소진 배는 정지. 전개중에는 끊지 않음(떨림 방지).
        ★ 이 휴리스틱이 곧 정책의 '교사 신호' — 배정·층상 배치·주기 재계획이 RL과 1:1 대응."""
        cfg = self.cfg
        for i in range(cfg.n_allies):
            if not self.a_alive[i]:
                self.a_paths[i] = []; self._plan_cluster[i] = -2
                continue
            k = int(self.assign[i])
            if k < 0 or self.a_nets[i] <= 0:        # 예비/그물소진 → 정지
                if not self.a_painting[i]:
                    self.a_paths[i] = []
                self._plan_cluster[i] = k
                continue
            if self.a_painting[i]:                  # 전개중 = 현재 그물 유지
                continue
            due = (self.t - self._plan_t[i]) >= cfg.decision_period
            if self._plan_cluster[i] == k and self.a_paths[i] and not due:
                continue                            # 같은 담당·경로 잔존·주기 미도래 → 유지
            self.a_paths[i] = self._build_cluster_path(i, k)
            self._plan_cluster[i] = k
            self._plan_t[i] = self.t

    def _covered_bearing_mask(self, nbin=120):
        """설치된 그물(net_installed)이 **각도상 커버하는 방위 빈** 마스크 [nbin] 반환.
        모선 기준, 적은 거의 방사형으로 접근하므로 '그 방위에 설치 그물이 있으면 차단됨'."""
        mask = np.zeros(nbin, bool)
        ii, jj = np.where(self.net_installed)
        if len(ii) == 0:
            return mask
        c = np.array(self.cfg.center, np.float64); cell = self.grid.cell
        x = (ii + 0.5) * cell; y = (jj + 0.5) * cell
        brg = np.degrees(np.arctan2(x - c[0], y - c[1])) % 360.0
        mask[(brg / (360.0 / nbin)).astype(np.int64) % nbin] = True
        return mask

    def _build_cluster_path(self, i, k):
        """담당 클러스터 k 를 막는 **방사형 부채꼴**(설치 순서=반경 순서) 경로 생성.
        먼저 까는 그물을 모선 가까이(near), 나중 그물을 바깥(far)으로 → 배가 모선 근처서
        시작해 **바깥으로 전진하며 살포**(후퇴 최소화). 측면축으로도 한 방향 스윕(끝끼리 이음).
        ★ 이미 설치된 그물이 **각도상 커버하는 방위는 후보에서 배제** → 덮이지 않은 각도로
          중심을 옮겨 중복 전개 방지(다 덮였으면 그물 아끼고 정지).
        경로 = [transit(near 시작점), paint, paint, …] (n그물 = n+1 분점)."""
        cfg = self.cfg; L = cfg.net_max_len
        c = np.array(cfg.center, np.float64)
        cent = self._cl_cent[k]
        D = float(np.hypot(cent[0] - c[0], cent[1] - c[1]))
        if D < 1e-3:
            return []
        Rcap = self._R_FEAS * cfg.net_deploy_reach
        r_far = float(np.clip(cfg.net_deploy_frac * D,
                              cfg.mothership_radius + 600.0, Rcap))   # 바깥(나중) 그물 반경
        r_near = float(np.clip(cfg.net_deploy_near * D,
                               cfg.mothership_radius + 400.0, r_far))  # 안쪽(먼저) 그물 반경
        n = int(min(self.a_nets[i], 3))
        if n <= 0:
            return []

        # ── 각도상 '덮이지 않은' 곳으로 부채꼴 중심 방위 선택 (중복 배제) ──
        theta_k = float(np.degrees(np.arctan2(cent[0] - c[0], cent[1] - c[1])) % 360.0)
        nbin = 120; binw = 360.0 / nbin
        cov = self._covered_bearing_mask(nbin)
        delta = float(np.degrees(np.arctan2(n * L / 2.0, r_far)))  # 부채꼴이 덮는 방위 반폭
        ext = max(0.5 * float(self._cl_spread[k]), delta)         # 클러스터 각도폭 내 탐색
        best_b, best_score = theta_k, -1
        for b in theta_k + np.linspace(-ext, ext, 11):
            lo = int(np.floor((b - delta) / binw)); hi = int(np.ceil((b + delta) / binw))
            idx = np.arange(lo, hi + 1) % nbin
            score = int((~cov[idx]).sum())                       # 그 창의 '미커버' 빈 수
            if score > best_score:
                best_score, best_b = score, b
        if best_score <= 0:                                      # 이미 각도상 다 덮임 → 정지
            return []
        br = np.deg2rad(best_b)
        rad = np.array([np.sin(br), np.cos(br)])                 # 부채꼴 중심 방위(미커버쪽)
        perp = np.array([-rad[1], rad[0]])

        # ── 방사형 부채꼴: 설치 순서=반경 순서(먼저=모선 가까이 → 나중=바깥) + 측면 스윕 ──
        radii = np.linspace(r_near, r_far, n + 1)                # step0=near … stepN=far
        edge = (np.arange(n + 1) - n / 2.0) * L                  # 측면 분점(끝끼리 이음)
        ship_lat = float(np.dot(self.a_pos[i] - c, perp))
        if ship_lat > 0.0:
            edge = edge[::-1]                                    # 배 가까운 wing 부터 스윕
        path = []
        for step in range(n + 1):
            pt = np.clip(c + rad * radii[step] + perp * edge[step], 0, cfg.world_size)
            path.append({"x": float(pt[0]), "y": float(pt[1]), "paint": step > 0,
                         "started": False, "active": False})
        return path

    # ── 렌더러용 frame_dict (DataSource 계약) ─────────────────────────

    def get_frame(self) -> dict:
        cfg = self.cfg
        alive = self.e_alive
        return {
            "world_size":  cfg.world_size,
            "cell_size":   cfg.cell_size,
            "t":           self.t,
            "done":        self.done,
            "mothership":  np.array(cfg.center),
            "mothership_radius": cfg.mothership_radius,
            "moback_size": cfg.moback_size,
            "moback_heading": cfg.moback_heading,
            # 적
            "enemy_pos":   self.e_pos.copy(),
            "enemy_hdg":   self.e_hdg.copy(),
            "enemy_alive": alive.copy(),
            "enemy_size":  cfg.enemy_size,
            # 아군
            "ally_pos":    self.a_pos.copy(),
            "ally_hdg":    self.a_hdg.copy(),
            "ally_paths":  [list(p) for p in self.a_paths],
            "ally_nets":   self.a_nets.copy(),
            "ally_painting": self.a_painting.copy(),
            "ally_alive":  self.a_alive.copy(),
            "ship_len":    cfg.ship_len,
            "ship_wid":    cfg.ship_wid,
            # 클러스터 배정 (시각화: 어느 배가 어느 클러스터를 맡았나 / 예비)
            "n_clusters":  cfg.n_clusters,
            "cluster_gap_deg": cfg.cluster_gap_deg,    # 적응형 클러스터 viz 일관(정책과 동일 gap)
            "enemy_speed": cfg.enemy_speed,
            "show_clusters": self.show_clusters,
            "show_residual": self.show_residual,      # j 토글: WP 잔차 허용범위 원
            "wp_adjust_max": cfg.wp_adjust_max,       # 잔차 반경 (m)
            "enemy_mode":  self.enemy_mode,
            "assign":      self.assign.copy(),
            "assignI":     self.assignI.copy(),
            # 격자
            "painted":     self.grid.painted,         # [G,G] bool (뷰 전용, 비복사)
            # UI/상태
            "selected":    self.selected,
            "manual":      self.manual,
            "running":     self.running,
            "net_stage":   self._net_stage,
            "stats":       dict(self.stats),
            "n_alive":     int(alive.sum()),
        }


# ── 헤드리스 스모크 (모듈 단독 실행) ─────────────────────────────────
if __name__ == "__main__":
    cfg = SimConfig()
    sim = Simulator(cfg)
    # 자동 데모: 아군 각자 모선 둘레를 가로지르는 그물 1개 전개
    cx, cy = cfg.center
    for i in range(cfg.n_allies):
        ang = 2 * np.pi * i / cfg.n_allies
        sx = cx + 700 * np.sin(ang);          sy = cy + 700 * np.cos(ang)
        ex = cx + 700 * np.sin(ang + 0.9);    ey = cy + 700 * np.cos(ang + 0.9)
        sim.a_paths[i] = [
            {"x": sx, "y": sy, "paint": False, "started": False, "active": False},
            {"x": ex, "y": ey, "paint": True,  "started": False, "active": False},
        ]
    for _ in range(cfg.max_steps):
        sim.step()
        if sim.done:
            break
    print(f"[smoke] steps={sim.t} painted={sim.grid.painted_ratio:.4f} "
          f"stats={sim.stats} n_alive={int(sim.e_alive.sum())}")
