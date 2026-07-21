"""
boatattack_sim/env/config.py — 시뮬레이터 설정 단일 정의 (Single Source of Truth)

학습 / 시뮬 / 시각화 / export 가 동일한 SimConfig 인스턴스를 공유해 불일치를 차단한다.
(삼성중공업 항해사모사 프로젝트의 config.py 패턴 차용: dataclass + to_dict/from_dict
 + __post_init__ 파생필드 + @property 계산값.)

좌표계: nav 규약.  x=East, y=North, heading 0°=North, 시계방향(CW) +.
        이동:  x += sin(hdg)·v,  y += cos(hdg)·v.
단위:   거리=meter, 시간=step(=dt초), 각도=degree.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class SimConfig:
    """BoatAttack 방어 시뮬레이터 전체 설정."""

    # ── 맵 / 격자 ─────────────────────────────────────────────────────
    #   모선(중앙)~가장자리 = 6.3km. wave 후속파 최대 6km 수용 + 약간 여유(11→12.6km).
    world_size: float = 12600.0    # 정사각 맵 한 변 (m). half=6.3km (wave 최대 6km 수용; 비-wave 스폰은 enemy_spawn_radius)
    grid_size:  int   = 200        # 고해상 격자 한 변 (cell). cell=50m. 충돌·painting·포획 전용
    dt:         float = 1.0        # 1 step = dt 초
    # ── ★ 행동 그리드 (모선중심 이산 좌표계; 셀선택 행동공간·시각화용) ──
    #   원점(0,0)=모선. 모선중심 정사각 박스 half=action_grid_half(m), N×N 셀. 정규화 좌표 = (world-center)/half ∈[-1,1].
    #   물리 grid_size(200, 충돌/painting)와 **별개** — 이건 계획·행동·관측용 coarse 격자.
    action_grid_n:    int   = 20       # 한 변 셀 수 (20×20=400셀)
    action_grid_half: float = 6000.0   # 모선중심 박스 반폭(m). 적 스폰 5.45km + 여유 → 셀=600m
    # ── ★ 셀선택 행동공간 (pointer 어텐션): 배가 극좌표 후보셀 중 cell_nets개 선택=그물 드롭 위치 ──
    #   클러스터 제거·적 원본 관측. 셀=요격 환형(annulus)의 (방위×반경) 격자. GRPO(정책경사)로 학습.
    cell_action:  bool  = False        # True=셀선택 pointer 모드(잔차/부채꼴 대신)
    cell_grid:    str   = "polar"      # 후보셀 격자: "polar"(방위×반경) | "cartesian"(정사각) | "annulus"(균일간격 환형)
    cell_cart_n:  int   = 26           # cartesian 격자 한 변 셀 수 (환형 밖 제외 후 사용)
    cell_spacing: float = 473.0        # ★ annulus 모드 셀 간격(m, 반경·호 모두 균일)
    cell_bearings: int  = 24           # 방위 sector 수
    cell_bands:    int  = 6            # 반경 band 수 (후보셀 = bearings×bands)
    cell_r_min:    float = 800.0       # 요격 환형 안쪽 반경(m, 모선기준)
    cell_r_max:    float = 4500.0      # 요격 환형 바깥 반경(m)
    # ★ 기본 액션 = 그물 1개(=셀 2개, 벽 양끝점). 실측상 1벽/배가 스윗스팟(K=2 ≥ K=3,4).
    #   여러 그물은 배정층에서 한 배에 net-task 여러 번 할당(HeteroMRTA식)으로 처리(관심사 분리).
    cell_nets:     int  = 2            # 선택할 셀 수. 그물 세그먼트 = cell_nets-1 (K=2 → 그물 1벽)
    cell_cluster_obs: bool = True      # ★ 적 관측: True=클러스터 토큰(휴리스틱 정렬, BC↑) | False=raw 원본10
    # ★ 배별 후보셀 pruning: 각 배 후보를 자기 배정요격점 근처 k개로 제한 → 배별 WP 구분 강제 + 행동공간 축소
    cell_prune:   bool  = True         # True=배별 pruning(붕괴/공유 방지)
    cell_prune_k: int   = 16           # 배당 유효 후보셀 수(요격점 최근접 k개)
    cell_prune_angle: float = 75.0     # ★ 후보 각도게이트: 요격점 방위 ±이 각도 안만(모선 감싸 반대편 wrap 방지). 75°=wrap 0
    cell_prune_disjoint: bool = True   # ★ 후보셀 Voronoi 분할: 각 셀을 최근접 배에만 배정 → 배별 세트 disjoint → WP 겹침 불가
    # ── ★ 하이브리드 행동: 셀 선택(이산) + 셀내 미세 연속 오프셋(Gaussian) → 격자 양자화 손실 회복 ──
    cell_hybrid:  bool  = False        # True=선택셀에 ±반셀 연속 오프셋 추가(coarse-to-fine). False=순수 이산(기존)
    cell_off_scale: float = 1.0        # 오프셋 반경 = cell_off_scale × 반셀크기(격자간격/2). 1.0=인접셀 경계까지
    # ── ★ 셀기준 슬림 관측 (cell_action 전용, 잔차모드 무관): 모든 개체를 '밟는 셀' 위치로 표현 ──
    #   own[5]=셀pos2·셀요격점2·배정1 | ally[2]=셀pos | enemy[3]=셀중심2·차지셀수1 | cell[5] 그대로
    cell_obs_slim: bool = False        # True=셀기준 대폭축소 관측 | False=full(기존 own9/ally6/enemy6)
    cell_recurrent: bool = False       # ★ actor 백본에 LSTM 1층 추가(시간기억). slim 관측과 짝일 때 동역학 복원
    # ── ★ 순차 부설: 한 결정에 1셀 선택 → 현위치→그셀 이동하며 그물 도색 → 방문셀 제외 → net_max_len 소진 종료 ──
    cell_sequential: bool = False      # True=적응적 순차 1셀 부설 | False=K셀 동시선택 벽(기존)

    # ── 모선 (방어 대상) ──────────────────────────────────────────────
    #   맵 정중앙 고정. breach = 적이 이 반경 안에 진입.
    mothership_radius: float = 260.0   # breach 반경 (항공모함 함체를 감싸는 스케일)

    # ── 적 선박 (공격) ────────────────────────────────────────────────
    n_enemies:          int   = 10      # 적 10대
    enemy_speed_mult:   float = 1.5     # 적 = 아군 × 1.5 (아군은 적의 0.667배 — 더 빨라 차단 난이도↑)
    enemy_max_turn:     float = 5.0     # deg/step
    enemy_weave_amp:    float = 14.0    # 위빙 진폭 (deg)
    enemy_weave_period: float = 32.0    # 위빙 주기 (step)
    # ── ★ 적 적응형 그물 회피(evasion): 헤딩 전방에 설치그물 있으면 측면으로 비껴감 ──
    #   정적 부채꼴 휴리스틱을 '뚫리게' 만들어 RL의 적응(반응·예측) 우위 무대를 만든다.
    enemy_evade:       bool  = True     # 적 net 회피 ON/OFF
    enemy_evade_look:  float = 500.0    # 전방 탐지 거리 (m)
    enemy_evade_deg:   float = 32.0     # 회피 조향 각 (deg, 측면)
    enemy_spawn_margin: float = 50.0    # 가장자리에서 안쪽 여유 (m). 맵 끝≈5km full 스폰
    # ★ 비-wave 포메이션 기본 스폰거리(m): 맵 확대와 무관하게 ~5.45km 유지(rotate/concentrated 등 불변).
    #   None 이면 world_half-margin(가장자리). wave 는 enemy_wave_near/gap 로 별도(5~6km staggered).
    enemy_spawn_radius: float = 5450.0
    enemy_spawn_frac:   float = 1.0     # 커리큘럼: 적 스폰 거리 비율(1.0=가장자리5km, ↓=가까이). 배포=1.0
    spawn_phase_lo:     float = 0.5     # 월드 비동기화: 월드별 시작거리 ×uniform(lo,1.0).
                                        #   1.0=동기화(끔). <1.0=에피소드 길이 분산→리셋 분산→valid↑
    # ── 적 그룹(클러스터) 스폰 ────────────────────────────────────────
    #   적은 산발이 아니라 '뭉쳐서' 온다. 각 그룹 = 모선 기준 한 방위에서 좁게 모인 무리.
    enemy_spawn_groups: int   = 3       # grouped/기본 모드의 공격 그룹 수
    enemy_group_jitter: float = 150.0   # 그룹 내부 산포 반경 (m, 작을수록 빽빽한 클러스터)
    #   ↑ 220→150 축소: 220 은 한 그룹이 여러 덩어리로 흩어져 클러스터 과다(정신없음) 유발.
    #     150 이면 diversionary 3그룹이 ~96% 깔끔히 3덩어리 유지(클러스터 응집).
    enemy_wave_ranks:   int   = 3       # 파상(wave) 모드의 단(rank) 수
    enemy_wave_gap:     float = 1000.0  # 파상 단 간 전후(거리) 간격 (m). near+,(nr-1)*gap=far
    # ★ wave 거리범위 [near, near+(ranks-1)*gap] 안으로 radial clamp. 가장 가까운 파부터 +gap 씩 뒤로.
    #   near=4000, gap=1000, ranks=3 → 4000(먼저 도달)/5000/6000(나중). 적은 모두 4~6km 이내.
    enemy_wave_near:    float = 4000.0
    # ★ wave 단(rank) 간 방위 분산(deg): 0=완전 같은 방향(1클러스터), >0=균등 부채. 7:3 치우침 방지용
    #   rank 별 방위 = b0 + (k-(nr-1)/2)*spread → nr=3,spread=18 → b0-18/b0/b0+18 (각 4/3/3 균등).
    enemy_wave_spread:  float = 18.0

    # ── 포메이션 도메인 랜덤화 (매 스폰마다 '구조 파라미터'를 흔들어 매번 다른 변형) ──────
    #   위치/방위/위상은 이미 매 스폰 랜덤(formations.py b0/jitter/phase). 여기서 추가로
    #   파상 간격·단수·퍼짐, 양동 각도, 그룹 산포 등 '모양' 자체를 ±frac 범위로 무작위화 →
    #   "정형 파상 한 종류"가 아니라 매 에피소드 다른 파상/양동을 학습(일반화↑).
    domain_rand:        bool  = False   # 기본 OFF(정형) — wave 특화 휴리스틱 초과 레시피가 정형 기준.
    #   학습서 켜려면 grpo --domain-rand. 시각화/compare 는 명시적 ON(끄려면 --no-domain-rand).
    domain_rand_frac:   float = 0.30    # 연속 파라미터 흔들기 폭 (±30%): gap/spread (거리 near 는 제외)
    domain_rand_ranks:  tuple = (2, 3)  # wave 단(rank) 수 무작위 선택지(4단=클러스터 과다 → 제외)
    domain_rand_angle:  float = 20.0    # diversionary 그룹 방위 흔들기 폭 (±deg)
    # ★ wave 거리 규제: near(첫 파 거리)는 DR 로 흔들지 않음(흔들면 wave 만 ~3600 으로 확 가까이 와
    #   다른 포메이션 ~5450 과 불일치·근접 과다). 정형 near(enemy_wave_near=4000) 고정 → 모양만 변형.

    # ── 아군 선박 (방어 에이전트) ─────────────────────────────────────
    #   배치: 모선 '바로 아래'에 한 줄로 옆(East)으로만 띄움. 전원 heading=아래(180°).
    n_allies:       int   = 3       # 아군 P대 (가변, ≤ max_pairs). vary_allies=True 면 이게 상한 P
    max_pairs:      int   = 8        # 정책 슬롯 상한 (마스킹/패딩용)
    # ── ★ 가변 아군 수: 월드(에피소드)마다 활성 아군 수를 ally_choices 중 균등 랜덤 ──
    #   P=n_allies(상한) 고정 + 부재 아군은 a_alive=False 로 비활성(할당·이동·obs·충돌서 제외).
    #   소수 아군으로 다수 적 방어 → 협조·동적 그물운용이 강제되는 무대(휴리스틱 1:1 배정 약점).
    vary_allies:    bool  = False    # True=월드별 활성 아군 수 랜덤 (학습 전 check_setup 로 검증)
    ally_choices:   tuple = (1, 2, 3)  # 활성 아군 수 후보 (균등 추출; 값 ≤ n_allies)
    ally_speed:     float = 6.0      # m/step. 적의 0.75배 (적 = 6 × 1.3333 ≈ 8)
    ally_max_turn:  float = 8.0      # deg/step (배는 천천히 돈다)
    ally_turn_gain: float = 0.6      # PD 비례게인 (heading 오차 → 선회 명령)
    ally_slow_min:  float = 0.30     # 급선회 시 속도 하한 비율 (cos 감속 바닥)
    arrive_radius:  float = 200.0    # WP 도착 판정 반경 (m). 선회한계로 타이트한 WP 맴도는 것 완화
                                     #   (200m 이내면 도착 처리 → 다음 WP/그물 leg 로 전환)
    ally_row_gap:   float = 550.0    # 모선 중심에서 아래로 줄을 놓는 거리 (m, 함미 바로 아래)
    ally_side_spacing: float = 330.0 # 아군 간 옆(East) 간격 (m)
    ally_heading:   float = 180.0    # 초기 heading (180°=South=아래)

    # ── 그물 (cell-painting 띠) ───────────────────────────────────────
    nets_per_ship:  int   = 5        # 선박당 그물 자원. ★ 5 → 휴리스틱 distinct WP = nnet+1 = 6 (행동공간 6WP 일치)
    # ── ★ 그물 전개 이동비용: 전개중(painting)엔 느리고 선회 제한 → '쉬었다 질주→재투척' 동적운용에
    #   의미부여(소수 아군이 다수 적 커버). 1.0=비용없음(기존 동작). 휴리스틱은 경직돼 더 손해 → RL 우위.
    deploy_speed_mult: float = 1.0   # 전개중 속도 배율 (<1=감속). 무대 학습 시 0.5 권장
    deploy_turn_mult:  float = 1.0   # 전개중 선회 배율 (<1=예인 중 선회제한). 0.6 권장
    net_width:      int   = 4        # 띠 폭 (cell) ≈ 220m (champion 조건; 3은 capture 손실로 원복)
    net_max_len:    float = 450.0    # 그물 길이 한계 (m, 상수). 끝점을 멀리/가깝게 찍어도
                                     #   '그 방향으로 이 길이만큼'만 전개된다.
    net_paint_step: int   = 1        # (예약) 스텝당 전진 cell 수

    # ── 휴리스틱 그물 전개 (AUTO 모드 자동 배치 위치 파라미터) ─────────
    #   배정된 배는 담당 클러스터의 코리도를 가로질러 그물들을 '측면 스윕'(끝끼리 이어
    #   한 방향으로만 쓸어) 깐다 → 방향 전환 없이 부드럽게 전개. 위치는 아래 값으로 조정.
    #   ★ '뿌리면서 전진': 그물을 설치 순서=반경 순서로 깐다. 먼저 까는 쪽을 모선에 가깝게
    #     (near), 이후 바깥(far)으로 → 배가 모선 근처서 시작해 바깥으로 전진하며 살포 → 후퇴↓.
    #     (적이 가까워질 때 재계획해도 안쪽은 이미 near 그물이 막아 되돌아갈 필요가 적음.)
    net_deploy_near:  float = 0.18   # **먼저** 까는 그물 반경 = near × 적거리 D (모선 가까이).
    net_deploy_frac:  float = 0.45   # **나중** 까는 그물 반경 = frac × 적거리 D (바깥/전방).
                                     #   near→frac 로 부채꼴 전개. 도달가능 반경(R_FEAS)서 상한.
    net_deploy_reach: float = 1.0    # 도달가능 반경(R_FEAS) 상한 배수. ↑면 더 멀리(전진↑·위험↑).

    # ── 충돌 (충돌 시 큰 패널티 + 비활성화) ──────────────────────────
    ally_collision_radius: float = 115.0  # 아군-아군 충돌 간주 거리 (m)
    ally_mother_radius:    float = 300.0  # 아군-모선(항공모함) 충돌 간주 거리 (m)
    # ★ 휴리스틱 충돌회피 안전층(reactive APF): 정책과 무관하게 물리적으로 회피.
    #   WP 추종 target 에 '가까운 아군/모선에서 멀어지는' 척력 오프셋(R−d, m)을 더해
    #   pd_follow 가 회피 경로를 타게 한다 → 보상 shaping(학습 의존) 위의 hard baseline.
    #   영향반경(R)·가중은 RewardCfg.avoid_r / mother_avoid_r / mother_avoid_w 공유.
    avoid_steer:      bool  = True    # 안전층 ON/OFF
    avoid_steer_gain: float = 1.2     # 오프셋 스케일 (×(R−d) m). ↑면 더 일찍/세게 회피.
                                      #   1.2/cap800 에서 휴리스틱 충돌 0/0 (포획률 유지) 검증.
    avoid_steer_cap:  float = 800.0   # 오프셋 크기 상한 (m): 극단 target 방지
    # ── ★ WP-레벨 척력(계획층): 몸체가 아니라 '계획된 WP들끼리' 밀어내 plan 을 벌린다.
    #   다른 배의 WP 쌍만 (R−d) 척력 → 결정마다 몇 번 변위(force-directed). 같은 배 내부 WP 는
    #   부채꼴 형태 보존 위해 제외. 동결 WP 는 commit 마스크가 보존. 몸체는 순수 PD 추종.
    wp_repulsion:   bool  = False   # WP 척력 ON/OFF (학습 기본 OFF; 뷰어서 켬)
    wp_repel_r:     float = 600.0   # 아군 WP↔WP 영향반경 (m)
    wp_repel_gain:  float = 0.35    # 변위 스케일 (×(R−d) m), 반복당
    wp_repel_iters: int   = 3       # 변위 반복 횟수
    wp_repel_mother_r: float = 600.0  # ★ 모선도 WP 를 밀어냄: 이 반경 안 WP 를 모선 밖으로 척력
                                      #   (death-disk ally_mother_radius=300 + 여유 → 경로가 모선 안 지남)

    # ── 모델 백본 (Actor) ─────────────────────────────────────────────
    #   "deepset"=순열불변 풀링(현재·stateless) / "lstm"=2단(집합인코더→LSTMCell 시간메모리).
    #   lstm 은 결정 간 hidden 을 carry(POMDP: 적 위빙·궤적 기억). 체크포인트에 저장돼 load 시 복원.
    backbone: str = "lstm"
    # ★ 통합 self-attention 백본(관계추론·협조): DeepSet 풀링 대신 [own,아군,적클러스터] 토큰
    #   self-attention → own 토큰(관계반영 ego). LSTM 과 병용(attention=공간관계, LSTM=시간메모리).
    #   그리디 1:1 휴리스틱이 못 하는 공동 그물벽·역할분담 협조를 학습. 재학습 필요.
    attn_backbone: bool = False
    attn_heads:    int  = 4
    attn_layers:   int  = 2

    # ── 결정 / 경로 (지속 풀 경로 + 잔차 보정) ────────────────────────
    #   각 아군은 Kw개 WP의 '지속 경로(route)'를 가지며, 매 결정마다 처음부터 다시 만들지 않고
    #   기존 경로를 **작은 델타로 미세 보정**만 한다 (확확 바뀌지 않게). 경로는 순환 순찰.
    decision_period: int = 25
    # 매 결정 '절대 재계획'(잔차-앵커 모드 OFF). True 면 결정마다 현재 위치서 경로를 새로
    #   배치(simulator 휴리스틱처럼 적을 추적). 휴리스틱 BC/가이드의 cont 타깃이 일관돼 학습↑.
    absolute_replan: bool = True
    # ★ 휴리스틱 baseline + 정책 잔차(refine): env 가 매 결정 강한 휴리스틱 부채꼴을 경로
    #   baseline 으로 깔고, 정책은 그 위에 ±wp_adjust_max 잔차만 학습. 잔차=0 이면 휴리스틱
    #   (cap_rate≥0.86) 보장 → GRPO 가 그 위를 개선. (정책이 부채꼴을 재현하는 취약성 회피.)
    #   그물 leg(net_mask)는 휴리스틱이 결정(견고). False 면 순수 정책(잔차/baseline OFF).
    heuristic_baseline: bool = True
    transit_wp:      int = 6      # 경로 WP 개수 (start→WP0..WP5). 정책이 4~6개 중 몇 개 따라갈지 결정
                                  #   (n_follow 액션) → 비효율/충돌 시 짧게 끊어 배회↓. 액션 변경 시 재학습 필수.
    route_step:      float = 360.0   # 초기 경로 WP 간격 (m). 240→360 (×1.5: 직선 구간 길게)
    # ★ WP 잔차 방식:
    #   "rigid"       = 전체 WP 동일 Δ 평행이동(형상보존, 요격선 통째 시프트). WP들이 똑같이 움직임.
    #   "independent" = 각 WP가 앵커에서 독립적으로 ±wp_adjust_max (기존 champion 방식, 자유도↓)
    #   "cumulative"  = 잔차를 체인 누적(cumsum): d_k 가 leg 벡터를 조정, 앞 d가 뒤 WP를 같이 끌어
    #     경로가 일관되게 휘고 먼 WP 도달범위↑(±k·wp_adjust_max). d=0 이면 동일하게 휴리스틱.
    wp_residual_mode: str = "independent"   # ★ GIF 시점 재편: 각 WP 독립 ±wp_adjust_max (cumulative→independent)
    # ── ★ A) Moving-anchor (TR-DPO): base 를 '고정 휴리스틱'이 아니라 '정책 경로 EMA'로 점진 이동.
    #   base = (1-w)·휴리스틱 + w·anchor_route, anchor_route ← (1-α)·anchor + α·정책경로(매 결정).
    #   trust-region(잔차 ±wp_adjust_max) 안정성은 유지하되 anchor 가 정책 따라 이동 → 휴리스틱서 발산(자유도↑).
    moving_anchor:   bool  = False   # A 활성화 (heuristic_baseline=True 에서만 의미)
    anchor_weight:   float = 0.5     # base 에서 anchor(정책경로 EMA) 비중 (0=순수 휴리스틱, 1=순수 anchor)
    anchor_alpha:    float = 0.15    # anchor 경로 EMA 속도(정책 추종). 0.05→0.15: 이기는 곳서 더 빨리 결박 해제
    wp_adjust_max:   float = 500.0   # 결정당 WP 잔차 스케일 (m). 468.75→500 (자유도↑, 휴리스틱 이탈 여지↑).
    # ★ 추종 중인 현재 WP(=ptr)도 매 결정 변동 허용. 기본 False=현재 WP 고정(과거+현재 고정, 미래만 적응
    #   → 훅훅 점프 억제, wp-jump-and-assign-stickiness). True=현재 WP도 잔차 반영(반응성↑·지터 위험).
    free_current_wp: bool  = False
    # ── 부채꼴 구조화 행동공간 (structured_action=True 시 fan[7], False=기존 잔차 보존) ──
    #   fan=[bearing, r_near, r_far, spread, aux_radial, aux_lateral, curve] ∈ [-1,1]. fan=0 → 휴리스틱.
    #   무구조 잔차의 anchor 딜레마(자유도↑=붕괴)를 '파라미터별 하드앵커'로 해소(방위 타이트·형상 자유).
    structured_action: bool  = False  # True=부채꼴 7파라미터, False=잔차(기본). 부채꼴 학습 시 True+transit_wp=5.
    net_radial_frac:   float = 0.0    # 0=한 반경 수직벽, 1=near→far 대각선(그물 형상)
    fan_bearing_max:   float = 25.0   # [0] 방위 회전 ±deg (far서 작은각=큰 요격점이동, 고레버리지)
    fan_rnear_amp:     float = 0.5    # [1] r_near ×(1±amp). 0.25→0.5: r_near 레버리지 강화(영향력 증대)
    fan_rfar_amp:      float = 0.35   # [2] r_far ×(1±amp) (더 멀리 요격)
    fan_spread_amp:    float = 0.40   # [3] 측면 스윕폭 ×(1±amp)
    fan_curve_max:     float = 400.0  # [6] 곡률: 그물을 호(arc)로 휨 ±m
    aux_radial_max:    float = 500.0  # [4] 보조WP 반경 이동 ±m
    aux_lateral_max:   float = 450.0  # [5] 보조WP 측면 이동 ±m
    fan_adjust_max:    tuple = (0.12, 0.35, 2.0, 2.0, 2.0, 2.0, 2.0)  # 파라미터별 하드앵커 캡(방위 타이트·형상 자유)
                                     #   independent=절대 한계, cumulative=leg 증분 한계.
    net_dir_adjust:  float = 0.15    # 그물 방향 잔차 보정 스케일 (정규화)
    max_steps:       int = 2000  # 맵↑·속도↓ 로 에피소드 길어짐 → 상향

    # ── 렌더링 (시각화 전용, 물리 무관) ───────────────────────────────
    ship_len:    float = 230.0   # 아군 선박 길이 (m)
    ship_wid:    float = 76.0    # 아군 선박 폭 (m)
    enemy_size:  float = 125.0   # 적 선박 기준 크기 (m)
    moback_size: float = 380.0   # 모선(항공모함) 기준 크기 (m). 길이≈1.9배·폭≈0.62배 → 웅장
    moback_heading: float = 0.0  # 모선 함수 방향 (0°=North=위). 웅장하게 정렬

    # ── 관측 정규화 (BoatAttack 패리티 — k=관심반경, 10km 스케일에 맞춤) ──
    #   norm_range/norm_close 의 k(=거리 0.5 지점). Unity normK 를 우리 맵으로 재스케일.
    norm_k_enemy:  float = 1500.0   # 적/클러스터 거리 (요격 임계 반경)
    norm_k_mother: float = 1000.0   # 모선 거리 (방어 근접)
    norm_k_ally:   float = 1000.0   # 아군쌍 거리
    n_clusters:    int   = 4        # 적 각도 클러스터 최대 수 (적응형 gap 클러스터링, ≤K개 사용)
    # ★ 배정 학습(b): 정책이 배별 클러스터 선호(Categorical) → _compute_assignment cost 를 soft 조정.
    #   휴리스틱 배정(greedy+sticky)을 base 로, 정책 선호만 얹어 협조 배정 학습(붕괴 회피). w_assign_bias=RewardCfg.
    learn_assign:  bool  = False
    # ★ WP 순회방향 학습: 정책이 배별 정/역방향(Bernoulli) 선택 → 역방향이면 route·net leg 를
    #   Kw축으로 뒤집어 거꾸로 순회. 첫 결정에 확정·동결(중간 변경 X). 아군끼리 경로 겹침/그물 중복
    #   (raycast redundancy) 완화 기대. 관측에도 노출(서로 방향 인지). heuristic_baseline 에서만 적용.
    learn_wpdir:   bool  = False
    # ★ route rotate 행동: 완성된 route 전체를 시작점(route[0]) 기준 ±rot_max_deg 만큼 회전(연속 스칼라).
    #   그물 벽도 같이 회전(레그 순서·시작점 불변→net-laying 안 깨짐). 부채꼴 방위 미세조정 → 아군 경로
    #   분리·redundancy 완화(역방향 wpdir 의 무손상 대체재). 매 결정 적용(래칭 X).
    learn_rot:     bool  = False
    rot_max_deg:   float = 15.0      # rotate 최대각(deg). tanh(rot)·rot_max_deg
    cluster_gap_deg: float = 11.97  # 적응형 클러스터 분할 임계 각도(deg). 무리 사이 gap 이 이보다
                                    #   크면 별도 클러스터로 분리; 작으면 한 무리=한 클러스터(중복 배정·
                                    #   고정빈 경계 분할 방지). ↑면 더 잘 합쳐짐(클러스터↓), ↓면 더 잘게.

    # ── 그물 위치 관측(net radar) — own 채널에 추가 ────────────────────
    #   각 배 헤딩기준 D방향 레이를 쏴 '설치된 그물(net_installed)까지 근접도'를 관측.
    #   정책이 자기/타선이 깐 그물을 *보고* 경로가 가로지르지 않게 하는 신호(net_touch 저감용).
    #   norm_close 규약(가까울수록 1, 탐지범위 내 그물 없으면 0). D=0 이면 비활성(구 관측 호환).
    net_probe_dirs:  int   = 0       # 레이 방향 수(헤딩 기준 360°/D 간격, 레이0=정면). 0=비활성(obs 슬림화 ablation)
    net_probe_range: float = 600.0   # 레이 최대 탐지거리(m). wp_max_len(450)보다 약간 김(앞 leg 인지)

    # ── 배경 지도 (위성영상 타일) ────────────────────────────────────
    #   앵커(lat,lon) 중심 world_size 박스를 배경으로.
    #   ★ 남해(통영 매물도·망태봉 근해): 중앙·좌측은 외해(물), 우측에 매물도 섬들(한려해상).
    use_basemap:   bool  = True
    geo_lat:       float = 34.625   # 남해 통영 매물도 서측 외해: 중앙=물, 우측에 매물도/망태봉 섬
    geo_lon:       float = 128.52
    basemap_zoom:  int   = 13       # 슬리피맵 줌 (높을수록 상세, 타일 수↑)

    # ── 난수 ──────────────────────────────────────────────────────────
    seed: int = 0

    def __post_init__(self):
        if self.n_allies > self.max_pairs:
            self.max_pairs = self.n_allies

    # ── 파생값 ────────────────────────────────────────────────────────
    @property
    def cell_size(self) -> float:
        """격자 한 cell의 물리 크기 (m). 액션·해상도 분리의 변환 상수."""
        return self.world_size / self.grid_size

    @property
    def wp_max_len(self) -> float:
        """WP 간 최대 간격 (m) = 그물 길이. 연속 WP 세그먼트는 이 길이를 못 넘는다.
        (단일 소스: net_max_len 과 항상 동일하게 강제.)"""
        return self.net_max_len

    @property
    def enemy_speed(self) -> float:
        """적 절대 속도 (m/step) = 아군 × 배수."""
        return self.ally_speed * self.enemy_speed_mult

    @property
    def center(self) -> tuple:
        """모선 위치 (맵 정중앙)."""
        return (self.world_size * 0.5, self.world_size * 0.5)

    # ── 직렬화 (체크포인트/export 호환) ───────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["cell_size"] = self.cell_size
        d["enemy_speed"] = self.enemy_speed
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class RewardCfg:
    """보상 설정 — Unity DefenseRewardCalculator 패리티 + GRPO용 포텐셜 shaping.

    GRPO는 후보당 스칼라 1개(윈도우 누적 return)를 쓰므로, 패리티 이벤트(정답)에
    potential-based shaping(Φ)을 더해 25스텝 윈도우에도 변별 가능한 밀집 신호를 만든다.
    Φ shaping 은 potential-based(γΦ(s')−Φ(s)) 라 최적 정책을 바꾸지 않는다.
    """
    # ── Unity 패리티 이벤트 ───────────────────────────────────────────
    #   ★ 안정화: rank 기반 advantage 라 '순서'만 중요 → 통제불가한 큰 이산 spike
    #     (돌파/잔존)를 줄이고, 통제가능한 dense 신호(Φ·placement)에 비중을 옮긴다.
    r_capture:        float = 13.5    # 포획 (팀). ★9→13.5(1.5배): 잡기 신호 추가 강화
    r_capture_indiv:  float = 0.3     # 포획한 배 개별 보너스 (joint 모드 전용; 현재 미연결)
    r_wipeout:        float = 8.0     # 전멸 (종료)
    r_breach:         float = -8.0    # 돌파. -6→-8: wave 미스 그래디언트 날카롭게(커버 붕괴 시 페널티↑)
    r_survive:        float = -5.0    # 종료 시 잔존 적 1척당
    r_ally_collision: float = -30.0   # 아군-아군 충돌. ★-100→-30(완화): spike 분산↓로 후반 안정화
    r_obstacle:       float = -30.0   # 아군-모선 충돌. ★-100→-30(완화): spike 분산↓로 후반 안정화
    r_net_touch:      float = -13.0   # 설치된(완성) 그물에 아군이 닿음. -10→-13: net_touch 비중 소폭↑
    time_penalty:     float = 0.0     # 스텝당 (제거: raycov 실험서 시간압 배제)

    # ── 경로 / 효율 / 그물 / WP 품질 ──────────────────────────────────
    w_path:    float = 0.5     # 경로효율(이동거리/시간) 페널티 가중
    w_net:     float = 0.3     # 그물 1개당 비용 (낭비 최소화: 포획 +5 보다 훨씬 작아 잡으면 이득).
                               #   1.0→0.3 완화: 그물 미배치(net_go=0) 붕괴 방지
    w_wp_good: float = 0.2     # WP 품질 보너스(도달가능·커버)
    w_wp_bad:  float = 0.2     # WP 품질 페널티(도달불가·중복)

    # ── GRPO 밀집화: 위협 포텐셜 shaping ──────────────────────────────
    #   Φ=근접도 합의 음수 → 적을 멀리 두거나 잡으면 ↑. 돌파의 '매끄러운(dense)' 대체 신호.
    #   1.0→3.0: 돌파 spike 대신 이 연속 신호가 방어의 주(主) gradient 가 되게 한다(매 윈도우 변별↑).
    w_threat: float = 3.0      # Φ 가중 (살아있는 적의 모선 근접도 합)
    gamma:    float = 0.99     # shaping 할인 (γΦ(s')−Φ(s))
    # ── ★ 레이캐스트 coverage 보상: 적→모선 ray가 그물에 막힌 적 비율 (방어 성공 직접 척도) ──
    #   포획의 선행지표(막힌 ray=잡힐 예정) → dense·원리적 신호. ray만 막으면 보상 차 → 그물 과설치
    #   유인↓ → net_touch 저감 기대. (배를 특정위치로 밀지 않아 회랑 이탈 없음.)
    w_coverage:  float = 0.1   # coverage 가중. 3.0→0.1: 잉여 shaping(휴리스틱이 이미 최대) 축소
    coverage_rays: int = 16    # ray 당 샘플 점 수
    # ── ★ 단일커버 레이캐스트: 적→모선 ray 를 아군 계획경로/그물이 '정확히 1대'로 막을 때만 보상 ──
    #   2대 이상 중복 차단 = 무효(0): 경로 겹침=충돌 예정 → 중복배정 억제 + 협조 분산을 직접 학습.
    #   그물 leg 로 막으면 raycov_net_bonus 배 우대. (휴리스틱이 못 잡는 '비중복 협조' niche)
    w_raycov:    float = 1.6    # 단일커버 레이캐스트 가중. 1.0→1.6: wave 커버 붕괴 직접 방어(성공 선행지표)
    raycov_net_bonus: float = 1.3  # 그물 차단 우대 배율
    #   redundancy(무효) = 두 아군 그물벽이 같은 ray 를 **충돌거리만큼 가까이** 가로지를 때만.
    #   (단순 '2대가 ray 통과'는 다층 방어(wave)도 벌점 → 충돌근접만 무효로 함. ship_len=230 기준)
    raycov_collide_m: float = 400.0
    # ── ★ 잔차 앵커(residual anchor): wp 잔차 크기 페널티 → 잔차=0(휴리스틱)을 바닥으로 ──
    #   휴리스틱이 거의 최적이라, 정책이 capture를 실제로 올릴 때만 deviate(앵커>capture gradient).
    #   드리프트로 휴리스틱보다 나빠지는 것(매 run 반복된 RL<휴리스틱) 방지 = trust-region.
    w_anchor:    float = 1.5    # mean(wp_residual²)∈[0,1] 페널티 가중 (0=비활성). 완화는 표류로 퇴행 확인→복원

    # ── 그물 배치 shaping + 경로 효율 (dense, 배치 gradient 직접 부여) ──
    #   ★ 품질중심 재설계: 휴리스틱이 이미 포획·정렬을 최대화 → place/approach/idle 은 정책이
    #     개선여지 없고 노이즈만 → 가중 대폭↓. 대신 정책 niche = '궤적 품질'(부드러움·마진·효율) 강화.
    w_place:     float = 0.1     # 0.3→0.1: 잉여 shaping 축소(휴리스틱이 이미 정렬 최대)
    place_scale: float = 300.0   # 경로선 수직거리 스케일 (m)
    w_eff:       float = 0.1     # 0.5→0.1: 잉여 shaping 축소
    # ── ★ 궤적 품질(정책 niche): 부드러움 + 그물 여유마진 + 시간 일관성 (포획 gate 보존 위해 작게) ──
    #   재균형: 품질 tier(0.3~0.5)로 통일. capture(5)·breach(8)보다 작아 tie-breaker로만 작동.
    w_smooth:    float = 0.3     # 선회량 페널티. 0.9→0.3: 셀 이산점프 재배치 자유(wave)
    w_clear_net: float = 0.0     # 설치 그물 근접 페널티. 제거: net_touch(-13)+Voronoi가 백스톱
    clear_net_r: float = 300.0   # 그물 여유마진 영향반경 (m, 충돌반경 밖 buffer) (champion 조건)
    w_consist:   float = 0.3     # ★ 결정 간 경로 일관성: ‖route[t]−route[t−1]‖/wp_max_len 페널티
                                 #   → 매 결정 경로가 튀지 않게(일관 추종). jittery 잔차 억제.

    # ── 클러스터 배정(assignment) 기반 dense 보상 ─────────────────────
    #   매 결정마다 위협 큰 클러스터를 '가장 가까운(효율적) 배'에 1:1 배정한다.
    #   배정된 배 = 자기 클러스터 교점으로 향하고(=경로 보상) 막으면(=place 보상) ↑.
    #   미배정 배 = 예비. 움직이지 않으면(정지 유지) ↑ → '클러스터 수만큼만' 출격.
    # ★ 배정 학습(b) soft-bias 강도: 정책 선호 클러스터의 배정 cost 차감(m). sticky_bonus 와 동급 스케일.
    w_assign_bias:      float = 2000.0
    w_assign_path:      float = 0.1   # 0.3→0.1: 잉여 shaping 축소(휴리스틱 중복)
    w_idle:             float = 0.2   # (0.5→0.2) 예비 배 정지 — 축소
    assign_intercept_t: float = 0.55  # 클러스터centroid→모선 교점 위치 비율(0=적,1=모선).
                                      #   아군이 적보다 느려, 모선에서 반경 ~2.4km 안쪽이라야
                                      #   적보다 먼저 도달 가능 → 0.55(모선寄り)로 차단 실현성 확보.
    # ── 충돌 회피 dense 척력(APF, Khatib) — 충돌 '전에' 떨어지도록 부드러운 그라디언트 ──
    #   기존 r_ally_collision/r_obstacle 는 충돌 '후' sparse −10. 이건 가까워질수록 ↑ 하는
    #   거리기반 페널티(½·η·(R/d−1)², d<R)로 회피를 학습시킨다(아군-아군 + 모선).
    w_obstacle:        float = 0.25  # 척력 페널티 전체 가중
    avoid_r:           float = 600.0 # 아군-아군 영향반경 R (m, 이 안에서만 척력)
    avoid_eta:         float = 0.6   # 척력 세기 η
    avoid_dmin:        float = 90.0  # 거리 하한 (m): 1/d 발산 방지 (≈충돌반경 가까이)
    mother_avoid_r:    float = 500.0 # 모선 death-disk 가장자리 밖 영향반경 (m)
    mother_avoid_w:    float = 2.0   # 모선 척력 가중 (아군-아군 대비; 모선 충돌이 더 치명)
    # ── 배정 안정화(stickiness): 매 결정 배정이 요동치면(churn) 배가 반쯤 깐 부채꼴을 버리고
    #   다른 클러스터로 재이동 → 자기 벽 횡단(self-touch)·포획 손실. 직전 클러스터가 여전히
    #   타겟이면 그 쌍 cost 를 낮춰 유지 → churn↓·포획↑(실측 churn 26→9, captures 1906→1984).
    #   (참고: '배p→WP0 직선이 그물 지나면 페널티'식 횡단-회피 배정은 실측상 churn↑·역효과라 폐기.)
    assign_sticky_bonus:      float = 6000.0 # 직전 배정 유지 보너스 (m; cost 차감, 0=비활성). 포화점(~6000)
                                              #   까지 빡세게: 정책 churn 17.4→5.8%(_prev_assign 연결 수정과 함께).

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardCfg":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class GRPOCfg:
    """GRPO 학습 설정 (critic 없음)."""
    num_worlds:   int = 512
    k_samples:    int = 16        # 결정당 후보 수 (그룹 크기)
    eval_period:  int = 100       # 후보 반사실 평가 horizon (>decision_period: 플랜 결과가 드러나도록)
    updates:      int = 600      # 결정-업데이트 횟수 (기본 총 에폭)
    lr:           float = 1.5e-4  # 안정 학습 (3e-4 → 후반 진동 완화)
    init_log_std: float = -1.6    # 탐색 표준편차 초기값 (exp≈0.20). −1.2→−1.6: 누적잔차 탐색폭주 억제
                                  #   (강한 휴리스틱 경로 덜 교란 → cap_rate가 잔차=0 baseline 회복·수렴)
    init_netgo_bias: float = 1.0  # net_go 전개 쪽 초기 바이어스(붕괴 완화). ↑일수록 초기 전개↑
    grad_clip:    float = 1.0
    ent_coef:     float = 0.01    # 엔트로피 보너스 초기값 (0.02→0.01: std 수렴 허용)
    ent_coef_end_ratio: float = 0.1   # 엔트로피 스케줄: 끝에서 초기값의 이 비율로 감쇠 (0.25→0.1)
    d_model:      int = 128       # Actor 임베딩 차원
    ma_mode:      str = "counterfactual"  # "joint" | "counterfactual"(배별 신용)
    # ★ advantage 산정: "rank"(K축 순위표준화) | "heur_rel"(후보0=휴리스틱 기준 A_k=(R_k−R_heur)/std)
    #   heur_rel: 모방=0, 휴리스틱 초과시만 + → 휴리스틱 천장 돌파 유도. heuristic_candidate 필수.
    adv_mode:     str = "rank"
    var_eps:      float = 1e-6    # 그룹 분산≈0 마스킹 임계
    # 커리큘럼: 적 스폰 거리 frac 을 spawn_frac0 → 1.0 (anneal_frac 구간 동안)
    curriculum:   bool = True
    spawn_frac0:  float = 0.35
    anneal_frac:  float = 0.8     # 쉬운 난이도를 더 오래 숙달 후 5km (0.6→0.8)
    save_every:   int = 100       # 체크포인트 주기(업데이트). 중간에 멈춰도 모델 남음
    log_every:    int = 20
    seed:         int = 0
    # 휴리스틱 가이드: 매 결정 K후보 중 1개를 휴리스틱(배정+부채꼴) 액션으로 교체.
    #   rank-advantage 가 휴리스틱 우수 시 정책을 그쪽으로 당김(초기 가이드→자연 소멸).
    heuristic_candidate: bool = True
    # 행동복제(BC) 워밍업 스텝(0=끔). heuristic_baseline=True 면 잔차=0 이 이미 휴리스틱급이라
    #   불필요(기본 0). heuristic_baseline=False 인 순수 정책 모드에서만 의미.
    bc_warmup_steps: int = 0
    bc_lr:           float = 3e-4


# 기본 설정 인스턴스 (간편 import용)
DEFAULT_CONFIG = SimConfig()
DEFAULT_REWARD = RewardCfg()
DEFAULT_GRPO = GRPOCfg()
