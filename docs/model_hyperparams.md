# U-Net 점수맵 정책 — 모델 구성 · 하이퍼파라미터 (발표자료용 표)

대상: `boatattack_sim/models/u-net_map.pt` (974 KB)
= CNN 저장소 `boatattack_sim/models/run_20260819-203248/best.pt` (md5 동일)

**출처 — 창작한 숫자 없음.**

| 항목 | 출처 |
|---|---|
| 신경망 구조 · 파라미터 수 | 체크포인트 `state_dict` 직접 집계 + `boatattack_sim/model/cnn_actor.py` |
| 관측 · 행동 · 시뮬 | 체크포인트에 저장된 `config` 174키 |
| 학습 설정 | `CNN/boatattack_sim/train/grpo_cnn.py` · `env/config.py::GRPOCfg` |
| 실제 실행값 · 결과 | 학습 로그 `CNN/_logs/dm_C_w3.log` (이 체크포인트를 낸 로그) |

---

## 1. 신경망 구조

**총 파라미터 244,389** (약 0.24 M)

| 모듈 | 구성 | 파라미터 | 비중 |
|---|---|---:|---:|
| `net` (UNetLite) | 아래 표 | 235,904 | 96.5 % |
| `own_mlp` | MLP `[8 → 128 → 32]`, GELU | 5,280 | 2.2 % |
| `off_mu` | MLP `[64 → 32 → 2]`, GELU (서브픽셀 오프셋) | 2,146 | 0.9 % |
| `q_proj` | Linear `32 → 32` | 1,056 | 0.4 % |
| `off_logstd` | 학습 파라미터 (2,) | 2 | — |
| `log_std` | 로깅 호환용 더미 (손실 무관) | 1 | — |

### 1.1 UNetLite 백본

기본 블록 `_cbr` = `Conv2d(3×3, bias=False)` → `GroupNorm(8)` → `SiLU`

| 단계 | 연산 | 해상도 | 채널 |
|---|---|---|---|
| 입력 | 래스터 스택 | 50 × 50 | **15** |
| `stem` | `_cbr(15→32)` + `_cbr(32→32)` | 50 × 50 | 32 |
| `enc1` | `_cbr(32→32, stride 2)` + `_cbr(32→32)` | 25 × 25 | 32 |
| `enc2` | `_cbr(32→64, stride 2)` + `_cbr(64→64)` | 13 × 13 | 64 |
| `bottle` | `_cbr(64→64)` × 2 | 13 × 13 | 64 |
| `film` | GAP → `Linear(64 → 128)` → (γ, β) FiLM 변조 | — | — |
| `dec2` | upsample(nearest) + skip(`enc1`) → `_cbr(96→32)` + `_cbr(32→32)` | 25 × 25 | 32 |
| `dec1` | upsample(nearest) + skip(`stem`) → `_cbr(64→32)` + `_cbr(32→32)` | 50 × 50 | 32 |
| `head` | `Conv2d(32→32, 1×1)` | 50 × 50 | **32** (픽셀별 key) |

- `cnn_width` (기본 폭 `w`) = **32**
- `cnn_d` (점수맵 feature 차원 `d`) = **32**
- 정규화 = GroupNorm(groups 8), 활성 = SiLU (MLP 만 GELU)

### 1.2 점수 · 행동 헤드

| 항목 | 값 |
|---|---|
| 점수 | `score = ⟨q_proj(q), feat⟩ / √d`, `d = 32` → `[B, 50, 50]` |
| query | `q = own_mlp(own[8])` → 32-d |
| 마스킹 | `valid == False` 인 픽셀은 `-inf` → spatial softmax |
| 선택 픽셀 수 `K` (`cnn_nets`) | **2** (두 점을 잇는 선분 = 그물 벽) |
| 자기회귀 | CNN 재-forward 없음. `feat` 재사용 + `q ← q + f_{k−1}` |
| 중복 배제 `cnn_dup_r` | **200 m** (2번째 픽셀은 1번째 반경 200 m 밖) |
| 도달성 상한 `cnn_reach_mask` / `slack` | **ON** / **1.5** → `450 × 1.5 = 675 m` |
| 서브픽셀 오프셋 `cnn_subpixel` | **ON**, ±½ 픽셀. 학습된 `std ≈ 0.304 px` (`off_logstd ≈ −1.19`) |
| 순환(LSTM) `cnn_recurrent` | **OFF** |
| 추론 | `greedy` 만 사용. `greedy_joint` 금지 (`cnn_gate_disjoint=False` 레짐) |

---

## 2. 관측 (입력 15채널)

격자 `cnn_grid_n = 50`, 반경 `cnn_extent = 6300 m` → **1 px = 252 m** (`2 × 6300 / 50`)

| 묶음 | 채널 수 | 이름 |
|---|---:|---|
| 전역 `gmap` (전 함정 공유) | **9** | `enemy_presence`, `enemy_vx`, `enemy_vy`, `enemy_threat`, `ally_presence`, `net_installed`, `origin`, `annulus`, `land` |
| 배별 `smap` | **3** | `self_marker`, `self_intercept`, `self_valid` |
| CoordConv (config 파생, 체크포인트 미저장) | **3** | `coord_x`, `coord_y`, `coord_r` |
| **합** | **15** | |

| 텐서 | 형태 |
|---|---|
| `gmap` | `[N, 9, 50, 50]` |
| `smap` | `[N, P, 3, 50, 50]` |
| `own` | `[N, P, 8]` |
| `valid` | `[N, P, 50, 50]` |

`own[8]` = `[(x−6300)/6300, (y−6300)/6300, sin(hdg), cos(hdg), 잔여그물/총, 부설중, (Ix−6300)/6300, (Iy−6300)/6300]`

### 2.1 행동 후보 마스크 `valid`

| 항목 | 값 |
|---|---|
| 환형 마스크 `cnn_mask_annulus` | ON |
| 육지 제외 `land_mask_action` | ON |
| 요격점 반경 게이트 `cnn_gate_r` | **3000 m** |
| 배정 방위 게이트 `cnn_gate_angle` | **±60°** |
| Voronoi 분할 `cnn_gate_disjoint` | **False** (겹침 허용 레짐) |
| 폴백 하한 `cnn_min_valid` | **150 px** (soft, 반경만 품) / hard = `K` = 2 |
| 실측 후보 수 | 학습 시 min 150 · median 273 / 2500 · 배포 시 평균 10.5 % |

---

## 3. 시뮬레이션 · 임무 설정

| 항목 | 값 |
|---|---|
| 월드 크기 `world_size` | 12,600 m |
| 지형 앵커 `geo_lat` / `geo_lon` | 34.625 , 128.52 |
| 아군 `n_allies` (P) | 3 |
| 적 `n_enemies` (M) | 10 |
| 클러스터 `n_clusters` | 4 (배포 시 3으로 오버라이드) |
| 아군 속도 `ally_speed` | **6.0 m/s** |
| 적 속도 `enemy_speed` | **9.0 m/s** (아군보다 빠름 → 추격 불가) |
| 아군 최대 선회 `ally_max_turn` | 8.0 °/step |
| 적 최대 선회 `enemy_max_turn` | 5.0 °/step |
| PD 이득 `ally_turn_gain` | 0.6 |
| 도달 반경 `arrive_radius` | 200 m |
| 결정 주기 `decision_period` | **25 step** |
| 시간 간격 `dt` | 1.0 s |
| 에피소드 상한 `max_steps` | 2000 step |
| 그물 최대 길이 `net_max_len` | **450 m** |
| 그물 폭 `net_width` | 4 셀 (`grid_size` 200²) |
| 배당 그물 `nets_per_ship` | **1** (학습) → 배포 시 3 |
| 모선 반경 `mothership_radius` | 260 m |
| 아군-모선 안전거리 `ally_mother_radius` | 300 m |
| 아군 충돌 반경 `ally_collision_radius` | 115 m |
| 적 스폰 반경 `enemy_spawn_radius` | 5,450 m |
| 육지 `land_obstacle` / `land_sites` | ON / `korea` 6개 사이트 로테이션 (평균 3.67 %) |
| 육지 임계 `land_threshold` | 0.30 |

---

## 4. 학습 — GRPO (critic 없음)

알고리즘: 그룹 상대 정책 최적화. 결정마다 후보 K개를 뽑아 **팀 보상으로 순위를 매기고**
advantage 를 그룹 안에서 표준화한다. value network 없음.

| 항목 | 값 | 비고 |
|---|---|---|
| 최적화기 | **Adam** | |
| 학습률 `lr` | **1.0 × 10⁻⁴** | `GRPOCfg` 기본 1.5e-4 → `grpo_cnn` 이 1e-4 로 덮음 |
| LR 스케줄 | **CosineAnnealingLR**, `T_max = updates`, `η_min = lr × 0.05` | 로그 확인: 1.00e-04 → 5.00e-06 |
| 업데이트 수 `updates` | **800** | |
| 병렬 월드 `num_worlds` | **24** | |
| 그룹 크기 `k_samples` (K후보) | **8** | 후보 0 = 휴리스틱 (`heuristic_candidate=True`) |
| 후보 평가 horizon `eval_period` | **180 step** | `decision_period` 25 보다 길게 — 계획 결과가 드러나도록 |
| advantage `adv_mode` | **`heur_rel`** | `A_k = (R_k − R_heur) / std` → 휴리스틱 초과분만 + |
| 멀티에이전트 신용 `ma_mode` | `joint` (기본값) | ⚠ 로그에 미기록 — 미확인 |
| 엔트로피 계수 `ent_coef` | **0.01** → 끝에서 `× 0.1` = 0.001 (선형 감쇠) | |
| 그래디언트 클립 `grad_clip` | **1.0** | |
| 그룹 분산 마스킹 `var_eps` | 1e-6 | 분산≈0 인 그룹은 학습에서 제외 |
| 커리큘럼 `curriculum` | **False** | 처음부터 최대 난이도 |
| 적 편성 `--enemy` | `rotate` | 포메이션 로테이션 |
| 시드 | 0 | |

### 4.1 행동복제(BC) 워밍업

| 항목 | 값 |
|---|---|
| 스텝 `bc_warmup` | **300** |
| 최적화기 / lr | Adam / **3 × 10⁻⁴** |
| 타깃 | 휴리스틱 픽셀 (가우시안 소프트 라벨) |
| 라벨 폭 `cnn_bc_sigma` | **0.8 px** |
| GRPO 중 BC 정규화 `bc_coef` | **0.0** (워밍업 후 BC 신호 없음) |
| 그래디언트 클립 | 1.0 |
| 로그상 손실 | 8.449 → 4.124 (300 step) |

### 4.2 보상 (`RewardCfg`) — 기본값 + dmin 2개만 변경

| 이벤트 | 가중 |
|---|---:|
| 포획 `r_capture` | **+13.5** |
| 전멸 `r_wipeout` | +8.0 |
| 돌파 `r_breach` | **−8.0** |
| 종료 시 잔존 적 1척 `r_survive` | −5.0 |
| 아군-아군 충돌 `r_ally_collision` | −30.0 |
| 아군-모선 충돌 `r_obstacle` | −30.0 |
| 아군-육지 충돌 `r_land` | −30.0 |
| 설치 그물 접촉 `r_net_touch` | −13.0 |
| 시간 페널티 `time_penalty` | 0.0 (끔) |

| Shaping / 비용 | 가중 |
|---|---:|
| 위협 포텐셜 `w_threat` (Φ = 적의 모선 근접도 합, potential-based) | **3.0** |
| shaping 할인 `gamma` | 0.99 |
| 경로 효율 `w_path` | 0.5 |
| 그물 1개 비용 `w_net` | 0.3 |
| WP 품질 보너스 / 페널티 `w_wp_good` / `w_wp_bad` | 0.2 / 0.2 |
| 레이 coverage `w_coverage` / `coverage_rays` | 0.1 / 16 |
| **모선 근접 `w_mother_dmin`** | **3.0** ← 이 실행에서 켬 |
| **아군 근접 `w_ally_dmin`** | **3.0** ← 이 실행에서 켬 |

`w_*_dmin` 은 25-step 윈도우 **최소거리** 기반: `prox = clip((영향반경 − d_min)/영향반경, 0, 1)²`
(붙을수록 급가중). 이 둘이 0 이 아니라서 `dmin_track = True` 로 켜졌다.

---

## 5. 결과 (학습 로그 기준)

| 지표 | 값 |
|---|---|
| 휴리스틱 baseline 포획률 | **0.744** (cap 9,663 / breach 3,320, 120 결정 × 3 시드) |
| 학습 중 best 포획률 | **0.805** (baseline 대비 **+0.061**) |
| 최종(upd 799) 포획률 | 0.73 |
| 저장 정책 | `best.pt` = best 시점 (0.805) |

> ⚠ 위 포획률은 **학습 루프 안의 평가**다. `results/` 의 독립 평가 수치와 직접 비교하지 말 것.

---

## 6. 배포 시 오버라이드 (`commander/unet_bridge.py`)

체크포인트 config 를 그대로 쓰되 다음 5개만 바꾼다. 나머지는 학습분포 유지.

| 키 | 학습값 → 배포값 | 이유 |
|---|---|---|
| `n_clusters` | 4 → **3** | LLM 지휘관이 최대 3그룹으로 다룸 |
| `spawn_phase_lo` | 0.6 → **1.0** | 웨이브 텀 설계대로 (스폰 랜덤 당김 끄기) |
| `nets_per_ship` | 1 → **3** | 1장 쓰면 `a_nets=0` → 정지 규칙에 걸려 배가 얼어붙음 |
| `land_sites` | `korea` → **`""`** | 운용 해역 고정 (로테이션 끔) |
| `mother_keepout` | — → **True** | 모선 전용 척력 (APF 는 OFF 유지) |

`own[4]` 는 배포에서 `min(a_nets, 1.0)` 으로 이진화한다 — 학습 때 항상 0/1 이었기 때문.
