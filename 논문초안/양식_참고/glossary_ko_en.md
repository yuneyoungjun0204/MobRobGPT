# 영문 번역 용어표 (KR → EN) — `USV_swarm_defense_EAAI_KR_real`

영문 원고 작성 시 **고정**할 용어. EN 열은 이미 그림 라벨(`boatattack_sim/eval/paper_figs*.py`)·영문 초록·키워드에 쓰인 표기와 맞췄다.
규칙: 한 개념에 하나의 영문명. 약어는 첫 등장에서 풀어 쓰고 이후 약어만. 조건 태그(`heur_unet` 등)는 코드 식별자이므로 번역하지 않는다.

## 1. 체계·계층

| KR | EN | 비고 |
|---|---|---|
| 방어정 | defender (USV) | 문맥상 "defending USV"도 가능. 적선은 attacker |
| 적선 / 적 USV | attacker / attacking USV | "enemy"는 그림 라벨(enemy presence)에만 |
| 모선 | mothership | 삽화의 항공모함은 "carrier glyph"로 설명 |
| 자폭 무인수상정 군집 | swarm of low-cost suicide USVs | 초록 첫 문장 고정 |
| 두 계층 방어 체계 | two-layer defence architecture | 영국식 defence (EAAI Elsevier 어느 쪽이든 일관성만) |
| 전략 계층 | strategic layer | = commander layer와 동의어로 쓰지 않는다 |
| 기동 계층 | maneuver layer | 그림 라벨 "Maneuver only" 와 일치 |
| 지휘관 / 언어모델 지휘관 | commander / LLM commander | §4.5 |
| 배정 (담당 무리 배정) | assignment (cluster-to-defender assignment) | 동사: assign |
| 기동 (그물벽 위치 결정) | maneuver (net-wall placement) | |
| 비동기 분리 | asynchronous decoupling | §4.1 |
| 결정 주기 | decision period (25 s) | step = 1 s |
| 재계획 (주기) | replanning (interval) | `--replan-every`; 재계획 간 → between replans |
| 온보드 배치 | onboard deployment | §1.3 |
| 로컬 / 클라우드 지휘관 | local / cloud commander | Qwen2.5 7B·14B (local), Gemini (cloud) |
| 능력 임계 | capability threshold | §6.2, 논제 문장 |
| 코드 휴리스틱 / 기하 휴리스틱 | rule-based (geometric) heuristic | 기준선 조건 `heur_heur` |
| 기준선 | baseline | heuristic baseline |

## 2. 그물·교전 규칙

| KR | EN | 비고 |
|---|---|---|
| 그물벽 | net wall | 그림에는 텍스트로 쓰지 않음(사용자 지시) |
| 부유식 그물 | floating net | Fig 3 |
| 후미 투하식 전개 장치 | stern-launched net deployment system | Fig 3 개념도 |
| 그물 전개 | net deployment | 동사: deploy |
| 프로펠러 얽힘 | propeller entanglement | 포획 기구 |
| 포획 | capture | 판정: captured on entering a painted net cell |
| 돌파 | breach | 모선 260 m 진입 |
| 접근 회랑 | approach corridor | |
| 요격점 / 요격 구역 | intercept point / intercept annulus | annulus [400, 4500] m |
| 선제 차단 | pre-emptive interception (blocking) | "추격 대신" contrast: instead of pursuit |
| 공격 포메이션: 집중·양동·파상 | concentrated / diversionary / wave | 그림 라벨과 동일, 소문자 |
| 단(파상의) | rank (of a wave attack) | |
| 적 무리 | (enemy) cluster | 군집화 → clustering |
| 유효 마스크 / 유효 픽셀 | validity mask / feasible cells (pixels) | 그림 라벨 "Feasible cells" |
| 방위 게이트 / 반경 게이트 | bearing gate / range gate | ±60°, 3 km |
| 육지 마스크 | land mask | OSM |

## 3. 정책·학습

| KR | EN | 비고 |
|---|---|---|
| 점수맵 | score map | 하이픈 없이 |
| 픽셀 지목 | pixel pointing (pointing to a pixel) | 포인터 네트워크 계보 |
| 다채널 격자 관측 | multi-channel grid observation | 15 channels |
| 좌표 채널 | coordinate channels (CoordConv) | |
| FiLM 조건화 | FiLM conditioning | |
| 부화소 오프셋 | sub-pixel offset | |
| 격자 상대 좌표 | grid-relative coordinates | Sim2Real 논거 |
| 경유점 | waypoint | 두 개 → two waypoints |
| 가치망 없는 그룹 상대 정책 최적화 | critic-free group-relative policy optimisation (GRPO) | |
| 그룹 상대 이득 | group-relative advantage | |
| 휴리스틱 후보 (k = 0) | heuristic candidate | 그룹 안 기준 후보 |
| 롤아웃 창 | rollout window (H = 180 steps) | |
| 반사실적 크레딧 배분 | counterfactual credit assignment | COMA 대비 |
| 결합(joint) 크레딧 배분 | joint credit assignment | 채택 모드 |
| 행동 복제 초기화 | behaviour-cloning (BC) warm-up | 300 updates |
| 코사인 어닐링 | cosine annealing | 학습률 |
| 엔트로피 계수 | entropy coefficient | |
| 근사 KL 발산 | approximate KL divergence | 수렴 진단 |
| 상대 파라미터 이동 | relative parameter shift | |
| 채택 런 / 재현 런 | adopted run / replication runs (3 seeds) | |
| greedy 평가 | greedy evaluation | |
| 시드 대응표본 | seed-paired samples | paired by seed |
| 시드 대응 차이 | seed-paired difference | 그림 라벨 "Δ capture rate (seed-paired)" |

## 4. 지표 (Table 2 순서, `paper_figs_extra.METRIC_EN`과 일치)

| KR | EN | 방향 |
|---|---|---|
| 포획률 | capture rate | ↑ |
| 돌파 수 | breaches | ↓ |
| 충돌률(척당) | collision rate (per defender) | ↓ |
| 그물 걸림 | net entanglements (own-net) | ↓ |
| 포획당 그물 | nets per capture | ↓ |
| 포획당 이동거리 | distance per capture | ↓ |
| 총 선회량 | total turning | ↓ |
| 포획 이격거리 | capture distance (from mothership) | ↑ |
| 평균 포획 시각 | capture time | ↓ |
| 전개 시 적 거리 | enemy distance at deploy | ↓ |

## 5. 계획 품질·지연

| KR | EN | 비고 |
|---|---|---|
| 배정 커버리지 | assignment coverage | |
| churn (재계획 간 담당 변경 비율) | assignment churn | 그대로 churn |
| 경로 교차 | crossing assignments | |
| 휴리스틱 일치율 | agreement with heuristic (assignment) | Fig 19 |
| 계획 안정성 | plan stability | 설계 원리 문장 |
| 응답 지연 | response latency | ECDF 그림 |
| 폴백(률) | fallback (rate) | 스키마 실패 → 휴리스틱 대체 |
| 구조적 출력 | structured output (JSON schema) | |
| 판단 근거 | rationale | |

## 6. 통계·설계

| KR | EN | 비고 |
|---|---|---|
| 2×2 조건 행렬 | 2×2 condition matrix | |
| 상호작용 | interaction (term) | 가산적 → additive |
| 가산적 | additive | 상승 → synergistic (부정에만 사용) |
| 효과크기 d_z | paired effect size d_z (Cohen's d_z) | |
| BCa 부트스트랩 신뢰구간 | BCa bootstrap confidence interval | |
| Bonferroni 보정 | Bonferroni correction (10 metrics) | |
| 천장 효과 | ceiling effect | 집중 포메이션 |
| 시뮬레이터 충실도 | simulator fidelity | Table 5 |
| 재현성 | reproducibility | Data availability |
