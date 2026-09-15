# 9_신규제안 — 논문 완성도 보강용 신규 도판 후보

본문에 **아직 그림 없이 표·문장으로만 버티는 주장**에 시각적 근거를 붙이기 위한 후보들이다.
전부 기존 원자료에서만 만들었고(새 실험 없음), 스타일은 P1–P8 과 같은 저널판(Times, 137 mm 판형, 영문 라벨)이다.
아직 본문에 연결하지 않았다 — 어느 것을 쓸지 정한 뒤 `논문초안/figs_snak/` 로 복사해 넣는다.

| 파일 | 내용 | 원자료 | 붙일 자리(제안) | 뒷받침하는 주장 |
|---|---|---|---|---|
| `N1_latency_ecdf` | 지휘관 3종 응답 지연 ECDF(로그축) + 25 s 결정 주기선, 범례에 n 과 주기 초과 비율 | `llm_metrics.latency_s` | §4.1 또는 §6.9 (Table 12 옆) | 14B 의 4.2 % 초과 꼬리가 *분포* 로 보인다 → 비동기 설계 근거 |
| `N2_stability_time` | (a) churn, (b) 교차 배정의 에피소드 시간축 추이(50 step 빈, 모델별 평균 ±95 % CI) | `llm_metrics.t, churn, crossings` | §6.7 (Fig 19 옆) | 계획 안정성이 시간에 따라 어떻게 달라지는가 — 7B 는 후반으로 갈수록 churn 이 커진다 |
| `N3_paired_diff` | 포메이션별 시드 대응차이 점들 + 평균·BCa 95 % CI (기동 계층 단독 / 제안 체계) | `episodes` heur_unet, llm_unet@gemini vs heur_heur | §6.1 (Table 7 옆) | 파상 +0.157 이 소수 시드가 끄는 것인지, 양동의 −0.033 이 어떤 분포인지 |
| `N4_metric_corr` | 10지표 시드 대응차이(개선 방향 통일)의 Spearman 상관 히트맵, n = 90 | `episodes` 10지표 | §6.4 또는 §7.4 | 선회량·포획당 이동거리·포획률 개선이 같은 시드에서 함께 온다 — "효율 지표는 한 메커니즘" |
| `N6_agreement` | (a) 모델별 휴리스틱 일치율 평균±CI 와 완전 일치 비율, (b) 일치율 vs llm_heur 포획률 | `llm_metrics.heuristic_agreement` (본문 미사용 열) | §6.7 | 7B 는 휴리스틱과 *다르게* 배정해서 진다(일치율 0.39 vs Gemini 0.59) — 새 관찰 |
| `N7_rationale` | (a) 근거 길이 분포(모델별), (b) 근거 길이 6분위 vs churn | `llm_metrics.rationale_chars` | §4.5 또는 §8.3 | 7B 는 근거가 길수록 churn 이 커지고 Gemini 는 반대 — 근거 길이가 판단 품질의 대리 지표가 아님 |
| `N8_bc_warmup` | (a) 행동복제 워밍업 손실 3시드, (b) 초기 300 갱신 정책 엔트로피 | `train_diag/seed*/{bc,metrics}.csv` | §4.4.5 초기 학습 안정화 | 그림 없던 소절에 실측 근거 |
| `N9_scoremap_sharpness` | (a) 결정별 유효칸 비율 분포, (b) 점수맵 조건부평균↔최빈 거리 분포 | 체크포인트 재실행(`docs/diagrams/scoremap_sharpness.py`, 원자료 `results/scoremap_sharpness.npz`) | §4.3.3 이산 격자 논거 | "좌표 회귀가 수렴할 평균점과 정책이 고르는 점이 얼마나 다른가" 실측 |

만들 수 없었던 것: **N5 시간축 누적 포획 곡선** — 에피소드 내부 시계열 로그가 없다(`cap_time_mean` 요약값만 있음).
새 실험이 필요한 것(규모 일반화, 재계획 주기 민감도, 학습률 어블레이션, 구성요소 절제)은 여기 없다.

재생성:
```
python -m boatattack_sim.eval.paper_figs_extra      # N1–N8
python docs/diagrams/scoremap_sharpness.py          # N9 (npz 캐시 있으면 시뮬 생략)
```
