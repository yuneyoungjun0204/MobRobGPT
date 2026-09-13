# 학습 로그 — 논문이 평가하는 정책의 학습 런

원본: `One-Way_Towing/CNN/boatattack_sim/models/run_20260819-203248/`
이 런의 `best.pt` 는 `boatattack_sim/models/u-net_map.pt` 와 md5 동일(`833e0e39…`)이다.
즉 **논문의 모든 결과가 이 런에서 나온 가중치로 측정되었다.**

- `metrics.csv` — 20 업데이트마다 기록. upd / R(후보 평균 보상) / best(후보 최대 보상) /
  valid(후보 간 분산>0 인 월드 비율) / loss / cap / br / cap_rate
- `evals.csv`   — greedy 평가 (200·400·600 업데이트). baseline 0.744 는 휴리스틱 기동.
  600 업데이트의 0.805 가 최고점이고 그 가중치가 `best.pt` 다.

논문 도판(Fig. 학습 경과)은 `boatattack_sim/eval/paper_figs.py --train-run` 이 이 두 파일에서 굽는다.
