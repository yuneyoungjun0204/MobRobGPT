# 파이프라인 다이어그램 — Diagram as Code

[mingrammer/diagrams](https://github.com/mingrammer/diagrams) 로 그린 의사결정 파이프라인.
그림이 아니라 **코드가 원본**이다. 파라미터가 바뀌면 `pipeline_diagram.py` 를 고치고 다시 렌더한다.

## 산출물

| 파일 | 용도 | 크기 |
|---|---|---|
| `mobrobgpt_overview.png` | **발표 본문 슬라이드** — 2계층 구조 한 장 | 2550 × 783 |
| `mobrobgpt_pipeline.png` | **부록 / 포스터** — ①~⑦ 전 단계 상세 | 6216 × 1676 |

## 준비

```bash
pip install diagrams                                  # graphviz 파이썬 래퍼 포함
winget install --id Graphviz.Graphviz -e --silent     # dot 실행파일 (필수)
```

`dot` 이 PATH 에 없어도 스크립트가 `C:\Program Files\Graphviz\bin` 을 직접 붙인다.

## 렌더

```bash
python docs/diagrams/pipeline_diagram.py
```

## 그림의 구조

```
①  입력            센서(GPS/IMU/적 탐지) 또는 시뮬 물리 → 원시 상태
②  전장상태 구성    무리 분할 + 기하 선계산 → BattlefieldState
③  전략 계층        LLM 지휘관 → CommanderPlan → _assign[P]      ← 느린 비동기 (25 step)
④  관측 래스터화    gmap/smap/own/valid  (15 ch × 50 × 50)
⑤  기동 계층        U-Net 점수맵 정책 → 픽셀 2점 → route[P,2,2]   ← 매 결정 스텝
⑥  저수준 제어      PD 조타 + 그물 도색 + ptr 전진               ← 매 스텝 (dt = 1 s)
⑦  액추에이터       /ally_i/waypoints  또는  시뮬 상태 갱신
```

- **주황 굵은 선** = LLM 이 실제로 결정하는 경로 (③ → ④ 배정 주입)
- **남색 선** = 학습 정책의 순전파 (④ → ⑤)
- **자홍 선** = 결정론적 저수준 제어 (⑤ → ⑥)
- **회색 파선** = 되먹임 (다음 스텝 · `net_installed` · 배정 연속성)

## 레이아웃 관련 주의

Graphviz 는 클러스터끼리 **랭크를 공유하면 상자를 겹쳐 그린다.** 이 스크립트가 넣은 세 가지 보정:

1. 들어오는 엣지가 없는 노드(`prompt`, `fb`, `gmap`)는 rank 0 으로 밀리므로
   `Edge(style="invis")` 로 앞 단계 뒤에 고정한다.
2. 단계를 건너뛰는 먼 엣지(`raw → gmap`, `route → pub`)는 `constraint="false"` 로
   랭킹에서 제외한다. 의미는 유지되고 배치만 안 비튼다.
3. 가장 긴 되먹임(⑦ → ①)은 선을 긋지 않고 **off-page connector 쌍**으로 표시한다.

## 수치 출처

전부 저장소에서 읽은 값이다. 창작한 숫자는 없다.

- `boatattack_sim/models/u-net_map.pt` 의 config —
  `decision_period 25` · `arrive_radius 200` · `net_max_len 450` · `net_width 4` ·
  `ally_speed 6.0` · `cnn_grid_n 50` · `cnn_extent 6300` (→ 1 px = 252 m) ·
  `cnn_dup_r 200` · `cnn_min_valid 150` · `cnn_gate_r 3000` · `cnn_width 32` ·
  `ally_max_turn 8.0` · `ally_turn_gain 0.6` · `ally_mother_radius 300` · `dt 1.0`
- `boatattack_sim/env/defense_env.py::_micro` — PD 추종 · 그물 도색 · `ptr` 전진
- `boatattack_sim/model/cnn_actor.py` — `UNetLite` · `q·key` · `greedy(K=2)`
- `commander/{prompts,sim_bridge,fallback,rl_bridge,ros2_sensor_bridge}.py`

## PPT 에 넣기

`docs/project_ppt.md` 의 「핵심 구조 — 판단 주기가 다른 2계층」 슬라이드에서
ASCII 다이어그램 대신 쓰면 된다.

```markdown
![w:1100](diagrams/mobrobgpt_overview.png)
```

---

# 슬라이드 도판 — 「왜 좌표 회귀가 아니라 점수맵 위 픽셀 지목인가」

| 파일 | 용도 |
|---|---|
| `scoremap_vs_regression.py` | **원본** — 실제 체크포인트를 돌려 그린다 |
| `why_scoremap.png` (2635 × 1393) | 슬라이드 본문 도판 |
| `why_scoremap_stats.json` | 도판에 찍힌 수치의 원본 |

```bash
python docs/diagrams/scoremap_vs_regression.py     # 약 2분 (4 에피소드 × 700 step)
```

## 논지 — 측정된 것만 쓴다

**행동공간의 88 %는 물리적으로 못 쓰는 곳이다.**
좌표 회귀의 출력 `(x, y) ∈ ℝ²` 에는 그 경계를 표현할 자리가 없고,
점수맵 지목은 마스킹된 softmax 로 **구조적으로** 후보 안에서만 고른다.

후보 축소 순서 (`defense_env._cnn_valid_mask`, 대표 프레임 seed 100 · t 61 · 배 0):

| 단계 | 남은 픽셀 |
|---|---:|
| 전체 격자 | 2500 |
| 환형(요격 가능 반경) | 1000 |
| − 육지 | 979 |
| − 요격점 반경 3000 m 밖 | 387 |
| − 배정 코리도 ±60° 밖 | **299** (12.0 %) |

집계 (6,652 결정 = 배정된 배 × 결정, diversionary 4 에피소드):
평균 후보 **10.5 %** · 중앙 11.2 % · 최소 6.0 %.

**배정 코리도는 LLM 지휘관이 정한 요격점 방위로 열린다** — 즉 상위 지휘의 배정이
소프트 페널티가 아니라 **행동공간 자체**로 강제된다. 이게 2계층 구조가 성립하는 이유다.

## 쓰지 않기로 한 논거 (반증됨)

「MSE 회귀는 다봉 분포의 조건부 평균으로 무너진다」는 교과서적 논거는
**이 체크포인트에서는 성립하지 않는다.** 실측:

- 조건부 평균이 유효영역 밖으로 나간 비율 **0.0 %**
- 조건부 평균 ↔ argmax 거리 **중앙 75 m** (0.3 px), 90 백분위 131 m
- 조건부 평균의 확률이 최대확률의 5 % 미만인 경우 **0.4 %**

학습된 점수맵이 거의 단봉이라 평균과 최빈이 사실상 같다.
그래서 도판은 **모드 평균이 아니라 제약 표현력**을 논지로 삼는다.
(뾰족하다는 사실 자체는 오른쪽 패널 각주로 실었다 — "확률로 뭉개지 않는다"는 반론 차단.)

## 재사용 가능한 기존 자료

| 파일 | 내용 |
|---|---|
| `논문_그래프/unet_scoremap.png` (3699 × 1665) | U-Net 구조 전체도 (PlotNeuralNet). 15 ch 입력 썸네일 + 우측에 실제 valid mask / score map k=0,1 |
| `tools/PlotNeuralNet/pyexamples/obs_score.png` (1402 × 583) | 위 그림 우측 패널 단독 — 「유효 244/2500 → 픽셀 2개 순차 선택 → 그물벽」 |
| `tools/PlotNeuralNet/pyexamples/obs_global.png` · `obs_self.png` | 전역 9채널 / 배별 3채널 + CoordConv 썸네일 |
