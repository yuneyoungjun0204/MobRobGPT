# Fig. 6 — Group-Relative Training with Per-Vessel Counterfactual Credit

- LaTeX label: `fig:grpo`
- 파일명: `fig6_grpo.png`
- 배치: single column, `\begin{figure}[H]`, `width=80mm`
- 캔버스: 2400 x 1800 px (4:3)
- **Register A (롤아웃 썸네일) + Register B (보상/이득 차트 + 반사실 행렬)**

---

## INSTRUCTION

### Image Purpose
학습 방식 도판. 한 결정 시점에서 정책이 여러 후보 행동을 뽑고, 각 후보를 시뮬레이터에서
일정 구간 실제로 진행시켜 팀 보상을 얻은 뒤, 후보들끼리의 상대 비교만으로 이득을 산정하므로
별도의 가치망이 필요 없다는 것, 그리고 한 척의 행동만 바꾸고 나머지를 고정해 평가함으로써
팀 보상 안에 묻힌 개별 기여를 분리한다는 것을 한 장에서 잇는다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 강화학습 전공이 아니어도 "후보를 여러 개 굴려 서로
비교한다"와 "한 척만 바꿔 차이를 그 배 몫으로 돌린다" 두 문장을 그림에서 직접 읽어야 한다.

### Key Message
후보 행동들을 실제로 진행시켜 서로 비교하고, 한 척만 바꾼 차이를 그 배의 기여로 돌린다.

### Scene Description
A white figure canvas organised as three stacked bands flowing downward, with no outer frame and no
background panel.

The top band holds a single decision state at the left: a square tile rendered as a genuine
viridis-colormapped score map with visible cell structure, a sector-shaped bright band, and a thin dark
edge. From its right side a fan of four solid dark arrows with filled triangular arrowheads spreads
rightward and slightly apart, each terminating at one of four small square candidate tiles arranged in
a vertical column. Each candidate tile repeats the same score map but with a different pair of cells
marked by bold blue square outlines, so the four candidates are visibly four different waypoint pairs
on one common map. The lowest of the four tiles is distinguished by a dashed blue border rather than a
solid dark one, marking it as the seeded candidate rather than a sampled one.

The middle band holds four photographic rollout panels in a horizontal row, one per candidate, each a
small near-vertical aerial view of the same sea area rendered at satellite orthophoto fidelity with
genuine wave texture. Every panel contains the same photorealistically rendered grey frigate at centre,
the same three small rigid-hull attack craft with white bow wakes converging on it, and the same three
light-hulled defending unmanned vessels — but in each panel the deployed net barriers sit in different
places, rendered as lines of small orange surface floats with dark mesh below the waterline, and each
panel shows a different set of thin dashed white trajectory traces recording where the vessels went
during the window. In two panels an attack craft is stopped dead in the water behind a barrier with its
bow wake collapsed; in the others one craft has slipped past and continues toward the frigate. A thin
grey horizontal time bracket with end ticks spans beneath the row of panels, indicating that all four
were advanced over the same interval. Directly under each panel a single vertical bar rises from a
common baseline, the four bars at visibly different heights, forming a small bar chart of the outcome
of each rollout.

The bottom band holds two data objects side by side. On the left, a second small bar chart shares the
same four-bar arrangement but is drawn about a horizontal zero line, with two bars extending above it
and two below, each bar dark-edged and filled pale blue, and a thin dashed grey line marks the group
mean at the zero level; a short curved dark arrow leads from the first bar chart into this one. On the
right sits a genuine matrix rendering: tall square brackets enclosing a grid of three rows by four
columns, the cells separated by faint rules, every cell containing a small angular vessel marker. In
the second row every cell's marker is drawn in solid blue and each is oriented differently from the
others; in the first and third rows every cell's marker is drawn in flat grey and all are oriented
identically across the row. The second row is further set off by a light tint band running behind it
and by a small bracket at the row's left edge. A short dark arrow leads from this matrix back toward
the bar chart pair, closing the flow.

CRITICAL: render only the exact English strings listed in the CONTENT block below and nothing else. Do
NOT invent, infer or add any further text, numeric values, axis numbers, or formulas. Never render
Korean, Chinese, Japanese or Cyrillic characters anywhere in the image.

### Rendering Style
- 롤아웃 패널은 **실사**다. 위성 정사영상 품질 해면 + 사진 수준 탑다운 선박 렌더.
  네 패널은 **같은 해역·같은 배치**여야 하고 그물 위치와 항적만 달라야 한다 —
  그래야 "같은 상태에서 갈라진 후보"라는 주장이 성립한다. 배를 아이콘으로 대체하면 즉시 실패다.
- 그물은 아이콘이 아니라 실물 — 주황 부표열 + 수면 아래 비치는 망.
- 항적은 사진 위에 얹은 얇은 흰 파선 오버레이. 후보마다 다르게 그린다.
- 막대 차트는 **진짜 차트**다 — 공통 기준선, 다른 높이, 이득 차트는 0선 위아래로 갈린다.
  축 눈금 숫자는 넣지 않는다(값을 창작하지 않기 위해서다).
- 반사실 행렬은 **진짜 행렬** — 양쪽 키 큰 대괄호, 셀 격자, 셀 안 마커.
  한 행만 색과 방향이 다르다는 사실이 "한 척만 바꾼다"를 형태로 못박는다.
- 씨앗 후보만 파란 파선 테두리로 구분한다. 나머지는 실선.
- 컬러맵은 viridis 하나로 고정. 무지개(jet) 금지.
- 배경: 순백. 드롭섀도·글로우 금지. 여백 15~25 %.

### Content Placement
상단 좌측 타일 아래에 'Decision state' 를 중 라벨로 놓는다. 네 후보 타일 열 오른쪽 바깥에
'Candidate actions' 를 대 라벨로 놓고, 파선 테두리 후보 옆에 'Heuristic seed' 를 소 라벨로 놓는다.
중간 띠 네 패널 아래 시간 브래킷 밑에 'Rollout window' 를 중 라벨로 놓는다. 첫 막대 차트 왼쪽
바깥에 'Team reward' 를 대 라벨로 세로로 놓는다. 하단 좌측 0선 차트 왼쪽 바깥에
'Group-relative advantage' 를 대 라벨로 세로로 놓고, 0선 오른쪽 끝에 'group mean' 을 소 라벨로 놓는다.
하단 우측 행렬 아래에 'Counterfactual' 을 대 라벨로 놓고, 강조된 둘째 행 왼쪽 브래킷 옆에
'perturbed' 를 소 라벨로, 회색 첫째 행 왼쪽에 'fixed' 를 소 라벨로 놓는다. 행렬 위쪽 바깥에
'candidates' 를 소 라벨로, 왼쪽 바깥 세로로 'vessels' 를 소 라벨로 놓아 두 축의 의미를 준다.
그 밖의 어떤 문자열도 렌더링하지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: three_band_downward_flow
- print_target: single journal column, 80 mm wide

### Color Palette
- ink `#1F2933` — 타일 외곽, 화살표, 행렬 대괄호, 막대 테두리, 라벨
- gray `#7B8794` — 시간 브래킷, 셀 격자, 평균 파선, 고정 행 마커, 지시선
- sea `#8FA8BC` → `#5B7C95` — 롤아웃 패널 수면
- blue `#1F5FA8` — 선택 칸 테두리, 씨앗 파선 테두리, 막대 채움, 섭동 행 마커
- red `#C0392B` — 실사 패널 위 적 오버레이 표시
- float `#E8862A` — 그물 부표 (실물 색)
- tint `#EAF2F8` — 섭동 행 배경 띠
- colormap **viridis** — 점수맵 타일
- white `#FFFFFF` — 배경, 항적 파선, 라벨 헤일로

### Typography
- 산세리프 단일 패밀리
- 크기 3단계: 구역명(대), 요소 라벨(중), 축 의미·상태(소)
- 차트 축 라벨은 세로로 한 번씩만 회전 배치, 나머지 라벨은 전부 수평
- 축 눈금 숫자를 넣지 않는다 — 값을 창작하지 않기 위한 의도적 선택이다

## CONTENT
stage_state: "Decision state"
stage_cand: "Candidate actions"
note_seed: "Heuristic seed"
stage_rollout: "Rollout window"
axis_reward: "Team reward"
axis_adv: "Group-relative advantage"
note_mean: "group mean"
stage_cf: "Counterfactual"
note_perturbed: "perturbed"
note_fixed: "fixed"
axis_cols: "candidates"
axis_rows: "vessels"

## FORBIDDEN ELEMENTS
- **롤아웃 패널을 아이콘·일러스트로 대체**: 속 빈 삼각형, 쉐브론, 픽토그램, 클립아트 보트
- 네 롤아웃 패널의 해역·초기 배치가 서로 달라지는 것 (그물·항적만 달라야 한다)
- 축 눈금 숫자, 보상 값, 이득 값 등 **창작된 수치** 표기
- 학습 곡선, 손실 그래프, 에폭 축 — 이 도판은 결과 그래프가 아니다
- 이미지 내부 캡션: "Fig. 6", "Figure 6", 도판 제목
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·수식·그리스 문자·기댓값 기호
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 네온, 글로우, 드롭섀도, 베벨, 어두운 배경, HUD
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇 등 AI 상투 모티프
- 폭발, 화염, 연기 — 포획이지 파괴가 아니다
- 기관 로고, 프레임워크 로고, 워터마크
