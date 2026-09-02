# Fig. 5 — Score-Map Policy: U-Net Features, Query Scoring, and Sequential Waypoint Selection

- LaTeX label: `fig:policy`
- 파일명: `fig5_policy.png`
- 배치: **full width**, `\begin{figure*}[t]`, `width=0.94\textwidth`
- 캔버스: 4200 x 1750 px (12:5)
- **Register B 주도 (U-Net 논문 관례) + 우측 끝 Register A (실사 해면)**

> 채널 수·해상도는 체크포인트 `boatattack_sim/models/u-net_map.pt` 의 실제 config 에서 온 값이다
> (`docs/unet_model_deploy.md` §1). 임의 수치가 아니므로 그대로 표기한다.

---

## INSTRUCTION

### Image Purpose
논문의 핵심 도판. 다채널 격자 관측이 인코더-디코더 합성곱을 지나 입력과 같은 해상도의
특징맵이 되고, 자기 상태에서 만든 질의 벡터와의 내적이 해면 전체에 점수를 그리며, 기하
마스크 밖을 배제한 뒤 점수가 높은 칸 두 개를 순차로 지목하되 먼저 고른 칸의 특징이 질의에
되먹여지고, 두 좌표를 잇는 구간이 실제 그물벽이 된다는 전 과정을 좌에서 우로 잇는다.
연속 좌표를 회귀하지 않고 칸을 지목한다는 설계 선택이 이 도판의 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. U-Net 계열 도해에 익숙한 독자는 채널 수 표기와 skip
화살표만으로 구조를 즉시 읽고, 익숙하지 않은 독자도 화살표를 따라가면 "지도를 받아 지도를
내고, 그 위에서 두 점을 찍는다"는 흐름을 얻는다.

### Key Message
점수맵 위의 칸 두 개를 순차로 지목하고, 그 두 점을 잇는 구간이 그물벽이 된다.

### Scene Description
A wide white figure canvas holding a single left-to-right pipeline drawn in the visual language of a
U-Net architecture figure. No outer frame, no background panel.

At the far left, the input is a stack of identical square raster slabs offset by a constant diagonal
step, each with a thin dark edge and a subtle oblique side face, the frontmost one rendered as a
genuine viridis-colormapped raster with visible cell structure. Its channel count is annotated above
the stack and its spatial resolution alongside it, in the conventional position.

The network body follows as a symmetric U-shaped arrangement of tensor slab boxes, each box drawn as a
front rectangle with a thin oblique top and side face, filled pale blue with a dark edge. The
contracting path descends in three levels: level one holds a pair of slabs at full height, level two a
pair at half height, level three a pair at quarter height, each successive level drawn lower on the
canvas so the descent is spatial as well as nominal. The expanding path mirrors this exactly,
ascending back through half height to full height, and the final slab is drawn at exactly the same
height as the first so equal input and output resolution is asserted by geometry. Every box carries its
channel count as a small numeral centred above it, and each level carries its spatial resolution as a
small numeral rotated ninety degrees alongside the leftmost box of that level. Arrows follow the
standard colour convention: short horizontal blue arrows between boxes within a level for convolution
blocks, red downward arrows between levels on the contracting side, green upward arrows between levels
on the expanding side, and long grey dashed horizontal arrows crossing the U from each contracting
level to its matching expanding level for skip connections. Beneath the lowest level a small pale
rounded box connects upward into the bottleneck by a thin grey arrow, denoting the global context
injection.

The network's final slab feeds a square feature-map tile of the same size as the input tile, drawn with
a fine grey cell grid and a small uniform dot at every cell centre to indicate a per-cell feature
vector. On a separate lane below the network sits a small pale rounded box containing four compact
pictograms in a row — an angular vessel marker, a short heading arrow, a pair of stacked bars, and a
hollow ring — and a short dark arrow leads from it to a genuine column vector rendering: a narrow
vertical arrangement of a few stacked cells enclosed by tall square brackets on both sides. A thin dark
arrow rises from this vector and meets the arrow from the feature tile at a small open circle
containing a centred dot, the standard inner-product operator glyph, and one arrow leaves that operator
into the scoring stage.

The scoring stage is a horizontal row of four square tiles, all exactly the same size as the input
tile. The first is the raw score map, a genuine viridis heatmap with one clear bright region and one
weaker region. The second is the mask, rendered as a true binary matrix: a black field with a
sector-shaped band of pure white cells, its cell grid visible, and a thin bracket pair at its sides so
it reads as a matrix. A small operator glyph, a circle enclosing a cross, sits between the first and
second tiles. The third tile is the masked score map, identical to the first but with everything
outside the sector band flattened to a single dark value, and one cell marked by a bold blue square
outline with a small filled centre dot. The fourth tile repeats the third with that first marked cell
retained, a circular grey hatched disc drawn around it to indicate exclusion, and a second distinct
cell marked with the same bold blue square. A vertical viridis colorbar with a thin dark border and end
ticks stands to the right of this row. A thin blue curved arrow with a filled triangular arrowhead runs
backward from the first marked cell, beneath the tile row, and returns into the column vector on the
lower lane; it is the only right-to-left arrow in the entire image.

At the far right, a photographic near-vertical aerial panel closes the pipeline: real sea surface at
satellite orthophoto fidelity, a photorealistically rendered light-hulled defending unmanned vessel
with a deck-mounted net canister under way with a visible wake, two small orange-float net endpoints
marked by hollow blue overlay circles, and between them a freshly deployed net barrier rendered as a
line of small orange surface floats with dark mesh visible below the waterline. A thin solid blue
overlay line with a filled triangular arrowhead runs from the vessel to the nearer circle. Two small
rigid-hull attack craft with white bow wakes sit just beyond the barrier.

CRITICAL: render only the exact English strings and numerals listed in the CONTENT block below and
nothing else. Do NOT invent, infer or add any further text, layer names, kernel sizes, or formulas.
Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the image.

### Rendering Style
- 신경망은 **U-Net 논문 도해의 관례 그대로**다: 오블리크 텐서 슬래브 박스, 박스 위 채널 수 숫자,
  레벨 옆 90도 회전한 해상도 숫자, 색으로 구분된 연산 화살표, U 를 가로지르는 회색 점선 skip.
  외곽선 실루엣 하나로 뭉뚱그리거나 납작한 사각형만 늘어놓으면 실패다.
- **모든 정사각 타일은 크기가 완전히 동일**하다. 크기를 달리하면 해상도 보존 주장이 무너진다.
- 점수맵·특징맵·관측 타일은 **진짜 렌더된 viridis 히트맵**. 손칠한 색 격자 금지.
- 마스크는 **진짜 이진 행렬** — 검정 바탕에 순백 셀, 셀 격자 가시, 양옆 대괄호.
- 질의는 **진짜 열벡터** — 키 큰 대괄호 안에 셀이 세로로 쌓인다.
- 연산자는 원 안 점(내적)과 원 안 십자(원소별 마스킹) 두 종류만. 그리스 문자·수식 금지.
- 역방향 화살표는 질의 되먹임 **단 하나**. 다른 역방향 선이 생기면 순차성이 깨진다.
- 우측 끝 패널만 **실사**다. 그물은 아이콘이 아니라 주황 부표열 + 수면 아래 망의 실물이다.
- 컬러맵은 viridis 하나로 고정. 무지개(jet) 금지.
- 배경: 순백. 드롭섀도·글로우·베벨 금지. 여백 15~25 %.

### Content Placement
좌측 입력 스택 아래에 'Observation' 을 대 라벨로 놓고, 스택 위 채널 수로 '15' 를, 옆 해상도로
'50 x 50' 을 소 라벨로 놓는다. 신경망 각 박스 위 채널 수로 좌에서 우로 '32', '32', '64', '64',
'32', '32' 를 소 라벨로 놓고, 각 레벨 왼쪽에 90도 회전한 해상도로 위에서 아래로 '50 x 50',
'25 x 25', '13 x 13' 을 소 라벨로 놓는다. 신경망 아래 중앙에 'Encoder-decoder' 를 대 라벨로 놓는다.
가장 위 skip 점선 위에 'skip connection' 을 소 라벨로 **한 번만** 놓는다. 병목 아래 작은 박스 옆에
'Global context' 를 소 라벨로 놓는다. 점 찍힌 타일 아래에 'Feature map' 을 대 라벨로, 위 채널 수로
'32' 를 소 라벨로 놓는다. 하단 레인 픽토그램 박스 아래에 'Own state' 를 대 라벨로, 열벡터 아래에
'Query' 를 대 라벨로 놓는다. 내적 연산자 아래에 'Inner product' 를 소 라벨로 놓는다.
네 타일 아래에 좌에서 우로 'Score map', 'Valid mask', 'First pick', 'Second pick' 을 대 라벨로 놓고,
네 번째 타일의 해칭 원판 옆에 'Exclusion' 을 소 라벨로 놓는다. 컬러바 위 끝에 'high', 아래 끝에
'low' 를 소 라벨로 놓는다. 되먹임 곡선 최저점 아래에 'Query update' 를 소 라벨로 놓는다.
우측 실사 패널 아래에 'Net wall' 을 대 라벨로 놓고, 아군에 가까운 원 옆에 'Transit waypoint' 를,
먼 원 옆에 'Net endpoint' 를 소 라벨로 놓는다.
그 밖의 어떤 문자열도 렌더링하지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1750
- aspect_ratio: 12:5
- layout: unet_pipeline_left_to_right
- print_target: full text width, 165 mm wide

### Color Palette
- ink `#1F2933` — 타일 외곽, 화살표, 대괄호, 라벨
- gray `#7B8794` — skip 점선, 셀 격자, 특징 점, 배제 해칭, 전역 문맥 박스
- tensor `#EAF2F8` 면 / `#2C3E50` 외곽 — 텐서 슬래브 박스
- conv blue `#1F5FA8` — 합성곱 화살표, 선택 칸 테두리, 되먹임 곡선, 그물 오버레이
- down red `#C0392B` — 다운샘플 화살표, 우측 실사의 적 오버레이
- up green `#2E8B57` — 업샘플 화살표
- float `#E8862A` — 그물 부표 (실물 색)
- sea `#8FA8BC` → `#5B7C95` — 우측 실사 수면
- colormap **viridis** — 관측·특징·점수 타일, 컬러바
- mask 순흑/순백 — 이진 마스크 타일 전용
- white `#FFFFFF` — 배경

### Typography
- 산세리프 단일 패밀리
- 크기 3단계: 단계명(대), 요소 라벨(중), 채널 수·해상도·부기(소)
- 채널 수는 박스 바로 위 중앙, 해상도는 레벨 왼쪽에 90도 회전 — U-Net 도해의 표준 위치
- 단계명은 대상 바로 아래 중앙 정렬로 통일하고 위치를 흔들지 않는다

## CONTENT
stage_obs: "Observation"
obs_ch: "15"
obs_dim: "50 x 50"
net_ch_1: "32"
net_ch_2: "32"
net_ch_3: "64"
net_ch_4: "64"
net_ch_5: "32"
net_ch_6: "32"
net_dim_1: "50 x 50"
net_dim_2: "25 x 25"
net_dim_3: "13 x 13"
stage_net: "Encoder-decoder"
note_skip: "skip connection"
note_film: "Global context"
stage_feat: "Feature map"
feat_ch: "32"
stage_own: "Own state"
stage_query: "Query"
note_dot: "Inner product"
stage_score: "Score map"
stage_mask: "Valid mask"
stage_pick1: "First pick"
stage_pick2: "Second pick"
note_exclude: "Exclusion"
cbar_hi: "high"
cbar_lo: "low"
note_refeed: "Query update"
stage_net_wall: "Net wall"
note_wp1: "Transit waypoint"
note_wp2: "Net endpoint"

## FORBIDDEN ELEMENTS
- 신경망을 **외곽선 실루엣 하나**로 뭉뚱그리기, 납작한 사각형만 늘어놓기
- 채널 수·해상도 주기 생략 (이 장르의 필수 표기다)
- 점수맵·마스크를 손칠한 회색 격자로 대체 (진짜 히트맵·진짜 이진 행렬이어야 함)
- 우측 패널을 일러스트·아이콘으로 대체, 그물을 아이콘으로 대체
- 이미지 내부 캡션: "Fig. 5", "Figure 5", 도판 제목
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·레이어 이름·커널 크기·수식·그리스 문자
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 아이소메트릭 큐브, 과장된 원근, 두꺼운 3D 판 옆면
- 네온, 글로우, 드롭섀도, 베벨, 반사, 어두운 배경, HUD
- 뇌 모양, 뉴런 노드-엣지 그물망, 회로기판, 로봇
- 질의 되먹임 외의 역방향 화살표, 순환 루프
- 기관 로고, 프레임워크 로고, 워터마크
- 원문에 없는 정량 성능 수치 표기
