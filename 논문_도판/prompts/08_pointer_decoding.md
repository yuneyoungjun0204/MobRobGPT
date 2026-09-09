# Fig. 8 — Query Scoring and Sequential Waypoint Selection

- 파일명: `fig8_pointer.png` · LaTeX label `fig:pointer`
- 배치: **양단** `figure*[t]`, `width=0.94\textwidth` · 캔버스 **4200 x 1500 (2.8:1)**
- 레지스터: **B 주도 + 우측 끝 A 실사** · 렌더링 문자열 **9개**

> 근거: `model/cnn_actor.py::_score` (질의·특징 내적), `_logits` (마스크 밖 -inf),
> `_decode` (K=2 순차 선택, 선택 특징을 질의에 가산, `_exclude` 로 중복·도달성 제한),
> `docs/unet_model_deploy.md` §6.3 (가까운 점=이동 WP, 먼 점=그물벽 끝점).

---

## INSTRUCTION

### Image Purpose
특징맵이 어떻게 행동이 되는지를 보이는 도판. 자기 상태에서 만든 질의 벡터와 칸 특징의 내적이
해면 전체에 점수를 그리고, 유효 마스크 밖이 배제된 뒤 점수가 높은 칸 두 개가 순차로 지목되되
먼저 고른 칸의 특징이 질의에 되먹여지며, 두 좌표를 잇는 구간이 실제 그물벽이 된다는 것을
좌에서 우로 잇는다. 연속 좌표를 회귀하지 않고 칸을 지목한다는 설계 선택이 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 화살표를 따라가면 "지도 위에서 두 점을 찍고 그 사이를
잇는다"는 결론에 도달해야 한다.

### Key Message
점수맵 위의 칸 두 개를 순차로 지목하고, 두 점을 잇는 구간이 그물벽이 된다.

### Scene Description
A wide off-white canvas holds a single left-to-right pipeline with no outer frame and no background
panel. At the far left a square tile with a fine faint grid and a small uniform dot at every cell
centre represents the per-cell feature map. On a separate lane below it a small rectangle outlined in
the primary tone with a very pale accent interior holds four compact pictograms in a row and no words —
an angular vessel marker, a short heading arrow, a pair of stacked bars, and a hollow ring — and a
short arrow leads from it to a genuine column vector rendering: a narrow vertical arrangement of a few
stacked cells enclosed by tall square brackets on both sides. A thin arrow rises from that vector and
meets the arrow leaving the feature tile at a small open circle enclosing a centred dot, the standard
inner-product operator glyph, and one arrow leaves that operator into the scoring stage. The scoring
stage is a horizontal row of four square tiles, all exactly the same size as the feature tile. The
first is the raw score map, a genuine viridis-colormapped heatmap with one clear bright region and one
weaker region and visible cell structure. Between it and the second tile sits a small circle enclosing
a cross, the elementwise masking operator. The second tile is the mask, rendered as a true binary
matrix: a black field with a sector-shaped band of pure white cells, its cell grid visible, and thin
bracket pairs at its left and right sides so it reads unmistakably as a matrix rather than a picture.
The third tile is the masked score map, identical to the first but with everything outside the sector
band flattened to a single dark value, and one cell within the band marked by a heavy accent square
outline with a small filled centre dot. The fourth tile repeats the third with that first marked cell
retained, a circular hatched disc drawn around it in the secondary tone to indicate the exclusion
neighbourhood, and a second distinct cell within the band marked by an identical heavy accent square.
A thin accent curved arrow with a filled triangular arrowhead runs backward from the first marked cell,
beneath the tile row, and returns into the column vector on the lower lane; it is the only right-to-left
arrow in the entire image and must be unmistakably singular. A vertical viridis colorbar with a thin
border and small end ticks stands to the right of the tile row, unlabelled. At the far right a
photographic near-vertical aerial panel closes the pipeline: real sea surface at satellite orthophoto
fidelity with genuine wave texture, a photorealistically rendered light-hulled defending unmanned
vessel with a deck-mounted net canister under way with a visible wake, two small hollow accent circles
marking the two selected positions, and between them a freshly deployed net barrier rendered as a line
of small orange surface floats with dark mesh visible below the waterline. A thin solid accent overlay
line with a filled triangular arrowhead runs from the vessel to the nearer circle, and the segment
between the two circles is drawn with a heavier accent stroke, so the near point reads as something the
vessel travels to and the far point as the end of the wall. Two small rigid-hull attack craft with
white bow wakes sit just beyond the barrier. No watermarks, no blurry text, no placeholder brackets,
no random artifacts, no figure caption numbers, no legend boxes, no axis labels, no tensor shape
notation, no mathematical formulas, no Greek letters. CRITICAL: render only the nine exact English
strings listed in the CONTENT block below and nothing else. Do NOT invent, infer or add any further
text. Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the image.

### Rendering Style
- 서피스: 모든 정사각 타일의 **크기가 완전히 동일**하다. 크기를 달리하면 해상도 보존 주장이
  무너진다. 점수맵은 **진짜 렌더된 viridis 히트맵**, 마스크는 **진짜 이진 행렬**(검정/순백 +
  셀 격자 + 양옆 대괄호), 질의는 **진짜 열벡터**(키 큰 대괄호 안에 셀이 세로로 쌓임).
- 배경: 오프 화이트 단색. 우측 끝 패널만 실사.
- 코너/경계: 직각. 선택 칸만 굵은 강조색 사각 테두리로 표시한다.
- 연결선: 주조색 실선 + 채워진 삼각 화살촉이 기본. 되먹임 곡선만 강조색이며 **유일한 역방향**이다.
  다른 역방향 선이나 순환 루프가 생기면 순차성이 깨진다.
- 시각장식: 연산자는 원 안 점(내적)과 원 안 십자(원소별 마스킹) 두 종류만. 그리스 문자·수식 금지.
  배제 원판은 보조색 해칭 — 색을 빼도 해칭으로 구분된다.
- 공간구성: 좌 특징맵 / 하단 질의 레인 / 중앙 4타일 행 / 우 실사의 단일 좌우 흐름. 여백 15~25%.
- 시각메타포: 지도 위에 두 개의 압정을 순서대로 꽂고 그 사이에 줄을 건다. 마지막 실사 패널이
  그 추상을 물리적 그물로 착지시킨다.

### Content Placement
좌측 타일 아래에 'Feature map' 을 중 라벨로 배치한다. 하단 레인 픽토그램 박스 아래에 'Own state'
를 중 라벨로, 열벡터 아래에 'Query' 를 중 라벨로 배치한다. 내적 연산자 아래에 'Inner product' 를
소 라벨로 배치한다. 네 타일 아래에 좌에서 우로 'Score map', 'Valid mask', 'First pick',
'Second pick' 을 중 라벨로 배치한다. 되먹임 곡선의 가장 낮은 지점 아래에 'Query update' 를 소
라벨로 배치한다. 우측 실사 패널 아래에 'Net wall' 을 중 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 이동 경유점과 그물 끝점의 구분은 화살표 유무와 선
굵기로만, 배제 반경은 해칭 원판으로만 표현하고 글로 적지 않는다. 컬러바에 라벨을 붙이지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1500
- aspect_ratio: 2.8:1
- layout: single_left_to_right_pipeline
- print_target: 학술지 양단 전폭, 165 mm

### Background Treatment
- base: 오프 화이트 단색 (우측 끝만 실사 해면)
- texture: 없음
- ornament: 없음

### Color Palette
- primary: #2C3E50 (네이비) — 타일 외곽, 화살표, 대괄호, 연산자, 라벨
- secondary: #5D6D7E (슬레이트 블루) — 셀 격자, 특징 점, 배제 원판 해칭, 픽토그램
- accent: #2980B9 (딥 블루) — 선택 칸 테두리, 되먹임 곡선, 그물벽 오버레이, 선택 위치 원
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: sea #8FA8BC→#5B7C95 (우측 실사 수면) · threat #B5493C (적 오버레이) ·
  float #E8862A (그물 부표 실물색) · mask 순흑/순백 (이진 마스크 타일 전용) ·
  colormap **viridis** (점수맵, 컬러바)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 2단계: 단계명(중), 연산·부기(소)
- 단계명은 대상 바로 아래 중앙 정렬로 통일하고 위치를 흔들지 않는다

## CONTENT
stage_feat: "Feature map"
stage_own: "Own state"
stage_query: "Query"
note_dot: "Inner product"
stage_score: "Score map"
stage_mask: "Valid mask"
stage_pick1: "First pick"
stage_pick2: "Second pick"
note_refeed: "Query update"
stage_net: "Net wall"

## FORBIDDEN ELEMENTS
- 타일 크기가 서로 달라지는 것
- 점수맵·마스크를 손칠한 회색 격자로 대체 (진짜 히트맵·진짜 이진 행렬이어야 함)
- 우측 패널을 일러스트·아이콘으로 대체, 그물을 아이콘으로 대체
- 되먹임 곡선 외의 역방향 화살표, 순환 루프
- 수식, 그리스 문자, 소프트맥스·시그마 표기, 텐서 shape, 확률값 숫자
- 컬러바 눈금 숫자
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 이미지 내부 캡션: "Fig. 8", "Figure 8", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 아이소메트릭 큐브, 원근, 3D 판 옆면
- 네온, 글로우, 드롭섀도, 베벨, 어두운 배경, HUD
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇
- 기관 로고, 워터마크
- 원문에 없는 정량 성능 수치 표기
