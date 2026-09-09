# Fig. 4 — Rasterized Multi-Channel Observation

- LaTeX label: `fig:observation`
- 파일명: `fig4_observation.png`
- 배치: single column, `\begin{figure}[H]`, `width=80mm`
- 캔버스: 2400 x 1800 px (4:3)
- **좌 Register A (실사 해면 + 격자 오버레이) → 우 Register B (텐서 스택 + 행렬 콜아웃 + 컬러바)**

---

## INSTRUCTION

### Image Purpose
기동 계층의 입력이 좌표 목록이 아니라 모선을 중심에 둔 고정 크기 다채널 격자 지도라는 사실을
보이는 도판. 실제 해면이 격자로 양자화되어 채널별 래스터로 분해되고, 그 래스터가 결국
수치 행렬이라는 점까지 한 장에서 내려간다. 가변 개체 수가 "칠하기"로 흡수되어 입력 형태가
불변이라는 것이 이 도판의 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 왼쪽 실사와 오른쪽 채널 더미의 대응, 그리고 확대된
숫자 행렬을 보고 "개체를 나열하지 않고 그려 넣는다"는 인코딩 방식을 파악할 수 있어야 한다.

### Key Message
전장을 모선 중심 고정 격자에 그려 넣으므로 표적 수가 변해도 입력 형태가 바뀌지 않는다.

### Scene Description
A white figure canvas holding a left scene panel and a right tensor panel joined by one horizontal
flow arrow, with a numeric matrix callout below the junction. No outer frame, no background panel.

The left panel is a photographic near-vertical aerial view of open sea rendered at satellite orthophoto
fidelity, with genuine fine wave texture and a natural depth gradient. At its exact centre sits a
photorealistically rendered grey naval frigate seen from above with a realistic trailing wake. Six
small rigid-hull fast attack craft, each photorealistically rendered from above with sharp white
V-shaped bow wakes, are scattered around it at varying distances. Two lighter defending unmanned
vessels with deck-mounted net canisters are present, and between two of the attack craft a deployed
net barrier is visible as a line of small orange surface floats with dark mesh below the waterline. A
real irregular coastline with muted terrain shading occupies one corner. Over the whole panel a regular
square grid of thin white-cored lines with a fine dark core is overlaid edge to edge, so the continuous
imagery is visibly being quantised into cells; the single cell at the exact centre is outlined in blue
and carries a small blue cross confirming the frigate is pinned to the grid centre. A short grey
dimension bracket with end ticks spans exactly one cell along the top edge, and a longer grey dimension
bracket spans the full grid width along the bottom edge.

A solid dark arrow with a filled triangular arrowhead runs from the right edge of the scene panel into
the tensor panel. The tensor panel is a stack of identical square raster tiles drawn in the standard
tensor-stack convention of a deep-learning paper: every tile the same size, each offset from the one
behind by a constant small diagonal step, rendered with a thin dark edge and a subtle oblique side face
so the stack reads as layered slabs rather than as flat squares. The stack is separated into three
contiguous groups by slightly wider diagonal gaps, and a thin square brace runs alongside each group
with a channel-count numeral at its centre. The frontmost tile of each group is fully rendered as a
genuine viridis-colormapped raster with visible cell structure, while tiles behind show only their
offset edges. The first group's front tile shows bright compact blobs clustered where the attack craft
sat in the left panel against a dark field. The second group's front tile shows a single bright cell
and one bright ring elsewhere. The third group's front tile shows a smooth monotone left-to-right ramp
across the full width. A vertical viridis colorbar with a thin dark border and small tick marks at its
two ends stands to the right of the stack. Spatial resolution is annotated once alongside the stack in
the conventional way.

Below the junction, a callout leads from a small square region outlined in blue on the frontmost
channel tile down to an enlarged numeric matrix: a genuine matrix rendering with tall square brackets
on both sides, a five by five arrangement of cells separated by faint rules, each cell containing a
short decimal number, and the cells lightly background-tinted with the same viridis ramp so the numeric
matrix and the raster tile are visibly the same object at two zoom levels. Two thin grey callout lines
connect the outlined region on the tile to the two upper corners of the matrix brackets. A single dark
arrow with a filled triangular arrowhead leaves the right edge of the tensor stack toward open space,
indicating the downstream consumer without drawing it.

CRITICAL: render only the exact English strings and numerals listed in the CONTENT block below and
nothing else, except that the matrix callout cells may contain short generic decimal numbers between
zero and one. Do NOT invent, infer or add any further text. Never render Korean, Chinese, Japanese or
Cyrillic characters anywhere in the image.

### Rendering Style
- 좌측은 **실사**다. 위성 정사영상 품질 해면 + 사진 수준 탑다운 선박 렌더.
  속 빈 삼각형·쉐브론·픽토그램으로 배를 대체하면 즉시 실패다.
- 우측은 딥러닝 논문의 **텐서 스택 관례**다. 정면 사각형 + 얇은 오블리크 측면을 가진 슬래브,
  등간격 대각 오프셋, 묶음별 각괄호와 채널 수 숫자, 공간 해상도 주기, 세로 컬러바.
  납작한 사각형만 늘어놓거나 아이소메트릭 큐브로 그리면 실패다.
- 채널 타일은 **진짜 렌더된 viridis 히트맵**이어야 한다. 회색 격자에 색칠한 시늉 금지.
- 행렬 콜아웃은 **진짜 행렬**이다 — 양쪽 키 큰 대괄호, 셀 격자, 셀 안 짧은 소수.
  이 요소가 "래스터 = 수치 행렬"을 못박는다. 생략하면 도판의 절반이 사라진다.
- 컬러맵은 viridis 하나로 고정. 무지개(jet) 금지.
- 흑백 안전성: viridis 는 명도 단조라 회색조 인쇄에서도 순서가 보존된다.
- 배경: 우측 순백, 좌측 실사. 드롭섀도·글로우 금지. 여백 15~25 %.

### Content Placement
좌측 실사 패널 아래에 'Battlefield' 를 대 라벨로 놓는다. 상단 한 칸 브래킷 위에 'Cell' 을,
하단 전폭 브래킷 아래에 'Grid extent' 를 소 라벨로 놓는다. 중앙 파란 셀 옆에 지시선을 빼
'Centred on mothership' 을 소 라벨로 놓는다. 우측 더미 아래에 'Channel stack' 을 대 라벨로 놓고,
스택 옆에 공간 해상도로 '50 x 50' 을 소 라벨로 한 번 놓는다. 세 묶음 각괄호 바깥에 위에서부터
'Global', 'Per-vessel', 'Coordinate' 를 중 라벨로 놓고, 각 각괄호 중앙에 채널 수로 각각
'9', '3', '3' 을 소 라벨로 놓는다. 각 묶음 라벨 바로 아래 한 단계 작게 각각
'enemy, ally, net, land', 'self, assigned, valid', 'x, y, r' 를 소 라벨로 놓는다.
컬러바 위 끝에 'high', 아래 끝에 'low' 를 소 라벨로 놓는다. 행렬 콜아웃 아래에
'Cell values' 를 중 라벨로 놓는다. 스택에서 나가는 오른쪽 화살표 위에 'To policy' 를 소 라벨로 놓는다.
그 밖의 어떤 문자열도 렌더링하지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: scene_to_tensor_stack_with_matrix_callout
- print_target: single journal column, 80 mm wide

### Color Palette
- ink `#1F2933` — 타일 외곽, 화살표, 행렬 대괄호, 라벨
- gray `#7B8794` — 격자선, 치수 브래킷, 콜아웃 선, 각괄호
- sea `#8FA8BC` → `#5B7C95` — 좌측 실사 수면
- land `#E3DCC9` / 해안선 `#B5AC93`
- blue `#1F5FA8` — 중앙 셀 강조, 콜아웃 영역 외곽, 그물 오버레이
- float `#E8862A` — 그물 부표 (실물 색)
- colormap **viridis** — 채널 타일, 컬러바, 행렬 셀 배경 틴트
- white `#FFFFFF` — 우측 배경, 격자 심, 라벨 헤일로

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 3단계: 구역명(대), 묶음명(중), 채널명·수치·부기(소)
- 채널명은 소문자, 쉼표로만 구분하고 줄바꿈하지 않는다
- 채널 수 숫자와 해상도 표기는 이 장르의 필수 주기이므로 반드시 넣는다

## CONTENT
scene: "Battlefield"
tick_cell: "Cell"
tick_extent: "Grid extent"
tick_centre: "Centred on mothership"
stack: "Channel stack"
stack_dim: "50 x 50"
group_1: "Global"
group_1_n: "9"
group_1_items: "enemy, ally, net, land"
group_2: "Per-vessel"
group_2_n: "3"
group_2_items: "self, assigned, valid"
group_3: "Coordinate"
group_3_n: "3"
group_3_items: "x, y, r"
cbar_hi: "high"
cbar_lo: "low"
callout: "Cell values"
edge_out: "To policy"

## FORBIDDEN ELEMENTS
- **좌측 선박을 아이콘으로 대체**: 속 빈 삼각형, 쉐브론, 픽토그램, 클립아트 보트
- 좌측을 플랫 벡터 바다·단색 면으로 대체
- 채널 타일을 회색 격자 + 손칠한 색으로 대체 (진짜 렌더된 히트맵이어야 함)
- 행렬 콜아웃 생략, 또는 대괄호 없는 표로 대체
- 이미지 내부 캡션: "Fig. 4", "Figure 4", 도판 제목
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·채널 이름
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 아이소메트릭 큐브, 원근 수축, 과장된 두께의 3D 판
- 네온, 글로우, 드롭섀도, 베벨, 어두운 배경, HUD
- 회로기판·뇌·로봇 등 AI 상투 모티프
- 기관 로고, 워터마크
- 원문에 없는 정량 성능 수치(포획률, 정확도, 손실값) 표기
