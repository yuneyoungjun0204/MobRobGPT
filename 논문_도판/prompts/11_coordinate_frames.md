# Fig. 11 — Three-Layer Coordinate Mapping

- 파일명: `fig11_coords.png` · LaTeX label `fig:coords`
- 배치: 단단 `figure[H]`, `width=80mm` · 캔버스 **2400 x 1800 (4:3)**
- 레지스터: **A (지리 실사) → B (격자)** · 렌더링 문자열 **8개**

> 근거: `docs/unet_model_deploy.md` §2 — WGS84 ↔ world metres ↔ score-map pixel 의
> 3층 고정 affine. 앵커는 맵 정중앙 = 모선 위치. 축 순서는 `[ix, iy] = [x, y]`.

---

## INSTRUCTION

### Image Purpose
실해역 적용 절의 도판. 센서가 주는 위경도, 시뮬레이터의 미터 좌표, 그리고 정책이 보는 픽셀
격자가 하나의 고정된 변환 사슬로 이어져 있고, 앵커가 모선 위치에 고정된다는 것을 보인다.
이 사슬이 흔들리면 정책이 조용히 엉뚱한 곳을 찍으므로, 고정성 자체가 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 세 층이 각각 무엇이고 어느 방향으로 변환되는지가
수식 없이 세 개의 판과 두 개의 양방향 화살표로 읽혀야 한다.

### Key Message
위경도, 미터 좌표, 픽셀 격자는 모선에 고정된 하나의 변환 사슬로 이어진다.

### Scene Description
An off-white canvas holds three square panels of identical size stacked in a single vertical column
with generous gutters, joined by two vertical double-headed arrows placed in the gutters, and there is
no outer frame and no background panel. The top panel is a photographic near-vertical satellite view of
a real coastal sea area at orthophoto fidelity, showing genuine water texture, a real irregular
coastline with a rocky headland entering one corner, and muted terrain shading; over it a sparse
graticule of thin faint lines runs horizontally and vertically in the manner of a chart overlay, and a
small filled square marks a single anchor position offset from the panel centre, with a short thin
crosshair through it. The middle panel shows the same sea area but rendered as a plain flat working
field with the coastline retained only as a thin outline and a light fill, the photographic texture
gone; a small filled square marks the same anchor, now placed at the exact centre of the panel, and
from it two straight axis arrows extend, one to the right and one upward, each with a filled triangular
arrowhead and a short tick near its end, while a thin dimension line with end ticks spans the full
panel width just inside its lower edge. A photorealistically rendered grey frigate seen from directly
above sits exactly on the anchor square, small enough not to dominate, confirming that the anchor is
the mothership. The bottom panel shows the same region reduced to a regular square grid of thin lines
covering the panel edge to edge, each cell plainly visible, with the centre cell outlined in the accent
tone and carrying a small cross; a few cells carry a flat accent tint in the positions where vessels
would fall, and a short dimension bracket with end ticks spans exactly one cell along the panel's top
edge. Along the bottom panel's left edge and bottom edge, two short index arrows run outward from the
centre cell, one horizontal and one vertical, each with a small filled arrowhead, establishing the
index order. Each of the two gutter arrows is double-headed with filled triangular heads at both ends,
drawn in the primary tone, and each carries a small circle at its midpoint containing a plain numeral.
No watermarks, no blurry text, no placeholder brackets, no random artifacts, no coordinate values, no
latitude or longitude numbers, no distance values, no mathematical formulas, no figure caption numbers,
no legend boxes, no compass roses, no north arrows. CRITICAL: render only the eight exact English
strings and the two numerals listed in the CONTENT block below and nothing else. Do NOT invent, infer
or add any further text or numeric values. Never render Korean, Chinese, Japanese or Cyrillic
characters anywhere in the image.

### Rendering Style
- 서피스: 위에서 아래로 **사진 → 도면 → 격자** 로 추상도가 단조 증가한다.
  이 세 단계의 질감 차이가 도판의 전부이므로, 중간 판을 사진으로 그리거나 아래 판을
  도면으로 그리면 논지가 무너진다.
- 배경: 오프 화이트. 세 판은 동일 크기 정사각형이고 앵커 위치가 세 판에서 대응한다.
- 코너/경계: 직각. 세 판의 테두리 굵기가 같다.
- 연결선: 거터의 **양방향 화살표 2개**만. 양쪽 다 채워진 삼각 화살촉이어야 가역 변환임이 읽힌다.
  단방향으로 그리면 안 된다. 중간의 작은 원 안 숫자가 변환 단계를 표시한다.
- 시각장식: 치수선 2개(월드 폭 하나, 픽셀 한 칸 하나), 축 화살표 2개, 인덱스 화살표 2개.
  그 외 장식 없음.
- 공간구성: 1열 3단 수직 스택. 여백 15~25%. 라벨은 각 판 오른쪽 바깥 여백에 배치한다.
- 시각메타포: 같은 바다가 세 번 다시 그려지며 점점 추상화된다. 앵커 사각형만 세 판을 관통해
  같은 자리에 남아 있어, 고정 앵커가 사슬 전체를 붙들고 있음을 형태로 말한다.

### Content Placement
최상단 판 오른쪽 바깥에 'Geodetic' 을 중 라벨로 배치하고, 그 아래 한 단계 작게 'lat, lon' 을
소 라벨로 배치한다. 중간 판 오른쪽 바깥에 'World metres' 를 중 라벨로, 그 아래에 'x, y' 를 소
라벨로 배치한다. 하단 판 오른쪽 바깥에 'Score map' 을 중 라벨로, 그 아래에 'ix, iy' 를 소 라벨로
배치한다. 중간 판의 앵커 사각형 옆에 'Anchor' 를 소 라벨로 배치한다. 하단 판 한 칸 브래킷 위에
'Cell' 을 소 라벨로 배치한다. 두 양방향 화살표 중점의 작은 원 안에 위에서 아래로 '1' 과 '2' 를
숫자로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 실제 위경도 값, 미터 값, 픽셀 크기 값은 **넣지 않는다**.
변환의 가역성은 양방향 화살촉으로만, 축 순서는 인덱스 화살표 방향으로만 표현한다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: vertical_three_layer_stack
- print_target: 학술지 단단, 80 mm

### Background Treatment
- base: 오프 화이트 단색
- top_panel: 위성 정사영상 품질 실사 + 얇은 그래티큘 오버레이
- mid_panel: 평면 작업 도면 (해안선 외곽선 + 옅은 면)
- bottom_panel: 정사각 격자만
- texture: 최상단 판에만 사진 질감

### Color Palette
- primary: #2C3E50 (네이비) — 판 테두리, 양방향 화살표, 앵커 사각형, 축 화살표, 라벨
- secondary: #5D6D7E (슬레이트 블루) — 그래티큘, 격자선, 치수선, 해안선 외곽
- accent: #2980B9 (딥 블루) — 중앙 셀 강조와 십자, 격자 틴트 셀, 단계 번호 원
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: sea #8FA8BC→#5B7C95 (최상단 실사 수면) · land #E3DCC9 / 해안선 #B5AC93

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 2단계: 좌표계 이름(중), 성분 표기·부기(소)
- 성분 표기(`lat, lon` 등)는 소문자 그대로 두어 좌표 성분임을 형태로 드러낸다
- 라벨은 각 판 오른쪽 바깥 여백에 좌측 정렬로 통일

## CONTENT
layer_1: "Geodetic"
layer_1_c: "lat, lon"
layer_2: "World metres"
layer_2_c: "x, y"
layer_3: "Score map"
layer_3_c: "ix, iy"
anchor: "Anchor"
cell: "Cell"
step_1: "1"
step_2: "2"

## FORBIDDEN ELEMENTS
- 실제 위경도 값, 미터 값, 픽셀 크기 값 등 **정량 좌표 수치**
- 수식, affine 행렬 표기, 그리스 문자, 첨자
- 단방향 변환 화살표 (반드시 양방향이어야 한다)
- 중간 판을 사진으로, 하단 판을 도면으로 그리는 것 (추상도 단조 증가가 논지다)
- 방위 장미, 북침 화살표, 축척 바, 좌표 눈금 숫자
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 이미지 내부 캡션: "Fig. 11", "Figure 11", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 지도 서비스 워터마크, 지명 라벨, 국경선, 행정구역 경계
- 45도 아이소메트릭 3D, 지구본, 위성 아이콘, GPS 픽토그램
- 네온, 글로우, 드롭섀도, 어두운 배경, HUD
- 기관 로고, 워터마크
