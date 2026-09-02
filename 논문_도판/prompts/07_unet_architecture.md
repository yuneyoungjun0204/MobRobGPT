# Fig. 7 — Score-Map Network Architecture

- 파일명: `fig7_unet.png` · LaTeX label `fig:unet`
- 배치: **양단** `figure*[t]`, `width=0.94\textwidth` · 캔버스 **4200 x 1600 (2.6:1)**
- 레지스터: **B (U-Net 논문 관례)** · 렌더링 문자열 **7개 + 채널/해상도 수치**

> 근거: `boatattack_sim/model/cnn_actor.py::UNetLite` — stem 15→32, enc1 32→32(H/2),
> enc2 32→64(H/4), bottleneck 64→64, GAP→FiLM, dec2 96→32(H/2), dec1 64→32(H), head 32→d=32.
> 해상도 50→25→13→25→50 (`docs/unet_model_deploy.md` §1).

---

## INSTRUCTION

### Image Purpose
기동 계층 신경망의 구조 도판. 다채널 격자 관측이 수축 경로를 거쳐 병목에 이르고, 전역 문맥이
병목에 주입된 뒤 확장 경로가 skip 연결을 받아 **입력과 같은 해상도**의 특징맵으로 복원된다는
것을 보인다. 해상도 보존이 점수맵을 가능하게 하는 전제이므로, 그것이 형태로 주장되어야 한다.

### Target Audience
조선해양공학 학술지 심사자와 독자. U-Net 계열 도해에 익숙한 독자는 채널 수 표기와 skip
화살표만으로 구조를 즉시 읽고, 익숙하지 않은 독자도 좌우 대칭과 화살표 색으로 흐름을 얻는다.

### Key Message
수축과 확장이 대칭을 이루며 입력과 같은 해상도의 특징맵을 낸다.

### Scene Description
A wide off-white canvas holds a single symmetric U-shaped architecture diagram drawn in the visual
language of a segmentation-network paper, with no outer frame and no background panel. At the far left
the input is a flat offset stack of identical square raster slabs, each with a thin edge and a subtle
oblique side face, the frontmost rendered as a genuine viridis-colormapped raster with visible cell
structure. The network body is a sequence of tensor slab boxes, each drawn as a front rectangle with a
thin oblique top face and side face, filled a very pale accent tint with a darker edge, so each box
reads as a solid volume rather than a flat rectangle. The contracting path descends through three
levels: level one holds a pair of tall slabs at full height, level two a pair at half height positioned
lower on the canvas, level three a pair at quarter height positioned lower still, so the descent is
spatial as well as nominal. The bottleneck sits at the lowest point as a pair of slabs at the same
quarter height. The expanding path mirrors the contracting path exactly, ascending back through half
height to full height, and the final slab is drawn at precisely the same height as the first slab of
the contracting path, so equal input and output resolution is asserted by geometry alone. Every box
carries its channel count as a small numeral centred immediately above it. Each level carries its
spatial resolution as a small numeral rotated ninety degrees placed just outside the leftmost box of
that level. Arrows follow the standard convention of the genre: short horizontal accent-coloured arrows
between boxes within a level denote convolution blocks; downward arrows between levels on the
contracting side are drawn in a distinct warm tone; upward arrows between levels on the expanding side
are drawn in a distinct cool green tone; and long horizontal dashed arrows in the secondary tone cross
the open interior of the U from each contracting level to its matching expanding level, ending in small
open arrowheads, denoting skip connections. Below the bottleneck a small rounded box outlined in the
secondary tone connects upward into the bottleneck by a thin arrow, and a very small circle enclosing a
centred dot sits where that arrow meets the bottleneck, denoting the modulation of the bottleneck by a
pooled global descriptor. At the far right the output is a single square tile drawn at exactly the same
size as the frontmost input slab, filled with a fine faint grid and a small uniform dot at every cell
centre to indicate a per-cell feature vector, with its channel count annotated above it in the same
manner as the network boxes. All arrowheads are filled triangles except the skip arrows. No watermarks,
no blurry text, no placeholder brackets, no random artifacts, no readable text inside any box, no
figure caption numbers, no legend boxes, no kernel size annotations, no mathematical formulas.
CRITICAL: render only the exact English strings and numerals listed in the CONTENT block below and
nothing else. Do NOT invent, infer or add any further text, layer names, or numbers. Never render
Korean, Chinese, Japanese or Cyrillic characters anywhere in the image.

### Rendering Style
- 서피스: **오블리크 텐서 슬래브 박스** (정면 사각형 + 얇은 상단·측면 사면). 납작한 사각형만
  늘어놓거나 외곽선 실루엣 하나로 뭉뚱그리면 실패. 박스 채움은 아주 옅은 강조색 틴트로 통일.
- 배경: 오프 화이트 단색. U 자 내부는 비워 두어 skip 화살표가 지나갈 공간을 확보한다.
- 코너/경계: 직각. 전 박스의 사면 각도와 두께가 동일하다.
- 연결선: 색으로 연산을 구분하는 **장르 관례를 그대로 따른다** — 강조색 수평=합성곱,
  따뜻한 톤 하향=다운샘플, 차가운 초록 상향=업샘플, 보조색 수평 파선=skip.
  파선 skip 만 열린 화살촉을 쓰고 나머지는 채워진 삼각 화살촉.
- 시각장식: 박스 위 채널 수, 레벨 옆 90도 회전 해상도, 병목 아래 전역 문맥 박스와 변조 연산자.
  그 외 장식 없음. 번호 배지·아이콘 금지.
- 공간구성: 정확한 좌우 대칭 U 자. 출력 박스 높이가 입력 박스 높이와 **정확히 같아야** 한다.
  여백 15~25%.
- 시각메타포: 내려갔다 올라오는 대칭 경로와 그 위를 가로지르는 수평 다리들. 다리(skip)가
  U 자 내부의 빈 공간을 건너는 모습 자체가 해상도 복원의 은유가 된다.

### Content Placement
좌측 입력 스택 아래에 'Observation' 을 중 라벨로 배치하고, 스택 위 채널 수로 '15' 를,
해상도 관례 위치에 '50 x 50' 을 소 라벨로 배치한다. 수축 경로 박스 위 채널 수로 좌에서 우로
'32', '32', '64' 를, 병목 박스 위에 '64' 를, 확장 경로 박스 위에 '32', '32' 를 소 라벨로 배치한다.
각 레벨 왼쪽 바깥에 90도 회전한 해상도로 위에서 아래로 '50 x 50', '25 x 25', '13 x 13' 을 소
라벨로 배치한다. 수축 경로 아래 중앙에 'Encoder' 를, 병목 아래에 'Bottleneck' 을, 확장 경로 아래
중앙에 'Decoder' 를 중 라벨로 배치한다. 가장 위 skip 파선 위에 'skip connection' 을 소 라벨로
**한 번만** 배치한다. 병목 아래 작은 박스 옆에 'Global context' 를 소 라벨로 배치한다.
우측 출력 타일 아래에 'Feature map' 을 중 라벨로 배치하고 그 위 채널 수로 '32' 를 소 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 커널 크기, 정규화 종류, 활성 함수는 이미지에 넣지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1600
- aspect_ratio: 2.6:1
- layout: symmetric_u_architecture
- print_target: 학술지 양단 전폭, 165 mm

### Background Treatment
- base: 오프 화이트 단색
- texture: 없음
- ornament: 없음

### Color Palette
- primary: #2C3E50 (네이비) — 박스 외곽, 출력 타일 테두리, 라벨 텍스트, 수치
- secondary: #5D6D7E (슬레이트 블루) — skip 파선, 전역 문맥 박스, 특징 점, 격자
- accent: #2980B9 (딥 블루) — 합성곱 화살표, 박스 내부 옅은 틴트
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: down #B5493C (다운샘플 화살표) · up #2E8B57 (업샘플 화살표) ·
  colormap **viridis** (입력 스택 앞면 타일 전용)

### Typography
- 산세리프 단일 패밀리
- 크기 2단계: 구역명(중), 채널 수·해상도·부기(소)
- 채널 수는 박스 바로 위 중앙, 해상도는 레벨 왼쪽에 90도 회전 — 이 장르의 표준 위치
- 구역명은 해당 경로 아래 중앙 정렬

## CONTENT
stage_in: "Observation"
in_ch: "15"
in_dim: "50 x 50"
enc_ch_1: "32"
enc_ch_2: "32"
enc_ch_3: "64"
bott_ch: "64"
dec_ch_1: "32"
dec_ch_2: "32"
dim_1: "50 x 50"
dim_2: "25 x 25"
dim_3: "13 x 13"
region_enc: "Encoder"
region_bott: "Bottleneck"
region_dec: "Decoder"
note_skip: "skip connection"
note_film: "Global context"
stage_out: "Feature map"
out_ch: "32"

## FORBIDDEN ELEMENTS
- 신경망을 외곽선 실루엣 하나로 뭉뚱그리기, 납작한 사각형만 늘어놓기
- 채널 수·해상도 주기 생략 (이 장르의 필수 표기다)
- 출력 박스 높이가 입력 박스 높이와 다른 것 (해상도 보존 주장이 무너진다)
- 커널 크기, 정규화·활성함수 이름, 텐서 shape, 수식, 그리스 문자
- 이미지 플레이스홀더: [Image 1], [아이콘]
- 이미지 내부 캡션: "Fig. 7", "Figure 7", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·레이어 이름
- 범례 박스, 축 라벨, 눈금 숫자, 축척 바
- 아이소메트릭 큐브, 과장된 원근, 두꺼운 3D 판 옆면
- 뉴런 노드-엣지 그물망, 뇌 모양, 회로기판, 로봇
- 네온, 글로우, 드롭섀도, 베벨, 반사, 어두운 배경, HUD
- 무지개(jet) 컬러맵
- 기관 로고, 프레임워크 로고, 워터마크
- 원문에 없는 정량 성능 수치 표기
