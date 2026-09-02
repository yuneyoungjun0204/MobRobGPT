# Fig. 2 — Two-Layer Architecture and Asynchronous Decision Cycles

- LaTeX label: `fig:architecture`
- 파일명: `fig2_architecture.png`
- 배치: **full width**, `\begin{figure*}[t]`, `width=0.92\textwidth`
- 캔버스: 4200 x 1750 px (12:5)
- **Register B (블록도) + 하단 실측 타이밍 차트 + 블록 내 실사/히트맵 썸네일**

---

## INSTRUCTION

### Image Purpose
논문 전체의 구조 도판. 전장 상태가 두 계층으로 갈라져 서로 다른 주기로 돌아가고,
느린 주기의 언어모델 지휘관이 배정을 확정하면 그 배정 아래에서 빠른 주기의 학습 정책이
매 결정마다 그물 투하 좌표를 고른다는 사실을, 상단 블록도와 하단 타이밍 차트로 동시에
성립시킨다. 두 계층이 서로를 기다리지 않는다는 비동기성이 이 도판의 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 블록 화살표만 따라가도 정보 흐름이, 하단 타이밍 차트만
봐도 왜 언어모델 응답 지연이 제어 주기를 막지 않는지가 읽혀야 한다.

### Key Message
높은 주기의 언어모델 배정과 낮은 주기의 학습 정책 기동이 비동기로 병렬 진행된다.

### Scene Description
A wide white figure canvas divided into an upper block-diagram register occupying the top two thirds
and a lower timing-chart register occupying the bottom third, separated by generous white space and by
one thin full-width hairline rule. There is no outer frame and no background panel.

The upper register is a left-to-right data-flow diagram in the style of a systems paper. At the far
left a rectangular panel with a thin dark border contains a genuine photographic aerial thumbnail of
the engagement scene: real sea texture seen from above, a photorealistically rendered grey frigate at
its centre, several small rigid-hull attack craft with white bow wakes around it, and two lighter
defending vessels, with a faint blue range ring overlaid. This panel is real imagery, not a diagram.
From it a solid dark arrow with a filled triangular arrowhead splits into an upper lane and a lower
lane, and this single fork is the visual claim that one state feeds two independent consumers.

The upper lane holds two stages in series. First, a light rounded rectangle containing a small
four-row quantity table drawn with real table rules: each row has a compact geometric pictogram on the
left (an angle wedge with an arc, a dimensioned segment with end ticks, a curved turn arrow, and two
crossing segments with a dot at the intersection) and a short right-aligned numeric field rendered as
a neutral grey placeholder bar, so it reads as a populated table of precomputed quantities without any
legible values. Second, the commander stage, a taller rounded rectangle with a blue border containing
a structured-response schema box: a small bracketed field list with three left-aligned rows in fixed
order, the first row's value area filled by three grey text-placeholder bars of decreasing length, the
second row's value area holding three small paired markers where a blue angular vessel marker is
joined by a short line to a red circular target ring, and the third row's value area holding a single
blue angular marker beside a short vertical pause bar.

The lower lane holds two stages in series. First, a light rounded rectangle containing a small stack of
square heatmap thumbnails drawn in the standard tensor-stack convention: identical squares offset by a
constant diagonal step, each a genuinely rendered viridis-colormapped raster with visible cell
structure, the frontmost one fully visible. Second, the policy stage, a rounded rectangle with a blue
border containing a compact symmetric encoder-decoder schematic drawn with oblique tensor slab boxes
of decreasing then increasing size, joined by short arrows, with two grey dashed horizontal skip arrows
arcing across the waist.

Both lanes converge into a terminal panel at the far right, which is again a real photographic aerial
thumbnail: sea seen from above, a defending vessel under way, and a deployed net barrier rendered as a
line of small orange surface floats with dark mesh visible below the surface, with a bold blue overlay
segment and two small hollow circles marking its endpoints. Additionally, one blue arrow descends from
the commander stage into the policy stage, entering it from directly above rather than from the left,
so the assignment is visibly a side conditioning input rather than part of the main data path.

The lower register is a proper two-track timing chart with a real axis. A horizontal axis line runs
nearly the full width with regular grey tick marks along it and a small arrowhead at its right end. The
upper track carries a sparse row of long blue bars separated by wide gaps, each bar representing one
commander cycle. The lower track carries a dense, evenly spaced row of short blue bars representing
policy decisions, at several times the rate of the upper track. One upper bar is drawn as a hollow
outlined bar spanning many lower bars, with a thin grey span bracket beneath it, showing that the
commander call is still in flight while the fast track continues without interruption. A blue dashed
vertical line drops from that hollow bar's trailing edge through both tracks to the axis, marking the
instant the new assignment takes effect. Each track has a short label at the left, outside the plotting
area, in the manner of a Gantt chart.

CRITICAL: render only the exact English strings listed in the CONTENT block below and nothing else. Do
NOT invent, infer or add any further text. Never render Korean, Chinese, Japanese or Cyrillic
characters anywhere in the image. Placeholder bars stand in for text and values and must remain
illegible — they must never resolve into readable words, JSON, or code.

### Rendering Style
- 블록: 동일 코너 반경의 라운드 사각형. 모든 블록의 반경과 테두리 굵기가 같다.
- **블록 안은 비우지 않는다.** 실사 썸네일, 실제 렌더된 viridis 히트맵, 실제 표 괘선,
  오블리크 텐서 슬래브 — 각 블록이 자기 내용물의 실물을 담는다. 이게 이 도판의 핵심이다.
- 텐서 슬래브는 정면 사각형 + 상단/측면 사면(oblique)의 입체 박스. 납작한 사각형 금지.
- 좌·우 끝 패널은 **실사 항공 사진**이다. 일러스트로 대체하면 실패다.
- 선: 주 흐름 2 px 잉크 실선 + 채워진 삼각 화살촉. skip 만 회색 점선. 배정 주입선만 파란 실선.
- 타이밍 차트는 **진짜 차트**다 — 축선, 눈금, 트랙 라벨, 스팬 브래킷을 갖춘다.
- 컬러맵은 viridis 하나로 고정. 다른 컬러맵 혼용 금지.
- 배경: 순백. 블록 뒤 패널·카드·프레임 없음. 드롭섀도 금지.
- 여백: 15~25 %. 상하 두 레지스터 사이 거터를 넉넉히.

### Content Placement
좌측 실사 패널 아래에 'Battlefield state' 를 대 라벨로 놓는다. 상단 레인 첫 블록 아래에
'Geometric preprocessing' 을, 다음 파란 블록 아래에 'LLM commander' 를 대 라벨로 놓고,
그 스키마 박스 세 행의 왼쪽 필드명 자리에 위에서 아래로 'rationale', 'assignment', 'hold' 를
소 라벨로 놓는다. 하단 레인 첫 블록 아래에 'Rasterization' 을, 다음 파란 블록 아래에
'Score-map policy' 를 대 라벨로 놓는다. 우측 실사 패널 아래에 'Net deployment' 를 대 라벨로 놓는다.
지휘관에서 정책으로 내려오는 파란 화살표 옆에 'Assigned target' 을 중 라벨로 놓는다.
하단 차트 좌측 트랙 라벨로 위에 'Commander cycle', 아래에 'Policy cycle' 을 중 라벨로 놓는다.
속 빈 막대 아래 브래킷 옆에 'Call in flight' 를, 파란 수직 파선 옆에 'Plan applied' 를 소 라벨로 놓는다.
축 오른쪽 끝에 'Time' 을 소 라벨로 한 번만 놓는다.
그 밖의 어떤 문자열도 렌더링하지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1750
- aspect_ratio: 12:5
- layout: two_register_dataflow_over_timing_chart
- print_target: full text width, 165 mm wide

### Color Palette
- ink `#1F2933` — 블록 테두리, 주 흐름 화살표, 축선, 라벨
- gray `#7B8794` — skip 점선, 표 괘선, 플레이스홀더 바, 스팬 브래킷, 눈금
- tensor `#EAF2F8` 면 / `#2C3E50` 외곽 — 텐서 슬래브
- blue `#1F5FA8` — 두 모델 블록 테두리, 배정 화살표, 타이밍 막대, 그물 오버레이
- red `#C0392B` — 스키마 박스 안 표적 링, 실사 썸네일 위 적 오버레이
- colormap **viridis** — 래스터 썸네일 전용
- white `#FFFFFF` — 배경

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 3단계: 블록명(대), 요소·트랙명(중), 필드명·부기(소)
- 스키마 필드명은 소문자 그대로 표기하여 구조적 출력의 필드임을 형태로 드러낸다
- 블록명은 블록 바로 아래 중앙 정렬로 통일

## CONTENT
block_state: "Battlefield state"
block_geo: "Geometric preprocessing"
block_llm: "LLM commander"
field_1: "rationale"
field_2: "assignment"
field_3: "hold"
block_raster: "Rasterization"
block_policy: "Score-map policy"
block_net: "Net deployment"
edge_assign: "Assigned target"
lane_slow: "Commander cycle"
lane_fast: "Policy cycle"
mark_inflight: "Call in flight"
mark_applied: "Plan applied"
axis_time: "Time"

## FORBIDDEN ELEMENTS
- **빈 블록**: 내용물 없이 라벨만 있는 상자 — 각 블록은 자기 내용의 실물을 담아야 한다
- 좌·우 끝 패널을 일러스트/아이콘으로 대체 (반드시 실사 항공 썸네일)
- 판독 가능한 가짜 본문: 읽히는 JSON, 코드 블록, 프롬프트 문장, 로그, 채팅 말풍선
- 이미지 내부 캡션: "Fig. 2", "Figure 2", 도판 제목
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 네온, 글로우, 드롭섀도, 베벨, 유리 질감, 어두운 배경, HUD 프레임
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇 얼굴 등 AI 상투 모티프
- 브라우저 창, 터미널 창, 채팅 UI, 앱 목업 프레임
- 기관 로고, 언어모델 제품명·회사명, 워터마크
- 원문에 없는 정량 수치(지연 ms, 파라미터 수, 주기 초) 표기
