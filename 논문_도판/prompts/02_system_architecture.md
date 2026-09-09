# Fig. 2 — Two-Layer Architecture and Asynchronous Decision Cycles

- 파일명: `fig2_architecture.png` · LaTeX label `fig:architecture`
- 배치: **양단** `figure*[t]`, `width=0.92\textwidth` · 캔버스 **4200 x 1750 (12:5)**
- 레지스터: **B 주도 + 좌우 끝 A 실사 썸네일** · 렌더링 문자열 **12개**

---

## INSTRUCTION

### Image Purpose
논문 전체의 구조 도판. 전장 상태가 두 계층으로 갈라져 서로 다른 주기로 돌아가고, 느린 주기의
언어모델 지휘관이 배정을 확정하면 그 배정 아래에서 빠른 주기의 학습 정책이 매 결정마다 그물
투하 좌표를 고른다는 사실을, 상단 블록도와 하단 타이밍 차트로 동시에 성립시킨다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 블록 화살표만 따라가도 정보 흐름이, 하단 타이밍 차트만
봐도 왜 언어모델 응답 지연이 제어 주기를 막지 않는지가 읽혀야 한다.

### Key Message
느린 주기의 언어모델 배정과 빠른 주기의 학습 정책 기동이 비동기로 병렬 진행된다.

### Scene Description
A wide off-white figure canvas divided into an upper block-diagram register occupying the top two
thirds and a lower timing-chart register occupying the bottom third, separated by generous white space
and one thin full-width hairline rule, with no outer frame and no background panel. At the far left of
the upper register a rectangular panel with a thin border contains a genuine photographic aerial
thumbnail of the engagement scene: real sea texture seen from above, a photorealistically rendered grey
frigate at its centre, several small rigid-hull attack craft with white bow wakes around it, and two
lighter defending vessels, with a faint range ring overlaid. This panel is real imagery, not a diagram.
From it a solid arrow with a filled triangular arrowhead splits into an upper lane and a lower lane,
and this single fork is the visual claim that one state feeds two independent consumers. The upper lane
holds two stages in series: first a light rectangle containing a small four-row quantity table drawn
with real table rules, each row carrying a compact geometric pictogram on the left — an angle wedge
with an arc, a dimensioned segment with end ticks, a curved turn arrow, and two crossing segments with
a dot at their intersection — and a neutral placeholder bar on the right standing in for a value; then
the commander stage, a taller rectangle with an accent border containing a structured-response schema
box, its contents indented under tall square brackets with three field rows in fixed order at visibly
unequal heights, the first and tallest row's value area filled with three placeholder text bars of
decreasing length suggesting prose without any legible characters, the second row holding three small
paired markers where an angular vessel marker joins by a short line to a circular target ring, and the
third and shortest row holding one angular marker beside a short vertical pause bar. The lower lane
holds two stages in series: first a light rectangle containing a small stack of square heatmap
thumbnails drawn in the standard tensor-stack convention, identical squares offset by a constant
diagonal step, each a genuinely rendered viridis-colormapped raster with visible cell structure; then
the policy stage, a rectangle with an accent border containing a compact symmetric encoder-decoder
schematic drawn with oblique tensor slab boxes of decreasing then increasing size joined by short
arrows, with two dashed horizontal skip arrows arcing across the waist. Both lanes converge into a
terminal panel at the far right which is again real photographic aerial imagery: sea seen from above, a
defending vessel under way, and a deployed net barrier rendered as a line of small orange surface
floats with dark mesh visible below the surface, with an accent overlay segment and two small hollow
circles marking its endpoints. One accent arrow descends from the commander stage into the policy
stage, entering from directly above rather than from the left, so the assignment is visibly a side
conditioning input rather than part of the main data path. The lower register is a proper two-track
timing chart with a real axis: a horizontal axis line runs nearly the full width with regular tick
marks and a small arrowhead at its right end, the upper track carries a sparse row of long bars
separated by wide gaps, the lower track carries a dense evenly spaced row of short bars at several
times the rate, and one upper bar is drawn as a hollow outlined bar spanning many lower bars with a
thin span bracket beneath it, showing that the commander call is still in flight while the fast track
continues without interruption; a dashed vertical line drops from that hollow bar's trailing edge
through both tracks to the axis, marking the instant the new assignment takes effect. No watermarks,
no blurry text, no placeholder brackets, no random artifacts, no readable body text inside any block,
no figure caption numbers, no legend boxes. CRITICAL: render only the twelve exact English strings
listed in the CONTENT block below and nothing else. Do NOT invent, infer or add any further text. Never
render Korean, Chinese, Japanese or Cyrillic characters anywhere in the image. Placeholder bars stand
in for text and values and must remain illegible — they must never resolve into readable words, JSON,
or code.

### Rendering Style
- 서피스: 모든 정보 단위는 얇은 실선 테두리의 직사각형 안에 들어간다. 코너 반경과 테두리 굵기가
  전 블록에서 같다. **블록을 비우지 않는다** — 각 블록은 자기 내용물의 실물(실사 썸네일,
  viridis 히트맵, 표 괘선, 오블리크 텐서 슬래브)을 담는다. 이게 이 도판의 핵심이다.
- 배경: 오프 화이트 단색. 블록 뒤에 패널·카드·프레임·상단 배너를 깔지 않는다. 드롭섀도 금지.
- 코너/경계: 직각. 라운딩을 쓸 경우 아주 작은 동일 반경으로 전 블록에 일관 적용한다.
- 연결선: 주 데이터 흐름은 주조색 실선 + 채워진 삼각 화살촉, 예외 없이 좌에서 우.
  skip 만 보조색 점선. 배정 주입선만 강조색 실선이며 위에서 아래로 들어간다.
- 시각장식: 좌·우 끝 **실사 항공 썸네일** 2개, 타이밍 차트의 스팬 브래킷과 수직 파선.
  그 외 장식 없음. 번호 배지·아이콘 클러터 금지.
- 공간구성: 상단 2/3 블록도, 하단 1/3 타이밍 차트의 2단 구조. 두 레지스터 사이 거터를 넉넉히.
  여백 15~25%.
- 시각메타포: 하나의 상태가 갈라져 서로 다른 속도로 흐르다 한 지점에서 다시 만난다.
  타이밍 차트의 성긴 막대와 촘촘한 막대의 대비가 비동기성을 형태로 못박는다.

### Content Placement
좌측 실사 패널 아래에 'Battlefield state' 를 중 라벨로 배치한다. 상단 레인 첫 블록 아래에
'Geometric preprocessing' 을, 다음 강조색 블록 아래에 'LLM commander' 를 중 라벨로 배치하고,
그 스키마 박스 세 행의 필드명 자리에 위에서 아래로 'rationale', 'assignment', 'hold' 를 소 라벨로
배치한다. 하단 레인 첫 블록 아래에 'Rasterization' 을, 다음 강조색 블록 아래에 'Score-map policy'
를 중 라벨로 배치한다. 우측 실사 패널 아래에 'Net deployment' 를 중 라벨로 배치한다.
하단 차트 좌측 바깥에 트랙 라벨로 위에 'Commander cycle', 아래에 'Policy cycle' 을 중 라벨로
배치한다. 축 오른쪽 끝에 'Time' 을 소 라벨로 한 번만 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 배정이 조건 입력이라는 것, 호출이 진행 중이라는 것,
계획이 적용되는 시점은 전부 화살표 진입 방향과 속 빈 막대·수직 파선으로만 표현하고 글로 적지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1750
- aspect_ratio: 12:5
- layout: two_register_dataflow_over_timing_chart
- print_target: 학술지 양단 전폭, 165 mm

### Background Treatment
- base: 오프 화이트 단색
- texture: 없음
- ornament: 두 레지스터를 가르는 얇은 헤어라인 1개. 상단 배너·하단 구분선 없음

### Color Palette
- primary: #2C3E50 (네이비) — 블록 테두리, 주 흐름 화살표, 축선, 라벨 텍스트
- secondary: #5D6D7E (슬레이트 블루) — skip 점선, 표 괘선, 플레이스홀더 바, 스팬 브래킷, 눈금
- accent: #2980B9 (딥 블루) — 두 모델 블록 테두리, 배정 주입 화살표, 타이밍 막대, 그물 오버레이
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: tensor 면 #EAF2F8 / 외곽 #2C3E50 · threat #B5493C (스키마 표적 링, 실사 적 오버레이) ·
  float #E8862A (그물 부표) · sea #8FA8BC→#5B7C95 · colormap **viridis** (래스터 썸네일 전용)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 3단계: 블록명(중), 트랙명(중), 필드명·부기(소)
- 스키마 필드명은 소문자 그대로 표기하여 구조적 출력의 필드임을 형태로 드러낸다
- 블록명은 블록 바로 아래 중앙 정렬로 통일하고 위치를 흔들지 않는다

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
lane_slow: "Commander cycle"
lane_fast: "Policy cycle"
axis_time: "Time"

## FORBIDDEN ELEMENTS
- **빈 블록**: 내용물 없이 라벨만 있는 상자 — 각 블록은 자기 내용의 실물을 담아야 한다
- 좌·우 끝 패널을 일러스트/아이콘으로 대체 (반드시 실사 항공 썸네일)
- 판독 가능한 가짜 본문: 읽히는 JSON, 코드 블록, 프롬프트 문장, 로그, 채팅 말풍선
- 이미지 플레이스홀더: [Image 1], [사진], [이미지], [아이콘]
- 위치 지시자: [상단], [하단], [좌측], [우측]
- 이미지 내부 캡션: "Fig. 2", "Figure 2", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 45도 아이소메트릭 3D 오브젝트, 입체 화살표, 매거진 헤드라인, 번호 배지
- 네온, 글로우, 드롭섀도, 베벨, 유리 질감, 어두운 배경, HUD 프레임
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇 얼굴 등 AI 상투 모티프
- 브라우저 창, 터미널 창, 채팅 UI, 앱 목업 프레임
- 기관 로고, 언어모델 제품명·회사명, 워터마크
- 한영 병기 표현: 배정 (Assignment), 정책 / Policy
- 원문에 없는 정량 수치(지연 ms, 파라미터 수, 주기 초) 표기
