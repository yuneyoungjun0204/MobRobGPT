# Fig. 4 — Commander Layer: Precomputed Geometry and Structured Output

- 파일명: `fig4_commander.png` · LaTeX label `fig:commander`
- 배치: 단단 `figure[H]`, `width=80mm` · 캔버스 **2400 x 1800 (4:3)**
- 레지스터: **B (표 + 스키마 + 실사 썸네일)** · 렌더링 문자열 **10개**

> 근거: `commander/schema.py` (BattlefieldState / CommanderPlan), `commander/geometry.py`
> (intercept_point, segments_cross, ship_cost), `commander/prompts.py` 의 출력 규약
> — rationale 이 deployments 보다 **먼저** 오는 필드 순서.

---

## INSTRUCTION

### Image Purpose
전략 계층의 내부를 여는 도판. 언어모델에 좌표 원시값이 아니라 기하량이 미리 계산된 표가
들어가고, 출력이 판단 근거를 맨 앞에 두는 고정 필드 순서로 강제되며, 그 결과가 적 무리별
담당 방어정과 대기 방어정으로 확정된다는 것을 한 장에 잇는다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 언어모델의 출력이 자유 서술이 아니라 스키마로 묶인
구조라는 점, 그리고 판단 근거가 결정보다 앞에 놓인다는 점이 형태만으로 읽혀야 한다.

### Key Message
기하량이 미리 계산된 상태가 들어가고, 판단 근거를 앞세운 고정 형식으로 배정이 확정된다.

### Scene Description
An off-white canvas arranged as a left-to-right three-stage flow, with no outer frame and no
background panel. The left third holds a compact quantity table drawn with genuine table rules in the
manner of a paper's data table: a thin horizontal rule above and below, one rule under the header row,
four body rows and three columns. The leftmost column of each body row holds a precise geometric
pictogram and no words — an angle wedge formed by two rays with a dimension arc between them, a
straight segment with tick marks at both ends, a curved arrow denoting a turn amount, and two crossing
straight segments with a small filled dot at their intersection. The middle and right columns hold
neutral placeholder bars standing in for a value and a unit; these bars must remain illegible and must
never resolve into readable digits. A solid arrow with a filled triangular arrowhead runs from this
table into the middle third. The middle third holds the structured-response object drawn as a schema
box: a tall rectangle with a thin accent border, its contents indented under a pair of tall square
brackets in the manner of a printed data structure. Three field rows appear in a fixed vertical order
at visibly unequal heights. The first and tallest row's value area is filled with three left-aligned
placeholder text bars of decreasing length, suggesting prose without any legible characters. The second
row's value area holds three small paired markers in a row, each pair being an angular vessel marker
joined by a short line to a circular target ring, so an assignment reads as a pairing. The third and
shortest row's value area holds one angular marker beside a short vertical pause bar. The deliberate
height ordering, largest at top and smallest at bottom, is the visual statement that the rationale is
produced first and at greatest length. The right third holds a small photographic aerial thumbnail of
the sea seen from above at satellite orthophoto fidelity: a photorealistically rendered grey frigate,
three small rigid-hull attack craft grouped at the upper right with white bow wakes, and two lighter
defending unmanned vessels. Over this imagery a light CAD overlay draws a thin line from the frigate to
the attacker group's centroid with a hollow circle and centre dot sitting on it, and two thin accent
lines pair each defending vessel to the group it has been given, while a third defending vessel is
drawn with a short vertical pause bar beside it and no pairing line at all, so assignment and hold are
distinguished purely by the presence or absence of a link. A single accent arrow with a filled
triangular arrowhead runs from the schema box's second field row into this thumbnail. No watermarks,
no blurry text, no placeholder brackets, no random artifacts, no readable body text inside any block,
no figure caption numbers, no legend boxes, no axis labels. CRITICAL: render only the ten exact English
strings listed in the CONTENT block below and nothing else. Do NOT invent, infer or add any further
text. Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the image. All
placeholder bars must remain illegible and must never resolve into readable words, numbers, JSON, or
code.

### Rendering Style
- 서피스: 좌측은 **진짜 표 괘선**을 가진 표, 중앙은 **진짜 대괄호**를 가진 스키마 박스,
  우측은 **실사 항공 썸네일**. 셋 다 자기 장르의 실물이어야 한다. 둥근 아이콘 상자로 대체하면 실패.
- 배경: 오프 화이트 단색. 블록 뒤에 패널·카드·프레임을 깔지 않는다.
- 코너/경계: 직각. 표는 가로 괘선만 쓰고 세로 괘선을 넣지 않는다(학술 표 관례).
- 연결선: 주 흐름은 주조색 실선 + 채워진 삼각 화살촉. 배정 주입선만 강조색.
  썸네일 안의 배정 짝짓기 선은 강조색 얇은 실선.
- 시각장식: 스키마 세 칸의 **높이 차이**가 유일한 강조 장치다. 색이나 배지로 강조하지 않는다.
- 공간구성: 좌 1/3 · 중 1/3 · 우 1/3 의 3단 좌우 흐름. 여백 15~25%.
- 시각메타포: 숫자 표가 스키마를 거쳐 해면 위 짝짓기로 내려앉는다. 대기 방어정만 연결선이
  없다는 사실이 "hold" 를 글 없이 말한다.

### Content Placement
좌측 표 아래에 'Precomputed geometry' 를 중 라벨로 배치하고, 표 각 행의 픽토그램 오른쪽 여백에
위에서 아래로 'bearing', 'range', 'turn', 'blocked' 를 소 라벨로 배치한다. 중앙 스키마 박스 아래에
'Structured output' 을 중 라벨로 배치하고, 세 필드 행의 필드명 자리에 위에서 아래로 'rationale',
'assignment', 'hold' 를 소 라벨로 배치한다. 우측 실사 썸네일 아래에 'Assignment' 를 중 라벨로
배치한다. 그 밖의 어떤 문자열도 렌더링하지 않는다. 판단 근거가 먼저 나온다는 것은 칸 높이로,
대기 상태는 연결선 부재와 정지 막대로만 표현하고 글로 적지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: three_stage_left_to_right
- print_target: 학술지 단단, 80 mm

### Background Treatment
- base: 오프 화이트 단색
- texture: 없음
- ornament: 없음. 상단 배너·하단 구분선 없음

### Color Palette
- primary: #2C3E50 (네이비) — 표 괘선, 스키마 대괄호, 픽토그램, 주 흐름 화살표, 라벨
- secondary: #5D6D7E (슬레이트 블루) — 플레이스홀더 바, 칸 구분선, 지시선
- accent: #2980B9 (딥 블루) — 스키마 박스 테두리, 배정 화살표, 썸네일 짝짓기 선, 요격점 원
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: sea #8FA8BC→#5B7C95 (썸네일 수면) · threat #B5493C (표적 링, 적 오버레이) ·
  white #FFFFFF (항적 거품, 라벨 헤일로)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 2단계: 단계명(중), 필드명·행 이름(소)
- 스키마 필드명과 표 행 이름은 소문자로 표기하여 데이터 필드임을 형태로 드러낸다
- 단계명은 각 블록 바로 아래 중앙 정렬로 통일

## CONTENT
stage_1: "Precomputed geometry"
geo_1: "bearing"
geo_2: "range"
geo_3: "turn"
geo_4: "blocked"
stage_2: "Structured output"
field_1: "rationale"
field_2: "assignment"
field_3: "hold"
stage_3: "Assignment"

## FORBIDDEN ELEMENTS
- 판독 가능한 가짜 본문: 읽히는 JSON, 중괄호 코드, 프롬프트 문장, 로그, 채팅 말풍선
- 플레이스홀더 바가 실제 글자·숫자로 해석되는 것
- 우측 썸네일을 일러스트·아이콘으로 대체
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 위치 지시자: [상단], [하단], [좌측], [우측]
- 이미지 내부 캡션: "Fig. 4", "Figure 4", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 범례 박스, 축 라벨, 좌표 눈금, 축척 바
- 45도 아이소메트릭 3D, 매거진 헤드라인, 번호 배지
- 네온, 글로우, 드롭섀도, 베벨, 어두운 배경, HUD, 채팅 UI, 터미널 창
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇 얼굴
- 기관 로고, 언어모델 제품명·회사명, 워터마크
- 한영 병기 표현: 배정 (Assignment), 근거 / Rationale
- 원문에 없는 정량 수치(각도, 반경, 척수) 표기
