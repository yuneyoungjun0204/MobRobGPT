# Fig. 3 — Commander Layer: Structured Assignment and the Resulting Valid Region

- LaTeX label: `fig:commander`
- 파일명: `fig3_commander.png`
- 배치: single column, `\begin{figure}[H]`, `width=80mm`
- 캔버스: 2400 x 1800 px (4:3)
- **상단 Register B (표 + 스키마) → 하단 Register A (실사 해면 + 오버레이)**

---

## INSTRUCTION

### Image Purpose
지휘관 계층의 내부를 여는 도판. 좌표 원시값이 아니라 기하량이 미리 계산된 표가 입력되고,
출력이 판단 근거를 먼저 두는 고정 순서의 구조적 형식으로 강제되며, 그 배정이 곧 기동
계층의 유효 후보 영역을 해면 위 부채꼴로 확정한다는 세 단계를 한 장에 잇는다.
배정이 왜 하위 계층의 선행 조건인지가 이 도판의 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 위쪽 절반이 표와 스키마라는 데이터 표현임이, 아래쪽
절반이 실제 해면 위 기하임이 한눈에 갈라져야 하고, 둘이 화살표 하나로 이어져야 한다.

### Key Message
지휘관의 배정이 각 방어정의 요격점과 후보 영역을 확정한다.

### Scene Description
A white figure canvas split into an upper data register and a lower scene register by generous white
space, with a single vertical flow arrow crossing between them. No outer frame, no background panel.

The upper register is a left-to-right pair of data objects. On the left, a compact quantity table
drawn with genuine table rules in the style of a paper's data table: a thin horizontal rule above and
below, one thin rule under the header row, four body rows, and three columns. The leftmost column of
each body row holds a precise geometric pictogram — an angle wedge formed by two rays with a dimension
arc between them, a straight segment with end ticks denoting a range, a curved arrow denoting a turn
amount, and two crossing straight segments with a small filled dot at their intersection denoting a
blocked route. The middle column holds a neutral grey placeholder bar standing in for a numeric value,
and the right column holds a shorter grey placeholder bar for its unit; these bars must remain
illegible and must never resolve into readable digits.

On the right of the upper register sits the structured-response object, drawn as a schema box: a tall
rectangle with a thin blue border, its contents indented under a pair of tall square brackets in the
manner of a printed data structure. Three field rows appear in a fixed vertical order and at visibly
unequal heights. The first and tallest row's value area is filled with three left-aligned grey
placeholder text bars of decreasing length, suggesting prose without any legible characters. The
second row's value area holds three small paired markers in a row, each pair being a blue angular
vessel marker joined by a short dark line to a red circular target ring. The third and shortest row's
value area holds one blue angular marker beside a short vertical pause bar. A solid dark arrow with a
filled triangular arrowhead runs from the table into this schema box.

A single blue arrow with a filled triangular arrowhead descends from the schema box's second field row
into the lower register, visually asserting that it is the assignment field that produces the geometry
below.

The lower register is a photographic near-vertical aerial view of open sea, rendered at satellite
orthophoto fidelity with real wave texture and a natural depth gradient. A photorealistically rendered
grey naval frigate sits at the lower-left of this register with a realistic wake. A loose group of
three small rigid-hull fast attack craft, each photorealistically rendered from above with sharp white
V-shaped bow wakes, sits at the upper right. One lighter defending unmanned vessel with a deck-mounted
net canister sits in the mid-field. Over this imagery a CAD-style vector overlay is applied: on the
straight line joining the frigate to the attacker group's centroid sits a hollow blue circle with a
centre dot marking the intercept point; a translucent blue sector opens from the frigate symmetric
about that same line, its two straight edges drawn as thin solid blue rays and its outer boundary as a
thin blue arc, and it is trimmed near the frigate by a thin grey inner arc so the usable region reads
as a sector-shaped band rather than a full wedge. A regular square candidate grid in thin white-cored
lines is overlaid across and slightly beyond this band; grid cells inside the band carry a light
translucent blue tint, while cells just outside the band edges carry a sparse grey forty-five degree
hatch, so admissible and inadmissible candidates are distinguished by hatch as well as by tint. The
overlay never fully obscures the water texture beneath it.

CRITICAL: render only the exact English strings listed in the CONTENT block below and nothing else. Do
NOT invent, infer or add any further text. Never render Korean, Chinese, Japanese or Cyrillic
characters anywhere in the image. All placeholder bars must remain illegible and must never resolve
into readable words, numbers, JSON, or code.

### Rendering Style
- 상단은 **데이터 객체의 실물**이다 — 진짜 표 괘선을 가진 표, 진짜 대괄호를 가진 스키마 박스.
  둥근 아이콘 상자로 대체하면 실패다.
- 상단 스키마 박스의 세 행 높이를 일부러 다르게 하여, 판단 근거 칸이 가장 크고 대기 칸이 가장
  작게 한다 — 출력 순서와 비중이 형태만으로 읽힌다.
- 플레이스홀더 바는 **끝까지 판독 불가**여야 한다. 가짜 JSON·가짜 문장·가짜 숫자로 채우면 실패다.
- 하단은 **실사**다. 위성 정사영상 품질 해면 + 사진 수준 탑다운 선박 렌더.
  속 빈 삼각형·쉐브론·픽토그램으로 배를 대체하면 즉시 실패다.
- 오버레이는 사진 위 CAD — 반투명 면(불투명도 25 % 이하) + 얇은 실선/파선 + 채워진 삼각 화살촉.
- 흑백 안전성: 유효 셀은 파란 틴트, 무효 셀은 회색 해칭 — 색을 빼도 해칭으로 갈린다.
- 배경: 상단 순백, 하단 실사. 드롭섀도·글로우 금지. 여백 15~25 %.

### Content Placement
상단 표 위 헤더 행에 좌에서 우로 'Quantity', 'Value', 'Unit' 을 소 라벨로 놓는다. 표 아래에
'Precomputed geometry' 를 대 라벨로 놓는다. 표 각 행의 픽토그램 오른쪽 여백에 위에서 아래로
'bearing', 'range', 'turn', 'blocked' 를 소 라벨로 놓는다. 스키마 박스 아래에 'Structured output'
을 대 라벨로 놓고, 세 필드 행의 필드명 자리에 위에서 아래로 'rationale', 'assignment', 'hold' 를
소 라벨로 놓는다. 하단으로 내려오는 파란 화살표 옆에 'Assigned target' 을 중 라벨로 놓는다.
하단 실사 장면에서 프리깃에서 지시선을 빼 'Mothership' 을 소 라벨로, 적 무리에서 지시선을 빼
'Assigned cluster' 를 소 라벨로, 파란 속 빈 원 옆에 'Intercept point' 를 중 라벨로 놓는다.
부채꼴 띠 안쪽 여백에 'Valid cells' 를 중 라벨로, 해칭된 바깥 셀 쪽에 'Masked out' 을 소 라벨로
놓는다. 부채꼴 정점 근처 두 직선 변 사이에 'Corridor' 를 소 라벨로 한 번만 놓는다.
그 밖의 어떤 문자열도 렌더링하지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: data_register_over_scene_register
- print_target: single journal column, 80 mm wide

### Color Palette
- ink `#1F2933` — 표 괘선, 스키마 대괄호, 픽토그램, 주 흐름 화살표, 라벨
- gray `#7B8794` — 플레이스홀더 바, 지시선, 내측 호, 무효 셀 해칭
- sea `#8FA8BC` → `#5B7C95` — 하단 실사 수면
- blue `#1F5FA8` — 스키마 박스 테두리, 배정 화살표, 요격점, 부채꼴 변과 호, 유효 셀 틴트, 아군 마커
- red `#C0392B` — 표적 링 마커, 적 오버레이
- white `#FFFFFF` — 상단 배경, 후보 격자 심, 라벨 헤일로

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 3단계: 단계명(대), 핵심 요소(중), 필드명·표 헤더·부기(소)
- 스키마 필드명과 표 행 이름은 소문자로 표기하여 데이터 필드임을 형태로 드러낸다
- 하단 실사 위 라벨은 얇은 흰 헤일로를 두어 판독되게 한다

## CONTENT
table_h1: "Quantity"
table_h2: "Value"
table_h3: "Unit"
stage1: "Precomputed geometry"
geo_1: "bearing"
geo_2: "range"
geo_3: "turn"
geo_4: "blocked"
stage2: "Structured output"
field_1: "rationale"
field_2: "assignment"
field_3: "hold"
edge_assign: "Assigned target"
scene_mother: "Mothership"
scene_cluster: "Assigned cluster"
scene_intercept: "Intercept point"
scene_valid: "Valid cells"
scene_masked: "Masked out"
scene_corridor: "Corridor"

## FORBIDDEN ELEMENTS
- **하단 선박을 아이콘으로 대체**: 속 빈 삼각형, 쉐브론, 픽토그램, 플랫 실루엣, 클립아트 보트
- 하단을 플랫 벡터 바다·단색 면으로 대체
- 판독 가능한 가짜 본문: 읽히는 JSON, 중괄호 코드, 프롬프트 문장, 로그, 채팅 말풍선
- 플레이스홀더 바가 실제 글자·숫자로 해석되는 것
- 이미지 내부 캡션: "Fig. 3", "Figure 3", 도판 제목
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 범례 박스, 좌표 눈금, 축척 바, 방위 장미
- 수평선, 하늘, 저각 시점
- 네온, 글로우, 드롭섀도, 베벨, 어두운 배경, HUD, 채팅 UI, 터미널 창
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇 얼굴
- 기관 로고, 언어모델 제품명·회사명, 워터마크
- 원문에 없는 정량 수치(각도, 반경, 척수) 표기
