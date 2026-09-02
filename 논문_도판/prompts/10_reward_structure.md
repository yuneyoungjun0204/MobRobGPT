# Fig. 10 — Reward Structure: Event Terms and Potential Shaping

- 파일명: `fig10_reward.png` · LaTeX label `fig:reward`
- 배치: 단단 `figure[H]`, `width=80mm` · 캔버스 **2400 x 1800 (4:3)**
- 레지스터: **A (사건 썸네일) + B (퍼텐셜 장)** · 렌더링 문자열 **7개**

> 근거: `boatattack_sim/env/reward.py` — 이벤트 지배항(capture / breach / ally net contact)에
> `threat_potential` 기반 potential-based shaping(γΦ' − Φ)을 더한다. 최적 정책을 보존한다
> (Ng et al., 1999). **계수 값은 도판에 넣지 않는다** — 부호와 구조만 보인다.

---

## INSTRUCTION

### Image Purpose
보상 설계 도판. 보상이 드문드문한 사건 항과 매 스텝 밀집한 퍼텐셜 항의 두 층으로 이루어져
있고, 사건 항은 부호로만 구분되며, 퍼텐셜 항은 적이 모선에 가까울수록 커지는 장의 차분으로
주어져 최적 정책을 바꾸지 않는다는 것을 보인다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 어떤 사건이 상이고 어떤 사건이 벌인지, 그리고 퍼텐셜이
공간의 함수라는 것이 수식 없이 읽혀야 한다.

### Key Message
드문 사건 항에 매 스텝의 퍼텐셜 차분을 더해 밀집 신호를 만든다.

### Scene Description
An off-white canvas split into a left column of event rows and a right potential-field block, joined at
the bottom by a single summation glyph, with no outer frame and no background panel. The left column
holds three horizontal rows of equal height, each row a small photographic aerial thumbnail on the left
paired with a large sign glyph on the right, and thin hairline rules separating the rows. Every
thumbnail is a near-vertical aerial view of sea at satellite orthophoto fidelity with genuine wave
texture, all three sharing the same water tone and sun direction. The first thumbnail shows a small
rigid-hull attack craft stopped dead in the water immediately behind a net barrier of small orange
surface floats with dark mesh below the waterline, its bow wake collapsed into a flat disturbed patch,
and beside it a large plus sign drawn in the accent tone. The second thumbnail shows a small attack
craft with an intact sharp white bow wake passing close alongside a photorealistically rendered grey
frigate, having reached it, and beside it a large minus sign drawn in a warm tone. The third thumbnail
shows a light-hulled defending unmanned vessel fouled against a net barrier, its own wake collapsed,
and beside it a large minus sign in the same warm tone. The three sign glyphs are drawn at identical
size and aligned in a single vertical column so the sign, not the magnitude, is what the reader
compares. The right block holds a potential field rendered as a genuine viridis-colormapped raster over
a square plan-view region with visible cell structure, brightest in a compact zone at the centre and
falling off smoothly outward, with a small filled square marking the mothership at that centre and
three small hollow triangular target outlines placed at differing distances from it, each carrying a
thin faint radial line back to the centre. Immediately to its right, a second identical field is drawn
with the same colormap but with the three target outlines shifted slightly inward along their radial
lines, and between the two fields sits a small circle enclosing a minus sign, so the pair reads as a
difference between two consecutive states rather than as two independent maps. A vertical viridis
colorbar with a thin border and small end ticks stands to the right of the pair, unlabelled. Beneath
both the left column and the right block, two short arrows converge into a single small circle
enclosing a plus sign centred at the bottom of the canvas, from which one short arrow points downward
into open space. No watermarks, no blurry text, no placeholder brackets, no random artifacts, no
coefficient values, no numeric weights, no axis tick numbers, no mathematical formulas, no Greek
letters, no figure caption numbers, no legend boxes. CRITICAL: render only the seven exact English
strings listed in the CONTENT block below and nothing else. Do NOT invent, infer or add any further
text or numeric values. Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the
image.

### Rendering Style
- 서피스: 좌측 사건 썸네일은 **실사** (세 폭의 수면 톤·태양 방향 동일). 우측 퍼텐셜 장은
  **진짜 렌더된 viridis 히트맵**. 두 성격이 좌우로 뚜렷이 갈려야 한다.
- 배경: 오프 화이트. 행 구분은 얇은 헤어라인만.
- 코너/경계: 직각. 썸네일과 장 블록 모두.
- 연결선: 하단 합류만 화살표를 쓴다. 채워진 삼각 화살촉. 그 외 연결선 없음.
- 시각장식: 부호 글리프(플러스/마이너스)가 이 도판의 유일한 강조 장치다. 세 부호의 크기가
  같아야 하고, 색으로만 상벌이 갈린다(강조색=상, 따뜻한 톤=벌). 배지·아이콘 클러터 금지.
- 공간구성: 좌 1/2 사건 3행 / 우 1/2 퍼텐셜 2연폭 / 하단 합류의 구조. 여백 15~25%.
- 시각메타포: 드문드문한 사건(사진 세 장)과 연속적인 장(히트맵)이 아래에서 하나로 합쳐진다.
  두 층의 밀도 차이가 시각적으로 대비되는 것이 논지다.

### Content Placement
좌측 세 행의 부호 글리프 오른쪽 바깥에 위에서 아래로 'Capture', 'Breach', 'Net contact' 를 소
라벨로 배치한다. 좌측 열 전체 아래에 'Event terms' 를 중 라벨로 배치한다. 우측 두 장 위쪽
가운데에 'Threat potential' 을 중 라벨로 배치한다. 우측 두 장 아래에 'Shaping' 을 소 라벨로
배치한다. 하단 합류 원 아래에 'Reward' 를 중 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 계수 값, 가중치, 할인율은 **어떤 형태로도 넣지 않는다**.
상벌 구분은 부호 글리프와 색으로만, 두 상태의 차분은 원 안 마이너스 기호로만 표현한다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: event_column_beside_field_pair
- print_target: 학술지 단단, 80 mm

### Background Treatment
- base: 오프 화이트 단색
- thumbnails: 위성 정사영상 품질 실사 해면 3폭 (수면 톤·조명 동일)
- texture: 사진 고유 질감만
- ornament: 행 구분 헤어라인만

### Color Palette
- primary: #2C3E50 (네이비) — 블록 테두리, 헤어라인, 화살표, 연산자, 라벨
- secondary: #5D6D7E (슬레이트 블루) — 지시선, 방사선, 셀 격자
- accent: #2980B9 (딥 블루) — 플러스 부호, 모선 마커, 그물 오버레이
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: penalty #B5493C (마이너스 부호, 적 오버레이) · sea #8FA8BC→#5B7C95 ·
  float #E8862A (그물 부표 실물색) · colormap **viridis** (퍼텐셜 장, 컬러바)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 2단계: 구역명(중), 사건명·부기(소)
- 세 사건명은 같은 크기·굵기여야 부호 비교로 읽힌다
- 실사 썸네일 위에는 라벨을 얹지 않고 바깥 여백에 배치한다

## CONTENT
event_1: "Capture"
event_2: "Breach"
event_3: "Net contact"
group_event: "Event terms"
field: "Threat potential"
field_note: "Shaping"
total: "Reward"

## FORBIDDEN ELEMENTS
- 계수 값, 가중치 숫자, 할인율, 보상 스케일 등 **어떤 정량 수치도 금지**
- 수식, 그리스 문자, 시그마, 감마, 파이 기호, 첨자
- 축 눈금 숫자, 컬러바 눈금 숫자
- 학습 곡선, 손실 그래프 — 이 도판은 결과 그래프가 아니다
- 사건 썸네일을 아이콘·일러스트로 대체
- 세 썸네일의 수면 톤·조명이 서로 달라지는 것
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 이미지 내부 캡션: "Fig. 10", "Figure 10", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 폭발, 화염, 연기, 피격 효과
- 45도 아이소메트릭 3D, 번호 배지, 매거진 헤드라인
- 네온, 글로우, 드롭섀도, 어두운 배경, HUD
- 기관 로고, 워터마크
