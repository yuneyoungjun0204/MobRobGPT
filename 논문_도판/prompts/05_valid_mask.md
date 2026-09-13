# Fig. 5 — Construction of the Valid Action Mask

- 파일명: `fig5_mask.png` · LaTeX label `fig:mask`
- 배치: **양단** `figure*[t]`, `width=0.94\textwidth` · 캔버스 **4200 x 1300 (3.2:1)**
- 레지스터: **B (기하 집합 연산 도해)** · 렌더링 문자열 **5개**

> 근거: `boatattack_sim/env/defense_env.py::_cnn_valid_mask` — 환형 ∩ 육지제외 ∩ 반경게이트 ∩
> 방위게이트. 학습 체크포인트는 `cnn_gate_disjoint=False`(겹침 허용 레짐)라 **Voronoi 분할을 쓰지
> 않는다**. 본문 4.3.3 의 4단계(2500→1000→979→387→299)와 동일한 순서다.

---

## INSTRUCTION

### Image Purpose
기동 계층의 행동공간이 어떻게 좁혀지는지를 보이는 도판. 요격 환형에서 출발해 육지 제외, 요격점 기준
반경, 배정 클러스터를 향한 방위 부채꼴을 차례로 교집합하여
최종 후보 칸 집합이 나온다는 것을 좌에서 우로 잇는다. 마스크가 정책보다 먼저 행동을
결정한다는 점이 이 도판의 논지다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 집합이 단계마다 줄어드는 것만 보면 되고, 각 단계의
이름은 하단 라벨 하나로 충분해야 한다.

### Key Message
후보 영역은 환형·육지제외·반경·방위의 교집합으로 좁혀진다.

### Scene Description
A wide off-white canvas holds four square plan-view panels of identical size arranged left to right in
a single row, separated by uniform gutters, with a small operator glyph centred in each gutter and a
final result panel at the right end, and there is no outer frame and no background panel. Every panel
shows the same top-down sea region drawn as a light neutral field with a regular square candidate grid
of thin faint lines covering it edge to edge, and every panel carries the same fixed reference marks so
the four are directly comparable: a small filled square marking the mothership at the exact centre, a
small hollow circle with a centre dot marking the intercept point offset up and to the right of centre,
a compact angular vessel marker marking the defending vessel, and two small hollow triangular target
outlines marking the assigned enemy group beyond the intercept point. In every panel the cells that
belong to the admissible set are filled with a flat light accent tint while all remaining cells are
left plain and additionally covered by a sparse forty-five degree hatch, so admissible and inadmissible
regions are separated by hatch as well as by tint and survive greyscale printing. The first panel's
admissible set is a wide annular band centred on the mothership, bounded by a thin inner circle and a
thin outer circle, and an irregular closed landmass outline near one corner overlaps the band without
yet being excluded from it. The second panel repeats the annular band but the landmass region is now
cut out of the admissible set, drawn with a denser stipple inside its outline and hatched like every
other inadmissible cell, so the only change between the first and second panels is the removal of the
land. The third panel's admissible set is the intersection of that land-free band with a filled disc
centred on the intercept point, the disc drawn with a thin boundary circle and a thin dimension line
with end ticks running from the intercept point to the boundary. The fourth panel's admissible set is
the further intersection with a wedge opening outward from the mothership, symmetric about the
straight line joining the mothership to the intercept point, that line drawn as a thin dashed axis and
the wedge bounded by two thin straight rays with a small dimension arc spanning between them at their
vertex. Between consecutive panels a small circle enclosing a centred intersection symbol
sits in the gutter. At the right end, after a slightly wider gutter containing a small circle enclosing
an equals symbol, the result panel repeats the same grid and reference marks but its admissible set is
the small sector-shaped patch that survives all four constraints, tinted and outlined with a slightly
heavier accent boundary, with every other cell plain and hatched. No watermarks, no blurry text, no
placeholder brackets, no random artifacts, no figure caption numbers, no legend boxes, no axis labels,
no coordinate ticks, no scale bars, no panel letter markers such as a, b, c, d. CRITICAL: render only
the five exact English strings listed in the CONTENT block below and nothing else. Do NOT invent, infer
or add any further text. Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the
image.

### Rendering Style
- 서피스: 평면 기하 도해. 다섯 패널의 크기·격자 간격·기준 마커 위치가 **완전히 동일**해야
  집합이 줄어드는 것이 비교로 읽힌다. 하나라도 어긋나면 도판이 무의미해진다.
- 배경: 오프 화이트. 패널 안은 아주 옅은 중립 면. 실사 사진을 쓰지 않는다(이 도판은 기하다).
- 코너/경계: 패널은 직각 정사각형. 집합 경계선만 곡선(원·호·부채꼴 변)을 갖는다.
- 연결선: 연결 화살표를 쓰지 않는다. 대신 거터의 **교집합 연산자 기호**가 관계를 말한다.
  마지막 거터만 등호 기호를 쓴다.
- 시각장식: 치수선 2개(반경 하나, 각도 호 하나)와 파선 축 1개, 육지 스티플. 그 외 장식 없음.
- 공간구성: 5패널 1행 균등 배치, 거터 폭 균일, 마지막 거터만 약간 넓다. 여백 15~25%.
- 시각메타포: 네 개의 제약이 순서대로 겹쳐 마지막 한 조각만 남는다. 둘째 패널은 첫째와
  육지 한 덩어리만 다르므로, 그 미세한 차이가 '육지는 어떤 폴백에서도 되살아나지 않는다'를 말한다. 남은 조각의 작음 자체가 논지다.

### Content Placement
좌측부터 각 패널 하단 중앙에 'Annulus', 'Land excluded', 'Range gate', 'Bearing gate' 를 소 라벨로
차례로 배치한다. 맨 오른쪽 결과 패널 하단 중앙에 'Candidate cells' 를 중 라벨로 배치하여
결과 패널만 한 단계 큰 라벨을 갖게 한다. 그 밖의 어떤 문자열도 렌더링하지 않는다.
교집합 관계는 거터의 연산자 기호로만, 육지 제외는 스티플과 해칭의 추가로만 표현하고 글로
적지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1300
- aspect_ratio: 3.2:1
- layout: set_intersection_row
- print_target: 학술지 양단 전폭, 165 mm

### Background Treatment
- base: 오프 화이트 단색
- panel_fill: 아주 옅은 중립 면 + 얇은 정사각 후보 격자
- texture: 육지 영역 스티플만 예외로 허용
- ornament: 없음

### Color Palette
- primary: #2C3E50 (네이비) — 패널 테두리, 기준 마커, 연산자 기호, 라벨
- secondary: #5D6D7E (슬레이트 블루) — 후보 격자선, 무효 영역 해칭, 치수선, 파선 축, 분할 경계
- accent: #2980B9 (딥 블루) — 유효 집합 틴트와 경계선, 결과 패널 강조 테두리, 요격점
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: land #E3DCC9 / 해안선 #B5AC93 (육지 제외 영역) · threat #B5493C (적 표적 외곽)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 2단계: 결과 라벨(중), 단계 라벨(소)
- 라벨은 각 패널 하단 중앙, 패널 바깥 여백에 배치한다

## CONTENT
step_1: "Annulus"
step_2: "Land excluded"
step_3: "Range gate"
step_4: "Bearing gate"
result: "Candidate cells"

## FORBIDDEN ELEMENTS
- Voronoi 분할, 방어정별 영역 분할선 — 학습 체크포인트는 겹침 허용 레짐이라 이 단계가 없다
- 다섯 패널의 크기·격자·기준 마커 위치가 서로 달라지는 것
- 패널 문자 마커: (a), (b), (c), (d)
- 수식 표기: 집합 기호를 제외한 어떤 수식·그리스 문자·첨자도 금지
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 위치 지시자: [상단], [하단], [좌측], [우측]
- 이미지 내부 캡션: "Fig. 5", "Figure 5", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·단계 설명
- 범례 박스, 색상 키, 축 라벨, 좌표 눈금, 축척 바, 방위 장미
- 실사 사진 배경 (이 도판은 기하 도해다)
- 45도 아이소메트릭 3D, 매거진 헤드라인, 번호 배지
- 네온, 글로우, 드롭섀도, 베벨, 어두운 배경, HUD
- 무지개 컬러맵, 연속 그라디언트 채움
- 기관 로고, 워터마크
- 원문에 없는 정량 수치(반경 m, 각도 도, 후보 칸 수) 표기
