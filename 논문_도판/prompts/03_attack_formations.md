# Fig. 3 — Attack Formation Scenarios

- 파일명: `fig3_formations.png` · LaTeX label `fig:formations`
- 배치: **양단** `figure*[t]`, `width=0.92\textwidth` · 캔버스 **4200 x 1400 (3:1)**
- 레지스터: **A 전면 (실사 해면 3연폭)** · 렌더링 문자열 **3개**

> 근거: `boatattack_sim/env/formations.py::spawn_enemies` 의 `concentrated` / `diversionary` /
> `wave` 세 모드. `30_model/` 아래 세 모드별 학습 모델이 각각 존재한다.

---

## INSTRUCTION

### Image Purpose
문제 설정 절의 도판. 적 군집이 단일 방위 집중, 양동 후 반대편 소수, 시차 웨이브의 세 가지
접근 양상을 취한다는 것을 실사 해면 세 폭으로 나란히 보인다. 배정 판단이 왜 고정 규칙으로
풀리지 않고 상황별 유연한 판단을 요구하는지를 그림 하나로 근거 짓는다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 세 폭의 차이를 글 없이 배치와 항적만으로 구별해야 한다.

### Key Message
적의 접근 양상이 시나리오마다 근본적으로 달라 고정 배정 규칙으로는 대응되지 않는다.

### Scene Description
Three equal-width photographic aerial panels stand side by side across a wide off-white canvas,
separated by narrow uniform gutters, with no outer frame and no background panel. Every panel is a
near-vertical aerial view of the same open sea area rendered at satellite orthophoto fidelity with
genuine fine wave texture, subtle swell banding and a natural depth gradient, lit by a single high sun
from the upper left so all hull shadows fall consistently. Every panel contains the identical
photorealistically rendered grey naval frigate at its exact centre, seen from directly above with its
hull taper, superstructure, mast and helicopter deck legible and a broad white wake astern, and the
identical two thin range rings overlaid on it, so the three panels differ only in how the attackers are
arranged. The left panel shows all ten small rigid-hull fast attack craft massed into one tight
cohesive body arriving from a single bearing, their white V-shaped bow wakes nearly parallel and their
thin dashed inbound tracks converging into one narrow bundle, giving the impression of a single
concentrated thrust along one axis. The centre panel shows a large body of attack craft arriving from
one bearing while a distinctly smaller detachment of two or three craft approaches from very nearly
the opposite side of the frigate, the two groups' dashed tracks arriving on opposing axes, so the
reader immediately reads a main effort and a separate small counter-side element. The right panel
shows the craft arranged as two successive echelons from two adjacent bearings, the leading echelon
already well inside the outer ring with long developed wakes and the trailing echelon still far out
near the frame edge with much shorter wakes just forming, and each craft's dashed track drawn with a
gentle serpentine weave rather than a straight run, so the staggered arrival in time is legible from
wake length and track curvature alone. Two defending unmanned surface vessels with lighter hulls and
deck-mounted net canisters sit in the mid-field of each panel in identical positions across all three,
holding station without deployed nets, so the reader sees the same defensive starting condition
against three different threats. Thin dashed inbound tracks are drawn in a restrained secondary tone
and never obscure the water texture. No watermarks, no blurry text, no placeholder brackets, no
duplicated labels, no random artifacts, no horizon, no sky, no explosions, no smoke, no figure caption
numbers, no legend boxes, no scale bars, no compass roses, no panel letter markers such as a, b, c.
CRITICAL: render only the three exact English strings listed in the CONTENT block below and nothing
else. Do NOT invent, infer or add any further text. Never render Korean, Chinese, Japanese or Cyrillic
characters anywhere in the image.

### Rendering Style
- 서피스: 위성 정사영상 품질 실사 해면 + 사진 수준 탑다운 선박 렌더. 세 폭의 해역·모선 위치·
  방어정 위치·거리환이 **완전히 동일**해야 하고 적 배치와 항적만 달라진다. 그래야 비교가 성립한다.
- 배경: 실제 수면. 세 폭의 수면 톤과 태양 방향이 동일하다.
- 코너/경계: 세 폭은 동일 크기 직사각형이며 거터 폭이 균일하다. 패널 테두리는 아주 얇거나 없다.
- 연결선: 적 항적은 보조색 얇은 파선. 화살촉을 붙이지 않는다(방향은 선수파가 이미 말한다).
- 시각장식: 동심 거리환 2개만. 부채꼴·후보격자·그물은 이 도판에 넣지 않는다(아직 미전개 상태).
- 공간구성: 3연폭 균등 분할. 각 폭 하단 중앙에 라벨 하나씩. 여백 15~25%.
- 시각메타포: 같은 무대에 세 개의 다른 위협 기하가 겹쳐 놓인 대조 구성. 차이가 곧 논지다.

### Content Placement
좌측 폭 하단 중앙에 'Concentrated' 를 중 라벨로 배치한다. 중앙 폭 하단 중앙에 'Diversionary' 를
중 라벨로 배치한다. 우측 폭 하단 중앙에 'Wave' 를 중 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 시차·주공/조공·집중도는 전부 항적 길이, 선수파 발달
정도, 트랙 곡률로만 표현하고 글로 적지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 4200 x 1400
- aspect_ratio: 3:1
- layout: three_panel_comparison
- print_target: 학술지 양단 전폭, 165 mm

### Background Treatment
- base: 오프 화이트 캔버스 위에 실사 패널 3개
- water: 세 폭 모두 동일한 미세 파문·스웰·깊이 색변화
- lighting: 좌상단 고각 단일 태양, 세 폭 그림자 방향 일치
- texture: 사진 고유 질감만

### Color Palette
- primary: #2C3E50 (네이비) — 라벨 텍스트, 패널 경계
- secondary: #5D6D7E (슬레이트 블루) — 적 파선 항적, 동심 거리환
- accent: #2980B9 (딥 블루) — 방어정 위치 마커 오버레이
- background: #F8F9FA (오프 화이트) — 캔버스 여백, 거터
- 확장: sea #8FA8BC→#5B7C95 · threat #B5493C (적 오버레이) · white #FFFFFF (항적 거품, 라벨 헤일로)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 1단계만: 세 폭 라벨이 모두 같은 크기·굵기여야 비교 도판으로 읽힌다
- 라벨은 각 폭 하단 중앙, 패널 바깥 여백에 배치한다

## CONTENT
panel_1: "Concentrated"
panel_2: "Diversionary"
panel_3: "Wave"

## FORBIDDEN ELEMENTS
- 세 폭의 해역·모선·방어정 배치가 서로 달라지는 것 (적 배치와 항적만 달라야 한다)
- 배를 아이콘으로 대체: 속 빈 삼각형, 쉐브론, 픽토그램, 클립아트 보트
- 패널 문자 마커: (a), (b), (c), a., b., c.
- 이미지 내부 캡션: "Fig. 3", "Figure 3", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·설명문
- 범례 박스, 축 라벨, 좌표 눈금, 축척 바, 방위 장미
- 그물벽, 후보 격자, 부채꼴 — 이 도판은 전개 이전 시점이다
- 수평선, 하늘, 저각 시점
- 45도 아이소메트릭 3D, 매거진 헤드라인, 번호 배지
- 네온, 글로우, 드롭섀도, HUD, 레이더 스코프
- 폭발, 화염, 연기
- 기관 로고, 함번, 국적 표식, 워터마크
- 원문에 없는 정량 수치(척수, 속력, 거리) 표기
