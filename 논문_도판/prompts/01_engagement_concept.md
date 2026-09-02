# Fig. 1 — Engagement Concept

- 파일명: `fig1_concept.png` · LaTeX label `fig:concept`
- 배치: 단단 `figure[H]`, `width=80mm` · 캔버스 **2400 x 1800 (4:3)**
- 레지스터: **A 전면 (실사 해면)** · 렌더링 문자열 **6개**

---

## INSTRUCTION

### Image Purpose
학술지 논문 서론의 작전 개념도. 저비용 자폭 무인수상정 군집이 고가치 모선으로 여러 방위에서
수렴하고, 그보다 느린 소수의 방어정이 추격 대신 접근 회랑을 앞질러 포획 그물벽으로 닫는다는
교전 개념을 성립시킨다. 실제 해역 항공영상 위에 CAD 주석을 얹은 2층 구조다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 캡션 한 줄과 이 도판만으로 "방어측이 더 느린데 왜 추격이
아니라 사전 차단인가"를 기하만으로 납득해야 한다.

### Key Message
느린 소수의 방어정이 적의 접근 회랑을 앞질러 그물벽으로 닫는다.

### Scene Description
A high-altitude near-vertical aerial view of a real coastal sea area, rendered photographically at the
fidelity of a satellite orthophoto. The water is genuine open sea seen from above with visible fine
wave texture, subtle swell banding, and a natural depth gradient from lighter shallows along the
upper-left margin to deeper water toward the lower right; it is emphatically not a flat colour fill. A
real irregular coastline with a small rocky headland and its reef shelf enters the upper-left corner
with muted terrain shading and a thin natural surf line. The sun is high and slightly to the upper
left, so every hull casts a short soft physically consistent shadow onto the water. At the geometric
centre sits a photorealistically rendered grey naval frigate seen from directly above at roughly one
fifth of the frame height, its hull taper, forecastle, superstructure, mast, funnel and helicopter deck
all legible, trailing a broad white wake of realistic turbulent foam. Ten small suicide craft are
distributed around the outer half of the frame in three loose groups arriving from three separated
bearings, each a photorealistically rendered rigid-hull fast attack boat seen from above at roughly one
twenty-fifth of the frame height, each cutting a sharp white V-shaped bow wake and a narrow foaming
trail that indicates heading and speed. Three defending unmanned surface vessels, visibly fewer than
the attackers, rendered with the same photographic fidelity but with a lighter hull and a compact
deck-mounted net canister aft, sit in the mid-field, one facing each arriving group; their wakes are
markedly shorter and narrower than the attackers' wakes so the speed disadvantage is legible from the
foam alone and needs no words. Across the mouths of two of the three approach corridors a capture net
barrier is already deployed and rendered as a real physical object: a taut line of small orange surface
floats strung at regular intervals with dark mesh visible just below the surface between them,
slightly disturbed water on either side, and a defending vessel sitting at one end having just laid it.
One attacking craft immediately behind one barrier is dead in the water, its bow wake collapsed into a
flat disturbed patch with no forward foam, fouled in the mesh. The third corridor is still open and its
defending vessel is under way along a short track toward where its barrier will go. Over this
photographic base sits a crisp vector annotation layer drawn as if a CAD overlay were placed on the
imagery: two thin range rings centred exactly on the frigate with a very light translucent wash in the
band between them that never obscures the water beneath; three translucent sector wedges, one per
arriving group, opening outward from the frigate along each group's bearing at low opacity with thin
dashed edges; a thin dashed track from each attacking craft running inward; a thin solid track from
each defending vessel to its net position terminated by a filled triangular arrowhead; and small
hollow circles marking the two endpoints of each deployed barrier. No watermarks, no blurry text, no
placeholder brackets, no duplicated labels, no random artifacts, no horizon, no sky, no explosions, no
smoke, no figure caption numbers, no legend boxes, no scale bars, no compass roses. CRITICAL: render
only the six exact English strings listed in the CONTENT block below and nothing else. Do NOT invent,
infer or add any further text. Never render Korean, Chinese, Japanese or Cyrillic characters anywhere
in the image. Any region without an assigned CONTENT string must be carried by imagery and geometry
alone, never by generated words.

### Rendering Style
- 서피스: 위성 정사영상 품질의 실사 해면 + 사진 수준 탑다운 선박 렌더. 갑판 구조물·함교·마스트·
  항적 거품·선체 그림자가 보인다. 아이콘·삼각형·쉐브론·플랫 실루엣으로 배를 대체하면 즉시 실패.
- 배경: 실제 수면. 미세 파문, 스웰 밴딩, 얕은곳→깊은곳 색 변화. 좌상단 실제 형상 해안선.
  단색 면, 벡터 바다, 만화풍 파도 금지.
- 코너/경계: 오버레이 도형만 기하 요소를 갖는다. 사진에는 프레임·테두리를 두르지 않는다.
- 연결선: 적 항적은 보조색 파선, 아군 항적은 강조색 실선 + 채워진 삼각 화살촉. 선종으로 갈린다.
- 시각장식: 동심 거리환 2개, 반투명 코리도 부채꼴 3개, 그물 양끝 속 빈 원. 그 외 장식 없음.
- 공간구성: 모선을 정중앙에 두고 세 방위에서 수렴하는 방사 구도. 여백 15~25%.
  라벨은 선박 위에 겹치지 않게 지시선으로 빼낸다.
- 시각메타포: 완성된 두 개의 닫힌 회랑과 아직 열린 한 개의 회랑이 나란히 놓여, 방어의 진행
  상태 자체가 장면의 논지가 된다. 사람도 무기도 등장하지 않는다.

### Content Placement
모선 프리깃에서 우측 여백으로 지시선을 빼내 'Mothership' 을 중 라벨로 배치한다. 가장 위쪽 적
무리에서 지시선을 빼 'Attacking USV' 를 중 라벨로 **한 번만** 배치한다. 좌측 방어정에서 지시선을
빼 'Defending USV' 를 중 라벨로 **한 번만** 배치한다. 좌측 그물 부표열 바깥에 'Capture net' 을
중 라벨로 **한 번만** 배치한다. 아직 열린 회랑의 반투명 부채꼴 안쪽 여백에 'Approach corridor' 를
소 라벨로 배치한다. 정지한 적 선체 옆에 'Captured' 를 소 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 속도 우열, 거리환의 의미, 회랑의 개수는 전부 기하와
항적 거품 길이로만 표현하고 글로 적지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: centred_radial_aerial_scene
- print_target: 학술지 단단, 80 mm

### Background Treatment
- base: 위성 정사영상 품질 실사 해면
- water: 미세 파문 + 스웰 밴딩 + 깊이에 따른 자연 색변화
- coast: 좌상단 실제 형상 해안선 + 암초 여울 + 파도 백사
- lighting: 좌상단 고각 단일 태양, 모든 그림자 방향 일치
- texture: 사진 고유 질감만. 도트 그리드·패턴·노이즈 오버레이 금지

### Color Palette
- primary: #2C3E50 (네이비) — 오버레이 주선, 라벨 텍스트, 지시선
- secondary: #5D6D7E (슬레이트 블루) — 적 파선 항적, 거리환, 코리도 부채꼴 테두리
- accent: #2980B9 (딥 블루) — 아군 실선 항적, 그물 오버레이 세그먼트, 요격 환형 워시
- background: #F8F9FA (오프 화이트) — 도판 여백
- 확장: sea #8FA8BC→#5B7C95 (수면) · land #E3DCC9 / 해안선 #B5AC93 · threat #B5493C (적 오버레이) ·
  float #E8862A (그물 부표 실물색) · white #FFFFFF (항적 거품, 라벨 헤일로)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 3단계: 요소명(중), 상태·부기(소). Bold 는 사용하지 않는다
- 실사 위 라벨은 얇은 흰 헤일로를 두어 판독되게 한다
- 라벨은 대상에서 떼어 놓고 얇은 지시선으로 연결한다

## CONTENT
label_mothership: "Mothership"
label_enemy: "Attacking USV"
label_ally: "Defending USV"
label_net: "Capture net"
label_corridor: "Approach corridor"
label_captured: "Captured"

## FORBIDDEN ELEMENTS
- 배를 아이콘으로 대체: 속 빈 삼각형, 쉐브론, 픽토그램, 플랫 실루엣, 클립아트 보트
- 플랫 벡터 바다, 단색 배경 해면, 만화풍 파도, 어린이책 일러스트 톤
- 이미지 플레이스홀더: [Image 1], [사진], [이미지], [아이콘]
- 위치 지시자: [상단], [하단], [좌측], [우측]
- 이미지 내부 캡션: "Fig. 1", "Figure 1", "그림 1", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장·설명문
- 범례 박스, 색상 키, 축 라벨, 좌표 눈금, 축척 바, 방위 장미
- 수평선, 하늘, 구름, 저각 시점, 3인칭 시네마틱 앵글
- 45도 아이소메트릭 3D 오브젝트, 매거진 헤드라인, 번호 배지
- 네온, 글로우, 렌즈 플레어, 홀로그램, 파티클, 빛나는 HUD 프레임
- 어두운 SF 배경, 레이더 스코프 스윕, 타깃 락온 브래킷
- 폭발, 화염, 연기, 피격 효과 — 이 도판은 포획이지 파괴가 아니다
- 기관 로고, 국적 표식, 함번, 워터마크
- 한영 병기 표현: 그물 (Net), 모선 / Mothership
- 원문에 없는 정량 수치(포획률, 속력 값, 거리 값) 표기
