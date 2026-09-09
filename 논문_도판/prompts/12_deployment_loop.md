# Fig. 12 — Real-World Deployment Loop

- 파일명: `fig12_deployment.png` · LaTeX label `fig:deployment`
- 배치: 단단 `figure[H]`, `width=80mm` · 캔버스 **2400 x 1800 (4:3)**
- 레지스터: **B (폐루프 블록도) + 실사 썸네일 2개** · 렌더링 문자열 **9개**

> 근거: `commander/ros2_unet_env.py::ROS2CnnEnv`, `commander/ros2_sensor_bridge.py`,
> `docs/unet_model_deploy.md` §10 (센서 → 입력 → WP tick 루프), §6.2 (픽셀 → 위경도).

---

## INSTRUCTION

### Image Purpose
실환경 적용 절의 도판. 학습된 정책이 시뮬레이터 밖에서 어떻게 도는지를 하나의 닫힌 루프로
보인다. 실선 센서가 상태를 만들고, 상태가 래스터로 바뀌어 정책에 들어가며, 정책이 낸 픽셀이
위경도 경유점으로 되돌아가 선박 제어로 내려가고, 그 결과가 다시 센서로 관측된다.
지휘관 호출은 이 루프 바깥에서 비동기로 붙는다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 루프가 닫혀 있다는 것과, 지휘관 가지가 루프를 막지
않는다는 것이 화살표 배치만으로 읽혀야 한다.

### Key Message
센서에서 경유점까지가 하나의 닫힌 주기로 돌고, 지휘관 호출은 그 바깥에 비동기로 붙는다.

### Scene Description
An off-white canvas holds a single closed cyclic block diagram arranged as a wide horizontal loop, with
no outer frame and no background panel. Five stages sit in sequence along the upper arc of the loop,
reading left to right, each drawn as a rectangle with a thin primary-tone border and a very pale
interior, joined by solid arrows with filled triangular arrowheads. The first stage is a photographic
near-vertical aerial thumbnail of real sea at satellite orthophoto fidelity showing a
photorealistically rendered grey frigate, two small rigid-hull attack craft with white bow wakes, and a
light-hulled defending unmanned vessel; this panel is real imagery, not a diagram, and three small
sensor glyphs sit along its lower edge, each a plain filled semicircle emitting two thin concentric
arcs, spaced evenly and drawn identically. The second stage holds a compact four-row table drawn with
genuine horizontal table rules and no vertical rules, each row carrying a small geometric pictogram at
the left and a neutral placeholder bar at the right that must remain illegible. The third stage holds a
flat offset stack of identical square raster tiles, each with a thin edge and a subtle oblique side
face, the frontmost rendered as a genuine viridis-colormapped raster with visible cell structure. The
fourth stage holds a compact symmetric encoder-decoder silhouette built from oblique tensor slab boxes
of decreasing then increasing size, with two dashed skip arrows arcing across its waist, and to its
immediate right within the same rectangle a small square heatmap tile carrying two cells marked by
heavy accent square outlines. The fifth stage holds a small plan-view panel showing a square grid with
two accent-tinted cells joined by a heavy accent segment, and directly beneath that grid the same two
points redrawn on a plain geodetic graticule of thin faint crossing lines, with a short double-headed
arrow between the two renderings, so the pixel-to-geodetic conversion is shown as a reversible mapping
rather than as a step. From the fifth stage a solid arrow descends and runs leftward along the lower arc
of the loop into a sixth stage placed beneath the first: a second photographic aerial thumbnail of the
same sea area, in which the defending unmanned vessel is now under way along a visible wake and a net
barrier of small orange surface floats with dark mesh below the waterline has been laid across the
attackers' path. From this sixth stage a solid arrow rises back into the first stage, closing the loop
unambiguously. Separately, a branch leaves the second stage upward into a detached rectangle sitting
clear above the main loop, drawn with an accent border and containing a schema box with tall square
brackets and three stacked field rows of unequal height whose value areas hold only illegible
placeholder bars and small paired markers; from this detached rectangle a dashed accent arrow descends
and enters the fourth stage from directly above. A small circular arrow glyph sits inside the loop's
open centre, indicating recurrence. No watermarks, no blurry text, no placeholder brackets, no random
artifacts, no readable body text inside any block, no timing values, no latency numbers, no figure
caption numbers, no legend boxes. CRITICAL: render only the nine exact English strings listed in the
CONTENT block below and nothing else. Do NOT invent, infer or add any further text or numeric values.
Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the image. Placeholder bars
must remain illegible and must never resolve into readable words, numbers, JSON, or code.

### Rendering Style
- 서피스: 각 블록은 **자기 내용물의 실물**을 담는다 — 실사 썸네일, 진짜 표 괘선, 오블리크 텐서
  슬래브, 진짜 viridis 히트맵, 격자와 그래티큘. 라벨만 있는 빈 상자는 실패다.
- 배경: 오프 화이트 단색. 루프 안쪽 중앙은 비워 두고 순환 기호 하나만 놓는다.
- 코너/경계: 직각. 전 블록의 테두리 굵기와 코너 처리가 같다.
- 연결선: 루프 본선은 주조색 실선 + 채워진 삼각 화살촉이며 **반드시 닫혀야** 한다.
  지휘관 가지만 강조색 **파선**이고 위에서 아래로 진입한다 — 실선/파선 구분이 동기와
  비동기를 가르는 유일한 장치다.
- 시각장식: 센서 글리프 3개(반원 + 동심호), 루프 중앙 순환 화살표 1개, 픽셀↔지리 양방향 화살표 1개.
  그 외 장식 없음. 번호 배지 금지.
- 공간구성: 상단 5단 + 하단 복귀 경로의 가로 폐루프. 지휘관 블록은 루프 위쪽에 **떨어뜨려**
  놓아 루프 바깥임이 위치로 읽히게 한다. 여백 15~25%.
- 시각메타포: 바다에서 시작해 바다로 돌아오는 닫힌 고리. 지휘관은 고리 위에 떠서 가끔
  내려꽂히는 가지로 그려져, 주기를 공유하지 않는다는 사실이 배치로 표현된다.

### Content Placement
루프 상단 다섯 블록 아래에 좌에서 우로 'Sensing', 'State', 'Observation', 'Policy', 'Waypoints'
를 중 라벨로 배치한다. 하단 복귀 블록 아래에 'Vessel control' 을 중 라벨로 배치한다.
루프 위에 떨어져 있는 강조색 블록 아래에 'Commander' 를 중 라벨로 배치하고, 그 블록에서
내려오는 파선 화살표 옆에 'asynchronous' 를 소 라벨로 배치한다. 다섯째 블록의 픽셀 격자와
지리 그래티큘 사이 양방향 화살표 옆에 'geodetic' 을 소 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 주기 길이, 지연 시간, 갱신율은 **어떤 숫자로도 넣지
않는다**. 비동기성은 파선과 블록 위치로만, 순환은 닫힌 화살표와 중앙 순환 기호로만 표현한다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: closed_loop_with_detached_branch
- print_target: 학술지 단단, 80 mm

### Background Treatment
- base: 오프 화이트 단색
- thumbnails: 위성 정사영상 품질 실사 해면 2폭 (같은 해역, 같은 조명, 전개 전/후)
- texture: 사진 고유 질감만
- ornament: 루프 중앙 순환 화살표 1개

### Color Palette
- primary: #2C3E50 (네이비) — 블록 테두리, 루프 본선 화살표, 표 괘선, 라벨
- secondary: #5D6D7E (슬레이트 블루) — skip 파선, 플레이스홀더 바, 그래티큘, 센서 동심호, 격자
- accent: #2980B9 (딥 블루) — 지휘관 블록 테두리와 파선 가지, 선택 셀, 그물 오버레이, 경유점
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: sea #8FA8BC→#5B7C95 · float #E8862A (그물 부표 실물색) · threat #B5493C (적 오버레이) ·
  tensor 면 #EAF2F8 · colormap **viridis** (래스터 타일)

### Typography
- 산세리프 단일 패밀리, 전 라벨 수평
- 크기 2단계: 단계명(중), 성질 표기(소)
- 'asynchronous' 와 'geodetic' 은 소문자로 두어 성질 표기임을 형태로 드러낸다
- 단계명은 블록 바로 아래 중앙 정렬로 통일

## CONTENT
stage_1: "Sensing"
stage_2: "State"
stage_3: "Observation"
stage_4: "Policy"
stage_5: "Waypoints"
stage_6: "Vessel control"
branch: "Commander"
branch_note: "asynchronous"
convert_note: "geodetic"

## FORBIDDEN ELEMENTS
- **빈 블록**: 내용물 없이 라벨만 있는 상자
- 루프가 닫히지 않은 것 (반드시 복귀 화살표로 닫혀야 한다)
- 지휘관 가지를 실선으로 그리거나 루프 본선 안에 배치하는 것
- 주기 길이, 지연 ms, 갱신율 Hz 등 **정량 시간 수치**
- 판독 가능한 가짜 본문: 읽히는 JSON, 코드, 로그, 프롬프트 문장
- 실제 위경도 값, 좌표 숫자
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 이미지 내부 캡션: "Fig. 12", "Figure 12", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 위성 아이콘, 안테나 픽토그램, 무선 신호 물결, 클라우드 아이콘, 서버 랙 그림
- 45도 아이소메트릭 3D, 번호 배지, 매거진 헤드라인
- 네온, 글로우, 드롭섀도, 어두운 배경, HUD, 대시보드 UI
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇
- 기관 로고, 미들웨어·프레임워크 로고, 워터마크
