# Fig. 9 — Group-Relative Training with Per-Vessel Counterfactual Credit

- 파일명: `fig9_grpo.png` · LaTeX label `fig:grpo`
- 배치: 단단 `figure[H]`, `width=80mm` · 캔버스 **2400 x 1800 (4:3)**
- 레지스터: **A (롤아웃 실사) + B (차트·행렬)** · 렌더링 문자열 **8개**

> 근거: 초안 §2.3 — 후보 행동 다수 표집 → 시뮬레이터 일정 구간 진행 → 팀 보상 →
> 후보 간 상대 비교로 이득 산정(가치망 없음) → 한 척만 바꾸고 나머지 고정하여 개별 기여 분리.
> 후보 하나는 휴리스틱 행동으로 채운다.

---

## INSTRUCTION

### Image Purpose
학습 방식 도판. 한 결정 시점에서 여러 후보 행동을 뽑아 각각을 실제로 진행시키고, 후보들끼리의
상대 비교만으로 이득을 산정하므로 별도의 가치망이 필요 없다는 것, 그리고 한 척의 행동만 바꾸고
나머지를 고정해 평가함으로써 팀 보상에 묻힌 개별 기여를 분리한다는 것을 한 장에서 잇는다.

### Target Audience
조선해양공학 학술지 심사자와 독자. 강화학습 전공이 아니어도 "후보를 여러 개 굴려 서로
비교한다"와 "한 척만 바꿔 차이를 그 배 몫으로 돌린다"를 그림에서 직접 읽어야 한다.

### Key Message
후보 행동들을 실제로 진행시켜 서로 비교하고, 한 척만 바꾼 차이를 그 배의 기여로 돌린다.

### Scene Description
An off-white canvas organised as three stacked bands flowing downward, with no outer frame and no
background panel. The top band holds a single decision state at the left: a square tile rendered as a
genuine viridis-colormapped score map with visible cell structure and a sector-shaped bright band. From
its right side a fan of four arrows with filled triangular arrowheads spreads rightward and apart, each
terminating at one of four small square candidate tiles arranged in a vertical column. Each candidate
tile repeats the same score map but with a different pair of cells marked by heavy accent square
outlines, so the four candidates are visibly four different waypoint pairs on one common map. The
lowest of the four tiles is distinguished by a dashed accent border rather than a solid one. The middle
band holds four photographic rollout panels in a horizontal row, one per candidate, each a small
near-vertical aerial view of the same sea area rendered at satellite orthophoto fidelity with genuine
wave texture. Every panel contains the identical photorealistically rendered grey frigate at centre,
the identical three small rigid-hull attack craft with white bow wakes converging on it, and the
identical three light-hulled defending unmanned vessels in the same starting positions, so the panels
are directly comparable. What differs between panels is only where the net barriers were laid, each
rendered as a line of small orange surface floats with dark mesh below the waterline, and the set of
thin dashed white trajectory traces recording where the vessels went during the window. In two panels
an attack craft is stopped dead in the water behind a barrier with its bow wake collapsed into a flat
disturbed patch; in the others one craft has slipped past and continues toward the frigate with an
intact wake. A thin horizontal time bracket with end ticks spans beneath the row of panels, indicating
that all four were advanced over the same interval. Directly under each panel a single vertical bar
rises from a common baseline, the four bars at visibly different heights, forming a small bar chart.
The bottom band holds two data objects side by side. On the left, a second small bar chart shares the
same four-bar arrangement but is drawn about a horizontal zero line, with two bars extending above it
and two below, each bar with a thin dark edge and a pale accent fill, and a thin dashed line marks the
group mean at the zero level; a short curved arrow leads from the first bar chart into this one. On the
right sits a genuine matrix rendering: tall square brackets enclosing a grid of three rows by four
columns, the cells separated by faint rules, every cell containing a small angular vessel marker. In
the second row every marker is drawn in the accent tone and each is oriented differently from the
others; in the first and third rows every marker is drawn in a flat neutral tone and all are oriented
identically across the row. The second row is further set off by a light tint band running behind it
and by a small bracket at the row's left edge. No watermarks, no blurry text, no placeholder brackets,
no random artifacts, no axis tick numbers, no reward values, no learning curves, no epoch axes, no
figure caption numbers, no legend boxes. CRITICAL: render only the eight exact English strings listed
in the CONTENT block below and nothing else. Do NOT invent, infer or add any further text or numeric
values. Never render Korean, Chinese, Japanese or Cyrillic characters anywhere in the image.

### Rendering Style
- 서피스: 롤아웃 패널은 **실사** (위성 정사영상 품질 해면 + 사진 수준 탑다운 선박 렌더).
  네 패널은 **같은 해역·같은 초기 배치**여야 하고 그물 위치와 항적만 달라야 한다.
  그래야 "같은 상태에서 갈라진 후보"라는 주장이 성립한다. 배를 아이콘으로 대체하면 즉시 실패.
- 배경: 오프 화이트. 차트와 행렬 영역은 사진을 쓰지 않는다.
- 코너/경계: 직각. 타일·패널·행렬 셀 모두.
- 연결선: 후보 분기는 부채꼴 화살표, 차트 간 이동은 짧은 곡선 화살표. 전부 채워진 삼각 화살촉.
- 시각장식: 막대 차트는 **진짜 차트**다 — 공통 기준선, 서로 다른 높이, 이득 차트는 0선 위아래로
  갈리고 그룹 평균이 파선으로 표시된다. **축 눈금 숫자는 넣지 않는다**(값을 창작하지 않기 위해).
  반사실 행렬은 **진짜 행렬**이며, 한 행만 색과 방향이 다르다는 사실이 "한 척만 바꾼다"를
  형태로 못박는다.
- 공간구성: 상 후보분기 / 중 롤아웃 4연폭 / 하 차트+행렬의 3단 하향 흐름. 여백 15~25%.
- 시각메타포: 같은 무대에서 네 갈래로 갈라진 미래를 나란히 놓고 높이로 비교한다.
  마지막 행렬의 한 줄만 다른 색인 것이 개별 기여 분리의 은유다.

### Content Placement
상단 좌측 타일 아래에 'Decision state' 를 중 라벨로 배치한다. 네 후보 타일 열 오른쪽 바깥에
'Candidate actions' 를 중 라벨로 배치하고, 파선 테두리 후보 옆에 'Heuristic seed' 를 소 라벨로
배치한다. 중간 띠 시간 브래킷 아래에 'Rollout window' 를 중 라벨로 배치한다. 첫 막대 차트 왼쪽
바깥에 'Team reward' 를 중 라벨로 세로로 배치한다. 하단 좌측 0선 차트 왼쪽 바깥에
'Group-relative advantage' 를 중 라벨로 세로로 배치한다. 하단 우측 행렬 아래에 'Counterfactual'
을 중 라벨로 배치하고, 강조된 둘째 행 왼쪽 브래킷 옆에 'perturbed' 를 소 라벨로, 회색 첫째 행
왼쪽에 'fixed' 를 소 라벨로 배치한다.
그 밖의 어떤 문자열도 렌더링하지 않는다. 그룹 평균은 파선 위치로만, 후보 수와 배 수는 격자
차원으로만 표현하고 글이나 숫자로 적지 않는다.

## CONFIGURATION

### Canvas Settings
- resolution: 2400 x 1800
- aspect_ratio: 4:3
- layout: three_band_downward_flow
- print_target: 학술지 단단, 80 mm

### Background Treatment
- base: 오프 화이트 단색
- rollout_panels: 위성 정사영상 품질 실사 해면 4폭 (해역·조명 동일)
- texture: 사진 고유 질감만
- ornament: 없음

### Color Palette
- primary: #2C3E50 (네이비) — 타일 외곽, 화살표, 행렬 대괄호, 막대 테두리, 라벨
- secondary: #5D6D7E (슬레이트 블루) — 시간 브래킷, 셀 격자, 평균 파선, 고정 행 마커, 지시선
- accent: #2980B9 (딥 블루) — 선택 칸 테두리, 씨앗 파선 테두리, 막대 채움, 섭동 행 마커, 틴트 띠
- background: #F8F9FA (오프 화이트) — 전체 배경
- 확장: sea #8FA8BC→#5B7C95 · threat #B5493C (적 오버레이) · float #E8862A (그물 부표) ·
  colormap **viridis** (점수맵 타일) · white #FFFFFF (항적 파선, 라벨 헤일로)

### Typography
- 산세리프 단일 패밀리
- 크기 2단계: 구역명(중), 상태·부기(소)
- 차트 축 의미 라벨만 90도 회전 배치, 나머지 라벨은 전부 수평
- 축 눈금 숫자를 넣지 않는다 — 값을 창작하지 않기 위한 의도적 선택이다

## CONTENT
stage_state: "Decision state"
stage_cand: "Candidate actions"
note_seed: "Heuristic seed"
stage_rollout: "Rollout window"
axis_reward: "Team reward"
axis_adv: "Group-relative advantage"
stage_cf: "Counterfactual"
note_perturbed: "perturbed"
note_fixed: "fixed"

## FORBIDDEN ELEMENTS
- 롤아웃 패널을 아이콘·일러스트로 대체
- 네 롤아웃 패널의 해역·초기 배치가 서로 달라지는 것 (그물·항적만 달라야 한다)
- 축 눈금 숫자, 보상 값, 이득 값 등 **창작된 수치** 표기
- 학습 곡선, 손실 그래프, 에폭 축 — 이 도판은 결과 그래프가 아니다
- 수식, 그리스 문자, 기댓값 기호, 표준편차 기호
- 이미지 플레이스홀더: [Image 1], [사진], [아이콘]
- 이미지 내부 캡션: "Fig. 9", "Figure 9", 도판 제목 텍스트
- 한글·한자·가나·키릴 문자
- CONTENT 에 없는 임의 생성 영문 문장
- 무지개(jet) 컬러맵, 다중 컬러맵 혼용
- 45도 아이소메트릭 3D, 번호 배지, 매거진 헤드라인
- 네온, 글로우, 드롭섀도, 어두운 배경, HUD
- 뇌 모양, 뉴런 그물망, 회로기판, 로봇
- 폭발, 화염, 연기 — 포획이지 파괴가 아니다
- 기관 로고, 워터마크
