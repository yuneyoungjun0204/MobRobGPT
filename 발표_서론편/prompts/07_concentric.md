## INSTRUCTION

### Image Purpose
사내 연구 세미나 발표에서 모델 입력의 구성과 정규화 설계를 설명하는 본문 슬라이드로 사용한다. 자선과 타선 채널이 얼마나 단출한지, 그리고 소프트사인 정규화가 왜 근거리에 해상도를 몰아주는지를 거리별 동심원 구조로 전달하는 도해를 산출한다.

### Target Audience
조선소 설계·운항 실무진, 사내 연구소 자율운항 연구원, 딥러닝 비전공 관리자. 동심원의 간격 변화만 보고도 "가까울수록 민감하고 멀수록 완만하다"를 수식 없이 이해할 수 있어야 하며, 채널 구성이 짧은 명사구로 읽히는 수준의 가독성을 기대한다.

### Key Message
자선 네 채널과 타선 두 채널이라는 단출한 입력을 소프트사인으로 정규화해 회피 판단에 필요한 해상도를 근거리에 집중시킨다.

### Scene Description
An off-white surface fills the entire 3840x2160 frame with a large editorial headline mass anchoring the upper-left corner and a concentric ring structure occupying the right two-thirds of the canvas, its centre pushed right of the canvas midline to create asymmetry. At the exact centre floats one large forty-five degree isometric photorealistic three-dimensional own-ship hull in brushed steel with a small heading vane on its bow decomposed into two orthogonal component arms, and a slender translucent glass vector arrow points from the hull toward a distant goal marker. Surrounding the hull, four concentric rings are drawn as flat two-dimensional bands lying on the ground plane, and their radial spacing is deliberately non-uniform, tight and closely packed near the hull then progressively wider and more relaxed toward the outer edge so that the sensitivity gradient is legible as pure geometry without any plotted curve. Small isometric ship markers sit on the rings, the ones near the hull rendered crisp and highly detailed while those on the outer rings become simplified and flatter, visually encoding that distant vessels saturate. The outermost ring carries a soft accent-coloured boundary and a compact isometric object of a rotating radar scanner sits just inside it, while two small crossed-out ghost objects, a faint position pin and a faint speed gauge, float outside the outer ring rendered in low-contrast grey with a thin diagonal strike to show what is excluded from the input. Rings sit on twelve-pixel soft-rounded section fields where labels are attached, connectors are limited to four short two-pixel secondary radial ticks with open triangular arrowheads pointing outward from the hull, and ornament is limited to accent-coloured threshold pips on two specific rings plus one thin accent rule beneath the headline. The central hull is rendered twenty percent larger than its nominal zone and the outer ring is stretched to touch the top and bottom safe margins so negative space stays at or below fifteen percent. No watermarks, no blurry text, no placeholder brackets, no duplicated labels, no random artifacts, no pedestals or stone bases under the three-dimensional objects, no figure caption numbers, no colour legend boxes, no axis labels, no data tables. CRITICAL: Only render the exact text strings listed in the CONTENT block below. Do NOT generate, infer, or add any additional Korean text beyond what is explicitly written in CONTENT. If a box or area has no CONTENT text assigned, fill it with icons or illustrations — never with AI-generated Korean sentences.

### Rendering Style
- 서피스: 45도 아이소메트릭 포토리얼리스틱 3D 오브젝트. 중심은 선수 방향 베인이 두 개의 직교 성분 팔로 분해된 브러시드 스틸 자선 선체, 링 위에는 거리에 따라 디테일이 감소하는 소형 타선 마커, 외곽에는 회전 레이더 스캐너. 제외 요소는 옅은 회색의 위치 핀과 속도 게이지 고스트로 표현하고 얇은 사선으로 지운다. 받침대 없이 부유시키고 접촉 그림자만 남기며 소프트 스튜디오 라이팅을 적용한다.
- 배경: 오프 화이트 단색 배경. 링 자체 외에는 어떤 격자나 눈금선도 넣지 않고 3D 오브젝트 하단 미세 앰비언트 그림자만 허용한다.
- 코너/경계: 네 개 동심 링은 지면 평면에 놓인 2D 밴드로 그리고, 각 링의 라벨이 붙는 지점에만 12픽셀 소프트 라운딩 섹션 필드를 1픽셀 보조색 헤어라인으로 얹는다. 최외곽 링만 강조색 경계로 처리해 수집 반경의 끝임을 표시한다.
- 연결선: 자선에서 바깥으로 향하는 2픽셀 보조색 짧은 방사 틱 4개만 배치하고 끝에 열린 삼각형 화살촉을 둔다. 링 사이를 잇는 추가 연결선은 넣지 않아 구조를 단순하게 유지한다.
- 시각장식: 두 개의 특정 링 위에 강조색 임계 핍을 하나씩 배치해 기준값 지점을 표시하고 헤드라인 아래 강조색 얇은 룰 한 줄을 넣는다. 링 간격을 안쪽은 촘촘하게 바깥쪽은 넓게 배분하는 것 자체가 핵심 장식이므로 다른 장식은 추가하지 않는다. 색상 범례 박스, 축 라벨, 데이터 테이블, 그래프 곡선은 넣지 않는다.
- 공간구성: 매거진 에디토리얼 레이아웃. 좌상단 대형 헤드라인이 앵커이고 동심원 중심을 캔버스 중앙보다 오른쪽으로 밀어 좌측에 텍스트 열, 우측에 도형 덩어리를 두는 비대칭 구도를 만든다. 자선 채널 항목은 중심 오브젝트 주변을 감싸기 패턴으로 흐르고, 링 라벨은 각 링 바깥쪽에 붙는다. 중심 오브젝트 20퍼센트 확대와 최외곽 링의 상하 세이프 마진 접촉으로 네거티브 스페이스를 15퍼센트 이하로 억제한다.
- 시각메타포: 에디토리얼 매거진 × 아이소메트릭 3D. 링 간격의 물리적 조밀도가 곧 정규화 해상도라는 등가를 만들어, 함수 그래프 없이도 소프트사인의 설계 의도를 사물의 배치만으로 전달한다. 지워진 위치 핀과 속도 게이지가 "무엇을 일부러 넣지 않았는가"를 조용히 말해준다.

### Content Placement
좌상단에 '단출한 입력과 소프트사인'을 캔버스에서 가장 큰 매거진 헤드라인으로 배치하고 그 아래 강조색 얇은 룰을 지나 '근거리에 해상도를 배분'을 보조색 중간 크기 문장으로 붙인다. 중심 자선 오브젝트 왼편에 '자선 4채널'을 소제목 크기로 배치하고 그 아래에 '선수각 사인 코사인'과 '목표 상대변위 2값'을 세로로 나열해 선체 윤곽을 따라 감싸기 패턴을 이루게 한다. 첫 번째 링 바깥쪽에 '타선 2채널'을 소제목 크기로 배치하고 그 아래에 '자선 기준 상대변위'와 '거리순 상위 10척'을 세로로 나열한다. 두 번째 링 바깥쪽에 '수집 반경 10km'를 배치하고 그 아래에 '관측 창 200스텝'을 한 단계 작은 크기로 배치한다. 지워진 고스트 오브젝트 두 개 옆에 각각 '절대 위치 미포함'과 '속도 미포함'을 옅은 보조색 작은 문구로 배치한다. 화면 좌하단 텍스트 열에 '소프트사인 정규화'를 소제목 크기로 배치하고 그 아래 강조색 칩 두 개 안에 '타선 기준 600m'와 '목표 기준 10km'를 나란히 배치하며, 두 칩 아래에 '기준값에서 0.5 출력'을 한 단계 작은 크기로 붙인다. 두 강조색 칩은 각각 해당하는 링의 임계 핍과 같은 색으로 처리해 숫자와 링이 눈으로 연결되게 한다. 화면 최하단 우측에 '먼 배일수록 완만한 포화'를 가장 작은 캡션 크기로 대비를 낮춰 배치한다.

## CONFIGURATION

### Canvas Settings
- resolution: 3840x2160
- aspect_ratio: 16:9
- layout: concentric_sensitivity_rings

### Background Treatment
- base: 오프 화이트 단색 평면
- ambient_shadow: 3D 오브젝트 하단 미세 접촉 그림자만 허용
- texture: 없음. 도트 그리드, 격자, 눈금선, 그래프 곡선 금지
- ornament: 헤드라인 하단 강조색 얇은 룰 1개, 특정 링 위 강조색 임계 핍 2개, 최외곽 링 강조색 경계

### Color Palette
- primary: #2C3E50 (네이비) - 헤드라인, 채널 소제목, 자선 선체의 어두운 금속 톤
- secondary: #5D6D7E (슬레이트 블루) - 채널 항목 문구, 링 밴드, 방사 틱, 제외 요소 고스트
- accent: #2980B9 (딥 블루) - 기준값 칩, 링 임계 핍, 최외곽 링 경계, 목표 벡터 화살표
- background: #F8F9FA (오프 화이트) - 전체 배경면

### Typography
All Korean text must be rendered with crisp, perfectly formed characters using heavy-weight Gothic-style sans-serif fonts. Each Korean syllable block must be complete and legible. Use Bold weight (700+) for titles, Medium weight (500) for body text.
- hierarchy: 헤드라인 > 채널 소제목 > 기준값 칩 문구 > 항목 본문 > 제외 표기와 캡션 위계
- contrast: 기준값 칩 문구는 항목 본문보다 크게 두어 숫자가 링과 함께 먼저 읽히게 한다
- alignment: 좌측 텍스트 열은 왼쪽 정렬, 링 라벨은 각 링 바깥 방향 정렬
- spacing: 채널 소제목과 항목 사이는 좁게, 서로 다른 채널군 사이는 넓게 두어 그룹을 구분

## CONTENT
title: "단출한 입력과 소프트사인"
lead_message: "근거리에 해상도를 배분"
core_label: "자선 4채널"
core_item1: "선수각 사인 코사인"
core_item2: "목표 상대변위 2값"
ring1_label: "타선 2채널"
ring1_item1: "자선 기준 상대변위"
ring1_item2: "거리순 상위 10척"
ring2_label: "수집 반경 10km"
ring2_item1: "관측 창 200스텝"
exclusion_note1: "절대 위치 미포함"
exclusion_note2: "속도 미포함"
softsign_label: "소프트사인 정규화"
softsign_metric1: "타선 기준 600m"
softsign_metric2: "목표 기준 10km"
softsign_note: "기준값에서 0.5 출력"
footnote: "먼 배일수록 완만한 포화"

## FORBIDDEN ELEMENTS
- 이미지 플레이스홀더: [Image 1], [Image 2], [사진], [이미지], [아이콘]
- 위치 지시자: [상단], [하단], [좌측], [우측]
- XML 태그 표기: scene, text_to_render, typography, canvas, layout
- 폰트 패밀리명 직접 지정: Noto Sans, Pretendard, Nanum Gothic, Apple SD Gothic Neo
- 색상 코드 노출 텍스트: #2C3E50, #5D6D7E, #2980B9, #F8F9FA
- 크기 단위 텍스트: 24pt, 32px, 18pt
- ASCII 레이아웃 힌트: +---+, |---|, ->
- 역할 라벨 텍스트: Main Title, 핵심 모듈명, 보조 지표, KPI 영역
- 플레이스홀더 문자열: [내용], {텍스트}, {{value}}
- 메타데이터 헤더 텍스트: 영역, 역할, 구성, 타입, 채널
- 한영 병기 표현: 관측 (Observation), 정규화 / Normalization, 자선(Own Ship)
- 렌더링 힌트 텍스트: (굵게), (강조), (이탤릭), (밑줄)
- 기관 로고, 조선소 마크, 국가 상징, 사명 워드마크
- 로렘 입숨, 의미 없는 더미 텍스트, 알아볼 수 없는 문자열
- CONTENT 내부 번호 목록, 표 형식, subsection 헤더
- 받침대와 플랫폼: 콘크리트 베이스, 스톤 플랫폼, 사각 받침대, 지층 단면
- 학술 논문 요소: Figure 1., 그림 1., 색상 범례 박스, X축, Y축, 축척 바, 함수 그래프 곡선
- 수식 표기: 분수, 시그마, 절댓값 기호, 함수 정의식
- AI가 자체 생성한 한글 설명문: CONTENT 블록에 명시되지 않은 어떤 한글 텍스트도 이미지 내부에 렌더링하는 것을 절대 금지
