## INSTRUCTION

### Image Purpose
사내 연구 세미나 발표에서 학습 라벨이 어떻게 만들어지는지를 설명하는 본문 슬라이드로 사용한다. 원본 항적에서 시간 오프셋을 거쳐 정답 경유점과 정답 곡률이 자동으로 산출되는 순차 처리 과정을 좌에서 우로 흐르는 파이프라인 도해로 산출한다.

### Target Audience
조선소 설계·운항 실무진, 사내 연구소 자율운항 연구원, 딥러닝 비전공 관리자. 화살표를 따라가는 것만으로 "사람이 라벨을 달지 않는다"는 결론에 도달할 수 있어야 하며, 각 단계가 한 줄 명사구로 요약되는 수준의 가독성을 기대한다.

### Key Message
별도 주석 작업 없이 항적 자체에서 정답 경유점과 정답 곡률이 자동으로 산출된다.

### Scene Description
An off-white surface fills the entire 3840x2160 frame with a full-width editorial headline band across the top and a left-to-right processing pipeline of four stage nodes occupying the central and lower canvas. The first node holds a large forty-five degree isometric photorealistic three-dimensional sea-surface slab on which a single recorded vessel track is embossed as a continuous curved ribbon with a small brushed-steel hull sitting at its present position. The second node holds an isometric mechanical time-offset caliper, a brushed-metal sliding gauge that grips the track ribbon at two points and lifts a translucent glass marker from the future position ahead of the hull, its jaws clearly spanning a forward stretch of the ribbon. The third node holds an isometric waypoint pin of polished metal planted on a small extracted patch of the sea surface with a straight glass chord line running back to the hull position. The fourth node holds an isometric curvature gauge shaped as a circular arc template of brushed metal laid over the track ribbon, with a small twin-arrow dial beside it whose two opposing arrows are rendered as a mirrored pair to express the sign convention physically. All nodes sit inside twelve-pixel soft-rounded section fields bounded by one-pixel secondary hairlines, and three two-pixel secondary connectors with open triangular arrowheads run strictly left to right between them so the flow direction is unambiguous across the whole canvas. Ornament is limited to accent-coloured numeric badges at the upper-left corner of each node, one accent rule beneath the headline band, and an accent tint on the extracted future marker so the eye tracks the same element through all four stages. The third node is rendered as the single enlarged emphasis stage at twenty percent larger than its neighbours while the other objects are fifteen percent larger than their nominal zones, so the pipeline presses against the left and right safe margins and negative space stays at or below fifteen percent. No watermarks, no blurry text, no placeholder brackets, no duplicated labels, no random artifacts, no pedestals or stone bases under the three-dimensional objects, no figure caption numbers, no colour legend boxes, no axis labels, no data tables. CRITICAL: Only render the exact text strings listed in the CONTENT block below. Do NOT generate, infer, or add any additional Korean text beyond what is explicitly written in CONTENT. If a box or area has no CONTENT text assigned, fill it with icons or illustrations — never with AI-generated Korean sentences.

### Rendering Style
- 서피스: 45도 아이소메트릭 포토리얼리스틱 3D 오브젝트. 1단계는 항적 리본이 양각된 해수면 슬래브와 브러시드 스틸 선체, 2단계는 리본을 물고 미래 위치 마커를 들어올리는 금속 시간 오프셋 캘리퍼, 3단계는 유리 현 라인이 연결된 광택 금속 경유점 핀, 4단계는 리본 위에 얹힌 원호 템플릿과 좌우 대칭 쌍화살 다이얼. 받침대 없이 부유시키고 접촉 그림자만 남기며 소프트 스튜디오 라이팅과 앰비언트 오클루전을 적용한다.
- 배경: 오프 화이트 단색 배경. 도트 그리드와 패턴 장식을 넣지 않고 3D 오브젝트 하단 미세 앰비언트 그림자만 허용한다.
- 코너/경계: 네 개 단계 노드를 12픽셀 소프트 라운딩 섹션 필드로 구획하고 1픽셀 보조색 헤어라인으로 경계를 잡는다. 강조 단계인 3단계 노드만 테두리를 강조색으로 처리해 핵심 변환 지점을 경계선으로 표시한다.
- 연결선: 노드 사이를 잇는 2픽셀 보조색 실선 화살표 3개를 좌에서 우로 일관되게 배치하고 끝에 열린 삼각형 화살촉을 둔다. 역방향 화살표나 되돌아오는 루프는 절대 넣지 않아 순차 처리임을 분명히 한다.
- 시각장식: 각 노드 좌상단에 강조색 사각 번호 배지를 동일 크기로 배치하고 헤드라인 밴드 아래 강조색 얇은 룰 한 줄을 넣는다. 2단계에서 추출된 미래 위치 마커에만 강조색 틴트를 입혀 그 요소가 3단계와 4단계까지 이어지는 것을 눈으로 추적할 수 있게 한다. 색상 범례 박스, 축 라벨, 데이터 테이블은 넣지 않는다.
- 공간구성: 매거진 에디토리얼 레이아웃. 상단 풀와이드 헤드라인 밴드가 앵커이고 그 아래 네 개 노드가 좌우로 펼쳐진다. 각 노드는 3D 오브젝트가 상단, 텍스트가 하단인 앵커 패턴을 쓰되 3단계 노드만 오브젝트를 키우고 텍스트를 옆으로 붙여 격자를 깬다. 파이프라인을 좌우 세이프 마진까지 늘리고 오브젝트를 15-20퍼센트 확대해 네거티브 스페이스를 15퍼센트 이하로 억제한다.
- 시각메타포: 에디토리얼 매거진 × 아이소메트릭 3D. 캘리퍼가 항적을 물고 미래 한 점을 집어 올리는 기계적 동작이 라벨 자동 산출을 사물화한다. 사람 손이나 주석 도구가 화면 어디에도 등장하지 않는다는 사실 자체가 "사람 라벨링 불필요"를 시각적으로 증명한다.

### Content Placement
상단 밴드에 '항적 자체가 정답'을 풀와이드 대형 헤드라인으로 펼치고 그 아래 강조색 얇은 룰을 지나 '별도 주석 작업 없음'을 보조색 중간 크기 문장으로 붙인다. 첫째 노드 좌상단 배지 안에 '01'을 흰 숫자로 배치하고, 해수면 슬래브 오브젝트 아래에 '원본 항적'을 소제목 크기로, 그 아래 한 단계 작은 크기로 '실제 조선 궤적'을 배치한다. 둘째 노드 배지 안에 '02'를 배치하고, 캘리퍼 오브젝트 아래에 '300초 오프셋'을 소제목 크기로, 그 아래에 '미래 시점 추출'을 배치한다. 셋째 노드 배지 안에 '03'을 배치하고, 확대된 경유점 핀 오브젝트 오른편에 '정답 경유점'을 이 슬라이드에서 가장 큰 소제목으로 배치한 뒤 그 아래에 '1.5km 이상 전방'을 강조색으로 배치한다. 넷째 노드 배지 안에 '04'를 배치하고, 원호 템플릿 오브젝트 아래에 '정답 곡률'을 소제목 크기로, 그 아래에 '선수각 변화 나누기 거리'를 배치한다. 넷째 노드의 쌍화살 다이얼 좌우에 각각 '플러스는 우선회'와 '마이너스는 좌선회'를 작은 크기로 나란히 배치해 화살표 방향과 문구가 물리적으로 대응하게 한다. 화면 최하단 우측에 '사람 라벨링 불필요'를 가장 작은 캡션 크기로 대비를 낮춰 배치한다.

## CONFIGURATION

### Canvas Settings
- resolution: 3840x2160
- aspect_ratio: 16:9
- layout: flow_four_stage_pipeline

### Background Treatment
- base: 오프 화이트 단색 평면
- ambient_shadow: 3D 오브젝트 하단 미세 접촉 그림자만 허용
- texture: 없음. 도트 그리드, 패턴, 노이즈 오버레이 금지
- ornament: 헤드라인 밴드 하단 강조색 얇은 룰 1개, 노드별 강조색 사각 번호 배지 4개, 3단계 노드 강조색 테두리

### Color Palette
- primary: #2C3E50 (네이비) - 헤드라인, 단계 소제목, 3D 오브젝트의 어두운 금속 톤
- secondary: #5D6D7E (슬레이트 블루) - 단계 설명 문구, 섹션 헤어라인, 좌우 흐름 화살표
- accent: #2980B9 (딥 블루) - 번호 배지, 추출된 미래 위치 마커, 3단계 테두리, 부호 규약 화살표
- background: #F8F9FA (오프 화이트) - 전체 배경면

### Typography
All Korean text must be rendered with crisp, perfectly formed characters using heavy-weight Gothic-style sans-serif fonts. Each Korean syllable block must be complete and legible. Use Bold weight (700+) for titles, Medium weight (500) for body text.
- hierarchy: 헤드라인 > 단계 소제목 > 단계 설명 본문 > 부호 규약 표기와 캡션 위계
- contrast: 3단계 소제목만 다른 단계보다 한 단계 크게 키워 핵심 변환 지점을 타이포로도 강조한다
- alignment: 각 노드 내부는 왼쪽 정렬로 통일하고 부호 규약 두 문구만 다이얼 좌우 대칭 정렬
- spacing: 노드 간 간격을 균일하게 유지하고 소제목과 설명 사이는 좁은 줄간격으로 묶는다

## CONTENT
title: "항적 자체가 정답"
lead_message: "별도 주석 작업 없음"
step1_number: "01"
step1_label: "원본 항적"
step1_desc: "실제 조선 궤적"
step2_number: "02"
step2_label: "300초 오프셋"
step2_desc: "미래 시점 추출"
step3_number: "03"
step3_label: "정답 경유점"
step3_desc: "1.5km 이상 전방"
step4_number: "04"
step4_label: "정답 곡률"
step4_desc: "선수각 변화 나누기 거리"
sign_note1: "플러스는 우선회"
sign_note2: "마이너스는 좌선회"
footnote: "사람 라벨링 불필요"

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
- 메타데이터 헤더 텍스트: 영역, 역할, 구성, 타입, 단계
- 한영 병기 표현: 곡률 (Curvature), 경유점 / Waypoint, 라벨(Label)
- 렌더링 힌트 텍스트: (굵게), (강조), (이탤릭), (밑줄)
- 기관 로고, 조선소 마크, 국가 상징, 사명 워드마크
- 로렘 입숨, 의미 없는 더미 텍스트, 알아볼 수 없는 문자열
- CONTENT 내부 번호 목록, 표 형식, subsection 헤더
- 받침대와 플랫폼: 콘크리트 베이스, 스톤 플랫폼, 사각 받침대, 지층 단면
- 학술 논문 요소: Figure 1., 그림 1., 색상 범례 박스, X축, Y축, 축척 바
- 수식 표기: 분수 기호, 나눗셈 기호, 시그마, 델타 기호가 포함된 계산식
- AI가 자체 생성한 한글 설명문: CONTENT 블록에 명시되지 않은 어떤 한글 텍스트도 이미지 내부에 렌더링하는 것을 절대 금지
