## INSTRUCTION

### Image Purpose
사내 연구 세미나 발표에서 모델 구현 구조를 설명하는 본문 슬라이드로 사용한다. 청중 절반이 딥러닝 비전공자이므로 텍스트 비중을 의도적으로 낮추고, 상단의 백본 공정 비교 개요와 하단의 주변선박 인코더 처리 흐름을 아이콘과 도식 중심으로 전달하는 도해를 산출한다.

### Target Audience
조선소 설계·운항 실무진, 사내 연구소 자율운항 연구원, 딥러닝 비전공 관리자. 그림만 훑어도 "두 구조를 같은 크기로 맞춰 비교한다"와 "몇 척이 보이든 같은 크기 요약이 나온다"는 두 가지 사실이 전달되어야 하며, 텍스트 없이도 도식이 자립하는 수준의 가독성을 기대한다.

### Key Message
파라미터 수를 거의 동일하게 맞춘 두 시계열 백본을 비교하고 가변 개수 타선 집합은 어텐션으로 고정 크기 벡터로 압축한다.

### Scene Description
An off-white surface fills the entire 3840x2160 frame and is organised into an upper overview zone occupying roughly the top half and a lower processing-flow strip occupying the bottom half, with a large editorial headline mass anchoring the upper-left corner. The upper zone is dominated by two forty-five degree isometric photorealistic three-dimensional module blocks of visibly identical outer volume standing side by side on a shared invisible baseline, and a brushed-metal balance beam spans between them sitting perfectly level to make the parameter-matched control condition legible as physical equilibrium rather than as a stated fact. The left module's interior is rendered as a lattice of cross-linked rods running across the variable direction so attention across variables is visible as horizontal cross-bracing, while the right module's interior is rendered as a chain of sequential cells linked one after another along the time direction, and both modules share the same identical inlet port on the left face and the same identical outlet port on the right face rendered in the same accent tone to show the fixed interface. The lower strip carries a left-to-right processing flow in three stations, the first a loose scatter of many small isometric ship markers of varying count enclosed in a dashed accent boundary to signal variability, the second a polished glass query prism that draws faint converging beams from every marker toward itself, and the third a single compact solid metal bar of fixed dimensions, and beside the third station a small hollow outline of the same bar sits empty to depict the zero-vessel case. Stations and modules sit inside twelve-pixel soft-rounded section fields with one-pixel secondary hairlines, two-pixel secondary connectors with open triangular arrowheads link the three lower stations left to right, and a single accent connector drops from the upper zone down into the flow strip to bind overview to detail. Objects are rendered twenty percent larger than their nominal zones and both zones stretch to the outer safe margins so negative space stays at or below fifteen percent, and the icon-to-text area ratio is pushed toward the upper limit of the theme so that diagrams clearly dominate. No watermarks, no blurry text, no placeholder brackets, no duplicated labels, no random artifacts, no pedestals or stone bases under the three-dimensional objects, no figure caption numbers, no colour legend boxes, no axis labels, no data tables. CRITICAL: Only render the exact text strings listed in the CONTENT block below. Do NOT generate, infer, or add any additional Korean text beyond what is explicitly written in CONTENT. If a box or area has no CONTENT text assigned, fill it with icons or illustrations — never with AI-generated Korean sentences.

### Rendering Style
- 서피스: 45도 아이소메트릭 포토리얼리스틱 3D 오브젝트. 상단은 외형 부피가 눈에 띄게 동일한 두 개의 모듈 블록으로, 좌측 내부는 변수 방향으로 교차 결합된 격자 로드, 우측 내부는 시간 방향으로 하나씩 이어진 순차 셀 체인. 두 모듈을 잇는 수평을 유지한 브러시드 메탈 저울대와 양쪽에 동일하게 달린 입출력 포트. 하단은 개수가 들쭉날쭉한 소형 선박 마커 무리, 수렴 빔을 끌어당기는 광택 유리 질의 프리즘, 치수가 고정된 단일 금속 바와 그 옆의 빈 윤곽. 받침대 없이 부유시키고 접촉 그림자만 남긴다.
- 배경: 오프 화이트 단색 배경. 도트 그리드와 패턴 장식을 넣지 않고 3D 오브젝트 하단 미세 앰비언트 그림자만 허용해 모듈 블록이 배경 위에 떠 있게 한다.
- 코너/경계: 상단 두 모듈 영역과 하단 세 스테이션 영역을 12픽셀 소프트 라운딩 섹션 필드로 구획하고 1픽셀 보조색 헤어라인으로 경계를 잡는다. 가변 개수를 뜻하는 첫 스테이션의 선박 무리만 강조색 파선 경계로 감싸 다른 영역과 성격을 구분한다.
- 연결선: 하단 세 스테이션을 잇는 2픽셀 보조색 실선 화살표 2개를 좌에서 우로 배치하고, 상단 개요 영역에서 하단 흐름 영역으로 내려오는 강조색 연결선 1개만 추가한다. 질의 프리즘으로 모이는 수렴 빔은 연결선이 아니라 옅은 광선 효과로 처리해 선 개수가 늘지 않게 한다.
- 시각장식: 상단 두 모듈의 입출력 포트를 동일한 강조색으로 칠해 인터페이스가 고정되었음을 색으로 표시하고, 저울대의 수평 상태 자체를 공정 비교의 시각 장식으로 삼는다. 헤드라인 아래 강조색 얇은 룰 한 줄만 추가한다. 색상 범례 박스, 축 라벨, 데이터 테이블, 아키텍처 다이어그램용 각주 번호는 넣지 않는다.
- 공간구성: 매거진 에디토리얼 레이아웃에 상단 개요와 하단 흐름의 2단 구성을 얹는다. 좌상단 헤드라인이 앵커이고 상단 절반을 두 모듈이 대칭으로 채우며 하단 절반을 세 스테이션이 좌우로 관통한다. 이 슬라이드는 기술 밀도가 높으므로 텍스트 항목을 최소로 유지하고 도식과 3D 오브젝트가 화면 면적의 대부분을 차지하게 하며, 오브젝트를 20퍼센트 확대해 네거티브 스페이스를 15퍼센트 이하로 억제한다.
- 시각메타포: 에디토리얼 매거진 × 아이소메트릭 3D. 수평을 유지한 저울대가 파라미터 정렬이라는 실험 통제를 한눈에 증명하고, 개수가 제각각인 선박 무리가 언제나 같은 치수의 금속 바 하나로 압축되는 장면이 고정 크기 요약이라는 개념을 사물로 번역한다. 빈 윤곽 바 하나가 타선이 없는 순간을 조용히 설명한다.

### Content Placement
좌상단에 '백본 비교와 주변선박 인코더'를 캔버스에서 가장 큰 매거진 헤드라인으로 배치하고 그 아래 강조색 얇은 룰을 지나 '구조 차이만 비교한다'를 보조색 중간 크기 문장으로 붙인다. 상단 좌측 모듈 블록 아래에 '아이트랜스포머'를 소제목 크기로 배치하고 그 아래 한 단계 작은 크기로 '변수축 어텐션'을, 다시 그 아래 강조색 칩 안에 '80,041개 파라미터'를 배치한다. 상단 우측 모듈 블록 아래에 '장단기 메모리'를 좌측 소제목과 완전히 동일한 크기와 동일한 세로 위치에 배치하고 그 아래에 '시계열 대조군'을, 다시 그 아래 강조색 칩 안에 '80,147개 파라미터'를 좌측 칩과 같은 세로 위치에 배치한다. 두 모듈 사이 수평 저울대 바로 위에 '파라미터 차이 0.1%'를 강조색 굵은 문구로 배치하고, 저울대 아래에 '입출력 인코더 동일'을 보조색 작은 문구로 배치해 통제 조건이 저울 그림과 함께 읽히게 한다. 하단 흐름 영역에서는 첫 스테이션 파선 경계 아래에 '가변 개수 타선'을, 가운데 질의 프리즘 아래에 '학습된 질의 벡터'를, 마지막 금속 바 아래에 '고정 크기 단일 벡터'를 각각 소제목보다 한 단계 작은 동일 크기로 배치한다. 빈 윤곽 바 옆에 '타선 0척은 영벡터'를 가장 작은 크기로 붙인다. 화면 최하단 우측에 '시간축 처리만 교체'를 캡션 크기로 대비를 낮춰 배치한다.

## CONFIGURATION

### Canvas Settings
- resolution: 3840x2160
- aspect_ratio: 16:9
- layout: section_flow_overview_plus_pipeline

### Background Treatment
- base: 오프 화이트 단색 평면
- ambient_shadow: 3D 오브젝트 하단 미세 접촉 그림자만 허용
- texture: 없음. 도트 그리드, 패턴, 노이즈 오버레이 금지
- ornament: 헤드라인 하단 강조색 얇은 룰 1개, 두 모듈의 동일 강조색 입출력 포트, 가변 개수 영역 강조색 파선 경계

### Color Palette
- primary: #2C3E50 (네이비) - 헤드라인, 백본 소제목, 3D 모듈 블록의 어두운 금속 톤
- secondary: #5D6D7E (슬레이트 블루) - 구조 설명 문구, 섹션 헤어라인, 좌우 흐름 화살표, 순차 셀 체인
- accent: #2980B9 (딥 블루) - 파라미터 칩, 통제 조건 문구, 입출력 포트, 질의 프리즘 수렴 빔
- background: #F8F9FA (오프 화이트) - 전체 배경면

### Typography
All Korean text must be rendered with crisp, perfectly formed characters using heavy-weight Gothic-style sans-serif fonts. Each Korean syllable block must be complete and legible. Use Bold weight (700+) for titles, Medium weight (500) for body text.
- hierarchy: 헤드라인 > 백본 소제목 > 통제 조건 문구 > 흐름 단계 라벨 > 캡션 위계
- contrast: 텍스트 총량을 의도적으로 억제하고 도식이 화면을 지배하도록 본문 크기를 절제한다
- alignment: 상단 두 모듈의 텍스트는 각 모듈 중심 기준 가운데 정렬, 하단 흐름 라벨은 각 스테이션 중심 기준 가운데 정렬
- spacing: 좌우 백본의 텍스트 세로 위치를 완전히 동일하게 맞추고 상단 개요와 하단 흐름 사이에 넉넉한 수평 여백을 둔다

## CONTENT
title: "백본 비교와 주변선박 인코더"
lead_message: "구조 차이만 비교한다"
backbone1_label: "아이트랜스포머"
backbone1_desc: "변수축 어텐션"
backbone1_metric: "80,041개 파라미터"
backbone2_label: "장단기 메모리"
backbone2_desc: "시계열 대조군"
backbone2_metric: "80,147개 파라미터"
control_note: "파라미터 차이 0.1%"
fixed_note: "입출력 인코더 동일"
flow_step1: "가변 개수 타선"
flow_step2: "학습된 질의 벡터"
flow_step3: "고정 크기 단일 벡터"
zero_case: "타선 0척은 영벡터"
footnote: "시간축 처리만 교체"

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
- 메타데이터 헤더 텍스트: 영역, 역할, 구성, 타입, 백본, 파라미터 열
- 한영 병기 표현: 어텐션 (Attention), 백본 / Backbone, 인코더(Encoder)
- 렌더링 힌트 텍스트: (굵게), (강조), (이탤릭), (밑줄)
- 기관 로고, 조선소 마크, 국가 상징, 사명 워드마크
- 로렘 입숨, 의미 없는 더미 텍스트, 알아볼 수 없는 문자열
- CONTENT 내부 번호 목록, 표 형식, subsection 헤더
- 받침대와 플랫폼: 콘크리트 베이스, 스톤 플랫폼, 사각 받침대, 지층 단면
- 학술 논문 요소: Figure 1., 그림 1., 색상 범례 박스, X축, Y축, 축척 바, 네트워크 층 번호 표기
- 수식 표기: 행렬 기호, 벡터 차원 표기, 소프트맥스 수식
- 원문에 없는 정량 성능 수치: 백본별 성능 비교 실측치, 정확도, 손실값, 추론 속도
- AI가 자체 생성한 한글 설명문: CONTENT 블록에 명시되지 않은 어떤 한글 텍스트도 이미지 내부에 렌더링하는 것을 절대 금지
