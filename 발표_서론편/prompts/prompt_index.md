# Prompt Index — 발표_서론편

- theme: seminar
- mood: technical-report
- canvas: 3840x2160 / 16:9
- style_sheet: `style_sheet.md` (슬라이드 1에서 create, 슬라이드 2-16은 follow)
- palette (16장 고정): primary #2C3E50 / secondary #5D6D7E / accent #2980B9 / background #F8F9FA
- 총 16장, 12종 레이아웃 (24종 정식 목록 대조 완료)

## 슬라이드 목록

| 번호 | 파일명 | 테마 | 레이아웃 | CONTENT 항목 수 | self-check |
|:----:|--------|------|----------|:---------------:|:----------:|
| 1 | `01_z-pattern.md` | seminar | Z-Pattern | 9 | PASS |
| 2 | `02_structure.md` | seminar | Structure | 11 | PASS |
| 3 | `03_card-grid.md` | seminar | Card-Grid | 15 | PASS |
| 4 | `04_contrast.md` | seminar | Contrast | 15 | PASS |
| 5 | `05_mind-map.md` | seminar | Mind Map | 12 | PASS |
| 6 | `06_stacked-progress.md` | seminar | Stacked Progress | 13 | PASS |
| 7 | `07_concentric.md` | seminar | Concentric | 17 | PASS |
| 8 | `08_flow.md` | seminar | Flow | 17 | PASS |
| 9 | `09_contrast.md` | seminar | Contrast | 14 | PASS |
| 10 | `10_cycle.md` | seminar | Cycle | 18 | PASS |
| 11 | `11_contrast.md` | seminar | Contrast | 15 | PASS |
| 12 | `12_bento-grid.md` | seminar | Bento Grid | 23 | PASS |
| 13 | `13_section-flow.md` | seminar | Section-Flow | 15 | PASS |
| 14 | `14_contrast.md` | seminar | Contrast | 16 | PASS |
| 15 | `15_group.md` | seminar | Group | 19 | PASS |
| 16 | `16_z-pattern.md` | seminar | Z-Pattern | 16 | PASS |

## Self-check 결과

| 검증 항목 | 기준 | 결과 |
|-----------|------|------|
| 줄 수 | 타이틀 ≥50, 본문 ≥80 | 전 슬라이드 83-98줄, 통과 |
| 4-block 구조 | INSTRUCTION / CONFIGURATION / CONTENT / FORBIDDEN ELEMENTS 순서 | 16/16 통과 |
| 서브섹션 | INSTRUCTION 6개 + CONFIGURATION 4개 | 16/16 모두 10개 통과 |
| CONTENT 항목 수 | 본문 ≥8, 타이틀 ≥3, seminar 최대 25 | 최소 9, 최대 23, 통과 |
| FORBIDDEN 항목 수 | ≥15 | 19-20개, 통과 |
| XML 태그 미포함 | scene, text_to_render 등 부재 | 통과 |
| CONTENT 금지 패턴 | 대괄호, 색상 코드, pt/px, 한영 병기 부재 | 통과 |
| Content Placement 실제 텍스트 인용 | 모든 CONTENT value가 작은따옴표로 인용 | 16/16 누락 0건 |
| 라벨 길이 | value 최대 15자 | 초과 0건 |
| Scene Description 텍스트 유출 | 한글 렌더 문자열 미포함 | 16/16 한글 0건 |
| anti-hallucination 문구 | CRITICAL 문장 포함 | 16/16 포함 |
| AI 자체 생성 한글 금지 항목 | FORBIDDEN에 포함 | 16/16 포함 |
| 팔레트 일치 | style_sheet 4색 완전 일치 | 16/16 일치 |
| 타이포 필수 문구 | heavy-weight Gothic-style 문장 | 16/16 포함 |
| 캔버스 사양 | 3840x2160, 16:9 | 16/16 명시 |

## 원문 충실도

- 원문 `docs/발표_내용_서론편.md`에 없는 정량 성능 수치는 삽입하지 않았다.
- CONTENT의 모든 숫자를 원문과 대조했으며 비대응 항목은 아래 세 종류뿐이다.
  - 번호 배지 `01`-`04`: 데이터가 아닌 구조적 순번 표기
  - `2026년 8월`: 표지의 발표 시점 표기, 원문 문서 일자와 일치
  - `비교군 8종`: 원문 §18의 "각 2개 백본 총 6종 + 규칙 기반 컨트롤러 + 사람 항적"의 합계
- 모델 학습·평가 결과값(정확도, 충돌률 실측치, 손실값, 학습 곡선)은 전 슬라이드에서 FORBIDDEN 항목으로 명시 차단했다.

## 리뷰 권고 반영

| 권고 | 반영 내용 |
|------|-----------|
| 슬라이드 1·2 메시지 중복 완화 | 슬라이드 1 제목을 '자동화의 빈 층' 후킹 카피 + '지금 얼마나 돌릴 것인가' 단일 훅으로 한정하고 3단 레이어 라벨을 전부 제거. 슬라이드 2는 제목을 '중간층 조선 판단'으로 바꾸고 3단 구조 설명 전용으로 분리. 중복 문구가 두 슬라이드에 동시에 렌더링되지 않도록 Content Placement에 명시적 제약 문장 추가 |
| 슬라이드 12·15 항목 8개 이상 시 2단 계층 | 12번은 '자동 채점 5기준' / '보수성 3장치' 두 클러스터로, 15번은 '평가 4축' / '비교군 8종' 두 클러스터로 분리. 각 클러스터에 그룹 헤더 스트립을 두고 클러스터 간 거터를 타일 간 거터의 2배 이상으로 벌려 계층을 시각적으로 강제 |
| 슬라이드 13 아이콘·도식 비중 확대 | CONTENT를 15개로 억제하고 Rendering Style·Scene Description에 "텍스트 총량 억제, 도식이 화면 지배" 지시를 명시. 파라미터 정렬을 수평 저울대로, 고정 크기 요약을 동일 치수 금속 바로, 타선 0척을 빈 윤곽으로 번역해 문장 없이도 개념이 전달되게 구성 |
| 비전공자 배려 (비유 시각화) | 슬라이드 9는 백지 필기시험지 + 도로 없는 조타륜, 슬라이드 11은 동그라미 친 문제집 한 문항 + 결승 게이트 도로로 원문 비유를 3D 오브젝트로 실체화 |
| 레이아웃 명칭 정식 24종 대조 | Z-Pattern(22), Structure(2), Card-Grid(15), Contrast(4), Mind Map(23), Stacked Progress(24), Concentric(9), Flow(1), Cycle(7), Bento Grid(20), Section-Flow(14), Group(8) — 12종 전부 정식 목록에 존재함을 확인 |

## 세션 일관성 장치

- 3D 오브젝트: 전 슬라이드 45도 아이소메트릭, 받침대 없음, 접촉 그림자만, 좌상단 소프트 스튜디오 라이팅
- 섹션 필드: 12픽셀 소프트 라운딩 + 1픽셀 보조색 헤어라인
- 연결선: 2픽셀 보조색 실선 또는 베지어 + 열린 삼각형 화살촉
- 번호 배지: 강조색 배지 + 흰 숫자 (3, 8, 10번 슬라이드)
- 수미상관: 슬라이드 1의 '절반만 형성된 반투명 조타륜'이 슬라이드 16에서 '위아래 모듈에 물린 완결된 조타 기구'로 대응
