# 논문초안 — 폴더 지도

마스터 `.tex` 는 **이 폴더 루트에서 컴파일**한다. `\input{sections_*/…}`·`\graphicspath{{figs_snak/…}}` 가
루트 기준 상대경로이고, `tools/` 의 스크립트(build_paper.sh, tex2hwp, make_paper_tables …)도
`논문초안/sections_*`·`tables`·`figs_snak` 경로를 직접 쓴다. 이 셋은 옮기지 말 것.

```
논문초안/
├── USV_swarm_defense_국방발표.tex       ← 최신 · 본문 sections_def/   (+ .pdf · .hwp · _hwp미리보기.pdf)
├── USV_swarm_defense_국방발표_15p.tex   ← 15쪽 축약판 · sections_kdu/ (+ .pdf · .hwp · _hwp미리보기.pdf)
├── USV_swarm_defense_EAAI_KR_real.tex   ← 실그림 판 · sections_real/
├── USV_swarm_defense_EAAI_KR.tex        ← 국문 EAAI 판 · sections/
├── USV_swarm_defense_EAAI.tex           ← elsarticle 투고판 · sections/ (tools/build_paper.sh → build/)
│
├── sections/  sections_real/  sections_def/  sections_kdu/   판별 본문
├── tables/          표 (tools/make_paper_tables.py 등이 생성)
├── figs_snak/       도판. real/ = real·국방발표 판 전용, _unused/ = 본문 미사용 잡파일
├── build/           build_paper.sh 출력 (git 무시)
│
├── 투고부속/        highlights(.tex/.txt/.pdf, tools/make_highlights.py 생성) · 영문초록 번역 기준본
├── 양식_참고/       논문작성 형식.hwp (tex2hwp 템플릿) · glossary_ko_en.md (국영 용어집)
├── 해설/            개념설명_sec2-4/ · 군집화_휴리스틱/ — 각 폴더 안에서 컴파일
├── PPT_그림/        발표용 그림·생성 프롬프트
└── _archive/        구판·복사본·백업 (컴파일 대상 아님 — 경로가 맞지 않는다)
```

## 판 ↔ 본문 폴더

| 마스터 | 본문 | 비고 |
|---|---|---|
| 국방발표 | `sections_def/` | 키워드만 `sections_real/00d_keywords_legacy` |
| 국방발표_15p | `sections_kdu/` | 키워드만 `sections_real/00d_keywords_legacy` |
| EAAI_KR_real | `sections_real/` | 그림 `figs_snak/real/` |
| EAAI_KR · EAAI | `sections/` | 그림 `figs_snak/` + `../논문_그래프/` |

## 빌드

- 본문: `xelatex USV_swarm_defense_국방발표.tex` (루트에서, 상호참조용 2회 이상)
- HWP: `python -m tools.tex2hwp.convert` — 마스터의 `.aux` 를 읽으므로 **먼저 xelatex** 를 돌린다
- EAAI 판별 PDF: `bash tools/build_paper.sh`
- 부산물(`.aux/.log/.out/.xdv …`)은 `.gitignore` 로 무시된다(하위 폴더 포함).
