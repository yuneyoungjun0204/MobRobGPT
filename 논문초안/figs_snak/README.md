# figs_snak — 도판 파일 위치

`.tex` 의 `\graphicspath{{figs_snak/}}` 가 이 폴더를 가리킨다.

현재 들어 있는 6개 PNG 는 **자리표시용 플레이스홀더**다. 화면에 "PLACEHOLDER" 라고
찍혀 있으므로 교체를 잊으면 바로 눈에 띈다. 생성한 실제 도판을 같은 이름으로 덮어쓰면 된다.

| 파일 | 도판 | 필요 비율 | 권장 해상도 |
|---|---|---|---|
| `fig1_concept.png` | 교전 개념 | 4:3 | 2400×1800 |
| `fig2_architecture.png` | 2계층 구조·비동기 주기 | 12:5 | 4200×1750 |
| `fig3_commander.png` | 지휘관 계층 | 4:3 | 2400×1800 |
| `fig4_observation.png` | 다채널 래스터 관측 | 4:3 | 2400×1800 |
| `fig5_policy.png` | 점수맵 정책 (U-Net) | 12:5 | 4200×1750 |
| `fig6_grpo.png` | 그룹상대 학습 | 4:3 | 2400×1800 |

생성 프롬프트는 `../../논문 그림/prompts/` 에 있다. 먼저 `style_sheet.md` 를 읽을 것.

비율이 다르면 `.tex` 의 `width=` 는 유지되고 높이만 달라져 플로트 배치가 흔들린다.
비율은 맞춰서 생성하는 편이 안전하다.

컴파일: `xelatex USV_swarm_defense_paper_CNN.tex` 2회.
