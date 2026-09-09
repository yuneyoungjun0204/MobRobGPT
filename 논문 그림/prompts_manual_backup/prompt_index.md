# Prompt Index — 논문 도판 6종

먼저 `style_sheet.md` 를 읽고 시작한다. 6장 전체에 공통 적용되는 규약이 거기 있다.

| # | 파일 | 도판 | LaTeX label | 배치 | 캔버스 | 레지스터 |
|---|---|---|---|---|---|---|
| 1 | `01_engagement_concept.md` | 교전 개념 (그물 포획) | `fig:concept` | 단단 80 mm | 2400×1800 | A (실사) |
| 2 | `02_system_architecture.md` | 2계층 구조 + 비동기 주기 | `fig:architecture` | **양단** `figure*` | 4200×1750 | B + 실사 썸네일 |
| 3 | `03_commander_assignment.md` | 지휘관 계층 · 배정 → 유효영역 | `fig:commander` | 단단 80 mm | 2400×1800 | B 위 / A 아래 |
| 4 | `04_raster_observation.md` | 다채널 래스터 관측 | `fig:observation` | 단단 80 mm | 2400×1800 | A 좌 / B 우 |
| 5 | `05_score_map_policy.md` | 점수맵 정책 (U-Net) | `fig:policy` | **양단** `figure*` | 4200×1750 | B + 실사 우측 |
| 6 | `06_grpo_training.md` | 그룹상대 학습 + 반사실 기여 | `fig:grpo` | 단단 80 mm | 2400×1800 | A + B |

## 산출물 배치

생성한 이미지는 다음 이름으로 `논문초안/figs_snak/` 에 넣는다 (`.tex` 의 `\graphicspath` 가 여기다):

```
fig1_concept.png
fig2_architecture.png
fig3_commander.png
fig4_observation.png
fig5_policy.png
fig6_grpo.png
```

## 두 레지스터 (style_sheet.md §0 요약)

- **A — 해면 장면**: 위성 정사영상 품질의 실사 해면 + 사진 수준 탑다운 선박 렌더 +
  그 위에 얹은 반투명 CAD 오버레이. 배를 아이콘으로 대체하면 실패.
- **B — 모델/데이터**: 딥러닝 논문 관례. 오블리크 텐서 슬래브, 박스 위 채널 수, 옆 해상도,
  색 구분 연산 화살표, 회색 점선 skip, 실제 렌더된 viridis 히트맵, 컬러바,
  대괄호를 갖춘 진짜 행렬·열벡터.

## 도판 간 일관성 체크리스트

생성 후 6장을 나란히 놓고 확인한다.

- [ ] 실사 패널의 해면 톤·태양 방향·선박 렌더 품질이 1·3·4·5·6에서 같은가
- [ ] 프리깃 / 공격정 / 방어정의 외형이 도판마다 동일한 모델인가
- [ ] 그물이 전 도판에서 주황 부표열 + 수면 아래 망으로 동일하게 그려졌는가
- [ ] 히트맵이 전부 viridis 단일 컬러맵인가 (무지개 섞이지 않았는가)
- [ ] 텐서 슬래브의 오블리크 각도와 박스 채움색이 2·4·5에서 같은가
- [ ] skip 화살표가 전부 회색 점선인가
- [ ] 이미지 안에 `Fig.` / `Figure` / 도판 제목이 하나도 없는가
- [ ] 한글이 단 한 글자도 없는가
- [ ] CONTENT 블록에 없는 문장이 생성되지 않았는가
- [ ] 창작된 성능 수치(포획률·정확도·손실값)가 없는가

## 수치 표기의 출처

`50 x 50`, `15`, `32`, `64`, `25 x 25`, `13 x 13`, 채널 묶음 `9 / 3 / 3` 은 전부 체크포인트
`boatattack_sim/models/u-net_map.pt` 의 실제 config 와 `model/cnn_actor.py::UNetLite` 구조에서
나온 값이다 (`docs/unet_model_deploy.md` §1, §3). 임의로 지어낸 수치가 아니므로 도판에 표기해도
사실성 원칙에 어긋나지 않는다. 반대로 보상값·이득값·포획률처럼 **측정되지 않은 값은 도판에
숫자로 넣지 않는다** — Fig. 6 의 막대 차트에 축 눈금 숫자를 뺀 이유가 이것이다.
