# MobRobGPT 실행 가이드 (복붙용 터미널 명령어)

> **LLM 지휘관 + 강화학습 셀 정책 기반 해상 방어 시뮬레이터.**
> 자연어 명령 → Ollama/OpenAI 지휘관이 아군 3척 배정 → 셀선택 RL 정책이 경로·그물을 생성 →
> 위성 배경 위에서 실시간 시청.
>
> 최종 갱신: 2026-08-09 (브랜치 `Basic`, `15598a7 --world 스케일 변환` 기준)

---

## 0. 30초 요약 — 뭘 실행하면 되나

| 하고 싶은 것 | 명령어 |
|---|---|
| **평소 쓰는 기본 실행** (LLM 지휘관 + 특화 셀 정책) | `python run_commander_ui.py --cell --specialized 30_model` |
| 33m 수조 스케일로 그대로 | `python run_commander_ui.py --cell --specialized 30_model --world 33` |
| LLM 없이 순수 셀 정책만 확인 | `python run_cell_play.py --enemy wave` |
| 창 없이 통계만 (헤드리스) | `python run_sim_headless.py wave` |
| 지휘관 프롬프트만 진단 | `python diag_prompt.py qwen2.5:7b` |
| 스케일 불변 검증 | `python tests/test_scale_e2e.py` |

---

## 1. 환경 준비 (최초 1회)

```powershell
cd C:\Users\ANSL\Desktop\MobRobGPT
.\.venv\Scripts\Activate.ps1
```

> `.venv` 는 이미 있다(Python 3.10). 새 worktree/새 머신이면 아래로 만든다:
> ```powershell
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> pip install torch numpy scipy matplotlib pandas pillow requests pydantic ollama openai
> ```
> 실행 정책 오류 시 한 번만: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

한글 콘솔 깨짐 방지(선택):
```powershell
$env:PYTHONUTF8 = "1"
chcp 65001
```

### Ollama (기본 지휘관 백엔드)
```powershell
ollama --version
ollama list
ollama pull qwen2.5:7b        # 기본값. 14b 대비 품질 동등·약 6배 빠름
ollama pull qwen2.5:14b       # VRAM 12GB+ 여유 시
```
자세한 설치는 `OLLAMA_SETUP.md` 참조.

### OpenAI 백엔드를 쓸 때만
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

---

## 2. 메인 실행 — LLM 지휘관 UI (`run_commander_ui.py`)

왼쪽에 해상 씬, 오른쪽에 지휘관 판단(rationale) 패널, 하단에 자연어 명령 입력창이 뜬다.
시작 시 기본 명령 **"모든 적군 포획"** 이 자동 적용된다.

### 2.1 자주 쓰는 조합 (그대로 복붙)

```powershell
# ★ 기본: LLM 지휘관 + 공격양상 특화 셀 정책 라우팅(집중/양동/파상 자동 판별)
python run_commander_ui.py --cell --specialized 30_model

# 특화 라우팅 없이 단일 범용 셀 정책(best_mixed_far)
python run_commander_ui.py --cell

# 모델 지정 (모델명은 반드시 맨 앞 — 플래그 값으로 오인되지 않게)
python run_commander_ui.py qwen2.5:14b --cell --specialized 30_model

# OpenAI 지휘관으로 (OPENAI_API_KEY 필요)
python run_commander_ui.py --openai --cell --specialized 30_model

# 적 대형 고정해서 시작
python run_commander_ui.py --cell --specialized 30_model --enemy concentrated
python run_commander_ui.py --cell --specialized 30_model --enemy wave
python run_commander_ui.py --cell --specialized 30_model --enemy diversionary

# 자동 재계획 주기 조정 (기본 100 step, 0=끄기)
python run_commander_ui.py --cell --specialized 30_model --replan 50
python run_commander_ui.py --cell --specialized 30_model --replan 0

# 충돌회피 APF 안전층 켜고 시작 (기본 OFF, 창에서 v 키로도 토글)
python run_commander_ui.py --cell --specialized 30_model --apf

# 대안 특화 세트(cos6k)로 라우팅
python run_commander_ui.py --cell --specialized specialized_cos6k_model

# 잔차(residual) RL 정책 경로 — 셀 정책 이전 방식
python run_commander_ui.py --rl --gain 1

# RL 없이 순수 휴리스틱 경로 (배정만 LLM)
python run_commander_ui.py
```

### 2.2 스케일 변환 — 임의 크기 실험장 (`--world`)

관측이 전부 길이/길이 비율이라 **재학습 없이** 아무 크기 실험장에 그대로 쓴다.

```powershell
# 33m × 33m 수조
python run_commander_ui.py --cell --specialized 30_model --world 33

# 100m 야외 실험장
python run_commander_ui.py --cell --specialized 30_model --world 100

# 기본 해상 스케일 (12,600 m) — --world 생략과 동일
python run_commander_ui.py --cell --specialized 30_model
```

> `--world` 를 주면 콘솔에 `★ 스케일 변환: world_size=33m (모든 길이 비례축소, 재학습 불필요)` 가 찍힌다.
> 근거·검증은 `docs/input_normalization.md` 와 `tests/test_scale_e2e.py`.

### 2.3 플래그 전체

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `<모델명>` (위치인자) | `qwen2.5:7b` | Ollama 모델 태그. **반드시 맨 앞에** 둘 것 |
| `--openai` | off | OpenAI 백엔드(`gpt-4o-mini`). `OPENAI_API_KEY` 필요 |
| `--enemy MODE` | `random` | `concentrated` / `diversionary` / `wave` / `grouped` / `random` / `rotate` / `mixed` |
| `--replan N` | `100` | N step 마다 LLM 자동 재계획. `0`=끄기 |
| `--cell` | off | 경로·그물을 셀선택 정책(CellPointerActor)으로. `--rl` 함의 |
| `--rl` | off | 경로를 잔차 RL 정책으로 (배정은 여전히 LLM) |
| `--specialized [ROOT]` | `30_model` | 공격양상 기하분류 → 대형 특화 정책 라우팅. **`--cell` 전용** |
| `--world M` | 없음 | 실험장 한 변 크기(m). 모든 길이 비례축소 |
| `--ckpt PATH` | `--cell`→`boatattack_sim/models/best_mixed_far.pt`, `--rl`→`boatattack_sim/models/rl_latest.pt` | 정책 체크포인트 |
| `--gain K` | `1` | `--rl` 잔차 배율. 셀 모델에는 무의미 |
| `--apf` | off | 충돌회피 APF 안전층 ON |

> ⚠️ **`--specialized` 는 값을 붙여 쓸 것.** 값 없이 다른 플래그 앞에 두면
> (`--specialized --world 33`) 다음 토큰인 `--world` 를 경로로 잘못 읽는다.
> 항상 `--specialized 30_model` 처럼 명시하거나 명령 맨 끝에 둔다.

### 2.4 창 조작키

| 키 | 동작 |
|---|---|
| `space` | 재생 / 일시정지 |
| `r` | 랜덤 시드로 리셋 + 기본 명령 재적용 |
| `1` / `2` / `3` | 집중 / 파상 / 양동 대형으로 리셋·재시작 (상단 버튼과 동일) |
| `a` | LLM 자동 재계획 ON/OFF |
| `v` | APF(충돌회피) 토글 |
| `c` | 경로 겹침 해소 토글 (RL 모드) |
| `z` | 후보셀 오버레이 토글 (`--cell` 모드) |
| `q` | 종료 |

하단 입력창 사용 예:
```
모든 적군 포획
정면 밀집 무리를 우선 차단
큰 무리에 2척, 1척은 예비
좌측 클러스터는 무시하고 우측만 막아라
```

---

## 3. LLM 없이 정책만 확인

### 3.1 셀선택 RL 정책 (`run_cell_play.py`)

```powershell
python run_cell_play.py                                          # 기본(best_mixed_far, diversionary)
python run_cell_play.py --enemy wave
python run_cell_play.py --enemy concentrated --seed 7
python run_cell_play.py --ckpt 30_model/wave/best.pt --enemy wave        # 대형 특화 모델 직접 지정
python run_cell_play.py --ckpt specialized_cos6k_model/concentrated/best.pt --enemy concentrated
python run_cell_play.py --heur                                   # 정책 대신 휴리스틱 셀선택(비교용)
python run_cell_play.py --no-joint                               # 배별 독립 greedy(교차잠금 끔)
python run_cell_play.py --apf --spf 5                            # APF ON, 프레임당 5 micro-step
```
조작키: `space` 일시정지 · `r` 리셋 · `c` 셀 오버레이 · `v` APF · `q` 종료

### 3.2 잔차 RL 정책 (`run_rl_play.py`)

```powershell
python run_rl_play.py                                            # rl_latest.pt, greedy
python run_rl_play.py --enemy wave
python run_rl_play.py --ckpt boatattack_sim/models/rl_latest_final.pt
python run_rl_play.py --sample                                   # greedy 대신 분포 샘플
python run_rl_play.py --gain 8                                   # 잔차를 8배 과장해 시각화
```
조작키: `space` · `r` · `g`(greedy/sample) · `[` `]`(잔차배율 ∓) · `v` · `q`

### 3.3 순수 휴리스틱 시뮬 (정책·LLM 모두 없이)

```powershell
python run_sim_headless.py                    # 창 없이 1 에피소드 통계
python run_sim_headless.py wave

python render_sim.py                          # 라이브 창
python render_sim.py --enemy wave
python render_sim.py --gif out.gif            # GIF 저장(헤드리스/공유용)
python render_sim.py --png shot.png --at 100  # 100 step 시점 PNG
```

---

## 4. 지휘관(LLM) 단독 진단

```powershell
# 같은 전장에 서로 다른 명령을 주고 배정이 실제로 바뀌는지 확인
python diag_prompt.py                 # 기본 Ollama
python diag_prompt.py qwen2.5:7b
python diag_prompt.py qwen2.5:14b
python diag_prompt.py --openai

# 지휘관 모듈 단독 데모
python -m commander.demo              # Ollama 호출
python -m commander.demo qwen2.5:7b
python -m commander.demo --fallback   # Ollama 없이 휴리스틱 폴백만 검증
```

**판독법**: rationale 이 `[휴리스틱...` 으로 시작하면 LLM 파싱 실패 → 콘솔의
`[commander] LLM 배정 실패(...)` 사유를 본다. `[LLM]` 인데 명령마다 배정이 똑같으면
모델이 명령을 안 따르는 것(프롬프트/모델 문제).

---

## 5. 검증 (테스트)

pytest 아님 — 그냥 스크립트로 돌린다.

```powershell
python tests\test_scale_invariance.py    # 12.6km vs 33m 정규화 관측이 완전히 동일한가
python tests\test_scale_e2e.py           # 실제 정책 1 에피소드 → 포획/돌파/충돌이 같게 나오는가
```
기대 출력: `결과: 동일 — 스케일 불변`

---

## 6. 체크포인트 지도

| 경로 | 내용 |
|---|---|
| `boatattack_sim/models/best_mixed_far.pt` | **셀 정책 기본값** (혼합 대형, 원거리 요격) |
| `boatattack_sim/models/cell_latest.pt`, `cell_latest_final.pt` | 이전 셀 정책 |
| `boatattack_sim/models/rl_latest.pt`, `rl_latest_final.pt` | 잔차 RL 정책 |
| `30_model/{concentrated,diversionary,wave}/best.pt` | **추론용 대형 특화 셀 정책 (기본 라우팅 대상)** |
| `specialized_cos6k_model/{...}/best.pt` | cos6k 스케줄로 학습한 대안 특화 세트 |

`--specialized <ROOT>` 는 `<ROOT>/<대형>/best.pt` 를 찾는다. 대형 판정은 LLM이 아니라
적 분포의 기하 특징(방위 집중도 R, 반경 퍼짐, 클러스터 수)으로 결정적으로 이뤄진다
(`commander/formation_router.py`).

---

## 7. 자주 겪는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 창이 안 뜨고 `모델 로딩 중…` 에서 멈춤 | Ollama 서버 미기동 | `ollama serve` 또는 Ollama 앱 실행 |
| `위성 배경 실패(오프라인?)` | Esri 타일 접근 불가 | 무시해도 됨 — 해색 배경으로 자동 폴백 |
| rationale 이 항상 `[휴리스틱` | LLM 스키마 위반/타임아웃 | `python diag_prompt.py` 로 원인 확인, 모델 교체 |
| `--specialized` 가 이상한 경로를 잡음 | 값 없이 다른 플래그 앞에 둠 | `--specialized 30_model` 처럼 명시 |
| `[오류] ... 는 셀선택(cell_action) 정책이 아닙니다` | 잔차 ckpt 를 `run_cell_play.py` 에 줌 | `run_rl_play.py` 사용 |
| 한글이 □ 로 깨짐 | 한글 폰트 없음 | Malgun Gothic 설치된 Windows면 자동. 콘솔은 `chcp 65001` |
| GPU 점유 충돌 | 다른 학습 진행 중 | `nvidia-smi` 로 먼저 확인 |

---

## 8. 부록 — 초기 pygame 데모 (`MobileRobot_Pygame_GPT.py`)

프로젝트 원형인 지상로봇 APF 경로계획 데모. 현재 파이프라인과는 별개다.

```powershell
pip install "openai>=1.40,<2" "scipy<1.14" pygame numpy matplotlib pandas
$env:OPENAI_API_KEY = "sk-..."
python MobileRobot_Pygame_GPT.py
```
> `scipy.interpolate.interp2d` 를 쓰므로 **`scipy < 1.14`** 필수(1.14에서 제거됨).
> `client.beta.assistants` 사용 → **openai v1.x** 계열.

---

## 9. 관련 문서

- `OLLAMA_SETUP.md` — Ollama 설치·모델 선택 기준·지휘관 API
- `docs/input_normalization.md` — 셀 정책 입력 정규화 사양 (스케일 불변의 근거)
- `docs/llm_commander_roadmap.md` — 지휘관 개선 로드맵
- `docs/llm_commander_ppt.md` — 발표용 정리
- `INTEGRATION_PLAN.md` / `MODEL_REVIEW.md` — 통합 설계·모델 검토
