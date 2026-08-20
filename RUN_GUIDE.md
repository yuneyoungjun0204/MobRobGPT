# MobRobGPT 실행 가이드 (터미널 명령어 정리)

> LLM(GPT-3.5) 기반 이동로봇 경로계획 시뮬레이터.
> `pygame`로 800×800 2D 환경을 그리고, 인공 포텐셜 필드(APF)로 경로를 계획하며,
> 자연어 명령("go to blue circle")을 OpenAI Assistants API로 해석해 목표/회피점을 설정합니다.

---

## 0. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 메인 스크립트 | `MobileRobot_Pygame_GPT.py` |
| 보조 모듈 | `robot.py` (로봇 기구학, 향후 확장용) |
| 핵심 의존성 | `pygame`, `numpy`, `scipy`, `matplotlib`, `pandas`, `openai` |
| 필수 환경변수 | `OPENAI_API_KEY` |
| 권장 Python | **3.11** (`.pyc` 기준) |

### ⚠️ 버전 호환성 주의 (Context7로 확인됨)
- **SciPy**: 코드가 `scipy.interpolate.interp2d`를 사용합니다. 이 함수는 **SciPy 1.10에서 deprecated → 1.14.0에서 완전히 제거**되었습니다.
  → 반드시 **`scipy < 1.14`** 를 설치해야 실행됩니다. (예: `scipy==1.13.1`)
- **OpenAI SDK**: `client.beta.assistants` / `client.beta.threads` (Assistants Beta API)를 사용합니다.
  → **`openai` v1.x** 계열이 안정적입니다. (예: `openai==1.55.3`)

---

## 1. 사전 준비: Python & Git 확인

```powershell
# Python 버전 확인 (3.11 권장)
python --version

# pip 최신화
python -m pip install --upgrade pip

# (선택) 저장소 클론 — 이미 로컬에 있다면 생략
git clone https://github.com/<your-account>/MobRobGPT.git
cd MobRobGPT
```

---

## 2. 가상환경 생성 및 활성화

### Windows (PowerShell) — 현재 환경
```powershell
cd C:\Users\ANSL\Desktop\MobRobGPT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> 실행 정책 오류 시 (한 번만):
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### macOS / Linux (bash) — 참고
```bash
cd MobRobGPT
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. 의존성 설치

`requirements.txt`가 없으므로 아래 명령으로 직접 설치합니다.
**호환 버전으로 고정 설치(권장):**

```powershell
pip install "openai>=1.40,<2" "scipy<1.14" pygame numpy matplotlib pandas
```

또는 `requirements.txt`를 만들어 관리하려면:

```powershell
# requirements.txt 생성
@"
openai>=1.40,<2
scipy<1.14
pygame
numpy
matplotlib
pandas
"@ | Out-File -Encoding utf8 requirements.txt

# 설치
pip install -r requirements.txt
```

설치 확인:
```powershell
pip list | Select-String "openai|scipy|pygame|numpy|matplotlib|pandas"
```

---

## 4. OpenAI API 키 설정 (필수)

스크립트는 `os.environ['OPENAI_API_KEY']`를 직접 참조하므로 키가 없으면 즉시 오류로 종료됩니다.

### Windows (PowerShell) — 현재 세션에만 적용
```powershell
$env:OPENAI_API_KEY = "sk-여기에-본인-키-입력"
```

### Windows — 영구 등록 (사용자 환경변수)
```powershell
setx OPENAI_API_KEY "sk-여기에-본인-키-입력"
# setx 이후에는 새 터미널을 열어야 반영됩니다.
```

### macOS / Linux (bash) — 참고
```bash
export OPENAI_API_KEY="sk-여기에-본인-키-입력"
```

키 설정 확인:
```powershell
echo $env:OPENAI_API_KEY
```

---

## 5. 실행

```powershell
# 가상환경이 활성화되고 OPENAI_API_KEY가 설정된 상태에서
python MobileRobot_Pygame_GPT.py
```

정상 실행 시 800×800 pygame 창("Robot Motion")이 열립니다.

### 사용 방법 (창 내부)
1. 하단 흰색 입력창에 자연어 명령 입력 후 **Enter**
   - 예: `go to blue circle`
   - 예: `go to the red circle but avoid the green ones`
2. GPT-3.5가 명령을 해석 → 목표(goal)와 회피점(repulsor)을 계산 → 로봇이 경로를 따라 이동
3. 키 조작
   - `↑` / `↓` : 이전/다음 명령 히스토리
   - `Esc` : 입력창 비우기
   - `exit` 입력 후 Enter 또는 창 닫기 : 종료

> 참고: 경로 계산 중 콘솔에 반복 횟수(iteration)와 상태가 출력됩니다. 로컬 미니마에 갇히면 코너를 임시 목표로 삼아 탈출합니다.

---

## 6. 자주 발생하는 문제 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| `ImportError: cannot import name 'interp2d'` | SciPy ≥ 1.14 설치됨 | `pip install "scipy<1.14"` 로 다운그레이드 |
| `KeyError: 'OPENAI_API_KEY'` | 환경변수 미설정 | 4번 단계 수행 |
| `openai.AuthenticationError` | 잘못되거나 만료된 키 | 유효한 API 키 재설정 |
| `AttributeError: client.beta ...` | openai SDK v2 이상에서 API 변경 | `pip install "openai>=1.40,<2"` |
| pygame 창이 안 뜸 / 멈춤 | GPT 응답 대기 중(동기 호출) | 콘솔 로그 확인, 네트워크/키 상태 점검 |
| `RuntimeError: main thread` (matplotlib) | 백엔드 충돌 | matplotlib 진단 플롯은 코드상 주석 처리되어 있어 무시 가능 |

---

## 7. 빠른 실행 요약 (복붙용 · PowerShell)

```powershell
cd C:\Users\ANSL\Desktop\MobRobGPT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install "openai>=1.40,<2" "scipy<1.14" pygame numpy matplotlib pandas
$env:OPENAI_API_KEY = "sk-여기에-본인-키-입력"
python MobileRobot_Pygame_GPT.py
```

---

## 8. 커스터마이징 포인트 (코드 수정)

`MobileRobot_Pygame_GPT.py` 상단 변수들:
- `circle_positions`, `circle_colors` : 목표 후보 점(원)의 위치/색상
- `obstacle_positions` : 사각형 장애물 `(x, y, width, height)`
- `start` : 로봇 시작 위치 (기본 `[400, 100]`)
- `resolution` : 포텐셜 필드 격자 해상도 (작을수록 정밀·느림)
- `k_attractive`, `k_obstacle`, `k_grid` (`pot_field` 내부) : 인력/척력 게인
- `model` (`assistants.create`) : `gpt-3.5-turbo-1106` → 다른 모델로 교체 가능

---

## 9. U-Net 점수맵 정책으로 기동 (`--unet`)

LLM 지휘관이 **배정**(어느 배가 어느 적 클러스터)을 내리고, 경로·그물은
**CNN 점수맵 정책**(`CnnScoreActor` / U-Net lite)이 결정한다. 셀선택 정책(`--cell`)과는
별개 계열이며 **플래그로 병행**한다 — 서로 간섭하지 않는다.

```powershell
# 시뮬 뷰어 (기본 ckpt = boatattack_sim/models/u-net_map.pt)
python run_commander_ui.py --unet
python run_commander_ui.py --unet --enemy wave
python run_commander_ui.py --unet --replan 100          # 100 step 마다 LLM 재계획
python run_commander_ui.py --unet --nets 5              # 배당 그물 5장 (기본 3)

# ROS2 실기 (실센서 → 점수맵 정책 → /ally_X/waypoints 발행)
python run_commander_ui.py --unet --ros2
```

`--cell` 과 `--unet` 은 **함께 쓸 수 없다**(서로 다른 정책 계열이라 즉시 오류).

### 동작 요약

| 단계 | 내용 |
|---|---|
| 관측 | 모선 중심 **50×50 픽셀 · 15채널** 래스터 (픽셀 = 252 m) |
| 배정 반영 | `self_intercept` 채널 + `own[6:8]` + 유효마스크의 방위 코리도(±60°) |
| 출력 | 배마다 **픽셀 2개**(+서브픽셀 보정) |
| 해석 | 가까운 점 = 이동 WP, 먼 점 = 그물 벽 끝점. 그 사이 구간이 그물 |
| 정지 | 미배정(`assign<0`) 또는 그물 소진(`a_nets<=0`) 배는 **제자리** |
| 그물 | 배당 **3장**(`--nets N` 으로 변경). 한 장 완성하면 다음 결정에 새 위치로 재전개 |

### 배당 그물 장수 (`--nets`, 기본 3)

체크포인트의 학습 config 는 `nets_per_ship=1` 이다. 그대로 두면 배가 **그물 한 장을 깔고 나면
`a_nets=0` → 정지 규칙에 걸려 영구히 얼어붙는다.** 운용에서는 배마다 여러 장을 싣는 것이 맞으므로
브릿지가 기본 3장으로 올린다.

재전개 흐름: 그물 완성(`doing_net=False`) → 다음 결정에서 `_apply_cnn_actions` 가 fresh route 를
깔며 `ptr`·`leg_netted`·`paint_dist` 를 리셋 → 새 픽셀 2점으로 다시 전개. `a_nets` 가 0이 되면
그 배는 정지한다.

> ★ **관측 왜곡 방지**: `own[4] = a_nets / nets_per_ship` 은 학습 때 항상 0 아니면 1이었다
> (계약서 §3.3). 장수를 3으로 올리면 1/3·2/3 같은 **학습 때 본 적 없는 중간값**이 들어간다.
> 그래서 `CommandedCnnEnv.build_cnn_obs` 가 이 채널을 **0/1 이진으로 되돌린다** — 남은 장수가
> 몇이든 "깔 그물이 있다"는 1, 소진은 0. 학습된 의미(행동이 세계를 바꿀 수 있는가)가 그대로 보존된다.

효과 (에피소드 4회 평균, 휴리스틱 배정):

| 대형 | 1장/척 포획·돌파 | 3장/척 포획·돌파 |
|---|---|---|
| wave | 8.25 · 1.75 | 8.25 · 1.75 |
| diversionary | 9.00 · 1.00 | 9.00 · 1.00 |
| concentrated | 5.25 · 4.75 | **10.00 · 0.00** |

wave·diversionary 는 그물이 떨어지기 전에 에피소드가 끝나 차이가 없고, 그물을 많이 쓰는
concentrated 에서 전부 잡는 쪽으로 바뀐다.

### 화면

`[z]` 키로 점수맵 오버레이를 켜고 끈다. 배색 heatmap = 그 배의 점수맵,
옅은 색칠 = 선택 가능한 유효 픽셀, `✕` 마커 2개 = 선택 픽셀, 잇는 선 = 그물 벽.
미배정 배는 점수맵이 균등분포(의미 없음)라 그리지 않는다.

### 주의 (모델 계약 — 자세한 내용은 `docs/unet_model_deploy.md`)

- **`greedy` 로만 추론한다.** 이 모델은 겹침 허용 레짐(`cnn_gate_disjoint=False`)에서 학습됐으므로
  `greedy_joint`(배 간 잠금)를 켜면 학습 레짐을 되돌려 성능이 떨어진다.
- **`land_sites` 는 `""` 로 고정한다.** 브릿지가 자동으로 처리한다 — 안 하면 해역이 매 에피소드
  무작위로 바뀌어 화면·마스크·물리가 서로 다른 섬을 가리킨다.
- **그물 벽은 450 m(`net_max_len`)에서 끊긴다.** 마스크는 675 m 까지 허용하지만 실제 도색 상한은
  450 m 다. 실기 제어도 이 상한을 지켜야 학습과 일치한다.
- **적 슬롯은 10개 고정.** 탐지가 모자라면 `e_alive=False`, 넘치면 모선 근접 상위 10개만 넣는다
  (`ROS2CnnEnv` 가 자동 처리).

### ROS2 실기에서 직접 관리하는 상태

센서가 주지 않아 `ROS2CnnEnv` 가 내부로 추적한다:
`a_nets`(남은 그물, 시작 = `--nets`) · `doing_net`(부설 중) ·
`net_installed[200,200]`(이미 깐 그물 래스터).
마지막 것을 빼먹으면 정책이 이미 친 벽을 못 보고 겹쳐 친다.

> ⚠ 그물 진행량(`paint_dist`)은 실기에 릴 엔코더가 없어 **항행 거리로 근사**한다.
> 실제 계측값이 있으면 `commander/ros2_unet_env.py::_advance_and_paint` 의 해당 줄을 교체할 것.

> ⚠ **시계 스케일**: `GeoBridge.fit` 이 t=0 에 잡는 배율(적 최원거리 → `target_sim_radius=5450`)에
> 맞춰 속도·`net_max_len`·`decision_period` 도 같이 스케일해야 학습과 같은 동역학이 된다.
> 현재 구현은 배율만 로그로 남기고 시계 보정은 하지 않는다 — 실기 튜닝 시
> `boatattack_sim/env/scaling.py::SimScale` 로 정식 처리할 것.
