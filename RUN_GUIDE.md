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
