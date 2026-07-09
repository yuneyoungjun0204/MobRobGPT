# Ollama 지휘관 실행 준비 가이드

> [[INTEGRATION_PLAN]]의 "LLM 지휘관"을 **로컬 Ollama(오프라인·무료추론)** 로 돌리기 위한 준비 완료 문서.
> 벤더 독립 `commander/` 모듈로 구현 — 어댑터만 바꾸면 Claude/Gemini/OpenAI로 교체 가능.
> 모델 검토·근거: [[MODEL_REVIEW]]. Ollama API는 Context7(`/ollama/ollama-python`)로 검증.

---

## 0. 준비된 것 (이미 생성됨)

```
commander/
  __init__.py            # 공개 API (OllamaCommander, BattlefieldState ...)
  schema.py              # 전이 계약: BattlefieldState → CommanderPlan (Pydantic)
  prompts.py             # 역할·규칙(고정) + 동적 전장 프롬프트
  fallback.py            # LLM 실패 시 위협비례 휴리스틱 배정 (안전망)
  ollama_commander.py    # Ollama 어댑터 (format=schema 강제 + 검증 + 폴백)
  demo.py                # 집중공격 샘플로 실행/검증
```

**검증 완료**: `python -m commander.demo --fallback` → 집중공격(적 8:2)에 다대일 배정
(`seats=[2,1]`, 클러스터당 1척 그물) 정상 출력. `pydantic 2.12.5` 확인.
남은 건 **Ollama 설치 + 모델 pull**뿐.

---

## 1. Ollama 설치

### Windows (현재 환경)
1. https://ollama.com/download 에서 Windows 설치 → 실행 (백그라운드 서버 `localhost:11434` 자동 기동)
2. 설치 확인:
```powershell
ollama --version
```

### 서버 수동 기동(필요 시)
```powershell
ollama serve      # 이미 앱이 떠 있으면 불필요
```

---

## 2. 모델 내려받기

```powershell
# 권장 (VRAM 12GB+ 급): qwen2.5 14B (기본 Q4_K_M, 약 9GB)
ollama pull qwen2.5:14b

# 대안 (VRAM 부족 시): 7B
ollama pull qwen2.5:7b

# 더 고정밀(VRAM 여유): 양자화 태그 지정
ollama pull qwen2.5:14b-instruct-q6_K
```

내려받은 모델 확인:
```powershell
ollama list
```

> **GPU/모델 선택 기준** ([[MODEL_REVIEW]] §속도):
> - RTX 4090/3090(24GB): `qwen2.5:14b` (Q4~Q6) — 배정 1회 ~1초
> - RTX 4070/3080(12~16GB): `qwen2.5:14b` Q4 — ~2~4초
> - 그 이하/CPU: `qwen2.5:7b` 권장 (14B는 과부하)

---

## 3. Python 의존성

```powershell
cd C:\Users\ANSL\Desktop\MobRobGPT
pip install ollama pydantic
```
(`pydantic`는 이미 설치되어 있음 — 2.12.5 확인됨. `ollama` 패키지만 추가하면 됨.)

---

## 4. 실행 / 검증

```powershell
# LLM 배정 (Ollama+qwen2.5:14b). 실패 시 자동으로 휴리스틱 폴백.
python -m commander.demo

# 다른 로컬 모델로
python -m commander.demo qwen2.5:7b
python -m commander.demo llama3.1:8b

# LLM 건너뛰고 폴백 로직만 확인 (Ollama 없이도 동작)
python -m commander.demo --fallback
```

정상 출력 예 (집중공격 시나리오):
```
=== CommanderPlan ===
  ally 0: role=intercept cluster= 0 net=Y prio=1.00 goal=(5200,2500)
  ally 1: role=flank     cluster= 0 net=- prio=1.00 goal=(5200,2500)
  ally 2: role=intercept cluster= 1 net=Y prio=0.25 goal=(8000,6000)
  rationale: ...
```
→ 적 8척 클러스터에 2척, 2척 클러스터에 1척 (다대일 배정) 이면 성공.

> **한글 깨짐 시**: PowerShell 콘솔 인코딩 문제일 뿐 로직은 정상.
> `chcp 65001` 또는 `$env:PYTHONUTF8=1` 로 UTF-8 강제 가능.

---

## 5. 코드에서 사용

```python
from commander import OllamaCommander, BattlefieldState  # 스키마 필드로 상태 구성
from commander.demo import sample_state

cmd = OllamaCommander(model="qwen2.5:14b", keep_alive="10m")
plan = cmd.plan(sample_state())      # -> CommanderPlan (항상 유효; 실패 시 폴백)

for order in plan.orders:
    print(order.ally_id, order.role, order.target_cluster_id, order.deploy_net)
```

주요 옵션 (`OllamaCommander(...)`):
| 인자 | 기본값 | 설명 |
|---|---|---|
| `model` | `"qwen2.5:14b"` | Ollama 모델 태그 |
| `host` | `None` | 원격 함정/엣지 서버 (`"http://ip:11434"`) |
| `keep_alive` | `"10m"` | 모델 상주 시간 → 매 호출 콜드로드 방지 |
| `num_ctx` | `4096` | 컨텍스트 길이 |
| `num_predict` | `800` | 최대 생성 토큰 (배정 JSON엔 충분) |

---

## 6. 신뢰성 설계 (왜 안전한가)

1. **구조적 출력 강제**: `format=CommanderPlan.model_json_schema()` → 스키마 벗어난 출력 불가.
   현행 GPT의 `$...$` 문자열 파싱 취약성([[INTEGRATION_PLAN]] §4.2) 제거.
2. **2단 검증**: 스키마 통과 후 `_validate_semantics`로 "아군 수 일치·id 유일·클러스터 존재" 확인.
3. **폴백 보장**: LLM 미설치·타임아웃·스키마위반·의미위반 → `heuristic_plan`(위협비례 배정)로
   **폐루프가 절대 멈추지 않음**. ONE-WAY 순수 휴리스틱 정신([[INTEGRATION_PLAN]] §5)과 동일.
4. **결정적**: `temperature=0` → 같은 전장엔 같은 배정(재현성).

---

## 7. 모델 교체 (벤더 독립)

`commander/`의 스키마·프롬프트·폴백은 그대로 두고 **어댑터만 추가**하면 된다:
- `ollama_commander.py` → `claude_commander.py`(messages.parse+Pydantic) / `gemini_commander.py`(response_schema) / `openai_commander.py`(responses.parse)
- 어느 것이든 `plan(state) -> CommanderPlan` 시그니처만 지키면 교체 완료.
- 개발기 클라우드 최고모델로 A/B → 검증된 프롬프트를 로컬 Ollama로 포팅([[MODEL_REVIEW]] §4).

---

## 8. 다음 단계 (ONE-WAY 결합)

이 지휘관을 `boatattack_sim`에 연결 ([[INTEGRATION_PLAN]] §6):
1. `defense_env`의 관측 → `BattlefieldState`로 변환하는 인코더 작성
2. `cmd.plan(state).orders` → `_compute_assignment` 오버라이드로 주입
3. `eval/policy_play`에 붙여 남해맵에서 폐루프 시청 → 집중공격 방어 개선 관찰

> 현재까지: **지휘관 모듈 단독 실행 준비 완료**. ONE-WAY 관측↔BattlefieldState 인코더가 다음 작업.
