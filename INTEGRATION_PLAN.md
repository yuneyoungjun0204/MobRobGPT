# MobRobGPT × ONE-WAY 통합 계획서 (LLM 기반 WP 생성 + 3-에이전트)

> 작성 목적: MobRobGPT(LLM→경로계획)의 프롬프트/WP 생성 구조를 분석하고,
> 이를 우리 태스크에 맞게 개조하여 **ONE-WAY Towing(`boatattack_sim`) 해상 다중 USV 방어**와 결합하기 위한 **설계 계획**.
> 본 문서는 **계획만** 담는다 (코드 변경 없음). 실제 구현은 승인 후 별도 진행.
>
> 방법론: Sequential Thinking (단계적 분해) + Context7 라이브러리 검증(OpenAI SDK / SciPy).

---

## 0. 한눈에 보는 결론 (TL;DR)

| 질문 | 답 |
|------|-----|
| GPT 프롬프트는 어디? | `MobileRobot_Pygame_GPT.py` **L57–L69** (`text_pre`+`text_loc`+`text_rules`=`guidelines`) 와 **L72–L78** (`assistants.create(instructions=...)`) |
| WP는 어떻게 생성? | **현재 LLM은 WP를 직접 만들지 않음.** LLM은 `goal`+`avoid`(repulsor)만 출력 → **APF(인공 포텐셜 필드) 경사하강**이 실제 경로(xdes/ydes)를 생성 (L252 `path()`) |
| LLM 입력(센서/환경)? | 원(목표 후보) 좌표+색, 사각 장애물 `(x,y,w,h)`, 로봇 **현재 위치** (L47–L55, L82–L94) |
| LLM 출력 형식? | 문자열 `$goal:(x,y); avoid:[(x1,y1);(x2,y2)]$` (L61–L67 규칙, L376+ 파싱) |
| 우리 태스크로 바꾸려면? | ① 환경 인코딩 교체 ② 출력을 **구조적 WP 리스트 JSON**으로 승격 ③ APF→ONE-WAY 디코더 연결 (§4) |
| 3-에이전트로? | LLM = **고수준 지휘관**(역할·클러스터 배정), ONE-WAY Actor = **저수준 WP/그물 전개**. 배정 방식 3안 (§5) |
| ONE-WAY 결합? | **2계층 아키텍처**: LLM 지휘관(느린 주기, 미션 의도) → ONE-WAY 정책(빠른 주기, WP 미세보정) (§3, §6) |

---

## 1. MobRobGPT 현재 구조 분석

### 1.1 GPT 프롬프트가 있는 파일과 위치
전부 **`MobileRobot_Pygame_GPT.py`** 한 파일에 있다.

| 구간 | 라인 | 내용 |
|------|------|------|
| `text_pre` | L57–L59 | 태스크 정의: "APF로 이동로봇 경로계획, 800×800 공간, 장애물 `(a,b,c,d)`, 색 원들. 목표/회피점을 정하라" |
| `text_loc` | L55 | **환경 상태 주입**: 원 위치+색 리스트, 장애물 리스트 (동적 문자열) |
| `text_rules` | L61–L67 | **출력 형식 강제**: `$goal:(x,y); avoid:[...]$` 포맷 규칙 5개 |
| `guidelines` | L69 | 위 3개 결합 → system instruction |
| `assistants.create` | L72–L78 | 이 guidelines를 `instructions`로, `model="gpt-3.5-turbo-1106"` Assistant 생성 |
| `create_message` | L82–L94 | 매 명령마다 `"현재 위치 {x,y}, {사용자 입력}"` 을 thread에 추가 |
| 응답 파싱 | L376–L424 | `$...$` 안을 `goal:` / `avoid:[...]` 로 split → `goal_position`, `repulsors` 세팅 |

> 핵심: **프롬프트 = "환경 서술(text_loc) + 규칙(text_rules)"**, 런타임 입력 = **"현재위치 + 자연어 명령"**.

### 1.2 "WP 생성" 방식 — 현재는 WP를 직접 만들지 않는다 (중요)
현재 파이프라인은 2단이다:

```
자연어 명령 ─(LLM)→ goal(1점) + repulsors(N점)  ─(APF 경사하강)→  경로 배열 xdes[],ydes[],tht[]
     L355               L379~L417 파싱                 L252 path()  → 최대 10만 스텝 궤적
```

- LLM은 **목표점 1개 + 회피점들**만 결정 (WP 리스트 아님).
- 실제 궤적(waypoint 시퀀스에 해당)은 **APF 포텐셜 필드의 gradient를 따라 step_size=0.05로 적분**해서 만든다 (L273–L314).
- 로컬 미니마 탈출: 10스텝 정체 감지 시 랜덤 코너를 임시 목표로 (L295–L309).
- 즉 **"WP 생성기"는 LLM이 아니라 APF**다. LLM은 APF의 **경계조건(goal/repulsor)만 세팅**.

### 1.3 LLM에 들어가는 입력(센서/환경 데이터)
"센서"라기보다 **사전에 알고 있는 맵 상태**를 텍스트로 넣는다:

| 입력 | 소스 | 형식 |
|------|------|------|
| 목표 후보(원) | `circle_positions`, `c_colors` (L41–L47) | `BLUE:(450,750); RED:(50,550); ...` (y축 반전 좌표) |
| 장애물 | `obstacle_positions` (L45) | `(x,y,w,h)` 사각형 리스트 (경계벽 4개 포함) |
| 로봇 현재 위치 | 런타임 `x_current,y_current` (L356) | `"현재 위치 x, y"` 문장 |
| 사용자 명령 | 입력창 텍스트 (L355) | `"go to blue circle"` 등 자연어 |

> 실제 센서(LiDAR/카메라/GPS)는 없음 — **완전 관측(맵 전체 좌표 제공)** 가정의 심볼릭 환경.

### 1.4 LLM 출력 형식
```
$goal:(x_goal,y_goal); avoid:[(x1,y1);(x2,y2)]$      # 회피 있을 때
$goal:(x_goal,y_goal)$                                # 회피 없을 때
```
- `$...$` 구분자로 감싼 **단일 goal + 다중 repulsor**. 좌표는 float.
- 파싱은 문자열 split (L376–L417) — **취약**(포맷 어긋나면 조용히 실패). → 우리 개조 시 1순위 교체 대상.

---

## 2. ONE-WAY(`boatattack_sim`)의 WP 생성 방식 (대조군)

> 출처: `One-Way_Towing/boatattack_sim/{README,OBSERVATION,FAN_ACTION_PORT}.md`

- **도메인**: 적 10척이 맵 가장자리→중앙 모선으로 전진. 아군 USV **3척(P=3)** 이 **그물(net)** 을 전개해 포획. 아군 속도 = 적×0.75.
- **WP 생성 주체 = 학습된 정책(Actor)** + 휴리스틱 디코더. LLM 아님.
  - `env/encoding.py decode_plan` / `defense_env.heuristic_route_netgo(fan)`: **fan 7파라미터 → WP 경로 + 그물 구간**으로 디코드.
  - 액션(부채꼴): `bearing / r_near / r_far / spread / aux_radial / aux_lateral / curve` (`FAN_ACTION_PORT.md §2.1`).
  - **지속 풀 5~6 WP 경로**를 매 결정마다 **작은 델타로 미세보정**(순환 순찰, 급변 금지).
- **관측(입력)**: per-agent **egocentric** (자기 정면 기준), 순열불변 집합, 유리함수 정규화 (`OBSERVATION.md`):
  - own[9]: 위치·모선거리/방위·잔여그물·전개중·배정·교점거리/방위
  - enemy[4×5]: 적 **클러스터**(각도 gap) 거리·방위·스프레드·멤버수·접근속도
  - ally[7×6]: 타 아군 거리·방위·헤딩차(cos/sin)·계획 그물끝점 (협조/겹침방지)
- **출력**: `wp_delta` 또는 `fan[7]` + `net_go`(그물 펼침 여부) + `net_dir`(방향) + `assign`.
- **학습**: GRPO(critic-free), Deep Sets Actor ~155k 파라미터.

### 대조 요약
| | MobRobGPT | ONE-WAY(boatattack_sim) |
|---|---|---|
| WP 생성기 | APF 경사하강 (LLM은 goal/repulsor만) | 학습 정책 + 부채꼴 디코더 |
| 에이전트 수 | 1 | **3 (파라미터 공유)** |
| 관측 | 전역 심볼릭(맵 전체) | egocentric 상대좌표 집합 |
| 출력 | goal + repulsors (문자열) | WP델타/fan[7] + net_go + assign |
| 고수준 의도 | **LLM(자연어)** ✔ | 없음(보상으로 암묵) ✘ |
| 저수준 최적화 | 없음(고정 APF) | **RL 정책** ✔ |

> **핵심 통찰**: 두 프로젝트는 **상보적**이다. MobRobGPT엔 "자연어 의도→목표"라는 상위 계층이 있고, ONE-WAY엔 "관측→최적 WP/그물"이라는 하위 계층이 있다. **결합 = 상위(LLM) + 하위(ONE-WAY)** 를 붙이는 것.

---

## 3. 통합 아키텍처 제안 — 2계층 지휘 구조

```
┌─────────────────────────────────────────────────────────────┐
│  L2  LLM 지휘관 (Commander)  — 느린 주기(수 초~미션 이벤트마다)   │
│      입력: 전장 요약(적 클러스터, 아군 상태, 모선 위험도)          │
│      출력: 미션 의도 = {에이전트별 역할·담당 클러스터·목표영역·제약} │
│      형식: 구조적 JSON (Context7: responses.parse + Pydantic)   │
└───────────────┬─────────────────────────────────────────────┘
                │  (배정/목표를 관측에 주입)
                ▼
┌─────────────────────────────────────────────────────────────┐
│  L1  ONE-WAY 정책 (3× 공유 Actor) — 빠른 주기(매 스텝)           │
│      입력: egocentric obs + L2가 준 배정/목표영역                │
│      출력: fan[7] + net_go + net_dir  → WP 경로 + 그물 구간      │
└───────────────┬─────────────────────────────────────────────┘
                ▼
       kinematics PD 추종 + grid 그물 painting + 포획/breach 판정
```

- **왜 2계층인가**: LLM은 매 시뮬 스텝(수십 ms) 호출이 불가능(지연·비용). 그래서 **LLM은 "무엇을/누가"(전략)**, **정책은 "어떻게"(전술 WP)** 로 분리. 이는 ONE-WAY 문서의 "receding-horizon 고수준 플래너"와 정확히 결선된다.
- **MobRobGPT에서 재사용할 부분**: 프롬프트 엔지니어링 골격(`text_pre/loc/rules` 3분할), 환경→텍스트 인코딩 아이디어, 자연어 명령 파이프라인.
- **버릴 부분**: APF `pot_field`/`path`(ONE-WAY가 대체), pygame 단일 로봇 뷰어(ONE-WAY renderer 사용), 취약한 문자열 파싱(구조적 출력으로 대체).

---

## 4. 우리 태스크에 맞춘 개조 설계 (입력·출력 재설계)

### 4.1 입력(프롬프트) 재설계 — "환경 서술"을 우리 도메인으로 교체
현재 `text_loc`(원+장애물)을 **해상 전장 요약**으로 교체:

```
전장 요약(JSON 직렬화 후 프롬프트에 삽입):
  mothership: {pos, radius, threat_level}
  enemy_clusters: [{id, center, bearing, spread, count, approach_speed}, ...]   # 4클러스터
  allies: [{id, pos, heading, nets_remaining, assigned_cluster}, ...]           # 3척
  constraints: {net_max_len, ally_speed, enemy_speed, arena_bounds}
```
- `text_pre`(태스크 정의)와 `text_rules`(출력 규칙)는 **골격 유지, 내용만 해상 방어로** 재작성.
- 자연어 명령(선택): "왼쪽 밀집 무리 우선 차단", "모선 정면 우선 방어" 등 인간 개입 여지.

### 4.2 출력 재설계 — 문자열 → **구조적 WP/배정 JSON** (Context7 검증)
> Context7(`/openai/openai-python`) 확인: 현재 SDK는 **Structured Outputs**를 지원.
> `client.responses.parse(model=..., text_format=PydanticModel)` 또는
> `client.chat.completions.parse(..., response_format=PydanticModel)` 로 **스키마 강제 + 자동 파싱**.
> `strict=True` json_schema로 변환되어 포맷 붕괴가 없다 → 현재 `$...$` split(L376) 취약성 완전 제거.

제안 스키마(Pydantic, 예시):
```python
class AllyOrder(BaseModel):
    ally_id: int
    role: Literal["intercept", "reserve", "flank"]
    target_cluster_id: int          # 담당 적 클러스터
    goal_area: tuple[float, float]  # 저수준 정책이 미세보정할 목표 영역 중심
    deploy_net: bool                # 이 배가 그물 전개 담당인지
    priority: float                 # 0~1

class CommanderPlan(BaseModel):
    orders: list[AllyOrder]         # 정확히 3개(3척)
    rationale: str                  # 판단 근거(디버깅/로그용)
```
- LLM은 **WP 좌표를 직접 뱉지 않는다**. `goal_area`+`role`+`target_cluster`만 주고, **정확한 WP는 ONE-WAY 정책**이 관측 기반으로 만든다. (LLM 좌표 정밀도 한계 회피 + RL 강점 활용)
- 단, **LLM 단독 baseline**(정책 없이 LLM이 WP 리스트까지 출력)도 비교군으로 만들 수 있음 → 스키마에 `waypoints: list[tuple]` 옵션 필드 추가.

### 4.3 모델 선택 (Context7)
- Assistants Beta API(현재 L72 `beta.assistants`)는 레거시 → **Responses API + Structured Outputs**로 이전 권장.
- 지연·비용 vs 추론력: 지휘관은 저빈도 호출이므로 상위 모델 사용 여유 있음. (모델 ID는 구현 시 최신으로 확정)

---

## 5. 3대 다중 에이전트 설계 (핵심 질문)

ONE-WAY는 이미 **아군 3척(P=3, 파라미터 공유 Actor)** 구조다. 관건은 **"LLM 지휘관이 3척을 어떻게 조정(배정)하느냐"**. 3가지 안:

### 안 A — 중앙집권 지휘관 (권장, 1차 구현)
- LLM **1회 호출로 3척 전체 배정**(`CommanderPlan.orders[3]`)을 한 번에 출력.
- 장점: 전역 최적 배정(집중 공격에 다대일 대응 — ONE-WAY의 `8.5 위협비례 다대일` 문제를 LLM이 자연어 추론으로 해결), 호출 1회로 저렴.
- 단점: LLM 지연 동안 배정 고정(receding-horizon 주기로 완화).
- **ONE-WAY 연결**: LLM 배정을 `defense_env._compute_assignment` **대체/오버라이드**로 주입 → 나머지(WP 미세보정·그물)는 학습 정책 유지.

### 안 B — 분산 에이전트 (각 배가 자기 LLM 페르소나)
- 3개 LLM 인스턴스가 각자 관점(egocentric obs)으로 자기 행동 협상.
- 장점: 통신 두절·부분관측에 강함, 창발적 협조 연구용.
- 단점: 비용 3배+, 충돌 조정 프로토콜 필요(누가 어느 클러스터). 연구 후순위.

### 안 C — 하이브리드 (지휘관 + 경량 로컬 재조정)
- LLM 지휘관이 큰 배정, 각 배는 로컬 규칙/소형 정책으로 실시간 미세조정(그물 겹침 회피 등).
- ONE-WAY의 ally obs(계획 그물끝점 공유)가 이미 겹침방지 신호를 제공 → 안 A + ONE-WAY 협조관측으로 사실상 C에 근접.

> **권장 경로: A → (필요시) C**. B는 멀티에이전트-LLM 연구 목적일 때만.

### 다중 에이전트 조정에서 반드시 정할 것
1. **배정 세분도**: 클러스터 단위(권장) vs 개별 적 단위.
2. **재계획 트리거**: 고정 주기(N스텝)? 이벤트(새 클러스터 출현/breach 임박/그물 소진)?
3. **충돌 해소**: 두 배가 같은 클러스터 → 방위 슬롯 분할(ONE-WAY `8.5 ①` 참조).
4. **LLM 실패 시 폴백**: 타임아웃/파싱실패 → ONE-WAY 순수 휴리스틱 배정으로 폴백(안전).

---

## 6. ONE-WAY 결합 상세 (인터페이스 계약)

두 시스템을 붙이는 **접점은 "배정/목표 주입"** 한 곳으로 최소화 (ONE-WAY 설계원칙 §단일소스 준수):

| 접점 | ONE-WAY 측 | 주입 내용 | 파일(참고) |
|------|-----------|-----------|-----------|
| 배정 오버라이드 | `_compute_assignment` | LLM `orders[].target_cluster_id`, `role` | `env/defense_env.py` |
| 목표영역 힌트 | obs own #8,#9 (배정 교점 거리/방위) | LLM `goal_area` → 교점으로 인코딩 | `env/encoding.py`, `OBSERVATION.md §2` |
| 그물 담당 | `net_go` 마스크 | LLM `deploy_net` | `FAN_ACTION_PORT.md §2` |
| 폴백 | `heuristic_action()` | LLM 실패 시 fan=0 휴리스틱 | `FAN_ACTION_PORT.md §4.4` |

- **torch 격리 원칙 유지**: LLM 호출은 `env` 코어 밖 **별도 지휘 모듈**(`commander/llm_commander.py` 신설 예정)에 두고, env엔 배정 배열만 넘긴다 → sim 단독 테스트·Unity 이식성 보존.
- **frame_dict 계약 확장**: 렌더러에 LLM 의도(역할 색상/담당 클러스터 라벨) 오버레이 추가 → 지휘 결정 시각 검증.
- **학습과의 관계**: 초기엔 LLM 배정 + **기존 학습 정책 그대로**(재학습 불필요). 이후 "LLM 배정 분포"를 학습 커리큘럼에 반영해 정책을 LLM-지휘에 특화시킬 수 있음(선택).

---

## 7. 단계별 로드맵 (각 단계 끝 = 검증 산출물)

| Phase | 내용 | 산출/검증 |
|-------|------|-----------|
| **P0** 스펙 | 지휘관 입출력 Pydantic 스키마 확정 + 프롬프트 3분할 초안 | `commander/schema.py`, 프롬프트 md |
| **P1** LLM 지휘관 단독 | 전장요약 JSON → `CommanderPlan` 파싱 (Responses API, strict) | 오프라인 샘플 전장 10종에 배정 출력, 포맷 100% |
| **P2** 배정 주입 | ONE-WAY `_compute_assignment` 오버라이드 훅 + 폴백 | fan=0 휴리스틱과 배정만 다른 스모크(경로 유한·경계내) |
| **P3** 폐루프 시청 | `eval/policy_play`에 LLM 지휘관 붙여 남해맵 렌더 | 집중공격(C0×10)에서 다대일 배정으로 방어 개선 관찰 |
| **P4** 평가 | LLM지휘 vs 순수 휴리스틱 배정 cap_rate/breach 비교 | `RESULTS.md`에 지표 표 |
| **P5** (선택) 특화학습 | LLM 배정 분포로 커리큘럼, 3-에이전트 협조 튜닝 | cap_rate 향상 |

---

## 8. 리스크 & 결정 필요 사항

- **지연/실시간성**: LLM 왕복(수백 ms~수 초)은 매 스텝 불가 → **저빈도 재계획 + 폴백** 필수. (아키텍처로 해소, §3)
- **좌표 정밀도**: LLM은 정확한 WP 좌표에 약함 → **좌표는 정책에, 의도만 LLM에** (§4.2). 결정: LLM에 WP까지 시킬지(baseline) 여부.
- **비용**: 안 A(1호출/재계획)로 최소화. 안 B는 3×.
- **SciPy 유산**: MobRobGPT의 `interp2d`(APF)는 통합 시 **폐기**되므로 SciPy 버전 문제 자연 소멸. (ONE-WAY는 numpy 코어)
- **Assistants→Responses 이전**: 레거시 API 탈피 (Context7 권장).
- **결정 대기 항목**:
  1. 배정 세분도(클러스터 vs 개별)?
  2. 재계획 트리거(주기 vs 이벤트)?
  3. LLM에 WP 좌표까지 위임하는 baseline도 만들지?
  4. 지휘관 모델 등급(추론력 vs 비용/지연)?
  5. 통합 코드 위치: `boatattack_sim` 내 `commander/` 신설 vs MobRobGPT 리포에 브릿지?

---

## 부록 A. 참고 파일 인덱스
- MobRobGPT 프롬프트/파싱: `MobileRobot_Pygame_GPT.py` L55–L94, L376–L424
- MobRobGPT APF(폐기 대상): 같은 파일 L112–L328
- ONE-WAY 개요/실행: `One-Way_Towing/boatattack_sim/README.md`
- ONE-WAY 관측 스키마: `.../OBSERVATION.md`
- ONE-WAY 액션(부채꼴)/디코드/배정: `.../FAN_ACTION_PORT.md` (특히 §2, §4.4, §8.5)
- ONE-WAY 전체 작업분해: `.../TASKS.md`

## 부록 B. Context7 검증 메모
- **OpenAI Python SDK**: Structured Outputs 지원 — `responses.parse` / `chat.completions.parse` + `pydantic_function_tool`, 내부적으로 `{"type":"json_schema","strict":True}` 변환 → 스키마 강제. (현재 코드의 `$...$` 문자열 파싱 대체 근거)
- **SciPy**: `interp2d`는 1.10 deprecated→1.14 제거. 통합 후 APF 폐기로 무관해짐(참고용).
