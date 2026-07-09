# 지휘관 LLM 모델 검토 (GPT 대안 비교)

> 목적: MobRobGPT×ONE-WAY 통합의 **"지휘관 LLM"**(전장 요약 → 3척 배정/목표 JSON 출력, [[INTEGRATION_PLAN]] §3~5) 역할에
> 현재 `gpt-3.5-turbo-1106` 대신 어떤 모델이 더 적합한지 검토한다.
> 방법론: Sequential Thinking(요구사항 도출 → 후보 평가 → 추천) + Context7(OpenAI/Gemini SDK)·`claude-api` 스킬(Claude 제원)로 근거 확보.
>
> ⚠️ 가격/모델ID는 시점에 따라 바뀐다. Claude 제원은 `claude-api` 스킬 기준(캐시 2026-06-24), OpenAI/Gemini/로컬은 구현 직전 공식 문서로 재확인 필요.

---

## 0. 결론 먼저 (TL;DR)

| 시나리오 | 1순위 추천 | 이유 |
|---|---|---|
| **개발/학습 루프 지휘관** (반복 호출, 온라인) | **Claude Haiku 4.5** 또는 **Sonnet 5** | 저지연·저비용·**네이티브 구조적 출력(strict)** → 배정 JSON 100% 파싱. 배정은 초당 호출 아님 → Haiku로 충분, 난도 높으면 Sonnet |
| **어려운 전술 추론이 필요할 때** | **Claude Opus 4.8** (adaptive thinking) | 집중공격 다대일 배정 같은 비자명 판단·"생각 과정(rationale)" 품질 최고 |
| **해상 실전/오프라인(인터넷 없음)** | **로컬 오픈웨이트**(Qwen2.5 / Llama 3.x Instruct, Ollama/vLLM) | 클라우드 API 불가 환경. 함정·임베디드 배치 → ONE-WAY의 임베디드 지향과 정합 |
| **현행 유지(최소 변경)** | OpenAI 최신 모델로 업그레이드 | gpt-3.5는 구식. 단 Assistants Beta→Responses API 이전 권장([[INTEGRATION_PLAN]] §4.3) |

**핵심 판단**: 지휘관은 **매 시뮬 스텝이 아니라 receding-horizon(수 초~이벤트)로 저빈도 호출**([[INTEGRATION_PLAN]] §3)이므로, **지연보다 "구조적 출력 신뢰성 + 배정 추론력"이 우선**. → **Claude(구조적 출력 네이티브) 우세**. 실전 오프라인이면 **로컬 모델 필수**.

---

## 1. 이 역할의 요구사항 (평가 축)

지휘관 LLM이 해야 하는 일 = 전장요약 → `CommanderPlan{orders[3], rationale}` 출력.

| # | 요구사항 | 왜 중요한가 | 가중치 |
|---|---|---|---|
| R1 | **구조적 출력 신뢰성** | 배정 JSON이 깨지면 폐루프 정지. 현행 `$...$` 문자열 파싱은 취약([[INTEGRATION_PLAN]] §4.2) | ★★★ |
| R2 | **전술 추론력** | 집중공격에 다대일 배정 등 비자명 판단(ONE-WAY `8.5`) | ★★★ |
| R3 | **지연(latency)** | 저빈도 호출이라 치명적이진 않으나 학습 루프에선 누적 | ★★ |
| R4 | **비용** | 학습/평가서 수천~수만 회 호출 → 토큰 단가 누적 | ★★ |
| R5 | **오프라인/엣지 가능성** | 해상·함정 배치 시 인터넷 부재. ONE-WAY 임베디드 지향 | ★★(실전 ★★★) |
| R6 | **툴/함수 호출** | 향후 지휘관이 시뮬 API를 직접 호출하는 확장 | ★ |

---

## 2. 후보별 검토

### 2-A. Claude (Anthropic) — **구조적 출력·추론 우위**
`claude-api` 스킬 기준(캐시 2026-06-24). 컨텍스트는 전 모델 넉넉(≥200K).

| 모델 | 모델 ID | 입력 $/1M | 출력 $/1M | 지연 | 이 역할 적합도 |
|---|---|---|---|---|---|
| Opus 4.8 | `claude-opus-4-8` | $5 | $25 | 느림(추론↑) | R2 최상 — 어려운 배정·rationale |
| Sonnet 5 | `claude-sonnet-5` | $3($2 인트로) | $15($10) | 중 | **균형 최적** — 대부분의 배정 |
| Haiku 4.5 | `claude-haiku-4-5` | $1 | $5 | **빠름** | **저빈도 배정에 충분·최저비용** |

- **R1 구조적 출력**: `client.messages.parse(..., output_format=PydanticModel)` 또는 `output_config.format`(json_schema, **strict**) → 스키마 강제·자동검증. 현행 문자열 파싱 취약성 제거. **툴 `strict:true`도 지원**(R6).
- **R2 추론**: Opus/Sonnet은 **adaptive thinking**(`thinking:{type:"adaptive"}`)+`effort`(low~max)로 사고 깊이 조절. rationale(생각 과정)을 구조적 출력에 그대로 담기 좋음(이전 논의 "생각 과정 보기").
- **주의**: 이 계열은 `temperature`/`top_p`/`budget_tokens` 제거(보내면 400) — 프롬프트로 제어.
- **R5**: 클라우드 API only(오프라인 불가) → 실전 함정 배치엔 부적합.

### 2-B. Google Gemini — 멀티모달·긴 컨텍스트, Vertex 온프렘 옵션
Context7(`/googleapis/python-genai`) 확인: Gemini Developer API + Vertex AI 통합 SDK(`google-genai`), 구조적 출력(`response_schema`/`response_mime_type=application/json`)·함수호출 지원.

- **장점**: 초대형 컨텍스트, **Flash 티어가 매우 저지연/저비용**(R3/R4), 멀티모달(향후 지도/카메라 관측을 직접 넣을 여지). Vertex AI 경유 시 조직 인프라 통합.
- **단점**: 정확한 모델ID/가격은 시점 의존 → **구현 직전 공식 문서 확인 필수**. 완전 오프라인은 불가(Vertex도 클라우드).
- **적합**: 온라인 환경에서 비용/지연 민감 + 멀티모달 확장 계획 시 대안.

### 2-C. OpenAI GPT — 현행, 업그레이드 경로
- 현행 `gpt-3.5-turbo-1106` + **Assistants Beta API**(`beta.assistants/threads`)는 구식. 최신 GPT + **Responses API/`responses.parse`(Pydantic·strict)**로 이전 시 R1 해결([[INTEGRATION_PLAN]] §4.2).
- **장점**: 코드 최소 변경(같은 벤더), 생태계 성숙. **단점**: 오프라인 불가(R5). 추론 품질은 최신 모델이어야 경쟁력.

### 2-D. 로컬 오픈웨이트 (Qwen2.5 / Llama 3.x / Mistral) — **오프라인 유일해**
- **Ollama**(간편) 또는 **vLLM**(고성능 서빙)으로 로컬 구동. **함정/엣지에서 인터넷 없이 동작**(R5 유일 충족) → ONE-WAY 임베디드 지향과 정합.
- **구조적 출력**: vLLM/Ollama의 **JSON 스키마/문법 강제(guided decoding, GBNF)** 로 R1 확보 가능. 7B~32B Instruct급이면 3척 배정 정도의 구조적 판단은 실현 가능.
- **단점**: R2 추론력은 프론티어 클라우드보다 낮음(모델 크기·양자화 트레이드오프), 자체 GPU/최적화 필요. **하드웨어 제약** 시 배정을 단순화하거나 규칙+소형LLM 하이브리드.
- **적합**: **실전 배치·데이터 주권·비용0(추론)**. 개발 단계에선 클라우드로 프로토타입 후 로컬 포팅 전략 권장.

---

## 3. 정리 비교표

| 축 | Claude Opus4.8 | Claude Sonnet5 | Claude Haiku4.5 | Gemini(Flash/Pro) | OpenAI(최신) | 로컬(Qwen/Llama) |
|---|---|---|---|---|---|---|
| R1 구조적출력 | ★★★ strict | ★★★ strict | ★★★ strict | ★★ schema | ★★★ strict | ★★ guided decode |
| R2 추론 | ★★★ | ★★★ | ★★ | ★★~★★★ | ★★~★★★ | ★~★★ |
| R3 지연 | ★ | ★★ | ★★★ | ★★★(Flash) | ★★ | ★★(HW의존) |
| R4 비용 | ★ | ★★ | ★★★ | ★★★(Flash) | ★★ | ★★★(추론0) |
| R5 오프라인 | ✘ | ✘ | ✘ | ✘ | ✘ | ★★★ |
| R6 툴호출 | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★ |

> 저빈도 호출(R3 완화)·구조적출력 필수(R1 ★★★)를 반영하면 **개발기: Claude Haiku4.5/Sonnet5**, **난제: Opus4.8**, **실전 오프라인: 로컬**의 조합이 최적.

---

## 4. 추천 전략 — 단계별 이원화

```
개발/학습·평가 (온라인)                       실전 배치 (오프라인 함정/엣지)
────────────────────────                      ────────────────────────────
Claude Sonnet5 (기본 배정)                     로컬 Qwen2.5/Llama3.x Instruct
  └ 어려운 케이스만 Opus4.8로 승격               (vLLM/Ollama + JSON 스키마 강제)
  └ 저난도/저비용은 Haiku4.5                     └ 개발기 프롬프트/스키마 그대로 포팅
구조적 출력: messages.parse(Pydantic, strict)   구조적 출력: guided decoding(GBNF)
```

- **왜 이원화?**: 지휘관 인터페이스(전장요약 JSON → `CommanderPlan` JSON)를 **벤더 독립 계약**([[INTEGRATION_PLAN]] §6)으로 두면, **모델 교체가 한 모듈(`commander/`) 안의 어댑터 교체**로 끝난다. 개발은 클라우드 최고모델로 빠르게, 실전은 동일 스키마로 로컬 포팅.
- **폴백**: LLM 실패/타임아웃 → ONE-WAY 순수 휴리스틱 배정([[INTEGRATION_PLAN]] §5, `heuristic_action` fan=0). 어떤 모델이든 이 안전망 위에서 동작.

---

## 5. 결정 필요 사항
1. **온라인 클라우드 vs 오프라인 로컬** — 실전 배치 환경(인터넷 유무)이 모델 선택을 좌우. 이게 1번 결정.
2. **개발기 기본 티어** — Sonnet5(균형) vs Haiku4.5(저비용) 중 시작점?
3. **멀티모달 확장** — 향후 지도/카메라 관측을 LLM에 직접 넣을 계획이면 Gemini 가중.
4. **벤더 종속 허용치** — 어댑터 계층으로 독립 유지할지, 단일 벤더로 단순화할지.

## 부록 — Context7/스킬 검증 메모
- **Claude**: `claude-api` 스킬 — 모델ID/가격(캐시 2026-06-24), `messages.parse`+Pydantic·`output_config.format`(strict), adaptive thinking·effort, `temperature`/`budget_tokens` 제거(400).
- **Gemini**: Context7 `/googleapis/python-genai` — 통합 `google-genai` SDK(Gemini API+Vertex), 구조적 출력·함수호출. 모델ID/가격은 공식 문서 재확인.
- **OpenAI**: Context7 `/openai/openai-python` — Responses API·`responses.parse`+`pydantic_function_tool`(strict json_schema). 현행 Assistants Beta는 레거시.
- **로컬**: 벤더 문서 아님(Context7 범위 밖) — vLLM/Ollama guided decoding은 구현 시 각 문서 확인.
