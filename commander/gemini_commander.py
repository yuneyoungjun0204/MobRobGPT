"""Google Gemini 기반 지휘관 어댑터 — OpenAI/Ollama 와 동일 인터페이스.

전장상태 → Gemini(structured output, 스키마 강제) → 검증된 CommanderPlan.
실패/거부/스키마위반 → 위협비례 휴리스틱 폴백(폐루프 안 멈춤).

google-genai SDK (`from google import genai`) 를 쓴다. 구버전
`google.generativeai` 와 API 가 다르므로 섞지 말 것.

구조적 출력:
    client.models.generate_content(
        model=..., contents=...,
        config={"response_mime_type": "application/json",
                "response_schema": CommanderPlan})
    → response.parsed (Pydantic 인스턴스) / response.text (JSON 문자열)

프롬프트: build_messages() 가 만든 OpenAI 형식(system/user 역할)을 Gemini 형식으로
옮긴다 — system 은 `system_instruction` 으로, 나머지는 contents 로.

환경변수 GEMINI_API_KEY (없으면 GOOGLE_API_KEY) 필요.
"""
from __future__ import annotations

import os

from ._validate import sanitize_plan
from .fallback import heuristic_plan
from .prompts import build_messages
from .schema import BattlefieldState, CommanderPlan


def _fallback(state: BattlefieldState, reason: str) -> CommanderPlan:
    """LLM 실패 시 전원 정지 대신 위협비례 휴리스틱으로 방어(전 클러스터 커버). 이유 병기."""
    plan = heuristic_plan(state)
    plan.rationale = f"{reason} → 휴리스틱 방어. {plan.rationale}"
    return plan


def _split_messages(messages) -> tuple[str, str]:
    """build_messages() 의 [{role, content}...] → (system_instruction, user_contents).

    Gemini 는 system 을 별도 필드로 받는다. system 이 여러 개면 이어 붙이고,
    나머지(user/assistant)는 역할 표시를 붙여 하나의 문자열로 합친다 —
    few-shot 예시가 assistant 역할로 들어와도 문맥이 유지된다.
    """
    sys_parts, body = [], []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            sys_parts.append(content)
        elif role == "assistant":
            body.append(f"[EXAMPLE OUTPUT]\n{content}")
        else:
            body.append(content)
    return "\n\n".join(sys_parts), "\n\n".join(body)


class GeminiCommander:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,     # None 이면 GEMINI_API_KEY / GOOGLE_API_KEY
        max_tokens: int = 1200,         # 정상 plan 은 ~300~500 토큰. 폭주 조기 차단.
        verbose: bool = True,
    ):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.verbose = verbose
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        try:
            from google import genai
            if not key:
                raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다")
            self.client = genai.Client(api_key=key)
        except Exception as e:          # SDK 미설치/키 없음 등
            self._log(f"Gemini 클라이언트 초기화 실패({e}) → 폴백 전용 모드")
            self.client = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[commander] {msg}")

    def plan(self, state: BattlefieldState) -> CommanderPlan:
        if self.client is None:
            return _fallback(state, "Gemini 미연결")
        try:
            sys_inst, contents = _split_messages(build_messages(state))
            cfg = {
                "response_mime_type": "application/json",
                "response_schema": CommanderPlan,   # 스키마 강제 + 자동 파싱
                "temperature": 0,                   # 결정적 배정
                "max_output_tokens": self.max_tokens,
            }
            if sys_inst:
                cfg["system_instruction"] = sys_inst
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=cfg)
            plan = getattr(resp, "parsed", None)
            if plan is None:                        # SDK 가 파싱을 못 하면 직접 시도
                txt = (getattr(resp, "text", "") or "").strip()
                if not txt:
                    raise ValueError("빈 응답 (max_output_tokens 초과 또는 안전 차단)")
                plan = CommanderPlan.model_validate_json(txt)
            plan = sanitize_plan(plan, state)       # 거부 대신 정제(중복 ally/무효 id 수리)
            self._log(f"LLM 배정 성공 ({self.model}): {len(plan.deployments)}개 클러스터 배분")
            return plan
        except Exception as e:
            self._log(f"LLM 배정 실패({type(e).__name__}: {e}) → 휴리스틱 방어")
            return _fallback(state, f"LLM 실패({type(e).__name__})")

    def warmup(self) -> None:
        """클라우드 API 는 사전 로드 개념이 없음 — 연결 여부만 로그."""
        if self.client is not None:
            self._log(f"Gemini 준비됨: {self.model}")
