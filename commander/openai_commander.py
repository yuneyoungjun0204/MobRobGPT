"""OpenAI(GPT) 기반 지휘관 어댑터 — OllamaCommander 와 동일 인터페이스.

전장상태 → OpenAI(structured outputs, 스키마 강제) → 검증된 CommanderPlan.
실패/거부/스키마위반 → 위협비례 휴리스틱 폴백(폐루프 안 멈춤).

Context7(/openai/openai-python) 확인 API:
- client.chat.completions.parse(model=, messages=, response_format=<Pydantic>)
  → completion.choices[0].message.parsed (CommanderPlan) / .refusal
- 구조적 출력은 gpt-4o-2024-08-06+ (gpt-4o-mini 포함) 지원. 구형 gpt-3.5 는 미지원.

환경변수 OPENAI_API_KEY 필요.
"""
from __future__ import annotations

from .schema import BattlefieldState, CommanderPlan
from .prompts import build_messages
from ._validate import sanitize_plan
from .fallback import heuristic_plan


def _fallback(state: BattlefieldState, reason: str) -> CommanderPlan:
    """LLM 실패 시 전원 정지 대신 위협비례 휴리스틱으로 방어(전 클러스터 커버). 이유 병기."""
    plan = heuristic_plan(state)
    plan.rationale = f"{reason} → 휴리스틱 방어. {plan.rationale}"
    return plan


class OpenAICommander:
    def __init__(
        self,
        model: str = "gpt-4o-mini",   # 구조적 출력 지원 모델 (gpt-3.5 는 미지원)
        api_key: str | None = None,   # None 이면 OPENAI_API_KEY 환경변수 사용
        max_tokens: int = 1200,       # 출력 상한 — 정상 plan 은 ~300~500 토큰. 폭주(루프) 조기 차단.
        verbose: bool = True,
    ):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.verbose = verbose
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        except Exception as e:  # 라이브러리 미설치/키 없음 등
            self._log(f"OpenAI 클라이언트 초기화 실패({e}) → 폴백 전용 모드")
            self.client = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[commander] {msg}")

    def plan(self, state: BattlefieldState) -> CommanderPlan:
        if self.client is None:
            return _fallback(state, "OpenAI 미연결")
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=build_messages(state),   # system + few-shot + 실제 STATE
                response_format=CommanderPlan,   # ← 스키마 강제 + 자동 파싱
                temperature=0,                   # 결정적 배정 (같은 전장 → 같은 계획)
                max_tokens=self.max_tokens,      # 출력 폭주(길이초과 파싱실패) 방지
            )
            msg = completion.choices[0].message
            if getattr(msg, "refusal", None):
                raise ValueError(f"모델 거부: {msg.refusal}")
            plan = msg.parsed
            if plan is None:
                raise ValueError("parsed 결과 없음")
            plan = sanitize_plan(plan, state)       # 거부 대신 정제(중복 ally/무효 id 수리)
            self._log(f"LLM 배정 성공 ({self.model}): {len(plan.deployments)}개 클러스터 배분")
            return plan
        except Exception as e:
            self._log(f"LLM 배정 실패({type(e).__name__}: {e}) → 휴리스틱 방어")
            return _fallback(state, f"LLM 실패({type(e).__name__})")

    def warmup(self) -> None:
        """클라우드 API 는 사전 로드 개념이 없음 — 연결 여부만 로그."""
        if self.client is not None:
            self._log(f"OpenAI 준비됨: {self.model}")


__all__ = ["OpenAICommander"]
