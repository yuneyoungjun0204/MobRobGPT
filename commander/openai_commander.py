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

from .schema import BattlefieldState, CommanderPlan, decode_plan_to_meters
from .prompts import SYSTEM_PROMPT, build_user_content
from ._validate import _validate_routes


def _idle_plan(reason: str) -> CommanderPlan:
    """LLM 미적용 시 빈 계획(전원 예비=정지). 폴백 휴리스틱 없음."""
    return CommanderPlan(routes=[], rationale=reason)


class OpenAICommander:
    def __init__(
        self,
        model: str = "gpt-4o-mini",   # 구조적 출력 지원 모델 (gpt-3.5 는 미지원)
        api_key: str | None = None,   # None 이면 OPENAI_API_KEY 환경변수 사용
        verbose: bool = True,
    ):
        self.model = model
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
            return _idle_plan("OpenAI 미연결 — 명령 미적용(정지 유지)")
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_content(state)},
                ],
                response_format=CommanderPlan,   # ← 스키마 강제 + 자동 파싱
            )
            msg = completion.choices[0].message
            if getattr(msg, "refusal", None):
                raise ValueError(f"모델 거부: {msg.refusal}")
            plan = msg.parsed
            if plan is None:
                raise ValueError("parsed 결과 없음")
            self._validate_semantics(plan, state)
            decode_plan_to_meters(plan, state)       # 중심 오프셋 → 절대좌표[m]
            self._log(f"LLM 배정 성공 ({self.model}): 투입 {len(plan.routes)}척(6-WP 경로)")
            return plan
        except Exception as e:
            self._log(f"LLM 배정 실패({type(e).__name__}: {e}) → 명령 미적용(정지)")
            return _idle_plan(f"LLM 실패({type(e).__name__}) — 명령 미적용(정지)")

    def warmup(self) -> None:
        """클라우드 API 는 사전 로드 개념이 없음 — 연결 여부만 로그."""
        if self.client is not None:
            self._log(f"OpenAI 준비됨: {self.model}")

    def _validate_semantics(self, plan: CommanderPlan, state: BattlefieldState) -> None:
        _validate_routes(plan, state)


__all__ = ["OpenAICommander"]
