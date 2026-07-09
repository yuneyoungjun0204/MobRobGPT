"""Ollama 기반 지휘관 어댑터.

전장상태 → Ollama(qwen2.5:14b 등, JSON Schema 강제) → 검증된 CommanderPlan.
LLM 호출 실패/타임아웃/스키마위반/의미검증실패 시 → 위협비례 휴리스틱 폴백.

Context7(/ollama/ollama-python) 확인 API:
- ollama.Client().chat(model=, messages=, format=<JSON schema>, options=, keep_alive=)
- format 에 Pydantic 의 model_json_schema() 를 넘기면 스키마 강제
- resp.message.content 에 JSON 문자열 → model_validate_json 으로 검증/파싱
"""
from __future__ import annotations

from .schema import BattlefieldState, CommanderPlan
from .prompts import SYSTEM_PROMPT, build_user_content
from ._validate import _validate_routes


def _idle_plan(reason: str) -> CommanderPlan:
    """LLM 미적용 시 빈 계획(전원 예비=정지). 폴백 휴리스틱 없음."""
    return CommanderPlan(routes=[], rationale=reason)


class OllamaCommander:
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        host: str | None = None,          # 예: "http://localhost:11434" / 원격 함정 서버
        keep_alive: str | float = "10m",  # 모델 상주 → 매 호출 콜드로드 방지
        num_ctx: int = 4096,
        num_predict: int = 800,
        verbose: bool = True,
    ):
        self.model = model
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.verbose = verbose
        # 지연 import: ollama 미설치 환경에서도 폴백만으로 동작
        try:
            from ollama import Client
            self.client = Client(host=host) if host else Client()
        except Exception as e:  # pragma: no cover
            self._log(f"ollama 클라이언트 초기화 실패({e}) → 폴백 전용 모드")
            self.client = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[commander] {msg}")

    def plan(self, state: BattlefieldState) -> CommanderPlan:
        """전장상태 → 배정계획. 어떤 오류에도 유효한 CommanderPlan 을 반환(폴백 보장)."""
        if self.client is None:
            return _idle_plan("Ollama 미연결 — 명령 미적용(정지 유지)")
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_content(state)},
                ],
                format=CommanderPlan.model_json_schema(),   # ← 스키마 강제
                options={
                    "temperature": 0,          # 결정적 배정
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
                keep_alive=self.keep_alive,
            )
            plan = CommanderPlan.model_validate_json(resp.message.content)
            self._validate_semantics(plan, state)   # 의미 검증(위반 시 예외)
            self._log(f"LLM 배정 성공 ({self.model}): 투입 {len(plan.routes)}척(6-WP 경로)")
            return plan
        except Exception as e:
            self._log(f"LLM 배정 실패({type(e).__name__}: {e}) → 명령 미적용(정지)")
            return _idle_plan(f"LLM 실패({type(e).__name__}) — 명령 미적용(정지)")

    def warmup(self) -> None:
        """모델을 미리 메모리에 로드 → 첫 명령 지연 제거."""
        if self.client is None:
            return
        try:
            self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ready"}],
                options={"num_predict": 1},
                keep_alive=self.keep_alive,
            )
            self._log(f"모델 로드 완료: {self.model}")
        except Exception as e:
            self._log(f"워밍업 실패({e}) — 첫 명령이 느릴 수 있음")

    def _validate_semantics(self, plan: CommanderPlan, state: BattlefieldState) -> None:
        _validate_routes(plan, state)


__all__ = ["OllamaCommander"]
