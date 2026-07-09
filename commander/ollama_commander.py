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
from ._validate import sanitize_plan
from .fallback import heuristic_plan


def _fallback(state: BattlefieldState, reason: str) -> CommanderPlan:
    """LLM 실패 시 전원 정지 대신 위협비례 휴리스틱으로 방어(전 클러스터 커버). 이유 병기."""
    plan = heuristic_plan(state)
    plan.rationale = f"{reason} → 휴리스틱 방어. {plan.rationale}"
    return plan


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
            return _fallback(state, "Ollama 미연결")
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
            plan = sanitize_plan(plan, state)       # 거부 대신 정제(중복 ally/무효 id 수리)
            self._log(f"LLM 배정 성공 ({self.model}): {len(plan.deployments)}개 클러스터 배분")
            return plan
        except Exception as e:
            self._log(f"LLM 배정 실패({type(e).__name__}: {e}) → 휴리스틱 방어")
            return _fallback(state, f"LLM 실패({type(e).__name__})")

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


__all__ = ["OllamaCommander"]
