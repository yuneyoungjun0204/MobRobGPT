"""LLM 지휘관 모듈 (벤더 독립).

전장상태 → 지휘관 → 3척 배정계획. 기본 백엔드는 Ollama(로컬/오프라인).
어댑터만 교체하면 Claude/Gemini/OpenAI 로 바꿀 수 있다(동일 스키마 계약).

사용:
    from commander import OllamaCommander, BattlefieldState
    cmd = OllamaCommander(model="qwen2.5:14b")
    plan = cmd.plan(state)   # -> CommanderPlan (항상 유효, 실패 시 휴리스틱 폴백)
"""
from .schema import (
    Point, Mothership, EnemyCluster, AllyShip, Constraints,
    BattlefieldState, ClusterDeployment, CommanderPlan,
)
from .fallback import heuristic_plan
from .ollama_commander import OllamaCommander
from .openai_commander import OpenAICommander


def make_commander(backend: str = "ollama", model: str | None = None, **kwargs):
    """백엔드 팩토리 — 같은 plan(state)->CommanderPlan 인터페이스.

    backend="ollama" (기본, 로컬/오프라인) | "openai" (GPT API, OPENAI_API_KEY 필요)
    model 미지정 시 백엔드별 기본값 사용.
    """
    b = backend.lower()
    if b in ("openai", "gpt"):
        return OpenAICommander(model=model or "gpt-4o-mini", **kwargs)
    if b == "ollama":
        return OllamaCommander(model=model or "qwen2.5:7b", **kwargs)
    raise ValueError(f"알 수 없는 backend: {backend} (ollama|openai)")


__all__ = [
    "Point", "Mothership", "EnemyCluster", "AllyShip", "Constraints",
    "BattlefieldState", "ClusterDeployment", "CommanderPlan",
    "heuristic_plan", "OllamaCommander", "OpenAICommander", "make_commander",
]
