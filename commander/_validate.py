"""경로 계획 의미 검증 (스키마 통과 후) — 두 어댑터 공용.

가벼운 검증만: 실존 아군 · 중복 없음 · 경로 비어있지 않음.
WP 개수(6개)나 좌표 범위는 여기서 실패시키지 않는다(브릿지가 클램프) → 폴백 남발 방지.
"""
from .schema import BattlefieldState, CommanderPlan


def _validate_routes(plan: CommanderPlan, state: BattlefieldState) -> None:
    ally_ids = {a.id for a in state.allies}
    seen = set()
    for r in plan.routes:
        if r.ally_id not in ally_ids:
            raise ValueError(f"존재하지 않는 아군 id: {r.ally_id}")
        if r.ally_id in seen:
            raise ValueError(f"아군 중복 경로: {r.ally_id}")
        seen.add(r.ally_id)
        if not r.waypoints:
            raise ValueError(f"빈 경로 (ally {r.ally_id})")
    # 빈 routes(전원 예비)도 유효.
