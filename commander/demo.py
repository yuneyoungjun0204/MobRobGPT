"""지휘관 데모/스모크 테스트.

집중공격(한 클러스터에 적 다수) 전장을 만들어 지휘관을 돌린다.
- Ollama + qwen2.5:14b 가 있으면 LLM 배정, 없으면 자동으로 휴리스틱 폴백.
실행:
    python -m commander.demo                 # 기본 qwen2.5:14b
    python -m commander.demo llama3.1:8b      # 다른 로컬 모델
    python -m commander.demo --fallback       # LLM 건너뛰고 폴백만 확인
"""
import sys

from .schema import (
    BattlefieldState, Mothership, EnemyCluster, AllyShip, Constraints, Point,
)
from .fallback import heuristic_plan
from . import make_commander


def sample_state() -> BattlefieldState:
    """집중공격 시나리오: C0에 8척 몰림, C1에 2척. 아군 3척."""
    return BattlefieldState(
        mothership=Mothership(pos=Point(x=5000, y=5000), radius=400, threat_level=0.7),
        enemy_clusters=[
            EnemyCluster(id=0, center=Point(x=5200, y=2500), bearing=-12, spread=18,
                         count=8, approach_speed=8.0),
            EnemyCluster(id=1, center=Point(x=8000, y=6000), bearing=65, spread=8,
                         count=2, approach_speed=7.5),
        ],
        allies=[
            AllyShip(id=0, pos=Point(x=4800, y=5400), heading=180, nets_remaining=3),
            AllyShip(id=1, pos=Point(x=5000, y=5400), heading=180, nets_remaining=3),
            AllyShip(id=2, pos=Point(x=5200, y=5400), heading=180, nets_remaining=3),
        ],
        constraints=Constraints(net_max_len=450, ally_speed=6.0, enemy_speed=12.0,
                                world_size=10000, max_intercept_radius=1670.0),
        command="정면 밀집 무리를 우선 차단하라.",
    )


def print_plan(plan, state) -> None:
    from .sim_bridge import plan_to_assign
    print("\n=== CommanderPlan (클러스터별 배정 USV) ===")
    for d in plan.deployments:
        who = d.ally_ids if d.ally_ids else "자동선택(효율/안전)"
        print(f"  cluster {d.cluster_id}: USV {who}  net={'Y' if d.deploy_net else '-'}")
    if plan.hold_ships:
        print(f"  HOLD(정지): USV {plan.hold_ships}")
    assign = plan_to_assign(plan, state)
    print(f"  → 아군별 배정 assign = {assign.tolist()}  (경로·그물은 시뮬이 기하로 생성)")
    print(f"  rationale: {plan.rationale}")


def main() -> None:
    args = sys.argv[1:]
    state = sample_state()

    if "--fallback" in args:
        print(">> 휴리스틱 폴백만 실행")
        print_plan(heuristic_plan(state), state)
        return

    backend = "openai" if "--openai" in args else "ollama"
    model = next((a for a in args if not a.startswith("-")), None)
    cmd = make_commander(backend, model)
    print(f">> {type(cmd).__name__}(model={cmd.model}) 실행 (실패 시 자동 폴백)")
    print_plan(cmd.plan(state), state)


if __name__ == "__main__":
    main()
