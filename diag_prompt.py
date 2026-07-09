"""프롬프트 진단: 같은 전장에 서로 다른 명령을 주고 결과가 바뀌는지 확인.

- 각 명령마다 [LLM] 인지 [폴백] 인지 표시 (폴백이면 rationale 이 '[휴리스틱'으로 시작).
- 모두 [폴백] → LLM 파싱 실패(스키마 문제). 콘솔의 '[commander] LLM 배정 실패(...)' 사유 확인.
- [LLM] 인데 deployments 가 명령마다 똑같음 → 모델이 명령을 안 따름(프롬프트/모델 문제).

실행:
    python diag_prompt.py                 # Ollama qwen2.5:14b
    python diag_prompt.py qwen2.5:7b
    python diag_prompt.py --openai
"""
import sys

from commander.schema import (
    BattlefieldState, Mothership, EnemyCluster, AllyShip, Constraints, Point,
)
from commander import make_commander


def make_state(command):
    return BattlefieldState(
        mothership=Mothership(pos=Point(x=6300, y=6300), radius=400, threat_level=0.6),
        enemy_clusters=[
            EnemyCluster(id=0, center=Point(x=6300, y=2500), bearing=0, spread=15, count=6, approach_speed=9),
            EnemyCluster(id=1, center=Point(x=9000, y=6300), bearing=90, spread=8, count=2, approach_speed=9),
            EnemyCluster(id=2, center=Point(x=3000, y=6300), bearing=270, spread=8, count=2, approach_speed=9),
        ],
        allies=[AllyShip(id=i, pos=Point(x=6300 + (i - 1) * 300, y=6600),
                         heading=180, nets_remaining=3) for i in range(3)],
        constraints=Constraints(net_max_len=450, ally_speed=6, enemy_speed=12,
                                world_size=12600, max_intercept_radius=2100.0),
        command=command,
    )


def main():
    import math
    from commander.sim_bridge import CommandedSimulator, build_battlefield
    backend = "openai" if "--openai" in sys.argv else "ollama"
    model = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    cmd = make_commander(backend, model)

    # 실제 40m 시뮬 상태로 진단 (LLM 배정 → 아군별 담당 클러스터)
    from commander.sim_bridge import plan_to_assign
    sim = CommandedSimulator(enemy_mode="random"); sim.reset(seed=0)
    bf = build_battlefield(sim, command="모든 적군 포획")
    R = bf.constraints.max_intercept_radius
    print(f"[진단] world={bf.constraints.world_size:.0f}m "
          f"클러스터(id,척수)={[(c.id, c.count) for c in bf.enemy_clusters]} R={R:.2f}")
    plan = cmd.plan(bf)
    fail = (not plan.deployments) and ("실패" in plan.rationale or "미연결" in plan.rationale)
    tag = "실패" if fail else "LLM "
    dep = [(d.cluster_id, d.n_ships) for d in plan.deployments]
    assign = plan_to_assign(plan, bf)
    print(f"[{tag}] deployments(클러스터,척수)={dep}")
    print(f"       assign(아군→클러스터)={assign.tolist()}  (경로·수직그물은 시뮬이 기하로 생성)")
    print(f"  rationale: {plan.rationale}")
    print("\n※ 배정이 위협 큰 클러스터에 척수를 몰면 정상. 경로/그물은 시뮬이 링에 수직으로 깐다.")


if __name__ == "__main__":
    main()
