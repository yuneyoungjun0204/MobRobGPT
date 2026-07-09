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

    # 실제 40m 시뮬 상태로 진단 (qwen 이 WP 를 링까지 보내는지 확인)
    sim = CommandedSimulator(enemy_mode="random"); sim.reset(seed=0)
    bf = build_battlefield(sim, command="모든 적군 포획")
    cx, cy = bf.mothership.pos.x, bf.mothership.pos.y
    R = bf.constraints.max_intercept_radius
    print(f"[진단] world={bf.constraints.world_size:.0f}m center=({cx:.1f},{cy:.1f}) "
          f"reachable_radius R={R:.2f} (그물 링 권장 0.6~0.85R = {0.6*R:.2f}~{0.85*R:.2f})")
    plan = cmd.plan(bf)
    tag = "실패" if (not plan.routes) and ("실패" in plan.rationale or "미연결" in plan.rationale) else "LLM "
    print(f"[{tag}] routes={len(plan.routes)}척")
    for r in plan.routes:
        radii = [math.hypot(w.x - cx, w.y - cy) for w in r.waypoints]
        nets = sum(1 for w in r.waypoints if w.deploy_net)
        print(f"  ally {r.ally_id}: WP 중심거리={[round(v,2) for v in radii]}  최대={max(radii):.2f}(R={R:.2f}) 그물{nets}구간")
        print(f"     좌표={[(round(w.x,1),round(w.y,1)) for w in r.waypoints]}")
    print(f"  rationale: {plan.rationale}")
    print("\n※ WP 중심거리 최대가 R 의 0.5 이상이면 링 도달(정상), 계속 작으면 '자기 앞에만'(문제).")
    return
    print(f"백엔드={type(cmd).__name__}  model={cmd.model}")
    print("전장: C0(정면, 6척)  C1(우, 2척)  C2(좌, 2척)  아군 3척\n")

    commands = [
        "정면(C0) 밀집 무리에 3척 모두 집중",
        "오른쪽(C1)만 1척으로 막고 나머지는 예비로 남겨",
        "세 무리에 1척씩 고르게 분산",
    ]
    for c in commands:
        bf = make_state(c)
        plan = cmd.plan(bf)
        summ = [(r.ally_id, len(r.waypoints), sum(1 for w in r.waypoints if w.deploy_net))
                for r in plan.routes]
        fail = (not plan.routes) and ("실패" in plan.rationale or "미연결" in plan.rationale)
        tag = "실패" if fail else "LLM "
        print(f"[{tag}] 명령: {c}")
        print(f"      routes (ally, #WP, #net구간) = {summ}")
        print(f"      rationale: {plan.rationale}\n")


if __name__ == "__main__":
    main()
