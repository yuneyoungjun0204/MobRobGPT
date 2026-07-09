"""복사해 온 boatattack_sim 시뮬레이터가 이 프로젝트에서 단독으로 도는지 검증(헤드리스).

아직 지휘관(LLM) 연동 전 — 시뮬 자체 AUTO 휴리스틱으로 1 에피소드를 끝까지 돌려
포획/breach/생존 통계를 출력한다. 오류 없이 통계가 나오면 이식 성공.

실행:
    python run_sim_headless.py
    python run_sim_headless.py wave     # 적 스폰 모드
"""
import sys
import numpy as np

from boatattack_sim.env.simulator import Simulator
from boatattack_sim.env.config import DEFAULT_CONFIG as C


def main() -> None:
    enemy_mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    sim = Simulator(enemy_mode=enemy_mode)
    sim.reset(seed=0)
    sim.running = True  # AUTO 모드(manual=False 기본) → 시뮬 휴리스틱이 자동 배치

    print(f"시작: 적 {int(sim.e_alive.sum())}척, 아군 {C.n_allies}척, "
          f"맵 {C.world_size:.0f}m, 최대 {C.max_steps} step")

    steps = 0
    while not sim.done:
        sim.step()
        steps += 1

    survived = int(sim.e_alive.sum())
    captured = 10 - survived  # n_enemies 기본 10 가정 (초기 생존 - 최종 생존)
    print(f"종료: {steps} step 진행")
    print(f"  최종 배정 assign = {sim.assign}")
    print(f"  생존 적(막지 못함) = {survived}")
    print(f"  stats = {sim.stats}")
    print("헤드리스 시뮬레이터 이식 검증 완료 [OK]")


if __name__ == "__main__":
    main()
