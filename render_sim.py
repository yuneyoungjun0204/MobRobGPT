"""복사해 온 boatattack_sim 시뮬레이터 시각화.

아직 지휘관(LLM) 연동 전 — 시뮬 자체 AUTO 휴리스틱 에피소드를 그림으로 본다.
배경 지도(basemap) 없이 해색 기본 배경으로 렌더(오프라인 OK).

실행:
    python render_sim.py                 # 라이브 창 (애니메이션)
    python render_sim.py --enemy wave    # 적 스폰 모드
    python render_sim.py --gif out.gif   # GIF 파일로 저장 (창 안 뜨는 환경/공유용)
    python render_sim.py --png shot.png   # 단일 프레임 PNG (기본 200 step 시점)
    python render_sim.py --png shot.png --at 100
"""
import sys


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


def main() -> None:
    enemy = _arg("--enemy", "random")
    gif_path = _arg("--gif")
    png_path = _arg("--png")
    at_step = int(_arg("--at", "200"))

    import matplotlib
    if gif_path or png_path:
        matplotlib.use("Agg")               # 헤드리스(파일 저장) 백엔드
    import matplotlib.pyplot as plt

    from boatattack_sim.env.simulator import Simulator
    from boatattack_sim.eval import renderer

    sim = Simulator(enemy_mode=enemy)
    sim.reset(seed=0)
    sim.running = True                       # AUTO 모드 자동 진행

    fig, ax = plt.subplots(figsize=(8.5, 9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06)

    # ── 단일 PNG ───────────────────────────────────────────────
    if png_path:
        for _ in range(at_step):
            if sim.done:
                break
            sim.step()
        renderer.draw_scene(ax, sim.get_frame())
        fig.savefig(png_path, dpi=110)
        print(f"PNG 저장: {png_path} ({sim.t} step 시점)")
        return

    # ── GIF ────────────────────────────────────────────────────
    if gif_path:
        from matplotlib.animation import FuncAnimation, PillowWriter
        max_frames = int(_arg("--frames", "700"))

        def update(_):
            if sim.running and not sim.done:
                sim.step()
            renderer.draw_scene(ax, sim.get_frame())
            return []

        anim = FuncAnimation(fig, update, frames=max_frames,
                             interval=40, blit=False, cache_frame_data=False)
        anim.save(gif_path, writer=PillowWriter(fps=25))
        print(f"GIF 저장: {gif_path} (stats={sim.stats})")
        return

    # ── 라이브 창 ──────────────────────────────────────────────
    from matplotlib.animation import FuncAnimation

    def update(_):
        if sim.running and not sim.done:
            sim.step()
        renderer.draw_scene(ax, sim.get_frame())
        return []

    _anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    print("라이브 창 실행 — 창을 닫으면 종료. (적 스폰: %s)" % enemy)
    plt.show()


if __name__ == "__main__":
    main()
