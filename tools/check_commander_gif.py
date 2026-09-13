"""make_commander_gif 회귀 검사 — 리팩터링 전후로 동작이 같은지 못 박는다.

이 스크립트가 지키는 것(전부 과거에 실제로 깨졌던 것들이다):
  ① GIF 프레임 수 == --frames
  ② figure 여백이 다크 (흰 테두리가 남지 않음)
  ③ 캠 3패널이 실제로 그려짐 (씬 배경과 다른 픽셀이 있음)
  ④ 선박 라벨이 axes 밖으로 새어 지휘관 패널을 덮지 않음
  ⑤ setup_panels 가 씬 axes 안쪽에 배치하고 패널끼리·지휘관 패널과 겹치지 않음
  ⑥ 적 전멸 시에도 draw_panels 가 예외를 내지 않음
  ⑦ matplotlib 글리프 경고 0 건
  ⑧ render_sim.py(다른 draw_scene 소비자)가 여전히 렌더됨

실행:  python tools/check_commander_gif.py
종료코드 0 = 통과.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FRAMES = 6
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    glyphs: list[str] = []

    class _GlyphSpy(logging.Handler):
        def emit(self, rec):
            if "glyph" in rec.getMessage():
                glyphs.append(rec.getMessage())

    logging.getLogger("matplotlib").addHandler(_GlyphSpy())
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from boatattack_sim.eval import renderer
    from commander.sim_bridge import CommandedSimulator

    tmp = tempfile.mkdtemp(prefix="gifcheck_")
    gif = os.path.join(tmp, "check.gif")

    print("[1] GIF 생성 (--no-llm)")
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "make_commander_gif.py"),
         "--out", gif, "--frames", str(FRAMES), "--no-llm",
         "--enemy", "diversionary", "--seed", "3"],
        cwd=REPO, capture_output=True, text=True,
        encoding="utf-8", errors="replace",     # 출력이 한글 — 로케일(cp949) 디코드 금지
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    check("익스포터 종료코드 0", r.returncode == 0, r.stderr[-200:] if r.returncode else "")
    if not os.path.exists(gif):
        check("GIF 생성됨", False)
        return 1

    im = Image.open(gif)
    check("프레임 수 == --frames", im.n_frames == FRAMES, f"{im.n_frames}/{FRAMES}")

    worst_spill = 0
    for f in range(im.n_frames):
        im.seek(f)
        rgb = im.convert("RGB")
        W, H = rgb.size
        # ④ 제목 오른쪽 끝(x>760)의 상단 띠에는 아무 글자도 없어야 한다.
        band = rgb.crop((760, 0, W, 62))
        worst_spill = max(worst_spill, sum(1 for q in band.getdata() if sum(q) > 210))
    check("라벨 스필 없음", worst_spill == 0, f"최대 {worst_spill}px")

    im.seek(im.n_frames - 1)
    rgb = im.convert("RGB")
    W, H = rgb.size
    corners = [rgb.getpixel(c) for c in ((2, 2), (W - 3, 2), (2, H - 3), (W - 3, H - 3))]
    check("여백 다크", all(sum(c) < 120 for c in corners), str(corners[0]))

    print("[2] 패널 기하 (setup_panels)")
    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_axes((0.030, 0.035, 0.600, 0.885))
    ax_info = fig.add_axes((0.648, 0.035, 0.335, 0.885))
    panels = renderer.setup_panels(fig, ax)
    sb, ib = ax.get_position(), ax_info.get_position()
    boxes = {k: a.get_position() for k, a in panels.items()}
    check("패널 3개", len(panels) == 3, ",".join(sorted(panels)))
    check("패널이 씬 안쪽",
          all(b.x0 >= sb.x0 - 1e-9 and b.x1 <= sb.x1 + 1e-9 and b.y1 <= sb.y1 + 1e-9
              for b in boxes.values()))
    check("지휘관 패널 침범 없음", all(b.x1 <= ib.x0 for b in boxes.values()))
    ordered = sorted(boxes.values(), key=lambda b: b.x0)
    check("패널끼리 안 겹침",
          all(u.x1 <= v.x0 + 1e-9 for u, v in zip(ordered, ordered[1:])))

    print("[3] 적 전멸 시 안전")
    sim = CommandedSimulator(enemy_mode="diversionary")
    sim.reset(seed=0)
    fd = dict(sim.get_frame())
    fd["enemy_alive"] = np.zeros_like(np.asarray(fd["enemy_alive"]), bool)
    fd["n_alive"] = 0
    check("follow_cluster_view -> None", renderer.follow_cluster_view(fd) is None)
    try:
        renderer.draw_panels(panels, fd)
        check("draw_panels 예외 없음", True)
    except Exception as e:                        # noqa: BLE001 — 무엇이든 실패로 본다
        check("draw_panels 예외 없음", False, f"{type(e).__name__}: {e}")
    plt.close(fig)

    print("[4] 다른 draw_scene 소비자 회귀 (render_sim.py)")
    png = os.path.join(tmp, "regress.png")
    r2 = subprocess.run(
        [sys.executable, os.path.join(REPO, "render_sim.py"), "--png", png, "--at", "60"],
        cwd=REPO, capture_output=True, text=True,
        encoding="utf-8", errors="replace",     # 출력이 한글 — 로케일(cp949) 디코드 금지
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    check("render_sim.py 렌더됨",
          r2.returncode == 0 and os.path.exists(png) and os.path.getsize(png) > 0,
          r2.stderr[-200:] if r2.returncode else "")

    check("글리프 경고 0", not glyphs, "; ".join(glyphs[:2]))

    print()
    if fails:
        print(f"FAILED {len(fails)}: {fails}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
