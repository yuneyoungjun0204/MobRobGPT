"""논문_그래프/ 의 그림·영상을 주제별 하위 폴더로 정리한다.

분류표는 boatattack_sim.eval.plots.FIG_GROUPS 하나뿐이다 — save_all 이 새로 저장할 때도
같은 표를 쓰므로, 재생성해도 정리가 흐트러지지 않는다. 여기서는 **이미 평평하게 쌓여
있던 기존 파일**만 옮긴다(1회성 이관).

안전장치
    · 같은 이름이 목적지에 이미 있고 내용이 다르면 옮기지 않고 보고만 한다.
    · --dry-run 으로 무엇이 어디로 갈지 먼저 볼 수 있다.
    · 하위 폴더 안에 이미 있는 파일은 건드리지 않는다(재실행해도 안전).

사용:
    python tools/organize_figures.py --dry-run
    python tools/organize_figures.py
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIGDIR = os.path.join(REPO, "논문_그래프")
#: 분류하지 않고 최상위에 두는 파일 — 색인이라 눈에 바로 보여야 한다.
KEEP_TOP = {"README.md"}
VIDEO_EXT = {".gif", ".mp4", ".webm"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="논문_그래프/ 를 주제별 폴더로 정리")
    ap.add_argument("--dry-run", action="store_true", help="옮기지 않고 계획만 출력")
    ap.add_argument("--figdir", default=FIGDIR)
    args = ap.parse_args(argv)

    from boatattack_sim.eval.plots import FIG_GROUP_VIDEO, fig_group

    figdir = os.path.abspath(args.figdir)
    if not os.path.isdir(figdir):
        print(f"[organize] 폴더 없음: {figdir}")
        return 1

    plan: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(figdir)):
        src = os.path.join(figdir, entry)
        if os.path.isdir(src) or entry in KEEP_TOP or entry.startswith("."):
            continue
        stem, ext = os.path.splitext(entry)
        grp = FIG_GROUP_VIDEO if ext.lower() in VIDEO_EXT else fig_group(stem)
        plan.append((src, os.path.join(figdir, grp, entry)))

    if not plan:
        print("[organize] 최상위에 정리할 파일이 없다 — 이미 정돈된 상태")
        return 0

    by_group: dict[str, int] = {}
    skipped: list[str] = []
    for src, dst in plan:
        grp = os.path.basename(os.path.dirname(dst))
        by_group[grp] = by_group.get(grp, 0) + 1

    print(f"[organize] {len(plan)}개 파일 → {len(by_group)}개 폴더")
    for grp in sorted(by_group):
        print(f"    {grp:<12s} {by_group[grp]:3d}개")

    if args.dry_run:
        print("\n[organize] --dry-run: 아무것도 옮기지 않았다")
        for src, dst in plan[:10]:
            print(f"    {os.path.basename(src)} -> {os.path.relpath(dst, figdir)}")
        if len(plan) > 10:
            print(f"    … 외 {len(plan) - 10}개")
        return 0

    moved = 0
    for src, dst in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            # 같은 내용이면 원본만 지우고, 다르면 손대지 않고 보고한다.
            if filecmp.cmp(src, dst, shallow=False):
                os.remove(src)
            else:
                skipped.append(os.path.basename(src))
            continue
        shutil.move(src, dst)
        moved += 1

    print(f"\n[organize] 이동 {moved}개")
    if skipped:
        print(f"[organize] ! 목적지에 다른 내용의 동명 파일이 있어 건너뜀 {len(skipped)}개:")
        for n in skipped:
            print(f"      {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
