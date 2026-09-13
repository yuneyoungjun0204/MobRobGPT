# -*- coding: utf-8 -*-
"""도판이 판면을 얼마나 쓰는지 잰다.

왜 절대 mm 로 재면 안 되나
    판면 폭이 판마다 다르다 --- elsarticle preprint 는 390pt(137mm), 게재본(5p)의
    양단 걸침은 522pt(184mm)다. "150mm 이상" 같은 절대 기준을 쓰면 preprint 에서는
    **달성 불가능한 조건**이 되어 거짓 실패가 난다(실제로 그렇게 잘못 판정했다).
    기준은 "판면 폭의 몇 %를 쓰는가" 여야 한다.

사용:
    python tools/check_figure_widths.py                       # 제출본
    python tools/check_figure_widths.py --pdf <경로> --min 0.9

종료 코드 0 = 전부 통과.
"""
from __future__ import annotations

import argparse
import sys

import fitz

#: 캡션의 앞부분으로 도판을 찾는다. (조각, 판면 대비 최소 폭)
#  본문 폭을 꽉 채워야 하는 것들만 검사한다 --- 작게 실어도 되는 삽화는 대상이 아니다.
TARGETS = (
    ("15채널 관측의 구성", 0.90),
    ("래스터화된 다채널 관측의 채널별 분해", 0.90),
)


def text_width_mm(page: fitz.Page) -> float:
    """본문 블록의 가로 폭. 여백을 뺀 실제 판면을 본문 텍스트에서 추정한다."""
    xs = [b[0] for b in page.get_text("blocks")] + [b[2] for b in page.get_text("blocks")]
    return (max(xs) - min(xs)) / 72 * 25.4 if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf",
                    default="논문초안/build/USV_swarm_defense_EAAI_submission.pdf")
    ap.add_argument("--min", type=float, default=None,
                    help="판면 대비 최소 폭 (기본: 도판별 기준)")
    a = ap.parse_args()

    d = fitz.open(a.pdf)
    print(f"{a.pdf}  ({d.page_count}쪽)\n")
    print(f"{'도판':<34}{'쪽':>4}{'폭 mm':>8}{'판면 mm':>9}{'비율':>7}  판정")
    fail = 0
    for kw, need in TARGETS:
        need = a.min if a.min is not None else need
        key = kw.replace(" ", "")
        for i in range(d.page_count):
            if key not in d[i].get_text().replace(" ", "").replace("\n", ""):
                continue
            boxes = [b["bbox"] for b in d[i].get_image_info()]
            w = max(((b[2] - b[0]) / 72 * 25.4 for b in boxes), default=0.0)
            tw = text_width_mm(d[i])
            r = w / tw if tw else 0.0
            okk = r >= need
            fail += (not okk)
            print(f"{kw:<34}{i + 1:>4}{w:>8.0f}{tw:>9.0f}{r:>7.2f}  "
                  f"{'ok' if okk else f'!! 판면의 {need:.0%} 이상이어야 한다'}")
            break
        else:
            print(f"{kw:<34}{'?':>4}{'?':>8}{'?':>9}{'?':>7}  못 찾음")
            fail += 1
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
