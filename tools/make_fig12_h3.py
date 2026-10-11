# -*- coding: utf-8 -*-
"""fig12.png(수렴 진단 2x2)에서 (b) 엔트로피 · (c) 근사 KL · (d) 파라미터 이동 세 패널만 잘라
가로 1x3 으로 다시 붙인 fig12_h3.png 를 만든다. 원본 fig12.png 는 건드리지 않는다.

원자료(results/train_diag/seed*)가 없는 PC 에서도 만들 수 있도록 600 dpi 래스터를 잘라 쓴다.
원자료가 있으면 boatattack_sim/eval/paper_figs_refstyle.py 로 다시 그리는 편이 낫다.
패널 제목은 (a)(b)(c) 로 다시 매겨 원본과 같은 글꼴(Times)·크기로 새로 쓴다.

  python tools/make_fig12_h3.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "논문초안", "figs_snak", "real")
SRC = os.path.join(ROOT, "fig12.png")
DST = os.path.join(ROOT, "fig12_h3.png")

INK = 200          # 이보다 어두우면 잉크로 본다
SPLIT_X = 1580     # 원본 좌·우 열 경계
ROW1 = (12, 600)   # 1행: 제목 시작 ~ x 라벨 끝 아래
ROW2 = (646, 1237) # 2행
TITLE_H = 50       # 원본 제목 띠 높이(px) — '(' 의 위아래 끝
PAD = 36           # 패널 사이·바깥 여백(px)
OVERHANG = 25      # 축 상단 눈금 라벨이 프레임 위로 튀어나오는 여유

PANELS = [  # (열, 행, 새 제목)
    ("R", ROW1, "(a) Policy entropy"),
    ("L", ROW2, "(b) Approx. KL per update"),
    ("R", ROW2, "(c) Relative parameter shift"),
]


def _frame_top(dark: np.ndarray, start: int) -> int:
    """제목 아래에서 처음으로 폭의 절반 이상이 잉크인 행 = 축 프레임 윗변."""
    w = dark.shape[1]
    for y in range(start, dark.shape[0]):
        if dark[y].sum() > 0.5 * w:
            return y
    raise RuntimeError("frame not found")


def _cut(gray: np.ndarray, rgb: np.ndarray, col: str, row: tuple[int, int]):
    """한 패널을 잘라 (컬러 조각, 조각 안 축 왼끝 x, 축 오른끝 x) 를 돌려준다. 제목 띠는 버린다."""
    x0, x1 = (0, SPLIT_X) if col == "L" else (SPLIT_X, gray.shape[1])
    y0, y1 = row
    dark = gray[y0:y1, x0:x1] < INK
    top = _frame_top(dark, TITLE_H)
    # 프레임 윗변 행에서 잉크가 이어진 구간 = 축의 가로 범위(제목 가운데 맞춤 기준)
    # (같은 행의 눈금 라벨이 섞이지 않게 가장 긴 연속 구간만 쓴다)
    fx = np.flatnonzero(dark[top])
    runs = np.split(fx, np.flatnonzero(np.diff(fx) > 1) + 1)
    run = max(runs, key=len)
    fl, fr = int(run[0]), int(run[-1])
    # 여유 띠 안에서 축 가로 범위에 걸친 잉크는 옛 제목의 아랫부분이다 --- 지운다
    dark = dark.copy()
    dark[top - OVERHANG:top, fl:fr + 1] = False
    d = dark[top - OVERHANG:]
    cols = np.flatnonzero(d.any(0))
    rows = np.flatnonzero(d.any(1))
    l, r, b = int(cols.min()), int(cols.max()) + 1, int(rows.max()) + 1
    piece = rgb[y0 + top - OVERHANG: y0 + top - OVERHANG + b, x0 + l: x0 + r].copy()
    piece[:OVERHANG, fl - l:fr - l + 1] = 255
    return Image.fromarray(piece), fl - l, fr - l


def _font(px: int) -> ImageFont.FreeTypeFont:
    path = r"C:\Windows\Fonts\times.ttf"
    # '(' 의 높이가 원본 제목 띠(TITLE_H)와 같아지는 크기를 찾는다
    for size in range(20, 120):
        f = ImageFont.truetype(path, size)
        bb = f.getbbox("(")
        if bb[3] - bb[1] >= px:
            return f
    return ImageFont.truetype(path, 60)


def main() -> None:
    src = Image.open(SRC).convert("RGB")
    rgb = np.array(src)
    gray = np.array(src.convert("L"))
    pieces = [(*_cut(gray, rgb, col, row), title) for col, row, title in PANELS]

    font = _font(TITLE_H)
    title_band = PAD // 3 + TITLE_H + 14
    H = title_band + max(p[0].height for p in pieces) + PAD // 2
    W = PAD + sum(p[0].width for p in pieces) + PAD * (len(pieces) - 1) + PAD
    out = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(out)
    x = PAD
    for im, ax_l, ax_r, title in pieces:
        out.paste(im, (x, title_band - OVERHANG))
        cx = x + (ax_l + ax_r) / 2
        tw = draw.textlength(title, font=font)
        bb = font.getbbox("(")
        draw.text((cx - tw / 2, PAD // 3 - bb[1]), title, font=font, fill="black")
        x += im.width + PAD
    out.save(DST, dpi=(600, 600))
    print("[ok]", DST, out.size)


if __name__ == "__main__":
    main()
