# -*- coding: utf-8 -*-
"""지형 6해역 실행 스냅샷을 2x3 타일 한 장으로 — 논문 부록용.

왜 만드는가
    PPT_그림/overpass_map.png 가 같은 내용이지만 360x250 px 라 인쇄가 안 된다.
    지형별 데모 GIF(논문_그래프/6_영상/*_<해역>_zoom.gif)는 1350x900 이므로 거기서
    같은 진행률의 프레임을 뽑아 씬 영역만 잘라 붙이면 고해상도로 다시 만들 수 있다.

    씬만 자르는 이유: 우측 지휘관 패널과 상단 제목은 6장 모두 같아 정보가 없고,
    해역 비교에서는 배경 지형과 그물·경로만 보이면 된다.

사용:
    python tools/make_sites_tile.py                      # 기본: 진행률 0.6 프레임
    python tools/make_sites_tile.py --frac 0.75 --out 논문초안/figs_snak/fig_sites6_run.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(REPO, "논문_그래프", "6_영상")

#: (해역 이름, 라벨) — terrain.KOREA_SITES 순서. 라벨의 육지 비율은 같은 파일 주석값이다.
SITES = (
    ("통영_매물도서", "통영 매물도 서측 · 육지 2.24 %"),
    ("통영_연화도서", "통영 연화도 서측 · 육지 7.47 %"),
    ("신안_홍도북",   "신안 홍도 북측 · 육지 3.16 %"),
    ("거제_매물도동", "거제 매물도 동측 · 육지 2.52 %"),
    ("통영_매물도북", "통영 매물도 북측 · 육지 6.62 %"),
    ("포항_호미곶동", "포항 호미곶 동측 · 육지 0 % (외해)"),
)
#: make_commander_gif.py 레이아웃의 씬 axes: fig 좌표 (0.030, 0.035, 0.600, 0.885).
#  1350x900 기준 픽셀로 환산해 자른다. 상단 캠 3패널은 포함한다(해역별 확대 캠도 정보다).
SCENE_BOX = (0.030, 0.035, 0.630, 0.920)   # x0, y0, x1, y1 (fig 좌표, y 는 아래가 0)


def find_gif(site: str) -> str:
    hits = sorted(glob.glob(os.path.join(VIDEO_DIR, f"*_{site}_zoom.gif")))
    if not hits:
        raise SystemExit(f"[tile] {site} 영상 없음: {VIDEO_DIR}")
    return hits[-1]


def crop_scene(img):
    W, H = img.size
    x0, y0, x1, y1 = SCENE_BOX
    return img.crop((int(x0 * W), int((1 - y1) * H), int(x1 * W), int((1 - y0) * H)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="지형 6해역 실행 스냅샷 2x3 타일")
    ap.add_argument("--frac", type=float, default=0.60,
                    help="추출할 프레임의 진행률(0~1). 0.6 = 그물이 펴지고 포획이 시작된 시점")
    ap.add_argument("--out", default=os.path.join(REPO, "논문초안", "figs_snak",
                                                  "fig_sites6_run.png"))
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--pad", type=int, default=14)
    ap.add_argument("--label-h", type=int, default=44)
    args = ap.parse_args(argv)

    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    for site, label in SITES:
        path = find_gif(site)
        im = Image.open(path)
        n = im.n_frames
        im.seek(max(0, min(n - 1, int(n * args.frac))))
        tiles.append((crop_scene(im.convert("RGB")), label, os.path.basename(path)))
        print(f"[tile] {site:<14s} {os.path.basename(path)}  frame {int(n * args.frac)}/{n}")

    tw, th = tiles[0][0].size
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    pad, lh = args.pad, args.label_h
    W = cols * tw + (cols + 1) * pad
    H = rows * (th + lh) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (10, 18, 31))
    draw = ImageDraw.Draw(canvas)

    font = None
    for cand in (r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"):
        if os.path.exists(cand):
            font = ImageFont.truetype(cand, 26)
            break
    if font is None:
        font = ImageFont.load_default()

    for i, (tile, label, _) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + lh + pad)
        draw.text((x + 8, y + 8), f"({chr(ord('a') + i)})  {label}",
                  fill=(236, 239, 244), font=font)
        canvas.paste(tile, (x, y + lh))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out, dpi=(200, 200))
    print(f"[tile] 저장: {args.out}  {canvas.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
