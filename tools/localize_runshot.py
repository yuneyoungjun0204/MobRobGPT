# -*- coding: utf-8 -*-
"""실행 화면 도판(fig:runshot)의 RATIONALE 블록만 영어로 바꿔 다시 굽는다.

논문 도판 fig14.png 는 make_commander_gif.py 가 만든 GIF 의 한 프레임을 잘라낸 것인데,
Gemini 지휘관의 rationale 이 한국어(commander/prompts.py 가 강제)라 영문 논문에 못 쓴다.
같은 장면을 다시 돌리면 LLM 출력이 달라져 배정·궤적이 바뀌므로, 프레임은 그대로 두고
패널의 rationale 텍스트 블록만 익스포터와 같은 글꼴·크기·색·줄간격으로 다시 그린다.

    python tools/localize_runshot.py \
        --gif 논문_그래프/6_영상/commander_unet_gemini-3.5-flash-lite_diversionary_seed1_replan25_zoom.gif \
        --frame 386 --crop 36 16 1296 858 \
        --text "In response to the diversionary ..." \
        --out 논문초안/figs_snak/real/fig14.png

블록 위치·줄 pitch 는 원본 프레임의 한글 잉크 행을 재서 잡으므로 다른 프레임에도 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import io
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

# make_commander_gif.py 와 같은 값 — 여기서 바꾸면 패널과 색이 어긋난다.
BG_PANEL = "#0d1b2a"
FG = "#ECEFF4"
FONT_PT = 7.8            # draw_commander_panel 의 rationale 글자 크기
ASCII_WRAP = 42          # _wrap_width(ascii_w)
MAX_LINES = 7
DPI = 100


def _frame(gif: Path, idx: int, crop: tuple[int, int, int, int] | None) -> Image.Image:
    g = Image.open(gif)
    g.seek(idx)
    im = g.convert("RGB")
    if crop:
        x, y, w, h = crop
        im = im.crop((x, y, x + w, y + h))
    return im


def _ink_rows(a: np.ndarray, bg: np.ndarray, x0: int, x1: int, y0: int, y1: int, thr: int = 60):
    """[y0,y1) 구간에서 배경과 다른 픽셀이 있는 행들을 (시작, 끝) 런으로 돌려준다."""
    d = np.abs(a[y0:y1, x0:x1].astype(int) - bg).sum(axis=2) > thr
    rows = d.any(axis=1)
    runs, start = [], None
    for i, r in enumerate(rows):
        if r and start is None:
            start = i
        elif not r and start is not None:
            runs.append((y0 + start, y0 + i - 1)); start = None
    if start is not None:
        runs.append((y0 + start, y1 - 1))
    return runs


def _locate_block(a: np.ndarray, bg: np.ndarray, x0: int, x1: int):
    """'RATIONALE' 헤더 다음 잉크 행부터 'ORDER LOG' 헤더 앞까지를 rationale 블록으로 본다.

    헤더는 굵은 청록(ACCENT)이라 흰 본문과 색으로 갈린다: 행 평균에서 G≈B>R 인 행이 헤더.
    """
    runs = _ink_rows(a, bg, x0, x1, 0, a.shape[0])

    def is_accent(run):
        y0, y1 = run
        sub = a[y0:y1 + 1, x0:x1].reshape(-1, 3).astype(int)
        m = np.abs(sub - bg).sum(axis=1) > 150
        if not m.any():
            return False
        c = sub[m].mean(axis=0)
        return c[1] > c[0] + 40 and c[2] > c[0] + 40          # 청록 계열

    accent_idx = [i for i, r in enumerate(runs) if is_accent(r)]
    # 패널 헤더 순서: LLM COMMANDER, COMMAND, DEPLOYMENT, (deploy 줄은 초록/흰) RATIONALE, ORDER LOG
    # 뒤에서 두 번째 accent 행이 RATIONALE, 마지막이 ORDER LOG 가 되도록 로그 [OK] 초록행은 걸러낸다.
    hdr = [i for i in accent_idx if (runs[i][1] - runs[i][0]) >= 6]
    if len(hdr) < 2:
        raise SystemExit("패널 헤더(RATIONALE / ORDER LOG)를 찾지 못했다 — --crop 이 오른쪽 패널을 담는지 확인")
    rat_i, log_i = hdr[-2], hdr[-1]
    body = runs[rat_i + 1:log_i]
    return runs[rat_i], runs[log_i], body


def localize(gif: Path, frame: int, crop, text: str, out: Path, keep_ko: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    names = {f.name for f in fm.fontManager.ttflist}
    kf = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in names), None)
    matplotlib.rcParams["font.family"] = [kf, "DejaVu Sans"] if kf else ["DejaVu Sans"]

    base = _frame(gif, frame, crop)
    if keep_ko and not keep_ko.exists():
        base.save(keep_ko)
    a = np.asarray(base)
    H, W = a.shape[:2]

    # 패널 배경색은 상수 대신 프레임에서 뽑는다 — GIF 팔레트 양자화로 미세하게 다를 수 있다.
    bg = np.array(Image.new("RGB", (1, 1), BG_PANEL)).reshape(3).astype(int)
    px = a[int(H * 0.56), int(W * 0.85)].astype(int)
    if np.abs(px - bg).sum() < 30:
        bg = px
    # 오른쪽 패널의 가로 위치는 익스포터 레이아웃(setup_panels) 기준 비율이다.
    # 다른 크롭·해상도의 프레임에 쓰면 아래 출력의 "x=" 가 패널 텍스트 시작과 맞는지 먼저 봐라.
    panel_x0 = int(W * 0.655)
    panel_x1 = W - 8
    rat_hdr, log_hdr, body = _locate_block(a, bg, panel_x0, panel_x1)
    if not body:
        raise SystemExit("rationale 본문 행을 찾지 못했다 — --crop 이 패널을 포함하는지 확인")
    pitch = (body[-1][0] - body[0][0]) / max(len(body) - 1, 1)
    text_x = panel_x0 + int(np.argmax((np.abs(a[body[0][0]:body[0][1] + 1, panel_x0:panel_x1].astype(int) - bg)
                                       .sum(axis=2) > 60).any(axis=0)))
    y_first_ink = body[0][0]

    lines = textwrap.wrap(text, width=ASCII_WRAP)
    if len(lines) > MAX_LINES:
        raise SystemExit(f"영문이 {len(lines)}줄 — {MAX_LINES}줄 안에 들어오게 줄여라:\n" + "\n".join(lines))

    # 블록 지우기: 헤더 아래 ~ ORDER LOG 헤더 위 (여백 4px)
    ya, yb = rat_hdr[1] + 4, log_hdr[0] - 4
    a2 = a.copy()
    a2[ya:yb, panel_x0 - 8:panel_x1] = bg.astype(np.uint8)

    def render(y_top_px: float, txt_lines):
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.imshow(a2, interpolation="nearest")
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        for k, ln in enumerate(txt_lines):
            fig.text(text_x / W, 1 - (y_top_px + k * pitch) / H, ln,
                     color=FG, fontsize=FONT_PT, va="top", ha="left")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=DPI, format="png")
        plt.close(fig)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))

    # 보정: va="top" 의 bbox 상단과 잉크 상단이 다르므로 한 번 그려 잉크 행을 재고 맞춘다.
    trial = render(y_first_ink, [lines[0]])
    tr = _ink_rows(trial, bg, panel_x0, panel_x1, ya, yb)
    off = (tr[0][0] - y_first_ink) if tr else 0
    final = render(y_first_ink - off, lines)

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(out)

    # 검증 출력
    outside = np.ones((H, W), bool); outside[ya:yb, panel_x0 - 8:panel_x1] = False
    mad = np.abs(final.astype(int) - a.astype(int)).sum(axis=2)[outside].mean()
    new_rows = _ink_rows(final, bg, panel_x0, panel_x1, ya, yb)
    print(f"frame {frame} crop {crop} → {out}  ({W}x{H})")
    print(f"  rationale 블록 y[{ya},{yb}) x[{panel_x0 - 8},{panel_x1})  원본 {len(body)}줄 pitch {pitch:.1f}px  x={text_x}")
    print(f"  영문 {len(lines)}줄, 첫 잉크행 원본 {y_first_ink} → 새 {new_rows[0][0] if new_rows else None} (보정 {off:+d}px)")
    print(f"  블록 밖 평균절대차 {mad:.3f}")
    for ln in lines:
        print("   |", ln)
    return final


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gif", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    p.add_argument("--text", required=True, help="영문 rationale (42자 폭으로 감싸 7줄 이내)")
    p.add_argument("--out", required=True)
    p.add_argument("--keep-ko", default=None, help="원본(한글) 프레임을 이 경로에 보존 (이미 있으면 건드리지 않음)")
    p.add_argument("--also", nargs="*", default=[], help="같은 결과를 추가로 저장할 경로들")
    args = p.parse_args(argv)
    final = localize(Path(args.gif), args.frame, tuple(args.crop) if args.crop else None,
                     args.text, Path(args.out), Path(args.keep_ko) if args.keep_ko else None)
    for extra in args.also:
        Image.fromarray(final).save(extra)
        print("  also →", extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
