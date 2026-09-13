"""지휘관 UI 틀 그대로 GIF/영상으로 굽는 헤드리스 익스포터.

왜 별도 스크립트인가
    render_sim.py --gif 는 **씬만** 그린다 — 지휘관 패널도, 확대 캠도 없다.
    run_commander_ui.py 는 그 UI 를 다 갖췄지만 TextBox/Button 위젯과 plt.show()
    에 묶여 있어 헤드리스에서 못 돈다. 그래서 같은 레이아웃을 Agg 백엔드로
    재구성하고, 입력창 대신 **명령 로그**를 둔다.

레이아웃 (run_commander_ui.py 와 같은 좌/우 배치)
    +--------------------------------------+------------------+
    | [브리핑] [적 캠] [아군 캠]  <- 오버레이 |  COMMANDER       |
    |                                      |  * 현재 명령      |
    |            해상 씬                    |  * 배정          |
    |                                      |  * 판단 근거      |
    |                                      |  ---------------  |
    |                                      |  ORDER LOG        |
    |                                      |  t=0   ...        |
    +--------------------------------------+------------------+

    상단 3패널은 renderer.setup_panels / draw_panels 를 그대로 쓴다. 이 함수들은
    예전에 만들어 두고 어디서도 호출하지 않던 것이다 — 새로 그리지 않는다.

실행:
    python tools/make_commander_gif.py --out demo.gif
    python tools/make_commander_gif.py --out demo.gif --enemy diversionary --frames 600
    python tools/make_commander_gif.py --out demo.gif --backend gemini --model gemini-3.5-flash-lite
    python tools/make_commander_gif.py --out demo.mp4 --fps 30        # ffmpeg 있으면 mp4
    python tools/make_commander_gif.py --out demo.gif --no-llm        # 휴리스틱만(오프라인)

지휘관 호출은 **동기**다. run_commander_ui 는 화면이 멈추면 안 되니 스레드로
비동기 호출하지만, 여기서는 프레임을 파일로 굽는 것이라 기다려도 된다 —
오히려 동기라야 "이 프레임에서 이 명령이 나왔다"가 정확히 맞는다.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── 색 (renderer.C 와 같은 계열. 패널 밖 여백까지 다크로 칠하기 위해 여기서도 쓴다) ──
BG_FIG = "#0A121F"
BG_PANEL = "#0d1b2a"
FG = "#ECEFF4"
FG_DIM = "#8A93A0"
ACCENT = "#26C6DA"
OK = "#00E676"
WARN = "#FFB74D"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="지휘관 UI(씬 + 확대 캠 + LLM 명령 패널)를 GIF/MP4 로 굽는다.")
    p.add_argument("--out", default=None,
                   help="출력 경로. 생략하면 논문_그래프/6_영상/ 아래에 조건을 담은 이름으로 "
                        "자동 저장한다. 확장자 .mp4 면 ffmpeg 시도 후 실패 시 GIF 폴백")
    p.add_argument("--frames", type=int, default=400, help="프레임 수 (기본 400)")
    p.add_argument("--fps", type=int, default=25, help="초당 프레임 (기본 25)")
    p.add_argument("--enemy", default="diversionary",
                   help="적 스폰 대형: concentrated|diversionary|wave|random")
    p.add_argument("--seed", type=int, default=0, help="난수 시드")
    p.add_argument("--replan", type=int, default=100,
                   help="지휘관 재계획 주기(step). 0=최초 1회만")
    p.add_argument("--backend", default="ollama", help="ollama|openai|gemini")
    p.add_argument("--model", default=None, help="모델명 (미지정 시 백엔드 기본값)")
    p.add_argument("--command", default="Capture all enemies",
                   help="지휘관에게 줄 자연어 명령")
    p.add_argument("--no-llm", action="store_true",
                   help="LLM 을 부르지 않고, 지휘관 자리에 휴리스틱 '계획'을 넣는다. "
                        "지휘관 계층은 그대로 있고 그 안의 LLM 만 빠진 상태다")
    p.add_argument("--no-commander", action="store_true",
                   help="★ 지휘관 계층 자체를 뺀다(assign_source='heuristic'). 시뮬 내장 배정이 "
                        "직접 돌고 계획 주입이 없다 — 평가에서 쓴 baseline 과 같은 구성이다. "
                        "--no-llm 과 다르다: 그쪽은 휴리스틱 계획을 LLM 경로로 주입한다")
    p.add_argument("--dpi", type=int, default=100,
                   help="저장 해상도(기본 100). ★ 용량을 줄이려고 이 값을 내리지 말 것 — "
                        "글자가 뭉개진다. 용량은 --step 과 --colors 로 조절한다")
    p.add_argument("--step", type=int, default=1,
                   help="렌더 1프레임당 시뮬 step 수(기본 2). 위성 배경을 쓰면 매 프레임이 "
                        "달라 GIF 팔레트 압축이 거의 안 먹는다 — 프레임 수가 곧 용량이므로 "
                        "같은 구간을 절반의 프레임으로 담는다")
    p.add_argument("--colors", type=int, default=0,
                   help="GIF 색 수. 0(기본)이면 후처리 없음 — 화질 우선. 값을 주면 저장 후 "
                        "전역 팔레트로 다시 양자화해 용량을 줄인다(위성 배경에서 약 45%% 감소, "
                        "대신 색 계단이 생긴다)")
    p.add_argument("--maneuver", choices=("unet", "heuristic"), default="unet",
                   help="기동 계층. unet(기본)=U-Net 점수맵 정책 — 이 프로젝트의 본 모델이다. "
                        "heuristic=시뮬 내장 휴리스틱 기동(비교용)")
    p.add_argument("--ckpt", default="boatattack_sim/models/u-net_map.pt",
                   help="U-Net 점수맵 체크포인트")
    p.add_argument("--nets", type=int, default=3,
                   help="배당 그물 장수(기본 3). 학습 config 는 1이라 한 장 깔면 배가 "
                        "정지한다 — 여러 장을 실어야 재전개한다")
    p.add_argument("--score-map", dest="score_map", action="store_true", default=True,
                   help="U-Net 점수맵 오버레이(유효마스크·히트맵·선택 픽셀). 기본 켜짐")
    p.add_argument("--no-score-map", dest="score_map", action="store_false",
                   help="점수맵 오버레이 끄기")
    p.add_argument("--satellite", dest="satellite", action="store_true", default=True,
                   help="위성 배경(기본 켜짐). 실패하면 해색 배경으로 떨어진다")
    p.add_argument("--no-satellite", dest="satellite", action="store_false",
                   help="위성 배경 끄기(오프라인/빠른 테스트)")
    p.add_argument("--zoom", action="store_true",
                   help="교전 구역으로 자동 줌. 전역(12.6km)을 그대로 보이면 실제 교전이 "
                        "화면의 30%%만 차지한다 — 켜면 모선+생존 적+아군을 꽉 채운다")
    p.add_argument("--zoom-margin", type=float, default=1200.0,
                   help="--zoom 여백(m, 기본 1200)")
    p.add_argument("--site", default=None,
                   help="해역(지형). terrain.KOREA_SITES 의 이름 또는 'lat,lon'. "
                        "미지정이면 config 기본(통영_매물도서). --list-sites 로 목록")
    p.add_argument("--list-sites", action="store_true", help="사용 가능한 해역을 찍고 종료")
    p.add_argument("--residual", dest="residual", action="store_true", default=False,
                   help="WP 후보 범위 원(action range 환형)을 보인다. 기본은 숨김 — "
                        "화면이 어지럽고 그물·경로를 가린다")
    p.add_argument("--clusters", dest="clusters", action="store_true", default=True,
                   help="적 클러스터 환형 부채꼴 표시(기본 켜짐). LLM 이 무엇을 묶어 배정했는지 보인다")
    p.add_argument("--no-clusters", dest="clusters", action="store_false",
                   help="클러스터 환형 부채꼴 숨김")
    a = p.parse_args(argv)
    if a.out is None:
        a.out = default_out(a)
    return a


def resolve_site(name: str | None):
    """--site 값 -> (lat, lon, 라벨). None 이면 (None, None, "default")."""
    from boatattack_sim.env.terrain import KOREA_SITES
    if not name:
        return None, None, "default"
    if "," in name:
        lat, lon = (float(v) for v in name.split(",", 1))
        return lat, lon, f"{lat:.4f}_{lon:.4f}"
    for nm, lat, lon in KOREA_SITES:
        if nm == name or nm.startswith(name) or name in nm:
            return lat, lon, nm
    raise SystemExit(f"[gif] 알 수 없는 해역: {name!r}. --list-sites 로 목록을 보라")


def default_out(args) -> str:
    """--out 생략 시 경로. 영상은 논문_그래프/6_영상/ 에 모으고, 이름에 조건을 담는다.

    조건을 파일명에 넣는 이유: 나중에 여러 개를 만들면 어느 것이 어떤 설정인지
    파일명만 보고 알 수 있어야 한다(지휘관/대형/시드/줌 여부).
    """
    from boatattack_sim.eval.plots import FIG_GROUP_VIDEO
    if args.no_commander:
        who = "nocommander"
    else:
        who = "heuristic" if args.no_llm else (args.model or args.backend)
    who = re.sub(r"[^0-9A-Za-z._-]+", "-", str(who))
    site = re.sub(r"[^0-9A-Za-z._가-힣-]+", "-", resolve_site(args.site)[2])
    name = f"commander_{args.maneuver}_{who}_{args.enemy}_seed{args.seed}"
    if site != "default":
        name += f"_{site}"
    if args.zoom:
        name += "_zoom"
    return os.path.join(REPO, "논문_그래프", FIG_GROUP_VIDEO, name + ".gif")


def _shrink_gif(path: str, colors: int) -> None:
    """저장된 GIF 를 전역 팔레트로 다시 양자화해 용량을 줄인다.

    PillowWriter 는 프레임마다 팔레트를 새로 만든다. 위성 배경처럼 프레임이 계속
    달라지는 영상에서는 그 팔레트가 프레임마다 통째로 다시 실려 파일이 폭증한다
    (실측 500프레임 134MB). 첫 프레임 기준 팔레트 하나로 통일하면 크게 준다.
    실패해도 원본은 그대로 두고 넘어간다 — 용량 최적화가 산출물을 잃게 하면 안 된다.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        im = Image.open(path)
        n = getattr(im, "n_frames", 1)
        if n < 2:
            return
        dur = im.info.get("duration", 40)
        im.seek(0)
        pal = im.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
        frames = []
        for f in range(n):
            im.seek(f)
            # ★ 디더링을 켜면 안 된다. 흩뿌린 노이즈가 프레임 간 차이를 늘려
            #   오히려 파일이 커진다(실측 11.1MB -> 14.8MB).
            frames.append(im.convert("RGB").quantize(palette=pal, dither=Image.NONE))
        before = os.path.getsize(path)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=dur, loop=0, optimize=True)
        after = os.path.getsize(path)
        print(f"[gif] 팔레트 {colors}색 재양자화: "
              f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    except Exception as e:                       # noqa: BLE001
        print(f"[gif] 용량 최적화 실패({type(e).__name__}: {e}) — 원본 유지")


def action_view(fd, *, margin: float = 1200.0, smooth=None, alpha: float = 0.12):
    """교전 구역을 감싸는 (cx, cy, half). draw_scene 의 view= 에 그대로 넣는다.

    모선 + 살아있는 적 + 살아있는 아군을 모두 담는 정사각 창을 만든다.
    `smooth` 에 직전 view 를 주면 지수이동평균으로 섞는다 — 안 그러면 적이 하나
    포획될 때마다 창이 툭툭 튀어 영상이 멀미난다.
    ※ half 를 단조 증가로 묶으면 안 된다. 이 시나리오는 적이 외곽에서 모선으로
      **모여드는** 구조라 창이 줄어드는 것이 정상 동작이다(실측 6300m → 2415m).
      단조 증가를 걸었더니 첫 프레임의 전역 크기에 그대로 못박혀 줌이 무의미해졌다.
      급변(포획으로 먼 적이 사라질 때)은 EMA 만으로 충분히 눌린다.
    """
    import numpy as np
    pts = [np.asarray(fd["mothership"], float).reshape(1, 2)]
    for pos_k, alive_k in (("enemy_pos", "enemy_alive"), ("ally_pos", "ally_alive")):
        pos = np.asarray(fd.get(pos_k, np.empty((0, 2))), float)
        if pos.size == 0:
            continue
        alive = np.asarray(fd.get(alive_k, np.ones(len(pos), bool)), bool)
        if alive.any():
            pts.append(pos[alive])
    p = np.vstack(pts)
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    half = float(max(np.abs(p[:, 0] - cx).max(), np.abs(p[:, 1] - cy).max())) + margin
    W = float(fd["world_size"])
    half = min(max(half, 1500.0), W / 2)
    if smooth is not None:
        pcx, pcy, phalf = smooth
        cx = pcx + alpha * (cx - pcx)
        cy = pcy + alpha * (cy - pcy)
        half = phalf + alpha * (half - phalf)
    # 창이 월드 밖으로 나가지 않게 중심을 당긴다.
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), W - half)
    return (cx, cy, half)


def _wrap_width(s: str, *, ascii_w: int = 42, cjk_w: int = 26) -> int:
    """줄바꿈 폭. 한글은 라틴 문자의 약 2배 폭이라 같은 글자수로 감싸면 패널을 넘친다.

    (실제로 Gemini 의 한국어 rationale 이 오른쪽으로 잘려 나갔다.)
    CJK 가 3할을 넘으면 좁은 폭을 쓴다 — 섞여 있어도 넘치지 않는 쪽으로 판단한다.
    """
    if not s:
        return ascii_w
    cjk = sum(1 for ch in s if "가" <= ch <= "힣" or "一" <= ch <= "鿿")
    return cjk_w if cjk / len(s) > 0.3 else ascii_w


# ── 지휘관 명령 로그 ────────────────────────────────────────────────────────
class OrderLog:
    """지휘관이 낸 계획을 시간순으로 쌓는다.

    폴백(LLM 실패 → 휴리스틱)을 **반드시 구분해서 보관한다**. 구분하지 않으면
    화면에는 명령이 흐르는데 실제로는 LLM 이 한 번도 안 불린 상태를 알 수 없다
    (이 프로젝트에서 gpt-4o-mini 가 폴백률 1.000 으로 전량 폐기된 전례가 있다).
    """

    MAX_SHOWN = 6

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, t: int, plan, assign_txt: str) -> dict:
        rat = str(getattr(plan, "rationale", "") or "")
        row = {
            "t": int(t),
            "fallback": "휴리스틱 방어" in rat,
            "alloc": "  ".join(f"C{d.cluster_id}:{list(d.ally_ids or [])}"
                               for d in (plan.deployments or [])) or "(none)",
            "assign": assign_txt,
            "rationale": rat,
        }
        self.rows.append(row)
        return row

    def recent(self):
        return self.rows[-self.MAX_SHOWN:]

    @property
    def n_fallback(self) -> int:
        return sum(1 for r in self.rows if r["fallback"])


def draw_commander_panel(ax, *, model: str, command: str, log: OrderLog,
                         status: str, flash: bool, headless: bool = False):
    """오른쪽 지휘관 패널: 현재 명령 / 배정 / 근거 / 명령 로그.

    `flash=True` 면 이번 프레임에 새 계획이 내려온 것 — 테두리를 밝혀 그 순간을 보이게 한다.
    """
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG_PANEL)
    edge = OK if flash else ACCENT
    for s in ax.spines.values():
        s.set_color(edge); s.set_linewidth(2.4 if flash else 1.2)

    last = log.rows[-1] if log.rows else None
    y = 0.975

    def line(txt, *, color=FG, size=8.5, bold=False, dy=0.030, x=0.035):
        nonlocal y
        ax.text(x, y, txt, color=color, fontsize=size, va="top", ha="left",
                fontweight="bold" if bold else "normal", transform=ax.transAxes)
        y -= dy

    # ★ 지휘관 계층이 없을 때 "LLM COMMANDER" 라고 쓰면 화면이 거짓말을 한다.
    #   제목·항목을 모드에 맞춰 바꾼다.
    if headless:
        line("NO COMMANDER LAYER", color=ACCENT, size=10.5, bold=True, dy=0.040)
        line("assignment   built-in heuristic (sim)", color=FG_DIM, size=8)
        line("maneuver     U-Net score map", color=FG_DIM, size=8, dy=0.042)
        line("설명", color=ACCENT, size=8.5, bold=True, dy=0.028)
        for ln in textwrap.wrap(
                "지휘관(LLM) 계층을 뺀 구성이다. 클러스터 배정은 시뮬 내장 "
                "휴리스틱이 매 결정마다 직접 계산하고, 계획 주입은 없다. "
                "논문 평가의 baseline 과 같은 설정이다.", width=26)[:8]:
            line(ln, size=7.8, dy=0.026)
        return
    line("LLM COMMANDER", color=ACCENT, size=10.5, bold=True, dy=0.040)
    line(f"model   {model}", color=FG_DIM, size=8)
    line(f"status  {status}", color=OK if "applied" in status.lower() else WARN, size=8,
         dy=0.042)

    line("COMMAND", color=ACCENT, size=8.5, bold=True, dy=0.028)
    for ln in textwrap.wrap(command, width=_wrap_width(command))[:3]:
        line(ln, size=8)
    y -= 0.014

    line("DEPLOYMENT", color=ACCENT, size=8.5, bold=True, dy=0.028)
    if last:
        for ln in textwrap.wrap(last["alloc"], width=40)[:2]:
            line(ln, size=8, color=OK)
        for ln in textwrap.wrap(last["assign"], width=40)[:2]:
            line(ln, size=8)
    else:
        line("(awaiting first plan)", color=FG_DIM, size=8)
    y -= 0.014

    line("RATIONALE", color=ACCENT, size=8.5, bold=True, dy=0.028)
    if last:
        rat = last["rationale"]
        # 폴백은 rationale 앞에 이유가 붙는다 — 색으로 갈라 LLM 출력과 혼동을 막는다.
        col = WARN if last["fallback"] else FG
        for ln in textwrap.wrap(rat, width=_wrap_width(rat))[:7]:
            line(ln, size=7.8, color=col, dy=0.026)
    else:
        line("(none)", color=FG_DIM, size=8)

    # ── 명령 로그 ──
    #   위 rationale 이 짧으면 빈 공간이 생기므로 고정 위치가 아니라 흐름 위치를 쓰되,
    #   너무 위로 올라와 rationale 과 붙지 않도록 하한(0.42)을 둔다.
    y = min(y - 0.030, 0.42)
    line("ORDER LOG", color=ACCENT, size=8.5, bold=True, dy=0.032)
    if not log.rows:
        line("(empty)", color=FG_DIM, size=8)
    for r in log.recent():
        mark, col = ("FB", WARN) if r["fallback"] else ("OK", OK)
        ax.text(0.035, y, f"[{mark}]", color=col, fontsize=7.5, va="top",
                ha="left", fontweight="bold", transform=ax.transAxes)
        ax.text(0.135, y, f"t={r['t']:<5d} {r['alloc'][:30]}", color=FG,
                fontsize=7.5, va="top", ha="left", transform=ax.transAxes)
        y -= 0.030

    if log.n_fallback:
        ax.text(0.035, 0.022,
                f"! {log.n_fallback}/{len(log.rows)} plans fell back to heuristic",
                color=WARN, fontsize=7.5, va="bottom", ha="left",
                transform=ax.transAxes)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_sites:
        from boatattack_sim.env.terrain import KOREA_SITES
        print("사용 가능한 해역 (--site 에 이름 또는 'lat,lon'):")
        for nm, lat, lon in KOREA_SITES:
            print(f"  {nm:<18s} {lat:.4f}, {lon:.4f}")
        return 0

    import matplotlib
    matplotlib.use("Agg")                    # ★ 창 없이 — DISPLAY 없는 환경에서도 돈다
    import warnings
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm

    # renderer 는 import 시점에 font.family 를 한글폰트 '단일'로 덮어쓴다.
    # 폴백 리스트가 이기려면 반드시 renderer import '뒤'에 폰트를 다시 잡아야 한다.
    from boatattack_sim.eval import renderer
    from commander.fallback import heuristic_plan
    from commander.sim_bridge import CommandedSimulator, build_battlefield, plan_to_assign

    _names = {f.name for f in _fm.fontManager.ttflist}
    _kf = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic")
                if f in _names), None)
    if _kf:
        matplotlib.rcParams["font.family"] = [_kf, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    # ── 기동 계층 ──
    #   기본은 U-Net 점수맵 정책이다. 이 프로젝트가 학습시킨 본 모델이므로 영상도
    #   그걸 보여야 한다(휴리스틱 기동은 --maneuver heuristic 로 비교용).
    #   ★ 전장 빌더가 다르다: DefenseVecEnv 계열은 build_battlefield_defense 를 쓴다.
    if args.maneuver == "unet":
        from commander.rl_bridge import build_battlefield_defense as build_bf
        from commander.unet_bridge import CommandedCnnEnv
        ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO, args.ckpt)
        lat, lon, site_label = resolve_site(args.site)
        print(f"[gif] U-Net 점수맵 정책 로딩… ({args.ckpt}, 배당 그물 {args.nets}장, "
              f"해역={site_label})")
        sim = CommandedCnnEnv(ckpt, enemy_mode=args.enemy, nets_per_ship=args.nets,
                              geo=None if lat is None else (lat, lon))
    else:
        build_bf = build_battlefield
        sim = CommandedSimulator(enemy_mode=args.enemy)
    sim.reset(seed=args.seed)
    sim.running = True

    # ── 지휘관 ──
    commander = None
    model_label = "heuristic (no LLM)"
    if args.no_commander:
        # 지휘관 계층 제거. 시뮬 내장 배정(_compute_assignment 의 super() 경로)이 돈다.
        sim.assign_source = "heuristic"
        model_label = "none (built-in assignment)"
        print("[gif] 지휘관 계층 없음 — 시뮬 내장 배정(assign_source='heuristic')")
    if not args.no_llm and not args.no_commander:
        try:
            # .env 의 GEMINI_API_KEY / OPENAI_API_KEY 를 읽는다. 없으면 어댑터가 조용히
            # 폴백 전용 모드로 떨어져 '전량 휴리스틱'이 LLM 결과처럼 보인다 —
            # 실제로 이 스크립트 첫 실행에서 9/9 폴백이 났다.
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(REPO, ".env"))
            except ImportError:
                pass
            from commander import make_commander
            commander = make_commander(args.backend, args.model)
            model_label = getattr(commander, "model", args.backend)
            commander.warmup()
            # ★ 사전 프로브: 실호출을 한 번 해 본다. 어댑터는 키가 없거나 쿼터가 막혀도
            #   예외 대신 휴리스틱 계획을 돌려주므로, 여기서 잡지 않으면 몇 분을 굽고 난
            #   뒤에야 전량 폴백이었다는 걸 알게 된다.
            probe = commander.plan(build_bf(sim, command=args.command))
            if "휴리스틱 방어" in str(getattr(probe, "rationale", "") or ""):
                reason = str(probe.rationale).split("→")[0].strip()
                print(f"[gif] ⚠ 지휘관 프로브 실패({reason}) — LLM 이 응답하지 않는다.")
                print("[gif]   그대로 진행하면 전량 휴리스틱이 LLM 결과처럼 보인다. "
                      "키/서버를 고치거나 --no-llm 을 명시하라.")
                return 2
            print(f"[gif] 지휘관 준비: {model_label} (프로브 통과)")
        except Exception as e:
            print(f"[gif] 지휘관 초기화 실패({type(e).__name__}: {e}) → 휴리스틱으로 진행")
            commander = None
            model_label = "heuristic (LLM unavailable)"

    # ── 위성 배경 (옵션) ──
    bg_img = bg_extent = None
    if args.satellite:
        try:
            from commander.satellite import fetch_satellite_bg
            res = fetch_satellite_bg(sim.cfg.geo_lat, sim.cfg.geo_lon, sim.cfg.world_size)
            if res:
                bg_img, bg_extent = res
                print("[gif] 위성 배경 로드 완료")
        except Exception as e:
            print(f"[gif] 위성 배경 실패({e}) → 해색 배경")

    # ── 레이아웃: run_commander_ui.py 와 같은 좌(씬)/우(패널) 배치 ──
    fig = plt.figure(figsize=(13.5, 9.0), facecolor=BG_FIG)
    ax = fig.add_axes((0.030, 0.035, 0.600, 0.885))       # 씬
    ax_info = fig.add_axes((0.648, 0.035, 0.335, 0.885))  # 지휘관 패널
    # 상단 오버레이 3패널(브리핑 / 적 캠 / 아군 캠).
    #   좌표를 여기서 다시 쓰지 않는다 — renderer.setup_panels 에 씬 axes 를 넘기면
    #   그 bbox 안쪽으로 배치해 준다. 기하가 한 곳(renderer)에만 있어야 서로 어긋나지 않는다.
    panels = renderer.setup_panels(fig, ax)
    for a in panels.values():
        a.set_facecolor(BG_PANEL)

    # 제목 두 줄. 한 줄에 몰면 긴 상태 문자열이 부제와 겹친다(실제로 겹쳤다).
    title = fig.text(0.030, 0.968, "", color=FG, fontsize=13, ha="left",
                     va="center", fontweight="bold")
    subtitle = fig.text(0.030, 0.939, "", color=FG_DIM, fontsize=8.5, ha="left",
                        va="center")

    log = OrderLog()
    state = {"cmd": args.command, "status": "standby", "flash": 0,
             "assign_txt": "(none)", "last_replan": -10 ** 9}

    def issue_order(t: int) -> None:
        """지휘관 호출 → 계획 주입 → 로그 적재. 동기 호출(파일로 굽는 중이라 기다려도 된다).

        --no-commander 면 아무것도 주입하지 않는다 — 시뮬이 스스로 배정한다.
        """
        if args.no_commander:
            state["status"] = "built-in assignment"
            state["last_replan"] = t
            return
        bf = build_bf(sim, command=state["cmd"])
        if commander is not None:
            plan = commander.plan(bf)
        else:
            # ★ sim.heuristic_plan() 이 아니다 — 그쪽은 '기동' 재계획이라 None 을 돌려준다.
            #   지휘관 수준(클러스터 배정) 휴리스틱은 commander.fallback 쪽이고,
            #   각 어댑터가 LLM 실패 시 쓰는 것과 같은 함수다.
            plan = heuristic_plan(bf)
            plan.rationale = f"LLM 미사용(--no-llm) → 휴리스틱 방어. {plan.rationale}"
        assign = plan_to_assign(plan, bf, mode="llm")
        sim.set_plan(plan, state["cmd"])
        committed = int((assign >= 0).sum())
        state["assign_txt"] = (f"deployed {committed}/{sim.cfg.n_allies}  "
                               + " ".join(f"#{i}->"
                                          + (f"C{a}" if a >= 0 else "res")
                                          for i, a in enumerate(assign.tolist())))
        row = log.add(t, plan, state["assign_txt"])
        state["status"] = "fallback applied" if row["fallback"] else "assignment applied"
        state["flash"] = 8            # 8 프레임 동안 패널 테두리 강조
        state["last_replan"] = t

    def scalar_t() -> int:
        t = sim.t
        return int(t.flat[0]) if hasattr(t, "flat") else int(t)

    def is_done() -> bool:
        d = sim.done
        return bool(d.flat[0]) if hasattr(d, "flat") else bool(d)

    issue_order(scalar_t())           # 첫 명령

    def update(_frame):
        t = scalar_t()
        for _ in range(max(1, args.step)):
            if not (sim.running and not is_done()):
                break
            sim.step()
            t = scalar_t()
            if args.replan > 0 and t - state["last_replan"] >= args.replan:
                issue_order(t)

        fd = sim.get_frame()
        fd["show_residual"] = bool(args.residual)   # 후보 범위 환형 (기본 숨김)
        fd["show_clusters"] = bool(args.clusters)
        view = None
        if args.zoom:
            view = action_view(fd, margin=args.zoom_margin, smooth=state.get("view"))
            state["view"] = view
        renderer.draw_scene(ax, fd, bg_img=bg_img, bg_extent=bg_extent,
                            show_help=False,          # 키보드 도움말은 영상에 무의미
                            view=view)
        # draw_scene 이 찍는 HUD 는 위 title 과 같은 정보라 지운다.
        #   ★ loc="left" 로 넣은 제목은 가운데 제목과 **다른 Text 객체**다.
        #     set_title("") 만으로는 안 지워진다 — loc 를 맞춰야 한다.
        ax.set_title("", loc="left")
        ax.set_xticks([]); ax.set_yticks([])          # 좌표 눈금 제거 — 화면이 훨씬 깨끗하다
        for s in ax.spines.values():
            s.set_color("#1E2B3F")
        # ★ U-Net 점수맵 오버레이 — 이 프로젝트의 본 모델이 무엇을 보고 어디를 골랐는지가
        #   영상의 핵심이다. 배별 유효마스크(옅게) + 점수맵 히트맵 + 선택 픽셀 2점.
        #   draw_panels 보다 먼저 그려야 캠 패널이 그 위에 얹힌다.
        if args.score_map and hasattr(sim, "cnn_viz"):
            from boatattack_sim.eval import cnn_overlay as CO
            viz = sim.cnn_viz()
            cfg_v, valid = viz["cfg"], viz["valid"]
            for pi in range(len(valid)):
                if int(viz["assign"][pi]) < 0:
                    continue          # 미배정 배는 점수맵이 균등분포라 의미 없다
                CO.draw_valid(ax, cfg_v, valid[pi], pi, alpha=0.07, z=2.0)
                if viz["prob"] is not None:
                    CO.draw_score(ax, cfg_v, viz["prob"][-1, pi], pi, alpha=0.38, z=2.2)
                if viz["pix"] is not None:
                    CO.draw_picks(ax, cfg_v, viz["pix"][pi],
                                  None if viz["offset"] is None else viz["offset"][pi],
                                  p=pi, z=7.0)
        renderer.draw_panels(panels, fd, bg_img, bg_extent)

        st = fd.get("stats", {})
        title.set_text(f"USV SWARM DEFENSE      t={t:<5d}"
                       f"captured {int(st.get('captures', 0))}   "
                       f"breached {int(st.get('breaches', 0))}   "
                       f"enemy alive {int(fd.get('n_alive', 0))}")
        subtitle.set_text(
            f"formation {args.enemy}   ·   seed {args.seed}   ·   "
            f"replan every {args.replan} steps   ·   "
            f"nets used {int(st.get('nets_used', 0))}   ·   "
            f"ally collisions {int(st.get('ally_collisions', 0))}")

        flash = state["flash"] > 0
        if flash:
            state["flash"] -= 1
        draw_commander_panel(ax_info, model=model_label, command=state["cmd"],
                             log=log, status=state["status"], flash=flash,
                             headless=args.no_commander)
        return []

    # ── 저장 ──
    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, update, frames=args.frames, interval=1000 // max(args.fps, 1),
                         blit=False, cache_frame_data=False)

    def save(path, writer):
        # facecolor 를 매 저장마다 넘겨야 한다 — 안 넘기면 프레임 여백이 흰색으로 굳는다.
        anim.save(path, writer=writer, dpi=args.dpi,
                  savefig_kwargs={"facecolor": BG_FIG})

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    t0 = time.perf_counter()
    if out.lower().endswith(".mp4"):
        # ffmpeg 는 writer 생성 시점이 아니라 **저장 시점**에 실행 파일을 찾는다.
        # 그래서 미리 확인할 수 없고 실제로 저장을 시도해 봐야 한다.
        try:
            from matplotlib.animation import FFMpegWriter
            save(out, FFMpegWriter(fps=args.fps))
            print(f"[gif] MP4 저장: {out}")
        except Exception as e:
            out = os.path.splitext(out)[0] + ".gif"
            print(f"[gif] ffmpeg 실패({type(e).__name__}: {e}) → GIF 로 폴백: {out}")
            save(out, PillowWriter(fps=args.fps))
    else:
        save(out, PillowWriter(fps=args.fps))

    plt.close(fig)
    if args.colors and out.lower().endswith(".gif"):
        _shrink_gif(out, args.colors)
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"[gif] 완료: {out}  ({size / 1e6:.2f} MB, {args.frames} 프레임, "
          f"{time.perf_counter() - t0:.1f}s)")
    print(f"[gif] 지휘관 호출 {len(log.rows)}회 (폴백 {log.n_fallback}회) · "
          f"최종 stats={sim.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
