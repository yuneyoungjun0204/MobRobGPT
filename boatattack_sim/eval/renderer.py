"""
boatattack_sim/eval/renderer.py — 순수 씬 렌더링 함수

matplotlib axes 만 의존 (데이터 소스/UI 무의존). frame_dict(시뮬레이터 get_frame)를
소비해 모선/적/아군/경로WP/painted 격자/그물 띠를 그린다.
(항해사모사 renderer.py 의 순수 draw_scene 패턴 차용.)
"""
import numpy as np
import matplotlib
from matplotlib import font_manager as _fm
from matplotlib.patches import Polygon as MplPolygon, Circle, Wedge, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from ..env import clustering

# 한글 라벨(전장 브리핑/캠 등) 깨짐 방지: 사용 가능한 한글 폰트 1개를 전역 지정.
_KFONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Batang")
_avail = {f.name for f in _fm.fontManager.ttflist}
for _kf in _KFONTS:
    if _kf in _avail:
        matplotlib.rcParams["font.family"] = _kf
        break
matplotlib.rcParams["axes.unicode_minus"] = False   # 한글폰트 음수기호 깨짐 방지

# ── 색상 ──────────────────────────────────────────────────────────────
C = {
    "bg":        "#0E1726",
    "grid":      "#1E2B3F",
    "mother":    "#FFD54F",
    "enemy":     "#EF5350",
    "enemy_dead":"#5A2A2A",
    "ally":      "#42A5F5",
    "ally_sel":  "#00E676",
    "path":      "#26C6DA",
    "wp":        "#26C6DA",
    "wp_net":    "#00E676",
    "paint":     "#7E57C2",
    "residual":  "#FFB74D",     # WP 잔차(±wp_adjust_max) 허용범위 원 — 앰버
    "text":      "#ECEFF4",
}

# painted 격자용 컬러맵 (0=투명, 1=보라)
_PAINT_CMAP = ListedColormap([(0, 0, 0, 0), (0.49, 0.34, 0.76, 0.55)])

# 클러스터 색 팔레트 (섹터 / 배정선 공유 → 어느 배가 어느 클러스터를 맡았나 한눈에)
_CLUSTER_PALETTE = ["#FF8A65", "#BA68C8", "#4FC3F7", "#FFD54F", "#81C784", "#F06292"]

# 테마: 지도 배경(라이트) vs 단색(다크)
_THEME_MAP  = {"sea": "#A8C8E8", "grid": "#FFFFFF", "tick": "#33414F", "text": "#1A2530"}
_THEME_DARK = {"sea": C["bg"],   "grid": C["grid"], "tick": "#6B7A90", "text": C["text"]}


def _rot(local, cx, cy, hdg_deg):
    """carrier/선박 로컬좌표(x=우현, y=함수) → world (nav: +y→(sin,cos))."""
    th = np.deg2rad(hdg_deg); c, s = np.cos(th), np.sin(th)
    local = np.asarray(local, dtype=np.float64)
    return np.column_stack([
        local[:, 0] * c + local[:, 1] * s + cx,
        -local[:, 0] * s + local[:, 1] * c + cy,
    ])


def _ship_poly(cx, cy, hdg_deg, length, width):
    """7-포인트 선박 다각형 (nav 좌표). 반환 [7,2]."""
    L, W = length / 2, width / 2
    local = [
        (0, L), (W * 0.55, L * 0.6), (W, -L * 0.15), (W * 0.7, -L),
        (-W * 0.7, -L), (-W, -L * 0.15), (-W * 0.55, L * 0.6),
    ]
    return _rot(local, cx, cy, hdg_deg)


def _local_rect(cxl, cyl, half_l, half_w, ang_deg):
    """carrier-로컬 프레임 안에서 (cxl,cyl) 중심, ang_deg 회전한 사각형 4점 (로컬좌표)."""
    r = np.deg2rad(ang_deg); c, s = np.cos(r), np.sin(r)
    base = np.array([(-half_w, -half_l), (half_w, -half_l),
                     (half_w, half_l), (-half_w, half_l)])
    return np.column_stack([base[:, 0] * c - base[:, 1] * s + cxl,
                            base[:, 0] * s + base[:, 1] * c + cyl])


# 항공모함 색
_CV = {"deck": "#6B7480", "deck2": "#565E68", "island": "#262C34",
       "edge": "#AEB6C2", "mark": "#F4D03F", "hull": "#3E454E"}


def draw_carrier(ax, cx, cy, hdg, size, z=6):
    """웅장한 항공모함 모선 렌더 (뾰족 함수 + 비행갑판 + 각진 착함갑판 + 아일랜드)."""
    hl = size * 0.95     # 반길이 (length ≈ 1.9·size)
    hw = size * 0.31     # 반폭   (width  ≈ 0.62·size)

    # 1) 선체 그림자 (살짝 큰 외곽)
    hull = [(0, hl * 1.06), (0.62 * hw * 1.15, 0.85 * hl), (hw * 1.12, 0.60 * hl),
            (hw * 1.12, -hl * 1.02), (-hw * 1.12, -hl * 1.02),
            (-hw * 1.12, 0.60 * hl), (-0.62 * hw * 1.15, 0.85 * hl)]
    ax.add_patch(MplPolygon(_rot(hull, cx, cy, hdg), closed=True,
                            facecolor=_CV["hull"], edgecolor="none",
                            alpha=0.55, zorder=z))

    # 2) 비행갑판 (뾰족한 함수)
    deck = [(0, hl), (0.60 * hw, 0.84 * hl), (hw, 0.58 * hl),
            (hw, -hl), (-hw, -hl), (-hw, 0.58 * hl), (-0.60 * hw, 0.84 * hl)]
    ax.add_patch(MplPolygon(_rot(deck, cx, cy, hdg), closed=True,
                            facecolor=_CV["deck"], edgecolor=_CV["edge"],
                            lw=1.4, zorder=z + 1))

    # 3) 각진 착함갑판 (port 쪽으로 ~9° 기운 패럴렐로그램, 살짝 색차)
    ang = _local_rect(-0.34 * hw, 0.04 * hl, 0.62 * hl, 0.40 * hw, 9.0)
    ax.add_patch(MplPolygon(_rot(ang, cx, cy, hdg), closed=True,
                            facecolor=_CV["deck2"], edgecolor=_CV["edge"],
                            lw=0.8, alpha=0.95, zorder=z + 2))

    # 4) 활주로 마킹: 주갑판 중심선(점선) + 각진갑판 중심선
    main = _rot([(0.14 * hw, -0.86 * hl), (0.14 * hw, 0.80 * hl)], cx, cy, hdg)
    ax.plot(main[:, 0], main[:, 1], c=_CV["mark"], lw=1.3, ls=(0, (5, 4)),
            alpha=0.9, zorder=z + 3)
    angc = _rot(_local_rect(-0.34 * hw, 0.04 * hl, 0.60 * hl, 0.0, 9.0),
                cx, cy, hdg)
    ax.plot(angc[:2, 0], angc[:2, 1], c=_CV["mark"], lw=1.3, ls=(0, (5, 4)),
            alpha=0.9, zorder=z + 3)
    # 함수 캐터펄트 2줄
    for off in (-0.18, 0.10):
        cat = _rot([(off * hw, 0.30 * hl), (off * hw, 0.78 * hl)], cx, cy, hdg)
        ax.plot(cat[:, 0], cat[:, 1], c="white", lw=0.8, alpha=0.65, zorder=z + 3)

    # 5) 아일랜드(함교) — 우현 + 작은 마스트 점
    island = _local_rect(0.74 * hw, -0.02 * hl, 0.20 * hl, 0.18 * hw, 0.0)
    ax.add_patch(MplPolygon(_rot(island, cx, cy, hdg), closed=True,
                            facecolor=_CV["island"], edgecolor="#11151A",
                            lw=0.8, zorder=z + 4))
    mast = _rot([(0.74 * hw, 0.06 * hl)], cx, cy, hdg)
    ax.scatter(mast[:, 0], mast[:, 1], s=10, c="#FF7043", zorder=z + 5)


def draw_clusters(ax, fd: dict, z=4.5):
    """정책이 보는 적 클러스터(적응형 gap 클러스터링)를 시각화.
      ★ 정책/배정과 **동일한** cluster_by_gaps_vec(labels) 사용(옛 고정-빈 enemy_clusters 아님).
      각 클러스터는 **자기 멤버만** 감싸는 환형 부채꼴(annular wedge: 방위 arc × 반경 범위)로 그린다.
      클러스터끼리 방위 arc 가 gap 으로 분리돼 있어 wedge 는 서로 겹치지 않고(=한 적이 두 클러스터에
      안 들어감), 한 클러스터가 다른 클러스터를 포함하지도 않는다(원 표현의 nesting/overlap 제거)."""
    K = int(fd.get("n_clusters", 0))
    if K <= 0:
        return
    epos = np.asarray(fd["enemy_pos"]); ealive = np.asarray(fd["enemy_alive"], bool)
    mx, my = fd["mothership"]
    if not ealive.any():
        return
    ehdg = np.asarray(fd.get("enemy_hdg", np.zeros(len(epos))), np.float64)
    gap = float(fd.get("cluster_gap_deg", 40.0))
    espeed = float(fd.get("enemy_speed", 8.0))                  # labels 무관(approach 전용)
    cl = clustering.cluster_by_gaps_vec(epos[None], ealive[None], ehdg[None],
                                        (mx, my), espeed, K, gap)
    labels = cl["labels"][0]; cnt = cl["count"][0]
    palette = _CLUSTER_PALETTE
    cen = np.array([mx, my])
    apad = max(2.0, min(4.0, gap * 0.25))                       # 각 패딩(<gap/2 → 겹침 없음)
    rpad = 260.0                                                # 반경 패딩(m)
    for k in range(K):
        if cnt[k] == 0:
            continue
        col = palette[k % len(palette)]
        mem = epos[(labels == k) & ealive]                      # 이 클러스터 멤버만
        d = mem - cen
        rad = np.hypot(d[:, 0], d[:, 1])                        # 멤버 반경
        brg = np.degrees(np.arctan2(d[:, 0], d[:, 1]))          # nav 방위(deg)
        # 원형(wrap) 안전한 각도 범위: 원형평균 기준 ±편차
        cb = np.degrees(np.arctan2(np.sin(np.deg2rad(brg)).mean(),
                                   np.cos(np.deg2rad(brg)).mean()))
        dev = (brg - cb + 180.0) % 360.0 - 180.0               # [-180,180]
        amin = cb + dev.min() - apad; amax = cb + dev.max() + apad
        rin = max(0.0, rad.min() - rpad); rout = rad.max() + rpad
        # nav 방위(0=N,CW+) → matplotlib Wedge 각(0=E,CCW+): theta=90−brg, 범위 뒤집힘
        th1 = 90.0 - amax; th2 = 90.0 - amin
        ax.add_patch(Wedge(cen, rout, th1, th2, width=rout - rin,
                           facecolor=col, edgecolor="none", alpha=0.12, zorder=z))
        ax.add_patch(Wedge(cen, rout, th1, th2, width=rout - rin,
                           facecolor="none", edgecolor=col, lw=1.6, alpha=0.85, zorder=z + 0.1))
        cx, cy = mem[:, 0].mean(), mem[:, 1].mean()
        ax.scatter([cx], [cy], marker="x", s=70, c=col, linewidths=2, zorder=z + 0.2)
        ax.text(cx, cy, f"C{k} ×{int(cnt[k])}", color=col, fontsize=8,
                ha="center", va="bottom", fontweight="bold", zorder=z + 0.2)


def draw_assignment(ax, fd: dict, z=4.7):
    """클러스터→배 배정 시각화: 배정된 배에서 담당 클러스터 교점까지 화살선(클러스터 색),
    교점에 ◇ 마커. 어느 배가 어느 클러스터를 '효율적으로' 맡았는지 한눈에 보인다."""
    assign = fd.get("assign"); assignI = fd.get("assignI")
    apos = fd.get("ally_pos"); alive = fd.get("ally_alive")
    if assign is None or assignI is None or apos is None:
        return
    assign = np.asarray(assign); assignI = np.asarray(assignI); apos = np.asarray(apos)
    for i in range(len(apos)):
        k = int(assign[i])
        if k < 0:
            continue                                   # 예비(미배정) 배는 선 없음
        if alive is not None and not bool(alive[i]):
            continue
        col = _CLUSTER_PALETTE[k % len(_CLUSTER_PALETTE)]
        sx, sy = apos[i]; ix, iy = assignI[i]
        ax.annotate("", xy=(ix, iy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                    alpha=0.85, shrinkA=8, shrinkB=4),
                    zorder=z)
        ax.scatter([ix], [iy], marker="D", s=55, facecolors="none",
                   edgecolors=col, linewidths=1.8, zorder=z + 0.1)


def _seg_cross(p1, p2, p3, p4):
    """선분 p1-p2 와 p3-p4 의 proper 교차 여부(2D)."""
    cr = lambda ax, ay, bx, by: ax * by - ay * bx
    gx, gy = p4[0] - p3[0], p4[1] - p3[1]
    d1 = cr(gx, gy, p1[0] - p3[0], p1[1] - p3[1])
    d2 = cr(gx, gy, p2[0] - p3[0], p2[1] - p3[1])
    hx, hy = p2[0] - p1[0], p2[1] - p1[1]
    d3 = cr(hx, hy, p3[0] - p1[0], p3[1] - p1[1])
    d4 = cr(hx, hy, p4[0] - p1[0], p4[1] - p1[1])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _net_leg_cross_pt(e, m, a, b):
    """ray(e→m) 와 leg(a→b) proper 교차 시 ray 파라미터 t(0=e,1=m) 반환, 아니면 None."""
    cr = lambda ax, ay, bx, by: ax * by - ay * bx
    d1x, d1y = m[0] - e[0], m[1] - e[1]       # ray
    d2x, d2y = b[0] - a[0], b[1] - a[1]       # leg
    denom = cr(d1x, d1y, d2x, d2y)
    o1 = cr(d2x, d2y, e[0] - a[0], e[1] - a[1])
    o2 = cr(d2x, d2y, m[0] - a[0], m[1] - a[1])
    o3 = cr(d1x, d1y, a[0] - e[0], a[1] - e[1])
    o4 = cr(d1x, d1y, b[0] - e[0], b[1] - e[1])
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and abs(denom) > 1e-9:
        return o3 / denom
    return None


def draw_raycast(ax, fd: dict, z=7.5):
    """★ 단일커버 레이캐스트 시각화 (env._raycast_single_cover_vec 와 동일 기하).
    적→모선 ray 를 아군 **그물벽(net leg)** 이 막는지 판정:
      · 굵은 초록 = 단일커버(유효)  · 주황 = 충돌근접 2대 중복(무효=충돌 예정)  · 빨강 = 누수."""
    epos = np.asarray(fd["enemy_pos"]); ealive = np.asarray(fd["enemy_alive"], bool)
    route = fd.get("route"); net_mask = fd.get("net_mask"); aalive = fd.get("ally_alive")
    if route is None or net_mask is None or not ealive.any():
        return
    route = np.asarray(route); net_mask = np.asarray(net_mask, bool)
    aalive = np.asarray(aalive, bool) if aalive is not None else np.ones(route.shape[0], bool)
    P, Kw = route.shape[0], route.shape[1]
    mx, my = fd["mothership"]; m = np.array([mx, my])
    thr = float(fd.get("raycov_collide_m", 400.0))
    pa = np.asarray(fd["painted"]) if fd.get("painted") is not None else None
    G = pa.shape[0] if pa is not None else 0; W = fd["world_size"]; cell = W / max(G, 1)
    pts_ts = np.linspace(0.05, 0.95, 24)
    n_cov = n_red = n_leak = n_live = 0
    for k in range(len(epos)):
        if not ealive[k]:
            continue
        n_live += 1
        e = np.asarray(epos[k])
        pts = []                                     # 막은 배별 최근접 교차점(계획 net-leg)
        for p in range(P):
            if not aalive[p]:
                continue
            tmin = None
            for lg in range(Kw - 1):
                if not net_mask[p, lg + 1]:
                    continue                          # 그물 레그만
                t = _net_leg_cross_pt(e, m, route[p, lg], route[p, lg + 1])
                if t is not None and (tmin is None or t < tmin):
                    tmin = t
            if tmin is not None:
                pts.append(e + tmin * (m - e))
        net_painted = False                           # 실제 설치 그물 차단(persistent)
        if pa is not None and pa.any():
            xs = e[0] + (mx - e[0]) * pts_ts; ys = e[1] + (my - e[1]) * pts_ts
            ci = np.clip((xs / cell).astype(int), 0, G - 1); cj = np.clip((ys / cell).astype(int), 0, G - 1)
            net_painted = bool(pa[ci, cj].any())
        covered = (len(pts) >= 1) or net_painted
        collide = any(np.hypot(*(pts[i] - pts[j])) < thr
                      for i in range(len(pts)) for j in range(i + 1, len(pts)))
        if covered and collide:
            n_red += 1; col = "#FF9100"; lw = 2.6; a = 0.95          # 주황=충돌근접 중복(무효)
        elif covered:
            n_cov += 1; col = "#00E676"; lw = 3.2; a = 0.95          # 굵은 초록=단일커버(그물)
        else:
            n_leak += 1; col = "#FF1744"; lw = 2.0; a = 0.9          # 빨강=누수
        ax.plot([e[0], mx], [e[1], my], color=col, lw=lw, alpha=a, zorder=z,
                solid_capstyle="round")
        ax.scatter([e[0]], [e[1]], s=22, c=col, zorder=z + 0.1, edgecolors="white", linewidths=0.5)
    ax.text(0.5, 1.035,
            f"RAYCAST single-cover = {n_cov}/{n_live}   (green=single net, orange=collide-redundant, red=leak)",
            transform=ax.transAxes, color="#00E676", fontsize=9, ha="center", va="bottom",
            zorder=z, weight="bold")


def draw_rrt(ax, rrt_res: dict, assign, z=4.6):
    """RRT* 경로생성 오버레이: 배별 트리(연하게) + 해 경로(short, 굵게) + start/goal.
    rrt_res = {ship: {nodes,parent,short,start,goal,reached}} (rrt_planner.plan_env_world).
    assign[P] = 배별 클러스터 idx(색상용, -1=회색)."""
    if not rrt_res:
        return
    palette = _CLUSTER_PALETTE
    for p, r in rrt_res.items():
        k = int(assign[p]) if assign is not None and assign[p] >= 0 else -1
        col = palette[k % len(palette)] if k >= 0 else "#9AA4B2"
        nodes = r["nodes"]; parent = r["parent"]
        segs = [[nodes[int(parent[i])], nodes[i]] for i in range(len(nodes)) if parent[i] >= 0]
        if segs:
            ax.add_collection(LineCollection(segs, colors=col, linewidths=0.4,
                                             alpha=0.22, zorder=z))                    # RRT 트리
        # 목표 그물 벽선 A–B (점선) — '여기에 벽을 깐다' 의도 표시
        nl = r.get("netline")
        if nl is not None:
            (A, B) = nl
            ax.plot([A[0], B[0]], [A[1], B[1]], ":", color=col, lw=1.4, alpha=0.6, zorder=z + 0.1)
        sp = r["short"]
        ls = "-" if r.get("reached", False) else "--"               # 미도달 transit=점선
        ax.plot(sp[:, 0], sp[:, 1], ls, color=col, lw=2.2, alpha=0.95, zorder=z + 0.2)
        # net leg(sweep) 구간은 굵게 강조 (그물 살포 기동)
        nw = r.get("net_wp")
        if nw is not None and len(nw) == len(sp):
            for i in range(len(sp) - 1):
                if nw[i + 1]:
                    ax.plot(sp[i:i + 2, 0], sp[i:i + 2, 1], "-", color=col, lw=4.5,
                            alpha=0.9, zorder=z + 0.25)
        ax.scatter(sp[:, 0], sp[:, 1], s=18, c=col, zorder=z + 0.3)
        ax.scatter([r["goal"][0]], [r["goal"][1]], marker="*", s=120, facecolors="none",
                   edgecolors=col, linewidths=1.6, zorder=z + 0.3)


def draw_scene(ax, fd: dict, bg_img=None, bg_extent=None, show_help: bool = True,
               view=None, show_paths: bool = True):
    """frame_dict 한 장을 ax 에 렌더링. bg_img/bg_extent 주면 실제 지도 배경.
    view=(cx,cy,half) 주면 그 영역만 확대(클로즈업 캠 모드: HUD/도움말/눈금/잔차원 생략).
    show_paths=False 면 경로선·WP 마커·잔차원 생략(캠에선 선박+그물만)."""
    if not fd:
        return
    W = fd["world_size"]
    has_bg = bg_img is not None
    th = _THEME_MAP if has_bg else _THEME_DARK
    minimap = view is not None

    ax.clear()
    ax.set_facecolor(th["sea"])
    if minimap:
        cx, cy, half = view
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    else:
        ax.set_xlim(0, W); ax.set_ylim(0, W)
    ax.set_aspect("equal")

    # 실제 해역 지도 배경 (있으면) — 없으면 해색 단색
    if has_bg:
        ax.imshow(bg_img, extent=bg_extent or (0, W, 0, W), origin="upper",
                  zorder=0, interpolation="bilinear", aspect="auto")
        ax.grid(True, color="white", lw=0.4, alpha=0.30, zorder=1)
    else:
        ax.grid(True, color=th["grid"], lw=0.5, alpha=0.5)
    ax.tick_params(colors=th["tick"], labelsize=6)
    if minimap:
        ax.set_xticks([]); ax.set_yticks([])

    # painted 격자 (transpose: grid[i=x, j=y] → imshow 행=y)
    painted = fd.get("painted")
    if painted is not None and painted.any():
        ax.imshow(painted.T, origin="lower", extent=(0, W, 0, W),
                  cmap=_PAINT_CMAP, vmin=0, vmax=1, zorder=1.5,
                  interpolation="nearest", aspect="auto")

    # 모선 = 웅장한 항공모함 + breach 반경
    mx, my = fd["mothership"]
    ax.add_patch(Circle((mx, my), fd["mothership_radius"], facecolor="none",
                        edgecolor=C["mother"], lw=1.0, ls="--", alpha=0.6, zorder=2))
    draw_carrier(ax, mx, my, fd.get("moback_heading", 0.0),
                 fd["moback_size"], z=6)

    # 적 — 아군과 동일한 7-포인트 선박 디자인 (heading 방향), 색만 빨강
    epos = fd["enemy_pos"]; ealive = fd["enemy_alive"]; ehdg = fd["enemy_hdg"]
    e_len = fd.get("enemy_size", 100.0) * 1.6      # 선박 길이 ≈ 사각 한 변의 1.6배
    e_wid = fd.get("enemy_size", 100.0) * 0.55
    for k in range(len(epos)):
        x, y = epos[k]
        col = C["enemy"] if ealive[k] else C["enemy_dead"]
        a = 0.95 if ealive[k] else 0.3
        verts = _ship_poly(x, y, ehdg[k], e_len, e_wid)
        ax.add_patch(MplPolygon(verts, closed=True, facecolor=col,
                                edgecolor="white", lw=0.9, alpha=a, zorder=5))

    # 적 클러스터 오버레이 (정책이 보는 각도 섹터 군집) + 배→클러스터 배정선
    if fd.get("show_clusters", False):
        draw_clusters(ax, fd)
        draw_assignment(ax, fd)

    # 레이캐스트 오버레이 (적→모선 ray; 막힘=초록, 누수=빨강) — coverage 보상 직관화
    if fd.get("show_raycast", False):
        draw_raycast(ax, fd)

    # 아군 + 경로
    apos = fd["ally_pos"]; ahdg = fd["ally_hdg"]; paths = fd["ally_paths"]
    # ★ 후보 경로(반투명): 매 결정 정책 분포서 K샘플한 '될 수 있는 경로들'(GRPO 후보) — 행동공간 범위.
    cr = fd.get("cand_routes")
    if cr is not None:
        for pi in range(min(cr.shape[1], len(apos))):
            for kk in range(cr.shape[0]):
                cp = np.vstack([apos[pi], cr[kk, pi]])
                ax.plot(cp[:, 0], cp[:, 1], color=C["wp"], lw=0.7, alpha=0.13, zorder=2)
    nets = fd["ally_nets"]; painting = fd["ally_painting"]; sel = fd["selected"]
    alive = fd.get("ally_alive", None)
    assign = fd.get("assign", None)
    for i in range(len(apos)):
        x, y = apos[i]
        # 경로 라인 + 그물 전개 구간 강조 + WP 번호
        #   show_paths=False(클로즈업 캠): 경로선/WP마커/잔차원 생략 — '선박+그물 띠'만.
        path = paths[i]
        if path:
            if show_paths:
                a_main = 0.9 if i == sel else 0.45
                pts = np.array([[x, y]] + [[w["x"], w["y"]] for w in path])
                ax.plot(pts[:, 0], pts[:, 1], c=C["path"], lw=1.2,
                        alpha=a_main, ls="--", zorder=3)
            # 그물 전개 구간(=paint WP 로 향하는 segment) 굵은 초록 띠 — 그물이므로 항상 표시
            prev = np.array([x, y])
            for w in path:
                p = np.array([w["x"], w["y"]])
                if w["paint"]:
                    ax.plot([prev[0], p[0]], [prev[1], p[1]], c=C["wp_net"],
                            lw=6, alpha=0.45 if i == sel else 0.25, zorder=3,
                            solid_capstyle="round")
                prev = p
            # WP 마커 + 번호 (그물=초록, 경유=청록)
            rmax = float(fd.get("wp_adjust_max", 0.0))
            show_res = show_paths and fd.get("show_residual", False) and rmax > 0
            for k, w in (enumerate(path) if show_paths else []):
                col = C["wp_net"] if w["paint"] else C["wp"]
                # ★ 잔차(residual) 허용범위: 정책이 이 WP 를 baseline 에서 ±wp_adjust_max(=반경 rmax)
                #   안으로 미세조정 → 앰버 점선 '원'으로 표시 (GIF 시점 디자인 복원).
                if show_res:
                    ax.add_patch(Circle((w["x"], w["y"]), rmax,
                                        facecolor=C["residual"], alpha=0.10, zorder=2))
                    ax.add_patch(Circle((w["x"], w["y"]), rmax, facecolor="none",
                                        edgecolor=C["residual"], lw=1.8, ls=(0, (3, 2)),
                                        alpha=0.95 if i == sel else 0.65, zorder=5))
                    if k == 0 and (i == sel or (sel < 0 and i == 0)):   # 대표 배 첫 WP 라벨
                        ax.text(w["x"] + rmax, w["y"] + rmax,
                                f"action range +-{rmax:.0f}m", color=C["residual"],
                                fontsize=7, fontweight="bold", va="bottom", ha="left", zorder=6)
                # ★ WP 마커 = 파란/시안 '별'(그물 WP=초록 별). GIF 시점 디자인 복원(빨간 원 → 별).
                ax.scatter([w["x"]], [w["y"]], marker="*", s=80, c=col,
                           edgecolors="white", linewidths=0.5, zorder=4)
                ax.text(w["x"], w["y"], f"  {k+1}", color=col, fontsize=7,
                        va="center", ha="left", zorder=4,
                        fontweight="bold" if w["paint"] else "normal")
        # 선박 (비활성=충돌로 격침된 아군은 회색·반투명)
        alive_i = True if alive is None else bool(alive[i])
        face = (C["ally_sel"] if i == sel else C["ally"]) if alive_i else "#555B66"
        verts = _ship_poly(x, y, ahdg[i], fd["ship_len"], fd["ship_wid"])
        ax.add_patch(MplPolygon(verts, closed=True, facecolor=face,
                                edgecolor="white", lw=1.0,
                                alpha=1.0 if alive_i else 0.35, zorder=8))
        if alive_i:
            role = ""
            if assign is not None:
                k = int(assign[i])
                role = f"  →C{k}" if k >= 0 else "  RSV"   # 담당 클러스터 / 예비
            tag = f"#{i}  net:{int(nets[i])}" + (" ▣" if painting[i] else "") + role
        else:
            tag = f"#{i} ✖"
        ax.text(x, y + fd["ship_len"] * 0.7, tag, color=face, fontsize=6,
                ha="center", va="bottom", zorder=9)

    # 클로즈업 캠 모드: HUD/도움말 없이 씬만. 호출측(_draw_cam)이 타이틀/테두리 처리.
    if minimap:
        return

    # HUD
    st = fd["stats"]
    mode = "MANUAL" if fd["manual"] else "AUTO(heuristic)"
    run = "▶" if fd["running"] else "Ⅱ"
    stage = fd.get("net_stage", 0)
    arm = "  [NET: click direction → auto-deploy]" if stage == 1 else ""
    emode = fd.get("enemy_mode", "")
    etag = f"[{emode}]  " if emode else ""
    nt = st.get("net_touches", 0)
    hud = (f"{run} {mode}  {etag}t={fd['t']}  alive={fd['n_alive']}   "
           f"cap={st['captures']} breach={st['breaches']} "
           f"coll={st['ally_collisions']} nets={st['nets_used']} "
           f"net_hit={nt}{arm}")
    ax.set_title(hud, color=th["text"], fontsize=9, pad=6, loc="left")

    if show_help:
        help_txt = ("AUTO=cluster-block heuristic   [m]manual adjust  [space]play/pause  "
                    "[k]clusters  [n]net  [c]clear  [r]reset  [q]quit")
        ax.text(0.5, -0.06, help_txt, transform=ax.transAxes, color=th["tick"],
                fontsize=7, ha="center", va="top")


# ── 전장 브리핑 패널 + 클로즈업(팔로우) 캠 오버레이 ──────────────────────
def cluster_centers(fd: dict):
    """활성 적 클러스터(정책과 동일한 적응형 gap 클러스터링)별
    (k, 중심[2], 멤버수, 모선거리, 멤버반경) 리스트. 살아있는 적 없으면 [].
    ★ draw_clusters 와 같은 cluster_by_gaps_vec 사용 → 메인맵 부채꼴과 캠 대상 일치."""
    K = int(fd.get("n_clusters", 0))
    epos = np.asarray(fd["enemy_pos"]); ealive = np.asarray(fd["enemy_alive"], bool)
    if K <= 0 or not ealive.any():
        return []
    mx, my = fd["mothership"]
    ehdg = np.asarray(fd.get("enemy_hdg", np.zeros(len(epos))), np.float64)
    gap = float(fd.get("cluster_gap_deg", 40.0))
    espeed = float(fd.get("enemy_speed", 8.0))
    cl = clustering.cluster_by_gaps_vec(epos[None], ealive[None], ehdg[None],
                                        (mx, my), espeed, K, gap)
    labels = cl["labels"][0]; cnt = cl["count"][0]
    out = []
    for k in range(K):
        if cnt[k] == 0:
            continue
        mem = epos[(labels == k) & ealive]
        cx, cy = mem[:, 0].mean(), mem[:, 1].mean()
        rad = float(np.hypot(mem[:, 0] - cx, mem[:, 1] - cy).max()) if len(mem) else 0.0
        out.append((k, np.array([cx, cy]), int(cnt[k]),
                    float(np.hypot(cx - mx, cy - my)), rad))
    return out


def follow_cluster_view(fd: dict):
    """모선과 가장 가까운(=가장 위협적인) 활성 적 클러스터 추적 캠. (cx,cy,half,label,color)|None."""
    cs = cluster_centers(fd)
    if not cs:
        return None
    k, c, cnt, mdist, rad = min(cs, key=lambda r: r[3])      # 모선 최근접 클러스터
    col = _CLUSTER_PALETTE[k % len(_CLUSTER_PALETTE)]
    half = max(rad + 700.0, 1100.0)                          # 클러스터가 화면에 꽉 차게
    return (float(c[0]), float(c[1]), half, f"ENEMY CAM  C{k} x{cnt}", col)


def follow_ally_view(fd: dict):
    """활성 적 클러스터에 가장 가까운 살아있는 아군 추적 캠. (cx,cy,half,label,color)|None."""
    cs = cluster_centers(fd)
    apos = np.asarray(fd["ally_pos"])
    alive = np.asarray(fd.get("ally_alive", np.ones(len(apos), bool)), bool)
    if not cs or not alive.any():
        return None
    cents = np.array([r[1] for r in cs])                    # [Nc,2]
    best_i, best_d = -1, np.inf
    for i in range(len(apos)):
        if not alive[i]:
            continue
        dmin = np.hypot(cents[:, 0] - apos[i, 0], cents[:, 1] - apos[i, 1]).min()
        if dmin < best_d:
            best_d, best_i = dmin, i
    if best_i < 0:
        return None
    c = apos[best_i]
    return (float(c[0]), float(c[1]), 1400.0, f"ALLY CAM  #{best_i}", C["ally_sel"])


def draw_briefing(ax, fd: dict):
    """좌상단 전장 상황판: 적 포획/돌파/생존 + 아군 출격(배정)/예비/격침."""
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((0.055, 0.090, 0.149, 0.92))           # #0E1726 (반투명)
    for s in ax.spines.values():
        s.set_color("#26C6DA"); s.set_linewidth(1.4)

    st = fd.get("stats", {})
    captured = int(st.get("captures", 0))
    breached = int(st.get("breaches", 0))
    n_alive = int(fd.get("n_alive", 0))                     # 생존 적 수
    assign = np.asarray(fd.get("assign", []))
    alive = np.asarray(fd.get("ally_alive", []), bool)
    P = len(alive)
    if P:
        active = int(((assign >= 0) & alive).sum())         # 클러스터 배정된 출격 아군
        reserve = int(((assign < 0) & alive).sum())         # 미배정(예비) 아군
        dead = int((~alive).sum())                          # 격침 아군
    else:
        active = reserve = dead = 0

    ax.text(0.5, 0.95, "BATTLEFIELD BRIEFING", color="#ECEFF4",
            fontsize=8.5, ha="center", va="top", fontweight="bold")
    rows = [
        ("Enemy Captured",  f"{captured}",   C["ally_sel"]),
        ("Enemy Breached",  f"{breached}",   C["enemy"]),
        ("Enemy Active",    f"{n_alive}",    C["enemy"]),
        ("Allies Deployed", f"{active}/{P}", C["ally"]),
        ("Allies Reserve",  f"{reserve}",    C["wp"]),
        ("Allies Lost",     f"{dead}",       "#8A93A0"),
    ]
    y = 0.745
    for name, val, col in rows:
        ax.text(0.07, y, name, color="#B0BEC5", fontsize=8, ha="left", va="center")
        ax.text(0.93, y, val, color=col, fontsize=8.5, ha="right", va="center",
                fontweight="bold")
        y -= 0.108


def _draw_cam(ax, fd, view, bg_img=None, bg_extent=None):
    """클로즈업 캠 한 칸: view 가 있으면 줌 씬(선박+그물만), 없으면 placeholder.
    bg_img/bg_extent 주면 위성맵 배경의 해당 영역을 확대해서 보여준다."""
    if view is None:
        ax.clear()
        ax.set_facecolor(_THEME_DARK["sea"]); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "CAM\n(대상 없음)", transform=ax.transAxes,
                color="#6B7A90", fontsize=8, ha="center", va="center")
        for s in ax.spines.values():
            s.set_color("#6B7A90"); s.set_linewidth(1.2)
        return
    cx, cy, half, label, col = view
    # 캠은 클러스터 부채꼴/잔차원은 생략하되 위성맵 배경(bg)은 그 영역만 확대해 표시.
    cam_fd = dict(fd)
    cam_fd["show_clusters"] = False
    cam_fd["show_residual"] = False
    draw_scene(ax, cam_fd, bg_img=bg_img, bg_extent=bg_extent, show_help=False,
               view=(cx, cy, half), show_paths=False)       # 선박+그물(+위성맵 줌)
    ax.set_title(label, color=col, fontsize=8, loc="left", pad=2, fontweight="bold")
    for s in ax.spines.values():
        s.set_color(col); s.set_linewidth(1.6)


def setup_panels(fig):
    """메인 ax 위 오버레이 패널 3개 생성 후 dict 반환.
      brief(좌상단 전장 상황판) / cam_enemy(우상단 모선근접 적 클러스터 캠) /
      cam_ally(우상단 적근접 아군 캠)."""
    P = {
        "brief":     fig.add_axes([0.066, 0.715, 0.185, 0.215]),
        "cam_enemy": fig.add_axes([0.548, 0.715, 0.212, 0.215]),
        "cam_ally":  fig.add_axes([0.763, 0.715, 0.212, 0.215]),
    }
    for a in P.values():
        a.set_zorder(20)
    return P


def draw_panels(P, fd, bg_img=None, bg_extent=None):
    """오버레이 패널 3개 갱신 (매 프레임 호출). bg 주면 캠도 위성맵 배경."""
    if not fd:
        return
    draw_briefing(P["brief"], fd)
    _draw_cam(P["cam_enemy"], fd, follow_cluster_view(fd), bg_img, bg_extent)
    _draw_cam(P["cam_ally"], fd, follow_ally_view(fd), bg_img, bg_extent)
