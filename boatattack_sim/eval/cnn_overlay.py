"""
boatattack_sim/eval/cnn_overlay.py — CNN 점수맵 오버레이 공용 유틸

점수맵을 **지도 좌표 그대로** 겹쳐 그린다. 축 규약은 env/cnn_map.py 단일 소스를 따르며,
그릴 때만 전치한다: imshow(arr.T, origin="lower", extent=CM.extent(cfg)).
(전치/origin 을 빠뜨리면 90° 회전·상하반전 버그가 확정적으로 난다 — cnn_map.md §A)

cnn_score_view / make_cnn_gif / cnn_channels 가 공유한다.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from ..env import cnn_map as CM

SHIP_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#A78BFA", "#FF9F43", "#7BE0AD"]


def ship_color(p):
    return SHIP_COLORS[p % len(SHIP_COLORS)]


def ship_cmap(p):
    """투명 → 배 색상 그라디언트 (점수맵 heatmap 용)."""
    r, g, b = to_rgb(ship_color(p))
    return LinearSegmentedColormap.from_list(
        f"ship{p}", [(r, g, b, 0.0), (r, g, b, 0.55), (r, g, b, 0.95)])


def imshow_map(ax, cfg, arr, cmap="magma", alpha=1.0, vmin=None, vmax=None, z=2.0):
    """[n,n] 배열을 world 좌표에 겹쳐 그린다 (축 규약 준수)."""
    return ax.imshow(np.asarray(arr).T, origin="lower", extent=CM.extent(cfg),
                     cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                     zorder=z, interpolation="nearest")


def draw_score(ax, cfg, prob, p, gamma=0.45, alpha=0.75, z=2.2):
    """배 p 의 점수맵(확률) heatmap. gamma<1 로 감마보정해 넓은 확률 꼬리도 보이게 한다.
    마스크 밖(확률 0)은 완전 투명."""
    a = np.asarray(prob, np.float64)
    mx = a.max()
    if mx <= 0:
        return None
    a = (a / mx) ** gamma
    a = np.where(np.asarray(prob) > 0, a, np.nan)          # 0 = 마스크 밖 → 투명
    return imshow_map(ax, cfg, a, cmap=ship_cmap(p), alpha=alpha, vmin=0.0, vmax=1.0, z=z)


def draw_valid(ax, cfg, valid, p, alpha=0.16, z=2.0):
    """유효 마스크(선택 가능 영역) 옅은 색칠 — 행동공간을 눈으로 확인."""
    a = np.where(np.asarray(valid), 1.0, np.nan)
    return imshow_map(ax, cfg, a, cmap=ship_cmap(p), alpha=alpha, vmin=0.0, vmax=1.0, z=z)


def draw_picks(ax, cfg, pix, offset=None, p=0, z=7.0, label=None):
    """선택 픽셀 K개 → world 마커 + 이들을 잇는 그물 벽 선분."""
    w = CM.flat_to_world(cfg, np.asarray(pix), offset)      # [K,2]
    col = ship_color(p)
    ax.plot(w[:, 0], w[:, 1], "-", color=col, lw=2.0, alpha=0.85, zorder=z)
    ax.scatter(w[:, 0], w[:, 1], s=90, marker="X", facecolors=col,
               edgecolors="white", lw=1.4, zorder=z + 0.5, label=label)
    return w


def enable_land(cfg, actor=None, tag="viz", source=None):
    """뷰어에서 `--land` 를 켤 때의 공통 처리 + **구 체크포인트 자동 강등**.

    육지를 켜면 관측 채널이 하나 늘어난다(13 → 14). 육지 없이 학습된 actor 는 stem 이
    13채널이라 그대로 넣으면 shape 오류가 난다. 그럴 때 **관측 채널만 끄고**
    (행동 마스크·적 APF 우회는 유지) 진행한다 — 정책은 섬을 '보지 못하지만'
    섬 위에 그물을 놓을 수도 없고 적은 섬을 돌아간다. 무엇이 강등됐는지 반드시 찍는다."""
    from ..env.rasterizer import obs_channels
    cfg.land_obstacle = True
    if source:
        cfg.land_source = source
    if actor is None:
        return cfg
    Cg, Cs, Cc = obs_channels(cfg)
    if (Cg, Cs, Cc) != (actor.Cg, actor.Cs, actor.Cc):
        cfg.cnn_ch_land = False
        print(f"[{tag}] ! 이 체크포인트는 육지 채널 없이 학습됐다 "
              f"(actor {actor.Cg}+{actor.Cs}+{actor.Cc} vs 요청 {Cg}+{Cs}+{Cc}) "
              f"→ 관측 채널만 끄고 행동 마스크·적 APF 우회는 유지한다")
    return cfg


def draw_land(ax, cfg, env=None, world=0, z=1.4, alpha=0.42, action_alpha=0.20):
    """육지(섬)를 두 겹으로 표시 — 물리격자(충돌 판정)와 점수맵(행동 후보 제외).

    위성 배경만 깔면 '보이는 섬'과 '물리적으로 막힌 섬'이 같은지 알 수 없다.
    실제 판정에 쓰이는 마스크를 그대로 겹쳐 그 둘을 눈으로 대조하게 한다.

    ★ `env` 를 주면 **그 월드가 실제로 서 있는 해역**의 마스크를 쓴다(지형 로테이션).
      안 주면 `cfg.geo_lat/lon` 앵커로 새로 받는데, 로테이션이 켜져 있으면 그건
      **항상 기본 앵커의 섬**이라 화면과 물리가 어긋난다(2026-08-19 실측: 6개 해역
      패널에 전부 같은 섬이 그려졌고, 육지 0 % 인 외해에도 섬이 나왔다).
      env 의 land_grid/land_map 은 물리·마스크가 쓰는 바로 그 배열이라 어긋날 수 없다."""
    from ..env import terrain as TR
    if not bool(getattr(cfg, "land_obstacle", False)):
        return
    if env is not None and getattr(env, "land_grid", None) is not None:
        si = int(env.world_site[int(world)]) if hasattr(env, "world_site") else 0
        lg = env.land_grid[si]
        ln = (env.land_map[si] if getattr(env, "land_map", None) is not None
              and env.land_map.shape[1:] == (CM.grid_n(cfg),) * 2 else None)
    else:
        lg = TR.land_mask(cfg, cfg.grid_size)
        ln = TR.land_mask(cfg, CM.grid_n(cfg))
    if not lg.any():
        return
    ext = CM.extent(cfg)
    W = float(cfg.world_size)
    full = (0.0, W, 0.0, W)
    ax.imshow(np.where(lg, 1.0, np.nan).T, origin="lower", extent=full,
              cmap="autumn", alpha=alpha, zorder=z, interpolation="nearest")
    if getattr(cfg, "land_mask_action", True) and ln is not None and ln.any():
        ax.imshow(np.where(ln, 1.0, np.nan).T, origin="lower", extent=ext,
                  cmap="cool", alpha=action_alpha, zorder=z + 0.05,
                  interpolation="nearest")


def draw_pixel_grid(ax, cfg, step=10, z=1.6, color="#FFD54F", alpha=0.18):
    """점수맵 픽셀 격자선 (step 픽셀마다) — 해상도를 눈으로 확인."""
    x0, x1, y0, y1 = CM.extent(cfg)
    n = CM.grid_n(cfg); ps = CM.px_size(cfg)
    for i in range(0, n + 1, step):
        ax.plot([x0 + i * ps, x0 + i * ps], [y0, y1], color=color, lw=0.5, alpha=alpha, zorder=z)
        ax.plot([x0, x1], [y0 + i * ps, y0 + i * ps], color=color, lw=0.5, alpha=alpha, zorder=z)


def zoom_to_action(ax, env, cfg, valid=None, pix=None, pad=700.0, min_half=2400.0):
    """실제 행동이 일어나는 영역(유효 마스크 ∪ 아군 ∪ 선택 ∪ 살아있는 적)에 맞춰 확대.
    전 맵(12.6km)은 대부분 여백이라 그대로 두면 정작 봐야 할 기하가 안 보인다."""
    c = np.asarray(env.center, float)
    R = [min_half]
    aa = env.a_alive[0]
    if aa.any():
        d = np.hypot(*(env.a_pos[0][aa] - c).T)
        R.append(float(d.max()))
    ea = env.e_alive[0]
    if ea.any():
        d = np.hypot(*(env.e_pos[0][ea] - c).T)
        R.append(float(np.percentile(d, 80)))          # 멀리 남은 낙오 적은 무시
    if valid is not None and np.asarray(valid).any():
        Rg, _ = CM.pixel_polar(cfg)
        v = np.asarray(valid)
        v = v.any(0) if v.ndim == 3 else v
        R.append(float(Rg[v].max()))
    if pix is not None:
        w = CM.flat_to_world(cfg, np.asarray(pix).ravel())
        R.append(float(np.hypot(*(w - c).T).max()))
    h = min(max(R) + pad, float(cfg.world_size) * 0.5)
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
    return h


def policy_maps(env, actor, device="cpu", world=0, joint=False, joint_mask_r=None):
    """정책의 단계별 점수맵·선택을 단일 월드에 대해 뽑는다.
    joint_mask_r: 추론 cross-ship 분리 반경(m). None=cfg 기본(겹침 레짐이면 자동 0).
    반환 (prob[K,P,n,n], pix[P,K], offset[P,K,2]|None, act_for_env)."""
    import torch
    from ..model.cnn_actor import cnn_obs_to_torch
    with torch.no_grad():
        p, _ = actor(cnn_obs_to_torch(env.build_cnn_obs(), device))
        a = (actor.greedy_joint(p, env.N, env.P, joint_mask_r) if joint else actor.greedy(p))
        st = actor.score_stages(p, picks=a["pix"])
    K = actor.K; N, P = env.N, env.P
    n = CM.grid_n(env.cfg)
    prob = st["prob"].view(K, N, P, n, n)[:, world].cpu().numpy()
    pix = a["pix"].view(N, P, K)[world].cpu().numpy()
    off = (a["offset"].view(N, P, K, 2)[world].cpu().numpy()
           if a.get("offset") is not None else None)
    act = {"pix": a["pix"].view(N, P, K).cpu().numpy()}
    if a.get("offset") is not None:
        act["offset"] = a["offset"].view(N, P, K, 2).cpu().numpy()
    return prob, pix, off, act


def heuristic_maps(env, world=0):
    """정책이 없을 때(휴리스틱 재생) 대응물: 유효 마스크 + 휴리스틱 픽셀.
    ★ 점수맵이 아니라 '행동공간'을 보여주는 것임을 호출부에서 명시할 것."""
    hp = env.heuristic_pix()
    valid = env._cnn_valid_mask()[world]                    # [P,n,n]
    act = {"pix": hp}
    return valid, hp[world], act
