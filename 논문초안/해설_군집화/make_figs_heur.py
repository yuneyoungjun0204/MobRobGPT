# -*- coding: utf-8 -*-
"""§4.3 기하 휴리스틱 식 (5)~(7) 해설 자료용 도판 + 예제 수치.
군집화 해설(make_figs.py)의 예제 무리 1·2 를 그대로 이어받는다."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Wedge, Circle, Rectangle
import json, os

for cand in ["Malgun Gothic", "HCR Dotum", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
C1, C2, C3, C4, GRAY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#888888"

# ── 원고·구현 상수 ──────────────────────────────────────────────
R_W = 6300.0          # 해역 반폭 (m)
LAM_I = 0.55          # 요격점 비율
BETA = 6000.0         # sticky 보너스 (m)
LAM_L = 0.18          # 잉여 배 층 증분
DBEAR = 40.0          # 층 방위 틀기 (deg)
L_NET = 450.0         # 그물벽 길이 (m)
ELL = 252.0           # 픽셀 한 변 (m)
NG = 50               # 격자
R_MIN, R_MAX, R_G, TH_G = 400.0, 4500.0, 3000.0, 60.0
V_E, V_A = 9.0, 6.0   # m/s (속력비 1.5)
P = 3
M = np.array([0.0, 0.0])

# 군집화 해설의 결과 (numbers.json 과 동일)
clusters = {1: dict(n=2, c=np.array([-2155.0, -4601.0])),
            2: dict(n=3, c=np.array([1432.0, 3902.0]))}
# 방어정 3척: 모선 함미 아래 한 줄 (ally_row_gap 550 m)
A = np.array([[-500.0, -550.0], [0.0, -550.0], [500.0, -550.0]])

def bearing(v):
    return np.degrees(np.arctan2(v[0], v[1])) % 360.0

# ── 식 (5) 위협도·요격점 ─────────────────────────────────────────
res = {}
for j, cl in clusters.items():
    d = np.linalg.norm(cl["c"] - M)
    close = float(np.clip(1 - d / R_W, 0.05, 1.0))
    tau = cl["n"] * close
    I = cl["c"] + LAM_I * (M - cl["c"])
    res[j] = dict(d=round(d), close=round(close, 3), tau=round(tau, 2), I=np.round(I).tolist(),
                  rI=round(float(np.linalg.norm(I))))
    cl["I"] = I; cl["tau"] = tau

# 근접도 곡선
fig, ax = plt.subplots(figsize=(7, 3.4))
dd = np.linspace(0, 7000, 400)
ax.plot(dd, np.clip(1 - dd / R_W, 0.05, 1), color=C1, lw=2)
ax.axhline(0.05, color=GRAY, ls=":", lw=1); ax.text(6400, 0.08, "하한 0.05", fontsize=9, color=GRAY)
ax.axvline(R_W, color=GRAY, ls="--", lw=1); ax.text(R_W + 60, 0.5, r"$R_w$ = 6300 m", fontsize=9, color=GRAY, rotation=90, va="center")
for j, cl in clusters.items():
    ax.scatter(res[j]["d"], res[j]["close"], s=70, c=[C1, C3][j - 1], zorder=3)
    ax.annotate(f"무리 {j}: d={res[j]['d']} m → 근접도 {res[j]['close']:.2f}\n"
                rf"$\tau_{j}$ = {cl['n']} × {res[j]['close']:.2f} = {res[j]['tau']:.2f}",
                (res[j]["d"], res[j]["close"]), xytext=(res[j]["d"] - 3600, res[j]["close"] + 0.22 - 0.3 * (j - 1)),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=GRAY))
ax.set_xlabel(r"무리 중심–모선 거리 $\|\mathbf{c}_j - \mathbf{m}\|$ (m)")
ax.set_ylabel(r"근접도 $\mathrm{clip}(1 - d/R_w,\ 0.05,\ 1)$")
ax.set_xlim(0, 7000); ax.set_ylim(0, 1.1)
ax.set_title("식 (5) 위협도의 근접도 항", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_h_threat.png", dpi=180); plt.close(fig)

# ── 요격점 λ_I 기하 + 도달 시간 경쟁 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
ax.set_aspect("equal"); ax.set_xlim(-3200, 3200); ax.set_ylim(-5200, 4600)
ax.add_patch(Circle(M, R_MAX, fc="none", ec=GRAY, ls=":", lw=1))
ax.scatter(*M, s=220, marker="s", c="k", zorder=4); ax.text(250, -150, "모선 m", fontsize=9)
for j, cl in clusters.items():
    col = [C1, C3][j - 1]
    ax.plot([cl["c"][0], M[0]], [cl["c"][1], M[1]], color=col, lw=1, ls="--")
    ax.scatter(*cl["c"], s=140, c=col, zorder=3)
    ax.text(cl["c"][0] + 150, cl["c"][1], rf"$\mathbf{{c}}_{j}$ (n={cl['n']})", fontsize=9, color=col)
    ax.scatter(*cl["I"], s=110, marker="X", c=col, zorder=4)
    ax.text(cl["I"][0] + 150, cl["I"][1], rf"$\mathbf{{I}}_{j}$ ({res[j]['rI']} m)", fontsize=9, color=col)
    for lam, lab in [(0.0, "λ=0"), (0.55, "λ=0.55"), (1.0, "λ=1")]:
        p = cl["c"] + lam * (M - cl["c"])
        if j == 2:
            ax.annotate(lab, p, xytext=(p[0] - 1500, p[1] + 250), fontsize=8, color=GRAY, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5))
for k, a in enumerate(A):
    ax.scatter(*a, s=80, marker="^", c=C4, zorder=4)
    ax.text(a[0] - 120, a[1] - 420, rf"$\mathbf{{a}}_{k + 1}$", fontsize=9, color=C4)
ax.set_title(r"식 (5) 요격점 $\mathbf{I}_j = \mathbf{c}_j + \lambda_I(\mathbf{m}-\mathbf{c}_j)$", fontsize=11)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

# 도달 시간 경쟁: 적은 5450 m 에서 출발, 방어정은 모선 550 m 옆에서 출발(같은 방위선 가정)
ax = axes[1]
D0 = 5450.0; a0 = 550.0
lam = np.linspace(0, 1, 200); r = (1 - lam) * D0
t_e = (D0 - r) / V_E
t_a_best = np.abs(r - a0) / V_A
t_a_worst = (r + a0) / V_A
ax.plot(lam, t_e, color=C2, lw=2, label=r"적 도달 시간 $(D-r)/v_e$")
ax.fill_between(lam, t_a_best, t_a_worst, color=C4, alpha=0.25, label="방어정 도달 시간 범위\n(출발 방위에 따라)")
ax.plot(lam, t_a_best, color=C4, lw=1.2); ax.plot(lam, t_a_worst, color=C4, lw=1.2)
# 교차점
lb = lam[np.argmin(np.abs(t_e - t_a_best))]; lw_ = lam[np.argmin(np.abs(t_e - t_a_worst))]
ax.axvline(LAM_I, color="k", ls="--", lw=1.2); ax.text(LAM_I + 0.01, 30, r"$\lambda_I$ = 0.55", fontsize=9)
ax.axvspan(lw_, 1.0, color=C3, alpha=0.08)
ax.text((lw_ + 1) / 2, 560, "방어정이 출발 방위와\n무관하게 먼저 도달", ha="center", fontsize=9, color=C3)
ax.set_xlabel(r"$\lambda_I$ (0 = 무리 중심, 1 = 모선)"); ax.set_ylabel("도달 시간 (s)")
ax.set_title(f"왜 0.55인가 — 도달 시간 경쟁 (D={D0:.0f} m, $v_e$={V_E:.0f}, $v_a$={V_A:.0f} m/s)", fontsize=10)
ax.legend(fontsize=8, loc="upper center"); ax.set_ylim(0, 1100)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_h_intercept.png", dpi=180); plt.close(fig)
res["lam_cross_best"] = round(float(lb), 2); res["lam_cross_worst"] = round(float(lw_), 2)
res["r_cross_best"] = round(float((1 - lb) * D0)); res["r_cross_worst"] = round(float((1 - lw_) * D0))
res["rI_at_D0"] = round((1 - LAM_I) * D0)

# ── 식 (6) 배정 ─────────────────────────────────────────────────
Js = sorted(clusters, key=lambda j: -clusters[j]["tau"])   # 위협 순
cost = np.array([[np.linalg.norm(a - clusters[j]["I"]) for j in Js] for a in A])   # [P, K]
def greedy(cost):
    work = cost.copy(); assign = {}
    order = []
    for _ in range(min(cost.shape)):
        p, k = np.unravel_index(np.argmin(work), work.shape)
        assign[p] = Js[k]; order.append((p, Js[k], round(float(cost[p, k]))))
        work[p, :] = np.inf; work[:, k] = np.inf
    return assign, order
assign0, order0 = greedy(cost)
# sticky 예: 직전 결정에서 a_2→무리1, a_1→무리2 였다고 하자(직전 결정 뒤 배·적이 움직여
# 지금은 거리만 보면 a_1→무리1 이 더 가까운 상황). β 가 없으면 배정이 뒤바뀐다(churn).
prev = {1: 1, 0: 2}
cost_s = cost.copy()
for p, j in prev.items():
    cost_s[p, Js.index(j)] -= BETA
assign_s, order_s = greedy(cost_s)
# 잉여 배: 1:1 뒤 남은 배 → 부하 = n/(1+배정수) 최대 무리, λ+0.18, 방위 ±40°
def surplus(assign):
    nass = {j: sum(1 for v in assign.values() if v == j) for j in clusters}
    free = [p for p in range(P) if p not in assign]
    out = {}
    for p in free:
        load = {j: clusters[j]["n"] / (1 + nass[j]) for j in clusters}
        jb = max(load, key=load.get)
        tl = np.clip(LAM_I + LAM_L * nass[jb], 0.05, 0.95)
        Ib = clusters[jb]["c"] + tl * (M - clusters[jb]["c"])
        jn = nass[jb]; sgn = 1 - 2 * (jn % 2); ang = np.deg2rad(DBEAR * sgn * ((jn + 1) // 2))
        rel = Ib - M; ca, sa = np.cos(ang), np.sin(ang)
        Ib = M + np.array([rel[0] * ca - rel[1] * sa, rel[0] * sa + rel[1] * ca])
        out[p] = dict(j=jb, load=load, lam=float(tl), I=np.round(Ib).tolist(), ang=float(np.degrees(ang)))
        nass[jb] += 1
    return out
sur0 = surplus(assign0); sur_s = surplus(assign_s)
res["Js"] = Js
res["cost"] = np.round(cost).astype(int).tolist()
res["greedy"] = order0; res["assign"] = {int(k): int(v) for k, v in assign0.items()}
res["cost_sticky"] = np.round(cost_s).astype(int).tolist()
res["greedy_sticky"] = order_s; res["assign_sticky"] = {int(k): int(v) for k, v in assign_s.items()}
res["surplus"] = {int(k): v for k, v in sur0.items()}
res["surplus_sticky"] = {int(k): v for k, v in sur_s.items()}

# 배정 그림
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (assign, sur, title) in zip(axes, [(assign0, sur0, "(a) 탐욕 1:1 배정 + 잉여 배 층 배정"),
                                            (assign_s, sur_s, "(b) 직전 배정(a₂→무리1, a₁→무리2)에 유지 보너스 β 적용")]):
    ax.set_aspect("equal"); ax.set_xlim(-3400, 3400); ax.set_ylim(-5200, 4600)
    ax.scatter(*M, s=220, marker="s", c="k", zorder=4)
    for j, cl in clusters.items():
        col = [C1, C3][j - 1]
        ax.scatter(*cl["c"], s=140, c=col, zorder=3); ax.text(cl["c"][0] + 150, cl["c"][1], rf"무리 {j} ($\tau$={cl['tau']:.2f})", fontsize=9, color=col)
        ax.scatter(*cl["I"], s=110, marker="X", c=col, zorder=4)
    for p, a in enumerate(A):
        ax.scatter(*a, s=90, marker="^", c=C4, zorder=5)
        ax.text(a[0] - 120, a[1] - 420, rf"$\mathbf{{a}}_{p + 1}$", fontsize=9, color=C4)
        if p in assign:
            I = clusters[assign[p]]["I"]; col = [C1, C3][assign[p] - 1]
            ax.annotate("", xy=I, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
            ax.text((a[0] + I[0]) / 2 + 100, (a[1] + I[1]) / 2, f"{np.linalg.norm(a - I):.0f} m", fontsize=8, color=col)
        elif p in sur:
            Ib = np.array(sur[p]["I"]); col = [C1, C3][sur[p]["j"] - 1]
            ax.scatter(*Ib, s=110, marker="X", c=col, zorder=4, alpha=0.6)
            ax.annotate("", xy=Ib, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, ls="--"))
            ax.text(Ib[0] + 150, Ib[1] - 250, rf"층 요격점 ($\lambda$={sur[p]['lam']:.2f}, {sur[p]['ang']:+.0f}°)", fontsize=8, color=col)
    ax.set_title(title, fontsize=10); ax.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_h_assign.png", dpi=180); plt.close(fig)

# ── 식 (7) 경유점 + 스냅 + 유효 마스크 ─────────────────────────────
j = 2; I = clusters[j]["I"]
rhat = I / np.linalg.norm(I); perp = np.array([-rhat[1], rhat[0]])
K = 2
t = [I + (k - (K - 1) / 2) * L_NET * perp for k in range(K)]
centers = -R_W + ELL / 2 + ELL * np.arange(NG)            # 픽셀 중심
XX, YY = np.meshgrid(centers, centers)
rr = np.hypot(XX, YY); thI = bearing(I); th = np.degrees(np.arctan2(XX, YY)) % 360
dth = np.abs((th - thI + 180) % 360 - 180)
valid = (rr >= R_MIN) & (rr <= R_MAX) & (np.hypot(XX - I[0], YY - I[1]) <= R_G) & (dth <= TH_G)
snap = []
avail = valid.copy()
for k in range(K):
    d2 = np.where(avail, (XX - t[k][0]) ** 2 + (YY - t[k][1]) ** 2, np.inf)
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    snap.append(np.array([XX[iy, ix], YY[iy, ix]]))
    avail &= np.hypot(XX - XX[iy, ix], YY - YY[iy, ix]) > ELL * 0.5   # 같은 픽셀 재선택 방지
res["I2"] = np.round(I).tolist(); res["rhat"] = np.round(rhat, 3).tolist(); res["perp"] = np.round(perp, 3).tolist()
res["t"] = [np.round(x).tolist() for x in t]; res["snap"] = [np.round(x).tolist() for x in snap]
res["snap_gap"] = round(float(np.linalg.norm(snap[0] - snap[1])))
res["t_gap"] = round(float(np.linalg.norm(t[0] - t[1])))
res["n_valid"] = int(valid.sum()); res["thI"] = round(float(thI), 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
ax = axes[0]
ax.set_aspect("equal")
edges = -R_W + ELL * np.arange(NG + 1)
ax.pcolormesh(edges, edges, valid.astype(float), cmap="Greens", vmin=0, vmax=2.5, edgecolors="none")
ax.add_patch(Circle(M, R_MIN, fc="none", ec=GRAY, lw=1)); ax.add_patch(Circle(M, R_MAX, fc="none", ec=GRAY, lw=1))
ax.add_patch(Circle(I, R_G, fc="none", ec=C2, lw=1, ls="--"))
w0 = 90 - thI  # 수학각
ax.add_patch(Wedge(M, R_MAX + 300, w0 - TH_G, w0 + TH_G, fc="none", ec=C1, lw=1, ls=":"))
ax.scatter(*M, s=150, marker="s", c="k", zorder=4)
ax.scatter(*clusters[j]["c"], s=120, c=C3, zorder=3); ax.text(clusters[j]["c"][0] + 200, clusters[j]["c"][1], r"$\mathbf{c}_2$", fontsize=9, color=C3)
ax.scatter(*I, s=110, marker="X", c=C3, zorder=4); ax.text(I[0] + 200, I[1], r"$\mathbf{I}_2$", fontsize=9, color=C3)
ax.set_xlim(-R_W, R_W); ax.set_ylim(-R_W, R_W)
ax.set_title(f"유효 픽셀 $\\mathcal{{V}}_p$ (초록, {int(valid.sum())}개): 환형 {R_MIN:.0f}–{R_MAX:.0f} m ∩ 요격점 반경 {R_G:.0f} m ∩ 방위 ±{TH_G:.0f}°", fontsize=9)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax = axes[1]
ax.set_aspect("equal")
lo, hi = I - 900, I + 900
sel = (centers > lo[0] - ELL) & (centers < hi[0] + ELL)
sely = (centers > lo[1] - ELL) & (centers < hi[1] + ELL)
for cx in centers[sel]:
    for cy in centers[sely]:
        v = valid[np.argmin(np.abs(centers - cy)), np.argmin(np.abs(centers - cx))]
        ax.add_patch(Rectangle((cx - ELL / 2, cy - ELL / 2), ELL, ELL, fc="#e4f2e4" if v else "#f0f0f0", ec="w", lw=0.8))
        ax.scatter(cx, cy, s=6, c=GRAY, zorder=2)
ax.scatter(*I, s=130, marker="X", c=C3, zorder=5); ax.text(I[0] + 60, I[1] + 60, r"$\mathbf{I}$", fontsize=10, color=C3)
ax.annotate("", xy=I + rhat * 700, xytext=I, arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))
ax.text(*(I + rhat * 760), r"$\hat{\mathbf{r}}$ (모선→요격점)", fontsize=8, color=GRAY)
ax.plot([t[0][0], t[1][0]], [t[0][1], t[1][1]], color=C2, lw=3, label=f"의도한 벽 $L_{{net}}$={L_NET:.0f} m")
for k in range(K):
    ax.scatter(*t[k], s=70, c=C2, zorder=6); ax.text(t[k][0] + 40, t[k][1] - 130, rf"$\mathbf{{t}}_{k}$", fontsize=9, color=C2)
    ax.scatter(*snap[k], s=120, marker="s", fc="none", ec=C1, lw=2, zorder=6)
    ax.annotate("", xy=snap[k], xytext=t[k], arrowprops=dict(arrowstyle="->", color=C1, lw=1))
ax.plot([snap[0][0], snap[1][0]], [snap[0][1], snap[1][1]], color=C1, lw=2, ls="--", label=f"스냅된 벽 {res['snap_gap']} m")
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.legend(fontsize=8, loc="lower left")
ax.set_title(f"식 (7) 경유점과 {ELL:.0f} m 픽셀 스냅 (요격점 주변 확대)", fontsize=10); ax.set_xlabel("x (m)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_h_waypoint.png", dpi=180); plt.close(fig)

# ── 파이프라인 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 2.3)); ax.axis("off")
steps = ["§4.2 무리 요약\n$n_j,\\ \\mathbf{c}_j$", "식 (5)\n위협도 $\\tau_j$", "위협 상위\n$P$개 선택", "식 (5)\n요격점 $\\mathbf{I}_j$",
         "식 (6)\n탐욕 1:1 배정\n(+직전 유지 $\\beta$)", "잉여 방어정\n안쪽 층 배정", "식 (7)\n벽 양끝 $\\mathbf{t}_k$", "유효 픽셀\n최근접 스냅"]
for i, s in enumerate(steps):
    x = i * 1.53
    ax.add_patch(Rectangle((x, 0), 1.38, 1, fc="#f3f3f3" if i in (0, 7) else "#fff3e0", ec="#864"))
    ax.text(x + 0.69, 0.5, s, ha="center", va="center", fontsize=8.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 1.53, 0.5), xytext=(x + 1.38, 0.5), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.set_xlim(-0.1, 12.2); ax.set_ylim(-0.1, 1.1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_h_pipeline.png", dpi=180); plt.close(fig)

def conv(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    return o
json.dump(conv(res), open(f"{OUT}/numbers_heur.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(conv(res), ensure_ascii=False, indent=1))
