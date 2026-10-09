# -*- coding: utf-8 -*-
"""식 (1)~(4) 해설 자료용 도판 + 예제 수치 계산."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import json, os

for cand in ["Malgun Gothic", "HCR Dotum", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
C1, C2, C3, GRAY = "#1f77b4", "#d62728", "#2ca02c", "#888888"
cols = [C1, C3, "#ff7f0e", "#9467bd"]
DELTA = 12.0


def polar_ax(ax, title=None):
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rticks([]); ax.set_ylim(0, 1.25)
    ax.set_thetagrids(range(0, 360, 45), [f"{d}°" for d in range(0, 360, 45)])
    if title: ax.set_title(title, pad=14)


def gaps_of(sorted_deg):
    n = len(sorted_deg)
    g = [sorted_deg[q + 1] - sorted_deg[q] for q in range(n - 1)]
    g.append(sorted_deg[0] + 360 - sorted_deg[-1])
    return g


def cluster(deg, delta, C=4):
    """논문 식 (2)(3) + 라벨링 문장 그대로. 반환: 정렬방위, 간격, Ĉ, 이음새 idx, 경계 mask, 원래순서 라벨."""
    deg = np.asarray(deg, float); order = np.argsort(deg); sb = deg[order]; n = len(sb)
    g = np.array(gaps_of(sb))
    Chat = int(np.clip((g > delta).sum(), 1, min(C, n)))
    desc = np.argsort(-g); boundary = np.zeros(n, bool); boundary[desc[:Chat]] = True
    seam = desc[0]
    lbl_sorted = np.zeros(n, int); cur = 0
    for k in range(n):
        pos = (seam + 1 + k) % n
        if k >= 1 and boundary[(seam + k) % n]:
            cur += 1
        lbl_sorted[pos] = cur
    labels = np.empty(n, int); labels[order] = lbl_sorted
    return sb, g, Chat, seam, boundary, labels, lbl_sorted


def resultant(degs):
    v = np.stack([np.cos(np.deg2rad(degs)), np.sin(np.deg2rad(degs))], 1)
    return v, v.mean(0)


# ---------- 예제 1: 5척, 두 무리 ----------
ex1 = [200, 10, 210, 30, 20]          # 적선 번호 m=1..5 (정렬 전)
sb1, g1, Chat1, seam1, bnd1, lbl1, lbls1 = cluster(ex1, DELTA)

fig = plt.figure(figsize=(11, 4.8))
ax = fig.add_subplot(1, 2, 1, projection="polar"); polar_ax(ax, "(a) 식 (1) 방위각 → 식 (2) 정렬·간격")
for q, d in enumerate(sb1):
    ax.scatter(np.deg2rad(d), 1.0, s=90, c=C1, zorder=3)
    ax.annotate(rf"$\vartheta_{{({q + 1})}}$", (np.deg2rad(d), 1.0), xytext=(np.deg2rad(d), 1.13), ha="center", fontsize=9)
n1 = len(sb1)
for q in range(n1):
    a = sb1[q]; b = sb1[(q + 1) % n1] + (360 if q == n1 - 1 else 0)
    big = g1[q] > DELTA
    arc = np.deg2rad(np.linspace(a, b, 40)); r = 0.78 if big else 0.92
    col = C2 if big else GRAY
    ax.plot(arc, np.full_like(arc, r), color=col, lw=2.5 if big else 1.5)
    mid = np.deg2rad((a + b) / 2)
    rt = r - 0.24 if big else (0.62 - 0.17 * (q % 2))
    ax.annotate(f"$g_{{({q + 1})}}$={g1[q]:.0f}°", (mid, r), xytext=(mid, rt), ha="center", fontsize=9, color=col,
                arrowprops=None if big else dict(arrowstyle="-", color=col, lw=0.6))
ax.text(0, 0, "모선", ha="center", va="center", fontsize=9, bbox=dict(boxstyle="circle", fc="w"))

ax = fig.add_subplot(1, 2, 2, projection="polar"); polar_ax(ax, rf"(b) 식 (3) $\hat C$={Chat1} → 라벨링 → 무리 확정")
seen = set()
for d, l in zip(sb1, lbls1):
    ax.scatter(np.deg2rad(d), 1.0, s=90, c=cols[l], zorder=3, label=None if l in seen else f"무리 {l + 1}"); seen.add(l)
ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.08), fontsize=9)
for q in np.where(bnd1)[0]:
    a = sb1[q]; b = sb1[(q + 1) % n1] + (360 if q == n1 - 1 else 0)
    mid = np.deg2rad((a + b) / 2)
    ax.plot([mid, mid], [0.55, 1.1], color=C2, lw=2, ls="--")
    ax.annotate("이음새\n(최대 간격)" if q == seam1 else "경계", (mid, 0.45), ha="center", fontsize=9, color=C2)
ax.text(0, 0, "모선", ha="center", va="center", fontsize=9, bbox=dict(boxstyle="circle", fc="w"))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_example1.png", dpi=180); plt.close(fig)

# ---------- 예제 2: 북쪽 걸침, 고정 bin vs gap ----------
ex2 = [350, 355, 5, 10, 180, 190]
sb2, g2, Chat2, seam2, bnd2, lbl2, lbls2 = cluster(ex2, DELTA)
fig = plt.figure(figsize=(11, 4.8))
ax = fig.add_subplot(1, 2, 1, projection="polar"); polar_ax(ax, "(a) 고정 90° 구간(bin): 북쪽 무리가 둘로 쪼개짐")
for k in range(4):
    ax.bar(np.deg2rad(k * 90 + 45), 1.25, width=np.deg2rad(90), bottom=0,
           color=["#dfe9f5", "#fde2e2", "#e2f5e2", "#f0e2f5"][k], alpha=0.9, edgecolor="w")
for d in ex2:
    b = int(d // 90)
    ax.scatter(np.deg2rad(d), 1.0, s=90, c=cols[b], zorder=3)
ax.annotate("빈 0 | 빈 3 경계(0°)에서\n한 무리가 보라/파랑으로 갈림", (0, 0.95), xytext=(np.deg2rad(100), 0.55), fontsize=9, color=C2,
            arrowprops=dict(arrowstyle="->", color=C2))
ax = fig.add_subplot(1, 2, 2, projection="polar"); polar_ax(ax, rf"(b) 간격(gap) 군집화: 이음새 항 덕에 한 무리 ($\hat C$={Chat2})")
for d, l in zip(ex2, lbl2):
    ax.scatter(np.deg2rad(d), 1.0, s=90, c=cols[l], zorder=3)
arc = np.deg2rad(np.linspace(355, 365, 20)); ax.plot(arc, np.full_like(arc, 0.85), color=C3, lw=3)
ax.annotate(r"$g_{(n)}$ = 5 + 360 - 355 = 10°" + "\n(이음새 항, 임계 12° 미만 → 안 자름)", (np.deg2rad(0), 0.85),
            xytext=(np.deg2rad(100), 0.55), fontsize=9, color=C3, arrowprops=dict(arrowstyle="->", color=C3))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_northwrap.png", dpi=180); plt.close(fig)

# ---------- 원형 결집도 ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
cases = [("모임: 10°, 20°, 30°", [10, 20, 30]), ("중간: 0°, 60°, 120°", [0, 60, 120]), ("흩어짐: 0°, 120°, 240°", [0, 120, 240])]
Rvals = []
for ax, (t, degs) in zip(axes, cases):
    v, mean = resultant(degs); R = float(np.hypot(*mean)); Rvals.append(R)
    circ = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(circ), np.sin(circ), color=GRAY, lw=0.8)
    for vi in v:
        ax.annotate("", xy=vi, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C1, lw=1.5))
    if R > 1e-6:
        ax.annotate("", xy=mean, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C2, lw=3))
    else:
        ax.scatter(0, 0, s=80, c=C2, zorder=4)
    ax.set_aspect("equal"); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.axis("off")
    sp = 90 * np.sqrt(max(0, 1 - R))
    ax.set_title(t + "\n" + rf"$\bar R$ = {R:.2f} → $\Delta\vartheta = 90^\circ\sqrt{{1-\bar R}}$ = {sp:.1f}°", fontsize=10)
fig.text(0.5, 0.02, r"파랑: 방위 단위벡터 $\hat{\mathbf{b}}_m=(\cos\vartheta_m,\ \sin\vartheta_m)$      빨강: 그 평균 벡터(길이 = $\bar R$)", ha="center", fontsize=9)
fig.subplots_adjust(top=0.8, bottom=0.1, left=0.02, right=0.98); fig.savefig(f"{OUT}/fig_resultant.png", dpi=180); plt.close(fig)

# ---------- 접근 속도 기하 ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
m = np.array([0.0, 0.0]); e = np.array([0.0, 3.0])
for ax, (t, psi) in zip(axes, [("정면 접근  ψ = 180°", 180), ("비스듬  ψ = 225°", 225), ("이탈  ψ = 0°", 0)]):
    h = np.array([np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi))])
    d = (m - e) / np.linalg.norm(m - e); dot = float(h @ d)
    ax.scatter(*m, s=200, marker="s", c="k", zorder=3); ax.text(0.25, -0.15, "모선 m", fontsize=9)
    ax.scatter(*e, s=120, c=C2, zorder=3); ax.text(0.25, 3.05, r"적선 $\mathbf{e}_m$", fontsize=9)
    ax.annotate("", xy=e + h * 1.3, xytext=e, arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.5))
    tip = e + h * 1.3
    ha = "left" if h[0] >= 0 else "right"; off = 0.15 if h[0] >= 0 else -0.15
    ax.text(tip[0] + off, tip[1], "침로 벡터\n" + r"$(\sin\psi_m,\ \cos\psi_m)$", fontsize=8, color=C2, ha=ha, va="center")
    ax.annotate("", xy=e + d * 1.3 + [-0.12, 0], xytext=e + [-0.12, 0], arrowprops=dict(arrowstyle="-|>", color=C1, lw=2, ls="--"))
    ax.text(-0.35, 1.5, r"$\hat{\mathbf{d}}_m$" + "\n(적→모선)", fontsize=8, color=C1, ha="right")
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1, 5.2); ax.axis("off")
    ax.set_title(t + "\n" + rf"$\langle$침로, $\hat{{\mathbf{{d}}}}_m\rangle$ = {dot:+.2f}  →  $v^{{\rm app}}$ = {dot:+.2f}$\,v_e$", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_approach.png", dpi=180); plt.close(fig)

# ---------- 파이프라인 흐름도 ----------
fig, ax = plt.subplots(figsize=(12, 2.3)); ax.axis("off")
steps = ["생존 적선\n위치 $\\mathbf{e}_m$, 침로 $\\psi_m$", "식 (1)\n방위각 $\\vartheta_m$", "식 (2)\n정렬·원형 간격 $g_{(q)}$",
         "식 (3)\n무리 수 $\\hat C$", "라벨링\n$j_m,\\ \\mathcal{A}_j$", "식 (4)\n$n_j,\\ \\mathbf{c}_j,\\ \\Delta\\vartheta_j,\\ v^{\\rm app}_j$",
         "관측 채널·위협도\n(§4.3, §4.4)"]
for i, s in enumerate(steps):
    x = i * 1.75
    ax.add_patch(plt.Rectangle((x, 0), 1.55, 1, fc="#eef3fa" if 0 < i < 6 else "#f3f3f3", ec="#345"))
    ax.text(x + 0.775, 0.5, s, ha="center", va="center", fontsize=9)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 1.75, 0.5), xytext=(x + 1.55, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.set_xlim(-0.1, 12.3); ax.set_ylim(-0.1, 1.1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png", dpi=180); plt.close(fig)

# ---------- 예제 1 식 (4) 수치 ----------
dist = {10: 4200, 20: 4000, 30: 4400, 200: 5000, 210: 5200}   # 모선까지 거리 (m)
psi = {10: 190, 20: 200, 30: 215, 200: 20, 210: 60}           # 침로 (북 0°, 시계)
v_e = 10.0
rows = []
for d in sb1:
    x = dist[d] * np.sin(np.deg2rad(d)); y = dist[d] * np.cos(np.deg2rad(d))
    rows.append(dict(bearing=float(d), dist=dist[d], x=round(x), y=round(y), psi=psi[d]))
summary = {}
for j in sorted(set(lbls1)):
    ds = [float(d) for d, l in zip(sb1, lbls1) if l == j]
    pos = np.array([[dist[d] * np.sin(np.deg2rad(d)), dist[d] * np.cos(np.deg2rad(d))] for d in ds])
    c = pos.mean(0)
    v, mean = resultant(ds); R = float(np.hypot(*mean)); spread = 90 * np.sqrt(1 - R)
    dots = []
    for d, p in zip(ds, pos):
        h = np.array([np.sin(np.deg2rad(psi[d])), np.cos(np.deg2rad(psi[d]))]); dd = (0 - p) / np.linalg.norm(p)
        dots.append(float(h @ dd))
    summary[int(j) + 1] = dict(members=ds, n=len(ds), c=[round(float(c[0])), round(float(c[1]))],
                               meancos=round(float(mean[0]), 4), meansin=round(float(mean[1]), 4),
                               R=round(R, 4), spread=round(float(spread), 2),
                               dots=[round(t, 3) for t in dots], vapp=round(v_e * float(np.mean(dots)), 2))
out = dict(ex1_input=ex1, ex1_sorted=sb1.tolist(), ex1_gaps=[round(float(x), 1) for x in g1], Chat=Chat1,
           seam=int(seam1) + 1, boundaries=[int(q) + 1 for q in np.where(bnd1)[0]], labels_sorted=(lbls1 + 1).tolist(),
           ex2_input=ex2, ex2_sorted=sb2.tolist(), ex2_gaps=[round(float(x), 1) for x in g2], Chat2=Chat2,
           labels2=(lbl2 + 1).tolist(), rows=rows, summary=summary, Rcases=[round(r, 3) for r in Rvals])
json.dump(out, open(f"{OUT}/numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
