#!/usr/bin/env python3
"""All curve families on ONE canvas, shared axes (1-78, 150-view, audit-corrected).

Series: HAC++ (5 pts incl. tight-lambda extension), ours base (3 pts),
progressive layered bitstream in TWO lambda families — each family is ONE
trained model cut into 4 prefixes, control vs +B3.
Bytes: level-major format + per-prefix params accounting (2026-09-28 audit).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HACPP = "#8a8a8a"
BASE = "#2e7d32"
FAM05_C, FAM05_B3 = "#1f6fb2", "#4fa3e8"   # lam 0.0005 family: navy pair
FAM04_C, FAM04_B3 = "#b3501e", "#e8863c"   # lam 0.004 family: rust pair

hacpp = [(12.10, 26.868), (16.34, 27.633), (20.03, 27.841), (26.27, 28.211), (40.56, 28.678)]
base = [(10.67, 27.394), (13.90, 27.695), (19.94, 27.813)]
f05_ctrl = [(15.72, 25.826), (20.88, 27.040), (24.66, 27.406), (30.48, 27.601)]
f05_b3 = [(15.97, 26.085), (19.49, 27.111), (22.11, 27.492), (26.23, 27.684)]
f04_ctrl = [(6.81, 18.817), (9.94, 23.908), (12.11, 26.265), (15.14, 27.251)]
f04_b3 = [(7.13, 19.733), (9.46, 23.747), (11.41, 26.183), (14.33, 27.225)]


def xs(p):
    return [x for x, _ in p]


def ys(p):
    return [y for _, y in p]


fig, ax = plt.subplots(figsize=(11.8, 7.0))

ax.axvspan(4.5, 12.10, color=FAM04_B3, alpha=0.07)
ax.text(8.2, 28.35, "<12.1 MB：HAC++ 无产品\n（我方独占区）", fontsize=10,
        color="#9a4317", ha="center",
        bbox=dict(fc="white", ec="#d8b7a5", alpha=0.85, boxstyle="round,pad=0.4"))

ax.plot(xs(hacpp), ys(hacpp), "D-", color=HACPP, lw=1.9, ms=8,
        label="HAC++（5 个独立完整文件，λ 延伸至 12.1 MB）")
ax.plot(xs(base), ys(base), "o-", color=BASE, lw=1.9, ms=7,
        label="我们 base 单码率（3 个独立文件）")

ax.plot(xs(f05_ctrl), ys(f05_ctrl), "s-", color=FAM05_C, lw=1.7, ms=7,
        label="渐进 λ0.0005 家族·对照（一个模型切 4 档）")
ax.plot(xs(f05_b3), ys(f05_b3), "s--", color=FAM05_B3, lw=2.0, ms=8, mfc="none",
        label="渐进 λ0.0005 家族·B3（w=0.01）")
ax.plot(xs(f04_ctrl), ys(f04_ctrl), "^-", color=FAM04_C, lw=1.7, ms=7,
        label="渐进 λ0.004 家族·对照（一个模型切 4 档）")
ax.plot(xs(f04_b3), ys(f04_b3), "^--", color=FAM04_B3, lw=2.0, ms=8, mfc="none",
        label="渐进 λ0.004 家族·B3（w=0.05）")

ax.annotate("P0", f05_ctrl[0], textcoords="offset points", xytext=(-26, -10),
            fontsize=8.5, color="#2a6da8")
ax.annotate("P0", f05_b3[0], textcoords="offset points", xytext=(11, -2),
            fontsize=8.5, color="#2a6da8")
ax.annotate("P3", f05_ctrl[-1], textcoords="offset points", xytext=(6, 4),
            fontsize=8.5, color="#2a6da8")
ax.annotate("P3", f05_b3[-1], textcoords="offset points", xytext=(5, 7),
            fontsize=8.5, color="#2a6da8")
ax.annotate("P0", f04_ctrl[0], textcoords="offset points", xytext=(-17, -6),
            fontsize=8.5, color="#b3501e")
ax.annotate("P0", f04_b3[0], textcoords="offset points", xytext=(6, -6),
            fontsize=8.5, color="#b3501e")
ax.annotate("P3", f04_ctrl[-1], textcoords="offset points", xytext=(-16, 13),
            fontsize=8.5, color="#b3501e")
ax.annotate("P3", f04_b3[-1], textcoords="offset points", xytext=(7, -17),
            fontsize=8.5, color="#b3501e")

ax.annotate("B3：P0 +0.26 dB", f05_b3[0], xytext=(17.4, 24.55), fontsize=10,
            color="#1f6fb2", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f6fb2", lw=1.2))
ax.annotate("B3：P0 +0.92 dB", f04_b3[0], xytext=(8.3, 18.9), fontsize=10,
            color="#b3501e", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#b3501e", lw=1.2))

ax.set_xlabel("码流体积（MB）")
ax.set_ylabel("PSNR（dB，150 视角）")
ax.set_title("1-78 全家族同图：两族渐进曲线（不同 λ 训练的模型各切 4 档）vs HAC++ 与 base\n"
             "（150 视角 / 1600px GT / 生产量化栅格，2026-09-28 审计统一口径）", fontsize=12.5)
ax.set_xlim(4.5, 42)
ax.set_ylim(16.9, 29.2)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.3)

fig.tight_layout()
out = "docs/figures/all_families_one_canvas_20260928.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
