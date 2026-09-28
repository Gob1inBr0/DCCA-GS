#!/usr/bin/env python3
"""B3-improved progressive family vs HAC++ (1-78, ALL series 150-view measured).

New: HAC++ curve extended to tight lambdas (0.008/0.016 -> 16.34/12.10 MB);
progressive B3 curves are now 150-view measured (no extrapolation).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

hacpp = [(12.10, 26.868), (16.34, 27.633), (20.03, 27.841), (26.27, 28.211), (40.56, 28.678)]
base = [(10.67, 27.394), (13.90, 27.695), (19.94, 27.813)]
prog = [(15.72, 25.826), (20.88, 27.040), (24.66, 27.406), (30.48, 27.601)]
prog_b3 = [(15.97, 26.085), (19.49, 27.111), (22.11, 27.492), (26.23, 27.684)]
p004_ctrl = [(6.81, 18.817), (9.94, 23.908), (12.11, 26.265), (15.14, 27.251)]
p004_b3 = [(7.13, 19.733), (9.46, 23.747), (11.41, 26.183), (14.33, 27.225)]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

ax = axes[0]
ax.plot([p[0] for p in hacpp], [p[1] for p in hacpp], "D-", color="#8a8a8a",
        lw=1.8, ms=7, label="HAC++（5 个独立文件，含新补 12.1/16.3 MB）")
ax.plot([p[0] for p in base], [p[1] for p in base], "o-", color="#2e7d32",
        lw=1.8, ms=6, label="我们 base 单码率")
ax.plot([p[0] for p in prog], [p[1] for p in prog], "s-", color="#5b7fa6",
        lw=1.6, ms=6, label="渐进分层码流（B3 前）")
ax.plot([p[0] for p in prog_b3], [p[1] for p in prog_b3], "s--", color="#d9622b",
        lw=1.8, ms=7, label="渐进 + B3（150 视角实测）")
ax.set_xlabel("码流体积（MB）")
ax.set_ylabel("PSNR（dB，150 视角）")
ax.set_title("全范围：四条曲线定位", fontsize=12)
ax.set_xlim(8, 42)
ax.set_ylim(24.4, 29.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8.8)

ax = axes[1]
ax.plot([p[0] for p in hacpp], [p[1] for p in hacpp], "D-", color="#8a8a8a",
        lw=1.8, ms=7, label="HAC++（延伸后）")
ax.plot([p[0] for p in base], [p[1] for p in base], "o-", color="#2e7d32",
        lw=1.8, ms=6, label="我们 base 单码率")
ax.plot([p[0] for p in p004_ctrl], [p[1] for p in p004_ctrl], "^-", color="#5b7fa6",
        lw=1.6, ms=6, label="渐进 λ0.004（B3 前）")
ax.plot([p[0] for p in p004_b3], [p[1] for p in p004_b3], "^--", color="#d9622b",
        lw=1.8, ms=7, label="渐进 λ0.004 + B3（P0 +0.92 dB）")
for i, p in enumerate(p004_b3):
    ax.annotate(f"P{i}", p, textcoords="offset points", xytext=(6, 6),
                fontsize=9, color="#d9622b")
ax.axvspan(5, 12.1, color="#d9622b", alpha=0.06)
ax.text(7.2, 27.2, "<12.1 MB：\nHAC++ 无产品", fontsize=9.5, color="#b3501e")
ax.set_xlabel("码流体积（MB）")
ax.set_ylabel("PSNR（dB，150 视角）")
ax.set_title("低码率放大：延伸后 HAC++ 与 λ0.004 渐进曲线", fontsize=12)
ax.set_xlim(5, 22)
ax.set_ylim(17, 28)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.8)

fig.suptitle("1-78：B3 渐进码流 vs 延伸后的 HAC++（全部 150 视角实测，栅格修复后）", fontsize=13, y=1.00)
fig.tight_layout()
out = "docs/figures/b3_progressive_vs_hacpp_measured_20260928.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
