#!/usr/bin/env python3
"""Final competitor-positioning figure: ours vs HAC++ vs PCGS vs Octree-GS.

All series 150-view test evals on 1-78. Octree-GS is a representation
method (uncompressed 392MB PLY) — shown as a quality bar, not an RD point.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

hacpp = [(12.10, 26.868), (16.34, 27.633), (20.03, 27.841), (26.27, 28.211), (40.56, 28.678)]
base = [(10.67, 27.394), (13.90, 27.695), (19.94, 27.813)]
pcgs = [(26.4, 28.128), (35.2, 28.331), (44.4, 28.366), (54.1, 28.366)]
prog_b3 = [(15.97, 26.085), (19.49, 27.111), (22.11, 27.492), (26.23, 27.684)]
p004_b3 = [(7.13, 19.733), (9.46, 23.747), (11.41, 26.183), (14.33, 27.225)]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

ax = axes[0]
ax.plot([p[0] for p in hacpp], [p[1] for p in hacpp], "D-", color="#8a8a8a", lw=1.8, ms=7,
        label="HAC++（非渐进，5 点）")
ax.plot([p[0] for p in pcgs], [p[1] for p in pcgs], "^-", color="#7b4ea3", lw=1.8, ms=8,
        label="PCGS（渐进压缩，4 步）")
ax.plot([p[0] for p in prog_b3], [p[1] for p in prog_b3], "s--", color="#d9622b", lw=1.8, ms=7,
        label="渐进 + B3（单文件四档）")
ax.plot([p[0] for p in base], [p[1] for p in base], "o-", color="#2e7d32", lw=1.6, ms=6,
        label="我们 base 单码率")
ax.axhline(24.94, color="#3f6fb5", lw=1.2, ls=":", alpha=0.8)
ax.text(21.5, 24.72, "Octree-GS 40k：24.94 dB（392 MB 无压缩 PLY）", fontsize=9, color="#3f6fb5")
ax.set_xlabel("码流体积（MB）")
ax.set_ylabel("PSNR（dB，150 视角）")
ax.set_title("全范围定位（1-78）", fontsize=12)
ax.set_xlim(8, 56)
ax.set_ylim(24.4, 29.0)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.8)

ax = axes[1]
ax.plot([p[0] for p in hacpp], [p[1] for p in hacpp], "D-", color="#8a8a8a", lw=1.8, ms=7,
        label="HAC++（延伸后）")
ax.plot([p[0] for p in p004_b3], [p[1] for p in p004_b3], "^--", color="#d9622b", lw=1.8, ms=7,
        label="渐进 λ0.004 + B3")
ax.axvspan(5, 12.1, color="#d9622b", alpha=0.06)
ax.text(7.0, 26.9, "<12.1 MB：HAC++/PCGS\n均无产品，我方独占", fontsize=9.5, color="#b3501e")
ax.set_xlabel("码流体积（MB）")
ax.set_ylabel("PSNR（dB，150 视角）")
ax.set_title("低码率放大：独占区间", fontsize=12)
ax.set_xlim(5, 22)
ax.set_ylim(17, 28)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.8)

fig.suptitle("1-78：渐进赛道竞品对比（PCGS / Octree-GS / HAC++ / 我方）", fontsize=13, y=1.00)
fig.tight_layout()
out = "docs/figures/competitor_positioning_20260929.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
