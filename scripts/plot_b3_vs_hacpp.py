#!/usr/bin/env python3
"""B3-improved progressive family vs HAC++ (1-78, 150-view protocol).

Defensible series (same 150-view evaluation):
  HAC++          3 single files
  base (ours)    3 single files, new-code entropy coder
  progressive    4 prefixes, layered bitstream, pre-B3
B3-improved progressive = pre-B3 points + paired deltas measured at 16-view
(dashed, open markers) — labeled as extrapolation, not a 150-view measurement.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

hacpp = [(20.03, 27.841), (26.27, 28.211), (40.56, 28.678)]
base = [(10.67, 27.394), (13.90, 27.695), (19.94, 27.813)]
prog = [(15.75, 25.011), (20.94, 26.187), (24.55, 26.535), (29.95, 26.703)]
d_psnr = [0.245, 0.075, 0.061, 0.054]
d_mb = [0.29, -1.24, -2.37, -4.05]
prog_b3 = [(x + dx, y + dy) for (x, y), dx, dy in zip(prog, d_mb, d_psnr)]

fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.plot([p[0] for p in hacpp], [p[1] for p in hacpp], "D-", color="#8a8a8a",
        lw=1.8, ms=8, label="HAC++（3 个独立文件，150 视角）")
ax.plot([p[0] for p in base], [p[1] for p in base], "o-", color="#2e7d32",
        lw=1.8, ms=7, label="我们 base 单码率（150 视角）")
ax.plot([p[0] for p in prog], [p[1] for p in prog], "s-", color="#5b7fa6",
        lw=1.8, ms=7, label="渐进分层码流（150 视角，B3 前）")
ax.plot([p[0] for p in prog_b3], [p[1] for p in prog_b3], "s--", color="#d9622b",
        lw=1.8, ms=8, mfc="none", label="渐进 + B3（配对增益外推，16 视角口径）")
for (x, y), (xb, yb) in zip(prog, prog_b3):
    ax.annotate("", (xb, yb), (x, y),
                arrowprops=dict(arrowstyle="->", color="#d9622b", lw=1.1, alpha=0.7))
ax.annotate("P0 +0.25", prog_b3[0], textcoords="offset points", xytext=(8, -14),
            fontsize=10, color="#d9622b", fontweight="bold")
ax.text(11.6, 27.52, "同画质省约 30% 体积", fontsize=9.5, color="#2e7d32", rotation=8)
ax.annotate("HAC++ 最小文件 20.0 MB：\n低码率区（<20 MB）只有我们的产品",
            (20.03, 27.841), xytext=(24.5, 25.7), fontsize=9.5, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
ax.set_xlabel("码流体积（MB）")
ax.set_ylabel("PSNR（dB，150 视角）")
ax.set_title("1-78：渐进分层码流（含 B3）与 HAC++、base 的相对定位", fontsize=13)
ax.set_xlim(8, 42)
ax.set_ylim(24.4, 29.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9.5)
fig.tight_layout()
out = "docs/figures/b3_progressive_vs_hacpp_20260927.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
