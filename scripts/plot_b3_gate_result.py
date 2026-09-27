#!/usr/bin/env python3
"""B3 gate-result visualization: paired layered RD curves, one panel per tier.

Data source: experiments.csv rows layered_b3w001_16view / layered_b3pair_004_16view
(16-view pipeline, paired same-code same-seed comparisons).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CTRL = "#5b7fa6"
B3 = "#d9622b"

TIERS = {
    "λ0.0005（松预算档）B3 w=0.01": dict(
        ctrl=[(15.51, 25.193), (20.67, 26.413), (24.43, 26.764), (30.24, 26.940)],
        b3=[(15.80, 25.438), (19.43, 26.488), (22.06, 26.825), (26.19, 26.994)],
        annot=(15.80, 25.438, "+0.245 dB @ P0", (18.6, 25.02)),
        full="满精度 +0.083 dB，体积 -4.1 MB",
        ylim=(24.8, 27.4),
    ),
    "λ0.004（紧预算档）B3 w=0.05": dict(
        ctrl=[(6.67, 18.076), (9.73, 23.033), (11.90, 25.521), (14.92, 26.506)],
        b3=[(7.00, 19.026), (9.35, 22.950), (11.26, 25.415), (14.21, 26.551)],
        annot=(7.00, 19.026, "+0.95 dB @ P0", (8.3, 17.4)),
        full="满精度 -0.026 dB（噪声级），体积 -0.7 MB",
        ylim=(17.0, 27.3),
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, (title, d) in zip(axes, TIERS.items()):
    c = d["ctrl"]
    b = d["b3"]
    ax.plot([p[0] for p in c], [p[1] for p in c], "o-", color=CTRL, lw=1.8, ms=7,
            label="对照（同一代码，B3 关）")
    ax.plot([p[0] for p in b], [p[1] for p in b], "s-", color=B3, lw=1.8, ms=7,
            label="B3 开（阶梯对齐惩罚）")
    for i, p in enumerate(c):
        ax.annotate(f"P{i}", p, textcoords="offset points", xytext=(6, -14),
                    fontsize=9, color=CTRL)
    for i, p in enumerate(b):
        ax.annotate(f"P{i}", p, textcoords="offset points", xytext=(6, 8),
                    fontsize=9, color=B3)
    x, y, txt, xy = d["annot"]
    ax.annotate(txt, (x, y), xytext=xy, fontsize=11, fontweight="bold", color=B3,
                arrowprops=dict(arrowstyle="->", color=B3, lw=1.4))
    ax.set_xlabel("码流体积（MB）")
    ax.set_ylabel("PSNR（dB，16 视角）")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(*d["ylim"])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.02, 0.98, d["full"], transform=ax.transAxes, fontsize=9.5,
            va="top", color="#444")

fig.suptitle("B3 阶梯对齐训练：配对渐进码流对比（同代码同种子，仅惩罚开关不同）",
             fontsize=13, y=1.00)
fig.tight_layout()
out = "docs/figures/b3_gate_result_20260927.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
