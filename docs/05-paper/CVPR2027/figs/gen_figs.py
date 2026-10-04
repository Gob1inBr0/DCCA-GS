# 论文图生成：数据全部来自 docs/data/experiments.csv（分支 feat/enh-layer-learned-coder，
# 363 行权威版），组 curve/layered_arscan_*（1-78，150 视角，新环境）。
# 生成：fig1_bitstream.pdf（码流截断示意，标注 r=0.50 实测档位数字）
#       fig2_budget_axis.pdf（锚点预算轴四条渐进曲线 + 单码率参照点）
# 运行：python3 gen_figs.py（在本目录下执行）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---- CSV 实测数据（MB, dB），layered_arscan_r0XX_150v 四行 ----
curves = {
    "r=0.85": [(15.88, 26.151), (21.06, 27.435), (24.57, 27.828), (30.05, 28.011)],
    "r=0.70": [(11.15, 25.504), (14.62, 26.631), (17.01, 27.080), (20.83, 27.244)],
    "r=0.60": [(8.67, 24.980), (11.25, 26.091), (13.07, 26.501), (15.91, 26.714)],
    "r=0.50": [(6.33, 24.252), (8.17, 25.419), (9.44, 25.941), (11.41, 26.141)],
}
# 生产单码率码流（学习式上下文编码，无渐进档）：env2_base_r085 与 arscan 各档 decoded eval
single_rate = {
    "r=0.85": (20.39, 27.8351),
    "r=0.70": (14.11, 27.0356),
    "r=0.60": (10.95, 26.5507),
    "r=0.50": (8.05, 25.9343),
}
PROGS_LOWEST = 10.75  # ProGS 论文报告的最低档（其自有场景与协议，仅作定位参考）

# Okabe-Ito 色板（色觉安全）+ 线型区分（黑白打印可辨）
C = {"r=0.85": "#0072B2", "r=0.70": "#E69F00", "r=0.60": "#009E73", "r=0.50": "#D55E00"}
M = {"r=0.85": "o", "r=0.70": "s", "r=0.60": "^", "r=0.50": "D"}

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
})

# ============ 图 1：单文件截断示意 ============
fig, ax = plt.subplots(figsize=(3.35, 2.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

# 码流条：header + base + L1 + L2 + L3（宽度按 r=0.50 实测字节比例）
blocks = [
    ("", 2.2, "#bbbbbb"),
    ("base layer\n(geometry +\nL0 symbols)", 51.0, "#9ecae1"),
    ("L1", 15.5, "#4292c6"),
    ("L2", 12.5, "#2171b5"),
    ("L3", 18.8, "#08519c"),
]
x = 0.0
block_edges = []
for name, w, col in blocks:
    ax.add_patch(Rectangle((x, 41), w, 11, facecolor=col, edgecolor="black", linewidth=0.6))
    if name:
        ax.text(x + w / 2, 46.5, name, ha="center", va="center", fontsize=5.6,
                color="white" if col.startswith("#2") or col.startswith("#0") else "black",
                linespacing=1.25)
    block_edges.append(x)
    x += w
block_edges.append(x)
# header 引出标注（块太窄，标签放上方）
hx = block_edges[1] / 2
ax.annotate("header", xy=(hx, 52), xytext=(hx + 9, 56.5), fontsize=6,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
# 文件首尾标注
ax.annotate("", xy=(100, 56.5), xytext=(30, 56.5),
            arrowprops=dict(arrowstyle="<->", lw=0.6, color="black"))
ax.text(65, 58.2, "single file", fontsize=6.5, ha="center", va="bottom")
# 截断点：header 之后、L0 之后、L1 之后、L2 之后
cuts = [block_edges[1], block_edges[2], block_edges[3]]
labels = ["$P_0$ cut", "$P_1$ cut", "$P_2$ cut"]
for cx, lab in zip(cuts, labels):
    ax.plot([cx, cx], [34.5, 41], color="#D55E00", lw=0.8, linestyle=(0, (3, 2)))
    ax.text(cx, 32.8, lab, ha="center", va="top", fontsize=6, color="#D55E00")
# 截断结果说明（r=0.50 实测档位）
tier_txt = (
    "truncate at a block boundary $\\Rightarrow$ valid stream:\n"
    "$P_0$ 6.33 MB / 24.25 dB    $P_1$ 8.17 MB / 25.42 dB\n"
    "$P_2$ 9.44 MB / 25.94 dB    $P_3$ (full) 11.41 MB / 26.14 dB"
)
ax.text(0, 27.5, tier_txt, fontsize=6.2, va="top", ha="left", linespacing=1.5)
ax.text(0, 11.5,
        "no learned model, codebook, hash grid, or quantization table precedes\n"
        "the first decodable byte; every step, group, and probability table is\n"
        "recomputed from received bytes (numbers: r=0.50 budget, 150-view eval)",
        fontsize=5.6, va="top", ha="left", color="#333333", linespacing=1.4)
fig.tight_layout(pad=0.25)
fig.savefig("fig1_bitstream.pdf", bbox_inches="tight")
plt.close(fig)

# ============ 图 2：锚点预算轴渐进曲线 ============
fig, ax = plt.subplots(figsize=(3.35, 2.35))
for name, pts in curves.items():
    mb = [p[0] for p in pts]; db = [p[1] for p in pts]
    ax.plot(mb, db, marker=M[name], ms=3.5, lw=1.1, color=C[name], label=name)
    ax.annotate("", xy=(mb[1], db[1]), xytext=(mb[0], db[0]),
                arrowprops=dict(arrowstyle="->", lw=0.0))  # no-op keeps marker order stable
    if name == "r=0.50":
        ax.text(mb[0] + 0.55, db[0] - 0.13, "entry", fontsize=5.2,
                ha="left", va="center", color=C[name])
# 生产单码率参照点（同一模型、学习式上下文编码、无渐进档）
for name, (mb, db) in single_rate.items():
    ax.plot([mb], [db], marker="x", ms=4.5, mew=1.2, color=C[name], linestyle="none")
ax.plot([], [], marker="x", ms=4.5, mew=1.2, color="gray", linestyle="none",
        label="single-rate (no tiers)")
# ProGS 最低档参考线（论文报告值，协议不同）
ax.axvline(PROGS_LOWEST, color="gray", lw=0.8, linestyle=(0, (4, 2)))
ax.text(PROGS_LOWEST + 0.35, 24.35, "lowest tier reported\nby ProGS (10.75 MB,\nits protocol)", fontsize=5.2,
        color="#444444", va="bottom")
ax.set_xlabel("decoded size (MiB)")
ax.set_ylabel("PSNR (dB)")
ax.set_xlim(4, 32.5)
ax.set_ylim(24.0, 28.35)
ax.tick_params(width=0.6, length=2.5)
ax.legend(fontsize=6, frameon=False, loc="lower right", handlelength=1.6)
for s in ax.spines.values():
    s.set_linewidth(0.6)
fig.tight_layout(pad=0.3)
fig.savefig("fig2_budget_axis.pdf", bbox_inches="tight")
plt.close(fig)
print("figs written")
