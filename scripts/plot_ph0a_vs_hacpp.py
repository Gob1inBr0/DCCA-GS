"""Phase 0a 剂量-响应(r_post 扫描)与 HAC++ 的 RD 点图(1-78 · 30k · λ=0.002 · seed42)。

数据:
* ph0a 四点:docs/data/experiments.csv ph0a 组(r_post ∈ {1.0, 0.85, 0.7, 0.55},base 无融合,
  post 相位 κ ramp,fp32 解码口径,bit-exact)。
* HAC++ 同协议三点:官方代码 30k 实测(HANDOVER_20260902 §6.1)——严格参考。
* HAC++ 共享表单点:allMothod.csv(协议未验证,仅位置参考)。

用法: python scripts/plot_ph0a_vs_hacpp.py
输出: docs/figures/ph0a_vs_hacpp.png
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcfg")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdgcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

STHEITI = "/System/Library/Fonts/STHeiti Light.ttc"
if os.path.exists(STHEITI):
    fm.fontManager.addfont(STHEITI)
    plt.rcParams["font.family"] = "Heiti TC"
plt.rcParams["axes.unicode_minus"] = False

# (MB, PSNR, label)
PH0A = [  # r_post 从 1.0(松)到 0.55(紧)
    (15.243, 27.862, "r100"),
    (14.453, 27.730, "r085"),
    (13.290, 27.637, "r070"),
    (11.101, 27.285, "r055"),
]
HACPP_SAME = [  # 官方代码 30k 同协议
    (20.0265, 27.8406, "λ=0.004"),
    (26.2734, 28.2105, "λ=0.002"),
    (40.5589, 28.6780, "λ=0.0005"),
]
HACPP_TABLE = (17.227, 27.140)  # 共享表单点(协议未验证)

fig, ax = plt.subplots(figsize=(9.5, 6.2))

# HAC++ 同协议曲线
xs = [p[0] for p in HACPP_SAME]
ys = [p[1] for p in HACPP_SAME]
ax.plot(xs, ys, color="black", ls="--", marker="x", ms=11, lw=2.0,
        label="HAC++ 30k(官方代码,同协议)", zorder=4)
for mb, psnr, lab in HACPP_SAME:
    ax.annotate(f"HAC++ {lab}\n{psnr:.2f} @ {mb:.1f} MB", (mb, psnr),
                textcoords="offset points", xytext=(8, -20), fontsize=8,
                color="black", zorder=5)

# 共享表 HAC++ 单点
ax.scatter(*HACPP_TABLE, s=150, marker="*", color="#718096", zorder=4,
           label="HAC++(共享表单点,口径未验证)")
ax.annotate(f"{HACPP_TABLE[1]:.2f} @ {HACPP_TABLE[0]:.1f} MB",
            HACPP_TABLE, textcoords="offset points", xytext=(6, 8),
            fontsize=8, color="#718096")

# ph0a 剂量-响应
xs = [p[0] for p in PH0A]
ys = [p[1] for p in PH0A]
ax.plot(xs, ys, color="#e53e3e", ls="-", marker="D", ms=9, lw=2.6,
        label="DCCA-GS base,r_post 剂量扫描(Phase 0,λ=0.002)", zorder=6)
for mb, psnr, lab in PH0A:
    ax.scatter(mb, psnr, s=90, color="#e53e3e", zorder=6,
               edgecolors="white", linewidths=1.0)
    ax.annotate(f"{lab}\n{psnr:.2f} @ {mb:.2f}", (mb, psnr),
                textcoords="offset points", xytext=(-6, 10),
                fontsize=8.5, color="#c53030", ha="right", zorder=6)

# r100 与 HAC++@0.004 的同预算对照标注
ax.annotate("同预算 30k 对照:\nr100 27.86 @ 15.2 MB vs\nHAC++ 27.84 @ 20.0 MB\n(+0.02 dB,体积 −24%)",
            xy=(15.243, 27.862), xytext=(22.5, 27.32),
            fontsize=9, color="#2d3748",
            arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#a0aec0"))

ax.set_xlabel("模型体积 total_MB(fp32 解码口径)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("1-78 · 30k · seed42 · Phase 0 预算剂量扫描 vs HAC++(越左上越好)")
ax.set_xlim(8, 44)
ax.set_ylim(27.0, 28.9)
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures",
                   "ph0a_vs_hacpp.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED", os.path.abspath(out))
