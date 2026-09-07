"""绘制 1-78 · 30k · 32-dim · DCCA-GS vs HAC++ 的 RD 曲线（log 与线性体积轴）。

背景
----
用户要求：画 RD 图，同时给出 log 体积轴与线性体积轴，重点看相比 HAC++ 的提升。

数据口径
--------
* 场景：1-78（无人机，1200 图像 / 1050 train / 150 val），与 HAC++ 同一份数据。
* 迭代：30000；seed：42；feat_dim：32；n_offsets：10。
* 指标来自「解码后真实 bitstream」；体积用 fp32 `total_MB`（与 HAC++ 原生 float32 MLP 口径一致）。
* HAC++ 1-78 30k 参考三点取自 docs/06-planning/HANDOVER_20260902.md §6.1。
* DCCA 四点来自 LYH 训练运行，`bit_exact_roundtrip=true`。

用法
----
    python scripts/plot_rd_vs_hacpp.py

依赖
----
matplotlib + numpy。中文渲染需先 addfont 本机 STHeiti（或任意中文字体），
并把 MPLCONFIGDIR / XDG_CACHE_HOME 指到可写目录（否则字体缓存报错但不影响出图）。

输出
----
  docs/figures/rd_1-78_vs_hacpp_log_linear.png
  docs/figures/rd_1-78_vs_hacpp_lpips_log_linear.png
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


# lambda -> (PSNR, SSIM, LPIPS, total_MB)
HACPP = {  # HAC++ 原生（HANDOVER_20260902 §6.1）
    0.004: (27.8406, 0.8829, 0.1597, 20.0265),
    0.002: (28.2105, 0.8903, 0.1514, 26.2734),
    0.0005: (28.6780, 0.9001, 0.1396, 40.5589),
}
BASELINE = {  # I2+I6（用同步后的运行时代码重跑，与融合剪枝同口径）
    0.004: (27.511, 0.8660, 0.1789, 11.7246),
    0.002: (27.750, 0.8721, 0.1695, 15.049),
    0.0005: (27.953, 0.8780, 0.1614, 23.6042),
}
FUSION = {  # 融合剪枝·修复后（覆盖主导 + ADMM 覆盖约束）
    0.004: (27.573, 0.8671, 0.1783, 11.6685),
    0.002: (27.817, 0.8747, 0.1672, 15.3021),
    0.0005: (28.069, 0.8800, 0.1590, 23.5236),
}
IMPA = {  # 重要感知损失（负结果，保留作参照）
    0.004: (26.937, 0.8557, 0.1887, 11.8523),
    0.002: (27.740, 0.8735, 0.1683, 15.3544),
    0.0005: (27.470, 0.8677, 0.1731, 23.2719),
}

LAMBDAS = [0.004, 0.002, 0.0005]

SERIES = [
    ("HAC++（原生）", HACPP, dict(color="black", ls="--", marker="x", ms=9, lw=2.2)),
    ("基线（I2+I6）", BASELINE, dict(color="#2b6cb0", ls="-", marker="o", ms=7, lw=2.2)),
    ("融合剪枝·修复后", FUSION, dict(color="#e53e3e", ls="-", marker="D", ms=7, lw=2.8)),
    ("重要感知损失（负）", IMPA, dict(color="#38a169", ls="-", marker="s", ms=7, lw=2.2)),
]


def plot_panel(ax, metric: int, logx: bool, title: str) -> None:
    """metric: 0=PSNR, 1=SSIM, 2=LPIPS."""
    for name, data, style in SERIES:
        xs = [data[lam][3] for lam in LAMBDAS]
        ys = [data[lam][metric] for lam in LAMBDAS]
        ax.plot(xs, ys, label=name, **style)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("体积 total_MB")
    ax.set_ylabel({0: "PSNR (dB)", 1: "SSIM", 2: "LPIPS"}[metric])
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)


def main() -> None:
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 图 1：PSNR + SSIM × (log, linear)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_panel(axes[0, 0], 0, True, "PSNR · 体积(对数轴)")
    plot_panel(axes[0, 1], 0, False, "PSNR · 体积(线性轴)")
    plot_panel(axes[1, 0], 1, True, "SSIM · 体积(对数轴)")
    plot_panel(axes[1, 1], 1, False, "SSIM · 体积(线性轴)")
    for r in range(2):
        for c in range(2):
            axes[r, c].legend(fontsize=9, loc="best")
    for r, c in [(0, 0), (0, 1)]:
        axes[r, c].annotate(
            "同质量 BD-rate:\n融合 vs HAC++ = -11.1%",
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            color="#e53e3e",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#e53e3e", alpha=0.9),
        )
    fig.suptitle(
        "1-78 · 30k · 32-dim · DCCA-GS vs HAC++（fp32 解码口径，seed 42）",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out1 = out_dir / "rd_1-78_vs_hacpp_log_linear.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    print("SAVED", out1)

    # 图 2：LPIPS × (log, linear)
    fig2, axs2 = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_panel(axs2[0], 2, True, "LPIPS · 体积(对数轴)")
    plot_panel(axs2[1], 2, False, "LPIPS · 体积(线性轴)")
    axs2[0].legend(fontsize=9, loc="best")
    axs2[1].legend(fontsize=9, loc="best")
    fig2.suptitle("1-78 · 30k · 32-dim · LPIPS vs 体积（fp32 解码口径）", fontsize=14)
    fig2.tight_layout(rect=[0, 0.02, 1, 0.93])
    out2 = out_dir / "rd_1-78_vs_hacpp_lpips_log_linear.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print("SAVED", out2)


if __name__ == "__main__":
    main()
