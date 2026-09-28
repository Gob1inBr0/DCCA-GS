"""绘制 1-78 · 融合剪枝 RD 曲线 vs 其他方法码率点散点（对数轴 + 线性轴）。

背景
----
用户要求：把最新融合剪枝（修复后）结果画成 RD 图，并叠放其他工作的码率点，
看我们的点能否落在「最左上」（体积更小、PSNR 更高）。

数据口径
--------
* 融合剪枝·修复后 / HAC++ 原生：1-78 · 30k · 32-dim · seed 42 · fp32 解码口径，
  与 scripts/plot_rd_vs_hacpp.py 相同（来源 docs/03-reports/RD图_vs_HAC++_流程记录.md §3）。
* 其他方法散点：docs/data/全数据集上各方案结果.csv 的 1-78 列，每个方法一个单码率点。
  该表未注明协议（迭代数 / data_factor / λ 均未知），表内 HAC++ 1-78（27.14 dB / 17.23 MB）
  与本项目 30k 协议的 HAC++ 原生三点不一致 → **口径未对齐，散点仅作位置参考**。
* reduced3dgs 1-78 PSNR 为 N/A，PSNR 面板跳过（LPIPS 面板保留）。

用法
----
    python scripts/plot_rd_fusion_vs_others.py

输出
----
  docs/figures/rd_1-78_fusion_vs_others_psnr.png    （PSNR × log/linear）
"""

from __future__ import annotations

import csv
import os
import pathlib

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "data" / "全数据集上各方案结果.csv"
SCENE = "1-78"

# lambda -> (PSNR, SSIM, LPIPS, total_MB)，与 plot_rd_vs_hacpp.py 同源
FUSION = {  # 融合剪枝·修复后（覆盖主导 + ADMM 覆盖约束），本项目最新
    0.004: (27.573, 0.8671, 0.1783, 11.6685),
    0.002: (27.817, 0.8747, 0.1672, 15.3021),
    0.0005: (28.069, 0.8800, 0.1590, 23.5236),
}
HACPP = {  # HAC++ 原生 30k（HANDOVER_20260902 §6.1，同协议参考）
    0.004: (27.8406, 0.8829, 0.1597, 20.0265),
    0.002: (28.2105, 0.8903, 0.1514, 26.2734),
    0.0005: (28.6780, 0.9001, 0.1396, 40.5589),
}
LAMBDAS = [0.004, 0.002, 0.0005]

# CSV 方法名 -> 展示名；HAC++ 单独处理
NAME_MAP = {
    "PC-GS": "PC-GS",
    "CAT": "CAT",
    "contextgs": "ContextGS",
    "HAC": "HAC",
    "RDO-gaussian": "RDO-Gaussian",
    "Scaffold-gs": "Scaffold-GS",
    "compact3d": "Compact3D",
    "reduced3dgs": "Reduced3DGS",
    "gsplat": "gsplat",
    "3DGS": "3DGS",
    "TC-GS": "TC-GS",
    "compact3dgs": "Compact3DGS",
    "Minisplatting": "Mini-Splatting",
    "FC-GS": "FC-GS",
    "Lightgaussian": "LightGaussian",
    "gaussianspa": "GaussianSpa",
    "Octree-GS": "Octree-GS",
}

# 手工标签偏移（points, dx, dy, ha），避免 18 个散点名互相压盖
LABEL_OFFSET = {
    "RDO-Gaussian": (0, 7, "center"),
    "PC-GS": (0, -13, "center"),
    "CAT": (0, 7, "center"),
    "TC-GS": (-7, -12, "right"),
    "HAC": (7, -11, "left"),
    "Compact3D": (7, -12, "left"),
    "Mini-Splatting": (7, -13, "left"),
    "Octree-GS": (7, -2, "left"),
    "ContextGS": (0, 7, "center"),
    "FC-GS": (0, 7, "center"),
    "Compact3DGS": (0, -14, "center"),
    "GaussianSpa": (0, 7, "center"),
    "Scaffold-GS": (0, 7, "center"),
    "LightGaussian": (0, -14, "center"),
    "3DGS": (-8, 6, "right"),
    "gsplat": (-8, -14, "right"),
}

# λ 标注偏移：0.002 放点上方，避开 0.004 标注与表内 HAC++ 星点（17.23 MB）
LAMBDA_OFFSET = {
    0.004: (4, -14, "left"),
    0.002: (0, 9, "center"),
    0.0005: (4, -14, "left"),
}


def load_csv_points(scene: str) -> dict[str, tuple[float, float, float, float]]:
    """读 CSV，返回 {方法名: (PSNR, SSIM, LPIPS, Size)}；N/A -> nan 并在上层过滤。"""
    import math

    out: dict[str, tuple[float, float, float, float]] = {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = (row.get("Method") or "").strip()
            if not method:
                continue

            def num(key: str) -> float:
                raw = (row.get(key) or "").strip()
                try:
                    return float(raw)
                except ValueError:
                    return math.nan

            out[method] = (
                num(f"{scene} PSNR"),
                num(f"{scene} SSIM"),
                num(f"{scene} LPIPS"),
                num(f"{scene} Size [MB]"),
            )
    return out


def plot_panel(ax, logx: bool, title: str, points: dict) -> None:
    """PSNR RD 面板：越左上越好（体积更小、PSNR 更高）。"""
    # 其他方法散点（CSV）
    for method, vals in points.items():
        x, y = vals[3], vals[0]
        if x != x or y != y:  # N/A
            continue
        ax.scatter(x, y, s=40, color="#4a5568", alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.8)
        name = NAME_MAP.get(method, method)
        dx, dy, ha = LABEL_OFFSET.get(name, (6, 4, "left"))
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=7.5, color="#2d3748", zorder=4)

    # HAC++ 原生（同协议曲线）
    xs = [HACPP[lam][3] for lam in LAMBDAS]
    ys = [HACPP[lam][0] for lam in LAMBDAS]
    ax.plot(xs, ys, color="black", ls="--", marker="x", ms=9, lw=2.2,
            label="HAC++ 原生 30k（同协议）", zorder=5)

    # 融合剪枝（本项目最新）
    xs = [FUSION[lam][3] for lam in LAMBDAS]
    ys = [FUSION[lam][0] for lam in LAMBDAS]
    ax.plot(xs, ys, color="#e53e3e", ls="-", marker="D", ms=8, lw=2.8,
            label="融合剪枝·修复后（DCCA-GS 30k）", zorder=6)
    for lam in LAMBDAS:
        dx, dy, ha = LAMBDA_OFFSET[lam]
        ax.annotate(f"λ={lam}", (FUSION[lam][3], FUSION[lam][0]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=7.5, color="#e53e3e", zorder=6)

    if logx:
        ax.set_xscale("log")
        ax.set_xlim(8, 900)
    else:
        ax.set_xlim(0, 62)
    ax.set_ylim(19, 29.6)
    ax.annotate("越左上越好 ↖", xy=(0.04, 0.95), xycoords="axes fraction",
                ha="left", va="top", fontsize=10, color="#2d3748")
    ax.set_xlabel("体积 total_MB")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)


def main() -> None:
    points = load_csv_points(SCENE)

    # HAC++ 表内单点（口径不同，单独标记）
    hacpp_csv = points.pop("HAC++")
    out_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    plot_panel(axes[0], True, "PSNR · 体积（对数轴）", points)
    plot_panel(axes[1], False, "PSNR · 体积（线性轴）", points)
    for ax in axes:
        ax.scatter(hacpp_csv[3], hacpp_csv[0], s=130, marker="*",
                   color="#1a202c", zorder=5, label="HAC++（表内单点·口径不同）")
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[1].legend(fontsize=8.5, loc="lower right")
    off_scale = sorted(
        (NAME_MAP.get(m, m), v[3]) for m, v in points.items() if v[3] > 62 and v[0] == v[0]
    )
    if off_scale:
        txt = "线性轴未显示（>62 MB）：" + " · ".join(f"{n} {s:.0f}" for n, s in off_scale)
        axes[1].annotate(txt, xy=(0.99, -0.16), xycoords="axes fraction",
                         ha="right", va="top", fontsize=7, color="#718096")
    axes[0].annotate(
        "散点：docs/data/全数据集上各方案结果.csv（1-78 列，单码率点，协议未注明，\n"
        "与本项目 30k·seed42·fp32 协议未对齐，仅供位置参考）",
        xy=(0.01, -0.16), xycoords="axes fraction", ha="left", va="top",
        fontsize=7, color="#718096",
    )
    fig.suptitle(
        "1-78 · 融合剪枝（修复后）vs 其他方法 · PSNR（曲线=解码后真实码流 fp32 口径）",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = out_dir / "rd_1-78_fusion_vs_others_psnr.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("SAVED", out)


if __name__ == "__main__":
    main()
