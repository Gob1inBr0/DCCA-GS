"""Plot paired room anchor-merge results and author-reported method references."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/data/experiments.csv"
OUTPUT = ROOT / "docs/figures/room_anchor_merge_comparison_20261122"

LOCAL_RUNS = [
    ("DCCA 未合并", "intrain_merge_saving_filter_room_l0002_s42_control_final",
     "#34445A", "o"),
    ("合并 · 保留字节下限", "intrain_merge_saving_filter_room_l0002_s42_minimum_one_byte_final",
     "#A86C13", "D"),
    ("合并 · 16 近邻", "intrain_merge_saving_filter_room_l0002_s42_no_minimum_final",
     "#286FAD", "s"),
    ("合并 · 64 近邻", "intrain_merge_neighbors_room_l0002_s42_k64_no_minimum_final",
     "#CC3D49", "*"),
]
REFERENCES = [
    ("COSA-GS", "room_ref_cosags_", "#188A78", "o"),
    ("Smol-GS", "room_ref_smolgs_", "#A46A1C", "^"),
    ("CRP-GS", "room_ref_crpgs_", "#9A65AB", "P"),
    ("HAC++", "room_ref_hacpp_", "#5275B9", "s"),
    ("ContextGS", "room_ref_contextgs_", "#8A8498", "D"),
    ("HAC", "room_ref_hac_", "#789AA4", "v"),
]


def load_points():
    with REGISTRY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    local = []
    for label, run_id, color, marker in LOCAL_RUNS:
        metrics = next(row for row in rows if row["run_id"] == run_id)
        payload = next(row for row in rows if row["run_id"] == run_id + "_payload")
        assert metrics["iteration"] == payload["iteration"] == "30000"
        local.append({
            "label": label,
            "size_mb": int(payload["metric_value"]) / 1_000_000,
            "psnr": float(metrics["psnr"]),
            "color": color,
            "marker": marker,
        })
    published = []
    for label, prefix, color, marker in REFERENCES:
        points = [
            {"size_mb": float(row["total_mb"]), "psnr": float(row["psnr"])}
            for row in rows
            if row["group"] == "room_published_reference"
            and row["scene"] == "room"
            and row["run_id"].startswith(prefix)
        ]
        assert points, label
        published.append((label, color, marker, sorted(points, key=lambda p: p["size_mb"])))
    return local, published


def style_axis(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color="#DFE5EC", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#ADB7C4")
    ax.tick_params(axis="both", labelsize=10.5, colors="#46556A", length=3)
    ax.set_ylabel("PSNR / dB  ↑", fontsize=12.5, color="#25364B", labelpad=9)
    ax.set_xlabel("解码文件大小 / MB  ↓", fontsize=12.5, color="#25364B", labelpad=9)


def main():
    chinese_font = next(
        (font.name for font in font_manager.fontManager.ttflist
         if font.name in ("PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "Heiti SC")),
        "DejaVu Sans",
    )
    plt.rcParams.update({
        "font.family": [chinese_font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "font.size": 11,
    })
    local, references = load_points()
    fig, axes = plt.subplots(
        1, 2, figsize=(14.7, 7.4), gridspec_kw={"width_ratios": [1, 1.32]},
    )
    fig.patch.set_facecolor("#F8FAFD")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.245, wspace=0.23)
    fig.text(0.065, 0.935, "room 场景的压缩体积与画质", fontsize=23, color="#172A43")
    fig.text(
        0.065, 0.884,
        "横轴越小，文件越小  ·  纵轴越高，PSNR 越高  ·  所有体积统一为 1 MB = 1,000,000 字节",
        fontsize=12, color="#586A80",
    )

    ax = axes[0]
    style_axis(ax)
    ax.set_title("同协议实测", loc="left", pad=20, fontsize=16, color="#172A43")
    ax.text(0, 1.035, "30k 终点 · 相同 18k 检查点 · 同一批 39 个评测视角",
            transform=ax.transAxes, fontsize=10, color="#617087")
    ax.set_xlim(1.02, 2.22)
    ax.set_ylim(30.10, 30.94)
    ax.set_xticks([1.1, 1.3, 1.5, 1.7, 1.9, 2.1])
    ax.set_yticks([30.2, 30.4, 30.6, 30.8])
    labels_at = [
        (1.72, 30.875, "right"),
        (1.58, 30.585, "left"),
        (1.37, 30.19, "left"),
        (1.07, 30.41, "left"),
    ]
    for point, (x, y, align) in zip(local, labels_at):
        ax.scatter(point["size_mb"], point["psnr"],
                   s=195 if point["marker"] == "*" else 82, marker=point["marker"],
                   color=point["color"], edgecolor="white", linewidth=1, zorder=5)
        label = (f'{point["label"]}\n'
                 f'{point["size_mb"]:.3f} MB · {point["psnr"]:.3f} dB')
        ax.annotate(
            label, xy=(point["size_mb"], point["psnr"]), xytext=(x, y),
            fontsize=10.5, color=point["color"], ha=align, va="center",
            arrowprops={"arrowstyle": "-", "color": point["color"], "lw": 0.9},
        )
    base, merged = local[0], local[3]
    saving = 100 * (1 - merged["size_mb"] / base["size_mb"])
    psnr_cost = base["psnr"] - merged["psnr"]
    ax.text(
        0.97, 0.045, f"64 近邻相对 DCCA\n文件小 {saving:.1f}%\nPSNR 低 {psnr_cost:.3f} dB",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#46556A",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F1F4F8", "edgecolor": "none"},
    )

    ax = axes[1]
    style_axis(ax)
    ax.set_title("与作者公开结果对照", loc="left", pad=20, fontsize=16, color="#172A43")
    ax.text(0, 1.035, "room 单场景参考点 · 各方法沿用作者训练、评测与打包设置",
            transform=ax.transAxes, fontsize=10, color="#617087")
    ax.set_xscale("log")
    ax.set_xlabel("解码文件大小 / MB（对数刻度） ↓", fontsize=12.5, color="#25364B", labelpad=9)
    ax.set_xlim(0.78, 9.35)
    ax.set_ylim(30.10, 32.72)
    ticks = [0.8, 1.0, 1.5, 2, 3, 4, 6, 8]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_yticks([30.5, 31, 31.5, 32, 32.5])
    handles = []
    for label, color, marker, points in references:
        x = [p["size_mb"] for p in points]
        y = [p["psnr"] for p in points]
        line, = ax.plot(x, y, linestyle="--", linewidth=1.3, color=color, alpha=0.88,
                        marker=marker, markersize=6.3, markerfacecolor="#FFFFFF",
                        markeredgewidth=1.35, label=label, zorder=3)
        handles.append(line)
    for point in local:
        ax.scatter(point["size_mb"], point["psnr"], marker="o", s=43,
                   color="#CC3D49", edgecolor="white", linewidth=0.7, zorder=6)
    ax.annotate(
        "本次 DCCA", xy=(base["size_mb"], base["psnr"]), xytext=(2.30, 30.68),
        fontsize=10, color="#B72F3D",
        arrowprops={"arrowstyle": "-", "color": "#CC3D49", "lw": 0.8},
    )
    ax.annotate(
        "本次合并 · 64", xy=(merged["size_mb"], merged["psnr"]), xytext=(1.55, 30.31),
        fontsize=10, color="#B72F3D",
        arrowprops={"arrowstyle": "-", "color": "#CC3D49", "lw": 0.8},
    )
    cosa_low = next(points[0] for label, _, _, points in references if label == "COSA-GS")
    ax.annotate(
        f"COSA-GS 低码率点\n{cosa_low['size_mb']:.3f} MB · {cosa_low['psnr']:.3f} dB",
        xy=(cosa_low["size_mb"], cosa_low["psnr"]), xytext=(0.89, 31.55),
        fontsize=10, color="#188A78",
        arrowprops={"arrowstyle": "-", "color": "#188A78", "lw": 0.85},
    )
    handles.append(Line2D([], [], marker="o", linestyle="", color="#CC3D49",
                          markersize=6, label="本次 DCCA / 合并模型"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, -0.17),
              ncol=4, frameon=False, columnspacing=1.1, handlelength=1.6, fontsize=9.8)
    fig.text(
        0.065, 0.115,
        "实测点的偏移与掩码参与续训。公开点为作者发布的 room 结果，训练预算、图像分辨率和打包方式存在差异。",
        fontsize=10.4, color="#586A80",
    )
    fig.text(
        0.065, 0.073,
        "来源  experiments.csv · HAC / HAC++ / ContextGS / Smol-GS 官方结果 · COSA-GS 表 11（15+30k） · CRP-GS 表 III",
        fontsize=10.1, color="#586A80",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT.with_suffix(suffix), dpi=230, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
