"""Plot independently decoded room checkpoints from the experiment registry."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/room_anchor_merge_45k_20261122"
ARMS = {
    "control": ("DCCA 未合并对照", "#34445A",
                "intrain_merge_saving_filter_room_l0002_s42_control_final"),
    "merged64": ("合并 · 64 近邻", "#CC3D49",
                 "intrain_merge_neighbors_room_l0002_s42_k64_no_minimum_final"),
}
STEPS = (30000, 32500, 35000, 37500, 40000, 42500, 45000)


def load_points():
    with (ROOT / "docs/data/experiments.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = {}
    for arm, (_, _, initial_id) in ARMS.items():
        points[arm] = []
        for step in STEPS:
            run_id = initial_id if step == 30000 else f"intrain_merge_45k_room_l0002_s42_{arm}_{step}"
            quality = [row for row in rows if row["run_id"] == run_id]
            payload = [row for row in rows if row["run_id"] == run_id + "_payload"]
            assert len(quality) == len(payload) == 1, (arm, step)
            quality, payload = quality[0], payload[0]
            assert quality["metric_type"] == "render_metrics"
            assert payload["metric_type"] == "compact_payload_bytes"
            assert int(quality["iteration"]) == int(payload["iteration"]) == step
            sampled_loss = None
            if step > 30000:
                loss_rows = [row for row in rows if row["run_id"] == run_id + "_sampled_loss"]
                assert len(loss_rows) == 1
                assert loss_rows[0]["metric_type"] == "training_loss_sample_mean"
                sampled_loss = float(loss_rows[0]["metric_value"])
            points[arm].append({
                "step": step / 1000,
                "psnr": float(quality["psnr"]),
                "ssim": float(quality["ssim"]),
                "lpips": float(quality["lpips"]),
                "size": int(payload["metric_value"]) / 1_000_000,
                "sampled_loss": sampled_loss,
            })
    return points


def main():
    font = next((font.name for font in font_manager.fontManager.ttflist
                 if font.name in ("PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "Heiti SC")),
                "DejaVu Sans")
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "font.size": 11})
    points = load_points()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    fig.patch.set_facecolor("#F8FAFD")
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.18, top=0.80, hspace=0.38, wspace=0.25)
    fig.text(0.08, 0.945, "room：延长训练到总计 45k", fontsize=24, color="#172A43")
    fig.text(0.08, 0.895, "同一相机序列 · 39 张留出视角 · 独立解码评测 · 单场景、单种子",
             fontsize=12, color="#586A80")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.grid(color="#DFE5EC", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#ADB7C4")
        ax.tick_params(colors="#46556A", labelsize=10)
        ax.set_xticks([30, 32.5, 35, 37.5, 40, 42.5, 45])
        ax.set_xlabel("总训练步数 / 千步")
        ax.axvspan(40, 45, color="#F2F5FA", zorder=0)
    psnr_ax, size_ax, gap_ax, loss_ax = axes.flat
    for arm, (label, color, _) in ARMS.items():
        series = points[arm]
        x = [p["step"] for p in series]
        psnr_ax.plot(x, [p["psnr"] for p in series], "o-", color=color, label=label, lw=1.8, ms=5)
        size_ax.plot(x, [p["size"] for p in series], "o-", color=color, lw=1.8, ms=5)
        loss_ax.plot([p["step"] - 1.25 for p in series[1:]],
                     [p["sampled_loss"] for p in series[1:]], "o-",
                     color=color, lw=1.8, ms=5)
        psnr_ax.annotate(f'{series[-1]["psnr"]:.3f}',
                         (45, series[-1]["psnr"]), (-7, 9), textcoords="offset points",
                         ha="right", color=color, fontsize=10)
        size_ax.annotate(f'{series[-1]["size"]:.3f} MB',
                         (45, series[-1]["size"]), (-7, 9), textcoords="offset points",
                         ha="right", color=color, fontsize=10)
    control, merged = points["control"], points["merged64"]
    gap_ax.plot([p["step"] for p in merged],
                [c["psnr"] - m["psnr"] for c, m in zip(control, merged)],
                "o-", color="#286FAD", lw=1.8, ms=5)
    gap_ax.axhline(0, color="#99A5B5", lw=0.8)
    psnr_ax.set_title("解码 PSNR", loc="left", fontsize=15, pad=10)
    psnr_ax.set_ylabel("PSNR / dB  ↑")
    psnr_ax.margins(y=0.18)
    psnr_ax.legend(frameon=False, fontsize=10, loc="best")
    size_ax.set_title("实际解码文件大小", loc="left", fontsize=15, pad=10)
    size_ax.set_ylabel("文件大小 / MB  ↓")
    size_ax.margins(y=0.18)
    gap_ax.set_title("相同步数的画质差距", loc="left", fontsize=15, pad=10)
    gap_ax.set_ylabel("未合并对照 − 合并 / dB  ↓")
    loss_ax.set_title("训练总损失：每 2,500 步的采样均值", loc="left", fontsize=15, pad=10)
    loss_ax.set_ylabel("总损失（渲染与码率等项） ↓")
    loss_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axes.flat:
        ax.set_xlim(29.25, 45.75)
    gains = [f"{label} {points[arm][-1]['psnr'] - points[arm][-3]['psnr']:+.3f} dB"
             for arm, (label, _, _) in ARMS.items()]
    fig.text(0.08, 0.106, "最后 5k 的 PSNR 净变化：" + "；".join(gains),
             fontsize=11, color="#25364B")
    fig.text(0.08, 0.068,
             "保持原学习率计划，30k 后沿用末值。浅色区域为最后 5k。1 MB = 1,000,000 字节。",
             fontsize=10.5, color="#586A80")
    fig.text(0.08, 0.033, "来源：experiments.csv。画质与字节均经独立解码核验；损失取日志每约 10 步采样，横坐标为分段中点。",
             fontsize=10.2, color="#586A80")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT.with_suffix(suffix), dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
