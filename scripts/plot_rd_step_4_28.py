"""Plot the 4-28 RD curve (q_scale sweep) and step sweep results."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RD_JSON = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/results.json"
BASE = "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32"
OUT = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/rd_step_4_28.png"


def main() -> None:
    rd = json.load(open(RD_JSON))
    rows = [r for r in rd["rows"] if not r.get("error")]
    rows.sort(key=lambda r: r["total_MB"])
    mb = [r["total_MB"] for r in rows]
    psnr = [r["psnr"] for r in rows]
    qs = [r["q_scale_feat"] for r in rows]

    steps = [30000, 60000, 90000]
    s_mb = []
    s_psnr = []
    for s in steps:
        meta = json.load(open(f"{BASE}/bitstreams_step_{s}/hac_meta.json"))
        met = json.load(open(f"{BASE}/eval_step_{s}/metrics.jsonl"))
        s_mb.append(meta["total_MB"])
        s_psnr.append(met["psnr"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(mb, psnr, "o-", color="tab:blue")
    for r, q in zip(rows, qs):
        ax.annotate(
            f"q={q:g}",
            (r["total_MB"], r["psnr"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    for label, x, y in [
        ("ours 90k", 6.3547, 28.563),
        ("HAC++", 6.9462, 28.311),
    ]:
        ax.plot(x, y, "x", color="red", ms=8)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, -2),
            fontsize=8,
            color="red",
        )
    ax.set_xlabel("total size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("4-28 RD curve (h32 90k, 1600-wide, joint q_scale)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    (l1,) = ax.plot(steps, s_psnr, "o-", color="tab:green", label="PSNR")
    (l2,) = ax2.plot(steps, s_mb, "s--", color="tab:orange", label="size")
    ax.set_xlabel("training steps")
    ax.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("total size (MB)")
    ax.set_title("4-28 step sweep (h32)")
    ax.grid(alpha=0.3)
    ax.legend([l1, l2], ["PSNR", "size"], loc="center right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
