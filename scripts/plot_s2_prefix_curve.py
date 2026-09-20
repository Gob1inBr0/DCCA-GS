#!/usr/bin/env python3
"""Plot the S2 bytes-PSNR prefix curve from prefix_sweep.json (design doc S2).

Usage:
  python scripts/plot_s2_prefix_curve.py analysis/s2_prefix_sweep/prefix_sweep.json \
      --out docs/figures/s2_prefix_bytes_psnr.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", default="docs/figures/s2_prefix_bytes_psnr.png")
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text())
    results = data["results"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    colors = {"contribution": "#1f77b4", "random": "#d62728"}
    for ordering in ("contribution", "random"):
        pts = [r for r in results if r["ordering"] == ordering]
        pts.sort(key=lambda r: r["bytes_est"])
        xs = [r["bytes_est"] / MB for r in pts]
        ys = [r["psnr_mean"] for r in pts]
        ax.plot(xs, ys, marker="o", color=colors[ordering], label=ordering)
        for r in pts:
            ax.annotate(
                f"{r['prefix']:.0%}",
                (r["bytes_est"] / MB, r["psnr_mean"]),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=7,
                color=colors[ordering],
            )
    full = next(r for r in results if r["prefix"] == 1.00)
    ax.axhline(full["psnr_mean"], ls=":", c="gray", lw=0.8)
    ax.set_xlabel("payload estimate (MB)")
    ax.set_ylabel("PSNR (dB), 16 fixed views")
    ax.set_title(
        "S2 prefix rendering: bytes vs PSNR "
        f"({data['n_anchors']:,} anchors, {data['n_views']} views)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    ov = data["a3_overlap"]
    print("A3 overlap (full prefix, 16 views):")
    print(
        "  blocks/anchor  mean={mean} median={median} p90={p90} max={max}".format(
            **ov["blocks_per_anchor"]
        )
    )
    print(
        "  anchors/block  mean={mean} median={median} p90={p90} max={max}".format(
            **ov["anchors_per_block"]
        )
    )
    gate = ov["blocks_per_anchor"]["mean"]
    verdict = "PASS (>=5)" if gate is not None and gate >= 5 else "FAIL (<5)"
    print(f"  greedy-on-real-edges gate: {verdict}")


if __name__ == "__main__":
    main()
