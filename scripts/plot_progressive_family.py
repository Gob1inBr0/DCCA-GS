#!/usr/bin/env python3
"""Complete comparison figure: progressive family (2 lambdas, 150-view) vs
our own single-rate base vs HAC++ reference, scene 1-78.

Curves/points (all decoded-payload MB, PSNR on the same eval protocol):
  - progressive lambda0.0005: 4 prefixes, 150-view (c25_fullview_fieldaware)
  - progressive lambda0.004:  5 prefixes, 16-view fast (c25_real_l0004_newproto)
  - ours base single-rate:    3 lambdas (learned-context coder, new code)
  - HAC++ reference:          3 rate points
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1e6
A = Path("/Users/chen/Documents/DCCA-GS/analysis/s2_prefix_sweep")

HACPP = [(20.0265, 27.8406), (26.2734, 28.2105), (40.5589, 28.6780)]
BASE = [(10.6675, 27.3940), (13.9042, 27.6946), (19.944, 27.8125)]


def main():
    fv5 = json.loads((A / "c25_fullview_fieldaware.json").read_text())
    fv5_bytes = {0: 15.75, 1: 20.94, 2: 24.55, 3: 29.95}
    l4 = json.loads((A / "c25_real_l0004_newproto.json").read_text())

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # HAC++
    ax.plot([p[0] for p in HACPP], [p[1] for p in HACPP], marker="s", ms=7,
            ls="--", color="#7f7f7f",
            label="HAC++ (3 independent files, official impl.)")

    # our single-rate base (learned-context coder)
    ax.plot([p[0] for p in BASE], [p[1] for p in BASE], marker="^", ms=7,
            ls=":", color="#2ca02c",
            label="Ours base: single-rate (learned context coder)")

    # progressive lambda0.0005 (150-view)
    px = [fv5_bytes[r["prefix"]] for r in fv5["results"]]
    py = [r["psnr_mean"] for r in fv5["results"]]
    pe = [r["psnr_std"] for r in fv5["results"]]
    ax.errorbar(px, py, yerr=pe, marker="o", ms=6, capsize=3,
                color="#1f77b4",
                label="Ours progressive λ=0.0005 (1 file, 4 truncation points, 150 views)")

    # progressive lambda0.004 (16-view fast)
    lx, ly = [], []
    for r in sorted(l4["results"], key=lambda r: r["real_bytes"]):
        lx.append(r["real_bytes"] / MB)
        ly.append(r["psnr_mean"])
    ax.plot(lx, ly, marker="D", ms=5, color="#9467bd",
            label="Ours progressive λ=0.004 (1 file, low-rate family)")

    ax.annotate("HAC++: 3 separate files,\nno progressive capability",
                (40.56, 28.678), textcoords="offset points", xytext=(-90, 8),
                fontsize=8, color="#7f7f7f")
    ax.annotate("gap = layered-coder\nefficiency (see report §7)",
                (29.95, 26.703), textcoords="offset points", xytext=(10, -24),
                fontsize=8, color="#1f77b4")

    ax.set_xlabel("decoded payload (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("1-78 scene: progressive layered bitstream family\n"
                 "vs single-rate base vs HAC++ (same eval protocol)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = "/Users/chen/Documents/DCCA-GS/docs/figures/progressive_family_vs_hacpp_1-78.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
