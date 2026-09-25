#!/usr/bin/env python3
"""Plot: layered-bitstream progressive curve vs HAC++ reference points (1-78).

Progressive curve = full-view (150-view) evaluation of the field-aware
layered bitstream (P0..P3). HAC++ = three official-implementation points on
1-78 (from docs/03-reports/1-78_30k_32dim_DCCA_vs_HACpp.md section 2).
Both axes: decoded-payload MB and decoded PSNR; per-view error bars for the
progressive points from the saved per-view arrays.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1e6

# HAC++ 1-78 reference points (official implementation, same eval protocol)
HACPP = [
    {"psnr": 27.8406, "mb": 20.0265},
    {"psnr": 28.2105, "mb": 26.2734},
    {"psnr": 28.6780, "mb": 40.5589},
]

# progressive curve byte sizes from the 16-view run (same code path)
P_BYTES = {0: 15.75, 1: 20.94, 2: 24.55, 3: 29.95}  # MB, real payload
P_LABELS = {
    0: "P0\nfeat/off ×8, scale ×2",
    1: "P1\nfeat/off ×4",
    2: "P2\nfeat/off ×2",
    3: "P3\nfull precision",
}


def main():
    fv = json.loads(Path(
        "/Users/chen/Documents/DCCA-GS/analysis/s2_prefix_sweep/"
        "c25_fullview_fieldaware.json").read_text())
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    # HAC++ points
    xs = [p["mb"] for p in HACPP]
    ys = [p["psnr"] for p in HACPP]
    ax.plot(xs, ys, marker="s", ms=7, ls="--", color="#7f7f7f",
            label="HAC++ (official implementation, 3 rate points)")

    # progressive curve with per-view std error bars
    px = [P_BYTES[r["prefix"]] for r in fv["results"]]
    py = [r["psnr_mean"] for r in fv["results"]]
    pe = [r["psnr_std"] for r in fv["results"]]
    ax.errorbar(px, py, yerr=pe, marker="o", ms=6, capsize=3,
                color="#1f77b4", label="Ours: progressive layered bitstream (150 views, ±std)")

    for r in fv["results"]:
        ax.annotate(P_LABELS[r["prefix"]].replace("\n", " "),
                    (P_BYTES[r["prefix"]], r["psnr_mean"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=7.5,
                    color="#1f77b4")
    for p in HACPP:
        ax.annotate(f"{p['psnr']:.2f} dB\n{p['mb']:.1f} MB",
                    (p["mb"], p["psnr"]),
                    textcoords="offset points", xytext=(6, -16),
                    fontsize=7.5, color="#7f7f7f")

    ax.set_xlabel("decoded payload (MB)")
    ax.set_ylabel("PSNR (dB), all 150 val views")
    ax.set_title("1-78: progressive layered bitstream vs HAC++\n"
                 "(error bars = per-view std, same eval protocol)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = "/Users/chen/Documents/DCCA-GS/docs/figures/progressive_vs_hacpp_1-78.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)

    # console comparison at matched quality
    print("\nmatched-quality reading:")
    for p in HACPP:
        # nearest progressive point
        best = min(fv["results"], key=lambda r: abs(r["psnr_mean"] - p["psnr"]))
        print(f"  HAC++ {p['psnr']:.2f} dB @ {p['mb']:.2f} MB  <->  "
              f"ours P{best['prefix']} {best['psnr_mean']:.2f} dB @ "
              f"{P_BYTES[best['prefix']]:.2f} MB")


if __name__ == "__main__":
    main()
