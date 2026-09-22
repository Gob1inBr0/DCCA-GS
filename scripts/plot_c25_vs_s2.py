#!/usr/bin/env python3
"""Plot the C2.5 layered curve against the S2 selection curves."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1e6


def main():
    c25 = json.loads(Path(sys.argv[1]).read_text())
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/c25_vs_s2_curves.png"

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pts = sorted(c25["results"], key=lambda r: r["bytes_est"])
    xs = [r["bytes_est"] / MB for r in pts]
    ys = [r["psnr_mean"] for r in pts]
    ax.plot(xs, ys, marker="s", color="#2ca02c", label="C2.5 layered (static coder)")
    for r in pts:
        ax.annotate(
            f"{r['step_factor']}xQ",
            (r["bytes_est"] / MB, r["psnr_mean"]),
            textcoords="offset points", xytext=(6, -10), fontsize=7, color="#2ca02c",
        )
    for ordering, color, ls in (("contribution", "#1f77b4", "-"), ("random", "#d62728", "--")):
        if "s2_selection_reference" in c25:
            pts2 = [r for r in c25["s2_selection_reference"] if r["ordering"] == ordering]
            pts2.sort(key=lambda r: r["bytes_est"])
            ax.plot([r["bytes_est"] / MB for r in pts2],
                    [r["psnr_mean"] for r in pts2],
                    marker="o", color=color, ls=ls,
                    label=f"selection: {ordering} (learned coder)")
    ax.set_xlabel("bytes on the curve's own accounting (MB)")
    ax.set_ylabel("PSNR (dB), 16 fixed views")
    ax.set_title("Layered (all anchors, coarse->fine) vs selection (anchor prefix)\nsame model, same views; NOTE: byte axes use different entropy coders")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
