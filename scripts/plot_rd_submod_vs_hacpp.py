"""RD figure v2: every method is its own complete line.

Scene 1-78, 30k iterations, decoded bitstreams, fp32 size accounting.
Numbers from docs/data/experiments.csv (groups ph0a/ph0c/ph1v3/fusA) and
official same-protocol HAC++ runs.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

HACPP = [(20.0265, 27.8406), (26.2734, 28.2105), (40.6, 28.678)]

BASE = [  # DCCA-GS base: SPA + depth-reinit + complexity multiplier
    (11.0835, 26.9829),  # lambda 0.004
    (14.4530, 27.7298),  # lambda 0.002, post budget r085
    (15.2430, 27.8622),  # lambda 0.002, post budget r100
    (21.4410, 27.5974),  # lambda 0.0005
]

SUBMOD_S42 = [  # + submodular cover (A2), seed 42
    (11.0557, 27.5038),  # lambda 0.004
    (14.4323, 27.8058),  # lambda 0.002
    (20.5475, 27.7186),  # lambda 0.0005
]
SUBMOD_S104 = [  # seed 104 replication
    (11.2760, 27.4046),
    (14.8113, 27.6452),
]

A3 = [  # + sens-weighted submodular cover (A3), seed 42
    (11.0557, 27.4173),  # lambda 0.004
    (14.4529, 26.9848),  # lambda 0.002
    (20.5311, 27.8411),  # lambda 0.0005
]


def bd_rate(ref, test) -> float:
    """% rate change of test vs ref; pchip(log-rate vs PSNR), extrapolated
    where PSNR ranges do not overlap (labelled by the caller)."""
    r1 = np.log([p[0] for p in ref])
    p1 = np.array([p[1] for p in ref])
    r2 = np.log([p[0] for p in test])
    p2 = np.array([p[1] for p in test])
    o = np.argsort(p1)
    f1 = PchipInterpolator(p1[o], r1[o], extrapolate=True)
    o = np.argsort(p2)
    f2 = PchipInterpolator(p2[o], r2[o], extrapolate=True)
    lo, hi = max(p1.min(), p2.min()), min(p1.max(), p2.max())
    xs = np.linspace(lo, hi, 200)
    return (float(np.exp(np.mean(f2(xs) - f1(xs)))) - 1.0) * 100.0


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bd_base = bd_rate(HACPP, BASE)
    bd_sub = bd_rate(HACPP, SUBMOD_S42)
    bd_a3 = bd_rate(HACPP, A3)
    print(f"BD-rate (extrapolated) vs HAC++: base {bd_base:+.1f}% | "
          f"submod {bd_sub:+.1f}% | A3 {bd_a3:+.1f}%")

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)

    h = sorted(HACPP)
    ax.plot([p[0] for p in h], [p[1] for p in h], "s--", color="black",
            ms=9, lw=1.5, label="HAC++ (official, 30k)", zorder=6)

    b = sorted(BASE)
    ax.plot([p[0] for p in b], [p[1] for p in b], "o-", color="#7f7f7f",
            ms=7, lw=1.4, label="DCCA-GS base (30k)", zorder=4)

    s = sorted(SUBMOD_S42)
    ax.plot([p[0] for p in s], [p[1] for p in s], "D-", color="#d62728",
            ms=8, lw=1.6, label="DCCA-GS + submodular cover (30k, s42)",
            zorder=5)
    ax.scatter([p[0] for p in SUBMOD_S104], [p[1] for p in SUBMOD_S104],
               marker="D", facecolors="none", edgecolors="#d62728", s=55,
               zorder=5, label="submodular, seed 104")

    a = sorted(A3)
    ax.plot([p[0] for p in a], [p[1] for p in a], "^-", color="#1f77b4",
            ms=8, lw=1.4, label="DCCA-GS + sens-weighted submod (30k)",
            zorder=3)

    for x, y in HACPP:
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8)

    ax.set_xlabel("decoded size (MB, fp32 accounting)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("DCCA-GS vs HAC++ — scene 1-78, 30k iterations (UAV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(
        0.02, 0.97,
        f"BD-rate vs HAC++ (pchip, extrapolated where ranges differ): "
        f"base {bd_base:+.1f}% | submod {bd_sub:+.1f}% | A3 {bd_a3:+.1f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout()
    out = "docs/figures/rd_submod_vs_hacpp_1-78.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
