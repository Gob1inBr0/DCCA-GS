"""RD figure + BD-rate: submodular selection vs current-best vs HAC++.

Scene 1-78, 30k iterations (decoded bitstreams, fp32 sizes), seed 42 (+104
for submodular). All numbers from docs/data/experiments.csv (groups ph0a/
ph0c/ph1v3/fusA) and the official HAC++ same-protocol runs.

BD-rate: log-rate pchip integral over the overlapping PSNR interval; where
the PSNR ranges do not overlap, the reference curve is pchip-EXTRAPOLATED
and the number is labelled as such (less reliable).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Data (MB, PSNR) — decoded
# ---------------------------------------------------------------------------

HACPP = [(20.0265, 27.8406), (26.2734, 28.2105), (40.6, 28.678)]

SUBMOD = [  # A2 submodular cover, seed 42 (+104 marked separately)
    (11.0557, 27.5038, "s42"),
    (11.2760, 27.4046, "s104"),
    (14.4323, 27.8058, "s42"),
    (14.8113, 27.6452, "s104"),
    (20.5475, 27.7186, "s42"),
]

BEST_NONSUB = [  # current best excluding submodular arms
    (11.0835, 26.9829, "base l0.004"),
    (14.4530, 27.7298, "base r085 l0.002"),
    (15.2430, 27.8622, "base r100 l0.002"),
    (20.5311, 27.8411, "A3 l0.0005"),
    (21.4410, 27.5974, "base l0.0005"),
]

OURS_60K = [(12.50, 27.949), (16.71, 28.372)]  # monotone prefix


def bd_rate(ref, test, lo=None, hi=None) -> float:
    """% rate change of `test` vs `ref` (negative = test cheaper).

    Piecewise-cubic (pchip) interpolation of log-rate over PSNR; integration
    over [lo, hi] defaults to the overlapping PSNR range (may extrapolate
    the curves outside their data — labelled by the caller).
    """
    r1 = np.log([p[0] for p in ref])
    p1 = np.array([p[1] for p in ref])
    r2 = np.log([p[0] for p in test])
    p2 = np.array([p[1] for p in test])
    order = np.argsort(p1)
    f1 = PchipInterpolator(p1[order], r1[order], extrapolate=True)
    order = np.argsort(p2)
    f2 = PchipInterpolator(p2[order], r2[order], extrapolate=True)
    lo = max(p1.min(), p2.min()) if lo is None else lo
    hi = min(p1.max(), p2.max()) if hi is None else hi
    xs = np.linspace(lo, hi, 200)
    avg = float(np.mean(f2(xs) - f1(xs)))
    return (np.exp(avg) - 1.0) * 100.0


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bd_sub_hac = bd_rate(HACPP, SUBMOD, lo=27.50, hi=27.81)     # extrapolated
    bd_best_hac = bd_rate(HACPP, BEST_NONSUB, lo=27.50, hi=27.86)  # extrapolated
    bd_sub_best = bd_rate(BEST_NONSUB, SUBMOD, lo=27.50, hi=27.73)  # clean overlap
    bd_60_hac = bd_rate(HACPP, OURS_60K)                        # clean overlap

    print(f"BD-rate submod vs HAC++  [extrapolated 27.50-27.81]: {bd_sub_hac:+.1f}%")
    print(f"BD-rate best  vs HAC++  [extrapolated 27.50-27.86]: {bd_best_hac:+.1f}%")
    print(f"BD-rate submod vs best   [overlap 27.50-27.73]:     {bd_sub_best:+.1f}%")
    print(f"BD-rate 60k   vs HAC++  [overlap 27.95-28.37]:      {bd_60_hac:+.1f}%")

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)

    h = sorted(HACPP)
    ax.plot([p[0] for p in h], [p[1] for p in h], "s--", color="black",
            ms=9, lw=1.4, label="HAC++ (official, 30k)", zorder=5)
    for x, y in HACPP:
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="black")

    b = sorted(BEST_NONSUB, key=lambda p: p[0])
    ax.plot([p[0] for p in b], [p[1] for p in b], "o-", color="#7f7f7f",
            ms=7, lw=1.3, label="DCCA-GS best (no submod, 30k)", zorder=4)
    for x, y, lab in BEST_NONSUB:
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(4, -12), fontsize=7, color="#7f7f7f")

    s42 = sorted([p[:2] for p in SUBMOD if p[2] == "s42"])
    ax.plot([p[0] for p in s42], [p[1] for p in s42], "D-", color="#d62728",
            ms=8, lw=1.6, label="DCCA-GS + submodular cover (30k)", zorder=6)
    s104 = [p[:2] for p in SUBMOD if p[2] == "s104"]
    ax.scatter([p[0] for p in s104], [p[1] for p in s104], marker="D",
               facecolors="none", edgecolors="#d62728", s=52, zorder=6,
               label="submodular, seed 104")

    ax.annotate("+0.52 dB @ ~11 MB\n(submodular vs no-submod best)",
                xy=(11.06, 27.50), xytext=(14.8, 27.02), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
                color="#d62728")
    ax.annotate("ties HAC++ @ ~20.5 MB", xy=(20.53, 27.84),
                xytext=(23.5, 27.55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.1))

    ax.set_xlabel("decoded size (MB, fp32 accounting)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("DCCA-GS vs HAC++ — scene 1-78, 30k iterations (UAV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.02, 0.02,
        f"BD-rate vs HAC++ (30k, pchip): submod {bd_sub_hac:+.1f}%* | "
        f"best-no-submod {bd_best_hac:+.1f}%*\n(* extrapolated) | "
        f"submod vs best-no-submod: {bd_sub_best:+.1f}% | "
        f"60k ours vs HAC++ 30k: {bd_60_hac:+.1f}% (iteration mismatch)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
        color="#444444",
    )
    fig.tight_layout()
    out = "docs/figures/rd_submod_vs_hacpp_1-78.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
