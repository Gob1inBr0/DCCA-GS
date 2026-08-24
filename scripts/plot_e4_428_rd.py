#!/usr/bin/env python3
"""Scientific RD figure: 4-28 E4 MiniSplat vs HAC++ official.

Data are decoded HAC++ bitstream results:
- HAC++ official 60k curve from old_hac_data/rd_main_curves_3scene_260708.csv
  (method_total_mb and decoded PSNR).
- E4 110k MiniSplat cells from 5090 runs:
  p0_e4_428_110k_spa_mini and p0_e4_428_110k_nospa_mini.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (lambda, method_total_MB, PSNR dB)
HAC_OFFICIAL_428 = [
    (0.001, 12.2742, 29.0592),
    (0.002, 9.7700, 28.8535),
    (0.004, 7.4309, 28.6702),
    (0.006, 5.9862, 28.4498),
]

# E4 4-28 110k decoded points (lambda=0.004 only).
MINI_SPA = (0.004, 5.4490, 28.7359)
MINI_NO_SPA = (0.004, 5.4329, 28.7594)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/figures"),
        help="output directory for e4_428_rd.png/pdf",
    )
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    hac_x = [row[1] for row in HAC_OFFICIAL_428]
    hac_y = [row[2] for row in HAC_OFFICIAL_428]
    ax.plot(
        hac_x,
        hac_y,
        "o-",
        color="#4C72B0",
        linewidth=1.8,
        markersize=6,
        label="HAC++ official (60k)",
        zorder=3,
    )
    for x, y, lam in HAC_OFFICIAL_428:
        ax.annotate(
            rf"$\lambda$={lam:g}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="#4C72B0",
        )

    ax.scatter(
        [MINI_SPA[1]],
        [MINI_SPA[2]],
        s=90,
        marker="s",
        color="#DD8452",
        label="MiniSplat + SPA (E4, 110k)",
        zorder=4,
    )
    ax.scatter(
        [MINI_NO_SPA[1]],
        [MINI_NO_SPA[2]],
        s=90,
        marker="^",
        color="#55A868",
        label="MiniSplat, no SPA (E4, 110k)",
        zorder=4,
    )
    ax.annotate(
        f"{MINI_SPA[2]:.3f} dB\n{MINI_SPA[1]:.3f} MiB",
        (MINI_SPA[1], MINI_SPA[2]),
        textcoords="offset points",
        xytext=(-12, 12),
        fontsize=8,
        color="#DD8452",
    )
    ax.annotate(
        f"{MINI_NO_SPA[2]:.3f} dB\n{MINI_NO_SPA[1]:.3f} MiB",
        (MINI_NO_SPA[1], MINI_NO_SPA[2]),
        textcoords="offset points",
        xytext=(10, -16),
        fontsize=8,
        color="#55A868",
    )

    ax.set_xlabel("Decoded payload size (MiB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("4-28 dataset: MiniSplat E4 vs HAC++ RD")
    ax.set_xlim(5.0, 13.1)
    ax.set_ylim(28.3, 29.2)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "e4_428_rd.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "e4_428_rd.pdf", bbox_inches="tight")
    print(f"Wrote {out_dir / 'e4_428_rd.png'}")
    print(f"Wrote {out_dir / 'e4_428_rd.pdf'}")


if __name__ == "__main__":
    main()
