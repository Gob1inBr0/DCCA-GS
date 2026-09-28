"""Plot LYH large-scene RD curves with matplotlib.

Usage:
    python scripts/plot_lyh_rd.py --summary docs/data/lyh_rd_1-78.json \
      --out docs/figures/lyh_rd_1-78.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_points(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [r for r in data.get("rows", []) if r.get("psnr") and r.get("total_MB")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Large-scene RD (decoded, 1600px)")
    args = p.parse_args()

    rows = _load_points(Path(args.summary))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    hac_points = [r for r in rows if r.get("method") == "hacpp"]
    dcca_points = [r for r in rows if r.get("method", "").startswith("dcca")]

    if hac_points:
        hac_points.sort(key=lambda r: r["total_MB"])
        ax.plot(
            [r["total_MB"] for r in hac_points],
            [r["psnr"] for r in hac_points],
            "o--",
            color="tab:gray",
            label="HAC++ (reference)",
        )
        for r in hac_points:
            ax.annotate(
                f"{r.get('tag', '')}",
                (r["total_MB"], r["psnr"]),
                textcoords="offset points",
                xytext=(5, -8),
                fontsize=7,
                color="tab:gray",
            )

    if dcca_points:
        dcca_points.sort(key=lambda r: r["total_MB"])
        ax.plot(
            [r["total_MB"] for r in dcca_points],
            [r["psnr"] for r in dcca_points],
            "s-",
            color="tab:red",
            label="DCCA-GS (no-SPA)",
        )
        for r in dcca_points:
            ax.annotate(
                f"λ={r.get('tag', '').split('_')[-1]}",
                (r["total_MB"], r["psnr"]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=7,
                color="tab:red",
            )

    ax.set_xscale("log")
    ax.set_xlabel("total size (MiB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
