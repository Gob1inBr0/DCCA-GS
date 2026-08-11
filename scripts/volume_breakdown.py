"""Print bitstream volume breakdown by category for a PHG bitstream dir.

Usage (5090, HAC_5090_a100 env):

    python scripts/volume_breakdown.py --bitstream-dir runs/<scene>/bitstreams
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hacplus.utils.codec_consistency import classify_codec_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bitstream-dir", required=True)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    root = Path(args.bitstream_dir)
    sizes: dict = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        category, required = classify_codec_file(path.name)
        if category == "unknown":
            raise RuntimeError(f"unclassified bitstream file: {path}")
        if not required:
            continue
        sizes.setdefault(category, 0)
        sizes[category] += path.stat().st_size

    total = sum(sizes.values())
    rows = []
    for category in sorted(sizes):
        mb = sizes[category] / 1e6
        rows.append(
            {
                "category": category,
                "bytes": sizes[category],
                "mb": round(mb, 6),
                "pct": round(100.0 * sizes[category] / total, 3) if total else 0.0,
            }
        )
    rows.append(
        {
            "category": "total",
            "bytes": total,
            "mb": round(total / 1e6, 6),
            "pct": 100.0,
        }
    )

    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "bytes", "mb", "pct"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
