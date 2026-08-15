"""Compare 4-28 q_scale RD curves across feat dims and report BD-rate."""

from __future__ import annotations

import json
from math import log10


def _load(path: str):
    d = json.load(open(path))
    rows = [r for r in d["rows"] if not r.get("error")]
    rows.sort(key=lambda r: r["total_MB"])
    return rows


def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def bd_rate(anchor, test):
    """BD-rate (percent) of test vs anchor over overlapping PSNR range.
    Positive means test needs more rate at the same quality."""
    xa = [r["psnr"] for r in anchor]
    ya = [log10(r["total_MB"]) for r in anchor]
    xt = [r["psnr"] for r in test]
    yt = [log10(r["total_MB"]) for r in test]
    lo = max(min(xa), min(xt))
    hi = min(max(xa), max(xt))
    if hi <= lo:
        return float("nan")
    total = 0.0
    n = 100
    for i in range(n):
        x = lo + (hi - lo) * i / (n - 1)
        total += _interp(xt, yt, x) - _interp(xa, ya, x)
    return total / n * 100.0


def main() -> None:
    base = "/home/fansonglin/xieliang/chentong/PHG/runs"
    dim50 = _load(f"{base}/rd_4_28_h32/results.json")
    for dim in (16, 32):
        try:
            rows = _load(f"{base}/rd_4_28_dim{dim}/results.json")
        except FileNotFoundError:
            print(f"=== dim{dim} vs dim50 === (results not ready)")
            continue
        print(f"=== dim{dim} vs dim50 ===")
        print("q\ttotal_MB\tPSNR\tSSIM\tLPIPS")
        for r in rows:
            print(
                f"{r['q_scale_feat']:g}\t{r['total_MB']:.4f}\t"
                f"{r['psnr']:.3f}\t{r['ssim']:.4f}\t{r['lpips']:.4f}"
            )
        bd = bd_rate(dim50, rows)
        print(f"BD-rate vs dim50: {bd:+.2f}%")
        print()


if __name__ == "__main__":
    main()
