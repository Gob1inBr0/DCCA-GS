"""Build the PHG quality-tier RD envelope (dim16/32/50 + lambda points).

The envelope is the Pareto frontier (max PSNR for a given size) over all
existing RD points: dim50/dim32/dim16 q_scale sweeps plus the PHG lambda
points. It demonstrates whether "lower dim at high lambda, higher dim at low
lambda" buys a curve above every single-dimension curve.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "/home/fansonglin/xieliang/chentong/PHG/runs"
LAMBDA_RUN = "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_l0p002"
OUT = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/rd_envelope_4_28.png"


def _load_rd(path: str):
    d = json.load(open(path))
    return [
        (r["total_MB"], r["psnr"], f"q={r['q_scale_feat']:g}")
        for r in d["rows"]
        if not r.get("error")
    ]


def _load_metrics(run_dir: str):
    for sub in ("decoded_eval_f1", "decoded_eval"):
        p = f"{run_dir}/{sub}/metrics.jsonl"
        if os.path.exists(p):
            return json.load(open(p))
    return None


def _pareto(points):
    """Return Pareto frontier (non-dominated by size/PSNR), sorted by size."""
    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    frontier = []
    best_psnr = -1e9
    for s, p, label in pts:
        if p > best_psnr + 1e-9:
            frontier.append((s, p, label))
            best_psnr = p
    return frontier


def main() -> None:
    curves = {
        "PHG dim50 h32 q_scale": _load_rd(f"{BASE}/rd_4_28_h32/results.json"),
        "PHG dim32 q_scale": _load_rd(f"{BASE}/rd_4_28_dim32/results.json"),
        "PHG dim16 q_scale": _load_rd(f"{BASE}/rd_4_28_dim16/results.json"),
    }
    colors = {
        "PHG dim50 h32 q_scale": "tab:blue",
        "PHG dim32 q_scale": "tab:purple",
        "PHG dim16 q_scale": "tab:green",
    }

    # PHG lambda points.
    lam = [(5.5604, 28.655061937967936, "λ=0.004")]
    try:
        meta = json.load(open(f"{LAMBDA_RUN}/bitstreams/hac_meta.json"))
        met = _load_metrics(LAMBDA_RUN)
        if met is not None:
            lam.append((meta["total_MB"], met["psnr"], "λ=0.002"))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    all_points = []
    for pts in curves.values():
        all_points.extend(pts)
    all_points.extend(lam)
    frontier = _pareto(all_points)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, pts in curves.items():
        pts_sorted = sorted(pts)
        ax.plot(
            [t[0] for t in pts_sorted],
            [t[1] for t in pts_sorted],
            "o-",
            ms=4,
            color=colors[name],
            alpha=0.7,
            label=name,
        )
    ax.plot(
        [t[0] for t in lam],
        [t[1] for t in lam],
        "s-",
        color="tab:red",
        lw=2,
        label="PHG lambda (dim50)",
    )
    for s, p, label in lam:
        ax.annotate(
            label,
            (s, p),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="tab:red",
        )

    fx = [t[0] for t in frontier]
    fy = [t[1] for t in frontier]
    ax.plot(
        fx,
        fy,
        "-",
        color="black",
        lw=2.5,
        marker="*",
        ms=10,
        label=f"quality-tier envelope ({len(frontier)} pts)",
    )
    for s, p, label in frontier:
        ax.annotate(
            label,
            (s, p),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=7,
        )

    ax.set_xlabel("total size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("4-28 quality-tier RD envelope (dim16/32/50 + lambda)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)

    print("frontier points (size_MB, psnr, config):")
    for s, p, label in frontier:
        print(f"  {s:.4f}  {p:.4f}  {label}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
