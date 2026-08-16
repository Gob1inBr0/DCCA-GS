"""Plot the 4-28 lambda-only RD curves (PHG vs old HAC lambda sweeps).

No q_scale points: only models trained with different lambda_rate.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAMBDA_RUN = "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_l0p002"
OLD_RD_CSV = "/home/fansonglin/xieliang/chentong/PHG/old_hac_data/rd_main_curves_3scene_260708.csv"
OUT = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/lambda_rd_4_28.png"


def main() -> None:
    # PHG lambda points: lambda=0.004 (known) + lambda=0.002 (from run dir).
    lam = [(0.004, 5.5604, 28.655061937967936)]
    try:
        meta = json.load(open(f"{LAMBDA_RUN}/bitstreams/hac_meta.json"))
        met = json.load(open(f"{LAMBDA_RUN}/decoded_eval/metrics.jsonl"))
        lam.append((0.002, meta["total_MB"], met["psnr"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    # Old HAC 4-28 lambda curves.
    old = {}  # method -> list of (lambda, size_mb, psnr)
    with open(OLD_RD_CSV) as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 9 or cols[0] != "4-28":
                continue
            method = cols[1]
            try:
                lmb = float(cols[2])
                p = float(cols[3])
                s = float(cols[8])
            except ValueError:
                continue
            old.setdefault(method, []).append((lmb, s, p))

    labels = {
        "official_hacpp_60k": "old official HAC++ (60k)",
        "ct_formula_i1_hybrid_90k": "old CT formula+I1 (90k)",
        "ct_shared_all_i1_hybrid_90k": "old CT shared+I1 (90k)",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "official_hacpp_60k": "tab:gray",
        "ct_formula_i1_hybrid_90k": "tab:orange",
        "ct_shared_all_i1_hybrid_90k": "tab:purple",
    }
    for method, pts in old.items():
        if method not in labels:
            continue
        pts.sort(key=lambda t: t[1])
        ls = [t[1] for t in pts]
        ps = [t[2] for t in pts]
        ax.plot(
            ls,
            ps,
            "o--",
            color=colors[method],
            alpha=0.7,
            label=labels[method],
        )
        for lmb, s, p in pts:
            ax.annotate(
                f"λ={lmb:g}",
                (s, p),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=colors[method],
            )

    # PHG lambda curve.
    lam_sorted = sorted(lam, key=lambda t: t[1])
    ax.plot(
        [t[1] for t in lam_sorted],
        [t[2] for t in lam_sorted],
        "s-",
        color="tab:red",
        lw=2,
        label="PHG (I2+I6, 90k, dim50 h32)",
    )
    for lmb, s, p in lam_sorted:
        ax.annotate(
            f"λ={lmb:g}",
            (s, p),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color="tab:red",
            fontweight="bold",
        )

    ax.set_xlabel("total size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("4-28 lambda RD (PHG vs old HAC lambda sweeps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
