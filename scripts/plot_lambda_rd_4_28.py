"""Plot the 4-28 lambda-only RD curves (PHG vs old HAC lambda sweeps).

No q_scale points: only models trained with different lambda_rate.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAMBDA_RUN = "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_l0p002"
DIM_RUNS = {
    "dim32": "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_dim32",
    "dim16": "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_dim16",
}
BEST_RUN_JSON = (
    "/home/fansonglin/xieliang/chentong/PHG/runs/"
    "mlp_quant_sens_cd8_rest16_110k/results.json"
)
OLD_RD_CSV = "/home/fansonglin/xieliang/chentong/PHG/old_hac_data/rd_main_curves_3scene_260708.csv"
OUT = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/lambda_rd_4_28_v4.png"


def _load_metrics(run_dir: str):
    """Return decoded metrics JSON, trying f1 (1600-wide) then default dir."""
    for sub in ("decoded_eval_f1", "decoded_eval"):
        path = f"{run_dir}/{sub}/metrics.jsonl"
        if os.path.exists(path):
            return json.load(open(path))
    return None


def main() -> None:
    # PHG lambda points: lambda=0.004 (known) + lambda=0.002 (from run dir).
    lam = [(0.004, 5.5604, 28.655061937967936)]
    try:
        meta = json.load(open(f"{LAMBDA_RUN}/bitstreams/hac_meta.json"))
        met = json.load(open(f"{LAMBDA_RUN}/decoded_eval/metrics.jsonl"))
        lam.append((0.002, meta["total_MB"], met["psnr"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    # PHG best operating point: 110k + conservative MLP quantization
    # (complexity/deform 8-bit, rest 16-bit; dim50 h32).
    best_point = None
    try:
        brd = json.load(open(BEST_RUN_JSON))
        brow = [r for r in brd["rows"] if not r.get("error")][0]
        best_point = (brow["total_MB"], brow["psnr"])
        print(
            f"[plot] best point: {brow['psnr']:.3f} dB / "
            f"{brow['total_MB']:.4f} MB"
        )
    except (FileNotFoundError, IndexError, KeyError, json.JSONDecodeError):
        print("[plot] WARNING: best point not available yet")

    # PHG dim16/dim32 single points (trained at lambda=0.004, 90k).
    dim_points = {}
    for name, run_dir in DIM_RUNS.items():
        try:
            meta = json.load(open(f"{run_dir}/bitstreams/hac_meta.json"))
            met = _load_metrics(run_dir)
            if met is not None:
                dim_points[name] = (meta["total_MB"], met["psnr"])
                print(f"[plot] loaded dim point {name}: "
                      f"{met['psnr']:.3f} dB / {meta['total_MB']:.4f} MB")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            print(f"[plot] WARNING: could not load dim point {name}: {exc}")
    if not dim_points:
        print("[plot] WARNING: no dim points loaded")

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

    # dim16 / dim32 markers (same lambda=0.004).
    dim_markers = {
        "dim32": ("D", "tab:green", "PHG dim32 (λ=0.004)"),
        "dim16": ("P", "tab:cyan", "PHG dim16 (λ=0.004)"),
    }
    for name, (marker, color, label) in dim_markers.items():
        if name not in dim_points:
            continue
        s, p = dim_points[name]
        ax.plot(s, p, marker, color=color, ms=9, label=label)
        ax.annotate(
            f"{name}\nλ=0.004",
            (s, p),
            textcoords="offset points",
            xytext=(0, -18),
            ha="center",
            fontsize=8,
            color=color,
        )

    if best_point is not None:
        s, p = best_point
        ax.plot(
            s,
            p,
            "*",
            color="gold",
            ms=20,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="PHG best (110k + MLP quant)",
        )
        ax.annotate(
            f"110k + quant\n{p:.2f} dB / {s:.2f} MB",
            (s, p),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
            fontweight="bold",
            color="black",
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
