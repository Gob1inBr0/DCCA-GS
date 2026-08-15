"""Plot the 4-28 RD curves (q_scale sweep + PHG lambda + old HAC data)
and the step sweep results."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RD_JSON = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/results.json"
BASE = "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32"
LAMBDA_RUN = "/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_l0p002"
OLD_RD_CSV = "/home/fansonglin/xieliang/chentong/PHG/old_hac_data/rd_main_curves_3scene_260708.csv"
OUT = "/home/fansonglin/xieliang/chentong/PHG/runs/rd_4_28_h32/rd_step_4_28.png"


def main() -> None:
    rd = json.load(open(RD_JSON))
    rows = [r for r in rd["rows"] if not r.get("error")]
    rows.sort(key=lambda r: r["total_MB"])
    mb = [r["total_MB"] for r in rows]
    psnr = [r["psnr"] for r in rows]
    qs = [r["q_scale_feat"] for r in rows]

    # PHG lambda points: lambda=0.004 (known) and lambda=0.002 (optional).
    lam = [(0.004, 5.5604, 28.655061937967936)]
    try:
        meta = json.load(open(f"{LAMBDA_RUN}/bitstreams/hac_meta.json"))
        met = json.load(open(f"{LAMBDA_RUN}/decoded_eval/metrics.jsonl"))
        lam.append((0.002, meta["total_MB"], met["psnr"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    # Old HAC 4-28 lambda curves (method_total_mb as size).
    old = {}  # method -> (lambdas, sizes, psnrs)
    try:
        with open(OLD_RD_CSV) as f:
            for line in f:
                cols = line.strip().split(",")
                if len(cols) < 9 or cols[0] != "4-28":
                    continue
                method, lmb = cols[1], float(cols[2])
                try:
                    p, s = float(cols[3]), float(cols[8])
                except ValueError:
                    continue
                old.setdefault(method, ([], [], []))
                old[method][0].append(lmb)
                old[method][1].append(s)
                old[method][2].append(p)
    except FileNotFoundError:
        pass

    steps = [30000, 60000, 90000]
    s_mb = []
    s_psnr = []
    for s in steps:
        meta = json.load(open(f"{BASE}/bitstreams_step_{s}/hac_meta.json"))
        met = json.load(open(f"{BASE}/eval_step_{s}/metrics.jsonl"))
        s_mb.append(meta["total_MB"])
        s_psnr.append(met["psnr"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(mb, psnr, "o-", color="tab:blue")
    for r, q in zip(rows, qs):
        ax.annotate(
            f"q={q:g}",
            (r["total_MB"], r["psnr"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    for label, x, y in [
        ("HAC++ paper", 6.9462, 28.311),
        ("old ours 90k", 6.3547, 28.563),
    ]:
        ax.plot(x, y, "x", color="red", ms=8)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, -2),
            fontsize=8,
            color="red",
        )
    for method, (ls, ss, pp) in old.items():
        order = sorted(range(len(ls)), key=lambda i: ss[i])
        label = method
        if method == "ct_formula_i1_hybrid_90k":
            label = "old CT formula+I1"
        elif method == "ct_shared_all_i1_hybrid_90k":
            label = "old CT shared+I1"
        elif method == "official_hacpp_60k":
            label = "old official HAC++"
        ax.plot(
            [ss[i] for i in order],
            [pp[i] for i in order],
            "o--",
            ms=4,
            alpha=0.55,
            label=label,
        )
    # PHG lambda 2-point curve.
    ax.plot(
        [l[1] for l in lam],
        [l[2] for l in lam],
        "s-",
        color="tab:red",
        label="PHG lambda",
    )
    for l, (_, s, p) in zip([0.004] + [0.002] * (len(lam) - 1), lam):
        ax.annotate(
            f"λ={l:g}",
            (s, p),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8,
            color="tab:red",
        )
    ax.set_xlabel("total size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("4-28 RD curves (PHG h32 90k vs old HAC)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    (l1,) = ax.plot(steps, s_psnr, "o-", color="tab:green", label="PSNR")
    (l2,) = ax2.plot(steps, s_mb, "s--", color="tab:orange", label="size")
    ax.set_xlabel("training steps")
    ax.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("total size (MB)")
    ax.set_title("4-28 step sweep (h32)")
    ax.grid(alpha=0.3)
    ax.legend([l1, l2], ["PSNR", "size"], loc="center right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
