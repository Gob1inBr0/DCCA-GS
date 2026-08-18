"""Plot Deep Blending RD curves: curated survey methods vs PHG lambda points.

Survey data: docs/data/DeepBlending_survey.csv (aggregate from
https://w-m.github.io/3dgs-compression-survey/, average of playroom+drjohnson).
PHG data: runs/db_rd_110k.json (per-scene lambda-RD; this script averages the
two scenes to match the survey aggregate basis).

Only high-quality compression methods are kept; low-quality / uncompressed
baselines (3DGS, FCGS, EAGLES, Compact3D, etc.) are dropped for readability.
The top-left cluster is zoomed and every point is annotated with a small
offset so overlapping points are separable.

Usage (5090):
    PYTHONPATH=$PWD python scripts/plot_db_rd.py \
      --survey docs/data/DeepBlending_survey.csv \
      --phg /home/fansonglin/data_space/web_scan/runs/db_rd_110k.json \
      --out /home/fansonglin/data_space/web_scan/runs/db_rd_curve.png
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mb_from_bytes(b: float) -> float:
    return b / 1024.0 / 1024.0


# curated high-quality compression methods (survey aggregate, DB)
INCLUDE = {
    "chen2024hac",
    "chen2025hac-plus",
    "liu2024compgs",
    "liu2024hemgs",
    "wang2024contextgs",
    "wang2026smolgs",
}
NAMES = {
    "chen2024hac": "HAC",
    "chen2025hac-plus": "HAC++",
    "liu2024compgs": "CompGS",
    "liu2024hemgs": "HEMGS",
    "wang2024contextgs": "ContextGS",
    "wang2026smolgs": "Smol-GS",
}

# per-point annotation offsets in points (dx, dy), keyed by (method, sub)
ANNO_OFFSETS = {
    ("chen2024hac", "-lowrate"): (6, -8),
    ("chen2024hac", "-highrate"): (6, 7),
    ("chen2025hac-plus", "-lowrate"): (7, 8),
    ("chen2025hac-plus", "-highrate"): (6, -9),
    ("liu2024compgs", ""): (8, 8),
    ("liu2024compgs", "Baseline"): (8, -6),
    ("liu2024hemgs", "-lowrate"): (-8, 7),
    ("liu2024hemgs", "-highrate"): (7, 7),
    ("wang2024contextgs", "_lowrate"): (7, 6),
    ("wang2024contextgs", "_highrate"): (7, -8),
    ("wang2026smolgs", "-base"): (-10, 7),
    ("wang2026smolgs", "-large"): (7, -7),
}

# zoom to the top-left region (MiB)
X_LIM = (2.0, 10.0)
PSNR_LIM = (29.0, 30.7)
SSIM_LIM = (0.885, 0.915)
LPIPS_LIM = (0.23, 0.34)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--survey", default="docs/data/DeepBlending_survey.csv")
    p.add_argument("--phg", default="runs/db_rd_110k.json")
    p.add_argument("--phg-spa", default="runs/db_spa_rd_110k.json")
    p.add_argument("--out", default="runs/db_rd_curve.png")
    args = p.parse_args()

    # --- survey rows ------------------------------------------------------
    survey = []
    with open(args.survey, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Method"] not in INCLUDE:
                continue
            try:
                size = float(row["Size [Bytes]"])
                psnr = float(row["PSNR"])
            except (ValueError, TypeError):
                continue
            survey.append(
                {
                    "method": row["Method"],
                    "sub": (row["Submethod"] or "").strip(),
                    "psnr": psnr,
                    "ssim": float(row["SSIM"]) if row["SSIM"] else None,
                    "lpips": float(row["LPIPS"]) if row["LPIPS"] else None,
                    "size_mib": _mb_from_bytes(size),
                }
            )
    survey.sort(key=lambda r: (r["method"], r["size_mib"]))

    # --- PHG aggregate (playroom + drjohnson average) ---------------------
    phg = json.load(open(args.phg))
    scenes = phg["scenes"]
    lams = [0.001, 0.002, 0.004]
    ours = {"baseline": {"psnr": [], "ssim": [], "lpips": [], "size": []},
            "quant": {"psnr": [], "ssim": [], "lpips": [], "size": []}}
    for i, _ in enumerate(lams):
        for variant, key in (("baseline", "baseline"), ("quant", "quant_cd8_rest16")):
            ours[variant]["psnr"].append(
                sum(scenes[s][i][key]["psnr"] for s in scenes) / 2.0
            )
            ours[variant]["ssim"].append(
                sum(scenes[s][i][key]["ssim"] for s in scenes) / 2.0
            )
            ours[variant]["lpips"].append(
                sum(scenes[s][i][key]["lpips"] for s in scenes) / 2.0
            )
            ours[variant]["size"].append(
                sum(scenes[s][i][key]["total_MB"] for s in scenes) / 2.0
            )

    # --- PHG + SPA aggregate (playroom + drjohnson average) ---------------
    spa = {}
    try:
        spa_json = json.load(open(args.phg_spa))
        spa_scenes = spa_json["scenes"]
        spa = {"quant": {"psnr": [], "ssim": [], "lpips": [], "size": []}}
        for i, _ in enumerate(lams):
            vals = []
            for s in spa_scenes:
                rows = spa_scenes[s]
                if i < len(rows):
                    q = rows[i]["quant_cd8_rest16"]
                    if q.get("psnr") is not None:
                        vals.append(q)
            if len(vals) == 2:
                spa["quant"]["psnr"].append(sum(v["psnr"] for v in vals) / 2.0)
                spa["quant"]["ssim"].append(sum(v["ssim"] for v in vals) / 2.0)
                spa["quant"]["lpips"].append(sum(v["lpips"] for v in vals) / 2.0)
                spa["quant"]["size"].append(sum(v["total_MB"] for v in vals) / 2.0)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass

    # --- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    metrics = [
        ("psnr", "PSNR (dB)", PSNR_LIM),
        ("ssim", "SSIM", SSIM_LIM),
        ("lpips", "LPIPS (lower is better)", LPIPS_LIM),
    ]
    for ax in axes:
        ax.set_xlabel("Size (MiB)")
        ax.set_xlim(*X_LIM)
        ax.grid(alpha=0.3, linestyle="--")

    for r in survey:
        label = NAMES[r["method"]]
        if r["sub"]:
            label = f"{label} {r['sub']}"
        off = ANNO_OFFSETS.get((r["method"], r["sub"]), (6, 6))
        for ax, (key, ylab, ylim) in zip(axes, metrics):
            y = r[key]
            if y is None:
                continue
            ax.set_ylim(*ylim)
            ax.scatter(r["size_mib"], y, s=55, marker="o", alpha=0.9,
                       color="tab:gray", edgecolors="black", linewidths=0.4)
            ax.annotate(label, (r["size_mib"], y), textcoords="offset points",
                        xytext=off, fontsize=6.5, color="tab:gray")

    # PHG ours: two lines — with / without SPA (quantized final payload)
    for variant, style, color in (
        ("quant", "o-", "tab:red"),
        ("spa", "s-", "tab:green"),
    ):
        data = spa if variant == "spa" else ours["quant"]
        if not data["size"]:
            continue
        lbl = "PHG + SPA (ratio 0.5, 110k)" if variant == "spa" else "PHG (no SPA, 110k)"
        for ax, (key, ylab, ylim) in zip(axes, metrics):
            ax.set_ylim(*ylim)
            ax.plot(data["size"], data[key], style, lw=2.2, ms=8, color=color,
                    label=lbl, zorder=5)
            for x, y, lam in zip(data["size"], data[key], lams[: len(data["size"])]):
                ax.annotate(
                    f"λ{lam:.3f}".rstrip("0"),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10) if variant == "quant" else (0, -14),
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

    axes[0].set_title("DB RD: PSNR vs Size")
    axes[1].set_title("DB RD: SSIM vs Size")
    axes[2].set_title("DB RD: LPIPS vs Size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Deep Blending (playroom + drjohnson average) — survey: "
        "w-m.github.io/3dgs-compression-survey (curated top-left methods)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot_db_rd] wrote {args.out}")


if __name__ == "__main__":
    main()
