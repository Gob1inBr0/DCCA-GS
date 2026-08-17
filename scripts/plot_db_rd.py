"""Plot Deep Blending RD curves: survey methods vs PHG lambda points.

Survey data: docs/data/DeepBlending_survey.csv (aggregate from
https://w-m.github.io/3dgs-compression-survey/, average of playroom+drjohnson).
PHG data: runs/db_rd_110k.json (per-scene lambda-RD; this script averages the
two scenes to match the survey aggregate basis).

Usage (5090):
    PYTHONPATH=$PWD python scripts/plot_db_rd.py \
      --survey docs/data/DeepBlending_survey.csv \
      --phg runs/db_rd_110k.json \
      --out runs/db_rd_curve.png
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--survey", default="docs/data/DeepBlending_survey.csv")
    p.add_argument("--phg", default="runs/db_rd_110k.json")
    p.add_argument("--out", default="runs/db_rd_curve.png")
    args = p.parse_args()

    # --- survey rows (aggregate = playroom + drjohnson average) ----------
    survey = []
    with open(args.survey, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                size = float(row["Size [Bytes]"])
                psnr = float(row["PSNR"])
            except (ValueError, TypeError):
                continue
            ssim = float(row["SSIM"]) if row["SSIM"] else None
            lpips = float(row["LPIPS"]) if row["LPIPS"] else None
            survey.append(
                {
                    "method": row["Method"],
                    "sub": (row["Submethod"] or "").strip(),
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "size_mib": _mb_from_bytes(size),
                }
            )

    # curated, readable subset
    include = {
        "chen2024hac",
        "chen2025hac-plus",
        "chen2025fcgs",
        "liu2024compgs",
        "liu2024hemgs",
        "lu2024scaffold",
        "wang2024contextgs",
        "wang2026smolgs",
        "zhang2024gaussianspa",
        "lee2024compact",
        "navaneet2023compact3d",
        "papantonakis2024reducing",
        "niedermayr2024compressed",
        "girish2024eagles",
        "kerbl3Dgaussians",
    }
    names = {
        "chen2024hac": "HAC",
        "chen2025hac-plus": "HAC++",
        "chen2025fcgs": "FCGS",
        "liu2024compgs": "CompGS",
        "liu2024hemgs": "HEMGS",
        "lu2024scaffold": "Scaffold-GS",
        "wang2024contextgs": "ContextGS",
        "wang2026smolgs": "Smol-GS",
        "zhang2024gaussianspa": "GaussianSpa",
        "lee2024compact": "Compact3D",
        "navaneet2023compact3d": "Compact3D*",
        "papantonakis2024reducing": "Reducing",
        "niedermayr2024compressed": "Compressed",
        "girish2024eagles": "EAGLES",
        "kerbl3Dgaussians": "3DGS",
    }
    survey = [r for r in survey if r["method"] in include]

    # --- PHG lambda points (average of playroom + drjohnson) ------------
    phg = json.load(open(args.phg))
    scenes = phg["scenes"]

    lams = [0.001, 0.002, 0.004]
    ours = {"baseline": {"psnr": [], "ssim": [], "lpips": [], "size": []},
            "quant": {"psnr": [], "ssim": [], "lpips": [], "size": []}}
    for i, lam in enumerate(lams):
        for variant in ("baseline", "quant"):
            key = "quant_cd8_rest16" if variant == "quant" else "baseline"
            ps = sum(scenes[s][i][key]["psnr"] for s in scenes) / 2.0
            ss = sum(scenes[s][i][key]["ssim"] for s in scenes) / 2.0
            lp = sum(scenes[s][i][key]["lpips"] for s in scenes) / 2.0
            mb = sum(scenes[s][i][key]["total_MB"] for s in scenes) / 2.0
            ours[variant]["psnr"].append(ps)
            ours[variant]["ssim"].append(ss)
            ours[variant]["lpips"].append(lp)
            ours[variant]["size"].append(mb)

    # --- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    metrics = [
        ("psnr", "PSNR (dB)", axes[0], True),
        ("ssim", "SSIM", axes[1], True),
        ("lpips", "LPIPS (lower is better)", axes[2], False),
    ]

    for ax in axes:
        ax.set_xlabel("Size (MiB)")
        ax.grid(alpha=0.3, linestyle="--")

    plotted = set()
    for r in survey:
        label = names[r["method"]]
        if r["sub"]:
            label = f"{label} {r['sub']}"
        for (key, ylab, ax, higher), sub in zip(metrics, (r["sub"], r["sub"], r["sub"])):
            y = r[key]
            if y is None:
                continue
            mk = "o"
            if (label, sub) in plotted:
                mk = "."
            ax.scatter(
                r["size_mib"], y, s=42, marker=mk, alpha=0.85,
                label=None if (label, sub) in plotted else label,
            )
            plotted.add((label, sub))

    # PHG ours: line through lambda points
    for variant, style in (("baseline", "o-"), ("quant", "s--")):
        lbl = "PHG (ours, 110k)" if variant == "baseline" else "PHG (ours) + MLP quant"
        for key, _, ax, _ in metrics:
            ax.plot(ours[variant]["size"], ours[variant][key], style, lw=2, ms=7, label=lbl)

    axes[0].set_title("DB RD: PSNR vs Size")
    axes[1].set_title("DB RD: SSIM vs Size")
    axes[2].set_title("DB RD: LPIPS vs Size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Deep Blending (playroom + drjohnson average) — survey: "
        "w-m.github.io/3dgs-compression-survey",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot_db_rd] wrote {args.out}")


if __name__ == "__main__":
    main()
