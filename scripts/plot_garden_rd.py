#!/usr/bin/env python3
"""Garden RD figure: DCCA progressive vs DCCA production vs HAC++.

Inputs:
  analysis/garden_rd/std_garden_l000{2,4}_s42 layered_curve_150v.json
    (written by scripts/full_view_eval.py: per-prefix real_bytes + psnr_mean
     on the run's val views, plus production_payload_bytes)
  docs/data/benchmark_hac_per_scene.csv  (official HAC++ garden, factor2-30k)

Output: analysis/garden_rd/garden_rd_curves.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "garden_rd"
OUT = DATA / "garden_rd_curves.png"
RUNS = [
    ("l0002", r"$\lambda$=0.002", "tab:blue"),
    ("l0004", r"$\lambda$=0.004", "tab:orange"),
]
V1_RUNS = [
    ("rate", "v1 rate-aware ($\\lambda$=0.002)", "tab:green"),
    ("p0", "v1 P0 render loss ($\\lambda$=0.002)", "tab:red"),
]

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)

for tag, lam, color in RUNS:
    d = json.loads((DATA / f"{tag}_layered_150v.json").read_text())
    res = sorted(d["results"], key=lambda r: r["prefix"])
    xs = [r["real_bytes"] / 1048576 for r in res]
    ys = [r["psnr_mean"] for r in res]
    ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.4, label=f"DCCA progressive {lam}")
    for r, x, y in zip(res, xs, ys):
        ax.annotate(f"P{r['prefix']}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=color)
    prod_mb = d["production_payload_bytes"] / 1048576
    prod_psnr = res[-1]["psnr_mean"]
    ax.plot([prod_mb], [prod_psnr], "D", color=color, ms=8, mfc="none", mew=1.6,
            label=f"DCCA production {lam}")

hac = [(float(r["size_mb"]), float(r["psnr"]))
       for r in csv.DictReader(open(HERE.parent / "docs/data/benchmark_hac_per_scene.csv"))
       if r["scene"] == "garden"]
hac.sort()
ax.plot([x for x, _ in hac], [y for _, y in hac], "s--", color="k", ms=7,
        label="HAC++ (official, factor2-30k)")

for tag, label, color in V1_RUNS:
    f = DATA / f"{tag}_layered_150v.json"
    if not f.exists():
        print(f"[skip] {f.name} not present yet")
        continue
    d = json.loads(f.read_text())
    res = sorted(d["results"], key=lambda r: r["prefix"])
    xs = [r["real_bytes"] / 1048576 for r in res]
    ys = [r["psnr_mean"] for r in res]
    ax.plot(xs, ys, "--^", color=color, ms=4, lw=1.4, label=f"{label} progressive")
    prod_mb = d["production_payload_bytes"] / 1048576
    ax.plot([prod_mb], [ys[-1]], "D", color=color, ms=8, mfc="none", mew=1.6,
            label=f"{label} production")

ax.set_xlabel("Bitstream size (MB)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Mip-NeRF360 garden — DCCA progressive vs production vs HAC++\n"
             "PSNR on val views @ factor2 / max-width 3200")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
for tag, _, _ in RUNS:
    d = json.loads((DATA / f"{tag}_layered_150v.json").read_text())
    print(tag, "n_views:", d.get("n_views"),
          "production_MB:", round(d["production_payload_bytes"] / 1048576, 3))
