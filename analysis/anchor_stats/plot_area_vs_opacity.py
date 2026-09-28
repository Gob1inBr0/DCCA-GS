"""Plot per-anchor area vs opacity from anchor_stats.npz (ph0c base run)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "ph0c_base_lam0004_s104")
z = np.load(os.path.join(D, "anchor_stats.npz"))

area = z["area"].astype(np.float64)
op_max = z["op_max"]
op_mean = z["op_mean"]
seen = z["seen"]

vis = np.nonzero(seen)[0]
a, om, omean = area[vis], op_max[vis], op_mean[vis]

# Strict check: does the conclusion hold with mean opacity (not per-view max)?
for name, o in (("op_max", om), ("op_mean", omean)):
    print(
        f"{name}: median={np.median(o):.3f}  frac<0.3={(o < 0.3).mean():.4f}  "
        f"frac>=0.9={(o >= 0.9).mean():.4f}"
    )
top = a >= np.quantile(a, 0.99)
print(
    f"top-1% area anchors (area>={np.quantile(a, 0.99):.2e}): "
    f"n={top.sum()}  op_mean median={np.median(omean[top]):.3f}  "
    f"frac op_mean<0.3={(omean[top] < 0.3).mean():.4f}"
)

fig, axes = plt.subplots(
    1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]}
)
ax = axes[0]
lg = np.log10(np.maximum(a, 1e-8))
h = ax.hexbin(
    lg, om, gridsize=(70, 28), mincnt=1, bins="log",
    cmap="viridis", extent=(-8, np.log10(a.max()), 0, 1),
)
fig.colorbar(h, ax=ax, label="count (log)")
# median op_max per decile of nonzero-area anchors
nz = a > 0
qs = np.quantile(a[nz], np.linspace(0, 1, 11))
mids, meds = [], []
for b in range(10):
    m = (a[nz] >= qs[b]) & ((a[nz] <= qs[b + 1]) if b == 9 else (a[nz] < qs[b + 1]))
    if m.sum():
        mids.append(np.sqrt(qs[b] * qs[b + 1]))
        meds.append(np.median(om[nz][m]))
ax.plot(
    np.log10(mids), meds, "r-", lw=2, marker="o", ms=4,
    label="median op (area deciles)",
)
ax.set_xlabel("contribution area (frame fraction, log10; −8 ≈ zero)")
ax.set_ylabel("anchor opacity  (max over 16 views, 10 offsets)")
ax.set_title("679k-anchor model, 528k visible: opacity vs coverage area")
ax.legend(loc="lower left", fontsize=9)
ax.axvline(-8, color="gray", lw=0.6, ls=":")

ax = axes[1]
dec_labels, dec_data = [], []
for b in range(10):
    m = (a[nz] >= qs[b]) & ((a[nz] <= qs[b + 1]) if b == 9 else (a[nz] < qs[b + 1]))
    dec_data.append(om[nz][m])
    dec_labels.append(f"D{b + 1}")
bp = ax.boxplot(
    dec_data, showfliers=False, tick_labels=dec_labels, patch_artist=True,
    medianprops={"color": "k"},
)
for patch in bp["boxes"]:
    patch.set_facecolor("#7fbf7b")
ax.axhline(0.3, color="crimson", ls="--", lw=1, label="0.3")
ax.set_ylim(0, 1.02)
ax.set_xlabel("contribution-area decile (D1 smallest → D10 largest)")
ax.set_ylabel("anchor opacity (max)")
ax.set_title("opacity is flat across area deciles")
ax.legend(fontsize=9)

fig.tight_layout()
out = os.path.join(D, "fig_area_vs_opacity.png")
fig.savefig(out, dpi=150)
print("saved", out)
