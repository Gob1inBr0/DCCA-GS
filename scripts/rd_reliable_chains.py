import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _f(x):
    try: return float(x)
    except Exception: return None

rows = {}
for r in csv.DictReader(open("docs/data/experiments.csv")):
    p, m = _f(r["psnr"]), _f(r["total_mb"])
    cur = rows.get(r["run_id"])
    if cur is None or (cur[0] is None and p is not None):
        rows[r["run_id"]] = (p, m)
def get(tag): return rows.get(tag, (None, None))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
base = [(11.72, 27.511), (15.05, 27.750), (23.60, 27.953)]
fus = [(11.987, 27.3081), (15.377, 27.8517), (23.593, 28.0347)]
ax = axes[0]
for pts, name, mk in [(base, "base (legacy)", "o"), (fus, "base + fusion pruning", "s")]:
    p = sorted(pts)
    ax.plot([x[0] for x in p], [x[1] for x in p], marker=mk, label=name)
ax.set_title("Chain A: fusion increment (1-78, 30k, legacy protocol)")
ax.set_xlabel("size (MB)"); ax.set_ylabel("PSNR (dB)"); ax.legend(); ax.grid(alpha=.3)

chains = {
    "seed42": {"base": ["ph0c_base_r085_lam0004_s42", "ph0c_base_r085_lam0002_s42", "ph0c_base_r085_lam0005_s42"],
               "sub":  ["ph1v3_A2_cover_r085_lam0004_s42", "ph1v3_A2_cover_r085_lam0002_s42", "ph1v3_A2_cover_r085_lam0005_s42"]},
    "seed104": {"base": ["ph0c_base_r085_lam0004_s104", "ph0c_base_r085_lam0005_s104"],
                "sub":  ["ph1v3_A2_cover_r085_lam0004_s104", "ph1v3_A2_cover_r085_lam0005_s104"]},
}
ax = axes[1]
styles = {("seed42", "base"): ("o-", "base s42"), ("seed42", "sub"): ("o--", "submodular s42"),
          ("seed104", "base"): ("^-", "base s104"), ("seed104", "sub"): ("^--", "submodular s104")}
for seed, cc in chains.items():
    for kind, tags in cc.items():
        pts = sorted([get(t) for t in tags if get(t)[0] is not None])
        ls, lab = styles[(seed, kind)]
        if pts:
            ax.plot([p[1] for p in pts], [p[0] for p in pts], ls, marker=ls[0], label=lab, alpha=.85)
ax.set_title("Chain B: submodular vs base (1-78, 30k, r_post=0.85, minifull code)")
ax.set_xlabel("size (MB)"); ax.set_ylabel("PSNR (dB)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("docs/figures/rd_reliable_chains.png", dpi=150)
print("saved")
for seed, cc in chains.items():
    for kind, tags in cc.items():
        print(seed, kind, [(t.split("_r085")[0], get(t)) for t in tags])
