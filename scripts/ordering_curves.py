"""Direction-3 offline step: bytes-vs-coverage curves for three anchor orderings.
Model: per-field streams reordered into per-anchor groups (re-encode variant);
x = cumulative attribute bytes fraction, y = cumulative area coverage fraction."""
import numpy as np, torch, json

base = "/home/project2/DCCA-GS-git/analysis/anchor_stats/ph0c_base_r085_lam0005_s42"
d = np.load(base + "/byte_account/per_anchor_bits.npz")
st = np.load(base + "/anchor_stats.npz")
ck = torch.load("/mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42/ckpts/ckpt_30000.pth", map_location="cpu", weights_only=False)
pos = ck["model_state"]["_anchor"].numpy()
N = pos.shape[0]
assert N == d["seen"].shape[0] == st["area"].shape[0]

bits = d["feat"] + d["scaling"] + d["offsets"] + d["masks"]
area = st["area"].astype(np.float64)
B_tot, A_tot = bits.sum(), area.sum()

def morton_codes(p, bits_per_axis=10):
    lo, hi = p.min(0), p.max(0)
    span = np.maximum(hi - lo, 1e-12)
    q = ((p - lo) / span * ((1 << bits_per_axis) - 1)).round().astype(np.uint64)
    m = np.zeros(len(p), dtype=np.uint64)
    for b in range(bits_per_axis):
        for ax in range(3):
            m |= ((q[:, ax] >> b) & 1).astype(np.uint64) << (3 * b + ax)
    return m

rng = np.random.default_rng(0)
orderings = {
    "morton": np.argsort(morton_codes(pos), kind="stable"),
    "random": rng.permutation(N),
    "area_desc": np.argsort(-area, kind="stable"),
}

def curve(idx):
    cb = np.cumsum(bits[idx]) / B_tot
    ca = np.cumsum(area[idx]) / A_tot
    return cb, ca

print("N=%d total_attr_MB=%.2f" % (N, B_tot / 8e6))
qs = [0.05, 0.10, 0.25, 0.50, 0.75]
print("bytes%% -> area coverage%%   (and bytes%% needed for 80%% area)")
res = {}
for name, idx in orderings.items():
    cb, ca = curve(idx)
    at = np.interp(qs, cb, ca) * 100
    need80 = np.interp(0.80, ca, cb) * 100
    res[name] = {"coverage_at": dict(zip(map(str, qs), at.round(1).tolist())), "bytes_for_80pct_area": round(need80, 1)}
    print("%-10s: %s  bytes_for_80%%area=%.1f%%" % (name, " ".join("%d%%->%.1f%%" % (q * 100, a) for q, a in zip(qs, at)), need80))
json.dump(res, open("/tmp/ordering_curves.json", "w"), indent=1)
print("saved /tmp/ordering_curves.json")
