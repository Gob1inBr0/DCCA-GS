"""S1 offline checks for the progressive-stream direction.
Data: ph0c_base_r085_lam0005_s42 (+ r2_base_newcode lam0002 for C2/D1).
Surrogate: coverage graph from spatial cells (not render edges); S2 renders validate."""
import numpy as np, torch, json, re, heapq, time
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

def entropy_of(arr):
    _, c = np.unique(arr, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

base = "/home/project2/DCCA-GS-git/analysis/anchor_stats/ph0c_base_r085_lam0005_s42"
d = np.load(base + "/byte_account/per_anchor_bits.npz")
st = np.load(base + "/anchor_stats.npz")
ck = torch.load("/mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42/ckpts/ckpt_30000.pth", map_location="cpu", weights_only=False)
dec = torch.load("/mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42/bitstreams/attributes.pth", map_location="cpu", weights_only=False)
pos = ck["model_state"]["_anchor"].numpy().astype(np.float64)
offs = dec["offset"].numpy().astype(np.float64)
sc = dec["scaling"].numpy().astype(np.float64)
mk = dec["mask"].numpy().astype(np.float64)
N = pos.shape[0]
area = st["area"].astype(np.float64)
bits = d["feat"] + d["scaling"] + d["offsets"] + d["masks"]
B_tot, A_tot = bits.sum(), area.sum()
out = {}

# ---- A1 metadata ----
a1 = {}
for G in (8, 16, 32, 64, 128):
    bits_meta = N * int(np.ceil(np.log2(G))) + G * 5 * 16 * 8
    a1[G] = round(bits_meta / 8e6, 3)
out["A1_meta_MB(pct)"] = {g: f"{v}MB ({v/(B_tot/8e6)*100:.2f}%)" for g, v in a1.items()}

# ---- cells ----
sub = pos[np.random.default_rng(0).choice(N, 4000, replace=False)]
tree = cKDTree(pos)
dist, _ = tree.query(sub, k=2)
nn_med = float(np.median(dist[:, 1]))
cell = nn_med * 2.0
lo = pos.min(0)
cid = np.floor((pos - lo) / cell).astype(np.int64)
dims = np.maximum(cid.max(0) + 1, 1)
key = (cid[:, 0] * dims[1] + cid[:, 1]) * dims[2] + cid[:, 2]
uc, cell_of = np.unique(key, return_inverse=True)
occ = pos + offs.mean(1) * (nn_med / max(float(np.median(np.abs(offs))), 1e-12))
cid2 = np.clip(np.floor((occ - lo) / cell).astype(np.int64), 0, dims - 1)
key2 = (cid2[:, 0] * dims[1] + cid2[:, 1]) * dims[2] + cid2[:, 2]
_, allk = np.unique(np.concatenate([key, key2]), return_inverse=True)
cell_of = allk[:N]          # own cell in COMBINED id space
cell2_of = allk[N:]         # offset cell in COMBINED id space
C = int(allk.max()) + 1
mass = np.zeros(C); np.add.at(mass, cell_of, area)
out["cells"] = {"n_cells": int(np.unique(cell_of).size), "cell_size": round(cell, 4)}

def curve_additive(idx):
    cb = np.cumsum(bits[idx]) / B_tot
    y = np.cumsum(area[idx]) / A_tot
    return cb, y

def curve_dedup(idx):
    """redundancy-aware: fraction of cell MASS whose cell first touched."""
    cb = np.cumsum(bits[idx]) / B_tot
    c = cell_of[idx]
    _, first = np.unique(c, return_index=True)
    gain = np.zeros(len(idx))
    gain[first] = mass[c[first]]
    y = np.cumsum(gain) / mass.sum()
    return cb, y

def auc(cb, y):
    return float(np.trapz(y, cb))

qs = [0.05, 0.10, 0.25, 0.50]
def at_points(cb, y):
    return {f"{int(q*100)}%": round(float(np.interp(q, cb, y)) * 100, 1) for q in qs}

rng = np.random.default_rng(0)
contrib = np.argsort(-area, kind="stable")
orders = {"contribution": contrib, "random": rng.permutation(N)}
out["eval"] = {}
for name, idx in orders.items():
    cb, y = curve_additive(idx); cb2, y2 = curve_dedup(idx)
    out["eval"][name] = {"additive": at_points(cb, y), "dedup": at_points(cb2, y2),
                          "auc_add": round(auc(cb, y), 4), "auc_dedup": round(auc(cb2, y2), 4)}

# ---- A3 greedy on cell-coverage surrogate ----
edges = np.concatenate([np.stack([np.arange(N), cell_of], 1), np.stack([np.arange(N), cell2_of], 1)])
edges = np.unique(edges, axis=0)
ua, uc2 = edges[:, 0], edges[:, 1]
by_a = {}
for a, c in zip(ua.tolist(), uc2.tolist()):
    by_a.setdefault(a, []).append(c)
covered = np.zeros(C, bool)
picked = np.zeros(N, bool)
gains0 = np.zeros(N)
np.add.at(gains0, ua, mass[uc2])
heap = [(-float(g), int(a), 0) for a, g in enumerate(gains0)]
heapq.heapify(heap)
stamp = np.zeros(N, int); cnt = 0
t0 = time.time()
greedy = []
while heap and len(greedy) < N:
    ng, a, stv = heapq.heappop(heap)
    if stv < stamp[a] or picked[a]:
        continue
    g = 0.0
    for c in by_a.get(a, ()):
        if not covered[c]:
            g += float(mass[c])
    if g <= 0:
        continue
    if -ng - g > 1e-9:
        cnt += 1; stamp[a] = cnt; heapq.heappush(heap, (-g, a, cnt)); continue
    picked[a] = True
    greedy.append(a)
    for c in by_a[a]:
        covered[c] = True
    if len(greedy) % 200000 == 0:
        print("greedy", len(greedy), f"{time.time()-t0:.0f}s", flush=True)
    if covered.all():
        break
greedy.extend(contrib[~picked[contrib]].tolist())   # complete the order regardless
gidx = np.array(greedy, dtype=np.int64)
cb, y = curve_additive(gidx); cb2, y2 = curve_dedup(gidx)
out["eval"]["greedy_surrogate"] = {"additive": at_points(cb, y), "dedup": at_points(cb2, y2),
                                    "auc_add": round(auc(cb, y), 4), "auc_dedup": round(auc(cb2, y2), 4)}

# ---- A2 decoder-side stats predict contribution ----
_cnt_cell = np.bincount(cell_of)
dens = _cnt_cell[cell_of]  # anchors sharing the anchor's own cell
aniso = sc[:, :3].max(1) / np.maximum(sc[:, :3].min(1), 1e-9)
off_e = np.linalg.norm(offs, axis=2).mean(1)
maskr = mk.reshape(N, -1).mean(1)
feats = {"density": dens, "scale_mean": sc.mean(1), "anisotropy": aniso, "offset_energy": off_e, "mask_ratio": maskr}
a2 = {k: round(float(spearmanr(v, area).statistic), 3) for k, v in feats.items()}
ranks = sum(np.argsort(np.argsort(v)) for v in feats.values())
pidx = np.argsort(-np.asarray(ranks, float), kind="stable")
cbp, yp = curve_dedup(pidx)
out["A2"] = {"spearman_vs_area": a2, "auc_pred_dedup": round(auc(cbp, yp), 4),
             "auc_true_dedup": out["eval"]["contribution"]["auc_dedup"],
             "pct_of_true": round(auc(cbp, yp) / out["eval"]["contribution"]["auc_dedup"] * 100, 1)}

# ---- C2/D1 on scaling symbols ----
def recover_Q(dsc):
    n = dsc.shape[0]
    qs_ = np.zeros(n); ok = np.zeros(n, bool)
    arr = np.abs(dsc)
    for i in range(n):
        v = arr[i][arr[i] > 0]
        if v.size == 0:
            continue
        q = v.min()
        for _ in range(24):
            r = v / q
            if np.all(np.abs(r - np.round(r)) < 1e-3 * np.maximum(1.0, r)):
                qs_[i], ok[i] = q, True
                break
            q /= 2.0
    return qs_, ok

res = {}
for tag, run in [("lam0005", "/mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42"),
                 ("lam0002", "/mnt/newproject2/dcca_runs/r2_base_newcode_r085_lam0002_s42")]:
    dsc = torch.load(run + "/bitstreams/attributes.pth", map_location="cpu", weights_only=False)["scaling"].numpy().astype(np.float64)
    qs_, ok = recover_Q(dsc)
    sym = np.round(dsc[ok] / qs_[ok][:, None]).astype(np.int64)
    sym -= int(np.median(sym))
    line = [l for l in open(run + "/compress.log") if "bit_feat" in l][-1]
    bit_s = int(re.search(r"'bit_scaling': (\d+)", line).group(1))
    r = res.setdefault(tag, {"current_bps": round(bit_s / sym.size, 3), "recovered_pct": round(ok.mean() * 100, 1)})
    for b in (2, 3):
        coarse = sym >> b; fine = sym & ((1 << b) - 1)
        Hf = entropy_of(fine.ravel()); Hj = entropy_of((coarse * (1 << b) + fine).ravel()); Hc = entropy_of(coarse.ravel())
        r[f"C2_b{b}"] = {"H_fine": round(Hf, 3), "H_fine|coarse": round(Hj - Hc, 3), "gain": round(Hf - (Hj - Hc), 3)}
    # greedy order only valid for the lam0005 run it was computed on
    flat = sym[gidx].ravel() if (tag == "lam0005" and ok.all() and gidx.max() < sym.shape[0]) else sym.ravel()
    for G in (16, 32, 64):
        segs = np.array_split(flat, G)
        cost = sum(s.size * entropy_of(s) for s in segs)
        r[f"D1_G{G}_loss_bps"] = round((cost - flat.size * entropy_of(flat)) / flat.size, 4)
out["C2_D1"] = res
json.dump(out, open("/tmp/s1_progressive.json", "w"), indent=1)
print(json.dumps(out, indent=1))
