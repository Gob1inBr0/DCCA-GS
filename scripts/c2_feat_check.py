"""C2 upper bound on the feat field (32 syms/anchor, scalar Q per anchor).
Same layered-conditional gate as scaling: H(fine) - H(fine|coarse)."""
import torch, numpy as np, json, re

def entropy_of(arr):
    _, c = np.unique(arr, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def recover_Q_scalar(vec):
    """vec: [D] decoded values of one anchor sharing one step. GCD-style."""
    v = np.abs(vec)
    v = v[v > 0]
    if v.size == 0:
        return None, False
    q = v.min()
    for _ in range(40):
        r = v / q
        if np.all(np.abs(r - np.round(r)) < 2e-3 * np.maximum(1.0, r)):
            return q, True
        q /= 2.0
    return q, False

res = {}
for tag, run in [("lam0005", "/mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42"),
                 ("lam0002", "/mnt/newproject2/dcca_runs/r2_base_newcode_r085_lam0002_s42")]:
    dec = torch.load(run + "/bitstreams/attributes.pth", map_location="cpu", weights_only=False)["anchor_feat"].numpy().astype(np.float64)
    n, D = dec.shape
    qs_ = np.zeros(n); ok = np.zeros(n, bool)
    step = max(1, n // 4000)
    idx = np.arange(0, n, step)  # sample anchors for Q recovery speed
    for i in idx:
        q, o = recover_Q_scalar(dec[i])
        qs_[i], ok[i] = (q or 0), o
    # interpolate recovered Q to all anchors via nearest sampled (steps vary smoothly with content-aware multiplier)
    ok_full = np.zeros(n, bool); ok_full[idx] = ok[idx]
    qs_full = np.interp(np.arange(n), idx[ok[idx]], qs_[idx][ok[idx]])
    sym_all = np.round(dec / qs_full[:, None]).astype(np.int64)
    # validate on sampled anchors: all decoded must be integer multiples
    valid = ok_full.sum() > 100
    r = res.setdefault(tag, {})
    r["recovered_pct_sampled"] = round(ok.mean() * 100, 1)
    r["Q_median"] = float(np.median(qs_[idx][ok[idx]])) if ok[idx].any() else None
    sym = sym_all[ok_full] if valid else sym_all
    sym = sym - int(np.median(sym))
    line = [l for l in open(run + "/compress.log") if "bit_feat" in l][-1]
    bit_f = int(re.search(r"'bit_feat': (\d+)", line).group(1))
    r["current_bps_feat"] = round(bit_f / (n * D), 3)
    for b in (3, 5):
        coarse = sym >> b; fine = sym & ((1 << b) - 1)
        Hf = entropy_of(fine.ravel()); Hj = entropy_of((coarse * (1 << b) + fine).ravel()); Hc = entropy_of(coarse.ravel())
        r[f"C2_b{b}"] = {"H_fine": round(Hf, 3), "H_fine|coarse": round(Hj - Hc, 3), "gain": round(Hf - (Hj - Hc), 3)}
    # prize pool: MB saved on feat if gain applied
    b = 5
    r[f"feat_MB_saved_b{b}"] = round(r[f"C2_b{b}"]["gain"] * sym.size / 8e6, 3)
    print(tag, json.dumps(r), flush=True)
json.dump(res, open("/tmp/c2_feat.json", "w"), indent=1)
