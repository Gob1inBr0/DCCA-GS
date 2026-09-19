"""Recover per-anchor quantization step via divisibility (decoded = k*Q),
then symbols -> entropy fits."""
import torch, numpy as np, re, json

run = "/mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42"
dec = torch.load(run + "/bitstreams/attributes.pth", map_location="cpu", weights_only=False)["scaling"].numpy().astype(np.float64)
N, D = dec.shape

qs = np.zeros(N)
ok = np.zeros(N, bool)
for i in range(N):
    v = np.abs(dec[i])
    v = v[v > 0]
    if v.size == 0:
        continue
    q = v.min()
    for _ in range(24):
        r = v / q
        if np.all(np.abs(r - np.round(r)) < 1e-3 * np.maximum(1.0, r)):
            qs[i], ok[i] = q, True
            break
        q /= 2.0

print("Q recovered for %.1f%% of anchors; median Q = %.6f (p10 %.6f p90 %.6f)" % (
    ok.mean() * 100, np.median(qs[ok]), *np.percentile(qs[ok], [10, 90])))

sel = ok
sym = np.round(dec[sel] / qs[sel][:, None]).astype(np.int64)
sym = sym - np.median(sym)
line = [l for l in open(run + "/compress.log") if "bit_feat" in l][-1]
bit_scaling = int(re.search(r"'bit_scaling': (\d+)", line).group(1))
n_all = N * D
cur_bps = bit_scaling / n_all

vals, counts = np.unique(sym, return_counts=True)
p = counts / counts.sum()
H_emp = -(p * np.log2(p)).sum()

x = sym.astype(np.float64).ravel()
rng = np.random.default_rng(0)
sub = rng.choice(x, size=min(300000, x.size), replace=False)

def laplace_ce(x, mu, b):
    return float((np.abs(x - mu) / b + np.log(2 * b)).mean() / np.log(2))

ce_lap = laplace_ce(sub, float(np.median(sub)), float(np.mean(np.abs(sub - np.median(sub)))))

def gmm_fit(x, K, iters=60):
    mu = np.quantile(x, (np.arange(K) + 0.5) / K)
    sd = np.full(K, max(x.std() / K, 1e-2))
    w = np.full(K, 1.0 / K)
    for _ in range(iters):
        lp = np.log(w[None]) - np.log(sd[None]) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x[:, None] - mu[None]) / sd[None]) ** 2
        m = lp.max(1, keepdims=True)
        r = np.exp(lp - m); r /= r.sum(1, keepdims=True)
        Nk = r.sum(0) + 1e-9
        mu = (r * x[:, None]).sum(0) / Nk
        sd = np.sqrt((r * (x[:, None] - mu[None]) ** 2).sum(0) / Nk + 1e-6)
        w = Nk / Nk.sum()
    ll = (np.log(w[None]) - np.log(sd[None]) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x[:, None] - mu[None]) / sd[None]) ** 2).sum(1)
    return -ll.mean() / np.log(2)

out = {"recovered_pct": round(ok.mean() * 100, 1),
       "Q_median": float(np.median(qs[ok])),
       "symbol_p99_range": int(np.percentile(np.abs(sym), 99)),
       "current_bits_per_symbol": round(cur_bps, 3),
       "empirical_iid_entropy": round(H_emp, 3),
       "laplace_ce": round(ce_lap, 3)}
for K in (2, 3, 4):
    out[f"gmm{K}_ce"] = round(gmm_fit(sub, K), 3)
out["saving_laplace_pct"] = round((cur_bps - ce_lap) / cur_bps * 100, 2)
out["saving_gmm3_pct"] = round((cur_bps - out["gmm3_ce"]) / cur_bps * 100, 2)
out["saving_iid_bound_pct"] = round((cur_bps - H_emp) / cur_bps * 100, 2)
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/scaling_entropy2.json", "w"), indent=1)
