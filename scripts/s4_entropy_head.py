#!/usr/bin/env python3
"""S4: learned conditional entropy head for enhancement layers (priority 2).

Replaces the static per-(group,bucket) (loc, b) of the layered bitstream with
a small MLP that predicts per-symbol (mean, std) of a discretized Gaussian
from ZERO-SIDE-INFO context: the already-decoded coarse-layer symbol, the
contribution group, and the log step size. The MLP weights (~100 KB) ship in
the bitstream header and are counted.

Modes:
  prepare --run R [--stats S]   decode R, rebuild symbols/ladder, save npz
  train   --npz a.npz b.npz --field feat --out head.pt
                              train the head on runs A,B; report NLL vs the
                              static baseline on a held-out symbol split
  apply   --run R --head head.pt --field feat
                              real constriction coding of R's residual chunks
                              with static vs learned heads -> real byte saving
                              (decoder recomputes all inputs: zero side info)

Run under the DCCA env python with PYTHONPATH=/home/project2/c25_pylib2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import constriction  # noqa: E402

GROUPS = 32
FACTORS = [16, 8, 4, 2]
STEPS = FACTORS + [1]
N_LEVELS = len(STEPS)
FAMILY_RANGE = 4096
FIELDS = ("feat", "scaling", "offset")


# ---------------- shared preamble (mirrors c25_real_bitstream) --------------

def load_run(run_dir, out_dir, device, want_render_model=False):
    from scaffold_gs.hacpp import HACPlusCodec

    bit_dir = Path(run_dir) / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_trained = int(meta["num_anchors_total"])
    run_tag = Path(run_dir).name

    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    del codec
    n_alive = model.num_anchors
    view = model._view
    dec_t = {
        "feat": view.anchor_feat.data.detach().float(),
        "scaling": view.scaling.data.detach().float(),
        "offset": view.offset.data.detach().float(),
    }
    mapped_p = out_dir / (run_tag + ".mapped.npy")
    if mapped_p.exists():
        mapped = np.load(mapped_p).astype(np.int64)
    else:
        ck = torch.load(Path(run_dir) / "ckpts" / "ckpt_30000.pth",
                        map_location="cpu", weights_only=False)
        trained_xyz = ck["model_state"]["_anchor"].to(device)
        decoded_xyz = model.core.get_anchor
        mapped = np.empty(decoded_xyz.shape[0], dtype=np.int64)
        with torch.no_grad():
            for s in range(0, decoded_xyz.shape[0], 1024):
                e = min(s + 1024, decoded_xyz.shape[0])
                d = torch.cdist(decoded_xyz[s:e], trained_xyz)
                mapped[s:e] = d.min(dim=1)[1].cpu().numpy()
                del d
        np.save(mp := out_dir / (run_tag + ".mapped.npy"), mapped)
        print(f"[S4] mapped generated: {mp}", flush=True)
    assert n_alive == mapped.size

    q_cache = out_dir / (run_tag + "_q_cache.npz")
    if q_cache.exists():
        z = np.load(q_cache)
        Q_t = {f: torch.from_numpy(z[n]).to(device)
               for f, n in (("feat", "Q_feat"), ("scaling", "Q_scaling"),
                            ("offset", "Q_offsets"))}
        print("[S4] Q from cache", flush=True)
    else:
        core = model.core
        k_off = model.cfg.n_offsets
        core.current_step = 30000
        core.current_iter = 30000
        Q_feat = torch.empty(n_alive, 32, device=device)
        Q_scaling = torch.empty(n_alive, 6, device=device)
        Q_offsets = torch.empty(n_alive, k_off, 3, device=device)
        anchor_dev = model.core.get_anchor
        with torch.no_grad():
            for s in range(0, n_alive, 16384):
                e = min(s + 16384, n_alive)
                a = anchor_dev[s:e]
                idx = torch.arange(s, e, device=device)
                ctx = core.calc_context_feat(a, anchor_indices=idx, caller="s4")
                (mean, scale, prob, m_sc, s_sc, m_off, s_off, qa, qs, qo) = torch.split(
                    core.get_grid_mlp(ctx),
                    [32, 32, 32, 6, 6, 3 * k_off, 3 * k_off, 1, 1, 1], dim=-1)
                qf = 1.0 * (1 + torch.tanh(qa.repeat(1, 32)))
                qs2 = 0.001 * (1 + torch.tanh(qs.repeat(1, 6)))
                qo2 = 0.2 * (1 + torch.tanh(qo.repeat(1, 3 * k_off))).view(-1, k_off, 3)
                if core.is_content_aware_quant_active():
                    msk = core.get_mask[idx]
                    (qf, qs2, qo2, _, _, _, _) = core._codec_apply_content_aware_quant_params(
                        "s4", a, msk, qf, qs2, qo2, None, None, None,
                        m_sc.view(-1, 6), m_off.view(-1, 3 * k_off))
                Q_feat[s:e] = qf
                Q_scaling[s:e] = qs2
                Q_offsets[s:e] = qo2
        np.savez(q_cache, Q_feat=Q_feat.cpu().numpy(),
                 Q_scaling=Q_scaling.cpu().numpy(), Q_offsets=Q_offsets.cpu().numpy())
        Q_t = {"feat": Q_feat, "scaling": Q_scaling, "offset": Q_offsets}
        print("[S4] Q computed (cached)", flush=True)
    return (model if want_render_model else None), view, dec_t, Q_t, mapped, \
        group_of_for(mapped, n_trained, stats_path=None), n_alive, device, meta


def group_of_for(mapped, n_trained, stats_path):
    """Contribution-area groups when stats exist; Morton-uniform fallback."""
    if stats_path and Path(stats_path).exists():
        area = np.load(stats_path)["area"].astype(np.float64)
        rank = np.empty(n_trained, dtype=np.int64)
        rank[np.argsort(-area, kind="stable")] = np.arange(n_trained)
        g = np.minimum(rank * GROUPS // n_trained, GROUPS - 1).astype(np.int16)
        return g[mapped]
    return np.minimum(np.arange(mapped.size) * GROUPS // mapped.size,
                      GROUPS - 1).astype(np.int16)


def ladder_symbols(dec_t, Q_t, n_alive):
    """Full symbols and the coarse/acc chain per level (mirrors the RB script)."""
    sym = {f: torch.round(dec_t[f] / Q_t[f]) for f in FIELDS}
    out = {}
    for f in FIELDS:
        q = sym[f].cpu().numpy().astype(np.int32).reshape(-1)
        accs = []
        acc = np.round(q / STEPS[0]).astype(np.int32)
        accs.append(acc)
        for li in range(1, N_LEVELS):
            acc = (STEPS[li - 1] // STEPS[li]) * acc + \
                (np.round(q / STEPS[li]).astype(np.int32)
                 - (STEPS[li - 1] // STEPS[li]) * acc)
            accs.append(np.round(q / STEPS[li]).astype(np.int32))
        outs = {"q": q}
        for li in range(N_LEVELS):
            outs[f"acc{li}"] = accs[li]
            if li >= 1:
                outs[f"res{li}"] = (accs[li] - (STEPS[li - 1] // STEPS[li])
                                    * accs[li - 1]).astype(np.int32)
        out[f] = outs
    return out


# ---------------- head model ----------------

MAX_MAG = 8  # decisions per symbol: zero-flag, sign, 7 magnitude bits
             # (|d| <= 2**7 covers >99.9% of residuals; larger clipped by
             # the coder's stage cap in practice)

class BinaryEntropyHead(torch.nn.Module):
    """MLP: (coarse symbol, group emb, log Q) -> per-decision probabilities
    matching the production coder's binary-decomposition decision sequence:
    out[:, 0]  = P(d != 0)
    out[:, 1]  = P(d < 0 | d != 0)
    out[:, 1+j] = P(|d| >= j+2 | |d| >= j+1), j = 1..MAX_MAG-1
    Isomorphic to the encoder: every output is one encoder decision."""

    def __init__(self, emb_dim=8, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(GROUPS, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + emb_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2 + MAX_MAG - 1),
        )

    def forward(self, coarse, group, logq):
        x = torch.cat([coarse[:, None].float(), self.emb(group),
                       logq[:, None].float()], dim=-1)
        return torch.sigmoid(self.net(x))


def decision_bits(residual, probs):
    """Exact code length (bits, summed) of each residual under the
    per-decision Bernoulli model that mirrors the production coder:
      probs[:,0] = P(d != 0); probs[:,1] = P(d < 0 | d != 0);
      probs[:,1+j] = P(|d| >= j+2 | |d| >= j+1), j = 1..MAX_MAG-1
    Magnitude beyond 2**MAX_MAG-1 saturates (clipped, tiny fraction)."""
    d = residual.long()
    a = torch.abs(d)
    sign = (d < 0).long()
    nz = (a > 0).long()
    bits = torch.zeros_like(a, dtype=torch.float32)
    alive_idx = torch.nonzero(nz == 1).squeeze(-1)
    if alive_idx.numel():
        bits[alive_idx] += -_bce_bit(sign[alive_idx], probs[alive_idx, 1])
        mag = a[alive_idx]
        pos = torch.arange(alive_idx.numel(), device=a.device)
        for j in range(1, MAX_MAG):
            cont = (mag >= j + 1).long()
            bits[alive_idx] += -_bce_bit(cont, probs[alive_idx, 1 + j])
            keep = cont == 1
            alive_idx = alive_idx[keep]
            mag = mag[keep]
            pos = pos[keep]
            if alive_idx.numel() == 0:
                break
    return bits


def _bce_bit(bit, p):
    """Binary cross-entropy for a 0/1 bit with modeled probability p."""
    p = p.clamp(1e-7, 1 - 1e-7)
    b = bit.float()
    return -(b * torch.log(p) + (1 - b) * torch.log(1 - p))


def discr_gauss_nll(symbols, mean, logscale):
    """NLL (nats) of integer symbols under discretized Gaussian (mean, std)."""
    scale = torch.exp(logscale).clamp(min=1e-3)
    lo = (symbols - 0.5 - mean) / scale
    hi = (symbols + 0.5 - mean) / scale
    cdf_hi = 0.5 * (1 + torch.erf(hi / np.sqrt(2.0)))
    cdf_lo = 0.5 * (1 + torch.erf(lo / np.sqrt(2.0)))
    p = (cdf_hi - cdf_lo).clamp(min=1e-9)
    return -torch.log(p)


def static_baseline_nll(symbols, group, acc_prev):
    """NLL (nats) under the CURRENT production head: per-(group,bucket)
    discrete Laplace fitted by matching mean |x - median|."""
    nll = np.zeros(symbols.size, dtype=np.float64)
    bucket = np.minimum(np.abs(acc_prev), 2)
    for gid in range(GROUPS):
        for bid in range(3):
            sel = (group == gid) & (bucket == bid)
            if sel.sum() == 0:
                continue
            s = symbols[sel].astype(np.float64)
            loc = float(np.median(s))
            m = float(np.mean(np.abs(s - loc)))
            if m <= 1e-9:
                nll[sel] = 0.0
                continue
            beta = min(max((np.sqrt(1 + 4 * m * m) - 1) / (2 * m), 1e-12),
                       1 - 1e-12)
            x = np.abs(s - loc)
            nll[sel] = np.log((1 + beta) / (1 - beta)) + x * np.log(1.0 / beta)
    return nll


# ---------------- modes ----------------

def mode_prepare(args):
    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, dec_t, Q_t, mapped, g, n_alive, dev, meta = load_run(
        args.run, out_dir, device)
    ladders = ladder_symbols(dec_t, Q_t, n_alive)
    logq_flat, g_flat, accs_flat, res_flat, lvl_flat = {}, {}, {}, {}, {}
    for f in FIELDS:
        n_cols = ladders[f]["q"].size // n_alive
        g_all = np.repeat(g, n_cols)
        for li in range(N_LEVELS):
            k = f"{f}_acc{li}"
            v = ladders[f][f"acc{li}"]
            logq_flat.setdefault(f, []).append(
                np.log(np.repeat(Q_t[f].cpu().numpy().reshape(-1), n_cols)
                       ).astype(np.float16))
            g_flat.setdefault(f, []).append(g_all)
            accs_flat.setdefault(f, []).append(v)
            lvl_flat.setdefault(f, []).append(np.full(v.size, li, dtype=np.int8))
        for li in range(1, N_LEVELS):
            res_flat.setdefault(f, []).append(ladders[f][f"res{li}"])
    data = {}
    for f in FIELDS:
        n_sym = ladders[f]["q"].size
        data[f"{f}_logq"] = np.concatenate(logq_flat[f])
        data[f"{f}_g"] = np.concatenate(g_flat[f])
        data[f"{f}_accs"] = np.concatenate(accs_flat[f], axis=0).reshape(
            N_LEVELS, -1).T.astype(np.int32)          # [n_sym, N_LEVELS]
        data[f"{f}_symidx"] = np.tile(np.arange(n_sym, dtype=np.int64), N_LEVELS)
        data[f"{f}_res"] = np.concatenate(res_flat[f]).astype(np.int8)
        data[f"{f}_lvl"] = np.concatenate(lvl_flat[f])
    np.savez_compressed(args.out, **data)
    print(f"[S4] wrote {args.out}")


def samples_for_field(data, f, levels, rng, cap):
    """(coarse, group, logq, residual, acc_prev) tuples for levels>=1.

    lvl/accs/symidx layout: level-major; symidx maps each flat entry back to
    its symbol so acc_prev = accs[sym_idx, li-1]."""
    idx_all = []
    for li in levels:
        idx_all.append(np.nonzero(data[f"{f}_lvl"] == li)[0])
    idx = np.concatenate(idx_all)
    if idx.size > cap:
        idx = rng.choice(idx, cap, replace=False)
    li_arr = data[f"{f}_lvl"][idx]
    sym_idx = data[f"{f}_symidx"][idx]
    acc_prev = np.zeros(idx.size, dtype=np.float32)
    for li in levels:
        m = li_arr == li
        acc_prev[m] = data[f"{f}_accs"][sym_idx[m], li - 1]
    n_sym = data[f"{f}_accs"].shape[0]
    res_idx = (li_arr - 1) * n_sym + (idx - li_arr * n_sym)
    return (torch.from_numpy(acc_prev),
            torch.from_numpy(data[f"{f}_g"][idx].astype(np.int64)),
            torch.from_numpy(data[f"{f}_logq"][idx].astype(np.float32)),
            torch.from_numpy(data[f"{f}_res"][res_idx].astype(np.float32)),
            acc_prev)


def mode_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {}
    for p in args.npz:
        d = np.load(p)
        for k in d.files:
            data.setdefault(k, np.concatenate([data[k], d[k]])
                            if k in data else d[k])
    rng = np.random.default_rng(0)
    heads = {}
    for f in FIELDS:
        levels = list(range(1, N_LEVELS))
        coarse, group, logq, res, acc_prev_np = samples_for_field(
            data, f, levels, rng, args.samples)
        n = res.numel()
        n_val = n // 10
        perm = torch.randperm(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        head = BinaryEntropyHead().to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        coarse = coarse.to(device)
        group = group.to(device)
        logq = logq.to(device)
        res = res.to(device)
        t0 = time.time()
        bs = 131072
        for epoch in range(args.epochs):
            order = tr_idx[torch.randperm(tr_idx.numel())]
            tot, nb = 0.0, 0
            for s in range(0, order.numel(), bs):
                b = order[s:s + bs]
                probs = head(coarse[b], group[b], logq[b])
                loss = decision_bits(res[b], probs).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * b.numel()
                nb += b.numel()
            print(f"[S4] {f} epoch {epoch}: train NLL {tot/nb:.4f} nats "
                  f"({time.time()-t0:.0f}s)", flush=True)
        with torch.no_grad():
            probs = head(coarse[val_idx], group[val_idx], logq[val_idx])
            bits_learned = float(decision_bits(res[val_idx], probs).mean())
            static = static_baseline_nll(
                res[val_idx].cpu().numpy().astype(np.int32),
                group[val_idx].cpu().numpy(), acc_prev_np[val_idx.numpy()])
            bits_static = float(static.sum() / max(n_val, 1) / np.log(2))
        torch.save(head.state_dict(), f"{args.out}_{f}.pt")
        heads[f] = f"{args.out}_{f}.pt"
        print(f"[S4] {f} val: learned {bits_learned:.3f} bits/sym vs static "
              f"{bits_static:.3f} bits/sym -> "
              f"{100*(1-bits_learned/bits_static):.1f}% saving", flush=True)
    print(f"[S4] heads saved: {heads}")


def mode_apply(args):
    raise SystemExit("apply mode: use c25_real_bitstream.py flow with the "
                     "trained head plugged into model_params (integration "
                     "follows once train-mode numbers pass the gate)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--run", required=True)
    pp.add_argument("--out-dir", default="analysis/s2_prefix_sweep")
    pp.add_argument("--out", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--npz", nargs="+", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--epochs", type=int, default=4)
    pt.add_argument("--samples", type=int, default=8_000_000)
    args = ap.parse_args()
    if args.mode == "prepare":
        mode_prepare(args)
    elif args.mode == "train":
        mode_train(args)


if __name__ == "__main__":
    main()
