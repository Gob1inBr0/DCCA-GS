#!/usr/bin/env python3
"""Learned-context probe v4: can a tiny MLP beat the PRODUCTION composite
conditional table on the enhancement-layer decision streams?

This is the honest restart of the learned-entropy line. The two earlier
attempts failed for reasons this probe is designed against:

  - S4 (learned Gaussian head) was integrated against production and found
    zero gain — its context (coarse, group, logQ) was a subset of what the
    static table already uses;
  - probe v3 measured a learned flag model against a REDUCED per-group
    table (+45.9%), which overstates what production integration can
    deliver, because the production table also conditions on the
    |coarse-symbol| bucket.

v4 rules (2026-09-26 lesson, registered in the negative-result ledger):
  1. the baseline IS the production conditional model — same
     (group x |coarse| bucket) cells, same fallback chain, same geometric
     tail — evaluated on the SAME decisions;
  2. anchor-level 90/10 split, fixed before any fitting; the static cells
     and the MLP both fit on train anchors only;
  3. bits = REAL arithmetic-coder bits (roundtrip asserted), not BCE;
  4. every MLP feature must be recomputable decoder-side at the moment the
     symbol is coded (stream order respected) — the zero-side-info contract
     decides feature legality, not convenience. Features derived here from
     the final symbols (accumulators, residuals) are bit-identical to what
     the decoder has already decoded at that point of the stream.

Decision streams (per field, one ladder level >= 1 per invocation):
  --stage flag : one decision per symbol, bit = (residual != 0);
                 static baseline = 1 - p0 of the (group, bucket) cell
  --stage cont : magnitude-continuation bits of the nonzero symbols,
                 one decision per (symbol, k), bit = |d| >= k+1;
                 static baseline = the cell's geometric beta

Feature groups (ablatable via --context; all decoder-recomputable):
  base      : |coarse| and signed coarse of the symbol's own accumulator,
              log Q, group embedding, bucket embedding, step-tier embedding
  spatial   : coarse symbols of the previous two anchors, same column
              (Morton row order = coding order)
  crosscol  : residual and coarse of the previous column of the SAME
              anchor (coded earlier within the chunk)
  crossfield: per-anchor mean |residual| of the fields coded earlier at
              the same level (feat before scaling before offset)

Gate (pre-registered): net payload saving >= 15% on the target stream
(val split, real bits, vs the isomorphic static-train baseline) with the
weights charged against the saving. Below that, the static table stays and
the learned-context direction closes for good.

Usage (training machine, c25 python + PYTHONPATH=/home/project2/c25_pylib2):
  python scripts/probe_ctx_v4.py \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --field offset --level 1 --stage flag --context all \
      --stats <run>/anchor_stats.npz \
      --out analysis/s2_prefix_sweep/probe_v4_offset_L1_flag.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from binary_prob_probe import bce_bits, code_bits_real  # noqa: E402
from c25_real_bitstream import FIELD_STEPS, bucket_of, fit_chunk_params  # noqa: E402

GROUPS = 32
N_BUCKETS = 3
CONTEXT_GROUPS = ("base", "spatial", "crosscol", "crossfield")
FIELD_ORDER = ("feat", "scaling", "offset")


# ---- ladder / grouping / tier helpers ----
# Keep byte-for-byte in sync with c25_real_bitstream (they are module-level
# functions on the s5-integration and 16x-ladder branches; import from
# there once all three land on the same line).
def parse_ladder(spec):
    """Parse 'feat=16,8,4,2,1 offset=...' -> {field: [steps,...,1]}.

    Validation: powers of two, strictly decreasing, ending at 1 (full
    precision); fields not named keep their preset ladder."""
    out = {}
    for part in spec.split():
        if "=" not in part:
            raise SystemExit(f"--ladder: expected field=steps, got {part!r}")
        f, _, s = part.partition("=")
        if f not in FIELD_ORDER:
            raise SystemExit(f"--ladder: unknown field {f!r}")
        steps = [int(x) for x in s.split(",") if x.strip()]
        if not steps or steps[-1] != 1:
            raise SystemExit(f"--ladder {f}: ladder must end at step 1 "
                             "(full precision)")
        for x in steps:
            if x < 1 or (x & (x - 1)) != 0:
                raise SystemExit(f"--ladder {f}: steps must be powers of "
                                 f"two, got {x}")
        for a, b in zip(steps[:-1], steps[1:]):
            if b >= a:
                raise SystemExit(f"--ladder {f}: steps must strictly "
                                 "decrease")
        out[f] = steps
    return out


def area_rank_groups(area, n_groups):
    """Contribution-area rank buckets: descending area -> group 0..n-1."""
    n = area.shape[0]
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(-area, kind="stable")] = np.arange(n)
    return np.minimum(rank * n_groups // n, n_groups - 1).astype(np.int16)


def uniform_groups(n, n_groups):
    """Uniform row-order buckets (documented fallback without stats)."""
    return np.minimum(np.arange(n) * n_groups // n,
                      n_groups - 1).astype(np.int16)


def per_anchor_step_tiers(qmul_flat, n_alive):
    """Per-anchor step-multiplier tercile tier (cut points fixed at the
    1/3, 2/3 quantiles — decoder-side recomputable)."""
    cols = qmul_flat.size // n_alive
    qmul_anchor = qmul_flat.reshape(-1, cols)[:, 0]
    t1, t2 = np.quantile(qmul_anchor, [1 / 3, 2 / 3])
    return np.digitize(qmul_anchor, [t1, t2]).astype(np.int16)


def telescoping_acc(q_flat, steps, upto):
    """Decoded accumulator at ladder level `upto` (None below level 0).

    Values equal what the decoder accumulates from the stream, so features
    derived from it are decoder-side recomputable.
    """
    if upto < 0:
        return None
    a = None
    for l2 in range(upto + 1):
        if l2 == 0:
            a = np.round(q_flat / steps[0]).astype(np.int32)
        else:
            ratio = steps[l2 - 1] // steps[l2]
            a = ratio * a + (np.round(q_flat / steps[l2]).astype(np.int32)
                             - ratio * a)
    return a


def build_stream(args):
    """Recompute the production encoding state for one (field, level).

    Mirrors c25_real_bitstream.main symbol-for-symbol: same Q source
    (production grid from the codec decode), same grouping, same step
    tiers, same ladder.
    """
    from scaffold_gs.hacpp import HACPlusCodec

    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    del codec
    n_alive = model.num_anchors
    dev = model.device
    view = model._view
    dec_t = {
        "feat": view.anchor_feat.data.detach().float(),
        "scaling": view.scaling.data.detach().float(),
        "offset": view.offset.data.detach().float(),
    }
    last_q = getattr(model.core, "last_decode_Q", None)
    assert last_q is not None, "need production-grid Q (post-audit codec)"
    Q_t = {
        "feat": last_q["feat"].to(dev).float(),
        "scaling": last_q["scaling"].to(dev).float(),
        "offset": last_q["offsets"].to(dev).float().reshape(
            dec_t["offset"].shape),
    }

    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_trained = int(meta["num_anchors_total"])
    if args.stats and Path(args.stats).exists():
        area = np.load(args.stats)["area"].astype(np.float64)
        assert area.shape[0] == n_trained
        group_of_trained = area_rank_groups(area, args.groups)
    else:
        group_of_trained = None
        print("[P4] WARNING: no stats — uniform-bucket grouping", flush=True)
    run_tag = run_dir.name
    if args.mapped and Path(args.mapped).exists():
        mapped = np.load(args.mapped).astype(np.int64)
    else:
        ck = torch.load(run_dir / "ckpts" / "ckpt_30000.pth",
                        map_location="cpu", weights_only=False)
        trained_xyz = ck["model_state"]["_anchor"].to(dev)
        decoded_xyz = model.core.get_anchor
        mapped = np.empty(decoded_xyz.shape[0], dtype=np.int64)
        with torch.no_grad():
            for start in range(0, decoded_xyz.shape[0], 1024):
                end = min(start + 1024, decoded_xyz.shape[0])
                dmat = torch.cdist(decoded_xyz[start:end], trained_xyz)
                mapped[start:end] = dmat.min(dim=1)[1].cpu().numpy()
                del dmat
        mp = Path(args.out).parent / (run_tag + ".mapped.npy")
        if not mp.exists():
            np.save(mp, mapped)
    assert mapped.size == n_alive
    g_base_alive = (group_of_trained[mapped] if group_of_trained is not None
                    else uniform_groups(n_alive, args.groups))

    fsteps = {f: list(FIELD_STEPS[f]) for f in FIELD_ORDER}
    if args.ladder:
        user = parse_ladder(args.ladder)
        fsteps = {f: user.get(f, fsteps[f]) for f in FIELD_ORDER}
    q_flat = {f: torch.round(dec_t[f] / Q_t[f]).cpu().numpy()
              .astype(np.int32).reshape(-1) for f in FIELD_ORDER}
    tiers = {f: per_anchor_step_tiers(Q_t[f].cpu().numpy().reshape(-1),
                                      n_alive) for f in FIELD_ORDER}
    n_cols = {f: q_flat[f].size // n_alive for f in FIELD_ORDER}
    g_flat = {f: np.repeat(g_base_alive * 3 + tiers[f], n_cols[f])
              for f in FIELD_ORDER}
    logq_flat = {f: np.log(Q_t[f].cpu().numpy().reshape(-1)
                           ).astype(np.float32) for f in FIELD_ORDER}

    f, li = args.field, args.level
    assert 1 <= li < len(fsteps[f]), \
        f"level {li} outside {f} ladder {fsteps[f]}"
    acc_prev_f = telescoping_acc(q_flat[f], fsteps[f], li - 1)
    ratio = fsteps[f][li - 1] // fsteps[f][li]
    layer = (np.round(q_flat[f] / fsteps[f][li]).astype(np.int32)
             - ratio * acc_prev_f)
    # cross-field accumulators/residuals at the level each earlier field
    # actually sits at when field f's level-li chunk is coded
    cross = {}
    for ff in FIELD_ORDER[:FIELD_ORDER.index(f)]:
        top = min(li, len(fsteps[ff]) - 1)
        acc_top = telescoping_acc(q_flat[ff], fsteps[ff], top)
        if top >= 1:
            r_ff = fsteps[ff][top - 1] // fsteps[ff][top]
            d_ff = (np.round(q_flat[ff] / fsteps[ff][top]).astype(np.int32)
                    - r_ff * telescoping_acc(q_flat[ff], fsteps[ff], top - 1))
        else:
            d_ff = acc_top
        cross[ff] = {"acc": acc_top, "res": d_ff, "level": top}
    return {
        "field": f, "level": li, "layer": layer, "ctx": bucket_of(acc_prev_f),
        "g_chunk": g_flat[f], "acc_prev": acc_prev_f, "cross": cross,
        "q_flat": q_flat, "logq_flat": logq_flat, "fsteps": fsteps,
        "n_alive": n_alive, "n_cols": n_cols, "dev": dev,
    }


def symbol_features(st, args):
    """Symbol-level feature arrays; every entry decoder-recomputable at the
    moment the symbol is coded (module docstring lists the legality of each
    group). Returns {name: float32 array (n_symbols, dim)}."""
    f, n_alive, cols = st["field"], st["n_alive"], st["n_cols"][st["field"]]
    acc = st["acc_prev"]
    layer = st["layer"]
    out = {}
    if "base" in args.context:
        out["abs_coarse"] = np.abs(acc).astype(np.float32)[:, None]
        out["coarse"] = acc.astype(np.float32)[:, None]
        out["logq"] = st["logq_flat"][f][:, None]
    if "spatial" in args.context:
        a2 = acc.reshape(n_alive, cols)
        nb1 = np.zeros_like(a2)
        nb2 = np.zeros_like(a2)
        nb1[1:] = a2[:-1]
        nb2[2:] = a2[:-2]
        out["spatial"] = np.stack([nb1.reshape(-1), np.abs(nb1).reshape(-1),
                                   nb2.reshape(-1), np.abs(nb2).reshape(-1)],
                                  -1).astype(np.float32)
    if "crosscol" in args.context:
        d2 = layer.reshape(n_alive, cols).astype(np.float32)
        a2 = acc.reshape(n_alive, cols).astype(np.float32)
        dc = np.zeros_like(d2)
        ac = np.zeros_like(a2)
        dc[:, 1:] = d2[:, :-1]
        ac[:, 1:] = a2[:, :-1]
        out["crosscol"] = np.stack([dc.reshape(-1), np.abs(dc).reshape(-1),
                                    ac.reshape(-1), np.abs(ac).reshape(-1)],
                                   -1).astype(np.float32)
    if "crossfield" in args.context:
        means = []
        for ff in FIELD_ORDER[:FIELD_ORDER.index(f)]:
            m = np.abs(st["cross"][ff]["res"].reshape(n_alive, -1)
                       ).mean(-1, keepdims=True)
            means.append(np.repeat(m, cols, axis=1).reshape(-1))
        while len(means) < 2:
            means.append(np.zeros(acc.size, dtype=np.float64))
        out["crossfield"] = np.stack(means[:2], -1).astype(np.float32)
    return out


class CtxMLP(nn.Module):
    """Embeddings for the discrete conditions + MLP over continuous context
    -> logit P(decision=1). Small on purpose: 32-group embeddings are ~1.3
    KB in float32, the MLP body ~17 KB."""

    def __init__(self, n_groups, ctx_groups, cont_dim, emb=8, hidden=64):
        super().__init__()
        self.ctx_groups = ctx_groups
        disc = 0
        if "base" in ctx_groups:
            self.emb_group = nn.Embedding(n_groups, emb)
            self.emb_bucket = nn.Embedding(N_BUCKETS, 4)
            self.emb_tier = nn.Embedding(3, 4)
            disc = emb + 4 + 4
        self.net = nn.Sequential(
            nn.Linear(cont_dim + disc, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, cont, disc):
        parts = list(cont)
        if "base" in self.ctx_groups:
            parts += [self.emb_group(disc["group"]),
                      self.emb_bucket(disc["bucket"]),
                      self.emb_tier(disc["tier"])]
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


def _clip(p):
    return min(max(float(p), 1e-4), 1 - 1e-4)


def static_flag_baseline(bits, cell_of_dec, g_of_dec, train_sel, n_cells,
                         n_groups):
    """Isomorphic static baseline for stage flag: per-(group,bucket) cell
    rate fit on TRAIN decisions only, with the production fallback chain
    composite cell -> contribution group (any bucket) -> global."""
    fallback = _clip(bits[train_sel].mean())
    cnt = np.bincount(cell_of_dec[train_sel], minlength=n_cells)
    pos = np.bincount(cell_of_dec[train_sel], weights=bits[train_sel],
                      minlength=n_cells)
    rate = np.where(cnt > 0, pos / np.maximum(cnt, 1), np.nan)
    gcnt = np.bincount(g_of_dec[train_sel], minlength=n_groups)
    gpos = np.bincount(g_of_dec[train_sel], weights=bits[train_sel],
                       minlength=n_groups)
    grate = np.where(gcnt > 0, gpos / np.maximum(gcnt, 1), np.nan)
    r = rate[cell_of_dec]
    miss = np.isnan(r)
    if miss.any():
        rg = grate[g_of_dec[miss]]
        rg[np.isnan(rg)] = fallback
        r[miss] = rg
    return np.clip(r, 1e-4, 1 - 1e-4)


def static_cont_baseline(surv_cell, surv_mag, surv_train, n_cells):
    """Isomorphic geometric-tail baseline for stage cont: per-cell beta
    from TRAIN survivor magnitudes (the production formula), global fit as
    the fallback. Returns per-cell beta (float64, size n_cells)."""
    cnt = np.bincount(surv_cell[surv_train], minlength=n_cells)
    m1 = np.bincount(surv_cell[surv_train], weights=surv_mag[surv_train],
                     minlength=n_cells)
    mean_abs = m1 / np.maximum(cnt, 1)
    beta = np.clip(1.0 - 1.0 / np.maximum(mean_abs, 1.0), 0.05, 0.98)
    gm = float(np.mean(surv_mag[surv_train])) if surv_train.any() else 1.0
    beta[cnt == 0] = np.clip(1.0 - 1.0 / max(gm, 1.0), 0.05, 0.98)
    return beta


def _cont_decisions(d, cell_sym, g_chunk):
    """Expansion of stage-C continuation bits.

    A nonzero symbol with magnitude m emits decisions k = 1..m with
    bit = (k < m); k = m terminates it. Returns (bit, sym_of_dec,
    cell_of_dec, g_of_dec, k_of_dec, counts_per_survivor).
    """
    nz = np.nonzero(d != 0)[0]
    mag_s = np.abs(d[nz]).astype(np.int64)
    counts = mag_s
    if nz.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return (empty, empty, empty, empty,
                np.zeros(0, dtype=np.float32), empty)
    sym_of_dec = np.repeat(nz, counts)
    cell_of_dec = np.repeat(cell_sym[nz], counts)
    g_of_dec = np.repeat(g_chunk[nz], counts)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    within = np.arange(int(counts.sum())) - np.repeat(offs, counts) + 1
    k_of_dec = within.astype(np.float32)
    bit = (within < np.repeat(mag_s, counts)).astype(np.int64)
    return bit, sym_of_dec, cell_of_dec, g_of_dec, k_of_dec, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--field", default="offset", choices=list(FIELD_ORDER))
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--stage", default="flag", choices=["flag", "cont"])
    ap.add_argument("--context", default="all",
                    help=f"comma list of {CONTEXT_GROUPS} or 'all'")
    ap.add_argument("--groups", type=int, default=GROUPS)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--mapped", default=None)
    ap.add_argument("--ladder", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--gate-net-pct", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.context = (list(CONTEXT_GROUPS) if args.context == "all"
                    else args.context.split(","))
    for c in args.context:
        assert c in CONTEXT_GROUPS, f"unknown context group {c}"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    st = build_stream(args)
    layer, ctx, g_chunk = st["layer"], st["ctx"], st["g_chunk"]
    n_alive, cols = st["n_alive"], st["n_cols"][st["field"]]

    # production table (deployment form) supplies the medians and the
    # reference-column probabilities; the decision bits are exactly what
    # code_chunk codes against those medians
    locs, p0s, bts = fit_chunk_params(layer, g_chunk, ctx)
    loc = locs[g_chunk, ctx]
    d = layer.astype(np.int64) - loc

    n_g = int(g_chunk.max()) + 1
    n_cells = n_g * N_BUCKETS
    cell_sym = g_chunk * N_BUCKETS + ctx          # per-symbol cell id

    # ---- decision stream ----
    if args.stage == "flag":
        bit = (d != 0).astype(np.int64)
        sym_of_dec = np.arange(bit.size)
        cell_of_dec = cell_sym
        g_of_dec = g_chunk
        counts = None
        # production reference column: 1 - p0 of the same cells
        p_prod_all = np.clip(1.0 - p0s[g_chunk, ctx], 1e-4, 1 - 1e-4)
    else:
        nz = np.nonzero(d != 0)[0]
        bit, sym_of_dec, cell_of_dec, g_of_dec, k_of_dec, counts = \
            _cont_decisions(d, cell_sym, g_chunk)
        # production reference column: the cells' all-fit beta
        p_prod_all = np.repeat(np.clip(bts[g_chunk[nz], ctx[nz]],
                                       0.05, 0.98), counts)
    print(f"[P4] {args.field} L{args.level} stage={args.stage}: "
          f"decisions={bit.size} positives={int(bit.sum())}", flush=True)

    # ---- anchor-level 90/10 split (fixed before any fitting) ----
    rng = np.random.default_rng(0)
    aperm = rng.permutation(n_alive)
    is_val_anchor = np.zeros(n_alive, dtype=bool)
    is_val_anchor[np.sort(aperm[:n_alive // 10])] = True
    val_sel = is_val_anchor[sym_of_dec // cols]
    tr_sel = ~val_sel
    print(f"[P4] split: train={int(tr_sel.sum())} val={int(val_sel.sum())}",
          flush=True)

    # ---- static baseline (isomorphic, train-only fit) ----
    if args.stage == "flag":
        p_stat = static_flag_baseline(bit, cell_of_dec, g_of_dec, tr_sel,
                                      n_cells, n_g)
    else:
        surv_train = ~is_val_anchor[nz // cols]
        beta_cell = static_cont_baseline(cell_sym[nz],
                                         np.abs(d[nz]).astype(np.float64),
                                         surv_train, n_cells)
        p_stat = beta_cell[cell_of_dec]

    # ---- learned model ----
    feats_sym = symbol_features(st, args)
    feats = ({k: v[sym_of_dec] for k, v in feats_sym.items()}
             if args.stage == "cont" else feats_sym)
    cont_names = list(feats)
    cont_dim = sum(feats[k].shape[1] for k in cont_names) \
        + (1 if args.stage == "cont" else 0)
    cont_cols = [torch.from_numpy(np.ascontiguousarray(feats[k]))
                 for k in cont_names]
    if args.stage == "cont":
        cont_cols.append(torch.from_numpy(
            (k_of_dec / max(float(k_of_dec.max()), 1.0)
             ).astype(np.float32))[:, None])
    cont_t = torch.cat(cont_cols, -1).to(st["dev"])
    disc = {
        "group": torch.from_numpy((g_of_dec // 3).astype(np.int64)),
        "bucket": torch.from_numpy(ctx[sym_of_dec].astype(np.int64)),
        "tier": torch.from_numpy((g_of_dec % 3).astype(np.int64)),
    } if "base" in args.context else {}
    disc = {k: v.to(st["dev"]) for k, v in disc.items()}
    bit_t = torch.from_numpy(bit.astype(np.float32)).to(st["dev"])
    tr_t = torch.from_numpy(np.nonzero(tr_sel)[0]).to(st["dev"])

    mlp = CtxMLP(n_g, args.context, cont_dim).to(st["dev"])
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for epoch in range(args.epochs):
        perm = tr_t[torch.randperm(tr_t.numel(), device=st["dev"])]
        tot = 0.0
        for s0 in range(0, perm.numel(), 131072):
            b = perm[s0:s0 + 131072]
            loss = bce(mlp(cont_t[b], {k: v[b] for k, v in disc.items()}),
                       bit_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * b.numel()
        print(f"[P4] mlp epoch {epoch}: train bits/sym "
              f"{tot / np.log(2) / max(perm.numel(), 1):.4f}", flush=True)

    mlp.eval()
    with torch.no_grad():
        p_mlp = torch.sigmoid(mlp(cont_t, disc)).cpu().numpy().astype(
            np.float64)

    # ---- evaluation: REAL coder bits on val decisions ----
    val_idx = np.nonzero(val_sel)[0]
    val_bit = bit[val_idx]
    val_stat = np.clip(p_stat[val_idx], 1e-4, 1 - 1e-4)
    val_mlp = np.clip(p_mlp[val_idx], 1e-4, 1 - 1e-4)
    bits_stat_real, back_s, p_eff_s = code_bits_real(val_bit, val_stat)
    bits_mlp_real, back_m, p_eff_m = code_bits_real(val_bit, val_mlp)
    assert (back_s == val_bit).all() and (back_m == val_bit).all(), \
        "coder roundtrip failed"
    val_prod = np.clip(p_prod_all[val_idx], 1e-4, 1 - 1e-4)
    bits_prod_real, _, _ = code_bits_real(val_bit, val_prod)

    weight_bytes = sum(pp.numel() * 4 for pp in mlp.parameters())
    raw_pct = 100 * (1 - bits_mlp_real / max(bits_stat_real, 1))
    net_pct = 100 * (1 - (bits_mlp_real + weight_bytes * 8)
                     / max(bits_stat_real, 1))
    verdict = "PASS" if net_pct >= args.gate_net_pct else "FAIL"

    print(f"[P4] static-train val: real {bits_stat_real/8/1e6:.3f} MB "
          f"(theory {bce_bits(val_bit, p_eff_s)/8/1e6:.3f} MB)", flush=True)
    print(f"[P4] learned      val: real {bits_mlp_real/8/1e6:.3f} MB "
          f"(theory {bce_bits(val_bit, p_eff_m)/8/1e6:.3f} MB)", flush=True)
    print(f"[P4] production-table reference (all-fit, optimistic): "
          f"{bits_prod_real/8/1e6:.3f} MB", flush=True)
    print(f"[P4] saving raw {raw_pct:.1f}% / net {net_pct:.1f}% "
          f"({weight_bytes/1024:.0f} KB weights) -> {verdict} "
          f"(gate net >= {args.gate_net_pct}%)", flush=True)

    out_path.write_text(json.dumps({
        "run": args.run, "field": args.field, "level": args.level,
        "stage": args.stage, "context": args.context, "groups": args.groups,
        "ladder": args.ladder,
        "n_decisions": int(bit.size), "positives": int(bit.sum()),
        "n_val": int(val_sel.sum()),
        "bits_static_real_val": bits_stat_real,
        "bits_mlp_real_val": bits_mlp_real,
        "bits_static_prod_ref_val": bits_prod_real,
        "weight_bytes": weight_bytes,
        "raw_saving_pct": round(raw_pct, 2),
        "net_saving_pct": round(net_pct, 2),
        "gate_net_pct": args.gate_net_pct,
        "verdict": verdict,
    }, indent=1))
    print(f"[P4] wrote {out_path}")


if __name__ == "__main__":
    main()
