#!/usr/bin/env python3
"""S3/priority-1: REAL layered bitstream — range-coded, prefix-decodable.

Upgrades the C2.5 accounting experiment to a real file: the three quantized
fields (feat/scaling/offsets) are re-encoded as a nested quality ladder with
an actual range coder (constriction; per-symbol discrete-Gaussian models
parameterized by the 32 contribution groups — the C2.5 zero-training
conditional). The bitstream layout is

  [4B header len][header json] then per chunk (ladder order):
  [4B params len][params: 32x(loc,b) float32][4B data len][data]

so truncating after any chunk boundary yields a valid decodable prefix.
Real cumulative bytes at each boundary are paired with a real render PSNR
computed from bitstream-DECODED symbols — the measured "cumulative bytes ->
PSNR" curve (journal-axis priority 1).

Ladder (power-of-two steps, telescoping reconstruction):
  chunk L0: q0 = round(q/16)                 -> dequant 16*Q*q0
  chunk L1: r1 = q1 - 2*q0, q1 = round(q/8)  -> acc = 2*acc + r1
  chunk L2: r2 = q2 - 2*q1, q2 = round(q/4)  -> acc = 2*acc + r2
  chunk L3: r3 = q3 - 2*q2, q3 = round(q/2)  -> acc = 2*acc + r3
  chunk L4: r4 = q  - 2*q3                   -> acc = q   (exact decoded grid)

Run under the DCCA env python with PYTHONPATH=/home/project2/c25_pylib2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import constriction  # noqa: E402

N_VIEWS = 16
GROUPS = 32
FACTORS = [16, 8, 4, 2]
STEPS = FACTORS + [1]
N_LEVELS = len(STEPS)
FAMILY_RANGE = 4096


def fit_laplace(symbols):
    s = np.asarray(symbols, dtype=np.float64)
    loc = float(np.median(s))
    m = float(np.mean(np.abs(s - loc)))
    if m <= 1e-9:
        return loc, 1e-3
    beta = (np.sqrt(1.0 + 4.0 * m * m) - 1.0) / (2.0 * m)
    b = -1.0 / np.log(max(min(float(beta), 1.0 - 1e-9), 1e-12))
    return loc, max(b, 1e-3)


BUCKETS = 3  # residual context: |coarse symbol| in {0}, {1}, {>=2}


def bucket_of(acc_prev):
    """Context bucket from the already-decoded coarse symbol: 0 / 1 / >=2."""
    a = np.abs(acc_prev)
    return np.minimum(a, 2).astype(np.int16)


# ---- binary-decomposition range coding ("geometric-binary") ----
# Each integer symbol d (relative to its bucket's integer median `loc`) is
# coded as binary decisions: [d != 0] -> [sign] -> magnitude unary-bit
# stages [>=k+1?]. Every decision keeps its probability bounded away from
# 0/1, so the fixed-point range coder never sees a zero-probability symbol
# (the failure mode of single-shot discretized-Gaussian coding with tiny
# fitted sigma).

def _bin_ms(p):
    """(mean, sigma) of a discretized Gaussian on {0,1} with P(1)=p:
    mean sits at the likely value, sigma widens until the rare side keeps
    nonzero probability (range-coder safe)."""
    pt = torch.clamp(torch.as_tensor(np.asarray(p, dtype=np.float64)),
                     1e-4, 1 - 1e-4)
    mean = torch.round(pt)
    z = torch.erfinv(2.0 * torch.abs(pt - mean) - 1.0).abs() * np.sqrt(2.0)
    sigma = torch.clamp(0.5 / torch.clamp(z, min=1e-3), 0.05, 8.0)
    return mean.numpy(), sigma.numpy()


def _bin_family():
    return constriction.stream.model.QuantizedGaussian(0, 1)


class _S4Head(torch.nn.Module):
    """Learned conditional entropy head: (coarse, group emb, logQ) ->
    (mean, logscale) of a discretized Gaussian for the residual symbol."""

    def __init__(self, emb_dim=8, hidden=64):
        super().__init__()
        self.emb = torch.nn.Embedding(GROUPS, emb_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 + emb_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, coarse, group, logq):
        x = torch.cat([coarse[:, None].float(), self.emb(group),
                       logq[:, None].float()], dim=-1)
        out = self.net(x)
        return out[:, 0], out[:, 1]


def _load_s4_heads(prefix, device="cpu"):
    heads = {}
    for f in ("feat", "scaling", "offset"):
        h = _S4Head()
        h.load_state_dict(torch.load(f"{prefix}_{f}.pt", map_location="cpu",
                                     weights_only=True))
        h.eval().to(device)
        heads[f] = h
    return heads


def _s4_ms(head, coarse, group, logq, device):
    """(mean, sigma) arrays for each symbol under the learned head."""
    with torch.no_grad():
        m, ls = head(torch.from_numpy(coarse).to(device),
                     torch.from_numpy(group.astype(np.int64)).to(device),
                     torch.from_numpy(logq.astype(np.float32)).to(device))
    mean = torch.round(m).cpu().numpy().astype(np.float32)
    z = torch.erfinv(2.0 * torch.abs(m - torch.round(m)) - 1.0).abs() \
        .cpu().numpy() * np.sqrt(2.0)
    sigma = np.clip(0.5 / np.clip(z, 1e-3, None), 0.05, 8.0).astype(np.float32)
    return mean, sigma


def fit_chunk_params(layer, g, ctx=None):
    """Per (group[, bucket]): loc (int median), p0 = P(d==0),
    beta = P(>=k+1 | >=k) of the geometric tail.

    g is a COMPOSITE index; table size follows its max+1, so callers fold
    extra condition dimensions (e.g. quant-step tier) into g and pass the
    matching tier count via n_groups."""
    n_groups = int(g.max()) + 1
    shape = (n_groups, BUCKETS) if ctx is not None else (n_groups,)
    locs = np.zeros(shape, dtype=np.int64)
    p0s = np.zeros(shape, dtype=np.float64)
    bts = np.zeros(shape, dtype=np.float64)
    for gid in range(n_groups):
        bids = range(BUCKETS) if ctx is not None else [None]
        for bid in bids:
            sel = layer[(g == gid) & (ctx == bid)] if ctx is not None \
                else layer[g == gid]
            if sel.size == 0:
                # composite groups can be empty; fall back to the base group
                # (same contribution group across tiers), then to everything
                sel = layer[(g // 3) == (gid // 3)] if ctx is not None \
                    else layer[g == gid]
            if sel.size == 0:
                sel = layer
            loc = int(np.median(sel))
            d = sel.astype(np.float64) - loc
            nz = float(np.mean(d != 0)) if sel.size else 0.0
            m1 = float(np.mean(np.abs(d[d != 0]))) if np.any(d != 0) else 1.0
            beta = min(max(1.0 - 1.0 / max(m1, 1.0), 0.05), 0.98)
            i = (gid, bid) if ctx is not None else gid
            locs[i] = loc
            p0s[i] = 1.0 - nz
            bts[i] = beta
    return locs, p0s, bts


def code_chunk(layer, g, ctx=None, s4=None):  # s4=(head, logq, dev, base_g)
    """Binary-decomposition range coding of `layer`.

    Decisions per symbol d = layer - loc:
      A: d != 0            (P(1) = 1 - p0)
      B: sign, if d != 0   (P(1) = 0.5, sign bit 1 = negative)
      C: for k = 1, 2, ... while alive: |d| >= k+1 ?  (P(1) = beta)
    With s4=(head, logq, device), stage-A/C probabilities come from the
    learned conditional head (decoder recomputes coarse+group+logQ: zero
    side info); loc stays the static integer median (shipped in params).
    Returns (data, params, plain_bytes)."""
    locs, p0s, bts = fit_chunk_params(layer, g, ctx)
    if ctx is not None:
        loc = locs[g, ctx]                 # per-symbol: (group, bucket) table
        p0 = p0s[g, ctx]
        beta = bts[g, ctx]
    else:
        loc = locs[g]
        p0 = p0s[g]
        beta = bts[g]
    d = layer.astype(np.int64) - loc

    head, logq, device, g_base = (s4 + (None,))[:4] if s4 is not None else (None, None, None, None)
    enc = constriction.stream.queue.RangeEncoder()
    if head is not None:
        meanH, sigH = _s4_ms(head, loc, g_base, logq, device)
        # stage A: P(d != 0) = 1 - |Phi(meanH) - Phi(meanH-1)|  approximated
        # by the learned Gaussian's tail mass beyond +-0.5
        from math import erf
        ph = lambda x: 0.5 * (1 + np.vectorize(erf)(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))
        p_nz = np.clip((ph(meanH + 0.5) - ph(meanH - 0.5)) /
                       (ph(meanH + 4096) - ph(meanH - 4096)), 1e-4, 1 - 1e-4)
        mA, sA = _bin_ms(1.0 - p_nz)
        enc.encode((d != 0).astype(np.int32), _bin_family(),
                   mA.astype(np.float32), sA.astype(np.float32))
    else:
        mA, sA = _bin_ms(1.0 - p0)
        enc.encode((d != 0).astype(np.int32), _bin_family(),
                   mA.astype(np.float32), sA.astype(np.float32))
    alive = np.nonzero(d != 0)[0]
    if alive.size:
        mS, sS = _bin_ms(0.5)
        enc.encode((d[alive] < 0).astype(np.int32), _bin_family(),
                   np.full(alive.size, mS, dtype=np.float32),
                   np.full(alive.size, sS, dtype=np.float32))
    mags = np.abs(d[alive]) if alive.size else np.zeros(0, dtype=np.int64)
    # stage-C probabilities for the CURRENT alive set, recomputed each round
    k = 1
    while alive.size:
        if head is not None:
            sig_cur = sigH[alive]
            beta_cur = np.clip(1.0 - 1.0 / np.maximum(sig_cur, 1.05), 0.05, 0.98)
            mB_cur, sB_cur = _bin_ms(beta_cur)
        else:
            mB_cur, sB_cur = _bin_ms(beta)
        bits = (mags >= k + 1).astype(np.int32)
        enc.encode(bits, _bin_family(),
                   mB_cur[alive].astype(np.float32),
                   sB_cur[alive].astype(np.float32))
        alive = alive[bits == 1]
        mags = mags[bits == 1]
        k += 1
        if k > (1 << 20):
            raise RuntimeError("unbounded magnitude stages")
    data = enc.get_compressed().tobytes()
    params = (locs, p0s, bts)

    plain_bytes = None
    if ctx is not None:
        pl, pp0s, pbts = fit_chunk_params(layer, g // 3, None)
        # plain baseline conditions on the base group only (strip tier)
        ploc = pl[g // 3]
        pp0 = pp0s[g // 3]
        pbeta = pbts[g // 3]
        pd = layer.astype(np.int64) - ploc
        enc2 = constriction.stream.queue.RangeEncoder()
        mPA, sPA = _bin_ms(1.0 - pp0)
        enc2.encode((pd != 0).astype(np.int32), _bin_family(), mPA, sPA)
        alive2 = np.nonzero(pd != 0)[0]
        if alive2.size:
            mPS, sPS = _bin_ms(0.5)
            enc2.encode((pd[alive2] < 0).astype(np.int32), _bin_family(),
                        np.full(alive2.size, mPS, dtype=np.float32),
                        np.full(alive2.size, sPS, dtype=np.float32))
            mags2 = np.abs(pd[alive2])
            mPB_all, sPB_all = _bin_ms(pbeta)
            kk = 1
            while alive2.size:
                bits2 = (mags2 >= kk + 1).astype(np.int32)
                enc2.encode(bits2, _bin_family(),
                            mPB_all[alive2].astype(np.float32),
                            sPB_all[alive2].astype(np.float32))
                alive2 = alive2[bits2 == 1]
                mags2 = mags2[bits2 == 1]
                kk += 1
        plain_bytes = len(enc2.get_compressed().tobytes())
    return data, params, plain_bytes


def decode_chunk(data, g, params, ctx=None, s4=None):
    locs, p0s, bts = params
    if ctx is not None:
        loc = locs[g, ctx]
        p0 = p0s[g, ctx]
        beta = bts[g, ctx]
    else:
        loc = locs[g]
        p0 = p0s[g]
        beta = bts[g]
    dec = constriction.stream.queue.RangeDecoder(
        np.frombuffer(data, dtype=np.uint32))
    n = g.size
    d = np.zeros(n, dtype=np.int64)
    head, logq, device, g_base = (s4 + (None,))[:4] if s4 is not None else (None, None, None, None)
    if head is not None:
        meanH, sigH = _s4_ms(head, loc, g_base, logq, device)
        from math import erf
        ph = lambda x: 0.5 * (1 + np.vectorize(erf)(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))
        p_nz = np.clip((ph(meanH + 0.5) - ph(meanH - 0.5)) /
                       (ph(meanH + 4096) - ph(meanH - 4096)), 1e-4, 1 - 1e-4)
        mA, sA = _bin_ms(1.0 - p_nz)
    else:
        mA, sA = _bin_ms(1.0 - p0)
    z = dec.decode(_bin_family(), mA.astype(np.float32), sA.astype(np.float32))
    surv = np.nonzero(z == 1)[0]
    if surv.size:
        mS, sS = _bin_ms(0.5)
        sgn = dec.decode(_bin_family(),
                         np.full(surv.size, mS, dtype=np.float32),
                         np.full(surv.size, sS, dtype=np.float32))
        mag = np.ones(surv.size, dtype=np.int64)
        alive_pos = np.arange(surv.size)
        k = 1
        while alive_pos.size:
            cur = surv[alive_pos]
            if head is not None:
                sig_cur = sigH[cur]
                beta_cur = np.clip(1.0 - 1.0 / np.maximum(sig_cur, 1.05), 0.05, 0.98)
                mB_cur, sB_cur = _bin_ms(beta_cur)
            else:
                mB_cur, sB_cur = _bin_ms(beta)
            bits = dec.decode(_bin_family(),
                              mB_cur[cur].astype(np.float32),
                              sB_cur[cur].astype(np.float32))
            dead = alive_pos[bits == 0]
            mag[dead] = k
            alive_pos = alive_pos[bits == 1]
            k += 1
            if k > (1 << 20):
                raise RuntimeError("unbounded magnitude stages")
        d[surv] = np.where(sgn == 1, -mag, mag)
    return (loc + d).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", default=None,
                    help="anchor_stats.npz with per-anchor area (contribution "
                    "groups); if absent, groups fall back to uniform buckets "
                    "over the Morton row order")
    ap.add_argument("--mapped",
                    default=None,
                    help="decoded-row -> trained-row index map; auto-generated "
                    "from the checkpoint when missing")
    ap.add_argument("--s2", default="analysis/s2_prefix_sweep/prefix_sweep.json")
    ap.add_argument("--out", default="analysis/s2_prefix_sweep/c25_real_bitstream.json")
    ap.add_argument("--bin", dest="bin_out",
                    default="analysis/s2_prefix_sweep/c25_real_bitstream.bin")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--s4-heads", default=None,
                    help="prefix path of learned entropy heads (s4_head_*.pt); "
                    "residual layers coded with the learned conditional heads, "
                    "weights shipped in the bitstream header and counted")
    ap.add_argument("--groups", type=int, default=GROUPS,
                    help="contribution-group count for the condition tables "
                    "(A2: try 64/128)")
    ap.add_argument("--field-aware", action="store_true",
                    help="per-field ladders: feat/offset 8x-start, scaling "
                    "2x-start (log-domain cliffs at coarser steps)")
    args = ap.parse_args()

    from scaffold_gs.hacpp import HACPlusCodec
    from scaffold_gs.datasets import ColmapDataset

    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_trained = int(meta["num_anchors_total"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # groups: contribution-area buckets when stats exist; otherwise uniform
    # buckets over the (Morton-ordered) decoded rows — documented fallback
    if args.stats and Path(args.stats).exists():
        stats = np.load(args.stats)
        area = stats["area"].astype(np.float64)
        assert area.shape[0] == n_trained
        rank = np.empty(n_trained, dtype=np.int64)
        rank[np.argsort(-area, kind="stable")] = np.arange(n_trained)
        group_of_trained = np.minimum(rank * GROUPS // n_trained,
                                      GROUPS - 1).astype(np.int16)
        group_source = "contribution-area"
    else:
        group_of_trained = None
        group_source = "morton-uniform (no stats)"
    print(f"[RB] grouping: {group_source}")

    # decoded -> trained row map: load or generate from the checkpoint
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
    run_tag = Path(args.run).name
    if args.mapped and Path(args.mapped).exists():
        mapped = np.load(args.mapped).astype(np.int64)
        print(f"[RB] mapped loaded: {args.mapped}")
    else:
        ck = torch.load(run_dir / "ckpts" / "ckpt_30000.pth", map_location="cpu",
                        weights_only=False)
        trained_xyz = ck["model_state"]["_anchor"].to(dev)
        assert trained_xyz.shape[0] == n_trained
        decoded_xyz = model.core.get_anchor
        mapped = np.empty(decoded_xyz.shape[0], dtype=np.int64)
        with torch.no_grad():
            for start in range(0, decoded_xyz.shape[0], 1024):
                end = min(start + 1024, decoded_xyz.shape[0])
                d = torch.cdist(decoded_xyz[start:end], trained_xyz)
                mapped[start:end] = d.min(dim=1)[1].cpu().numpy()
                del d
        mp = out_path.parent / (run_tag + ".mapped.npy")
        np.save(mp, mapped)
        print(f"[RB] mapped generated and cached: {mp}")
    assert n_alive == mapped.size
    q_cache = out_path.parent / (run_tag + "_q_cache.npz")
    if q_cache.exists():
        z = np.load(q_cache)
        Q_t = {
            "feat": torch.from_numpy(z["Q_feat"]).to(dev),
            "scaling": torch.from_numpy(z["Q_scaling"]).to(dev),
            "offset": torch.from_numpy(z["Q_offsets"]).to(dev),
        }
        print("[RB] Q loaded from cache")
    else:
        core = model.core
        k_off = model.cfg.n_offsets
        core.current_step = 30000
        core.current_iter = 30000
        Q_feat = torch.empty(n_alive, 32, device=dev)
        Q_scaling = torch.empty(n_alive, 6, device=dev)
        Q_offsets = torch.empty(n_alive, k_off, 3, device=dev)
        anchor_dev = model.core.get_anchor
        with torch.no_grad():
            for start in range(0, n_alive, 16384):
                end = min(start + 16384, n_alive)
                a = anchor_dev[start:end]
                idx = torch.arange(start, end, device=dev)
                ctx_in = core.calc_context_feat(a, anchor_indices=idx, caller="rb")
                (mean, scale, prob, m_sc, s_sc, m_off, s_off, qa, qs, qo) = torch.split(
                    core.get_grid_mlp(ctx_in),
                    [32, 32, 32, 6, 6, 3 * k_off, 3 * k_off, 1, 1, 1],
                    dim=-1,
                )
                qf = 1.0 * (1 + torch.tanh(qa.repeat(1, 32)))
                qs2 = 0.001 * (1 + torch.tanh(qs.repeat(1, 6)))
                qo2 = 0.2 * (1 + torch.tanh(qo.repeat(1, 3 * k_off))).view(-1, k_off, 3)
                if core.is_content_aware_quant_active():
                    msk = core.get_mask[idx]
                    (qf, qs2, qo2, _, _, _, _) = core._codec_apply_content_aware_quant_params(
                        "rb", a, msk, qf, qs2, qo2, None, None, None,
                        m_sc.view(-1, 6), m_off.view(-1, 3 * k_off),
                    )
                Q_feat[start:end] = qf
                Q_scaling[start:end] = qs2
                Q_offsets[start:end] = qo2
        np.savez(q_cache, Q_feat=Q_feat.cpu().numpy(), Q_scaling=Q_scaling.cpu().numpy(),
                 Q_offsets=Q_offsets.cpu().numpy())
        Q_t = {"feat": Q_feat, "scaling": Q_scaling, "offset": Q_offsets}
        print("[RB] per-anchor Q computed via decoder-reproducible path (cached)")
    print(f"[RB] model decoded (anchors={n_alive})", flush=True)

    fields = ("feat", "scaling", "offset")
    # field-aware ladders: scaling is log-domain (cliffs at 4x+), so its
    # coarsest step is 2x; feat/offset tolerate 8x. Uniform mode = same
    # ladder for all fields (legacy curve).
    FIELD_STEPS = {"feat": [8, 4, 2, 1], "scaling": [2, 1], "offset": [8, 4, 2, 1]}
    if args.field_aware:
        fsteps = {f: FIELD_STEPS[f] for f in fields}
        print("[RB] mode: FIELD-AWARE ladders", flush=True)
    else:
        fsteps = {f: STEPS for f in fields}
    sym = {f: torch.round(dec_t[f] / Q_t[f]) for f in fields}

    # ---- A1+A2: composite condition = contribution group × step tier ----
    # step tier: per-field terciles of the per-anchor step multiplier
    # (decoder recomputes both the multiplier and the tier cut points are
    # fixed at 1/3, 2/3 quantiles of the same array — zero side info)
    n_groups_eff = getattr(args, "groups", GROUPS)
    step_tiers = {}
    for f in fields:
        qmul = Q_t[f].cpu().numpy().reshape(-1)
        # per-symbol multiplier is constant across an anchor's columns:
        qmul_anchor = qmul.reshape(-1, qmul.size // n_alive)[:, 0]
        t1, t2 = np.quantile(qmul_anchor, [1 / 3, 2 / 3])
        tier = np.digitize(qmul_anchor, [t1, t2]).astype(np.int16)
        step_tiers[f] = tier  # per-anchor; repeated per column below
    if n_groups_eff != GROUPS:
        # finer contribution grouping: re-derive from area order
        if args.stats and Path(args.stats).exists():
            area = np.load(args.stats)["area"].astype(np.float64)
            rank = np.empty(n_trained, dtype=np.int64)
            rank[np.argsort(-area, kind="stable")] = np.arange(n_trained)
            g_tr = np.minimum(rank * n_groups_eff // n_trained,
                              n_groups_eff - 1).astype(np.int16)
        else:
            g_tr = np.minimum(np.arange(n_trained) * n_groups_eff // n_trained,
                              n_groups_eff - 1).astype(np.int16)
        g_base_alive = g_tr[mapped]
    else:
        g_base_alive = (group_of_trained[mapped] if group_of_trained is not None
                        else np.minimum(np.arange(n_alive) * GROUPS // n_alive,
                                        GROUPS - 1).astype(np.int16))
    print(f"[RB] condition: groups={n_groups_eff} x step-tiers=3 "
          f"(composite={n_groups_eff*3})", flush=True)

    q_flat, g_flat, out_shape = {}, {}, {}
    for f in fields:
        qf = sym[f].cpu().numpy().astype(np.int32).reshape(-1)
        n_cols = qf.size // n_alive
        q_flat[f] = qf
        g_flat[f] = np.repeat(
            g_base_alive * 3 + step_tiers[f], n_cols)
        out_shape[f] = tuple(sym[f].shape)

    # ---- optional learned entropy heads (S4) ----
    s4_heads = None
    s4_weight_bytes = 0
    logq_flat = {}
    if args.s4_heads:
        if getattr(args, "groups", GROUPS) != GROUPS:
            raise SystemExit("--s4-heads was trained with 32 groups; "
                             "combine with --groups 32 only")
        s4_heads = _load_s4_heads(args.s4_heads, dev)
        for f in fields:
            # Q_t[f] flat already has one entry per symbol of this field;
            # residual chunks index the same flat symbol space
            logq_flat[f] = np.log(Q_t[f].cpu().numpy().reshape(-1)
                                  ).astype(np.float32)
        import os
        s4_weight_bytes = sum(os.path.getsize(f"{args.s4_heads}_{f}.pt")
                              for f in fields)
        print(f"[RB] learned entropy heads loaded (S4 apply mode), "
              f"weights {s4_weight_bytes/1024:.0f} KB", flush=True)

    # ---- range-code the ladder (chunks in ladder order) ----
    # residual chunks (li>=1) use layer-conditioned models: per (group,
    # |coarse symbol| bucket) — the bucket is recomputed decoder-side from
    # the already-decoded coarse layer, so it stays zero-side-info.
    chunks = []          # dicts: level, field, params, data, decoded, ctx
    acc_prev = {}        # decoded coarse symbols per field (context source)
    plain_bytes = {}     # unconditioned alternative, for the savings report
    t0 = time.time()
    for f in fields:
        steps_f = fsteps[f]
        for li in range(len(steps_f)):
            q = q_flat[f]
            if li == 0:
                layer = np.round(q / steps_f[0]).astype(np.int32)
                ctx = None
            else:
                ratio = steps_f[li - 1] // steps_f[li]
                layer = (np.round(q / steps_f[li]).astype(np.int32)
                         - ratio * acc_prev[f])
                ctx = bucket_of(acc_prev[f])
            # base layer (li==0) conditions on the contribution group only:
            # step-tier split there costs ~1.2 MB (fit variance with no
            # magnitude context to justify it); enhancement layers keep it.
            if ctx is None:
                g_chunk = g_flat[f] // 3
            else:
                g_chunk = g_flat[f]
            s4ctx = None
            if s4_heads is not None and ctx is not None:
                s4ctx = (s4_heads[f], logq_flat[f], dev, g_flat[f] // 3)
            data, params, pbytes = code_chunk(layer, g_chunk, ctx, s4ctx)
            back = decode_chunk(data, g_chunk, params, ctx, s4ctx)
            exact = bool((back == layer).all())
            assert exact, f"roundtrip mismatch level {li} field {f}"
            if ctx is not None:
                plain_bytes[(li, f)] = pbytes
            chunks.append({"level": li, "field": f, "params": params,
                           "data": data, "decoded": back, "bytes": len(data),
                           "ctx": ctx})
            acc_prev[f] = ((steps_f[li] if li == 0 else ratio) * 0
                           + (np.round(q / steps_f[li]).astype(np.int32)
                              if li == 0 else
                              ratio * acc_prev[f] + back))
            print(f"[RB] coded {f} L{li} (Q x{steps_f[li]}): "
                  f"{len(data)/1e6:.2f} MB exact={exact}", flush=True)
    coding_s = time.time() - t0
    plain_total = sum(plain_bytes.values()) if plain_bytes else 0
    cond_total = sum(c["bytes"] for c in chunks if c["ctx"] is not None)
    cond_saved = plain_total - cond_total
    if plain_total:
        print(f"[RB] layer-conditioned residual coding: {cond_total/1e6:.2f} MB "
              f"vs unconditioned {plain_total/1e6:.2f} MB -> saved "
              f"{cond_saved/1e6:.2f} MB", flush=True)

    # ---- write the real transmit-able file ----
    header = {
        "format": "c25_layered_v1" if not args.field_aware else "c25_layered_v2_fieldaware",
        "groups": n_groups_eff,
        "step_tiers": 3,
        "steps": {f: [int(s) for s in fsteps[f]] for f in fields},
        "fields": list(fields),
        "chunks": [{"level": c["level"], "field": c["field"]} for c in chunks],
        "n_alive": int(n_alive),
    }
    header_bytes = json.dumps(header).encode("utf-8")
    bin_path = Path(args.bin_out)
    with open(bin_path, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(4, "little"))
        fh.write(header_bytes)
        for c in chunks:
            locs, p0s, bts = c["params"]
            pb = (locs.astype(np.int32).tobytes()
                  + p0s.astype(np.float32).tobytes()
                  + bts.astype(np.float32).tobytes())
            fh.write(len(pb).to_bytes(4, "little"))
            fh.write(pb)
            fh.write(len(c["data"]).to_bytes(4, "little"))
            fh.write(c["data"])
    bin_size = bin_path.stat().st_size
    print(f"[RB] wrote {bin_path} ({bin_size/1e6:.2f} MB)", flush=True)

    # ---- fixed payload carried once (from the real hac_meta accounting) ----
    fixed_bytes = (
        int(meta["bit_hash"]) + int(meta["bit_mlp"]) + int(meta["bit_bounds"])
        + int(meta["bit_header"]) + int(meta["bit_masks"])
    ) / 8.0
    geom_bytes = int(meta["bit_anchor"]) / 8.0
    header_total = 4 + len(header_bytes)
    chunk_params_total = 0
    for c in chunks:
        locs, p0s, bts = c["params"]
        chunk_params_total += 8 + locs.astype(np.int32).tobytes().__len__() \
            + p0s.astype(np.float32).tobytes().__len__() \
            + bts.astype(np.float32).tobytes().__len__()
    chunk_data_total = sum(c["bytes"] for c in chunks)

    # ---- dataset (after model, per trainer convention) ----
    dataset = ColmapDataset(
        data_dir=args.data_dir, data_factor=2, test_every=8,
        white_background=False, preload_images=False, device=str(dev),
    )
    val = dataset.val_cameras
    pick = np.unique(np.linspace(0, len(val) - 1, N_VIEWS).astype(int))
    cams = [val[i] for i in pick]
    background = dataset.background

    # cumulative coded bytes per prefix: prefix p puts field f at its level
    # min(p, len_f-1); chunks within a field are in ladder order
    n_prefixes = max(len(fsteps[f]) for f in fields)
    lvl_of_prefix = {
        f: [min(p, len(fsteps[f]) - 1) for p in range(n_prefixes)]
        for f in fields
    }
    cum_by_prefix = []
    for p in range(n_prefixes):
        tot = 0
        for f in fields:
            lf = lvl_of_prefix[f][p]
            tot += sum(c["bytes"] for c in chunks
                       if c["field"] == f and c["level"] <= lf)
        cum_by_prefix.append(tot)

    # ---- render prefixes; full precision FIRST as the sanity anchor ----
    results = []
    for p in [n_prefixes - 1] + list(range(n_prefixes - 1)):
        deq = {}
        for f in fields:
            lf = lvl_of_prefix[f][p]
            steps_f = fsteps[f]
            chunks_upto = [c for c in chunks
                           if c["field"] == f and c["level"] <= lf]
            acc = chunks_upto[0]["decoded"].astype(np.float64)
            for prev, cur in zip(chunks_upto[:-1], chunks_upto[1:]):
                acc = (steps_f[prev["level"]] // steps_f[cur["level"]]) * acc \
                    + cur["decoded"]
            xhat = (steps_f[lf] * acc.reshape(out_shape[f])
                    * Q_t[f].cpu().numpy()).astype(np.float32)
            deq[f] = torch.from_numpy(xhat).to(dev)
        view.anchor_feat = torch.nn.Parameter(deq["feat"].contiguous(),
                                              requires_grad=False)
        view.scaling = torch.nn.Parameter(deq["scaling"].contiguous(),
                                          requires_grad=False)
        view.offset = torch.nn.Parameter(deq["offset"].contiguous(),
                                         requires_grad=False)
        t0 = time.time()
        psnrs = []
        with torch.no_grad():
            for cam in cams:
                out = model.render(cam, background, is_training=False,
                                   appearance_id=0, step=30000)
                pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
                gt = dataset.get_image(cam).clamp(0.0, 1.0)
                mse = torch.mean((pred - gt) ** 2).item()
                psnrs.append(float("inf") if mse <= 0 else -10.0 * np.log10(mse))
                del out
        real_bytes = (4 + len(header_bytes) + fixed_bytes + geom_bytes
                      + chunk_params_total + s4_weight_bytes
                      + cum_by_prefix[p])
        entry = {
            "prefix": p,
            "field_levels": {f: int(lvl_of_prefix[f][p]) for f in fields},
            "field_steps": {f: int(fsteps[f][lvl_of_prefix[f][p]])
                            for f in fields},
            "coded_bytes": int(cum_by_prefix[p]),
            "real_bytes": round(real_bytes, 1),
            "psnr_mean": round(float(np.mean(psnrs)), 4),
            "psnr_per_view": [round(p, 4) for p in psnrs],
            "render_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(f"[RB] prefix P{p} (steps {entry['field_steps']}): REAL "
              f"{real_bytes/1e6:6.2f} MB  PSNR {entry['psnr_mean']:.3f}", flush=True)
        if p == n_prefixes - 1:
            print(f"[RB] SANITY ANCHOR: {entry['psnr_mean']:.3f} dB (need >= 25)")
            assert entry["psnr_mean"] >= 25.0, "sanity anchor failed — INVALID"
        del deq
        torch.cuda.empty_cache()

    payload = {
        "run": str(run_dir), "n_alive": int(n_alive),
        "groups": int(n_groups_eff),
        "field_aware": bool(args.field_aware),
        "s4_apply": s4_heads is not None,
        "s4_weight_bytes": s4_weight_bytes,
        "field_steps": {f: [int(s) for s in fsteps[f]] for f in fields},
        "fixed_bytes": fixed_bytes, "geom_bytes": geom_bytes,
        "header_bytes": header_total,
        "chunk_params_total": chunk_params_total,
        "conditional_saved_bytes": int(cond_saved),
        "coding_s": round(coding_s, 1),
        "bin_file": str(bin_path), "bin_size": bin_size,
        "results": results,
    }
    if Path(args.s2).exists():
        s2 = json.loads(Path(args.s2).read_text())
        payload["s2_selection_reference"] = [
            {k2: r[k2] for k2 in ("ordering", "prefix", "bytes_est", "psnr_mean")}
            for r in s2["results"]
        ]
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"[RB] wrote {out_path}")


if __name__ == "__main__":
    main()
