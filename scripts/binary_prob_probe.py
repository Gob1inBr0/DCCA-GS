#!/usr/bin/env python3
"""Binary-probability probe v3 — ONE magnitude decision, minimal verified
loop, isolated from the production script on purpose.

Symbols: q = round(t / Q) (int32, exact); base layer coarse = round(q/8);
enhancement residual e = q - 8*coarse (int32, exact, |e| <= 4).

Decision set (stage s >= 0): population = symbols with |e| >= s (the decoder
knows this from earlier stages); decision bit = (|e| >= s + 1).
s = 0 is the needs-refinement flag itself (over all symbols), s >= 1 are the
unary magnitude-continuation bits.

Two probability models over the SAME decisions:
  static : per-group continuation rate, fitted on the train split only
  learned: tiny MLP on decoder-side continuous context (|coarse|, log Q,
           stage; group embedding optional via --group-emb, off by default)

Evaluation: anchor-level 90/10 split (split first, then use decisions — no
leakage between columns of the same anchor); on the val split, theory bits
(BCE at the coder's effective p) and real arithmetic-coder bits (roundtrip
asserted) for static and learned.

Usage:
  PYTHONPATH=/home/project2/c25_pylib2 python scripts/binary_prob_probe.py \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --field offset --stage 0 --groups 32 \
      --mapped analysis/s2_prefix_sweep/ph0c_base_r085_lam0005_s42.mapped.npy \
      --out analysis/s2_prefix_sweep/probe_offset_s0.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import constriction  # noqa: E402  (PYTHONPATH=/home/project2/c25_pylib2)


def _bin_ms(p):
    """(mean, sigma, p_eff) of discretized Gaussian on {0,1}.

    p_eff is the P(1) the coder actually implements after clamping — theory
    bits must be computed at p_eff, not at the raw p, or the G1 gap is fake.
    """
    pt = torch.clamp(torch.as_tensor(np.asarray(p, dtype=np.float64)),
                     1e-4, 1 - 1e-4)
    mean = torch.round(pt)
    z = torch.erfinv(2.0 * torch.abs(pt - mean) - 1.0).abs() * np.sqrt(2.0)
    sigma = torch.clamp(0.5 / torch.clamp(z, min=1e-3), 0.05, 512.0)
    p_eff = 1.0 - 0.5 * (1.0 + torch.erf((0.5 - mean.double()) /
                                          (sigma.double() * np.sqrt(2.0))))
    return (mean.numpy().astype(np.float32), sigma.numpy().astype(np.float32),
            p_eff.numpy().astype(np.float64))


def _bin_family():
    return constriction.stream.model.QuantizedGaussian(0, 1)


def code_bits_real(bits01, p_ones):
    """REAL arithmetic coding of 0/1 decisions.

    Returns (bits, decoded, p_eff) where p_eff is the probability actually
    implemented by the coder (use it for theory bits).
    """
    m, s, p_eff = _bin_ms(p_ones)
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(bits01.astype(np.int32), _bin_family(), m, s)
    comp = enc.get_compressed()
    dec = constriction.stream.queue.RangeDecoder(comp)
    back = dec.decode(_bin_family(), m, s).astype(np.int64)
    return comp.size * 32, back, p_eff


def bce_bits(bits01, p):
    """Theoretical code length: -sum[y log2 p + (1-y) log2 (1-p)]."""
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    y = bits01.astype(np.float64)
    return float(-np.sum(y * np.log2(pc) + (1 - y) * np.log2(1 - pc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--field", default="offset",
                    choices=["feat", "scaling", "offset"])
    ap.add_argument("--stage", type=int, default=0,
                    help="decision stage s: population |e|>=s, bit=|e|>=s+1; "
                         "0 = needs-refinement flag")
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--group-emb", action="store_true",
                    help="add group embedding to the MLP (first round: off)")
    ap.add_argument("--mapped", default=None,
                    help="decoded->trained row map; default "
                         "<out dir>/<run tag>.mapped.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_tag = Path(args.run).name
    assert args.stage >= 0

    from scaffold_gs.hacpp import HACPlusCodec

    bit_dir = Path(args.run) / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())

    # ---- decode model (ground-truth symbols) ----
    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    del codec
    n_alive = model.num_anchors
    dev = model.device
    t = {"feat": model._view.anchor_feat.data,
         "scaling": model._view.scaling.data,
         "offset": model._view.offset.data}[args.field].detach().float()

    # ---- per-anchor step multiplier: cache or error ----
    q_cache = out_path.parent / (run_tag + "_q_cache.npz")
    if not q_cache.exists():
        raise SystemExit(f"Q cache missing: {q_cache} — run "
                         "c25_real_bitstream.py once to create it")
    z = np.load(q_cache)
    qmul = z[{"feat": "Q_feat", "scaling": "Q_scaling",
              "offset": "Q_offsets"}[args.field]]

    # ---- decoded -> trained row mapping (existence check only; symbols are
    # taken from the decoded model itself, so rows already align) ----
    mapped_p = (Path(args.mapped) if args.mapped else
                out_path.parent / (run_tag + ".mapped.npy"))
    mapped = np.load(mapped_p).astype(np.int64)
    assert mapped.size == n_alive

    # ---- GATE 0: exact int32 enhancement residuals over the 8x base layer ----
    # qmul may be stored 3-D (e.g. offsets (n,10,3)) while the model view is
    # flattened: align by numel; both sides flatten row-major, so symbol
    # order is identical
    assert qmul.size == t.numel(), f"qmul {qmul.shape} vs t {t.shape}"
    Qf = torch.from_numpy(qmul).to(dev).float().reshape(t.shape)
    q = torch.round(t / Qf).to(torch.int32).reshape(-1)   # final symbols
    coarse = torch.round(q.float() / 8).to(torch.int32)   # base layer (8x)
    e = (q - 8 * coarse).to(torch.int32)                  # residual, |e|<=4
    assert torch.equal(8 * coarse + e, q)
    r_np = e.cpu().numpy()
    print(f"[probe] G0 residuals: n={r_np.size} min={r_np.min()} "
          f"max={r_np.max()} absmax={np.abs(r_np).max()} "
          f"|r|>127={int((np.abs(r_np) > 127).sum())} "
          f"|r|>32767={int((np.abs(r_np) > 32767).sum())}", flush=True)

    # groups by decoded-row uniform buckets (documented fallback)
    g_anchor = np.minimum(np.arange(n_alive) * args.groups // n_alive,
                          args.groups - 1).astype(np.int32)

    # ---- stage decision ----
    absr_np = np.abs(r_np).astype(np.int64)
    alive_idx = np.nonzero(absr_np >= args.stage)[0]      # stage population
    bit = (absr_np[alive_idx] >= args.stage + 1).astype(np.int64)
    # symbol -> anchor: fields are (n_alive, cols) flattened row-major
    cols = r_np.size // n_alive
    anchor_of_sym = np.arange(r_np.size) // cols
    g_dec = g_anchor[anchor_of_sym[alive_idx]]
    print(f"[probe] stage {args.stage}: decisions={bit.size} "
          f"positives={int(bit.sum())}", flush=True)

    # per-symbol decoder-side context: |coarse| (base layer) and log Q
    ctx_np = np.abs(coarse.cpu().numpy()).astype(np.float32)
    lq_np = np.log(np.clip(qmul.reshape(-1).astype(np.float32),
                           1e-9, None))

    # ---- anchor-level 90/10 split (split FIRST, then use decisions) ----
    # all symbols of one anchor go to the same side: no spatial leakage
    # between train and val through neighboring columns of the same anchor
    rng = np.random.default_rng(0)
    aperm = rng.permutation(n_alive)
    val_anchors = np.sort(aperm[:n_alive // 10])
    is_val_anchor = np.zeros(n_alive, dtype=bool)
    is_val_anchor[val_anchors] = True
    sym_is_val = is_val_anchor[anchor_of_sym[alive_idx]]
    val_sel = np.nonzero(sym_is_val)[0]
    tr_sel = np.nonzero(~sym_is_val)[0]
    val_bit, tr_bit = bit[val_sel], bit[tr_sel]
    val_g, tr_g = g_dec[val_sel], g_dec[tr_sel]
    print(f"[probe] split (anchor-level): train={tr_bit.size} "
          f"val={val_bit.size}", flush=True)

    # ---- static model: per-group rate (train-split fit, global fallback) ----
    global_rate = min(max(float(tr_bit.mean()), 1e-4), 1 - 1e-4)
    p_stat_val = np.full(val_bit.size, global_rate, dtype=np.float64)
    for gid in range(args.groups):
        m_tr, m_va = tr_g == gid, val_g == gid
        if m_tr.sum() == 0 or m_va.sum() == 0:
            continue
        rate = float(tr_bit[m_tr].mean())
        p_stat_val[m_va] = min(max(rate, 1e-4), 1 - 1e-4)

    # ---- learned model: tiny MLP on continuous context ----
    # first round: |coarse|, logQ, stage ONLY — no group embedding, so the
    # comparison against the per-group static table is about continuous
    # context, not about memorizing group ids
    class ContHead(nn.Module):
        """|coarse|, logQ, stage (+optional group emb) -> logit P(continue)."""

        def __init__(self, groups, use_emb=False, emb=8, hidden=32):
            super().__init__()
            self.use_emb = use_emb
            if use_emb:
                self.emb = nn.Embedding(groups, emb)
            self.net = nn.Sequential(
                nn.Linear(3 + (emb if use_emb else 0), hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1))

        def forward(self, abs_coarse, logq, group, stage):
            feats = [abs_coarse[:, None], logq[:, None],
                     float(stage) * torch.ones_like(abs_coarse[:, None])]
            if self.use_emb:
                feats.append(self.emb(group))
            return self.net(torch.cat(feats, dim=-1)).squeeze(-1)

    head = ContHead(args.groups, use_emb=args.group_emb).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    ctx_t = torch.from_numpy(ctx_np).to(dev)
    lq_t = torch.from_numpy(lq_np).to(dev)
    absr_t = torch.from_numpy(absr_np.astype(np.float32)).to(dev)
    gt_anchor = torch.from_numpy(g_anchor.astype(np.int64)).to(dev)
    g_sym_t = gt_anchor[torch.from_numpy(anchor_of_sym).to(dev)]
    tr_t = torch.from_numpy(alive_idx[tr_sel].astype(np.int64)).to(dev)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    # train on the train split ONLY (val symbols never seen — no leakage)
    for epoch in range(4):
        perm_t = tr_t[torch.randperm(tr_t.numel())]
        tot = 0.0
        for s0 in range(0, perm_t.numel(), 131072):
            b = perm_t[s0:s0 + 131072]
            logits = head(ctx_t[b], lq_t[b], g_sym_t[b], args.stage)
            loss = bce(logits,
                       (absr_t[b] >= args.stage + 1).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * b.numel()
        print(f"[probe] mlp epoch {epoch}: train bits/sym "
              f"{tot / np.log(2) / max(tr_t.numel(), 1):.4f}", flush=True)

    # ---- MLP evaluation on val decisions: theory + real coder ----
    head.eval()
    with torch.no_grad():
        logits_val = head(
            torch.from_numpy(ctx_np[alive_idx][val_sel]).to(dev),
            torch.from_numpy(lq_np[alive_idx][val_sel]).to(dev),
            torch.from_numpy(g_dec[val_sel].astype(np.int64)).to(dev),
            args.stage)
    p_mlp_val = torch.sigmoid(logits_val).cpu().numpy().astype(np.float64)

    bits_mlp_real, back_m, p_eff_m = code_bits_real(val_bit, p_mlp_val)
    assert (back_m == val_bit).all(), "mlp roundtrip failed"
    bits_mlp_theory = bce_bits(val_bit, p_eff_m)

    bits_static_real, back_s, p_eff_s = code_bits_real(val_bit, p_stat_val)
    assert (back_s == val_bit).all(), "static roundtrip failed"
    bits_static_theory = bce_bits(val_bit, p_eff_s)

    g1_static = abs(bits_static_real - bits_static_theory) / \
        max(bits_static_theory, 1)
    g1_mlp = abs(bits_mlp_real - bits_mlp_theory) / max(bits_mlp_theory, 1)
    gate2 = 100 * (1 - bits_mlp_real / max(bits_static_real, 1))
    weight_bytes = sum(pp.numel() * 4 for pp in head.parameters())
    net_saving = 100 * (1 - (bits_mlp_real + weight_bytes * 8) /
                        max(bits_static_real, 1))

    print(f"[probe] G1 static: theory {bits_static_theory/1e6:.3f} Mbit, "
          f"real {bits_static_real/8/1e6:.3f} MB, gap {g1_static:.2%}",
          flush=True)
    print(f"[probe] G1 mlp:    theory {bits_mlp_theory/1e6:.3f} Mbit, "
          f"real {bits_mlp_real/8/1e6:.3f} MB, gap {g1_mlp:.2%}", flush=True)
    verdict = "PASS" if gate2 >= 5 else "FAIL"
    print(f"[probe] G2 val payload: mlp {bits_mlp_real/8/1e6:.3f} MB vs "
          f"static {bits_static_real/8/1e6:.3f} MB -> {gate2:.1f}% "
          f"({verdict}, gate >= 5%)", flush=True)
    print(f"[probe] G3 net (with {weight_bytes/1024:.0f}KB weights): "
          f"{net_saving:.1f}%", flush=True)

    out_path.write_text(json.dumps({
        "field": args.field, "stage": args.stage, "groups": args.groups,
        "group_emb": bool(args.group_emb), "split": "anchor_90_10",
        "n_decisions_val": int(val_bit.size),
        "positives_val": int(val_bit.sum()),
        "bits_static_theory_val": bits_static_theory,
        "bits_mlp_theory_val": bits_mlp_theory,
        "bits_static_real_val": bits_static_real,
        "bits_mlp_real_val": bits_mlp_real,
        "weight_bytes": weight_bytes,
        "g1_static_gap": round(g1_static, 5),
        "g1_mlp_gap": round(g1_mlp, 5),
        "gate2_saving_pct": round(gate2, 2),
        "gate3_net_saving_pct": round(net_saving, 2),
        "verdict": verdict,
    }, indent=1))
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
