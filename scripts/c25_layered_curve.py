#!/usr/bin/env python3
"""S3 step 1 (C2.5, zero-training): layered attribute re-encoding curve.

Builds a nested quality ladder over the three quantized fields (feat /
scaling / offsets) of an already-encoded run:

  L0:  round(x / (4*Q))            coarse symbols for ALL anchors
  L1:  residual to round(x / (2*Q))  in {-1,0,1}
  L2:  residual to full Q           in {-1,0,1}

Ground truth = the DECODED MODEL's own attribute tensors (codec.decode):
attributes.pth on disk predates decode-side bound clamping and differs on a
few rows by huge amounts (scaling up to ~21 in log space), so disk values
must never be rendered. Q steps come from the decoder-reproducible path
(cached). Entropy accounting: per-group discrete-Laplace (32 contribution
groups, fitted per layer — slight optimism, disclosed, params counted).
Masks/geometry/MLP/header bytes carried once from hac_meta.

Levels are DEQUANTIZED and RENDERED on 16 fixed views. The full-precision
level renders the decoded model's own tensors verbatim and is asserted to
reproduce known quality (>= 25 dB) before anything else is trusted.

Usage (server):
  python scripts/c25_layered_curve.py \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --data-dir /dev/shm/dcca_data/1-78/data \
      --stats analysis/anchor_stats/ph0c_base_r085_lam0005_s42/anchor_stats.npz \
      --mapped analysis/s2_prefix_sweep/prefix_sweep.mapped.npy \
      --s2 analysis/s2_prefix_sweep/prefix_sweep.json \
      --out analysis/s2_prefix_sweep/c25_layered_curve.json
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

N_VIEWS = 16
GROUPS = 32
FACTORS = [4, 2]  # coarse -> fine ladder (powers of two)
LN2 = float(np.log(2.0))


def laplace_bits(symbols, loc, b):
    """Total bits for integer `symbols` under a discrete Laplace
    P(x) = (1-b)/(1+b) * b^|x-loc| (geometric on |x-loc|); never negative."""
    scale = max(float(b), 1e-3)
    beta = min(max(float(np.exp(-1.0 / scale)), 1e-12), 1.0 - 1e-12)
    x = np.abs(np.asarray(symbols, dtype=np.float64) - loc)
    return float(np.sum(np.log2((1.0 + beta) / (1.0 - beta)) + x * np.log2(1.0 / beta)))


def fit_laplace(symbols):
    s = np.asarray(symbols, dtype=np.float64)
    loc = float(np.median(s))
    m = float(np.mean(np.abs(s - loc)))
    if m <= 1e-9:
        return loc, 1e-3
    beta = (np.sqrt(1.0 + 4.0 * m * m) - 1.0) / (2.0 * m)
    b = -1.0 / np.log(max(min(float(beta), 1.0 - 1e-9), 1e-12))
    return loc, max(b, 1e-3)


def group_bits(layer, g, n_groups):
    bits = 0.0
    for gid in range(n_groups):
        sel = layer[g == gid]
        if sel.size == 0:
            continue
        loc, b = fit_laplace(sel)
        bits += laplace_bits(sel, loc, b)
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument(
        "--mapped",
        default="analysis/s2_prefix_sweep/prefix_sweep.mapped.npy",
        help="decoded-row -> trained-row index map (encoder Morton-sorts anchors)",
    )
    ap.add_argument("--s2", default="analysis/s2_prefix_sweep/prefix_sweep.json")
    ap.add_argument("--out", default="analysis/s2_prefix_sweep/c25_layered_curve.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from scaffold_gs.hacpp import HACPlusCodec
    from scaffold_gs.datasets import ColmapDataset

    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_trained = int(meta["num_anchors_total"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = np.load(args.stats)
    area = stats["area"].astype(np.float64)
    assert area.shape[0] == n_trained
    order = np.argsort(-area, kind="stable")
    rank = np.empty(n_trained, dtype=np.int64)
    rank[order] = np.arange(n_trained)
    group_of = np.minimum(rank * GROUPS // n_trained, GROUPS - 1).astype(np.int16)

    mapped = np.load(args.mapped).astype(np.int64)  # decoded j -> trained row
    print(f"[C25] decoded rows {mapped.size} mapped into trained rows {n_trained}")

    # ---- decode ALWAYS: the model's own tensors are the ground truth ----
    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    del codec
    n_alive = model.num_anchors
    dev = model.device
    assert n_alive == mapped.size, f"{n_alive} vs mapped {mapped.size}"
    view = model._view
    dec_feat = view.anchor_feat.data.detach().float()
    dec_scaling = view.scaling.data.detach().float()
    dec_offset = view.offset.data.detach().float()
    print(f"[C25] decoded anchors={n_alive} (ground truth = model tensors)")

    # ---- exact per-anchor Q via the decoder-reproducible path (cached) ----
    q_cache = out_path.parent / "c25_q_cache.npz"
    if q_cache.exists():
        z = np.load(q_cache)
        assert z["Q_feat"].shape[0] == n_alive
        Q_feat = torch.from_numpy(z["Q_feat"]).to(dev)
        Q_scaling = torch.from_numpy(z["Q_scaling"]).to(dev)
        Q_offsets = torch.from_numpy(z["Q_offsets"]).to(dev)
        print(f"[C25] per-anchor Q loaded from cache {q_cache}")
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
                ctx = core.calc_context_feat(a, anchor_indices=idx, caller="c25")
                (mean, scale, prob, m_sc, s_sc, m_off, s_off, qa, qs, qo) = torch.split(
                    core.get_grid_mlp(ctx),
                    [32, 32, 32, 6, 6, 3 * k_off, 3 * k_off, 1, 1, 1],
                    dim=-1,
                )
                qf = 1.0 * (1 + torch.tanh(qa.repeat(1, 32)))
                qs2 = 0.001 * (1 + torch.tanh(qs.repeat(1, 6)))
                qo2 = 0.2 * (1 + torch.tanh(qo.repeat(1, 3 * k_off))).view(-1, k_off, 3)
                if core.is_content_aware_quant_active():
                    msk = core.get_mask[idx]
                    (qf, qs2, qo2, _, _, _, _) = core._codec_apply_content_aware_quant_params(
                        "c25", a, msk, qf, qs2, qo2, None, None, None,
                        m_sc.view(-1, 6), m_off.view(-1, 3 * k_off),
                    )
                Q_feat[start:end] = qf
                Q_scaling[start:end] = qs2
                Q_offsets[start:end] = qo2
        np.savez(q_cache, Q_feat=Q_feat.cpu().numpy(), Q_scaling=Q_scaling.cpu().numpy(),
                 Q_offsets=Q_offsets.cpu().numpy())
        print("[C25] per-anchor Q computed via decoder-reproducible path (cached)")

    # ---- symbols from the MODEL's own values ----
    fields = ("feat", "scaling", "offsets")
    dec_t = {"feat": dec_feat, "scaling": dec_scaling, "offsets": dec_offset}
    Q_t = {"feat": Q_feat, "scaling": Q_scaling, "offsets": Q_offsets}
    sym = {f: torch.round(dec_t[f] / Q_t[f]) for f in fields}
    # off-grid audit: decoded values should sit on the Q grid; clamped rows may
    # deviate — count, don't fail (tiny fraction expected)
    off_grid = {}
    for f in fields:
        recon = Q_t[f] * sym[f]
        bad = ((recon - dec_t[f]).abs() > 0.51 * Q_t[f]).float().mean().item()
        off_grid[f] = round(bad, 6)
    print(f"[C25] off-grid fraction per field: {off_grid}")

    q_flat, g_flat = {}, {}
    for f in fields:
        qf = sym[f].cpu().numpy().astype(np.int32).reshape(-1)
        n_cols = qf.size // n_alive
        q_flat[f] = qf
        g_flat[f] = np.repeat(group_of[mapped], n_cols)
        assert Q_t[f].numel() == qf.size

    # ---- nested ladder entropy accounting ----
    fixed_bytes = (
        int(meta["bit_hash"]) + int(meta["bit_mlp"]) + int(meta["bit_bounds"])
        + int(meta["bit_header"]) + int(meta["bit_masks"])
    ) / 8.0
    geom_bytes = int(meta["bit_anchor"]) / 8.0
    n_levels = len(FACTORS) + 1
    steps = FACTORS + [1]
    layer_bits = {f: [0.0] * n_levels for f in fields}
    t0 = time.time()
    for f in fields:
        q = q_flat[f]
        g = g_flat[f]
        sym0 = np.round(q / steps[0]).astype(np.int32)
        layer_bits[f][0] = group_bits(sym0, g, GROUPS)
        for li in range(1, n_levels):
            sym_prev = np.round(q / steps[li - 1]).astype(np.int32)
            sym_cur = np.round(q / steps[li]).astype(np.int32)
            step = steps[li - 1] // steps[li]
            resid = sym_cur - step * sym_prev
            layer_bits[f][li] = group_bits(resid, g, GROUPS)
        del sym0
        print(f"[C25] {f}: " + " ".join(
            f"L{i}={layer_bits[f][i]/8/1e6:.2f}MB" for i in range(n_levels)
        ))
    print(f"[C25] entropy accounting done in {time.time()-t0:.0f}s")

    # ---- dataset (after model, per trainer convention) ----
    dataset = ColmapDataset(
        data_dir=args.data_dir, data_factor=2, test_every=8,
        white_background=False, preload_images=False, device=str(dev),
    )
    val = dataset.val_cameras
    pick = np.unique(np.linspace(0, len(val) - 1, N_VIEWS).astype(int))
    cams = [val[i] for i in pick]
    background = dataset.background

    # ---- render levels; full precision FIRST as the sanity anchor ----
    results = []
    order_levels = [n_levels - 1] + list(range(n_levels - 1))
    for li in order_levels:
        if li == n_levels - 1:
            deq = dec_t  # verbatim decoded tensors
        else:
            k = steps[li]
            deq = {}
            for f in fields:
                deq[f] = k * Q_t[f] * torch.round(sym[f] / k)
        view.anchor_feat = torch.nn.Parameter(
            deq["feat"].contiguous(), requires_grad=False
        )
        view.scaling = torch.nn.Parameter(
            deq["scaling"].contiguous(), requires_grad=False
        )
        view.offset = torch.nn.Parameter(
            deq["offsets"].contiguous(), requires_grad=False
        )
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
        cum_bits = sum(layer_bits[f][j] for f in fields for j in range(li + 1))
        total_bytes = fixed_bytes + geom_bytes + cum_bits / 8.0
        entry = {
            "level": li,
            "step_factor": int(steps[li]),
            "bytes_est": round(total_bytes, 1),
            "psnr_mean": round(float(np.mean(psnrs)), 4),
            "psnr_per_view": [round(p, 4) for p in psnrs],
            "render_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(
            f"[C25] level {li} (Q x{entry['step_factor']}): "
            f"{total_bytes/1e6:6.2f} MB  PSNR {entry['psnr_mean']:.3f}"
        )
        if li == n_levels - 1:
            print(f"[C25] SANITY ANCHOR: {entry['psnr_mean']:.3f} dB (need >= 25)")
            assert entry["psnr_mean"] >= 25.0, (
                "full-precision sanity anchor failed — results INVALID"
            )
        del deq
        torch.cuda.empty_cache()

    payload = {
        "run": str(run_dir), "stats": args.stats, "n_trained": n_trained,
        "n_alive": n_alive, "groups": GROUPS, "factors": FACTORS,
        "fixed_bytes": fixed_bytes, "geom_bytes": geom_bytes,
        "off_grid_fraction": off_grid,
        "results": results,
    }
    if Path(args.s2).exists():
        s2 = json.loads(Path(args.s2).read_text())
        payload["s2_selection_reference"] = [
            {k2: r[k2] for k2 in ("ordering", "prefix", "bytes_est", "psnr_mean")}
            for r in s2["results"]
        ]
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"[C25] wrote {out_path}")


if __name__ == "__main__":
    main()
