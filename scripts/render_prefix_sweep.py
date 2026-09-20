#!/usr/bin/env python3
"""S2: bytes-vs-PSNR prefix rendering sweep (progressive-stream design doc, stage S2).

Renders the scene using only the first k% of anchors under a given ordering,
for k in {5,10,25,50,100}, and reports PSNR per prefix — the measured
bytes-quality curve that validates the offline coverage surrogate (S1).

Also reports the A3 gate statistics from real render coverage: how many
16px tiles each anchor's gaussians cover (blocks-per-anchor) and how many
distinct anchors cover each tile (anchors-per-block). The design doc sets
the gate at ">= 5 blocks/anchor with high multiplicity" before a greedy
order on real edges is worth running.

Run AFTER the r3 queue frees a whitelisted card (GPU-light: ~10 min).
Usage (server, env /home/project2/miniconda3/envs/DCCA/bin/python):

  python scripts/render_prefix_sweep.py \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --data-dir /dev/shm/dcca_data/1-78/data \
      --stats analysis/anchor_stats/ph0c_base_r085_lam0005_s42/anchor_stats.npz \
      --bits analysis/anchor_stats/ph0c_base_r085_lam0005_s42/byte_account/per_anchor_bits.npz \
      --out analysis/s2_prefix_sweep/prefix_sweep.json

Implementation notes:
- Model rebuild mirrors eval_decoded.py: HACPlusCodec().decode(bitstreams)
  loads the decoded attributes; the decoded_version path in
  generate_gaussians uses them as-is (no re-quantization), so rendering a
  subset of anchors is exact, not an approximation of the full render.
- Prefix = subset of anchor rows. The renderer's frustum prefilter is
  intersected with the prefix keep-mask, and gaussians are generated ONLY
  for kept anchors (design doc D2 pre-registered strategy).
- Orderings: contribution-desc (anchor_stats area), random (seed 0).
- Bytes axis: fixed overhead (hash+mlp+bounds+header, from hac_meta.json)
  + geometry scaled by k + per-anchor attribute bits summed over kept
  anchors (from per_anchor_bits.npz). Entropy re-coding of a prefix subset
  would differ slightly; this is the linear accounting the design doc
  prescribes for the curve's x-axis.
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

ORDERINGS = ["contribution", "random"]
PREFIXES = [0.05, 0.10, 0.25, 0.50, 1.00]
N_VIEWS = 16  # fixed camera subset for comparability across prefixes
TILE = 16     # pixel-block size for the A3 overlap statistics
HIST_CAP = 32  # histogram bins 1..HIST_CAP, then ">HIST_CAP"


def hist_counts(values, cap=HIST_CAP):
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return {str(i): 0 for i in range(1, cap + 1)} | {f">{cap}": 0}
    bins = np.bincount(np.minimum(values, cap + 1), minlength=cap + 2)
    out = {str(i): int(bins[i]) for i in range(1, cap + 1)}
    out[f">{cap}"] = int(bins[cap + 1])
    return out


def gaussian_anchor_ids(model, out):
    """Map each rendered gaussian row to its global anchor id.

    Mirrors the eval-path filters inside generate_gaussians: rows survive
    opacity MLP > 0 (selection_mask), then binary grid mask != 0.
    """
    masks_all = model.core.get_mask.reshape(-1)  # [N * K * 1]
    per_anchor = masks_all.numel() // model.num_anchors

    sel_idx = torch.nonzero(
        out.gaussians.selection_mask.reshape(-1), as_tuple=False
    ).squeeze(-1)
    la = torch.div(sel_idx, per_anchor, rounding_mode="floor")
    sub = sel_idx % per_anchor
    glob_row = out.gaussians.anchor_indices[la] * per_anchor + sub
    keep = masks_all[glob_row].bool()
    anchor_ids = out.gaussians.anchor_indices[la][keep]
    assert anchor_ids.shape[0] == out.gaussians.xyz.shape[0], (
        "anchor mapping out of sync with generate_gaussians eval filters"
    )
    return anchor_ids.cpu().numpy().astype(np.int64)


def overlap_stats_for_view(model, out, width, height):
    """Blocks-per-anchor and anchors-per-block from one rendered view."""
    means2d = out.meta["means2d"][0].detach().float().cpu().numpy()  # [nnz, 2]
    radii = out.meta["radii"][0].detach().float().cpu().numpy()      # [nnz, 2]
    gids = out.meta["gaussian_ids"].detach().cpu().numpy()           # [nnz]
    ok = (radii > 0).all(axis=-1)
    means2d, radii, gids = means2d[ok], radii[ok], gids[ok]
    if gids.size == 0:
        return [], []
    anchor_of_g = gaussian_anchor_ids(model, out)[gids]

    ntx = max(int(np.ceil(width / TILE)), 1)
    nty = max(int(np.ceil(height / TILE)), 1)
    x0 = np.clip((means2d[:, 0] - radii[:, 0]) // TILE, 0, ntx - 1).astype(np.int64)
    x1 = np.clip((means2d[:, 0] + radii[:, 0]) // TILE, 0, ntx - 1).astype(np.int64)
    y0 = np.clip((means2d[:, 1] - radii[:, 1]) // TILE, 0, nty - 1).astype(np.int64)
    y1 = np.clip((means2d[:, 1] + radii[:, 1]) // TILE, 0, nty - 1).astype(np.int64)

    sx = x1 - x0 + 1
    sy = y1 - y0 + 1
    spans = sx * sy
    row = np.repeat(np.arange(spans.size), spans)
    starts = np.zeros(spans.size, dtype=np.int64)
    np.cumsum(spans[:-1], out=starts[1:])
    off = np.arange(spans.sum(), dtype=np.int64) - starts[row]
    tx = x0[row] + off % sx[row]
    ty = y0[row] + off // sx[row]
    tile_id = ty * ntx + tx

    pair_key = anchor_of_g[row] * (ntx * nty) + tile_id
    pair_key = np.unique(pair_key)
    anchors = pair_key // (ntx * nty)
    tiles = pair_key % (ntx * nty)

    blocks_per_anchor = np.bincount(anchors)
    anchors_per_block = np.bincount(tiles)
    anchors_per_block = anchors_per_block[anchors_per_block > 0]
    return blocks_per_anchor.tolist(), anchors_per_block.tolist()


def render_prefix(model, cam, background, keep_mask):
    """Render one camera restricted to anchors in keep_mask."""
    base_prefilter = model.prefilter_anchors
    model.prefilter_anchors = lambda c: base_prefilter(c) & keep_mask
    try:
        with torch.no_grad():
            out = model.render(
                cam,
                background,
                is_training=False,
                appearance_id=0,
                step=30_000,
            )
    finally:
        del model.prefilter_anchors  # restore the class method
    return out


def view_psnr(model, cam, background, dataset, keep_mask):
    out = render_prefix(model, cam, background, keep_mask)
    pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
    gt = dataset.get_image(cam).clamp(0.0, 1.0)
    mse = torch.mean((pred - gt) ** 2).item()
    psnr = float("inf") if mse <= 0 else -10.0 * np.log10(mse)
    return psnr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing bitstreams/")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument(
        "--stats",
        default=str(
            REPO
            / "analysis/anchor_stats/ph0c_base_r085_lam0005_s42/anchor_stats.npz"
        ),
        help="anchor_stats.npz with per-anchor area (contribution order)",
    )
    ap.add_argument(
        "--bits",
        default=str(
            REPO
            / "analysis/anchor_stats/ph0c_base_r085_lam0005_s42/byte_account/per_anchor_bits.npz"
        ),
        help="per_anchor_bits.npz with per-anchor feat/scaling/offsets/masks bits",
    )
    ap.add_argument("--out", default="analysis/s2_prefix_sweep/prefix_sweep.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from scaffold_gs.datasets import ColmapDataset
    from scaffold_gs.hacpp import HACPlusCodec

    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_total_meta = int(meta["num_anchors_total"])

    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    n = model.num_anchors
    assert n == n_total_meta, f"anchor count mismatch: {n} vs meta {n_total_meta}"
    print(f"[S2] decoded anchors={n}")

    stats = np.load(args.stats)
    area = stats["area"].astype(np.float64)
    assert area.shape[0] == n, "stats row count mismatch with this run"
    rng = np.random.default_rng(0)
    orders = {
        "contribution": np.argsort(-area, kind="stable"),
        "random": rng.permutation(n),
    }

    bits = np.load(args.bits)
    attr_bits = (
        bits["feat"].astype(np.float64)
        + bits["scaling"].astype(np.float64)
        + bits["offsets"].astype(np.float64)
        + bits["masks"].astype(np.float64)
    )
    assert attr_bits.shape[0] == n, "per-anchor bits row count mismatch"
    fixed_bytes = (
        int(meta["bit_hash"])
        + int(meta["bit_mlp"])
        + int(meta["bit_bounds"])
        + int(meta["bit_header"])
    ) / 8.0
    geo_bytes_full = int(meta["bit_anchor"]) / 8.0

    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=args.data_factor,
        test_every=args.test_every,
        white_background=False,
        preload_images=True,
        device=args.device,
    )
    val = dataset.val_cameras
    pick = np.unique(np.linspace(0, len(val) - 1, N_VIEWS).astype(int))
    cams = [val[i] for i in pick]
    background = dataset.background
    print(f"[S2] rendering {len(cams)} of {len(val)} val views")

    results = []
    bpa_all, apb_all = [], []  # pooled A3 overlap observations (per view)
    for ordering in ORDERINGS:
        order = orders[ordering]
        for k in PREFIXES:
            n_keep = int(n * k)
            keep_idx = order[:n_keep]
            keep_mask = torch.zeros(n, dtype=torch.bool, device=model.device)
            keep_mask[torch.from_numpy(keep_idx).to(model.device)] = True

            t0 = time.time()
            psnrs = []
            for ci, cam in enumerate(cams):
                psnr, out = view_psnr(model, cam, background, dataset, keep_mask)
                psnrs.append(psnr)
                if k == 1.00 and ordering == "contribution":
                    bpa, apb = overlap_stats_for_view(
                        model, out, cam.width, cam.height
                    )
                    bpa_all.append(bpa)
                    apb_all.append(apb)
                del out

            payload_bytes = fixed_bytes + geo_bytes_full * k + attr_bits[keep_idx].sum() / 8.0
            entry = {
                "ordering": ordering,
                "prefix": k,
                "n_kept": n_keep,
                "bytes_est": round(payload_bytes, 1),
                "psnr_mean": round(float(np.mean(psnrs)), 4),
                "psnr_per_view": [round(p, 4) for p in psnrs],
                "elapsed_s": round(time.time() - t0, 1),
            }
            results.append(entry)
            print(
                f"[S2] {ordering:>12s} k={k:5.2f} anchors={n_keep:>7d} "
                f"bytes={payload_bytes/1e6:8.2f}MB PSNR={entry['psnr_mean']:.3f}"
            )

    bpa_flat = np.concatenate([np.asarray(x) for x in bpa_all]) if bpa_all else np.array([])
    apb_flat = np.concatenate([np.asarray(x) for x in apb_all]) if apb_all else np.array([])
    overlap = {
        "n_views": len(bpa_all),
        "tile": TILE,
        "blocks_per_anchor": {
            "mean": round(float(bpa_flat.mean()), 3) if bpa_flat.size else None,
            "median": float(np.median(bpa_flat)) if bpa_flat.size else None,
            "p90": float(np.percentile(bpa_flat, 90)) if bpa_flat.size else None,
            "max": int(bpa_flat.max()) if bpa_flat.size else None,
            "hist": hist_counts(bpa_flat),
        },
        "anchors_per_block": {
            "mean": round(float(apb_flat.mean()), 3) if apb_flat.size else None,
            "median": float(np.median(apb_flat)) if apb_flat.size else None,
            "p90": float(np.percentile(apb_flat, 90)) if apb_flat.size else None,
            "max": int(apb_flat.max()) if apb_flat.size else None,
            "hist": hist_counts(apb_flat),
        },
        "gate_note": "design doc: greedy-on-real-edges only worth it if blocks/anchor >= ~5 with high multiplicity",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": str(run_dir),
        "stats": args.stats,
        "bits": args.bits,
        "data_dir": args.data_dir,
        "n_anchors": n,
        "n_views": len(cams),
        "prefixes": PREFIXES,
        "fixed_bytes": fixed_bytes,
        "geo_bytes_full": geo_bytes_full,
        "results": results,
        "a3_overlap": overlap,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"[S2] wrote {out_path}")
    print(
        "[S2] A3 overlap: blocks/anchor mean="
        f"{overlap['blocks_per_anchor']['mean']} "
        f"anchors/block mean={overlap['anchors_per_block']['mean']}"
    )


if __name__ == "__main__":
    main()
