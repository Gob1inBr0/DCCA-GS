#!/usr/bin/env python3
"""Full-validation-view evaluation of layered-bitstream prefixes (paper-grade).

Re-renders every prefix of an already-coded layered bitstream on ALL val
views (150 by protocol) and reports per-prefix PSNR/SSIM/LPIPS with
per-view numbers saved for error bars. This replaces the 16-view fast
口径 used during iteration; it changes NO encoding — the .bin files are
read as-is and their symbols re-decoded per prefix, exactly as the
fast sweep did.

Usage (server):
  PYTHONPATH=/home/project2/c25_pylib2 python scripts/full_view_eval.py \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --data-dir /dev/shm/dcca_data/1-78/data \
      --stats analysis/anchor_stats/ph0c_base_r085_lam0005_s42/anchor_stats.npz \
      --mapped analysis/s2_prefix_sweep/ph0c_base_r085_lam0005_s42.mapped.npy \
      --field-aware \
      --out analysis/s2_prefix_sweep/c25_fullview_fieldaware.json
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

import c25_real_bitstream as rb  # reuse ladder/chunk machinery verbatim

GROUPS = rb.GROUPS
FIELDS = ("feat", "scaling", "offset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--mapped", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from scaffold_gs.hacpp import HACPlusCodec
    from scaffold_gs.datasets import ColmapDataset

    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_tag = run_dir.name

    # --- decode model (ground truth tensors) + grouping + ladder chunks ---
    n_trained = int(json.loads((bit_dir / "hac_meta.json").read_text())["num_anchors_total"])
    if args.stats and Path(args.stats).exists():
        area = np.load(args.stats)["area"].astype(np.float64)
        rank = np.empty(n_trained, dtype=np.int64)
        rank[np.argsort(-area, kind="stable")] = np.arange(n_trained)
        g_trained = np.minimum(rank * GROUPS // n_trained, GROUPS - 1).astype(np.int16)
    else:
        g_trained = None
    mapped = (np.load(args.mapped).astype(np.int64) if args.mapped
              and Path(args.mapped).exists() else None)

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
    if mapped is None:
        ck = torch.load(run_dir / "ckpts" / "ckpt_30000.pth", map_location="cpu",
                        weights_only=False)
        tr_xyz = ck["model_state"]["_anchor"].to(dev)
        dec_xyz = model.core.get_anchor
        mapped = np.empty(dec_xyz.shape[0], dtype=np.int64)
        with torch.no_grad():
            for s in range(0, dec_xyz.shape[0], 1024):
                e = min(s + 1024, dec_xyz.shape[0])
                d = torch.cdist(dec_xyz[s:e], tr_xyz)
                mapped[s:e] = d.min(dim=1)[1].cpu().numpy()
                del d
        np.save(out_path.parent / (run_tag + ".mapped.npy"), mapped)
    assert n_alive == mapped.size
    g_of_alive = (g_trained[mapped] if g_trained is not None else
                  np.minimum(np.arange(n_alive) * GROUPS // n_alive,
                             GROUPS - 1).astype(np.int16))

    z = np.load(out_path.parent / (run_tag + "_q_cache.npz"))
    Q_t = {f: torch.from_numpy(z[n]).to(dev)
           for f, n in (("feat", "Q_feat"), ("scaling", "Q_scaling"),
                        ("offset", "Q_offsets"))}
    q_flat, g_flat, out_shape = {}, {}, {}
    for f in FIELDS:
        qf = torch.round(dec_t[f] / Q_t[f]).cpu().numpy().astype(np.int32).reshape(-1)
        n_cols = qf.size // n_alive
        q_flat[f] = qf
        g_flat[f] = np.repeat(g_of_alive, n_cols)
        out_shape[f] = tuple(dec_t[f].shape)

    # --- rebuild the ladder exactly as the coding script does ---
    steps = {f: ([8, 4, 2, 1] if f in ("feat", "offset") else [2, 1])
             for f in FIELDS}  # field-aware ladders
    n_prefixes = max(len(v) for v in steps.values())
    lvl_of_prefix = {f: [min(p, len(steps[f]) - 1) for p in range(n_prefixes)]
                     for f in FIELDS}

    chunk_sets = {}  # (field, level) -> decoded symbols (bit-exact re-encode)
    for f in FIELDS:
        acc = None
        for li, k in enumerate(steps[f]):
            if li == 0:
                layer = np.round(q_flat[f] / k).astype(np.int32)
            else:
                layer = (np.round(q_flat[f] / steps[f][li]).astype(np.int32)
                         - (steps[f][li - 1] // steps[f][li]) * acc)
            data, params, _ = rb.code_chunk(layer, g_flat[f], None)
            back = rb.decode_chunk(data, g_flat[f], params, None)
            assert (back == layer).all(), f"roundtrip mismatch {f} L{li}"
            if acc is None:
                acc = back.copy()
            else:
                acc = (steps[f][li - 1] // steps[f][li]) * acc + back
            chunk_sets[(f, li)] = acc.copy()

    # --- dataset: ALL val views ---
    dataset = ColmapDataset(
        data_dir=args.data_dir, data_factor=2, test_every=8,
        white_background=False, preload_images=False, device=str(dev),
    )
    val = dataset.val_cameras
    cams = list(val)
    background = dataset.background
    print(f"[FV] full evaluation: {len(cams)} views x {n_prefixes} prefixes",
          flush=True)

    results = []
    for p in [n_prefixes - 1] + list(range(n_prefixes - 1)):
        deq = {}
        for f in FIELDS:
            lf = lvl_of_prefix[f][p]
            xhat = (steps[f][lf] * chunk_sets[(f, lf)].reshape(out_shape[f])
                    * Q_t[f].cpu().numpy()).astype(np.float32)
            deq[f] = torch.from_numpy(xhat).to(dev)
        view.anchor_feat = torch.nn.Parameter(deq["feat"].contiguous(),
                                              requires_grad=False)
        view.scaling = torch.nn.Parameter(deq["scaling"].contiguous(),
                                          requires_grad=False)
        view.offset = torch.nn.Parameter(deq["offset"].contiguous(),
                                         requires_grad=False)
        t0 = time.time()
        psnrs, ssims = [], []
        with torch.no_grad():
            for cam in cams:
                out = model.render(cam, background, is_training=False,
                                   appearance_id=0, step=30000)
                pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
                gt = dataset.get_image(cam).clamp(0.0, 1.0)
                mse = torch.mean((pred - gt) ** 2).item()
                psnrs.append(float("inf") if mse <= 0 else -10.0 * np.log10(mse))
                del out
        entry = {
            "prefix": p,
            "field_steps": {f: int(steps[f][lvl_of_prefix[f][p]]) for f in FIELDS},
            "psnr_mean": round(float(np.mean(psnrs)), 4),
            "psnr_std": round(float(np.std(psnrs)), 4),
            "psnr_per_view": [round(x, 4) for x in psnrs],
            "render_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(f"[FV] prefix P{p} ({entry['field_steps']}): FULL-VIEW "
              f"PSNR {entry['psnr_mean']:.3f} +- {entry['psnr_std']:.3f} "
              f"({len(psnrs)} views, {entry['render_s']}s)", flush=True)
        if p == n_prefixes - 1:
            print(f"[FV] SANITY ANCHOR: {entry['psnr_mean']:.3f} dB (need >= 25)")
            assert entry["psnr_mean"] >= 25.0, "sanity anchor failed — INVALID"
        del deq
        torch.cuda.empty_cache()

    out_path.write_text(json.dumps({
        "run": str(run_dir), "n_views": len(cams),
        "field_steps": {f: [int(s) for s in steps[f]] for f in FIELDS},
        "results": results,
    }, indent=1))
    print(f"[FV] wrote {out_path}")


if __name__ == "__main__":
    main()
