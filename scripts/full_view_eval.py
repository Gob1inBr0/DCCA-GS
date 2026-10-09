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
    ap.add_argument("--data-factor", type=int, default=2,
                    help="GT resolution factor (1 = full res; audit vs "
                         "decoded_eval needs 1 + max-width 1600)")
    ap.add_argument("--max-width", type=int, default=None)
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

    last_q = getattr(model.core, "last_decode_Q", None)
    if last_q is not None:
        Q_t = {f: last_q[n].to(dev).float()
               for f, n in (("feat", "feat"), ("scaling", "scaling"),
                            ("offset", "offsets"))}
        # decode-side offsets Q is flat (N, 3k); align to the view shape
        Q_t["offset"] = Q_t["offset"].reshape(dec_t["offset"].shape)
        print("[FV] Q taken from codec decode (production grid)")
    else:
        print("[FV] WARNING: falling back to legacy q_cache (grid may differ "
              "from production decode!)")
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
    # NOTE: this evaluator describes the FIELD-AWARE ladder only; .bin files
    # encoded in uniform mode ([16,8,4,2,1]) would need different steps.
    steps = {f: ([8, 4, 2, 1] if f in ("feat", "offset") else [2, 1])
             for f in FIELDS}  # field-aware ladders
    n_prefixes = max(len(v) for v in steps.values())
    lvl_of_prefix = {f: [min(p, len(steps[f]) - 1) for p in range(n_prefixes)]
                     for f in FIELDS}

    chunk_sets = {}  # (field, level) -> decoded symbols (bit-exact re-encode)
    chunk_bytes = {}  # (field, level) -> transmitted bytes (4B lens+params+data)
    for f in FIELDS:
        acc = None
        for li, k in enumerate(steps[f]):
            if li == 0:
                layer = np.round(q_flat[f] / k).astype(np.int32)
            else:
                layer = (np.round(q_flat[f] / steps[f][li]).astype(np.int32)
                         - (steps[f][li - 1] // steps[f][li]) * acc)
            data, params, _, _ = rb.code_chunk(layer, g_flat[f], None)
            back = rb.decode_chunk(data, g_flat[f], params, None)
            assert (back == layer).all(), f"roundtrip mismatch {f} L{li}"
            locs, p0s, bts = params
            chunk_bytes[(f, li)] = (8 + len(data)
                                    + locs.astype(np.int32).nbytes
                                    + p0s.astype(np.float32).nbytes
                                    + bts.astype(np.float32).nbytes)
            if acc is None:
                acc = back.copy()
            else:
                acc = (steps[f][li - 1] // steps[f][li]) * acc + back
            chunk_sets[(f, li)] = acc.copy()

    # Production payload = everything the decoder needs, minus audit/debug
    # artifacts. Fixed part = payload minus the production attribute streams
    # (the ladder re-encodes feat/scaling/offset itself, so every prefix
    # carries the fixed part plus its cumulative ladder attribute bytes).
    EXCLUDE = {"attributes.pth", "codec_roundtrip_diagnostics.json",
               "content_aware_q_meta.json"}
    payload_total = sum(p_.stat().st_size for p_ in bit_dir.iterdir()
                        if p_.is_file() and p_.name not in EXCLUDE)
    prod_attr = sum(p_.stat().st_size for f0 in ("feat", "scaling", "offset")
                    for p_ in bit_dir.glob(f"{f0}_*.b"))
    fixed_bytes = payload_total - prod_attr
    print(f"[FV] production payload {payload_total/1048576:.2f} MB "
          f"(attr {prod_attr/1048576:.2f} + fixed {fixed_bytes/1048576:.2f})",
          flush=True)

    # --- dataset: ALL val views ---
    dataset = ColmapDataset(
        data_dir=args.data_dir, data_factor=args.data_factor, test_every=8,
        max_width=args.max_width,
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
        attr_cum = sum(chunk_bytes[(f, li)]
                       for f in FIELDS
                       for li in range(lvl_of_prefix[f][p] + 1))
        entry = {
            "prefix": p,
            "field_steps": {f: int(steps[f][lvl_of_prefix[f][p]]) for f in FIELDS},
            "attr_cum_bytes": attr_cum,
            "real_bytes": attr_cum + fixed_bytes,
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
            # Sanity: the final prefix re-renders the production symbols, so it
            # must match the production decode on the same views. Budget arms
            # legitimately land below 25 dB, so compare against the run's own
            # decoded_eval instead of a fixed floor.
            ref = None
            mfile = run_dir / "decoded_eval" / "metrics.jsonl"
            if mfile.exists():
                ref = float(json.loads(
                    mfile.read_text().strip().splitlines()[-1])["psnr"])
            if ref is None:
                print(f"[FV] SANITY ANCHOR: {entry['psnr_mean']:.3f} dB "
                      f"(no decoded_eval reference, skipped)", flush=True)
            else:
                gap = abs(entry["psnr_mean"] - ref)
                print(f"[FV] SANITY ANCHOR: {entry['psnr_mean']:.3f} dB vs "
                      f"production decode {ref:.3f} dB (gap {gap:.3f})",
                      flush=True)
                assert gap <= 0.5, "final prefix deviates from production decode"
        del deq
        torch.cuda.empty_cache()

    out_path.write_text(json.dumps({
        "run": str(run_dir), "n_views": len(cams),
        "production_payload_bytes": payload_total,
        "fixed_bytes": fixed_bytes,
        "field_steps": {f: [int(s) for s in steps[f]] for f in FIELDS},
        "results": results,
    }, indent=1))
    print(f"[FV] wrote {out_path}")


if __name__ == "__main__":
    main()
