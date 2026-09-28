"""方案 C offline prototype: same checkpoint, encode/decode with the
background-feature codebook ON vs OFF, evaluate both bitstreams on the val
set. No training. Run on any free GPU (~10 min).

Usage:
  python scripts/bg_codebook_proto.py --ckpt <ckpt_30000.pth> \
    --data-dir <1-78 data> --flags-npz <anchor_stats.npz> \
    --out-dir <proto_out> --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--flags-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--quantile", type=float, default=0.9)
    args = ap.parse_args()

    from scaffold_gs.datasets import ColmapDataset
    from scaffold_gs.trainer import load_checkpoint, evaluate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _optim, iteration, _meta = load_checkpoint(args.ckpt, args.device)
    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=1,
        test_every=8,
        white_background=False,
        preload_images=False,
        max_width=1600,
        device=args.device,
    )
    background = dataset.background
    model.cfg.bg_flags_path = args.flags_npz
    model.cfg.bg_area_quantile = args.quantile

    results: dict = {"iteration": int(iteration), "ckpt": args.ckpt}

    # --- OFF (baseline bitstream from this same checkpoint) -------------
    off_dir = out_dir / "off"
    meta_off = model.encode_attributes(
        off_dir, q_scale_feat=1.0, q_scale_scaling=1.0, q_scale_offsets=1.0
    )
    model.decode_attributes(off_dir)
    ev_off = evaluate(model, dataset, off_dir / "decoded_eval", iteration)
    results["off"] = {
        "total_MB": meta_off["total_MB"],
        "psnr": ev_off["psnr"],
        "ssim": ev_off["ssim"],
    }

    # --- ON (background-feature codebook) -------------------------------
    model2, _o2, it2, _m2 = load_checkpoint(args.ckpt, args.device)
    model2.cfg.bg_flags_path = args.flags_npz
    model2.cfg.bg_area_quantile = args.quantile
    model2.cfg.bg_codebook_enabled = True
    model2.cfg.bg_codebook_size = args.codebook_size
    on_dir = out_dir / "on"
    meta_on = model2.encode_attributes(
        on_dir, q_scale_feat=1.0, q_scale_scaling=1.0, q_scale_offsets=1.0
    )
    model2.decode_attributes(on_dir)
    ev_on = evaluate(model2, dataset, on_dir / "decoded_eval", it2)
    results["on"] = {
        "total_MB": meta_on["total_MB"],
        "psnr": ev_on["psnr"],
        "ssim": ev_on["ssim"],
        "bit_bg_codebook": meta_on.get("bit_bg_codebook"),
        "num_bg_anchors": meta_on.get("num_bg_anchors"),
    }

    d_mb = results["off"]["total_MB"] - results["on"]["total_MB"]
    d_psnr = results["on"]["psnr"] - results["off"]["psnr"]
    results["delta"] = {
        "size_saved_MB": round(d_mb, 4),
        "size_saved_pct": round(100.0 * d_mb / results["off"]["total_MB"], 2),
        "d_psnr": round(d_psnr, 4),
    }
    with open(out_dir / "proto_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
