"""I5 stage-1 RD probe on a trained checkpoint.

For every VQ configuration the script encodes the *already trained* model
(no retraining), decodes it and evaluates PSNR/SSIM/LPIPS, so the real-codec
rate is compared against the scalar baseline on the same weights.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/i5_rd_probe.py \
      --ckpt <ckpt_90000.pth> --data-dir <4-28> \
      --configs "scalar;d4:4:4:4;d4:4:4:4:dither;e8:8:4:4" --out runs/i5_rd.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.trainer import evaluate, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--configs",
        default="scalar;d4:4:4:4:ca;d4:4:4:4:ca:dither;e8:8:4:4:ca",
        help="Semicolon-separated modes: scalar | <lattice>:<gf>:<gs>:<go>[:ca][:dither]",
    )
    p.add_argument("--max-width", type=int, default=1600)
    p.add_argument("--out", default="runs/i5_rd.json")
    args = p.parse_args()

    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=1,
        max_width=args.max_width,
        test_every=8,
        white_background=False,
        preload_images=False,
        device="cuda",
    )
    rows = []
    for spec in args.configs.split(";"):
        parts = spec.split(":")
        tag = parts[0] if len(parts) == 1 else parts[0] + "_" + "_".join(parts[1:])
        model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
        if tag != "scalar":
            lattice, gf, gs, go = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            model.cfg.vq_enabled = True
            model.cfg.vq_lattice = lattice
            model.cfg.vq_group_feat = gf
            model.cfg.vq_group_scaling = gs
            model.cfg.vq_group_offsets = go
            model.cfg.vq_content_aware = "ca" in parts[4:]
            model.cfg.dither_enabled = "dither" in parts[4:]
            model.cfg.dither_seed = 7
        else:
            model.cfg.vq_enabled = False
            model.cfg.dither_enabled = False

        out_dir = Path(args.out).parent / tag
        codec = HACPlusCodec()
        print(f"\n=== {tag} ===", flush=True)
        meta = codec.encode(model, out_dir)
        decoded = codec.decode(out_dir)
        metrics = evaluate(decoded, dataset, out_dir / "eval", iteration)
        row = {
            "tag": tag,
            "total_MB": meta["total_MB"],
            "num_anchors": meta["num_anchors"],
            "psnr": round(metrics["psnr"], 4),
            "ssim": round(metrics["ssim"], 4),
            "lpips": round(metrics["lpips"], 4),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, decoded
        torch.cuda.empty_cache()

    summary = {"configs": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("\n=== SUMMARY ===")
    for r in rows:
        print(
            f"{r['tag']}: PSNR {r['psnr']:.3f} | SSIM {r['ssim']:.4f} | "
            f"LPIPS {r['lpips']:.4f} | {r['total_MB']:.4f}MB"
        )


if __name__ == "__main__":
    main()
