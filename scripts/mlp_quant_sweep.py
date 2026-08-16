"""MLP weight quantization + entropy-coding sweep.

For each bit width (and optionally a single MLP group), quantize the MLP
weights with per-channel symmetric PTQ, re-run the attribute codec with the
dequantized weights, compress the indices with a static arithmetic coder
(raw int16 for >=16-bit), decode, and evaluate.  ``total_MB`` uses the
official accounting but replaces ``bit_mlp = params*32`` with the real
compressed MLP payload.

Usage (5090):
    PYTHONPATH=$PWD python scripts/mlp_quant_sweep.py \
      --ckpt <ckpt_90000.pth> \
      --data-dir <4-28> \
      --result-dir runs/mlp_quant_sweep \
      --bits 16 8 6 4 \
      --max-width 1600 --data-factor 1 --no-preload-images

    # per-MLP sensitivity at 8-bit (repeat --groups mlp_grid, ...)
    --bits 8 --groups mlp_grid

    # mixed precision: complexity+deform 8-bit, rest 16-bit
    --group-bits mlp_complexity:8 mlp_deform:8 mlp_opacity:16 mlp_cov:16 \
                 mlp_color:16 mlp_grid:16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple

import torch

from hacplus.scene.gaussian_model import bit2MB_scale
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.mlp_quant import MLP_GROUPS, quantize_core_mlps, save_mlp_payload
from scaffold_gs.trainer import evaluate, load_checkpoint


def run_one(
    ckpt: str,
    bits: int,
    groups: Tuple[str, ...],
    bits_map: dict | None,
    data_dir: str,
    out_dir: Path,
    max_width: int,
    data_factor: int,
    no_preload: bool,
    device: str,
) -> dict:
    model, _, iteration, _ = load_checkpoint(ckpt, device)
    if bits_map:
        quant_meta = quantize_core_mlps(model.core, bits_map=bits_map)
    elif bits < 32:
        quant_meta = quantize_core_mlps(model.core, bits, groups=groups)
    else:
        quant_meta = {}

    bs = out_dir / "bitstreams"
    if bs.exists():
        shutil.rmtree(bs)
    codec = HACPlusCodec()
    enc = codec.encode(model, bs)
    if quant_meta:
        sizes = save_mlp_payload(quant_meta, bs)
        mlp_bits = (
            sizes["mlp_quant_bin_bytes"] + sizes["mlp_quant_meta_bytes"]
        ) * 8
    else:
        mlp_bits = int(enc["bit_mlp"])

    attr_bits = int(enc["total_bits"] - enc["bit_mlp"])
    total_bits = attr_bits + mlp_bits
    total_MB = total_bits / bit2MB_scale

    decoded = codec.decode(bs)
    dataset = ColmapDataset(
        data_dir=data_dir,
        data_factor=data_factor,
        test_every=8,
        white_background=False,
        preload_images=not no_preload,
        max_width=max_width,
        device=device,
    )
    metrics = evaluate(decoded, dataset, out_dir / "eval", iteration)
    row = {
        "bits": "mixed" if bits_map else bits,
        "bits_map": bits_map,
        "groups": list(groups),
        "mlp_payload_MB": round(mlp_bits / bit2MB_scale, 4),
        "attr_MB": round(attr_bits / bit2MB_scale, 4),
        "total_MB": round(total_MB, 4),
        "psnr": round(float(metrics["psnr"]), 4),
        "ssim": round(float(metrics["ssim"]), 4),
        "lpips": round(float(metrics["lpips"]), 4),
        "num_anchors": int(enc["num_anchors"]),
    }
    print(json.dumps(row, sort_keys=True))
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", default="runs/mlp_quant_sweep")
    p.add_argument("--bits", type=int, nargs="+", default=[16, 8, 6, 4])
    p.add_argument(
        "--group-bits",
        nargs="+",
        default=None,
        help="Per-group bit widths, e.g. mlp_complexity:8 mlp_deform:8 "
        "mlp_opacity:16 ... (mixed precision; overrides --bits).",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="MLP groups to quantize (default: all); e.g. mlp_grid mlp_color",
    )
    p.add_argument("--data-factor", type=int, default=1)
    p.add_argument("--max-width", type=int, default=1600)
    p.add_argument("--no-preload-images", action="store_true")
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the 32-bit baseline row (used for per-MLP sensitivity runs).",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    groups = tuple(args.groups) if args.groups else MLP_GROUPS
    bits_map = None
    if args.group_bits:
        bits_map = {}
        for item in args.group_bits:
            g, b = item.split(":")
            bits_map[g] = int(b)
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    if not args.skip_baseline:
        rows.append(
            run_one(
                args.ckpt,
                32,
                groups,
                None,
                args.data_dir,
                out / "b32",
                args.max_width,
                args.data_factor,
                args.no_preload_images,
                args.device,
            )
        )
    for bits in (args.bits if not bits_map else [min(bits_map.values())]):
        rows.append(
            run_one(
                args.ckpt,
                bits,
                groups,
                bits_map,
                args.data_dir,
                out / f"b{bits}",
                args.max_width,
                args.data_factor,
                args.no_preload_images,
                args.device,
            )
        )

    summary = {"ckpt": args.ckpt, "groups": list(groups), "rows": rows}
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
