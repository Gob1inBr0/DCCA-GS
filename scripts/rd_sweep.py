"""Post-hoc rate-distortion sweep for HAC++ (quantization steps + anchor mask).

Runs encode -> decode -> eval for every combination of quantization-step
scales and post-hoc anchor-mask pruning ratios on an already-trained
checkpoint, then writes ``results.json`` / ``results.csv`` with bitrate
breakdown and PSNR/SSIM/LPIPS per configuration.

Usage (5090, HAC_5090_a100 env, GPU 1):

    CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
    PATH="$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH" \\
    python scripts/rd_sweep.py \\
      --ckpt "$HOME/data_space/web_scan/runs/ours_hacpp_30k_capped/ckpts/ckpt_30000.pth" \\
      --data-dir "$HOME/data_space/web_scan/第一组数据" \\
      --result-dir runs/rd_sweep \\
      --mask-keep-ratio 0.85 0.7 \\
      --q-scale-feat 2.0 4.0

All ``--q-scale-*`` defaults include 1.0, so the official baseline is always
included in the grid. ``--q-scale-joint 2.0`` adds a config where feat,
scaling and offsets are all scaled together.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.trainer import evaluate, load_checkpoint


def _raw_attribute_mb(model) -> float:
    total = 0
    for t in model.export_attributes().values():
        if torch.is_tensor(t):
            total += int(t.nelement() * t.element_size())
    return round(total / 1e6, 3)


def _configs(
    mask_ratios: List[float],
    q_feat: List[float],
    q_scaling: List[float],
    q_offsets: List[float],
    q_joint: List[float],
) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for mr, qf, qs, qo in itertools.product(
        mask_ratios, q_feat, q_scaling, q_offsets
    ):
        cfg = {
            "mask_keep_ratio": None if abs(mr - 1.0) < 1e-9 else mr,
            "q_scale_feat": float(qf),
            "q_scale_scaling": float(qs),
            "q_scale_offsets": float(qo),
        }
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    for j in q_joint:
        if abs(j - 1.0) < 1e-9:
            continue
        cfg = {
            "mask_keep_ratio": None,
            "q_scale_feat": float(j),
            "q_scale_scaling": float(j),
            "q_scale_offsets": float(j),
        }
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def _tag(cfg: Dict[str, Any]) -> str:
    mr = cfg["mask_keep_ratio"]
    return (
        f"mask{'off' if mr is None else f'{mr:g}'}"
        f"_qf{cfg['q_scale_feat']:g}"
        f"_qs{cfg['q_scale_scaling']:g}"
        f"_qo{cfg['q_scale_offsets']:g}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", default="runs/rd_sweep")
    p.add_argument("--data-factor", type=int, default=2)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-keep-ratio", type=float, nargs="+", default=[1.0])
    p.add_argument("--q-scale-feat", type=float, nargs="+", default=[1.0])
    p.add_argument("--q-scale-scaling", type=float, nargs="+", default=[1.0])
    p.add_argument("--q-scale-offsets", type=float, nargs="+", default=[1.0])
    p.add_argument("--q-scale-joint", type=float, nargs="+", default=[])
    args = p.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    model, _, iteration, _ = load_checkpoint(args.ckpt, args.device)
    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=args.data_factor,
        test_every=args.test_every,
        white_background=False,
        preload_images=True,
        device=args.device,
    )

    # Uncompressed reference (quality of the trained model itself).
    orig_metrics = evaluate(
        model, dataset, result_dir / "orig", iteration
    )
    raw_mb = _raw_attribute_mb(model)
    print(
        f"[Sweep] original PSNR {orig_metrics['psnr']:.3f} "
        f"SSIM {orig_metrics['ssim']:.4f} LPIPS {orig_metrics['lpips']:.4f} "
        f"raw_attrs {raw_mb:.1f} MB"
    )

    configs = _configs(
        args.mask_keep_ratio,
        args.q_scale_feat,
        args.q_scale_scaling,
        args.q_scale_offsets,
        args.q_scale_joint,
    )
    rows: List[Dict[str, Any]] = []
    baseline_key: Optional[str] = None
    baseline_mb = None
    baseline_psnr = None

    for cfg in configs:
        tag = _tag(cfg)
        out_dir = result_dir / tag
        try:
            codec = HACPlusCodec()
            meta = codec.encode(model, out_dir, **cfg)
            decoded = codec.decode(out_dir, **cfg)
            metrics = evaluate(decoded, dataset, out_dir / "eval", iteration)
            row: Dict[str, Any] = {
                "tag": tag,
                **cfg,
                "num_anchors": int(meta["num_anchors"]),
                "num_anchors_total": int(meta["num_anchors_total"]),
                "bit_anchor": int(meta["bit_anchor"]),
                "bit_feat": int(meta["bit_feat"]),
                "bit_scaling": int(meta["bit_scaling"]),
                "bit_offsets": int(meta["bit_offsets"]),
                "bit_hash": int(meta["bit_hash"]),
                "bit_masks": int(meta["bit_masks"]),
                "total_MB": float(meta["total_MB"]),
                "psnr": float(metrics["psnr"]),
                "ssim": float(metrics["ssim"]),
                "lpips": float(metrics["lpips"]),
                "raw_attribute_MB": raw_mb,
                "error": None,
            }
            if (
                cfg["mask_keep_ratio"] is None
                and cfg["q_scale_feat"] == 1.0
                and cfg["q_scale_scaling"] == 1.0
                and cfg["q_scale_offsets"] == 1.0
            ):
                baseline_key = tag
                baseline_mb = float(meta["total_MB"])
                baseline_psnr = float(metrics["psnr"])
            print(
                f"[Sweep] {tag}: {row['total_MB']:.3f} MB | "
                f"anchors {row['num_anchors']} | "
                f"PSNR {row['psnr']:.3f} SSIM {row['ssim']:.4f} "
                f"LPIPS {row['lpips']:.4f}"
            )
        except Exception as exc:  # pragma: no cover - remote env issues
            print(f"[Sweep] {tag} FAILED: {type(exc).__name__}: {exc}")
            row = {
                "tag": tag,
                **cfg,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        del decoded
        torch.cuda.empty_cache()

    for row in rows:
        if row.get("error"):
            continue
        if baseline_mb is not None:
            row["delta_MB_vs_baseline"] = round(row["total_MB"] - baseline_mb, 4)
        if baseline_psnr is not None:
            row["delta_psnr_vs_baseline"] = round(row["psnr"] - baseline_psnr, 4)
        row["orig_psnr"] = float(orig_metrics["psnr"])
        row["orig_ssim"] = float(orig_metrics["ssim"])
        row["orig_lpips"] = float(orig_metrics["lpips"])

    summary = {
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "iteration": int(iteration),
        "baseline_key": baseline_key,
        "orig_metrics": {
            "psnr": float(orig_metrics["psnr"]),
            "ssim": float(orig_metrics["ssim"]),
            "lpips": float(orig_metrics["lpips"]),
        },
        "raw_attribute_MB": raw_mb,
        "rows": rows,
    }
    with open(result_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "tag",
        "mask_keep_ratio",
        "q_scale_feat",
        "q_scale_scaling",
        "q_scale_offsets",
        "num_anchors",
        "num_anchors_total",
        "bit_anchor",
        "bit_feat",
        "bit_scaling",
        "bit_offsets",
        "bit_hash",
        "bit_masks",
        "total_MB",
        "delta_MB_vs_baseline",
        "psnr",
        "delta_psnr_vs_baseline",
        "ssim",
        "lpips",
        "raw_attribute_MB",
        "orig_psnr",
        "orig_ssim",
        "orig_lpips",
        "error",
    ]
    with open(result_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[Sweep] wrote {result_dir / 'results.csv'} and "
          f"{result_dir / 'results.json'}")


if __name__ == "__main__":
    main()
