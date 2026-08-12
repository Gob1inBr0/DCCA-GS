"""I6 doc experiments: sensitivity/complexity correlation + strength sweep.

Usage (5090, HAC_5090_a100 env):

    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \\
    PATH="$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH" \\
    python scripts/sensitivity_rd_sweep.py \\
      --ckpt <ckpt_1500.pth> --data-dir <scene> --result-dir runs/i6_sweep \\
      --steps 200 --strengths 0.5,1.0,2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec, anchor_codec_order, sensitivity_multiplier
from scaffold_gs.trainer import evaluate, load_checkpoint


def _pearson(a, b):
    a = a.float()
    b = b.float()
    if a.numel() < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", default="runs/i6_sweep")
    p.add_argument("--data-factor", type=int, default=2)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--strengths", default="0.5,1.0,2.0")
    args = p.parse_args()
    strengths = [float(x) for x in args.strengths.split(",")]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=args.data_factor,
        test_every=args.test_every,
        white_background=False,
        preload_images=False,
        device="cuda",
    )

    model, optim_cfg, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    model.cfg.sensitivity_enabled = True
    model.cfg.sensitivity_weight = 1e-3
    model.cfg.sensitivity_start_iter = 0
    model.create_optimizer(optim_cfg)

    core = model.core
    ema_list, logits_list = [], []
    cams = list(dataset.train_cameras)
    model.train()
    for i in range(args.steps):
        cam = cams[i % len(cams)]
        out = model.render(
            cam,
            dataset.background,
            is_training=True,
            retain_grad=True,
            appearance_id=cam.appearance_id,
            step=30_000 + i,
        )
        gt = dataset.get_image(cam)
        pred = out.image[0].permute(2, 0, 1)
        loss = ((pred - gt) ** 2).mean()
        loss.backward()
        model.accumulate_sensitivity(out.gaussians)
        if out.gaussians.complexity_logits is not None:
            idx = out.gaussians.anchor_indices
            ema = torch.stack(
                [
                    core.sensitivity_feat[idx].squeeze(-1),
                    core.sensitivity_scaling[idx].squeeze(-1),
                    core.sensitivity_offsets[idx].squeeze(-1),
                ],
                dim=-1,
            )
            ema_list.append(ema.detach().cpu())
            logits_list.append(out.gaussians.complexity_logits.detach().cpu())
        model.optimizer.zero_grad(set_to_none=True)

    corr = {}
    if ema_list:
        ema = torch.cat(ema_list, dim=0)
        logits = torch.cat(logits_list, dim=0)
        for j, name in enumerate(["feat", "scaling", "offsets"]):
            corr[name] = round(_pearson(ema[:, j], logits[:, j]), 4)

    z = {
        name: (
            (core.sensitivity_feat - core.sensitivity_mean[0])
            / core.sensitivity_mean[0].clamp_min(1e-12),
            (core.sensitivity_scaling - core.sensitivity_mean[1])
            / core.sensitivity_mean[1].clamp_min(1e-12),
            (core.sensitivity_offsets - core.sensitivity_mean[2])
            / core.sensitivity_mean[2].clamp_min(1e-12),
        )
        for name in ["_"]
    }
    z_feat, z_scaling, z_offsets = z["_"]

    def make_override(strength):
        names = ["feat", "scaling", "offsets"]
        zs = [z_feat, z_scaling, z_offsets]
        codec_idx = anchor_codec_order(model).cpu().numpy()
        paths = {}
        for name, zz in zip(names, zs):
            mult = (
                sensitivity_multiplier(zz, strength)
                .cpu()
                .numpy()
                .astype(np.float32)
            )[codec_idx]
            path = result_dir / f"q_override_{name}_s{strength:g}.npy"
            np.save(path, mult)
            paths[name] = path
        return paths

    codec = HACPlusCodec()
    formula_dir = result_dir / "formula"
    meta_f = codec.encode(model, formula_dir)
    dec_f = codec.decode(formula_dir)
    metrics_f = evaluate(dec_f, dataset, formula_dir / "eval", iteration)

    rows = []
    for strength in strengths:
        ov = make_override(strength)
        out_dir = result_dir / f"sens_s{strength:g}"
        meta_s = codec.encode(
            model,
            out_dir,
            q_override_feat=str(ov["feat"]),
            q_override_scaling=str(ov["scaling"]),
            q_override_offsets=str(ov["offsets"]),
        )
        dec_s = codec.decode(
            out_dir,
            q_override_feat=str(ov["feat"]),
            q_override_scaling=str(ov["scaling"]),
            q_override_offsets=str(ov["offsets"]),
        )
        metrics_s = evaluate(dec_s, dataset, out_dir / "eval", iteration)
        rows.append(
            {
                "strength": strength,
                "psnr": round(metrics_s["psnr"], 4),
                "ssim": round(metrics_s["ssim"], 4),
                "lpips": round(metrics_s["lpips"], 4),
                "total_MB": meta_s["total_MB"],
                "delta_MB_vs_formula": round(meta_s["total_MB"] - meta_f["total_MB"], 4),
            }
        )

    summary = {
        "correlation": corr,
        "formula": {
            "psnr": round(metrics_f["psnr"], 4),
            "ssim": round(metrics_f["ssim"], 4),
            "lpips": round(metrics_f["lpips"], 4),
            "total_MB": meta_f["total_MB"],
        },
        "sensitivity_strengths": rows,
    }
    with open(result_dir / "i6_sweep.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
