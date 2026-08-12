"""I6 stage gates: sensitivity/complexity correlation and offline RD upper bound.

Usage (5090, HAC_5090_a100 env, GPU 1):

    CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
    PATH="$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH" \\
    python scripts/sensitivity_gate.py \\
      --ckpt <ckpt_30000.pth> --data-dir <scene> --result-dir runs/sens_gate

The script never writes back to the checkpoint; it loads fresh copies for the
correlation gate and for the offline RD comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.trainer import evaluate, load_checkpoint


def _collect_sensitivity(model, dataset, steps: int):
    """Run a short sensitivity accumulation loop and return (ema, logits, idx)."""
    core = model.core
    ema_list, logits_list, idx_list = [], [], []
    model.train()
    cams = list(dataset.train_cameras)
    for i in range(steps):
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
            idx_list.append(idx.cpu())
        model.optimizer.zero_grad(set_to_none=True)
    if not ema_list:
        return None
    return (
        torch.cat(ema_list, dim=0),
        torch.cat(logits_list, dim=0),
        torch.cat(idx_list, dim=0),
    )


def _pearson(a, b):
    a = a.float()
    b = b.float()
    if a.numel() < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _build_override(model, path: Path):
    core = model.core
    std = torch.sqrt(core.sensitivity_var.clamp_min(1e-8))
    fields = ["feat", "scaling", "offsets"]
    tensors = [
        core.sensitivity_feat,
        core.sensitivity_scaling,
        core.sensitivity_offsets,
    ]
    strength = model.cfg.sensitivity_strength
    paths = {}
    for i, (name, t) in enumerate(zip(fields, tensors)):
        z = (t - core.sensitivity_mean[i]) / torch.sqrt(
            core.sensitivity_var[i] + 1e-8
        )
        mult = (1.0 + strength * torch.tanh(-z)).cpu().numpy().astype(np.float32)
        p = path / f"q_override_{name}.npy"
        np.save(p, mult)
        paths[name] = p
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", default="runs/sens_gate")
    p.add_argument("--data-factor", type=int, default=2)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

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

    # ---- Correlation gate on an in-memory copy ----
    model, optim_cfg, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    model.cfg.sensitivity_enabled = True
    model.cfg.sensitivity_weight = 1e-3
    model.cfg.sensitivity_start_iter = 0
    model.create_optimizer(optim_cfg)
    collected = _collect_sensitivity(model, dataset, args.steps)
    corr = {}
    corr_pass = True
    if collected is not None:
        ema, logits, _ = collected
        for i, name in enumerate(["feat", "scaling", "offsets"]):
            r = _pearson(ema[:, i], logits[:, i])
            corr[name] = round(r, 4)
            if r > 0.9:
                corr_pass = False
    del model
    torch.cuda.empty_cache()

    # ---- Offline RD upper bound on a fresh copy ----
    model2, optim_cfg2, iteration2, _ = load_checkpoint(args.ckpt, "cuda")
    model2.cfg.sensitivity_enabled = True
    model2.cfg.sensitivity_weight = 1e-3
    model2.cfg.sensitivity_start_iter = 0
    model2.create_optimizer(optim_cfg2)
    _collect_sensitivity(model2, dataset, args.steps)
    override_paths = _build_override(model2, result_dir)

    codec = HACPlusCodec()
    formula_dir = result_dir / "formula"
    sens_dir = result_dir / "sensitivity"
    meta_f = codec.encode(model2, formula_dir)
    dec_f = codec.decode(formula_dir)
    metrics_f = evaluate(dec_f, dataset, formula_dir / "eval", iteration2)

    meta_s = codec.encode(
        model2,
        sens_dir,
        q_override_feat=str(override_paths["feat"]),
        q_override_scaling=str(override_paths["scaling"]),
        q_override_offsets=str(override_paths["offsets"]),
    )
    dec_s = codec.decode(
        sens_dir,
        q_override_feat=str(override_paths["feat"]),
        q_override_scaling=str(override_paths["scaling"]),
        q_override_offsets=str(override_paths["offsets"]),
    )
    metrics_s = evaluate(dec_s, dataset, sens_dir / "eval", iteration2)

    rd_pass = (
        metrics_s["psnr"] >= metrics_f["psnr"] - 1e-3
        and meta_s["total_MB"] <= meta_f["total_MB"] + 1e-3
    )
    summary = {
        "correlation": corr,
        "correlation_pass": corr_pass,
        "formula": {
            "psnr": round(metrics_f["psnr"], 4),
            "ssim": round(metrics_f["ssim"], 4),
            "lpips": round(metrics_f["lpips"], 4),
            "total_MB": meta_f["total_MB"],
        },
        "sensitivity": {
            "psnr": round(metrics_s["psnr"], 4),
            "ssim": round(metrics_s["ssim"], 4),
            "lpips": round(metrics_s["lpips"], 4),
            "total_MB": meta_s["total_MB"],
        },
        "offline_rd_pass": rd_pass,
    }
    with open(result_dir / "gate.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
