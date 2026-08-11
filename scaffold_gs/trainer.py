"""Training loop, evaluation and checkpoint I/O for Scaffold-GS."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import imageio
import numpy as np
import torch
import tqdm

from .config import DataConfig, ModelConfig, OptimConfig, TrainConfig
from .datasets import ColmapDataset
from .losses import l1_loss, ssim_loss
from .model import BaseGaussianModel, get_model_class
from .utils import set_random_seed


def save_checkpoint(
    model: BaseGaussianModel,
    optim_cfg: OptimConfig,
    data_cfg: DataConfig,
    iteration: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model.model_name,
            "model_config": asdict(model.cfg),
            "optim_config": asdict(optim_cfg),
            "data_config": asdict(data_cfg),
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": (
                model.optimizer.state_dict() if model.optimizer is not None else None
            ),
            "stats": {
                "opacity_accum": model.opacity_accum,
                "offset_gradient_accum": model.offset_gradient_accum,
                "offset_denom": model.offset_denom,
                "anchor_demon": model.anchor_demon,
                "max_radii2D": model.max_radii2D,
            },
            "voxel_size": float(model.voxel_size),
            "spatial_lr_scale": float(model.spatial_lr_scale),
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str
) -> tuple[BaseGaussianModel, OptimConfig, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model_cls = get_model_class(ckpt["model_name"])
    model = model_cls(model_cfg, device)
    model.voxel_size = ckpt["voxel_size"]
    model.spatial_lr_scale = ckpt["spatial_lr_scale"]
    sd = ckpt["model_state"]
    if "decoder.embedding_appearance.weight" in sd:
        model.set_appearance(int(sd["decoder.embedding_appearance.weight"].shape[0]))
    model.load_state_dict(sd)
    stats = ckpt.get("stats")
    if stats is not None:
        model.opacity_accum = stats["opacity_accum"].to(device)
        model.offset_gradient_accum = stats["offset_gradient_accum"].to(device)
        model.offset_denom = stats["offset_denom"].to(device)
        model.anchor_demon = stats["anchor_demon"].to(device)
        model.max_radii2D = stats["max_radii2D"].to(device)
    optim_cfg = OptimConfig(**ckpt["optim_config"])
    return model, optim_cfg, int(ckpt["iteration"]), ckpt


def _get_metrics(device: str):
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = None
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        ).to(device)
    except Exception as exc:  # pragma: no cover - offline / missing weights
        print(f"[Eval] LPIPS unavailable, skipping: {exc}")
    return psnr, ssim, lpips


def evaluate(
    model: BaseGaussianModel,
    dataset: ColmapDataset,
    out_dir: str | Path,
    iteration: int,
) -> Dict[str, float]:
    """Render all val cameras and report PSNR / SSIM / LPIPS."""
    out_dir = Path(out_dir)
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    background = dataset.background

    psnr_fn, ssim_fn, lpips_fn = _get_metrics(str(model.device))
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for cam in dataset.val_cameras:
            out = model.render(
                cam,
                background,
                is_training=False,
                appearance_id=0,
                step=iteration,
            )
            pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
            gt = dataset.get_image(cam).clamp(0.0, 1.0)

            psnr_list.append(float(psnr_fn(pred[None], gt[None])))
            ssim_list.append(float(ssim_fn(pred[None], gt[None])))
            if lpips_fn is not None and len(lpips_list) < 64:
                try:
                    lpips_list.append(float(lpips_fn(pred[None], gt[None])))
                except Exception as exc:  # pragma: no cover
                    print(f"[Eval] LPIPS failed, skipping: {exc}")
                    lpips_fn = None

            render_np = (pred.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            imageio.imwrite(render_dir / f"{iteration:06d}_{cam.uid:04d}.png", render_np)

    metrics = {
        "iteration": iteration,
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)) if lpips_list else float("nan"),
        "num_val_views": len(dataset.val_cameras),
    }
    print(
        f"[Eval @{iteration}] PSNR {metrics['psnr']:.3f} | "
        f"SSIM {metrics['ssim']:.4f} | LPIPS {metrics['lpips']:.4f}"
    )
    (out_dir / "metrics.jsonl").open("a").write(json.dumps(metrics) + "\n")
    model.train()
    return metrics


def run_training(cfg: TrainConfig) -> Dict[str, float]:
    set_random_seed(cfg.seed)
    device = cfg.device
    result_dir = Path(cfg.data.result_dir)
    ckpt_dir = result_dir / "ckpts"
    ply_dir = result_dir / "ply"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Create the model first: for hac_pp this imports torch_scatter via the
    # vendored HAC++ core, and pycolmap must load afterwards (see datasets.py).
    model = get_model_class(cfg.model.model_name)(cfg.model, device)
    dataset = ColmapDataset(
        data_dir=cfg.data.data_dir,
        data_factor=cfg.data.data_factor,
        test_every=cfg.data.test_every,
        white_background=cfg.data.white_background,
        preload_images=cfg.data.preload_images,
        device=device,
    )

    model.init_from_pcd(dataset.points, dataset.points_rgb, dataset.scene_scale)
    model.set_appearance(dataset.num_cameras)
    model.create_optimizer(cfg.optim)

    eval_steps = set(cfg.optim.eval_steps)
    save_steps = set(cfg.optim.save_steps)
    eval_steps.add(cfg.optim.max_steps)
    save_steps.add(cfg.optim.max_steps)

    background = dataset.background
    train_cams = list(dataset.train_cameras)
    optim = cfg.optim

    pbar = tqdm.tqdm(range(1, optim.max_steps + 1), desc="Scaffold-GS training")
    final_metrics: Dict[str, float] = {}
    for iteration in pbar:
        model.update_learning_rate(iteration)
        cam = random.choice(train_cams)
        retain_grad = iteration < optim.update_until
        out = model.render(
            cam,
            background,
            is_training=True,
            retain_grad=retain_grad,
            appearance_id=cam.appearance_id,
            step=iteration,
        )

        gt = dataset.get_image(cam)
        pred = out.image[0].permute(2, 0, 1)
        ll1 = l1_loss(pred, gt).mean()
        ssim = ssim_loss(pred[None], gt[None])
        if out.gaussians.xyz.shape[0] > 0:
            scale_reg = out.gaussians.scales.prod(dim=1).mean()
        else:
            scale_reg = torch.zeros((), device=device)
        loss = (
            (1.0 - optim.lambda_dssim) * ll1
            + optim.lambda_dssim * ssim
            + optim.scale_reg_lambda * scale_reg
        )
        rate_term = getattr(model, "rate_loss_term", None)
        if rate_term is not None:
            loss = loss + rate_term(out.gaussians, iteration)

        loss.backward()

        if optim.update_until > iteration > optim.start_stat:
            model.training_statis(
                out.meta["means2d"],
                (out.meta["radii"] > 0).all(dim=-1),
                out.gaussians,
                out.meta["width"],
                out.meta["gaussian_ids"],
                out.meta["height"],
            )
        if (
            optim.update_from < iteration < optim.update_until
            and iteration % optim.update_interval == 0
        ):
            model.adjust_anchor(
                check_interval=optim.update_interval,
                success_threshold=optim.success_threshold,
                grad_threshold=optim.densify_grad_threshold,
                min_opacity=optim.min_opacity,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if iteration % 10 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                anchors=f"{model.num_anchors}",
            )
        if torch.cuda.is_available() and iteration % 100 == 0:
            gids = out.meta.get("gaussian_ids")
            gids_n = gids.numel() if gids is not None else 0
            print(
                f"[Mem @{iteration}] alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "
                f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB "
                f"gids={gids_n}",
                flush=True,
            )

        # Drop references to this step's render output / graph before the next
        # iteration; otherwise the previous step's autograd graph and packed
        # rasterizer buffers stay alive while the next step is built.
        del out, loss, pred, gt
        if torch.cuda.is_available() and iteration % 100 == 0:
            torch.cuda.empty_cache()

        if iteration in eval_steps:
            final_metrics = evaluate(model, dataset, result_dir, iteration)
        if iteration in save_steps:
            save_checkpoint(
                model, optim, cfg.data, iteration, ckpt_dir / f"ckpt_{iteration}.pth"
            )
            model.save_ply(ply_dir / f"iteration_{iteration}" / "point_cloud.ply")
            model.save_mlp_checkpoints(
                ply_dir / f"iteration_{iteration}" / "mlp_checkpoints"
            )
            print(f"[Save @{iteration}] anchors={model.num_anchors}")

    print(f"Training finished. Final anchors: {model.num_anchors}")
    return final_metrics
