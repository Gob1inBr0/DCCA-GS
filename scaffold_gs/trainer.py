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


@torch.no_grad()
def holdout_psnr(
    model: BaseGaussianModel,
    dataset: ColmapDataset,
    cams,
    iteration: int,
) -> float:
    """Quick fixed-camera PSNR probe for the Phase-2a recovery gate.

    MSE-based (no metric deps), ~8 renders per projection cycle, decode-like
    path (model.eval + is_training=False) so it tracks quantized quality.
    appearance_id=0 matches the zero-appearance protocol of every round-2
    arm; revisit if appearance conditioning is ever enabled.
    """
    model.eval()
    try:
        background = dataset.background
        mses = []
        for cam in cams:
            out = model.render(
                cam,
                background,
                is_training=False,
                appearance_id=0,
                step=iteration,
            )
            pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
            gt = dataset.get_image(cam).clamp(0.0, 1.0)
            mses.append(torch.mean((pred - gt) ** 2).item())
    finally:
        model.train()
    mse = float(np.mean(mses)) if mses else 1.0
    return -10.0 * np.log10(max(mse, 1e-12))


def run_training(cfg: TrainConfig) -> Dict[str, float]:
    print(
        "[TrainerVer] post_branch=loaded "
        f"spa_post_ratio={getattr(cfg.model, 'spa_post_ratio', 'MISSING')} "
        f"spa_post_window={getattr(cfg.model, 'spa_post_window', 'MISSING')} "
        f"mini_splat_enabled={cfg.model.mini_splat_enabled} "
        f"spa_rate_aware={getattr(cfg.model, 'spa_rate_aware', 'MISSING')} "
        f"spa_rate_tau={getattr(cfg.model, 'spa_rate_tau', 'MISSING')} "
        f"spa_bit_budget={getattr(cfg.model, 'spa_bit_budget', 'MISSING')} "
        f"sensitivity_target_mode={getattr(cfg.model, 'sensitivity_target_mode', 'MISSING')} "
        f"sensitivity_use_fisher={getattr(cfg.model, 'sensitivity_use_fisher', 'MISSING')} "
        f"sensitivity_second_order={getattr(cfg.model, 'sensitivity_second_order', 'MISSING')} "
        f"spa_holdout_gate={getattr(cfg.model, 'spa_holdout_gate', 'MISSING')}",
        flush=True,
    )
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
        max_width=cfg.data.max_width,
        cache_images_cpu=cfg.data.cache_images_cpu,
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

    fusion_pool: list = []
    fusion_views = int(getattr(cfg.model, "mini_splat_views", 8))
    if getattr(cfg.model, "fusion_prune", False) and getattr(
        cfg.model, "mini_splat_enabled", False
    ):
        # Pass the FULL camera pool: hacpp.adjust_anchor rotates an 8-view
        # window every projection, so coverage accumulates over the whole
        # training set instead of 8 fixed (never-rotating) views.
        fusion_pool = list(train_cams)
        print(
            f"[FusionPrune] rotating coverage pool: {len(fusion_pool)} cams "
            f"x {fusion_views} views/cycle",
            flush=True,
        )
    optim = cfg.optim

    # D4b (Phase 2a) holdout gate: fixed training-camera probe; fires when
    # the probe PSNR drops beyond max(3*sigma_recent, eps) below its EMA
    # baseline and freezes the post-reinit kappa ramp via core state.
    holdout_state = None
    if getattr(cfg.model, "spa_holdout_gate", False):
        if float(getattr(cfg.model, "spa_post_ratio", 1.0)) >= 1.0:
            # The gate freezes the post-phase kappa ramp; with no post phase
            # there is nothing to freeze — say so instead of arming silently.
            print(
                "[HoldoutGate] NOT armed: spa_post_ratio=1.0 leaves no "
                "post-phase ramp to freeze",
                flush=True,
            )
        elif len(train_cams) > 0:
            rng = random.Random(cfg.seed)
            n_hold = max(
                1,
                min(
                    int(getattr(cfg.model, "spa_holdout_views", 8)),
                    len(train_cams),
                ),
            )
            holdout_state = {
                "cams": rng.sample(train_cams, n_hold),
                "hist": [],
                "baseline": None,
                "triggers": 0,
            }
            print(
                f"[HoldoutGate] armed: {n_hold} cams, "
                f"eps={float(getattr(cfg.model, 'spa_holdout_gate_eps', 0.05)):.3f}dB "
                f"freeze={int(getattr(cfg.model, 'spa_holdout_freeze_window', 2))} cycles",
                flush=True,
            )

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
        if getattr(cfg.model, "importance_weighted_loss", False):
            w = out.alpha[0, :, :, 0].detach()          # [H,W] rendered opacity
            w = w / w.max().clamp_min(1e-8)
            floor = float(getattr(cfg.model, "importance_weight_floor", 0.2))
            scale = float(getattr(cfg.model, "importance_weight_scale", 1.0))
            w = floor + (scale - floor) * w
            diff = (pred - gt).abs().mean(dim=0)        # [H,W]
            ll1 = (w * diff).sum() / w.sum()
        else:
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
        sens_loss_fn = getattr(model, "sensitivity_supervision", None)
        if sens_loss_fn is not None:
            loss = loss + sens_loss_fn(out.gaussians)
        sem_loss_fn = getattr(model, "semantic_supervision", None)
        if sem_loss_fn is not None:
            loss = loss + sem_loss_fn(out.gaussians)
        spa_loss_fn = getattr(model, "spa_loss_term", None)
        if spa_loss_fn is not None:
            loss = loss + spa_loss_fn()
        # B3: coarse-ladder alignment penalty, filled in during render when
        # coarse_ladder_align is on and past coarse_ladder_start_iter
        ladder_pen = getattr(out.gaussians, "ladder_penalty", None)
        if ladder_pen is not None:
            loss = loss + ladder_pen
            if iteration % 500 == 0:
                raw = getattr(out.gaussians, "ladder_penalty_raw", None)
                raw_val = float(raw) if raw is not None else float("nan")
                print(f"[B3] iter {iteration}: ladder raw {raw_val:.4f} "
                      f"weighted {float(ladder_pen):.5f}", flush=True)

        loss.backward()

        sens_accum = getattr(model, "accumulate_sensitivity", None)
        if (
            sens_accum is not None
            and iteration >= cfg.model.sensitivity_start_iter
        ):
            sens_accum(out.gaussians)

        if optim.update_until > iteration > optim.start_stat:
            model.training_statis(
                out.meta["means2d"],
                (out.meta["radii"] > 0).all(dim=-1),
                out.gaussians,
                out.meta["width"],
                out.meta["gaussian_ids"],
                out.meta["height"],
            )
        if iteration % 250 == 0:
            print(
                f"[PostDebug] iter={iteration} post_ratio="
                f"{getattr(cfg.model, 'spa_post_ratio', None)} "
                f"ms_enabled={getattr(cfg.model, 'mini_splat_enabled', None)} "
                f"ms_done={getattr(model.core, 'mini_splat_done', None)}",
                flush=True,
            )
        if (
            iteration > int(getattr(cfg.model, "mini_splat_reinit_iter", 0))
            and iteration <= int(getattr(cfg.model, "mini_splat_reinit_iter", 0))
            + int(getattr(cfg.model, "spa_post_window", 2000)) + 300
            and iteration % optim.update_interval == 0
        ):
            c1 = float(getattr(cfg.model, "spa_post_ratio", 1.0)) < 1.0
            c2 = (
                getattr(model.core, "mini_splat_done", False)
                or not getattr(cfg.model, "mini_splat_enabled", False)
            )
            c3 = iteration % optim.update_interval == 0
            print(
                f"[PostDebug2] iter={iteration} c1_ratio={c1} c2_doneorenabled={c2} "
                f"c3_interval={c3} fire={c1 and c2 and c3}",
                flush=True,
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
                fusion_pool=fusion_pool,
                fusion_views=fusion_views,
                background=background,
                post_phase=False,
            )
        elif (
            float(getattr(cfg.model, "spa_post_ratio", 1.0)) < 1.0
            and (
                getattr(model.core, "mini_splat_done", False)
                or not getattr(cfg.model, "mini_splat_enabled", False)
            )
            and iteration > int(getattr(cfg.model, "mini_splat_reinit_iter", 0))
            and iteration <= int(getattr(cfg.model, "mini_splat_reinit_iter", 0))
            + int(getattr(cfg.model, "spa_post_window", 2000))
            and iteration % optim.update_interval == 0
        ):
            # Phase 0 post-reinit selection: pure selection (no growth) while
            # kappa ramps down to spa_post_ratio of the post-reinit count.
            # With mini-splat disabled there is no reinit event; the window
            # still anchors at reinit_iter so ±depth-reinit arms share the
            # schedule (core falls back to kappa = spa_final_n * spa_ratio).
            model.adjust_anchor(
                check_interval=optim.update_interval,
                success_threshold=optim.success_threshold,
                grad_threshold=optim.densify_grad_threshold,
                min_opacity=optim.min_opacity,
                fusion_pool=fusion_pool,
                fusion_views=fusion_views,
                background=background,
                post_phase=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if holdout_state is not None:
                p = holdout_psnr(
                    model, dataset, holdout_state["cams"], iteration
                )
                hist = holdout_state["hist"]
                hist.append(p)
                if len(hist) > 10:
                    hist.pop(0)
                sigma = float(np.std(hist)) if len(hist) >= 3 else 0.0
                eps_eff = max(
                    3.0 * sigma,
                    float(getattr(cfg.model, "spa_holdout_gate_eps", 0.05)),
                )
                core = model.core
                frozen = int(getattr(core, "spa_gate_freeze_cycles", 0)) > 0
                max_trig = int(
                    getattr(cfg.model, "spa_holdout_max_triggers", 5)
                )
                if holdout_state["baseline"] is None:
                    holdout_state["baseline"] = p
                if frozen:
                    # Frozen window: hold the baseline, wait for the ramp to
                    # resume (core decrements its counter per projection).
                    print(
                        f"[HoldoutGate] iter={iteration} psnr={p:.3f} "
                        f"baseline={holdout_state['baseline']:.3f} "
                        f"eps={eps_eff:.3f} FROZEN "
                        f"left={int(core.spa_gate_freeze_cycles)}",
                        flush=True,
                    )
                else:
                    # Judge against the pre-drift baseline, THEN drift it on
                    # healthy cycles (drift-first would raise the effective
                    # threshold by 1/0.9).
                    dropped = holdout_state["baseline"] - p
                    if (
                        holdout_state["triggers"] < max_trig
                        and dropped > eps_eff
                    ):
                        holdout_state["triggers"] += 1
                        core.spa_gate_fired = (
                            getattr(core, "spa_gate_fired", 0) + 1
                        )
                        # Same source as the core ramp formula
                        # (spa_reinit_step, not the configured iteration) so
                        # the frozen progress matches the schedule it holds.
                        reinit_step = int(
                            getattr(core, "spa_reinit_step", iteration)
                        )
                        w = max(
                            1, int(getattr(cfg.model, "spa_post_window", 2000))
                        )
                        core.spa_gate_frozen_progress = min(
                            1.0,
                            max(0.0, (iteration - reinit_step) / float(w)),
                        )
                        core.spa_gate_freeze_cycles = int(
                            getattr(cfg.model, "spa_holdout_freeze_window", 2)
                        )
                        print(
                            f"[HoldoutGate] FIRED "
                            f"#{holdout_state['triggers']} iter={iteration} "
                            f"psnr={p:.3f} baseline={holdout_state['baseline']:.3f} "
                            f"eps={eps_eff:.3f} -> kappa ramp frozen at "
                            f"progress={core.spa_gate_frozen_progress:.3f}",
                            flush=True,
                        )
                        if int(getattr(core, "spa_post_base", 0)) <= 0:
                            print(
                                "[HoldoutGate] WARNING spa_post_base=0: no "
                                "post-phase ramp exists, freeze is inert",
                                flush=True,
                            )
                    else:
                        # Healthy cycle: the baseline drifts with the model,
                        # so slow training-wide improvements never trip it.
                        holdout_state["baseline"] = (
                            0.9 * holdout_state["baseline"] + 0.1 * p
                        )
                        print(
                            f"[HoldoutGate] iter={iteration} psnr={p:.3f} "
                            f"baseline={holdout_state['baseline']:.3f} "
                            f"eps={eps_eff:.3f} "
                            f"triggers={holdout_state['triggers']}",
                            flush=True,
                        )

        if (
            getattr(cfg.model, "semantic_enabled", False)
            and getattr(cfg.model, "semantic_cache_dir", None)
            and iteration == int(optim.update_until)
            and not getattr(model.core, "semantic_refreshed", False)
        ):
            from scaffold_gs.semantic_targets import refresh_semantic_targets

            print(f"[semantic] refreshing targets at iteration {iteration}",
                  flush=True)
            refresh_semantic_targets(
                model,
                dataset,
                cfg.model.semantic_cache_dir,
                pca_dims=8,
                min_views=cfg.model.semantic_min_visible_views,
                device=str(model.device),
            )

        if (
            getattr(cfg.model, "mini_splat_enabled", False)
            and iteration == int(getattr(cfg.model, "mini_splat_reinit_iter", 0))
            and not getattr(model.core, "mini_splat_done", False)
        ):
            reinit_fn = getattr(model, "mini_splat_reinit", None)
            if reinit_fn is not None:
                print(f"[MiniSplat] reinit at iteration {iteration}", flush=True)
                reinit_fn(dataset, background)
                model.core.mini_splat_done = True

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
