"""Mini-Splatting-style anchor spatial re-organization (depth-driven).

This is the anchor/HAC++ port of Mini-Splatting (Fang & Wang, ECCV 2024).
It implements the full two-stage spatial re-organization loop:

1. **depth reinitialization**: back-project rendered depth to surface points,
   voxelize, and add anchors into under-covered regions;
2. **blur split**: render per-pixel argmax contribution area using the packed
   gsplat intersection API, and split anchors whose maximum contribution area
   exceeds ``mini_splat_blur_threshold``;
3. **intersection-preserving simplification**: keep the anchors with the
   largest contribution area, then let SPA continue with an importance-weighted
   score so the fixed budget selects good placements.

The ``scaffold_gs/model.py`` change attaches ``gaussian_anchor_indices`` to
each rasterized Gaussian, giving the missing per-pixel contributor mapping
without modifying the packed gsplat CUDA kernel.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .model import BaseGaussianModel, NeuralGaussians


MINI_SPLAT_ALPHA_MIN = 0.05


def valid_depth_alpha_mask(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    alpha_min: float = MINI_SPLAT_ALPHA_MIN,
) -> torch.Tensor:
    """Mask valid surface pixels rendered by gsplat.

    gsplat fills missing/background depth with 0; the depth channel is also
    unreliable where accumulated alpha is negligible. Keeping only
    ``alpha > alpha_min`` and finite positive depth prevents invalid
    camera-origin points from being back-projected as anchors.
    """
    if depth.dim() == 4:
        depth = depth[0, :, :, 0]
    elif depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if alpha.dim() == 4:
        alpha = alpha[0, :, :, 0]
    elif alpha.dim() == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
    elif alpha.dim() == 3:
        alpha = alpha[..., 0]
    if alpha.dim() == 1:
        alpha = alpha.reshape(depth.shape)
    return torch.isfinite(depth) & (depth > 0) & (alpha > alpha_min)


def _backproject_depth(
    depth: torch.Tensor,  # [1, H, W, 1] camera-z (gsplat render_mode="D")
    cam,
    device: torch.device,
) -> torch.Tensor:
    """Back-project a depth map to world-space points -> [N, 3]."""
    h, w = cam.height, cam.width
    K = torch.from_numpy(cam.K).float().to(device)
    c2w = torch.from_numpy(cam.c2w).float().to(device)
    depth = depth[0, :, :, 0]  # [H, W]
    # Pixel grid in image coords (u right, v down).  gsplat mixes a small
    # convention shift; using K directly with the pixel center is standard and
    # well within voxel tolerance for a densification cue.
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    z = depth
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    ones = torch.ones_like(z)
    cam_pts = torch.stack([x_cam, y_cam, z, ones], dim=-1)  # [H,W,4]
    world = (c2w @ cam_pts.reshape(-1, 4).T).T  # [N,4]
    return world[:, :3]


def render_scene_depth(
    model: BaseGaussianModel,
    cam,
    background: torch.Tensor,
    device: torch.device,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[NeuralGaussians],
    Optional[torch.Tensor],
    Optional[dict],
]:
    """Render depth/alpha and return the rasterization meta plus anchor mapping."""
    from gsplat.rendering import rasterization

    visible_mask = model.prefilter_anchors(cam)
    if visible_mask.sum() == 0:
        return None, None, None, None, None
    gaussians = model.generate_gaussians(
        cam,
        visible_mask=visible_mask,
        is_training=False,
        appearance_id=0,
        step=0,
    )
    if gaussians.xyz.shape[0] == 0:
        return None, None, None, None, None
    viewmats, Ks = cam.to_gsplat(device)
    render_depths, render_alphas, meta = rasterization(
        means=gaussians.xyz,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=torch.zeros_like(gaussians.colors),
        viewmats=viewmats,
        Ks=Ks,
        width=cam.width,
        height=cam.height,
        near_plane=cam.near_plane,
        far_plane=cam.far_plane,
        render_mode="D",
        packed=True,
        tile_size=int(getattr(model.cfg, "tile_size", 16)),
    )
    return (
        render_depths,
        render_alphas,
        gaussians,
        gaussians.gaussian_anchor_indices,
        meta,
    )


def _voxel_subsample(
    points: torch.Tensor,
    voxel_size: float,
    max_new: int,
    device: torch.device,
) -> torch.Tensor:
    """Voxel-dedup points and deterministically cap at ``max_new``."""
    if points.shape[0] == 0 or max_new <= 0:
        return torch.zeros(0, 3, device=device)
    cell = torch.round(points / voxel_size).int()
    uniq, inv = torch.unique(cell, return_inverse=True, dim=0)
    scatter = torch.zeros(uniq.shape[0], 3, device=device)
    count = torch.zeros(uniq.shape[0], 1, device=device)
    scatter.scatter_reduce_(
        0,
        inv.view(-1, 1).expand(-1, 3),
        points,
        reduce="sum",
        include_self=False,
    )
    count.scatter_reduce_(
        0,
        inv.view(-1, 1),
        torch.ones(points.shape[0], 1, device=device),
        reduce="sum",
        include_self=False,
    )
    cell_xyz = (scatter / count.clamp_min(1.0)).cpu()
    if cell_xyz.shape[0] > max_new:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42)
        idx = torch.randperm(cell_xyz.shape[0], generator=rng)[:max_new]
        cell_xyz = cell_xyz[idx]
    return cell_xyz.to(device)


def _intersection_weights(
    meta: dict,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return per-Gaussian max-contribution pixel area, normalized by H*W.

    We reuse gsplat's packed intersection API so no CUDA kernel change is
    needed. The per-pixel argmax is recovered with the alpha-compositing
    formula from ``gsplat.cuda._torch_impl.accumulate``.
    """
    from gsplat import rasterize_to_indices_in_range

    means2d = meta["means2d"]
    conics = meta["conics"]
    opacities = meta["opacities"]
    # rasterization() may return means2d/conics/opacities without the leading
    # camera/batch dimension while isect_offsets carries it; normalize both to
    # the same image_dims expected by the low-level intersection API.
    if means2d.dim() == 2:
        means2d = means2d[None]
        conics = conics[None]
        opacities = opacities[None]
    width = int(meta["width"])
    height = int(meta["height"])
    tile_size = int(meta["tile_size"])
    n_gauss = int(means2d.shape[-2])
    if n_gauss == 0:
        return None
    transmittances = torch.ones(
        means2d.shape[:-2] + (height, width), device=device
    )
    gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        0,
        1_000_000_000,
        transmittances,
        means2d,
        conics,
        opacities,
        width,
        height,
        tile_size,
        meta["isect_offsets"],
        meta["flatten_ids"],
    )
    if gs_ids.numel() == 0:
        return torch.zeros(n_gauss, device=device)

    pixel_x = pixel_ids % width
    pixel_y = pixel_ids // width
    pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1) + 0.5
    deltas = pixel_coords - means2d[image_ids, gs_ids]
    c = conics[image_ids, gs_ids]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )
    alphas = torch.clamp_max(
        opacities[image_ids, gs_ids] * torch.exp(-sigmas), 1.0
    )
    ray_ids = image_ids * height * width + pixel_ids
    total_pixels = height * width

    order = torch.argsort(ray_ids, stable=True)
    ray = ray_ids[order]
    gs = gs_ids[order]
    alphas = alphas[order]
    log_one_minus = torch.log1p(-alphas.clamp(max=1.0 - 1e-8))
    group_start = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=device),
            ray[1:] != ray[:-1],
        ]
    )
    group_id = torch.cumsum(group_start.long(), dim=0) - 1
    cum = torch.cumsum(log_one_minus, dim=0)
    group_last = torch.zeros(
        int(group_id.max()) + 1, dtype=cum.dtype, device=device
    )
    group_last.scatter_reduce_(
        0, group_id, cum, reduce="amax", include_self=False
    )
    group_base = torch.cat(
        [torch.zeros(1, dtype=cum.dtype, device=device), group_last[:-1]]
    )[group_id]
    w = torch.exp(cum - group_base) * alphas

    best = torch.full(
        (total_pixels,), -torch.inf, dtype=w.dtype, device=device
    )
    best.scatter_reduce_(0, ray, w, reduce="amax", include_self=False)
    candidate = w == best[ray]
    key = torch.where(candidate, gs.long(), n_gauss)
    best_gid = torch.full(
        (total_pixels,), n_gauss, dtype=torch.long, device=device
    )
    best_gid.scatter_reduce_(
        0, ray, key, reduce="amin", include_self=False
    )
    valid = best_gid < n_gauss
    counts = torch.bincount(
        best_gid[valid], minlength=n_gauss + 1
    )[:n_gauss].to(device=device, dtype=torch.float32)
    return counts / float(total_pixels)


def compute_contribution_areas(
    model: BaseGaussianModel,
    cameras: List,
    background: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Per-anchor max-contribution pixel area over the sampled cameras."""
    n_total = int(model.num_anchors)
    area = torch.zeros(n_total, device=device)
    for cam in cameras:
        _, _, gaussians, gids, meta = render_scene_depth(
            model, cam, background, device
        )
        if gaussians is None or gids is None or meta is None:
            continue
        counts = _intersection_weights(meta, device)
        if counts is None:
            continue
        global_gids = meta.get("gaussian_ids")
        if global_gids is None or global_gids.numel() == 0:
            continue
        anchor_gids = gids.long()[global_gids.long()]
        area.scatter_add_(0, anchor_gids, counts)
    return area


def collect_blur_split_anchors(
    model: BaseGaussianModel,
    cameras: List,
    background: torch.Tensor,
    area: torch.Tensor,
    blur_threshold: float,
    voxel_size: float,
    max_new: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect child anchor positions for anchors with excessive area."""
    if max_new <= 0:
        return torch.zeros(0, 3, device=device)
    blur = area.squeeze(-1) > blur_threshold
    if blur.sum() == 0:
        return torch.zeros(0, 3, device=device)
    parts: List[torch.Tensor] = []
    for cam in cameras:
        _, _, gaussians, gids, meta = render_scene_depth(
            model, cam, background, device
        )
        if gaussians is None or gids is None or meta is None:
            continue
        global_gids = meta.get("gaussian_ids")
        if global_gids is None or global_gids.numel() == 0:
            continue
        active_anchor = gids.long()[global_gids.long()]
        active_xyz = gaussians.xyz[global_gids.long()]
        keep = blur[active_anchor]
        parts.append(active_xyz[keep])
    if not parts:
        return torch.zeros(0, 3, device=device)
    candidates = torch.cat(parts, dim=0)
    return _voxel_subsample(candidates, voxel_size, max_new, device)


def collect_depth_surface_anchors(
    model: BaseGaussianModel,
    cameras: List,
    background: torch.Tensor,
    voxel_size: float,
    max_new: int,
    device: torch.device,
) -> torch.Tensor:
    """Back-project depth over cameras -> candidate anchor positions [M, 3].

    Returns a voxel-deduped, well-spread set of world-space anchor centers
    sampled from the scene surface, capped at ``max_new``. The caller (official
    HAC++ core) appends these as anchors and pins the SPA budget beforehand.
    """
    all_candidate: List[torch.Tensor] = []
    for cam in cameras:
        depth, alpha, _, _, _ = render_scene_depth(
            model, cam, background, device
        )
        if depth is None or alpha is None:
            continue
        world = _backproject_depth(depth, cam, device)
        mask = valid_depth_alpha_mask(depth, alpha)
        world = world[mask.reshape(-1)]
        # Keep a sparse, well-spread subset of surface points.
        if world.shape[0] == 0:
            continue
        cell = torch.round(world / voxel_size).int()
        uniq, inv = torch.unique(cell, return_inverse=True, dim=0)
        # Mean world point per occupied cell (robust surface estimate).
        scatter = torch.zeros(uniq.shape[0], 3, device=device)
        count = torch.zeros(uniq.shape[0], 1, device=device)
        scatter.scatter_reduce_(
            0, inv.view(-1, 1).expand(-1, 3), world, reduce="sum", include_self=False
        )
        count.scatter_reduce_(
            0,
            inv.view(-1, 1),
            torch.ones(world.shape[0], 1, device=device),
            reduce="sum",
            include_self=False,
        )
        cell_xyz = scatter / count.clamp_min(1.0)
        all_candidate.append(cell_xyz)

    if not all_candidate:
        return torch.zeros(0, 3, device=device)
    candidates = torch.cat(all_candidate, dim=0)
    return _voxel_subsample(candidates, voxel_size, max_new, device)
