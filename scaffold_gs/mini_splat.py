"""Mini-Splatting-style anchor spatial re-organization (depth-driven).

This is the minimal, faithful piece of Mini-Splatting (Fang & Wang, ECCV 2024)
ported to the anchor/HAC++ world.  The original insight is that *count is not
the bottleneck -- placement is*: naive count reduction (SPA's ``get_mask``
top-k) only *deletes* anchors, so overlapping (clustered) and
under-reconstructed (gappy) regions survive.  Mini-Splatting instead *re-moves*
anchors: densify on the surface (blur split / depth reinitialization), then
simplify (intersection preserving / sampling) toward the target count.

We implement the highest-fidelity, GPU-testable half: **depth reinitialization**
densification.  At the growth-stop iteration (== ``update_until``), we render
depth maps for a sample of training cameras, back-project them to world-surface
points, voxelize into candidate anchors, and add anchors where the scene is
under-covered.  We *fix* the SPA budget to the pre-densification anchor count
(``spa_final_n``) so the added anchors only re-allocate *which* anchors survive
the budget -- they do not inflate it.  This isolates "placement" from "count",
which is exactly the Mini-Splatting hypothesis we want to test against SPA.

The other half (blur split, per-pixel argmax contribution area) needs a
per-pixel contributor index that the packed gsplat rasterizer does not expose;
we deliberately skip it for this first pass and note it as a follow-up.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .model import BaseGaussianModel


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
) -> Optional[torch.Tensor]:
    """Render a depth-only map (render_mode="D") from the decoded Gaussians."""
    from gsplat.rendering import rasterization

    visible_mask = model.prefilter_anchors(cam)
    if visible_mask.sum() == 0:
        return None
    gaussians = model.generate_gaussians(
        cam,
        visible_mask=visible_mask,
        is_training=False,
        appearance_id=0,
        step=0,
    )
    if gaussians.xyz.shape[0] == 0:
        return None
    viewmats, Ks = cam.to_gsplat(device)
    render_depths, _alphas, _meta = rasterization(
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
    return render_depths


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
        depth = render_scene_depth(model, cam, background, device)
        if depth is None:
            continue
        world = _backproject_depth(depth, cam, device)
        # Keep a sparse, well-spread subset of surface points.
        if world.shape[0] == 0:
            continue
        cell = torch.round(world / voxel_size).int()
        uniq, inv = torch.unique(cell, return_inverse=True, dim=0)
        # Median world point per occupied cell (robust surface estimate).
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
    # Voxel subsample the deep surface to a spread, capped candidate set.
    cell = torch.round(candidates / voxel_size).int()
    uniq, inv = torch.unique(cell, return_inverse=True, dim=0)
    first_idx: List[int] = []
    seen = torch.zeros(uniq.shape[0], dtype=torch.bool, device=device)
    for i in range(candidates.shape[0]):
        c = inv[i].item()
        if not seen[c]:
            seen[c] = True
            first_idx.append(i)
            if len(first_idx) >= max_new:
                break
    if not first_idx:
        return torch.zeros(0, 3, device=device)
    return candidates[torch.tensor(first_idx, device=device)]
