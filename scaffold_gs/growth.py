"""Anchor growing / pruning and optimizer-state bookkeeping.

Ported from the official Scaffold-GS ``scene/gaussian_model.py``, with two
deliberate adaptations:

- ``torch.scatter_reduce`` replaces the ``torch_scatter`` dependency.
- The offset-selection bookkeeping is corrected so gradient statistics map
  masked (decoded) Gaussians back to their global ``anchor * K + offset``
  indices even when some anchors were frustum-culled.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict

import torch
import torch.nn as nn

from .model import BaseGaussianModel, NeuralGaussians
from .utils import inverse_sigmoid

PER_ANCHOR_NAMES = ("anchor", "offset", "anchor_feat", "scaling", "rotation", "opacity")


def _is_anchor_group(name: str) -> bool:
    return name in PER_ANCHOR_NAMES


def _pad_stats(model: BaseGaussianModel) -> None:
    """Grow statistics buffers to match the current anchor count."""
    n = model.num_anchors
    k = model.cfg.n_offsets
    device = model.device
    if model.opacity_accum is None:
        model.opacity_accum = torch.zeros(n, 1, device=device)
    if model.opacity_accum.shape[0] < n:
        model.opacity_accum = torch.cat(
            [
                model.opacity_accum,
                torch.zeros(n - model.opacity_accum.shape[0], 1, device=device),
            ],
            dim=0,
        )
    if model.anchor_demon is None:
        model.anchor_demon = torch.zeros(n, 1, device=device)
    if model.anchor_demon.shape[0] < n:
        model.anchor_demon = torch.cat(
            [
                model.anchor_demon,
                torch.zeros(n - model.anchor_demon.shape[0], 1, device=device),
            ],
            dim=0,
        )
    if model.offset_gradient_accum is None:
        model.offset_gradient_accum = torch.zeros(n * k, 1, device=device)
    if model.offset_gradient_accum.shape[0] < n * k:
        model.offset_gradient_accum = torch.cat(
            [
                model.offset_gradient_accum,
                torch.zeros(
                    n * k - model.offset_gradient_accum.shape[0], 1, device=device
                ),
            ],
            dim=0,
        )
    if model.offset_denom is None:
        model.offset_denom = torch.zeros(n * k, 1, device=device)
    if model.offset_denom.shape[0] < n * k:
        model.offset_denom = torch.cat(
            [
                model.offset_denom,
                torch.zeros(n * k - model.offset_denom.shape[0], 1, device=device),
            ],
            dim=0,
        )
    model.max_radii2D = torch.zeros(n, device=device)


def cat_params_and_optimizer(
    model: BaseGaussianModel, tensors: Dict[str, torch.Tensor]
) -> None:
    """Append new rows to per-anchor params and matching optimizer states."""
    model.anchor_params.cat_(tensors)
    if model.optimizer is None:
        return
    for group in model.optimizer.param_groups:
        name = group["name"]
        if not _is_anchor_group(name) or name not in tensors:
            continue
        old = group["params"][0]
        new_param = getattr(model.anchor_params, name)
        state = model.optimizer.state.get(old)
        if state is not None:
            state["exp_avg"] = torch.cat(
                [state["exp_avg"], torch.zeros_like(tensors[name])], dim=0
            )
            state["exp_avg_sq"] = torch.cat(
                [state["exp_avg_sq"], torch.zeros_like(tensors[name])], dim=0
            )
            del model.optimizer.state[old]
            model.optimizer.state[new_param] = state
        group["params"][0] = new_param


def prune_params_and_optimizer(model: BaseGaussianModel, mask: torch.Tensor) -> None:
    """Keep per-anchor rows selected by ``mask`` (params + optimizer state)."""
    model.anchor_params.prune_(mask)
    if model.optimizer is None:
        return
    for group in model.optimizer.param_groups:
        name = group["name"]
        if not _is_anchor_group(name):
            continue
        old = group["params"][0]
        new_param = getattr(model.anchor_params, name)
        state = model.optimizer.state.get(old)
        if state is not None:
            state["exp_avg"] = state["exp_avg"][mask]
            state["exp_avg_sq"] = state["exp_avg_sq"][mask]
            del model.optimizer.state[old]
            model.optimizer.state[new_param] = state
        group["params"][0] = new_param


def training_statis(
    model: BaseGaussianModel,
    means2d: torch.Tensor,
    visibility_filter: torch.Tensor,
    gaussians: NeuralGaussians,
) -> None:
    """Accumulate opacity / 2D-gradient statistics for anchor refinement.

    Args:
        means2d: screen-space means from gsplat, ``[1, M, 2]`` (grad retained).
        visibility_filter: ``[1, M]`` bool, ``radii > 0`` for decoded Gaussians.
        gaussians: the ``NeuralGaussians`` object returned by the renderer.
    """
    assert model.opacity_accum is not None
    assert model.offset_gradient_accum is not None
    assert model.offset_denom is not None
    assert model.anchor_demon is not None

    k = model.cfg.n_offsets
    device = model.device
    vis = gaussians.visible_mask
    global_visible = torch.nonzero(vis).squeeze(-1)
    if global_visible.numel() == 0:
        return

    # Opacity / visit statistics for visible anchors.
    temp_opacity = gaussians.neural_opacity.clamp(min=0.0)  # [n, K]
    model.opacity_accum[global_visible] += temp_opacity.sum(dim=1, keepdim=True)
    model.anchor_demon[global_visible] += 1.0

    # Map masked (decoded) Gaussians back to global anchor*K+offset indices.
    selection_mask = gaussians.selection_mask  # [n*K]
    active_local = torch.nonzero(selection_mask).squeeze(-1)  # [M]
    local_anchor = active_local // k
    local_offset = active_local % k
    global_idx = global_visible[local_anchor] * k + local_offset

    vis2d = visibility_filter[0]  # [M]
    idx = global_idx[vis2d]
    if idx.numel() > 0:
        assert means2d.grad is not None, "means2d grad missing; retain_grad was not set"
        grad_norm = means2d.grad[0][vis2d, :2].norm(dim=-1, keepdim=True)
        model.offset_gradient_accum.index_add_(0, idx, grad_norm)
        model.offset_denom.index_add_(
            0, idx, torch.ones_like(grad_norm, device=device)
        )


def grow_anchors(
    model: BaseGaussianModel,
    grads_norm: torch.Tensor,
    threshold: float,
    offset_mask: torch.Tensor,
) -> None:
    """Hierarchical anchor growing (official ``anchor_growing``)."""
    device = model.device
    k = model.cfg.n_offsets
    feat_dim = model.cfg.feat_dim
    init_length = model.num_anchors * k

    for depth in range(model.cfg.update_depth):
        cur_threshold = threshold * (
            (model.cfg.update_hierachy_factor // 2) ** depth
        )
        candidate_mask = (grads_norm >= cur_threshold) & offset_mask
        rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (depth + 1))
        candidate_mask = candidate_mask & rand_mask

        length_inc = model.num_anchors * k - init_length
        if length_inc == 0:
            if depth > 0:
                continue
        else:
            candidate_mask = torch.cat(
                [
                    candidate_mask,
                    torch.zeros(length_inc, dtype=torch.bool, device=device),
                ],
                dim=0,
            )

        all_xyz = (
            model.anchor_params.anchor.unsqueeze(1)
            + model.anchor_params.offset * torch.exp(model.anchor_params.scaling[:, :3]).unsqueeze(1)
        )  # [N, K, 3]
        size_factor = model.cfg.update_init_factor // (
            model.cfg.update_hierachy_factor ** depth
        )
        cur_size = model.voxel_size * size_factor

        grid_coords = torch.round(model.anchor_params.anchor / cur_size).int()
        selected_xyz = all_xyz.view(-1, 3)[candidate_mask]
        selected_grid = torch.round(selected_xyz / cur_size).int()
        unique_grid, inverse = torch.unique(
            selected_grid, return_inverse=True, dim=0
        )

        # Remove candidate cells that already contain an anchor.
        remove = torch.zeros(unique_grid.shape[0], dtype=torch.bool, device=device)
        chunk_size = 4096
        for start in range(0, unique_grid.shape[0], chunk_size):
            chunk = unique_grid[start : start + chunk_size]
            dup = (chunk.unsqueeze(1) == grid_coords.unsqueeze(0)).all(-1).any(-1)
            remove[start : start + chunk_size] = dup
        keep = ~remove

        candidate_anchor = unique_grid[keep] * cur_size
        if candidate_anchor.shape[0] == 0:
            continue

        new_scaling = torch.log(
            torch.full((candidate_anchor.shape[0], 6), cur_size, device=device)
        )
        new_rotation = torch.zeros(candidate_anchor.shape[0], 4, device=device)
        new_rotation[:, 0] = 1.0
        new_opacities = inverse_sigmoid(
            torch.full((candidate_anchor.shape[0], 1), 0.1, device=device)
        )
        parent_feat = (
            model.anchor_params.anchor_feat.unsqueeze(1)
            .repeat(1, k, 1)
            .view(-1, feat_dim)[candidate_mask]
        )
        scatter_index = inverse.unsqueeze(1).expand(-1, feat_dim)
        new_feat = torch.zeros(
            unique_grid.shape[0], feat_dim, device=device
        ).scatter_reduce(
            0, scatter_index, parent_feat, reduce="amax", include_self=False
        )[keep]
        new_offsets = torch.zeros(
            candidate_anchor.shape[0], k, 3, device=device
        )

        cat_params_and_optimizer(
            model,
            {
                "anchor": candidate_anchor,
                "offset": new_offsets,
                "anchor_feat": new_feat,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "opacity": new_opacities,
            },
        )


def adjust_anchor(
    model: BaseGaussianModel,
    check_interval: int = 100,
    success_threshold: float = 0.8,
    grad_threshold: float = 0.0002,
    min_opacity: float = 0.005,
) -> None:
    """Grow then prune anchors (official ``adjust_anchor``)."""
    k = model.cfg.n_offsets
    device = model.device
    assert model.offset_gradient_accum is not None
    assert model.offset_denom is not None

    grads = model.offset_gradient_accum / model.offset_denom.clamp_min(1.0)
    grads[grads.isnan()] = 0.0
    grads_norm = grads.norm(dim=-1)
    offset_mask = (
        model.offset_denom > check_interval * success_threshold * 0.5
    ).squeeze(dim=1)

    grow_anchors(model, grads_norm, grad_threshold, offset_mask)

    # Reset statistics for the offsets we just examined, then pad for any
    # anchors that were grown.
    model.offset_denom[offset_mask] = 0.0
    model.offset_gradient_accum[offset_mask] = 0.0
    _pad_stats(model)

    # Prune anchors with consistently low opacity that were visited enough.
    prune_mask = (model.opacity_accum < min_opacity * model.anchor_demon).squeeze(1)
    anchors_mask = (
        model.anchor_demon > check_interval * success_threshold
    ).squeeze(1)
    prune_mask = prune_mask & anchors_mask

    # Reset per-anchor statistics for anchors that were visited this round
    # (must happen before pruning slices the buffers to a smaller size).
    if anchors_mask.any():
        model.opacity_accum[anchors_mask] = 0.0
        model.anchor_demon[anchors_mask] = 0.0

    if prune_mask.any():
        model.offset_denom = model.offset_denom.view(-1, k)[~prune_mask].reshape(-1, 1)
        model.offset_gradient_accum = model.offset_gradient_accum.view(-1, k)[
            ~prune_mask
        ].reshape(-1, 1)
        model.opacity_accum = model.opacity_accum[~prune_mask]
        model.anchor_demon = model.anchor_demon[~prune_mask]
        prune_params_and_optimizer(model, ~prune_mask)
    model.max_radii2D = torch.zeros(model.num_anchors, device=device)
