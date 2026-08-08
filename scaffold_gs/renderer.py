"""gsplat rasterization wrapper for Scaffold-GS neural Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .model import BaseGaussianModel, NeuralGaussians


@dataclass
class RenderOutput:
    """Result of rendering one camera."""

    image: torch.Tensor  # [1, H, W, 3]
    alpha: torch.Tensor  # [1, H, W, 1]
    meta: Dict[str, Any]
    gaussians: NeuralGaussians
    visible_mask: torch.Tensor  # [N] anchors in front of the camera


def _empty_output(
    camera, device: torch.device, n_offsets: int = 0
) -> RenderOutput:
    h, w = camera.height, camera.width
    gaussians = NeuralGaussians(
        xyz=torch.zeros(0, 3, device=device),
        colors=torch.zeros(0, 3, device=device),
        opacities=torch.zeros(0, device=device),
        scales=torch.zeros(0, 3, device=device),
        quats=torch.zeros(0, 4, device=device),
        neural_opacity=torch.zeros(0, n_offsets, device=device),
        selection_mask=torch.zeros(0, dtype=torch.bool, device=device),
        visible_mask=torch.zeros(0, dtype=torch.bool, device=device),
        anchor_indices=torch.zeros(0, dtype=torch.long, device=device),
    )
    return RenderOutput(
        image=torch.zeros(1, h, w, 3, device=device),
        alpha=torch.zeros(1, h, w, 1, device=device),
        meta={
            "means2d": torch.zeros(1, 0, 2, device=device),
            "radii": torch.zeros(1, 0, 2, device=device),
        },
        gaussians=gaussians,
        visible_mask=torch.zeros(0, dtype=torch.bool, device=device),
    )


def prefilter_anchors(
    model: BaseGaussianModel, camera
) -> torch.Tensor:
    """Return anchors that survive frustum culling.

    Uses the same rasterizer as rendering, but with the anchor's own scale
    (first 3 dims of ``scaling``), identity quaternions, unit opacity and
    depth-only output -- the gsplat equivalent of the official
    ``visible_filter`` pass.
    """
    from gsplat.rendering import rasterization

    p = model.anchor_params
    n = p.num_anchors
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=model.device)

    viewmats, Ks = camera.to_gsplat(model.device)
    scales = torch.exp(p.scaling[:, :3])
    quats = F.normalize(p.rotation, dim=-1)
    opacities = torch.ones(n, device=model.device)

    _, _, meta = rasterization(
        means=p.anchor,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=torch.zeros(n, 1, device=model.device),  # ignored for depth-only
        viewmats=viewmats,
        Ks=Ks,
        width=camera.width,
        height=camera.height,
        near_plane=camera.near_plane,
        far_plane=camera.far_plane,
        render_mode="D",
        packed=False,
    )
    radii = meta["radii"]  # [1, N, 2]
    return (radii[0] > 0).all(dim=-1)


def render(
    model: BaseGaussianModel,
    camera,
    background: torch.Tensor,
    is_training: bool = False,
    retain_grad: bool = False,
    appearance_id: Optional[int] = None,
    **kwargs,
) -> RenderOutput:
    """Render one camera with the decoded neural Gaussians."""
    from gsplat.rendering import rasterization

    visible_mask = model.prefilter_anchors(camera)
    if visible_mask.sum() == 0:
        return _empty_output(camera, model.device, model.cfg.n_offsets)

    gaussians = model.generate_gaussians(
        camera,
        visible_mask=visible_mask,
        is_training=is_training,
        appearance_id=appearance_id,
        **kwargs,
    )
    if gaussians.xyz.shape[0] == 0:
        out = _empty_output(camera, model.device, model.cfg.n_offsets)
        out.visible_mask = visible_mask
        out.gaussians = gaussians
        return out

    viewmats, Ks = camera.to_gsplat(model.device)
    render_colors, render_alphas, meta = rasterization(
        means=gaussians.xyz,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
        viewmats=viewmats,
        Ks=Ks,
        width=camera.width,
        height=camera.height,
        near_plane=camera.near_plane,
        far_plane=camera.far_plane,
        backgrounds=background[None],
        render_mode="RGB",
        packed=False,
        sh_degree=None,
    )
    if retain_grad:
        meta["means2d"].retain_grad()

    return RenderOutput(
        image=render_colors,
        alpha=render_alphas,
        meta=meta,
        gaussians=gaussians,
        visible_mask=visible_mask,
    )
