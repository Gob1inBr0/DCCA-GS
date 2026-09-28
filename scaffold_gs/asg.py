"""Anisotropic spherical Gaussian (ASG) color decoding helpers.

This module implements the two-stage ASG appearance path used by the
``color_mode="asg"`` option:

1. ``mlp_asg`` maps anchor/view context to ASG parameters
   (axis, two sharpness values, signed amplitude and a small latent vector).
2. The ASG is evaluated on the unit view direction, producing an anisotropic
   view-dependent feature.
3. ``mlp_color2`` maps ``[latent, asg_feature, view_dir]`` to RGB.

The convention follows the anisotropic spherical Gaussian used in
Spec-Gaussian [2402.15870, Eq. (4)]:

    ASG(v | [x, y, z], [lambda, mu], xi)
        = xi * max(v . z, 0) * exp(-lambda (v . x)^2 - mu (v . y)^2)

The axes are generated deterministically from a single learned axis so the
same function can be re-run on the decoder side without side information.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def asg_per_lobe_width(latent_dim: int) -> int:
    """Number of per-lobe ASG parameters excluding the latent part.

    axis(3) + lambda_x(1) + lambda_y(1) + amplitude(3).
    """
    return 3 + 1 + 1 + 3


def build_asg_modules(
    color_in: int,
    feat_dim: int,
    n_offsets: int,
    lobes: int,
    latent_dim: int,
    hidden: int | None = None,
) -> Tuple[nn.Sequential, nn.Sequential]:
    """Create the two MLPs used by the ASG color path."""
    h = feat_dim if hidden is None or hidden <= 0 else int(hidden)
    per_lobe = asg_per_lobe_width(latent_dim) + latent_dim
    mlp_asg = nn.Sequential(
        nn.Linear(int(color_in), h),
        nn.ReLU(True),
        nn.Linear(h, int(n_offsets * lobes * per_lobe)),
    )
    mlp_color2 = nn.Sequential(
        nn.Linear(int(n_offsets * (latent_dim + 3 + 3)), h),
        nn.ReLU(True),
        nn.Linear(h, int(3 * n_offsets)),
        nn.Sigmoid(),
    )
    return mlp_asg, mlp_color2


def _orthonormal_basis(
    axis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Determine a right-handed orthonormal basis from a learned axis."""
    z = F.normalize(axis, dim=-1, eps=1e-8)
    ref = torch.tensor(
        [0.0, 0.0, 1.0], dtype=axis.dtype, device=axis.device
    ).expand_as(z)
    c = torch.cross(z, ref, dim=-1)
    fallback = torch.tensor(
        [1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device
    ).expand_as(z)
    c = c + (c.norm(dim=-1, keepdim=True) < 1e-6).to(c.dtype) * fallback
    x = F.normalize(c, dim=-1, eps=1e-8)
    y = torch.cross(z, x, dim=-1)
    return x, y, z


def evaluate_asg(
    params: torch.Tensor,
    view_dir: torch.Tensor,
    n_offsets: int,
    lobes: int = 1,
    latent_dim: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ASG lobes and aggregate the view-dependent feature.

    Args:
        params: ``[n, K*L*(8+latent_dim)]`` raw MLP output.
        view_dir: ``[n, 3]`` unit camera-to-anchor direction.

    Returns:
        ``(asg_feature [n, K, 3], latent [n, K, latent_dim])``.
    """
    n, _ = params.shape
    per_lobe = asg_per_lobe_width(latent_dim) + latent_dim
    p = params.reshape(n, n_offsets, lobes, per_lobe)
    axis_raw, lam_x_raw, lam_y_raw, amp, latent = torch.split(
        p,
        [3, 1, 1, 3, latent_dim],
        dim=-1,
    )
    x, y, z = _orthonormal_basis(axis_raw.reshape(-1, 3))
    x = x.reshape(n, n_offsets, lobes, 3)
    y = y.reshape(n, n_offsets, lobes, 3)
    z = z.reshape(n, n_offsets, lobes, 3)

    v = view_dir[:, None, None, :]  # [n, 1, 1, 3]
    lam_x = F.softplus(lam_x_raw)
    lam_y = F.softplus(lam_y_raw)
    dot_x = (v * x).sum(dim=-1, keepdim=True)
    dot_y = (v * y).sum(dim=-1, keepdim=True)
    dot_z = (v * z).sum(dim=-1, keepdim=True)
    smooth = F.relu(dot_z)
    gaussian = torch.exp(-lam_x * dot_x.square() - lam_y * dot_y.square())
    weights = smooth * gaussian  # [n, K, L, 1]
    asg_feature = (weights * amp).sum(dim=2)  # [n, K, 3]
    latent_agg = latent.mean(dim=2)  # [n, K, latent_dim]
    return asg_feature, latent_agg


def evaluate_asg_rgb(
    params: torch.Tensor,
    view_dir: torch.Tensor,
    mlp_color2: nn.Module,
    n_offsets: int,
    lobes: int = 1,
    latent_dim: int = 8,
) -> torch.Tensor:
    """Evaluate the complete ASG color path to RGB."""
    asg_feature, latent = evaluate_asg(
        params, view_dir, n_offsets, lobes=lobes, latent_dim=latent_dim
    )
    view = view_dir[:, None, :].expand(-1, n_offsets, -1)
    mlp_in = torch.cat([latent, asg_feature, view], dim=-1).reshape(
        params.shape[0], n_offsets * (latent_dim + 3 + 3)
    )
    return mlp_color2(mlp_in).reshape(
        params.shape[0], n_offsets, 3
    )
