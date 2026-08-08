"""Photometric losses used by Scaffold-GS.

Kept local (instead of importing ``gsplat.losses``) so the project works with
both gsplat 1.5.3 and newer versions: the official 3DGS SSIM implementation
is ~30 lines and version-independent.
"""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F
from torch import Tensor


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean L1 loss."""
    return F.l1_loss(pred, target)


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> Tensor:
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(3, 1, window_size, window_size).contiguous().to(device)


def ssim_loss(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
    """Return ``1 - SSIM`` for two ``(..., 3, H, W)`` images in [0, 1]."""
    channel = img1.shape[-3]
    window = _gaussian_window(window_size, 1.5, img1.device)
    window = window[:channel] if channel < 3 else window
    window = window.type_as(img1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return (1.0 - ssim_map).mean()
