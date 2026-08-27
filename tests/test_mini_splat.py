"""Mini-Splatting depth-reinit config contract tests (CPU-safe).

GPU model-level checks (depth rendering, candidate collection, anchor append
and SPA-budget pinning) run on 5090 inside the playroom Stage-B smoke.
"""

from __future__ import annotations

import pytest
import torch

from scaffold_gs.config import ModelConfig
from scaffold_gs.model import NeuralGaussians
from scaffold_gs.mini_splat import valid_depth_alpha_mask


def test_mini_splat_default_on() -> None:
    cfg = ModelConfig()
    assert cfg.mini_splat_enabled is True
    assert cfg.mini_splat_reinit_iter == 15_000
    assert cfg.mini_splat_max_new == 4_000
    assert cfg.mini_splat_views == 8
    assert cfg.mini_splat_voxel == 0.0


def test_mini_splat_reinit_iter_must_be_positive() -> None:
    with pytest.raises(ValueError, match="mini_splat_reinit_iter"):
        ModelConfig(mini_splat_enabled=True, mini_splat_reinit_iter=0)


def test_mini_splat_max_new_non_negative() -> None:
    with pytest.raises(ValueError, match="mini_splat_max_new"):
        ModelConfig(mini_splat_enabled=True, mini_splat_max_new=-1)


def test_mini_splat_enables_cleanly_with_content_aware() -> None:
    cfg = ModelConfig(
        mini_splat_enabled=True,
        content_aware_quant=True,
        mini_splat_reinit_iter=15_000,
    )
    assert cfg.mini_splat_enabled is True
    assert cfg.mini_splat_reinit_iter == 15_000


def test_valid_depth_alpha_mask_filters_background_and_low_alpha() -> None:
    depth = torch.zeros(1, 4, 5, 1)
    alpha = torch.zeros(1, 4, 5, 1)
    depth[..., 2, 2, 0] = 1.0
    alpha[..., 2, 2, 0] = 0.5
    mask = valid_depth_alpha_mask(depth, alpha)
    assert mask.dtype == torch.bool
    assert mask.sum() == 1
    assert mask[2, 2].item() is True


def test_neural_gaussians_can_carry_anchor_map() -> None:
    g = NeuralGaussians(
        xyz=torch.zeros(2, 3),
        colors=torch.zeros(2, 3),
        opacities=torch.zeros(2),
        scales=torch.zeros(2, 3),
        quats=torch.zeros(2, 4),
        neural_opacity=torch.zeros(1, 2),
        selection_mask=torch.zeros(2, dtype=torch.bool),
        visible_mask=torch.ones(1, dtype=torch.bool),
        anchor_indices=torch.zeros(1, dtype=torch.long),
        gaussian_anchor_indices=torch.tensor([0, 0], dtype=torch.long),
    )
    assert g.gaussian_anchor_indices is not None
    assert g.gaussian_anchor_indices.shape == (2,)


def test_mini_splat_full_config_contract() -> None:
    cfg = ModelConfig(
        mini_splat_enabled=True,
        mini_splat_full=True,
        mini_splat_blur_threshold=0.02,
        mini_splat_importance_weight=0.5,
    )
    assert cfg.mini_splat_full is True
    assert 0.0 < cfg.mini_splat_blur_threshold < 1.0
    assert cfg.mini_splat_importance_weight >= 0.0

    with pytest.raises(ValueError, match="mini_splat_blur_threshold"):
        ModelConfig(mini_splat_blur_threshold=0.0)
