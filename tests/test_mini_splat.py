"""Mini-Splatting depth-reinit config contract tests (CPU-safe).

GPU model-level checks (depth rendering, candidate collection, anchor append
and SPA-budget pinning) run on 5090 inside the playroom Stage-B smoke.
"""

from __future__ import annotations

import pytest

from scaffold_gs.config import ModelConfig


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
