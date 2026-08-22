"""Semantic-prior (DINOv2) tests: config validation + target file format.

GPU model-level checks (supervision loss, growth/prune sync) run on 5090
inside the Stage-B smoke; here we keep the CPU-safe contract tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from scaffold_gs.config import ModelConfig


def test_semantic_enabled_requires_content_aware_quant() -> None:
    with pytest.raises(ValueError, match="content_aware_quant"):
        ModelConfig(semantic_enabled=True, content_aware_quant=False)


def test_semantic_enabled_requires_positive_weight() -> None:
    with pytest.raises(ValueError, match="semantic_weight"):
        ModelConfig(semantic_enabled=True, semantic_weight=0.0)


def test_semantic_signal_restricted_to_dino() -> None:
    with pytest.raises(ValueError, match="only 'dino'"):
        ModelConfig(semantic_enabled=True, semantic_signal="depth")


def test_semantic_target_dims_must_be_three() -> None:
    with pytest.raises(ValueError, match="exactly 3 dims"):
        ModelConfig(semantic_enabled=True, semantic_target_dims=[0, 1])


def test_default_semantic_config_is_off() -> None:
    cfg = ModelConfig()
    assert cfg.semantic_enabled is False
    assert cfg.semantic_proj_head is False
    assert cfg.semantic_signal == "dino"
    assert cfg.semantic_target_dims == [0, 3, 4]


def test_target_npz_format_roundtrip(tmp_path) -> None:
    """Exported target format consumed by HACPlusModel._load_semantic_targets."""
    path = tmp_path / "targets.npz"
    np.savez(
        path,
        target=np.zeros((10, 8), dtype=np.float32),
        cov=np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=bool),
    )
    data = np.load(path)
    assert data["target"].shape == (10, 8)
    assert data["cov"].shape == (10,)
    assert int(data["cov"].sum()) == 7
