"""Typed configuration objects for Scaffold-GS on gsplat.

All configs are plain dataclasses so they can be parsed by ``tyro`` from the
command line, stored inside checkpoints, and later re-used by HAC/HAC++.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Dataset / rendering settings."""

    data_dir: str = "data/garden"
    """Root of a COLMAP scene (contains ``images/`` and ``sparse/0/``)."""

    result_dir: str = "results/garden"
    """Where checkpoints, renders and logs are written."""

    data_factor: int = 4
    """Down-sampling factor applied to images and intrinsics."""

    test_every: int = 8
    """Every ``test_every``-th image (sorted by name) is held out for eval."""

    white_background: bool = False
    """Render background color: white when True, black otherwise."""

    near_plane: float = 0.01
    far_plane: float = 1e10

    preload_images: bool = True
    """Load all images into GPU memory up front (default, fast for small scenes)."""


@dataclass
class ModelConfig:
    """Scaffold-GS model architecture."""

    model_name: str = "scaffold_gs"
    """Registered model name; future options: ``hac``, ``hac_pp``."""

    feat_dim: int = 32
    """Anchor feature dimension."""

    n_offsets: int = 10
    """Number of neural Gaussians spawned per anchor (K in the paper)."""

    voxel_size: float = 0.001
    """Voxel size for anchor initialization; <=0 means median 1-NN distance."""

    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4

    use_feat_bank: bool = False
    """Multi-resolution feature bank (official BungeeNeRF option; default off)."""

    appearance_dim: int = 32
    """Per-training-camera appearance embedding dim; 0 disables it."""

    ratio: int = 1
    """Sample every ``ratio``-th SfM point before voxelization."""


@dataclass
class OptimConfig:
    """Training hyper-parameters (defaults follow the official Scaffold-GS)."""

    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [15_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [15_000, 30_000])

    # Per-parameter-group learning rates / schedules.
    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000

    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 30_000

    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002

    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 30_000

    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 30_000

    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 30_000

    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 30_000

    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.0005
    appearance_lr_delay_mult: float = 0.01
    appearance_lr_max_steps: int = 30_000

    lambda_dssim: float = 0.2
    scale_reg_lambda: float = 0.01

    # Anchor densification / pruning.
    start_stat: int = 500
    update_from: int = 1_500
    update_interval: int = 100
    update_until: int = 15_000
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002


@dataclass
class TrainConfig:
    """Top-level config for the ``train`` command."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    device: str = "cuda"
    seed: int = 42
    enable_tensorboard: bool = False


@dataclass
class EvalConfig:
    """Top-level config for the ``eval`` command."""

    ckpt: str
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda"
    out_dir: Optional[str] = None


@dataclass
class ExportConfig:
    """Top-level config for the ``export`` command."""

    ckpt: str
    out_dir: str
    device: str = "cuda"


@dataclass
class CompressConfig:
    """Top-level config for the ``compress`` command (HAC/HAC++ reserved)."""

    ckpt: str
    out_dir: str
    device: str = "cuda"
    codec: str = "none"
    """Codec name; ``none`` writes the uncompressed attribute baseline."""
