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

    max_width: Optional[int] = None
    """Cap image width like official HAC++ ``resolution=-1`` (e.g. 1600).
    When set and the original width exceeds it, images/K are rescaled to
    ``(max_width, round(h * max_width / w))``. Takes precedence over
    ``data_factor`` for sizing (data_factor still selects the image folder)."""


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

    tile_size: int = 16
    """gsplat rasterization tile size; larger values cut packed-intersection
    memory (intersections scale as 1/tile_size^2) at a small speed cost."""

    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4

    use_feat_bank: bool = False
    """Multi-resolution feature bank (official BungeeNeRF option; default off)."""

    appearance_dim: int = 32
    """Per-training-camera appearance embedding dim; 0 disables it."""

    ratio: int = 1
    """Sample every ``ratio``-th SfM point before voxelization."""

    # HAC++ hash-grid settings (defaults match the official HAC-plus args).
    n_features_per_level: int = 4
    log2_hashmap_size: int = 13
    log2_hashmap_size_2D: int = 15

    # I1: scale-aware hierarchical anchor-hash context (pure coordinate,
    # decoder-recomputable; default OFF until the I1 on/off ablation settles it).
    hierarchical_context: bool = False
    hierarchical_context_start_iter: int = 12_000

    # I2: content-aware formula quantization (default ON, formula mode only).
    content_aware_quant: bool = True
    content_aware_q_mode: str = "formula"
    complexity_scale: float = 0.35
    content_aware_start_iter: int = 20_000
    content_aware_ramp_iters: int = 10_000
    mlp_complexity_hidden: Optional[int] = None
    """Hidden width of the complexity MLP; None -> feat_dim // 2."""

    mlp_complexity_layers: int = 1
    """Number of hidden layers in the complexity MLP (>= 1)."""

    level_threshold_low: float = 0.33
    level_threshold_high: float = 0.66
    """Spatial-distance thresholds used by I1 level ids (decoder-recomputable)."""

    # I5: lattice vector quantization (VQ) + dithered quantization.
    vq_enabled: bool = False
    """Enable lattice vector quantization of feat/scaling/offsets."""

    vq_lattice: str = "d4"
    """Lattice type: ``z`` (scalar), ``d4`` (4D), ``e8`` (8D)."""

    vq_group_feat: int = 4
    """VQ group size for features (4 for D4, 8 for E8; leftover dims are
    quantized element-wise)."""

    vq_group_scaling: int = 4
    """VQ group size for scaling (6 dims -> 1 D4 group + 2 scalar leftovers)."""

    vq_group_offsets: int = 4
    """VQ group size for offsets (30 dims -> 7 D4 groups + 2 scalar leftovers)."""

    vq_content_aware: bool = False
    """Use the adaptive/content-aware per-group step (geometric mean) instead
    of the fixed base steps in the VQ path."""

    dither_enabled: bool = False
    """Subtractive dithering on top of VQ (seed is stored in the codec header)."""

    dither_seed: int = 0
    """Deterministic dither seed shared by encode and decode."""
    sensitivity_enabled: bool = False

    # I6: render-sensitivity weighted supervision (default OFF).
    sensitivity_weight: float = 1e-3
    """Weight of L_sens = MSE(pred multiplier, sensitivity target)."""

    sensitivity_ema: float = 0.99
    """EMA smoothing for per-anchor sensitivity gradient norms."""

    sensitivity_strength: float = 1.0
    """Bounded mapping strength: 1 + strength * tanh(.)."""

    sensitivity_start_iter: int = 20_000
    """First iteration at which sensitivity supervision/accumulation runs."""

    def __post_init__(self) -> None:
        if self.content_aware_q_mode != "formula":
            raise ValueError(
                "content_aware_q_mode must be 'formula' in PHG v1; "
                f"got {self.content_aware_q_mode!r}"
            )
        if self.mlp_complexity_layers < 1:
            raise ValueError("mlp_complexity_layers must be >= 1")
        if not (0.0 <= self.level_threshold_low < self.level_threshold_high <= 1.0):
            raise ValueError(
                "level thresholds must satisfy "
                "0 <= low < high <= 1"
            )
        if self.vq_lattice not in ("z", "d4", "e8"):
            raise ValueError(
                f"vq_lattice must be one of 'z', 'd4', 'e8'; got {self.vq_lattice!r}"
            )
        for name, g in (
            ("vq_group_feat", self.vq_group_feat),
            ("vq_group_scaling", self.vq_group_scaling),
            ("vq_group_offsets", self.vq_group_offsets),
        ):
            if g < 1:
                raise ValueError(f"{name} must be >= 1, got {g}")
        if self.dither_enabled and not self.vq_enabled:
            raise ValueError("dither_enabled requires vq_enabled=True")
        if self.vq_content_aware and not self.content_aware_quant:
            raise ValueError(
                "vq_content_aware requires content_aware_quant=True"
            )
        if self.sensitivity_enabled:
            if self.sensitivity_weight <= 0.0:
                raise ValueError("sensitivity_weight must be > 0 when enabled")
            if not self.content_aware_quant:
                raise ValueError(
                    "sensitivity_enabled requires content_aware_quant=True "
                    "(supervision targets mlp_complexity used by I2)"
                )


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

    # HAC++-specific schedules (mask / hash-grid / context MLPs).
    mask_lr_init: float = 0.01
    mask_lr_final: float = 0.0001
    mask_lr_delay_mult: float = 0.01
    mask_lr_max_steps: int = 30_000

    encoding_xyz_lr_init: float = 0.005
    encoding_xyz_lr_final: float = 0.00001
    encoding_xyz_lr_delay_mult: float = 0.33
    encoding_xyz_lr_max_steps: int = 30_000

    mlp_grid_lr_init: float = 0.005
    mlp_grid_lr_final: float = 0.00001
    mlp_grid_lr_delay_mult: float = 0.01
    mlp_grid_lr_max_steps: int = 30_000

    mlp_deform_lr_init: float = 0.005
    mlp_deform_lr_final: float = 0.0005
    mlp_deform_lr_delay_mult: float = 0.01
    mlp_deform_lr_max_steps: int = 30_000

    # I2 complexity MLP schedule.
    mlp_complexity_lr_init: float = 0.005
    mlp_complexity_lr_final: float = 0.0005
    mlp_complexity_lr_delay_mult: float = 0.01
    mlp_complexity_lr_max_steps: int = 30_000

    lambda_rate: float = 0.004
    """Rate-distortion weight for HAC++ (official default 0.004)."""

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
