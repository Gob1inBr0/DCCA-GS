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

    cache_images_cpu: bool = True
    """Cache resized images as uint8 in CPU RAM on first access (used when
    ``preload_images`` is False). 4-28: ~5.2GB RAM, saves ~150ms/step that
    would otherwise be spent decoding the full-resolution JPEG every step."""

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

    # Reserved future stages: enabling them must fail loudly, not half-implement.
    vq_enabled: bool = False
    dither_enabled: bool = False
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

    # Semantic-prior supervision (design: docs/语义先验实验设计.md v0.2).
    semantic_enabled: bool = False
    """Enable DINOv2 semantic supervision of mlp_complexity (training only)."""
    semantic_signal: str = "dino"
    """Which semantic prior is used; Stage A keeps only 'dino'."""
    semantic_weight: float = 1e-3
    """Weight of the semantic MSE term."""
    semantic_start_iter: int = 20_000
    """First iteration at which semantic supervision runs."""
    semantic_ramp_iters: int = 10_000
    """Ramp length for semantic supervision (informational; MSE applied as-is)."""
    semantic_min_visible_views: int = 3
    """Minimum visible views for a semantic target to be trusted."""
    semantic_proj_head: bool = False
    """T-A2: train an 8-dim projection head on mlp_complexity hidden and
    regress the DINO PCA target; head is dropped at inference (zero side
    info). False = T-A: supervise the 3 output logits directly."""
    semantic_target_path: Optional[str] = None
    """Path to the exported per-anchor DINO target npz (see
    semantic_gate.py --export-targets)."""
    semantic_cache_dir: Optional[str] = None
    """Path to the per-view DINO cache dir; when set, the trainer refreshes
    the per-anchor targets on the training model's own anchors once growth
    stops (iteration == update_until) -- the correct Stage-B protocol."""
    semantic_target_dims: List[int] = field(
        default_factory=lambda: [0, 3, 4]
    )
    """T-A only: which DINO PCA dims are used as the 3-dim supervision target
    (Stage A gate: dims 0/3/4 had the highest Pearson r)."""

    # S (GaussianSpa-style): training-side ADMM anchor sparsity.
    # NOTE: cell2 (MiniSplat+SPA) is now the PRIMARY training path -- SPA budget
    # + Mini-Splatting depth-reinit are both ON by default. (cell2 = 30.420dB /
    # 1.905MB vs cell1 baseline-SPA 30.221/1.859 on playroom, decoded.)
    spa_enabled: bool = True
    """Enforce an explicit anchor budget with ADMM hard projection."""
    spa_ratio: float = 0.85
    """Target fraction of surviving anchors (kappa = ratio * N)."""
    spa_rho: float = 1e-3
    """ADMM augmented-Lagrange weight for ||a - z + u||^2."""
    spa_u_clamp: float = 1.0
    """Clamp bound for the ADMM multiplier u."""

    # Mini-Splatting anchor spatial re-organization (depth reinitialization).
    # ON by default (primary path with SPA): depth-reinit re-places anchors onto
    # the scene surface so the SPA budget is spent on well-distributed anchors.
    mini_splat_enabled: bool = True
    """Densify anchors from back-projected depth at the growth-stop point."""
    mini_splat_reinit_iter: int = 15_000
    """Iteration at which depth reinitialization densification runs."""
    mini_splat_max_new: int = 4_000
    """Max new depth-reinit anchors added per reinit pass."""
    mini_splat_views: int = 8
    """Number of training cameras used to sample the scene surface."""
    mini_splat_voxel: float = 0.0
    """Voxel size for depth-surface anchor sampling; <=0 uses model voxel_size."""

    def __post_init__(self) -> None:
        if self.content_aware_q_mode != "formula":
            raise ValueError(
                "content_aware_q_mode must be 'formula' in PHG v1; "
                f"got {self.content_aware_q_mode!r}"
            )
        if self.mini_splat_enabled and self.mini_splat_reinit_iter < 1:
            raise ValueError("mini_splat_reinit_iter must be >= 1")
        if self.mini_splat_max_new < 0:
            raise ValueError("mini_splat_max_new must be >= 0")
        if self.mlp_complexity_layers < 1:
            raise ValueError("mlp_complexity_layers must be >= 1")
        if not (0.0 <= self.level_threshold_low < self.level_threshold_high <= 1.0):
            raise ValueError(
                "level thresholds must satisfy "
                "0 <= low < high <= 1"
            )
        for name, enabled in (
            ("vq_enabled", self.vq_enabled),
            ("dither_enabled", self.dither_enabled),
        ):
            if enabled:
                raise NotImplementedError(
                    f"{name} is reserved for a future PHG stage and is not "
                    "implemented in v1."
                )
        if self.sensitivity_enabled:
            if self.sensitivity_weight <= 0.0:
                raise ValueError("sensitivity_weight must be > 0 when enabled")
            if not self.content_aware_quant:
                raise ValueError(
                    "sensitivity_enabled requires content_aware_quant=True "
                    "(supervision targets mlp_complexity used by I2)"
                )
        if self.semantic_enabled:
            if self.semantic_weight <= 0.0:
                raise ValueError("semantic_weight must be > 0 when enabled")
            if not self.content_aware_quant:
                raise ValueError(
                    "semantic_enabled requires content_aware_quant=True "
                    "(supervision targets mlp_complexity used by I2)"
                )
            if self.semantic_signal != "dino":
                raise ValueError(
                    f"semantic_signal {self.semantic_signal!r} is closed; "
                    "only 'dino' passed the Stage-A gate"
                )
            if len(self.semantic_target_dims) != 3:
                raise ValueError(
                    "semantic_target_dims must have exactly 3 dims for T-A"
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
    attr_ctx: Optional[str] = None
    """Optional R4 predictor file (fitted by ``scripts/fit_attr_ctx.py``).
    When set, ``hac_pp`` codes scaling/offsets with conditional entropy
    parameters adjusted by the predictor and charges its payload to total_MB."""
    mask_keep_ratio: Optional[float] = None
    """Post-hoc encode-time anchor topk ratio (0,1]; None keeps all anchors.
    Used as the MaskTopk-only control group for SPA experiments."""
