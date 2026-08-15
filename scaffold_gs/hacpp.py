"""HAC++ module integrated with gsplat2hac.

This wraps the vendored official HAC++ core (``hacplus/``) so that:

- the model is registered as ``hac_pp`` and trains with a rate-distortion
  loss (photometric + ``lambda_rate * bits``) exactly like the official code,
- rendering is done with gsplat instead of ``diff_gaussian_rasterization``,
- ``compress --codec hac_pp`` entropy-codes anchor attributes (features,
  scaling, offsets, masks, binarized hash grid) with the official
  ``arithmetic`` range coder.

The CUDA extensions ``_gridencoder``, ``arithmetic`` and ``simple_knn`` must
be importable (the 5090 ``HAC_5090_a100`` conda env already provides them).
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from hacplus.scene.gaussian_model import (
        MAX_batch_size,
        GaussianModel as _OfficialHACModel,
    )
    from hacplus.utils.encodings import STE_multistep, get_binary_vxl_size
    from hacplus.utils.encodings_cuda import (
        decoder,
        decoder_gaussian_chunk,
        decoder_gaussian_mixed_chunk,
        encoder,
        encoder_gaussian_chunk,
        encoder_gaussian_mixed_chunk,
    )
    from hacplus.utils.gpcc_utils import (
        calculate_morton_order,
        compress_gpcc,
        decompress_gpcc,
    )
except Exception as exc:  # pragma: no cover - missing HAC++ extensions
    _OfficialHACModel = None
    _HACPP_IMPORT_ERROR = exc

from .codec import CompressionCodec
from .config import ModelConfig, OptimConfig
from .hac_core import HACCoreView
from .model import BaseGaussianModel, NeuralGaussians
from .utils import inverse_sigmoid, knn_distances, median_nn_distance, voxelize_points


def _tensor_sha256(t: torch.Tensor) -> str:
    data = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _load_override(value, n: int, device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        arr = np.load(value)
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float32).reshape(n, 1)
    return torch.from_numpy(arr).to(device)


def anchor_codec_order(model) -> torch.Tensor:
    """Global anchor indices in codec order (mask_anchor + Morton sort).

    Any per-anchor side-channel (e.g. I6 ``q_override_*``) must be written in
    this exact order so encode and decode see the same per-anchor mapping.
    """
    core = model.core
    mask_anchor = core.get_mask_anchor.to(torch.bool)[:, 0]
    anchor = core.get_anchor.detach()[mask_anchor]
    anchor_int = torch.round(anchor / model.voxel_size)
    sorted_indices = calculate_morton_order(anchor_int)
    return torch.nonzero(mask_anchor).squeeze(-1)[sorted_indices]


def sensitivity_multiplier(z, strength: float):
    """Bounded I6 Q multiplier ``1 + strength * tanh(-z)``.

    Clamped to [0.1, 2.0]: near-zero Q steps make the arithmetic coder's CDF
    produce NaN (observed at strength >= 1 with the relative z-score), and
    unbounded multipliers would make the strength sweep degenerate.
    """
    return (1.0 + float(strength) * torch.tanh(-z)).clamp(0.1, 2.0)


class _ParamsView:
    """Duck-typed accessor so the shared gsplat prefilter/renderer can consume
    the HAC++ core without a Scaffold-style ``AnchorParams`` module."""

    def __init__(self, model: "HACPlusModel") -> None:
        self._m = model

    @property
    def anchor(self) -> torch.Tensor:
        return self._m._view.get_anchor()

    @property
    def scaling(self) -> torch.Tensor:
        return self._m._view.get_scaling()

    @property
    def rotation(self) -> torch.Tensor:
        return self._m._view.get_rotation()

    @property
    def num_anchors(self) -> int:
        return self._m._view.num_anchors


def _empty_gaussians(model: "HACPlusModel", n_total: int) -> NeuralGaussians:
    device = model.device
    k = model.cfg.n_offsets
    empty = torch.zeros(0, device=device)
    return NeuralGaussians(
        xyz=empty.clone().reshape(0, 3),
        colors=empty.clone().reshape(0, 3),
        opacities=empty.clone(),
        scales=empty.clone().reshape(0, 3),
        quats=empty.clone().reshape(0, 4),
        neural_opacity=empty.clone().reshape(0, k),
        selection_mask=empty.clone().bool(),
        visible_mask=torch.zeros(n_total, dtype=torch.bool, device=device),
        anchor_indices=empty.clone().long(),
    )


class HACPlusModel(BaseGaussianModel):
    """HAC++ (hash-grid assisted context) model rendered with gsplat."""

    model_name = "hac_pp"

    def __init__(self, cfg: ModelConfig, device: str) -> None:
        super().__init__(cfg, device)
        if _OfficialHACModel is None:
            raise ImportError(
                "HAC++ core is unavailable in this environment. "
                "Use the HAC_5090_a100 conda env (gridencoder + arithmetic). "
                f"Reason: {_HACPP_IMPORT_ERROR}"
            )
        self.core = _OfficialHACModel(
            feat_dim=cfg.feat_dim,
            n_offsets=cfg.n_offsets,
            voxel_size=cfg.voxel_size,
            update_depth=cfg.update_depth,
            update_init_factor=cfg.update_init_factor,
            update_hierachy_factor=cfg.update_hierachy_factor,
            use_feat_bank=cfg.use_feat_bank,
            n_features_per_level=cfg.n_features_per_level,
            log2_hashmap_size=cfg.log2_hashmap_size,
            log2_hashmap_size_2D=cfg.log2_hashmap_size_2D,
            resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
            resolutions_list_2D=(130, 258, 514, 1026),
            ste_binary=True,
            ste_multistep=False,
            add_noise=False,
            Q=1,
            use_2D=True,
            decoded_version=False,
            is_synthetic_nerf=False,
            hierarchical_context=cfg.hierarchical_context,
            hierarchical_context_start_iter=cfg.hierarchical_context_start_iter,
            content_aware_quant=cfg.content_aware_quant,
            content_aware_q_mode=cfg.content_aware_q_mode,
            complexity_scale=cfg.complexity_scale,
            content_aware_start_iter=cfg.content_aware_start_iter,
            content_aware_ramp_iters=cfg.content_aware_ramp_iters,
            mlp_complexity_hidden=cfg.mlp_complexity_hidden,
            mlp_complexity_layers=cfg.mlp_complexity_layers,
            level_threshold_low=cfg.level_threshold_low,
            level_threshold_high=cfg.level_threshold_high,
        )
        self.core.to(self.device)
        self._view = HACCoreView(self.core)
        self.anchor_params = _ParamsView(self)
        self.optim_cfg: Optional[OptimConfig] = None
        self.voxel_size = cfg.voxel_size

    # ------------------------------------------------------------------
    # BaseGaussianModel interface
    # ------------------------------------------------------------------

    @property
    def num_anchors(self) -> int:
        return self.core.get_anchor.shape[0]

    @property
    def opacity_accum(self) -> torch.Tensor:
        return self.core.opacity_accum

    @opacity_accum.setter
    def opacity_accum(self, value: torch.Tensor) -> None:
        self.core.opacity_accum = value

    @property
    def offset_gradient_accum(self) -> torch.Tensor:
        return self.core.offset_gradient_accum

    @offset_gradient_accum.setter
    def offset_gradient_accum(self, value: torch.Tensor) -> None:
        self.core.offset_gradient_accum = value

    @property
    def offset_denom(self) -> torch.Tensor:
        return self.core.offset_denom

    @offset_denom.setter
    def offset_denom(self, value: torch.Tensor) -> None:
        self.core.offset_denom = value

    @property
    def anchor_demon(self) -> torch.Tensor:
        return self.core.anchor_demon

    @anchor_demon.setter
    def anchor_demon(self, value: torch.Tensor) -> None:
        self.core.anchor_demon = value

    @property
    def max_radii2D(self) -> torch.Tensor:
        return self.core.max_radii2D

    @max_radii2D.setter
    def max_radii2D(self, value: torch.Tensor) -> None:
        self.core.max_radii2D = value

    def state_dict(self, *args, **kwargs):
        sd = self.core.state_dict(*args, **kwargs)
        sd["_x_bound_min"] = self._view.x_bound_min
        sd["_x_bound_max"] = self._view.x_bound_max
        sd["_decoded_version"] = torch.tensor(int(self._view.decoded_version))
        sd["_phg_state"] = self._view.state_tensors()
        sd["_sensitivity_state"] = self._view.sensitivity_state()
        return sd

    def load_state_dict(self, *args, **kwargs):
        sd = dict(args[0])
        if not any(key.startswith("mlp_complexity.") for key in sd):
            raise RuntimeError(
                "PHG v1 codec requires an mlp_complexity checkpoint; "
                "old pre-PHG checkpoints are not codec-compatible."
            )
        if "_x_bound_min" in sd:
            self._view.x_bound_min = sd.pop("_x_bound_min").to(self.device)
            self._view.x_bound_max = sd.pop("_x_bound_max").to(self.device)
            self._view.decoded_version = bool(sd.pop("_decoded_version").item())
        if "_phg_state" in sd:
            self._view.load_state_tensors(sd.pop("_phg_state"))
        if "_sensitivity_state" in sd:
            self._view.load_sensitivity_state(sd.pop("_sensitivity_state"))
        for key in (
            "_anchor",
            "_offset",
            "_mask",
            "_anchor_feat",
            "_scaling",
            "_rotation",
            "_opacity",
        ):
            if key in sd and not isinstance(getattr(self.core, key, None), nn.Parameter):
                setattr(
                    self.core,
                    key,
                    nn.Parameter(
                        sd[key].to(self.device),
                        requires_grad=key not in ("_rotation", "_opacity"),
                    ),
                )
        result = self.core.load_state_dict(sd, strict=kwargs.get("strict", True))
        if self.core.sensitivity_feat.numel() == 0 and self.num_anchors > 0:
            n = self.num_anchors
            self.core.sensitivity_feat = torch.zeros(n, 1, device=self.device)
            self.core.sensitivity_scaling = torch.zeros(n, 1, device=self.device)
            self.core.sensitivity_offsets = torch.zeros(n, 1, device=self.device)
            self.core.sensitivity_mean = torch.zeros(3, device=self.device)
            self.core.sensitivity_var = torch.ones(3, device=self.device)
        return result

    def train(self, mode: bool = True):
        # The official HAC++ GaussianModel overrides train()/eval() without a
        # mode argument, so route calls explicitly.
        if mode:
            self.core.train()
        else:
            self.core.eval()
        return self

    def eval(self):
        self.core.eval()
        return self

    def save_ply(self, path) -> None:
        self.core.save_ply(path)

    def save_mlp_checkpoints(self, path) -> None:
        self.core.save_mlp_checkpoints(path)

    def load_ply(self, path) -> None:
        self.core.load_ply_sparse_gaussian(path)

    def set_appearance(self, num_cameras: int) -> None:
        del num_cameras  # HAC++ has no appearance embedding.

    def init_from_pcd(
        self, points: np.ndarray, rgbs: np.ndarray, spatial_lr_scale: float
    ) -> None:
        core = self.core
        self.spatial_lr_scale = float(spatial_lr_scale)
        points = np.asarray(points, dtype=np.float32)
        if self.cfg.ratio > 1:
            points = points[:: self.cfg.ratio]
        if self.voxel_size <= 0:
            self.voxel_size = median_nn_distance(points)
            print(f"[HAC++] Auto voxel_size = {self.voxel_size:.6f}")
        points = voxelize_points(points, self.voxel_size)
        n = len(points)
        k = self.cfg.n_offsets
        print(f"[HAC++] {n} anchors after voxelization.")

        device = self.device
        anchor = torch.from_numpy(points).float().to(device)
        dist = knn_distances(points, k=2)[:, 1]
        scales_log = np.log(np.clip(dist, 1e-7, None))[:, None].repeat(6, axis=1)
        rot = torch.zeros(n, 4, device=device)
        rot[:, 0] = 1.0

        self._view.anchor = nn.Parameter(anchor, requires_grad=True)
        self._view.offset = nn.Parameter(
            torch.zeros(n, k, 3, device=device), requires_grad=True
        )
        self._view.mask = nn.Parameter(
            torch.ones(n, k + 1, 1, device=device), requires_grad=True
        )
        self._view.anchor_feat = nn.Parameter(
            torch.zeros(n, self.cfg.feat_dim, device=device), requires_grad=True
        )
        self._view.scaling = nn.Parameter(
            torch.from_numpy(scales_log).float().to(device), requires_grad=True
        )
        self._view.rotation = nn.Parameter(rot, requires_grad=False)
        self._view.opacity = nn.Parameter(
            inverse_sigmoid(torch.full((n, 1), 0.1, device=device)),
            requires_grad=False,
        )
        core.spatial_lr_scale = self.spatial_lr_scale
        core.opacity_accum = torch.zeros(n, 1, device=device)
        core.offset_gradient_accum = torch.zeros(n * k, 1, device=device)
        core.offset_denom = torch.zeros(n * k, 1, device=device)
        core.anchor_demon = torch.zeros(n, 1, device=device)
        core.max_radii2D = torch.zeros(n, device=device)
        core.sensitivity_feat = torch.zeros(n, 1, device=device)
        core.sensitivity_scaling = torch.zeros(n, 1, device=device)
        core.sensitivity_offsets = torch.zeros(n, 1, device=device)
        core.sensitivity_mean = torch.zeros(3, device=device)
        core.sensitivity_var = torch.ones(3, device=device)
        core.update_anchor_bound()
        # Guard against anchor grids whose min/max sit exactly on 0, which the
        # official calc_interp_feat assertion rejects.
        if (self._view.x_bound_min == 0).any():
            self._view.x_bound_min = self._view.x_bound_min - 1e-4
        if (self._view.x_bound_max == 0).any():
            self._view.x_bound_max = self._view.x_bound_max + 1e-4

    def create_optimizer(self, optim_cfg: OptimConfig) -> None:
        self.optim_cfg = optim_cfg
        args = SimpleNamespace(
            percent_dense=0.01,
            position_lr_init=optim_cfg.position_lr_init,
            position_lr_final=optim_cfg.position_lr_final,
            position_lr_delay_mult=optim_cfg.position_lr_delay_mult,
            position_lr_max_steps=optim_cfg.position_lr_max_steps,
            offset_lr_init=optim_cfg.offset_lr_init,
            offset_lr_final=optim_cfg.offset_lr_final,
            offset_lr_delay_mult=optim_cfg.offset_lr_delay_mult,
            offset_lr_max_steps=optim_cfg.offset_lr_max_steps,
            mask_lr_init=optim_cfg.mask_lr_init,
            mask_lr_final=optim_cfg.mask_lr_final,
            mask_lr_delay_mult=optim_cfg.mask_lr_delay_mult,
            mask_lr_max_steps=optim_cfg.mask_lr_max_steps,
            feature_lr=optim_cfg.feature_lr,
            opacity_lr=optim_cfg.opacity_lr,
            scaling_lr=optim_cfg.scaling_lr,
            rotation_lr=optim_cfg.rotation_lr,
            mlp_opacity_lr_init=optim_cfg.mlp_opacity_lr_init,
            mlp_opacity_lr_final=optim_cfg.mlp_opacity_lr_final,
            mlp_opacity_lr_delay_mult=optim_cfg.mlp_opacity_lr_delay_mult,
            mlp_opacity_lr_max_steps=optim_cfg.mlp_opacity_lr_max_steps,
            mlp_cov_lr_init=optim_cfg.mlp_cov_lr_init,
            mlp_cov_lr_final=optim_cfg.mlp_cov_lr_final,
            mlp_cov_lr_delay_mult=optim_cfg.mlp_cov_lr_delay_mult,
            mlp_cov_lr_max_steps=optim_cfg.mlp_cov_lr_max_steps,
            mlp_color_lr_init=optim_cfg.mlp_color_lr_init,
            mlp_color_lr_final=optim_cfg.mlp_color_lr_final,
            mlp_color_lr_delay_mult=optim_cfg.mlp_color_lr_delay_mult,
            mlp_color_lr_max_steps=optim_cfg.mlp_color_lr_max_steps,
            mlp_featurebank_lr_init=optim_cfg.mlp_featurebank_lr_init,
            mlp_featurebank_lr_final=optim_cfg.mlp_featurebank_lr_final,
            mlp_featurebank_lr_delay_mult=optim_cfg.mlp_featurebank_lr_delay_mult,
            mlp_featurebank_lr_max_steps=optim_cfg.mlp_featurebank_lr_max_steps,
            encoding_xyz_lr_init=optim_cfg.encoding_xyz_lr_init,
            encoding_xyz_lr_final=optim_cfg.encoding_xyz_lr_final,
            encoding_xyz_lr_delay_mult=optim_cfg.encoding_xyz_lr_delay_mult,
            encoding_xyz_lr_max_steps=optim_cfg.encoding_xyz_lr_max_steps,
            mlp_grid_lr_init=optim_cfg.mlp_grid_lr_init,
            mlp_grid_lr_final=optim_cfg.mlp_grid_lr_final,
            mlp_grid_lr_delay_mult=optim_cfg.mlp_grid_lr_delay_mult,
            mlp_grid_lr_max_steps=optim_cfg.mlp_grid_lr_max_steps,
            mlp_deform_lr_init=optim_cfg.mlp_deform_lr_init,
            mlp_deform_lr_final=optim_cfg.mlp_deform_lr_final,
            mlp_deform_lr_delay_mult=optim_cfg.mlp_deform_lr_delay_mult,
            mlp_deform_lr_max_steps=optim_cfg.mlp_deform_lr_max_steps,
            mlp_complexity_lr_init=optim_cfg.mlp_complexity_lr_init,
            mlp_complexity_lr_final=optim_cfg.mlp_complexity_lr_final,
            mlp_complexity_lr_delay_mult=optim_cfg.mlp_complexity_lr_delay_mult,
            mlp_complexity_lr_max_steps=optim_cfg.mlp_complexity_lr_max_steps,
        )
        self.core.training_setup(args)
        self.optimizer = self.core.optimizer

    def update_learning_rate(self, iteration: int) -> None:
        self.core.update_learning_rate(iteration)

    def prefilter_anchors(self, camera) -> torch.Tensor:
        from .renderer import prefilter_anchors

        return prefilter_anchors(self, camera)

    def generate_gaussians(
        self,
        camera,
        visible_mask: Optional[torch.Tensor] = None,
        is_training: bool = False,
        appearance_id: Optional[int] = None,
        step: int = 0,
        retain_grad: bool = False,
    ) -> NeuralGaussians:
        del appearance_id
        core = self.core
        device = self.device
        if step > 0:
            core.current_step = int(step)
            core.current_iter = int(step)
        n_total = core.get_anchor.shape[0]
        if n_total == 0:
            return _empty_gaussians(self, n_total)
        if visible_mask is None:
            visible_mask = torch.ones(n_total, dtype=torch.bool, device=device)
        anchor_indices = torch.nonzero(visible_mask).squeeze(-1)
        n = anchor_indices.shape[0]
        if n == 0:
            return _empty_gaussians(self, n_total)

        anchor = core.get_anchor[anchor_indices]
        feat = self._view.anchor_feat[anchor_indices]
        grid_offsets = self._view.offset[anchor_indices]
        grid_scaling = core.get_scaling[anchor_indices]
        binary_grid_masks = core.get_mask[anchor_indices]  # [n, K, 1]
        k = self.cfg.n_offsets

        Q_feat = 1.0
        Q_scaling = 0.001
        Q_offsets = 0.2
        bit_per_param = None
        bit_per_feat_param = None
        bit_per_scaling_param = None
        bit_per_offsets_param = None
        complexity_logits = None

        if is_training:
            if 3000 < step <= 10000:
                feat = feat + (torch.rand_like(feat) - 0.5) * Q_feat
                grid_scaling = grid_scaling + (torch.rand_like(grid_scaling) - 0.5) * Q_scaling
                grid_offsets = grid_offsets + (torch.rand_like(grid_offsets) - 0.5) * Q_offsets
            if step == 10000:
                core.update_anchor_bound()
            if step > 10000:
                # Quantization noise on the rendering attributes (official
                # renderer behavior): the model must learn to survive the
                # hard quantization applied at eval/encode time.
                feat_context_orig = core.calc_context_feat(
                    anchor, anchor_indices=anchor_indices, caller="generate_gaussians"
                )
                ctx_out = core.get_grid_mlp(feat_context_orig)
                (
                    _mean,
                    _scale,
                    _prob,
                    _mean_scaling,
                    _scale_scaling,
                    _mean_offsets,
                    _scale_offsets,
                    qa,
                    qs,
                    qo,
                ) = torch.split(
                    ctx_out,
                    [
                        self.cfg.feat_dim,
                        self.cfg.feat_dim,
                        self.cfg.feat_dim,
                        6,
                        6,
                        3 * k,
                        3 * k,
                        1,
                        1,
                        1,
                    ],
                    dim=-1,
                )
                Q_feat = 1.0 * (1 + torch.tanh(qa.repeat(1, self.cfg.feat_dim)))
                Q_scaling = 0.001 * (1 + torch.tanh(qs.repeat(1, 6)))
                Q_offsets = 0.2 * (
                    1 + torch.tanh(qo.repeat(1, 3 * k))
                ).view(-1, k, 3)
                if core.is_content_aware_quant_active():
                    (
                        Q_feat,
                        Q_scaling,
                        Q_offsets,
                        _,
                        _,
                        _,
                        complexity_logits,
                    ) = core._codec_apply_content_aware_quant_params(
                        "generate_gaussians",
                        anchor,
                        binary_grid_masks,
                        Q_feat,
                        Q_scaling,
                        Q_offsets,
                        None,
                        None,
                        None,
                        _mean_scaling,
                        _mean_offsets,
                    )
                feat = feat + (torch.rand_like(feat) - 0.5) * Q_feat
                grid_scaling = grid_scaling + (
                    torch.rand_like(grid_scaling) - 0.5
                ) * Q_scaling
                grid_offsets = grid_offsets + (
                    torch.rand_like(grid_offsets) - 0.5
                ) * Q_offsets
                (
                    bit_per_param,
                    bit_per_feat_param,
                    bit_per_scaling_param,
                    bit_per_offsets_param,
                ) = self._estimate_rate_terms(anchor, feat, grid_scaling, grid_offsets)
        elif not self._view.decoded_version:
            feat_context = core.calc_context_feat(
                anchor, anchor_indices=anchor_indices, caller="generate_gaussians"
            )
            (
                _mean,
                _scale,
                _prob,
                _mean_scaling,
                _scale_scaling,
                _mean_offsets,
                _scale_offsets,
                Q_feat_adj,
                Q_scaling_adj,
                Q_offsets_adj,
            ) = torch.split(
                core.get_grid_mlp(feat_context),
                [
                    self.cfg.feat_dim,
                    self.cfg.feat_dim,
                    self.cfg.feat_dim,
                    6,
                    6,
                    3 * k,
                    3 * k,
                    1,
                    1,
                    1,
                ],
                dim=-1,
            )
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj.repeat(1, self.cfg.feat_dim)))
            Q_scaling = Q_scaling * (
                1 + torch.tanh(Q_scaling_adj.repeat(1, 6))
            )
            Q_offsets = Q_offsets * (
                1 + torch.tanh(Q_offsets_adj.repeat(1, 3 * k))
            ).view(-1, k, 3)
            if core.is_content_aware_quant_active():
                (
                    Q_feat,
                    Q_scaling,
                    Q_offsets,
                    _,
                    _,
                    _,
                    _,
                ) = core._codec_apply_content_aware_quant_params(
                    "generate_gaussians",
                    anchor,
                    binary_grid_masks,
                    Q_feat,
                    Q_scaling,
                    Q_offsets,
                    None,
                    None,
                    None,
                    _mean_scaling,
                    _mean_offsets,
                )
            feat = STE_multistep.apply(feat, Q_feat, self._view.anchor_feat.mean()).detach()
            grid_scaling = STE_multistep.apply(
                grid_scaling, Q_scaling, core.get_scaling.mean()
            ).detach()
            grid_offsets = STE_multistep.apply(
                grid_offsets, Q_offsets, self._view.offset.mean()
            ).detach()

        sens_active = (
            self.cfg.sensitivity_enabled
            and is_training
            and step >= self.cfg.sensitivity_start_iter
        )
        if sens_active:
            feat.retain_grad()
            grid_scaling.retain_grad()
            grid_offsets.retain_grad()

        # Decode neural Gaussians in anchor chunks so peak memory does not
        # scale with the total anchor count (same math as the official renderer).
        camera_center = camera.camera_center(device)
        neural_opacity_parts = []
        selection_parts = []
        xyz_parts, color_parts, opacity_parts, scale_parts, quat_parts = (
            [],
            [],
            [],
            [],
            [],
        )
        chunk = 8_192
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            a = anchor[start:end]
            f = feat[start:end]
            go = grid_offsets[start:end]
            gs = grid_scaling[start:end]
            bm = binary_grid_masks[start:end]
            c = end - start

            ob_view = a - camera_center
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            ob_view = ob_view / ob_dist.clamp_min(1e-8)
            cat_local_view = torch.cat([f, ob_view, ob_dist], dim=-1)

            no = core.get_opacity_mlp(cat_local_view)  # [c, K]
            sel = (no.reshape(-1) > 0.0)
            neural_opacity_parts.append(no)
            selection_parts.append(sel)

            color = core.get_color_mlp(cat_local_view).reshape(c * k, 3)[sel]
            scale_rot = core.get_cov_mlp(cat_local_view).reshape(c * k, 7)[sel]
            offsets_c = go.reshape(-1, 3)[sel]
            scaling_repeat = (
                gs.unsqueeze(1).repeat(1, k, 1).reshape(c * k, 6)[sel]
            )
            anchor_repeat = a.unsqueeze(1).repeat(1, k, 1).reshape(c * k, 3)[sel]
            scales_c = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
            quats_c = F.normalize(scale_rot[:, 3:7], dim=-1)
            xyz_c = anchor_repeat + offsets_c * scaling_repeat[:, :3]

            binary_flat = bm.reshape(-1)[sel]
            opacity_c = no.reshape(-1)[sel]
            if is_training:
                opacity_c = opacity_c * binary_flat
                scales_c = scales_c * binary_flat.unsqueeze(-1)
            else:
                keep = binary_flat.bool()
                xyz_c = xyz_c[keep]
                color = color[keep]
                opacity_c = opacity_c[keep]
                scales_c = scales_c[keep]
                quats_c = quats_c[keep]

            xyz_parts.append(xyz_c)
            color_parts.append(color)
            opacity_parts.append(opacity_c)
            scale_parts.append(scales_c)
            quat_parts.append(quats_c)

        neural_opacity = torch.cat(neural_opacity_parts, dim=0)
        selection_mask = torch.cat(selection_parts, dim=0)
        xyz = torch.cat(xyz_parts, dim=0)
        color = torch.cat(color_parts, dim=0)
        opacity = torch.cat(opacity_parts, dim=0)
        scales = torch.cat(scale_parts, dim=0)
        quats = torch.cat(quat_parts, dim=0)

        return NeuralGaussians(
            xyz=xyz,
            colors=color,
            opacities=opacity,
            scales=scales,
            quats=quats,
            neural_opacity=neural_opacity,
            selection_mask=selection_mask,
            visible_mask=visible_mask,
            anchor_indices=anchor_indices,
            bit_per_param=bit_per_param,
            bit_per_feat_param=bit_per_feat_param,
            bit_per_scaling_param=bit_per_scaling_param,
            bit_per_offsets_param=bit_per_offsets_param,
            pre_quant_feat=feat if sens_active else None,
            pre_quant_scaling=grid_scaling if sens_active else None,
            pre_quant_offsets=grid_offsets if sens_active else None,
            complexity_logits=complexity_logits,
        )

    def render(self, camera, background, **kwargs):
        from .renderer import render

        return render(self, camera, background, **kwargs)

    def training_statis(
        self,
        means2d: torch.Tensor,
        visibility_filter: torch.Tensor,
        gaussians: NeuralGaussians,
        width: float,
        gaussian_ids: torch.Tensor,
        height: float,
    ) -> None:
        """Shared growth statistics; see ``growth.accumulate_growth_stats``."""
        from .growth import accumulate_growth_stats

        accumulate_growth_stats(
            self, means2d, visibility_filter, gaussians, width, gaussian_ids, height
        )

    def adjust_anchor(
        self,
        check_interval: int,
        success_threshold: float,
        grad_threshold: float,
        min_opacity: float,
    ) -> None:
        self.core.adjust_anchor(
            check_interval=check_interval,
            success_threshold=success_threshold,
            grad_threshold=grad_threshold,
            min_opacity=min_opacity,
        )

    def rate_loss_term(self, gaussians: NeuralGaussians, iteration: int) -> torch.Tensor:
        del iteration
        if gaussians.bit_per_param is None or self.optim_cfg is None:
            return torch.zeros((), device=self.device)
        _, bit_hash, _, _ = get_binary_vxl_size(
            (self.core.get_encoding_params() + 1) / 2
        )
        denom = self.num_anchors * (
            self.cfg.feat_dim + 6 + 3 * self.cfg.n_offsets
        )
        return self.optim_cfg.lambda_rate * (
            gaussians.bit_per_param + bit_hash / denom
        )

    def sensitivity_supervision(self, gaussians: NeuralGaussians) -> torch.Tensor:
        """L_sens = weight * MSE(pred multiplier, bounded sensitivity target).

        ``complexity_logits`` is kept differentiable so gradients reach
        ``mlp_complexity``; the target side (previous-step EMA z-score) is
        detached.
        """
        core = self.core
        if (
            not self.cfg.sensitivity_enabled
            or gaussians.complexity_logits is None
            or self.cfg.sensitivity_weight <= 0.0
            or core.current_step < self.cfg.sensitivity_start_iter
        ):
            return torch.zeros((), device=self.device)
        idx = gaussians.anchor_indices
        ema = torch.stack(
            [
                core.sensitivity_feat[idx].squeeze(-1),
                core.sensitivity_scaling[idx].squeeze(-1),
                core.sensitivity_offsets[idx].squeeze(-1),
            ],
            dim=-1,
        )
        # I6 mapping follows the design doc's relative normalization
        # s_norm = accum / (mean(accum) + eps). The variance-EMA z-score
        # never converges from its unit init at alpha=0.99 and flattens the
        # signal (grad norms span several orders of magnitude per anchor).
        z_score = (ema - core.sensitivity_mean) / core.sensitivity_mean.clamp_min(
            1e-12
        )
        strength = self.cfg.sensitivity_strength
        pred = 1.0 + strength * torch.tanh(gaussians.complexity_logits)
        target = sensitivity_multiplier(z_score, strength).detach()
        return self.cfg.sensitivity_weight * torch.nn.functional.mse_loss(
            pred, target
        )

    def accumulate_sensitivity(self, gaussians: NeuralGaussians) -> None:
        """EMA-update per-anchor render-sensitivity gradient norms."""
        if (
            not self.cfg.sensitivity_enabled
            or gaussians.pre_quant_feat is None
            or gaussians.pre_quant_feat.grad is None
        ):
            return
        core = self.core
        alpha = self.cfg.sensitivity_ema
        idx = gaussians.anchor_indices
        grads = [
            gaussians.pre_quant_feat.grad.norm(dim=-1, keepdim=True),
            gaussians.pre_quant_scaling.grad.norm(dim=-1, keepdim=True),
            gaussians.pre_quant_offsets.grad.flatten(1).norm(
                dim=-1, keepdim=True
            ),
        ]
        for ema_tensor, g in zip(
            (
                core.sensitivity_feat,
                core.sensitivity_scaling,
                core.sensitivity_offsets,
            ),
            grads,
        ):
            ema_tensor.mul_(alpha)
            ema_tensor.index_add_(0, idx, (1.0 - alpha) * g)
        batch_mean = torch.stack([g.mean() for g in grads])
        batch_var = torch.stack([g.var(unbiased=False) for g in grads])
        core.sensitivity_mean.mul_(alpha).add_((1.0 - alpha) * batch_mean)
        core.sensitivity_var.mul_(alpha).add_((1.0 - alpha) * batch_var)

    # ------------------------------------------------------------------
    # Export / codec support
    # ------------------------------------------------------------------

    def export_attributes(self) -> Dict[str, Any]:
        decoder_state = self._view.decoder_state()
        return {
            "model_name": self.model_name,
            "anchor": self._view.anchor.detach().cpu().clone(),
            "offset": self._view.offset.detach().cpu().clone(),
            "mask": self._view.mask.detach().cpu().clone(),
            "anchor_feat": self._view.anchor_feat.detach().cpu().clone(),
            "scaling": self._view.scaling.detach().cpu().clone(),
            "rotation": self._view.rotation.detach().cpu().clone(),
            "opacity": self._view.opacity.detach().cpu().clone(),
            "decoder": decoder_state,
            "config": self.cfg.__dict__.copy(),
            "voxel_size": float(self.voxel_size),
            "spatial_lr_scale": float(self.spatial_lr_scale),
            "x_bound_min": self._view.x_bound_min.detach().cpu().clone(),
            "x_bound_max": self._view.x_bound_max.detach().cpu().clone(),
            "decoded_version": bool(self._view.decoded_version),
        }

    @classmethod
    def from_attributes(cls, attrs: Dict[str, Any], device: str) -> "HACPlusModel":
        cfg = ModelConfig(**attrs["config"])
        model = cls(cfg, device)
        model.voxel_size = attrs["voxel_size"]
        model.spatial_lr_scale = attrs["spatial_lr_scale"]
        view = model._view
        view.anchor = nn.Parameter(attrs["anchor"].to(device), requires_grad=True)
        view.offset = nn.Parameter(attrs["offset"].to(device), requires_grad=True)
        view.mask = nn.Parameter(attrs["mask"].to(device), requires_grad=True)
        view.anchor_feat = nn.Parameter(
            attrs["anchor_feat"].to(device), requires_grad=True
        )
        view.scaling = nn.Parameter(attrs["scaling"].to(device), requires_grad=True)
        view.rotation = nn.Parameter(attrs["rotation"].to(device), requires_grad=False)
        view.opacity = nn.Parameter(attrs["opacity"].to(device), requires_grad=False)
        view.x_bound_min = attrs["x_bound_min"].to(device)
        view.x_bound_max = attrs["x_bound_max"].to(device)
        view.decoded_version = bool(attrs.get("decoded_version", False))
        view.load_decoder_state(attrs["decoder"])
        return model

    # ------------------------------------------------------------------
    # HAC++ entropy encoding / decoding
    # ------------------------------------------------------------------

    def _estimate_rate_terms(self, anchor, feat, grid_scaling, grid_offsets):
        """Official 5%-subsample entropy estimate (used by the RD loss)."""
        core = self.core
        k = self.cfg.n_offsets
        choose = torch.rand_like(core.get_anchor[:, 0]) <= 0.05
        anchor_c = core.get_anchor[choose]
        feat_c = self._view.anchor_feat[choose]
        offsets_c = self._view.offset[choose]
        scaling_c = core.get_scaling[choose]
        masks_c = core.get_mask[choose]
        mask_anchor_c = core.get_mask_anchor[choose]

        ctx = core.calc_context_feat(anchor_c, caller="_estimate_rate_terms")
        out = core.get_grid_mlp(ctx)
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, qa, qs, qo = torch.split(
            out,
            [
                self.cfg.feat_dim,
                self.cfg.feat_dim,
                self.cfg.feat_dim,
                6,
                6,
                3 * k,
                3 * k,
                1,
                1,
                1,
            ],
            dim=-1,
        )
        qa = qa.repeat(1, self.cfg.feat_dim)
        qs = qs.repeat(1, 6)
        qo = qo.repeat(1, 3 * k)
        Q_feat = 1.0 * (1 + torch.tanh(qa))
        Q_scaling = 0.001 * (1 + torch.tanh(qs))
        Q_offsets = 0.2 * (1 + torch.tanh(qo))
        if core.is_content_aware_quant_active():
            (
                Q_feat,
                Q_scaling,
                Q_offsets,
                _,
                _,
                _,
                _,
            ) = core._codec_apply_content_aware_quant_params(
                "_estimate_rate_terms",
                anchor_c,
                masks_c,
                Q_feat,
                Q_scaling,
                Q_offsets,
                None,
                None,
                None,
                mean_scaling,
                mean_offsets,
            )

        feat_c = feat_c + (torch.rand_like(feat_c) - 0.5) * Q_feat
        mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
            feat_c, torch.cat([mean, scale, prob], dim=-1)
        )
        probs = torch.softmax(torch.stack([prob, prob_adj], dim=-1), dim=-1)
        scaling_c = scaling_c + (torch.rand_like(scaling_c) - 0.5) * Q_scaling
        offsets_c = offsets_c + (torch.rand_like(offsets_c) - 0.5) * Q_offsets.view(
            -1, k, 3
        )
        offsets_c = offsets_c.view(-1, 3 * k)
        masks_c = masks_c.repeat(1, 1, 3).view(-1, 3 * k)

        bit_feat = core.EG_mix_prob_2.forward(
            feat_c,
            mean,
            mean_adj,
            scale,
            scale_adj,
            probs[..., 0],
            probs[..., 1],
            Q=Q_feat,
            x_mean=self._view.anchor_feat.mean(),
        )
        bit_feat = bit_feat * mask_anchor_c
        bit_scaling = core.entropy_gaussian.forward(
            scaling_c, mean_scaling, scale_scaling, Q_scaling, core.get_scaling.mean()
        )
        bit_scaling = bit_scaling * mask_anchor_c
        bit_offsets = core.entropy_gaussian.forward(
            offsets_c,
            mean_offsets,
            scale_offsets,
            Q_offsets,
            self._view.offset.mean(),
        )
        bit_offsets = bit_offsets * mask_anchor_c * masks_c

        s_feat, n_feat = torch.sum(bit_feat), bit_feat.numel()
        s_scaling, n_scaling = torch.sum(bit_scaling), bit_scaling.numel()
        s_offsets, n_offsets = torch.sum(bit_offsets), bit_offsets.numel()
        bit_per_param = (s_feat + s_scaling + s_offsets) / (
            n_feat + n_scaling + n_offsets
        )
        return (
            bit_per_param,
            s_feat / n_feat,
            s_scaling / n_scaling,
            s_offsets / n_offsets,
        )

    @torch.no_grad()
    def encode_attributes(
        self,
        out_dir: Path,
        q_scale_feat: float = 1.0,
        q_scale_scaling: float = 1.0,
        q_scale_offsets: float = 1.0,
        mask_keep_ratio: Optional[float] = None,
        q_override_feat=None,
        q_override_scaling=None,
        q_override_offsets=None,
    ) -> Dict[str, Any]:
        """Entropy-code anchor attributes with the official arithmetic codec.

        Args:
            out_dir: directory where bitstream artifacts are written.
            q_scale_feat / q_scale_scaling / q_scale_offsets: post-hoc
                multipliers applied to the learned quantization steps on both
                encode and decode. 1.0 preserves official behavior; larger
                values coarsen quantization (fewer bits, more distortion).
            mask_keep_ratio: optional post-hoc anchor pruning. If in (0, 1),
                keep only the top fraction of anchors ranked by the learned
                anchor-mask rate instead of the official ``mask_anchor`` rule.
                ``None`` or 1.0 preserves official behavior.
        """
        from hacplus.scene.gaussian_model import bit2MB_scale
        from hacplus.utils.codec_consistency import (
            CODEC_HEADER_FILENAME,
            CONTENT_AWARE_Q_META_FILENAME,
            FORMULA_INPUT_VERSION,
        )

        core = self.core
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        k = self.cfg.n_offsets
        cg = int(core.feat_channel_group)
        device = self.device

        n_total = self._view.anchor.shape[0]
        if mask_keep_ratio is not None and mask_keep_ratio < 1.0:
            if not (0.0 < mask_keep_ratio < 1.0):
                raise ValueError(
                    f"mask_keep_ratio must be in (0, 1), got {mask_keep_ratio}"
                )
            mask_rate = core.get_mask.mean(dim=1)[:, 0]  # [N] soft anchor score
            keep_n = max(1, int(round(n_total * mask_keep_ratio)))
            keep = torch.zeros(n_total, dtype=torch.bool, device=device)
            keep[torch.topk(mask_rate, keep_n).indices] = True
            mask_anchor = keep
        else:
            mask_anchor = core.get_mask_anchor.to(torch.bool)[:, 0]
        anchor = core.get_anchor[mask_anchor]
        feat = self._view.anchor_feat[mask_anchor]
        offsets = self._view.offset[mask_anchor]
        scaling = core.get_scaling[mask_anchor]
        masks = core.get_mask[mask_anchor]  # [N, K, 1]
        N = anchor.shape[0]

        anchor_int = torch.round(anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(anchor_int)
        anchor_int = anchor_int[sorted_indices]
        feat = feat[sorted_indices]
        offsets = offsets[sorted_indices]
        scaling = scaling[sorted_indices]
        masks = masks[sorted_indices]
        ov_feat = _load_override(q_override_feat, N, device)
        ov_scaling = _load_override(q_override_scaling, N, device)
        ov_offsets = _load_override(q_override_offsets, N, device)
        means_strings = compress_gpcc(anchor_int)
        np.savez_compressed(
            out_dir / "xyz_gpcc.npz",
            means_strings=means_strings,
            voxel_size=np.float32(self.voxel_size),
        )
        bits_xyz = (out_dir / "xyz_gpcc.npz").stat().st_size * 8
        torch.save(self._view.x_bound_min, out_dir / "x_bound_min.pkl")
        torch.save(self._view.x_bound_max, out_dir / "x_bound_max.pkl")
        anchor = anchor_int.float() * self.voxel_size

        codec_header = {
            "format": "phg_v1",
            "codec": "hac_pp",
            "codec_iteration": int(core.current_step),
            "num_anchors": int(N),
            "num_anchors_total": int(n_total),
            "model": {
                "feat_dim": self.cfg.feat_dim,
                "n_offsets": k,
                "voxel_size": float(self.voxel_size),
                "hierarchical_context": bool(core.hierarchical_context),
                "hierarchical_context_start_iter": int(
                    core.hierarchical_context_start_iter
                ),
                "content_aware_quant": bool(core.content_aware_quant),
                "content_aware_q_mode": str(core.content_aware_q_mode),
                "complexity_scale": float(core.complexity_scale),
                "content_aware_start_iter": int(core.content_aware_start_iter),
                "content_aware_ramp_iters": int(core.content_aware_ramp_iters),
                "mlp_complexity_hidden": (
                    None
                    if core.mlp_complexity_hidden is None
                    else int(core.mlp_complexity_hidden)
                ),
                "mlp_complexity_layers": int(core.mlp_complexity_layers),
                "level_threshold_low": float(core.level_threshold_low),
                "level_threshold_high": float(core.level_threshold_high),
            },
            "formula_input_version": FORMULA_INPUT_VERSION,
            "anchor_int_sha256": _tensor_sha256(anchor_int),
            "masks_sha256": _tensor_sha256(masks),
        }
        with open(out_dir / CODEC_HEADER_FILENAME, "w") as f:
            json.dump(codec_header, f, indent=2, sort_keys=True)

        if core.content_aware_quant:
            q_meta = {
                "mode": str(core.content_aware_q_mode),
                "version": FORMULA_INPUT_VERSION,
                "complexity_scale": float(core.complexity_scale),
                "start_iter": int(core.content_aware_start_iter),
                "ramp_iters": int(core.content_aware_ramp_iters),
            }
            with open(out_dir / CONTENT_AWARE_Q_META_FILENAME, "w") as f:
                json.dump(q_meta, f, indent=2, sort_keys=True)

        steps = math.ceil(N / MAX_batch_size)
        bit_feat_list: list = []
        bit_scaling_list: list = []
        bit_offsets_list: list = []

        for s in range(steps):
            start = s * MAX_batch_size
            end = min((s + 1) * MAX_batch_size, N)
            feat_b = str(out_dir / f"feat_{s}.b")
            scaling_b = str(out_dir / f"scaling_{s}.b")
            offsets_b = str(out_dir / f"offsets_{s}.b")

            anchor_slice = anchor[start:end]
            ctx = core.calc_context_feat(anchor_slice, caller="encode_attributes")
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, qa, qs, qo = torch.split(
                core.get_grid_mlp(ctx),
                [
                    self.cfg.feat_dim,
                    self.cfg.feat_dim,
                    self.cfg.feat_dim,
                    6,
                    6,
                    3 * k,
                    3 * k,
                    1,
                    1,
                    1,
                ],
                dim=-1,
            )
            qa = qa.repeat(1, self.cfg.feat_dim)
            qs = qs.repeat(1, 6)
            qo = qo.repeat(1, 3 * k)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = scale_scaling.contiguous().view(-1).clamp(min=1e-9)
            scale_offsets = scale_offsets.contiguous().view(-1).clamp(min=1e-9)
            Q_feat = q_scale_feat * 1.0 * (1 + torch.tanh(qa))
            Q_scaling = q_scale_scaling * 0.001 * (1 + torch.tanh(qs))
            Q_offsets = q_scale_offsets * 0.2 * (1 + torch.tanh(qo))
            if core.is_content_aware_quant_active():
                masks_slice = masks[start:end]
                (
                    Q_feat,
                    Q_scaling,
                    Q_offsets,
                    _,
                    _,
                    _,
                    _,
                ) = core._codec_apply_content_aware_quant_params(
                    "encode_attributes",
                    anchor_slice,
                    masks_slice,
                    Q_feat,
                    Q_scaling,
                    Q_offsets,
                    None,
                    None,
                    None,
                    mean_scaling.view(-1, 6),
                    mean_offsets.view(-1, 3 * k),
                )
            if ov_feat is not None:
                Q_feat = Q_feat * ov_feat[start:end]
            if ov_scaling is not None:
                Q_scaling = Q_scaling * ov_scaling[start:end]
            if ov_offsets is not None:
                Q_offsets = Q_offsets * ov_offsets[start:end]
            Q_feat_flat = Q_feat.contiguous().view(-1)
            Q_scaling_flat = Q_scaling.contiguous().view(-1)
            Q_offsets_flat = Q_offsets.contiguous().view(-1)

            # features (channel-context, cg channels per step)
            feat_slice = feat[start:end]
            feat_q = STE_multistep.apply(feat_slice, Q_feat, self._view.anchor_feat.mean())
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = scale.clamp(min=1e-9)
            bit_feat = 0
            for cc in range(self.cfg.feat_dim // cg):
                mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
                    feat_q, mean_scale, to_dec=cc
                )
                probs = torch.softmax(
                    torch.stack([prob[:, cc * cg : cc * cg + cg], prob_adj], dim=-1),
                    dim=-1,
                )
                feat_tmp = feat_q[:, cc * cg : cc * cg + cg].contiguous().view(-1)
                Q_tmp = Q_feat[:, cc * cg : cc * cg + cg].contiguous().view(-1)
                bit_feat += encoder_gaussian_mixed_chunk(
                    feat_tmp,
                    [
                        mean[:, cc * cg : cc * cg + cg].contiguous().view(-1),
                        mean_adj.contiguous().view(-1),
                    ],
                    [
                        scale[:, cc * cg : cc * cg + cg].contiguous().view(-1),
                        scale_adj.contiguous().view(-1),
                    ],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_tmp,
                    file_name=feat_b.replace(".b", f"_{cc}.b"),
                    chunk_size=500_000,
                )
            bit_feat_list.append(bit_feat)

            # scaling
            scaling_slice = scaling[start:end].view(-1)
            scaling_q = STE_multistep.apply(
                scaling_slice, Q_scaling_flat, core.get_scaling.mean()
            )
            bit_scaling_list.append(
                encoder_gaussian_chunk(
                    scaling_q,
                    mean_scaling,
                    scale_scaling,
                    Q_scaling_flat,
                    file_name=scaling_b,
                    chunk_size=100_000,
                )
            )

            # offsets (masked)
            mask_slice = masks[start:end].repeat(1, 1, 3).view(-1, 3 * k).view(-1)
            offsets_slice = offsets[start:end].view(-1, 3 * k).view(-1)
            offsets_q = STE_multistep.apply(
                offsets_slice, Q_offsets_flat, self._view.offset.mean()
            )
            offsets_q[~mask_slice.bool()] = 0.0
            bit_offsets_list.append(
                encoder_gaussian_chunk(
                    offsets_q[mask_slice.bool()],
                    mean_offsets[mask_slice.bool()],
                    scale_offsets[mask_slice.bool()],
                    Q_offsets_flat[mask_slice.bool()],
                    file_name=offsets_b,
                    chunk_size=100_000,
                )
            )

        hash_params = core.get_encoding_params()
        bit_hash = encoder(
            ((hash_params.view(-1) + 1) / 2), file_name=str(out_dir / "hash.b")
        )
        bit_masks = encoder(masks, file_name=str(out_dir / "masks.b"))

        # Official HAC++ size accounting: decoder MLP weights at 32 bit/param
        # plus the xyz bounds (2 x [3] float32), see get_mlp_size().
        bit_mlp = (
            sum(p.numel() for n, p in core.named_parameters() if "mlp" in n)
            * 32
        )
        bit_bounds = 32 * 3 * 2
        total_bits = (
            bits_xyz
            + sum(bit_feat_list)
            + sum(bit_scaling_list)
            + sum(bit_offsets_list)
            + bit_hash
            + bit_masks
            + bit_mlp
            + bit_bounds
        )
        aux_bytes = (out_dir / CODEC_HEADER_FILENAME).stat().st_size
        if (out_dir / CONTENT_AWARE_Q_META_FILENAME).exists():
            aux_bytes += (out_dir / CONTENT_AWARE_Q_META_FILENAME).stat().st_size
        total_bits += aux_bytes * 8
        meta = {
            "codec": "hac_pp",
            "num_anchors": int(N),
            "num_anchors_total": int(n_total),
            "q_scale_feat": float(q_scale_feat),
            "q_scale_scaling": float(q_scale_scaling),
            "q_scale_offsets": float(q_scale_offsets),
            "mask_keep_ratio": mask_keep_ratio,
            "bit_anchor": int(bits_xyz),
            "bit_feat": int(sum(bit_feat_list)),
            "bit_scaling": int(sum(bit_scaling_list)),
            "bit_offsets": int(sum(bit_offsets_list)),
            "bit_hash": int(bit_hash),
            "bit_masks": int(bit_masks),
            "bit_mlp": int(bit_mlp),
            "bit_bounds": int(bit_bounds),
            "bit_header": int(aux_bytes * 8),
            "total_bits": int(total_bits),
            "total_MB": round(total_bits / bit2MB_scale, 4),
        }
        with open(out_dir / "hac_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    @torch.no_grad()
    def decode_attributes(
        self,
        artifact_dir: Path,
        q_scale_feat: float = 1.0,
        q_scale_scaling: float = 1.0,
        q_scale_offsets: float = 1.0,
        q_override_feat=None,
        q_override_scaling=None,
        q_override_offsets=None,
    ) -> None:
        """Restore quantized attributes from the bitstream (official decode).

        The ``q_scale_*`` values must match the ones used at encode time.
        """
        from hacplus.scene.gaussian_model import bit2MB_scale
        from hacplus.utils.codec_consistency import (
            CODEC_HEADER_FILENAME,
            CONTENT_AWARE_Q_META_FILENAME,
            FORMULA_INPUT_VERSION,
        )

        core = self.core
        artifact_dir = Path(artifact_dir)
        k = self.cfg.n_offsets
        cg = int(core.feat_channel_group)
        device = self.device
        header_path = artifact_dir / CODEC_HEADER_FILENAME
        if not header_path.is_file():
            raise FileNotFoundError(f"PHG v1 decode requires {CODEC_HEADER_FILENAME}")
        with open(header_path) as f:
            codec_header = json.load(f)
        if codec_header.get("format") != "phg_v1":
            raise RuntimeError(
                f"unsupported codec header format: {codec_header.get('format')!r}"
            )
        if codec_header.get("formula_input_version") != FORMULA_INPUT_VERSION:
            raise RuntimeError(
                "formula input version mismatch: "
                f"{codec_header.get('formula_input_version')!r}"
            )
        core.current_step = int(codec_header.get("codec_iteration", 0))
        core.current_iter = core.current_step
        if codec_header.get("model", {}).get("content_aware_quant"):
            q_meta_path = artifact_dir / CONTENT_AWARE_Q_META_FILENAME
            if not q_meta_path.is_file():
                raise FileNotFoundError(
                    f"content-aware quantization requires {CONTENT_AWARE_Q_META_FILENAME}"
                )
            with open(q_meta_path) as f:
                q_meta = json.load(f)
            if q_meta.get("mode") != "formula" or q_meta.get("version") != FORMULA_INPUT_VERSION:
                raise RuntimeError("content-aware Q meta is not the PHG v1 formula contract")
            if abs(float(q_meta.get("complexity_scale", -1.0)) - float(core.complexity_scale)) > 1e-6:
                raise RuntimeError(
                    "complexity_scale mismatch between bitstream and model: "
                    f"{q_meta.get('complexity_scale')} vs {core.complexity_scale}"
                )
        self._view.x_bound_min = torch.load(
            artifact_dir / "x_bound_min.pkl", map_location=device, weights_only=False
        )
        self._view.x_bound_max = torch.load(
            artifact_dir / "x_bound_max.pkl", map_location=device, weights_only=False
        )

        npz = np.load(artifact_dir / "xyz_gpcc.npz")
        voxel_size = float(npz["voxel_size"])
        means_strings = npz["means_strings"].tobytes()
        anchor_int = decompress_gpcc(means_strings).to(device)
        sorted_indices = calculate_morton_order(anchor_int)
        anchor_int = anchor_int[sorted_indices]
        anchor = anchor_int * voxel_size
        N = anchor.shape[0]
        ov_feat = _load_override(q_override_feat, N, device)
        ov_scaling = _load_override(q_override_scaling, N, device)
        ov_offsets = _load_override(q_override_offsets, N, device)
        if _tensor_sha256(anchor_int) != codec_header.get("anchor_int_sha256"):
            raise RuntimeError("anchor_int hash mismatch after GPCC round-trip")

        masks_decoded = decoder(
            N * k, str(artifact_dir / "masks.b"), device=str(device)
        ).float().view(-1, k, 1)
        if _tensor_sha256(masks_decoded) != codec_header.get("masks_sha256"):
            raise RuntimeError("masks hash mismatch after arithmetic decode")
        n_hash = int(core.get_encoding_params().numel())
        hash_decoded = decoder(
            n_hash, str(artifact_dir / "hash.b"), device=str(device)
        ).float()
        hash_decoded = (hash_decoded * 2 - 1).view(-1, core.n_features_per_level)

        steps = math.ceil(N / MAX_batch_size)
        feat_list, scaling_list, offsets_list = [], [], []
        for s in range(steps):
            start = s * MAX_batch_size
            end = min((s + 1) * MAX_batch_size, N)
            feat_b = str(artifact_dir / f"feat_{s}.b")
            scaling_b = str(artifact_dir / f"scaling_{s}.b")
            offsets_b = str(artifact_dir / f"offsets_{s}.b")
            anchor_slice = anchor[start:end]
            ctx = core.calc_context_feat(anchor_slice, caller="decode_attributes")
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, qa, qs, qo = torch.split(
                core.get_grid_mlp(ctx),
                [
                    self.cfg.feat_dim,
                    self.cfg.feat_dim,
                    self.cfg.feat_dim,
                    6,
                    6,
                    3 * k,
                    3 * k,
                    1,
                    1,
                    1,
                ],
                dim=-1,
            )
            qa = qa.repeat(1, self.cfg.feat_dim)
            qs = qs.repeat(1, 6)
            qo = qo.repeat(1, 3 * k)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = scale_scaling.contiguous().view(-1).clamp(min=1e-9)
            scale_offsets = scale_offsets.contiguous().view(-1).clamp(min=1e-9)
            Q_feat = q_scale_feat * 1.0 * (1 + torch.tanh(qa))
            Q_scaling = q_scale_scaling * 0.001 * (1 + torch.tanh(qs))
            Q_offsets = q_scale_offsets * 0.2 * (1 + torch.tanh(qo))
            if core.is_content_aware_quant_active():
                masks_slice = masks_decoded[start:end]
                (
                    Q_feat,
                    Q_scaling,
                    Q_offsets,
                    _,
                    _,
                    _,
                    _,
                ) = core._codec_apply_content_aware_quant_params(
                    "decode_attributes",
                    anchor_slice,
                    masks_slice,
                    Q_feat,
                    Q_scaling,
                    Q_offsets,
                    None,
                    None,
                    None,
                    mean_scaling.view(-1, 6),
                    mean_offsets.view(-1, 3 * k),
                )
            if ov_feat is not None:
                Q_feat = Q_feat * ov_feat[start:end]
            if ov_scaling is not None:
                Q_scaling = Q_scaling * ov_scaling[start:end]
            if ov_offsets is not None:
                Q_offsets = Q_offsets * ov_offsets[start:end]
            Q_feat_flat = Q_feat.contiguous().view(-1)
            Q_scaling_flat = Q_scaling.contiguous().view(-1)
            Q_offsets_flat = Q_offsets.contiguous().view(-1)

            n_num = end - start
            feat_decoded = torch.zeros(n_num, self.cfg.feat_dim, device=device)
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = scale.clamp(min=1e-9)
            for cc in range(self.cfg.feat_dim // cg):
                mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
                    feat_decoded, mean_scale, to_dec=cc
                )
                probs = torch.softmax(
                    torch.stack([prob[:, cc * cg : cc * cg + cg], prob_adj], dim=-1),
                    dim=-1,
                )
                Q_tmp = Q_feat[:, cc * cg : cc * cg + cg].contiguous().view(-1)
                dec = decoder_gaussian_mixed_chunk(
                    [
                        mean[:, cc * cg : cc * cg + cg].contiguous().view(-1),
                        mean_adj.contiguous().view(-1),
                    ],
                    [
                        scale[:, cc * cg : cc * cg + cg].contiguous().view(-1),
                        scale_adj.contiguous().view(-1),
                    ],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_tmp,
                    file_name=feat_b.replace(".b", f"_{cc}.b"),
                    chunk_size=500_000,
                )
                feat_decoded[:, cc * cg : cc * cg + cg] = dec.view(n_num, cg)

            scaling_decoded = decoder_gaussian_chunk(
                mean_scaling,
                scale_scaling,
                Q_scaling_flat,
                file_name=scaling_b,
                chunk_size=100_000,
            ).view(n_num, 6)
            masks_tmp = (
                masks_decoded[start:end]
                .repeat(1, 1, 3)
                .view(-1, 3 * k)
                .view(-1)
                .bool()
            )
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = decoder_gaussian_chunk(
                mean_offsets[masks_tmp],
                scale_offsets[masks_tmp],
                Q_offsets_flat[masks_tmp],
                file_name=offsets_b,
                chunk_size=100_000,
            )
            offsets_decoded = offsets_decoded.view(n_num, k, 3)
            feat_list.append(feat_decoded)
            scaling_list.append(scaling_decoded)
            offsets_list.append(offsets_decoded)

        feat = torch.cat(feat_list, dim=0)
        scaling = torch.cat(scaling_list, dim=0)
        offsets = torch.cat(offsets_list, dim=0)
        mask = torch.zeros(N, k + 1, 1, device=device)
        mask[:, :k] = masks_decoded

        self._view.decoded_version = True
        self._view.anchor = nn.Parameter(anchor, requires_grad=False)
        self._view.anchor_feat = nn.Parameter(feat, requires_grad=False)
        self._view.offset = nn.Parameter(offsets, requires_grad=False)
        self._view.scaling = nn.Parameter(scaling, requires_grad=False)
        self._view.mask = nn.Parameter(mask, requires_grad=False)
        # The official decode leaves _rotation/_opacity at the pre-mask size;
        # resize them so prefilter/render stay consistent (they are not used
        # for decoded neural Gaussians beyond the identity prefilter rotation).
        rot = torch.zeros(N, 4, device=device)
        rot[:, 0] = 1.0
        self._view.rotation = nn.Parameter(rot, requires_grad=False)
        self._view.opacity = nn.Parameter(
            inverse_sigmoid(torch.full((N, 1), 0.1, device=device)),
            requires_grad=False,
        )

        self._view.set_hash_params(hash_decoded)
        diag = {
            "bit_exact_roundtrip": True,
            "anchor_int_sha256": _tensor_sha256(anchor_int),
            "masks_sha256": _tensor_sha256(masks_decoded),
        }
        last_input = getattr(core, "_last_formula_complexity_input", None)
        if last_input is not None:
            diag["formula_input_sha256"] = _tensor_sha256(last_input)
        with open(artifact_dir / "codec_roundtrip_diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2, sort_keys=True)
        print(f"[HAC++] Decoded {N} anchors from {artifact_dir}")


class HACPlusCodec(CompressionCodec):
    """Entropy-codes a trained HAC++ model into bitstreams and back."""

    name = "hac_pp"

    def encode(
        self,
        model: BaseGaussianModel,
        output_dir: Path,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not isinstance(model, HACPlusModel):
            raise TypeError("hac_pp codec requires a HACPlusModel.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.export_attributes(), output_dir / "attributes.pth")
        meta = model.encode_attributes(output_dir, **kwargs)
        meta["raw_attribute_MB"] = round(
            sum(
                int(t.nelement() * t.element_size())
                for t in model.export_attributes().values()
                if torch.is_tensor(t)
            )
            / 1e6,
            3,
        )
        return meta

    def decode(
        self,
        artifact_dir: Path,
        **kwargs: Any,
    ) -> BaseGaussianModel:
        artifact_dir = Path(artifact_dir)
        attrs = torch.load(
            artifact_dir / "attributes.pth", map_location="cuda", weights_only=False
        )
        model = HACPlusModel.from_attributes(attrs, "cuda")
        model.decode_attributes(artifact_dir, **kwargs)
        return model
