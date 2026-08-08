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
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hacplus  # noqa: F401  (registers sys.path)
    from scene.gaussian_model import MAX_batch_size, GaussianModel as _OfficialHACModel
    from utils.encodings import STE_multistep, get_binary_vxl_size
    from utils.encodings_cuda import (
        decoder,
        decoder_gaussian_chunk,
        decoder_gaussian_mixed_chunk,
        encoder,
        encoder_gaussian_chunk,
        encoder_gaussian_mixed_chunk,
    )
    from utils.gpcc_utils import (
        calculate_morton_order,
        compress_gpcc,
        decompress_gpcc,
    )
except Exception as exc:  # pragma: no cover - missing HAC++ extensions
    _OfficialHACModel = None
    _HACPP_IMPORT_ERROR = exc

from .codec import CompressionCodec
from .config import ModelConfig, OptimConfig
from .model import BaseGaussianModel, NeuralGaussians
from .utils import inverse_sigmoid, knn_distances, median_nn_distance, voxelize_points


class _ParamsView:
    """Duck-typed accessor so the shared gsplat prefilter/renderer can consume
    the HAC++ core without a Scaffold-style ``AnchorParams`` module."""

    def __init__(self, model: "HACPlusModel") -> None:
        self._m = model

    @property
    def anchor(self) -> torch.Tensor:
        return self._m.core.get_anchor

    @property
    def scaling(self) -> torch.Tensor:
        return self._m.core.get_scaling

    @property
    def rotation(self) -> torch.Tensor:
        return self._m.core.get_rotation

    @property
    def num_anchors(self) -> int:
        return self._m.core.get_anchor.shape[0]


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
        )
        self.core.to(self.device)
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
        sd["_x_bound_min"] = self.core.x_bound_min
        sd["_x_bound_max"] = self.core.x_bound_max
        sd["_decoded_version"] = torch.tensor(int(self.core.decoded_version))
        return sd

    def load_state_dict(self, *args, **kwargs):
        sd = dict(args[0])
        if "_x_bound_min" in sd:
            self.core.x_bound_min = sd.pop("_x_bound_min").to(self.device)
            self.core.x_bound_max = sd.pop("_x_bound_max").to(self.device)
            self.core.decoded_version = bool(sd.pop("_decoded_version").item())
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
        return self.core.load_state_dict(sd, strict=kwargs.get("strict", True))

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

        core._anchor = nn.Parameter(anchor, requires_grad=True)
        core._offset = nn.Parameter(
            torch.zeros(n, k, 3, device=device), requires_grad=True
        )
        core._mask = nn.Parameter(
            torch.ones(n, k + 1, 1, device=device), requires_grad=True
        )
        core._anchor_feat = nn.Parameter(
            torch.zeros(n, self.cfg.feat_dim, device=device), requires_grad=True
        )
        core._scaling = nn.Parameter(
            torch.from_numpy(scales_log).float().to(device), requires_grad=True
        )
        core._rotation = nn.Parameter(rot, requires_grad=False)
        core._opacity = nn.Parameter(
            inverse_sigmoid(torch.full((n, 1), 0.1, device=device)),
            requires_grad=False,
        )
        core.spatial_lr_scale = self.spatial_lr_scale
        core.opacity_accum = torch.zeros(n, 1, device=device)
        core.offset_gradient_accum = torch.zeros(n * k, 1, device=device)
        core.offset_denom = torch.zeros(n * k, 1, device=device)
        core.anchor_demon = torch.zeros(n, 1, device=device)
        core.max_radii2D = torch.zeros(n, device=device)
        core.update_anchor_bound()
        # Guard against anchor grids whose min/max sit exactly on 0, which the
        # official calc_interp_feat assertion rejects.
        if (core.x_bound_min == 0).any():
            core.x_bound_min = core.x_bound_min - 1e-4
        if (core.x_bound_max == 0).any():
            core.x_bound_max = core.x_bound_max + 1e-4

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
    ) -> NeuralGaussians:
        del appearance_id
        core = self.core
        device = self.device
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
        feat = core._anchor_feat[anchor_indices]
        grid_offsets = core._offset[anchor_indices]
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
                feat_context_orig = core.calc_interp_feat(anchor)
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
        elif not core.decoded_version:
            feat_context = core.calc_interp_feat(anchor)
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
            feat = STE_multistep.apply(feat, Q_feat, core._anchor_feat.mean()).detach()
            grid_scaling = STE_multistep.apply(
                grid_scaling, Q_scaling, core.get_scaling.mean()
            ).detach()
            grid_offsets = STE_multistep.apply(
                grid_offsets, Q_offsets, core._offset.mean()
            ).detach()

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
    ) -> None:
        """Same statistics as official ``training_statis``, but reads the
        screen-space gradients from gsplat's retained ``means2d.grad``."""
        core = self.core
        k = self.cfg.n_offsets
        vis2d = visibility_filter  # [nnz] bool (packed rows with radii > 0)
        full_visible = torch.zeros(
            self.num_anchors, dtype=torch.bool, device=self.device
        )
        full_visible[gaussians.anchor_indices] = True

        temp_opacity = gaussians.neural_opacity.clone().view(-1).clamp(min=0.0).view(
            -1, k
        )
        core.opacity_accum[full_visible] += temp_opacity.sum(dim=1, keepdim=True)
        core.anchor_demon[full_visible] += 1.0

        active_local = torch.nonzero(gaussians.selection_mask).squeeze(-1)  # [M]
        local_anchor = active_local // k
        local_offset = active_local % k
        global_idx = gaussians.anchor_indices[local_anchor] * k + local_offset
        idx = global_idx[gaussian_ids][vis2d]
        if idx.numel() > 0:
            assert means2d.grad is not None, "means2d grad missing; retain_grad was not set"
            grad_norm = (
                means2d.grad[vis2d, :2].norm(dim=-1, keepdim=True)
                * (width / 2.0)
            )
            core.offset_gradient_accum[idx] += grad_norm
            core.offset_denom[idx] += 1.0

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

    # ------------------------------------------------------------------
    # Export / codec support
    # ------------------------------------------------------------------

    def export_attributes(self) -> Dict[str, Any]:
        core = self.core
        decoder_state = {
            "mlp_opacity": core.mlp_opacity.state_dict(),
            "mlp_cov": core.mlp_cov.state_dict(),
            "mlp_color": core.mlp_color.state_dict(),
            "mlp_grid": core.mlp_grid.state_dict(),
            "mlp_deform": core.mlp_deform.state_dict(),
            "encoding_xyz": core.encoding_xyz.state_dict(),
        }
        if core.use_feat_bank:
            decoder_state["mlp_feature_bank"] = core.mlp_feature_bank.state_dict()
        return {
            "model_name": self.model_name,
            "anchor": core._anchor.detach().cpu().clone(),
            "offset": core._offset.detach().cpu().clone(),
            "mask": core._mask.detach().cpu().clone(),
            "anchor_feat": core._anchor_feat.detach().cpu().clone(),
            "scaling": core._scaling.detach().cpu().clone(),
            "rotation": core._rotation.detach().cpu().clone(),
            "opacity": core._opacity.detach().cpu().clone(),
            "decoder": decoder_state,
            "config": self.cfg.__dict__.copy(),
            "voxel_size": float(self.voxel_size),
            "spatial_lr_scale": float(self.spatial_lr_scale),
            "x_bound_min": core.x_bound_min.detach().cpu().clone(),
            "x_bound_max": core.x_bound_max.detach().cpu().clone(),
            "decoded_version": bool(core.decoded_version),
        }

    @classmethod
    def from_attributes(cls, attrs: Dict[str, Any], device: str) -> "HACPlusModel":
        cfg = ModelConfig(**attrs["config"])
        model = cls(cfg, device)
        model.voxel_size = attrs["voxel_size"]
        model.spatial_lr_scale = attrs["spatial_lr_scale"]
        core = model.core
        core._anchor = nn.Parameter(attrs["anchor"].to(device), requires_grad=True)
        core._offset = nn.Parameter(attrs["offset"].to(device), requires_grad=True)
        core._mask = nn.Parameter(attrs["mask"].to(device), requires_grad=True)
        core._anchor_feat = nn.Parameter(
            attrs["anchor_feat"].to(device), requires_grad=True
        )
        core._scaling = nn.Parameter(attrs["scaling"].to(device), requires_grad=True)
        core._rotation = nn.Parameter(attrs["rotation"].to(device), requires_grad=False)
        core._opacity = nn.Parameter(attrs["opacity"].to(device), requires_grad=False)
        core.x_bound_min = attrs["x_bound_min"].to(device)
        core.x_bound_max = attrs["x_bound_max"].to(device)
        core.decoded_version = bool(attrs.get("decoded_version", False))
        ds = attrs["decoder"]
        core.mlp_opacity.load_state_dict(ds["mlp_opacity"])
        core.mlp_cov.load_state_dict(ds["mlp_cov"])
        core.mlp_color.load_state_dict(ds["mlp_color"])
        core.mlp_grid.load_state_dict(ds["mlp_grid"])
        core.mlp_deform.load_state_dict(ds["mlp_deform"])
        core.encoding_xyz.load_state_dict(ds["encoding_xyz"])
        if core.use_feat_bank:
            core.mlp_feature_bank.load_state_dict(ds["mlp_feature_bank"])
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
        feat_c = core._anchor_feat[choose]
        offsets_c = core._offset[choose]
        scaling_c = core.get_scaling[choose]
        masks_c = core.get_mask[choose]
        mask_anchor_c = core.get_mask_anchor[choose]

        ctx = core.calc_interp_feat(anchor_c)
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
            x_mean=core._anchor_feat.mean(),
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
            core._offset.mean(),
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
    def encode_attributes(self, out_dir: Path) -> Dict[str, Any]:
        """Entropy-code anchor attributes with the official arithmetic codec."""
        from scene.gaussian_model import bit2MB_scale

        core = self.core
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        k = self.cfg.n_offsets
        device = self.device

        mask_anchor = core.get_mask_anchor.to(torch.bool)[:, 0]
        anchor = core.get_anchor[mask_anchor]
        feat = core._anchor_feat[mask_anchor]
        offsets = core._offset[mask_anchor]
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
        means_strings = compress_gpcc(anchor_int)
        np.savez_compressed(
            out_dir / "xyz_gpcc.npz",
            means_strings=means_strings,
            voxel_size=np.float32(self.voxel_size),
        )
        bits_xyz = (out_dir / "xyz_gpcc.npz").stat().st_size * 8
        torch.save(core.x_bound_min, out_dir / "x_bound_min.pkl")
        torch.save(core.x_bound_max, out_dir / "x_bound_max.pkl")
        anchor = anchor_int.float() * self.voxel_size

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
            ctx = core.calc_interp_feat(anchor_slice)
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
            qs = qs.repeat(1, 6).view(-1)
            qo = qo.repeat(1, 3 * k).view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = scale_scaling.contiguous().view(-1).clamp(min=1e-9)
            scale_offsets = scale_offsets.contiguous().view(-1).clamp(min=1e-9)
            Q_feat = 1.0 * (1 + torch.tanh(qa))
            Q_scaling = 0.001 * (1 + torch.tanh(qs))
            Q_offsets = 0.2 * (1 + torch.tanh(qo))

            # features (channel-context, 10 channels per step)
            feat_slice = feat[start:end]
            feat_q = STE_multistep.apply(feat_slice, Q_feat, core._anchor_feat.mean())
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = scale.clamp(min=1e-9)
            bit_feat = 0
            for cc in range(self.cfg.feat_dim // 10):
                mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
                    feat_q, mean_scale, to_dec=cc
                )
                probs = torch.softmax(
                    torch.stack([prob[:, cc * 10 : cc * 10 + 10], prob_adj], dim=-1),
                    dim=-1,
                )
                feat_tmp = feat_q[:, cc * 10 : cc * 10 + 10].contiguous().view(-1)
                Q_tmp = Q_feat[:, cc * 10 : cc * 10 + 10].contiguous().view(-1)
                bit_feat += encoder_gaussian_mixed_chunk(
                    feat_tmp,
                    [
                        mean[:, cc * 10 : cc * 10 + 10].contiguous().view(-1),
                        mean_adj.contiguous().view(-1),
                    ],
                    [
                        scale[:, cc * 10 : cc * 10 + 10].contiguous().view(-1),
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
                scaling_slice, Q_scaling, core.get_scaling.mean()
            )
            bit_scaling_list.append(
                encoder_gaussian_chunk(
                    scaling_q,
                    mean_scaling,
                    scale_scaling,
                    Q_scaling,
                    file_name=scaling_b,
                    chunk_size=100_000,
                )
            )

            # offsets (masked)
            mask_slice = masks[start:end].repeat(1, 1, 3).view(-1, 3 * k).view(-1)
            offsets_slice = offsets[start:end].view(-1, 3 * k).view(-1)
            offsets_q = STE_multistep.apply(
                offsets_slice, Q_offsets, core._offset.mean()
            )
            offsets_q[~mask_slice.bool()] = 0.0
            bit_offsets_list.append(
                encoder_gaussian_chunk(
                    offsets_q[mask_slice.bool()],
                    mean_offsets[mask_slice.bool()],
                    scale_offsets[mask_slice.bool()],
                    Q_offsets[mask_slice.bool()],
                    file_name=offsets_b,
                    chunk_size=100_000,
                )
            )

        hash_params = core.get_encoding_params()
        bit_hash = encoder(
            ((hash_params.view(-1) + 1) / 2), file_name=str(out_dir / "hash.b")
        )
        bit_masks = encoder(masks, file_name=str(out_dir / "masks.b"))

        total_bits = (
            bits_xyz
            + sum(bit_feat_list)
            + sum(bit_scaling_list)
            + sum(bit_offsets_list)
            + bit_hash
            + bit_masks
        )
        meta = {
            "codec": "hac_pp",
            "num_anchors": int(N),
            "bit_anchor": int(bits_xyz),
            "bit_feat": int(sum(bit_feat_list)),
            "bit_scaling": int(sum(bit_scaling_list)),
            "bit_offsets": int(sum(bit_offsets_list)),
            "bit_hash": int(bit_hash),
            "bit_masks": int(bit_masks),
            "total_bits": int(total_bits),
            "total_MB": round(total_bits / bit2MB_scale, 4),
        }
        with open(out_dir / "hac_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    @torch.no_grad()
    def decode_attributes(self, artifact_dir: Path) -> None:
        """Restore quantized attributes from the bitstream (official decode)."""
        from scene.gaussian_model import bit2MB_scale

        core = self.core
        artifact_dir = Path(artifact_dir)
        k = self.cfg.n_offsets
        device = self.device
        core.x_bound_min = torch.load(
            artifact_dir / "x_bound_min.pkl", map_location=device, weights_only=False
        )
        core.x_bound_max = torch.load(
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

        masks_decoded = decoder(
            N * k, str(artifact_dir / "masks.b"), device=str(device)
        ).float().view(-1, k, 1)
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
            ctx = core.calc_interp_feat(anchor_slice)
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
            qs = qs.repeat(1, 6).view(-1)
            qo = qo.repeat(1, 3 * k).view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = scale_scaling.contiguous().view(-1).clamp(min=1e-9)
            scale_offsets = scale_offsets.contiguous().view(-1).clamp(min=1e-9)
            Q_feat = 1.0 * (1 + torch.tanh(qa))
            Q_scaling = 0.001 * (1 + torch.tanh(qs))
            Q_offsets = 0.2 * (1 + torch.tanh(qo))

            n_num = end - start
            feat_decoded = torch.zeros(n_num, self.cfg.feat_dim, device=device)
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = scale.clamp(min=1e-9)
            for cc in range(self.cfg.feat_dim // 10):
                mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
                    feat_decoded, mean_scale, to_dec=cc
                )
                probs = torch.softmax(
                    torch.stack([prob[:, cc * 10 : cc * 10 + 10], prob_adj], dim=-1),
                    dim=-1,
                )
                Q_tmp = Q_feat[:, cc * 10 : cc * 10 + 10].contiguous().view(-1)
                dec = decoder_gaussian_mixed_chunk(
                    [
                        mean[:, cc * 10 : cc * 10 + 10].contiguous().view(-1),
                        mean_adj.contiguous().view(-1),
                    ],
                    [
                        scale[:, cc * 10 : cc * 10 + 10].contiguous().view(-1),
                        scale_adj.contiguous().view(-1),
                    ],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_tmp,
                    file_name=feat_b.replace(".b", f"_{cc}.b"),
                    chunk_size=500_000,
                )
                feat_decoded[:, cc * 10 : cc * 10 + 10] = dec.view(n_num, 10)

            scaling_decoded = decoder_gaussian_chunk(
                mean_scaling,
                scale_scaling,
                Q_scaling,
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
                Q_offsets[masks_tmp],
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

        core.decoded_version = True
        core._anchor = nn.Parameter(anchor, requires_grad=False)
        core._anchor_feat = nn.Parameter(feat, requires_grad=False)
        core._offset = nn.Parameter(offsets, requires_grad=False)
        core._scaling = nn.Parameter(scaling, requires_grad=False)
        core._mask = nn.Parameter(mask, requires_grad=False)
        # The official decode leaves _rotation/_opacity at the pre-mask size;
        # resize them so prefilter/render stay consistent (they are not used
        # for decoded neural Gaussians beyond the identity prefilter rotation).
        rot = torch.zeros(N, 4, device=device)
        rot[:, 0] = 1.0
        core._rotation = nn.Parameter(rot, requires_grad=False)
        core._opacity = nn.Parameter(
            inverse_sigmoid(torch.full((N, 1), 0.1, device=device)),
            requires_grad=False,
        )

        if core.use_2D:
            len_3d = core.encoding_xyz.encoding_xyz.params.shape[0]
            len_2d = core.encoding_xyz.encoding_xy.params.shape[0]
            core.encoding_xyz.encoding_xyz.params = nn.Parameter(
                hash_decoded[0:len_3d], requires_grad=False
            )
            core.encoding_xyz.encoding_xy.params = nn.Parameter(
                hash_decoded[len_3d : len_3d + len_2d], requires_grad=False
            )
            core.encoding_xyz.encoding_xz.params = nn.Parameter(
                hash_decoded[len_3d + len_2d : len_3d + 2 * len_2d], requires_grad=False
            )
            core.encoding_xyz.encoding_yz.params = nn.Parameter(
                hash_decoded[len_3d + 2 * len_2d : len_3d + 3 * len_2d],
                requires_grad=False,
            )
        else:
            core.encoding_xyz.params = nn.Parameter(hash_decoded, requires_grad=False)
        print(f"[HAC++] Decoded {N} anchors from {artifact_dir}")


class HACPlusCodec(CompressionCodec):
    """Entropy-codes a trained HAC++ model into bitstreams and back."""

    name = "hac_pp"

    def encode(self, model: BaseGaussianModel, output_dir: Path) -> Dict[str, Any]:
        if not isinstance(model, HACPlusModel):
            raise TypeError("hac_pp codec requires a HACPlusModel.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.export_attributes(), output_dir / "attributes.pth")
        meta = model.encode_attributes(output_dir)
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

    def decode(self, artifact_dir: Path) -> BaseGaussianModel:
        artifact_dir = Path(artifact_dir)
        attrs = torch.load(
            artifact_dir / "attributes.pth", map_location="cuda", weights_only=False
        )
        model = HACPlusModel.from_attributes(attrs, "cuda")
        model.decode_attributes(artifact_dir)
        return model
