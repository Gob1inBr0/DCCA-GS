"""Scaffold-GS model: anchors, neural-Gaussian decoders and gsplat integration.

The model is split into three layers so that future anchor-based methods
(HAC / HAC++) can reuse the training loop and rasterizer:

1. :class:`AnchorParams` -- the explicit per-anchor tensors.
2. :class:`AnchorDecoder` -- the MLPs that turn anchor attributes into neural
   Gaussians (this is the natural swap point for a hash-grid context model).
3. :class:`ScaffoldGSModel` -- the full model registered in :data:`MODELS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, OptimConfig
from .utils import (
    get_expon_lr_func,
    inverse_sigmoid,
    knn_distances,
    median_nn_distance,
    voxelize_points,
)


@dataclass
class NeuralGaussians:
    """Decoded, opacity-masked neural Gaussians ready for rasterization."""

    xyz: torch.Tensor  # [M, 3]
    colors: torch.Tensor  # [M, 3]
    opacities: torch.Tensor  # [M]
    scales: torch.Tensor  # [M, 3]
    quats: torch.Tensor  # [M, 4]
    neural_opacity: torch.Tensor  # [n, K] raw decoder output (pre-mask)
    selection_mask: torch.Tensor  # [n * K] bool, True where opacity > 0
    visible_mask: torch.Tensor  # [N] bool anchors used for this camera
    anchor_indices: torch.Tensor  # [n] global indices of visible anchors
    # Optional HAC++ rate statistics (None for Scaffold-GS).
    bit_per_param: Optional[torch.Tensor] = None
    bit_per_feat_param: Optional[torch.Tensor] = None
    bit_per_scaling_param: Optional[torch.Tensor] = None
    bit_per_offsets_param: Optional[torch.Tensor] = None


class AnchorParams(nn.Module):
    """Explicit anchor tensors, kept in one module for easy save/load."""

    def __init__(self, n_offsets: int, feat_dim: int) -> None:
        super().__init__()
        self.n_offsets = n_offsets
        self.feat_dim = feat_dim
        self.anchor = nn.Parameter(torch.empty(0, 3))
        self.offset = nn.Parameter(torch.empty(0, n_offsets, 3))
        self.anchor_feat = nn.Parameter(torch.empty(0, feat_dim))
        self.scaling = nn.Parameter(torch.empty(0, 6))
        # Fixed identity rotation / opacity, kept only for checkpoint parity
        # with the official Scaffold-GS PLY format.
        self.rotation = nn.Parameter(torch.empty(0, 4), requires_grad=False)
        self.opacity = nn.Parameter(torch.empty(0, 1), requires_grad=False)

    @property
    def num_anchors(self) -> int:
        return self.anchor.shape[0]

    def _leaf(self, name: str, new_value: torch.Tensor) -> nn.Parameter:
        learnable = name not in ("rotation", "opacity")
        return nn.Parameter(new_value.detach(), requires_grad=learnable)

    def cat_(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Append new anchors to all per-anchor parameters."""
        for name in (
            "anchor",
            "offset",
            "anchor_feat",
            "scaling",
            "rotation",
            "opacity",
        ):
            old = getattr(self, name)
            new = torch.cat([old.detach(), tensors[name].to(old.device)], dim=0)
            setattr(self, name, self._leaf(name, new))

    def prune_(self, mask: torch.Tensor) -> None:
        """Keep only rows where ``mask`` is True."""
        for name in (
            "anchor",
            "offset",
            "anchor_feat",
            "scaling",
            "rotation",
            "opacity",
        ):
            p = getattr(self, name)
            setattr(self, name, self._leaf(name, p.detach()[mask]))


class AnchorDecoder(nn.Module):
    """MLPs that decode neural-Gaussian attributes from anchor features.

    This is the intended extension point for HAC / HAC++: a hash-grid context
    model can replace (or pre-process) :meth:`predict_gaussians` while the
    rest of the pipeline stays unchanged.
    """

    def __init__(
        self,
        feat_dim: int = 32,
        n_offsets: int = 10,
        appearance_dim: int = 32,
        use_feat_bank: bool = False,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.appearance_dim = appearance_dim
        self.use_feat_bank = use_feat_bank

        # Inputs: [anchor_feat (feat_dim), view_dir (3)].
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh(),
        )
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * n_offsets),
        )
        color_in = feat_dim + 3 + appearance_dim
        self.mlp_color = nn.Sequential(
            nn.Linear(color_in, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid(),
        )
        if use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(4, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1),
            )
        self.embedding_appearance: Optional[nn.Embedding] = None

    def set_appearance(self, num_cameras: int) -> None:
        if self.appearance_dim > 0:
            self.embedding_appearance = nn.Embedding(
                num_cameras, self.appearance_dim
            )

    def blend_feature_bank(
        self, feat: torch.Tensor, ob_view: torch.Tensor, ob_dist: torch.Tensor
    ) -> torch.Tensor:
        """Official multi-resolution feature bank (used only when enabled)."""
        weights = self.mlp_feature_bank(
            torch.cat([ob_view, ob_dist], dim=-1)
        )  # [n, 3]
        l1, l2 = self.feat_dim // 4, self.feat_dim // 2
        f0 = feat[:, :l1].repeat(1, 4)
        f1 = feat[:, :l2].repeat(1, 2)
        return weights[:, 0:1] * f0 + weights[:, 1:2] * f1 + weights[:, 2:3] * feat

    def predict_gaussians(
        self,
        params: AnchorParams,
        camera_center: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
        appearance_id: Optional[int] = None,
    ) -> NeuralGaussians:
        """Decode neural Gaussians from anchors visible to one camera."""
        device = params.anchor.device
        n_total = params.num_anchors
        if n_total == 0:
            empty = torch.zeros(0, device=device)
            return NeuralGaussians(
                xyz=empty.clone().reshape(0, 3),
                colors=empty.clone().reshape(0, 3),
                opacities=empty.clone(),
                scales=empty.clone().reshape(0, 3),
                quats=empty.clone().reshape(0, 4),
                neural_opacity=empty.clone().reshape(0, self.n_offsets),
                selection_mask=empty.clone().reshape(0).bool(),
                visible_mask=empty.clone().bool(),
                anchor_indices=empty.clone().long(),
            )

        if visible_mask is None:
            visible_mask = torch.ones(n_total, dtype=torch.bool, device=device)
        anchor_indices = torch.nonzero(visible_mask).squeeze(-1)
        n = anchor_indices.shape[0]
        if n == 0:
            empty = torch.zeros(0, device=device)
            return NeuralGaussians(
                xyz=empty.clone().reshape(0, 3),
                colors=empty.clone().reshape(0, 3),
                opacities=empty.clone(),
                scales=empty.clone().reshape(0, 3),
                quats=empty.clone().reshape(0, 4),
                neural_opacity=empty.clone().reshape(0, self.n_offsets),
                selection_mask=empty.clone().reshape(0).bool(),
                visible_mask=visible_mask,
                anchor_indices=anchor_indices,
            )

        feat = params.anchor_feat[anchor_indices]
        anchor = params.anchor[anchor_indices]
        offsets = params.offset[anchor_indices]
        anchor_scaling = torch.exp(params.scaling[anchor_indices])  # [n, 6]

        ob_view = anchor - camera_center
        ob_dist = ob_view.norm(dim=-1, keepdim=True)
        ob_view = ob_view / ob_dist.clamp_min(1e-8)

        if self.use_feat_bank:
            feat = self.blend_feature_bank(feat, ob_view, ob_dist)

        cat_local = torch.cat([feat, ob_view], dim=-1)  # [n, feat_dim + 3]
        neural_opacity = self.mlp_opacity(cat_local)  # [n, K]
        selection_mask = (neural_opacity > 0.0).reshape(-1)

        if self.appearance_dim > 0:
            assert self.embedding_appearance is not None
            idx = torch.full(
                (n,),
                int(appearance_id if appearance_id is not None else 0),
                dtype=torch.long,
                device=device,
            )
            appearance = self.embedding_appearance(idx)
            color_input = torch.cat([cat_local, appearance], dim=-1)
        else:
            color_input = cat_local

        color = self.mlp_color(color_input).reshape(n, self.n_offsets, 3)
        scale_rot = self.mlp_cov(cat_local).reshape(n, self.n_offsets, 7)

        scales = anchor_scaling[:, 3:].unsqueeze(1) * torch.sigmoid(
            scale_rot[..., :3]
        )
        quats = F.normalize(scale_rot[..., 3:7], dim=-1)
        xyz = anchor.unsqueeze(1) + offsets * anchor_scaling[:, :3].unsqueeze(1)

        k = self.n_offsets
        xyz = xyz.reshape(n * k, 3)
        color = color.reshape(n * k, 3)
        scales = scales.reshape(n * k, 3)
        quats = quats.reshape(n * k, 4)
        neural_opacity_flat = neural_opacity.reshape(-1)

        return NeuralGaussians(
            xyz=xyz[selection_mask],
            colors=color[selection_mask],
            opacities=neural_opacity_flat[selection_mask],
            scales=scales[selection_mask],
            quats=quats[selection_mask],
            neural_opacity=neural_opacity,
            selection_mask=selection_mask,
            visible_mask=visible_mask,
            anchor_indices=anchor_indices,
        )


class BaseGaussianModel(nn.Module, ABC):
    """Interface implemented by every model the trainer can run.

    HAC / HAC++ will register subclasses here under ``hac`` / ``hac_pp`` and
    re-use the trainer, renderer and growth code.
    """

    model_name: ClassVar[str] = "base"

    def __init__(self, cfg: ModelConfig, device: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_schedulers: Dict[str, Callable[[int], float]] = {}
        self.spatial_lr_scale = 1.0
        self.voxel_size = cfg.voxel_size

    @abstractmethod
    def init_from_pcd(
        self, points: np.ndarray, rgbs: np.ndarray, spatial_lr_scale: float
    ) -> None:
        """Initialize anchors from SfM points."""

    @abstractmethod
    def prefilter_anchors(self, camera) -> torch.Tensor:
        """Return a bool mask of anchors visible to ``camera``."""

    @abstractmethod
    def generate_gaussians(self, camera, visible_mask=None, **kwargs) -> NeuralGaussians:
        """Decode neural Gaussians for the visible anchors."""

    @abstractmethod
    def render(self, camera, background, **kwargs):
        """Render one camera; returns a ``RenderOutput``."""

    @abstractmethod
    def create_optimizer(self, optim_cfg: OptimConfig) -> None:
        """Create ``self.optimizer`` and LR schedulers."""

    @abstractmethod
    def update_learning_rate(self, iteration: int) -> None:
        """Update per-group learning rates for one iteration."""

    @abstractmethod
    def export_attributes(self) -> Dict[str, Any]:
        """Detached, named anchor attributes + decoder weights (for codecs)."""

    def training_statis(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def adjust_anchor(self, *args, **kwargs) -> None:
        raise NotImplementedError


class ScaffoldGSModel(BaseGaussianModel):
    """The faithful Scaffold-GS model."""

    model_name: ClassVar[str] = "scaffold_gs"

    def __init__(self, cfg: ModelConfig, device: str) -> None:
        super().__init__(cfg, device)
        self.anchor_params = AnchorParams(cfg.n_offsets, cfg.feat_dim).to(device)
        self.decoder = AnchorDecoder(
            feat_dim=cfg.feat_dim,
            n_offsets=cfg.n_offsets,
            appearance_dim=cfg.appearance_dim,
            use_feat_bank=cfg.use_feat_bank,
        ).to(device)

        # Running statistics for anchor growing / pruning.
        self.opacity_accum: Optional[torch.Tensor] = None
        self.offset_gradient_accum: Optional[torch.Tensor] = None
        self.offset_denom: Optional[torch.Tensor] = None
        self.anchor_demon: Optional[torch.Tensor] = None
        self.max_radii2D: Optional[torch.Tensor] = None

    @property
    def num_anchors(self) -> int:
        return self.anchor_params.num_anchors

    def set_appearance(self, num_cameras: int) -> None:
        self.decoder.set_appearance(num_cameras)

    def init_from_pcd(
        self, points: np.ndarray, rgbs: np.ndarray, spatial_lr_scale: float
    ) -> None:
        self.spatial_lr_scale = float(spatial_lr_scale)
        points = np.asarray(points, dtype=np.float32)
        if self.cfg.ratio > 1:
            points = points[:: self.cfg.ratio]

        if self.voxel_size <= 0:
            self.voxel_size = median_nn_distance(points)
            print(f"[Model] Auto voxel_size = {self.voxel_size:.6f}")
        points = voxelize_points(points, self.voxel_size)
        n = len(points)
        print(f"[Model] {n} anchors after voxelization.")

        device = self.device
        anchor = torch.from_numpy(points).float().to(device)
        dist = knn_distances(points, k=2)[:, 1]
        dist = np.clip(dist, 1e-7, None)
        scales_log = np.log(dist)[:, None].repeat(6, axis=1)

        self.anchor_params.anchor = nn.Parameter(anchor, requires_grad=True)
        self.anchor_params.offset = nn.Parameter(
            torch.zeros(n, self.cfg.n_offsets, 3, device=device),
            requires_grad=True,
        )
        self.anchor_params.anchor_feat = nn.Parameter(
            torch.zeros(n, self.cfg.feat_dim, device=device),
            requires_grad=True,
        )
        self.anchor_params.scaling = nn.Parameter(
            torch.from_numpy(scales_log).float().to(device),
            requires_grad=True,
        )
        rot = torch.zeros(n, 4, device=device)
        rot[:, 0] = 1.0
        self.anchor_params.rotation = nn.Parameter(rot, requires_grad=False)
        self.anchor_params.opacity = nn.Parameter(
            inverse_sigmoid(
                torch.full((n, 1), 0.1, device=device)
            ),
            requires_grad=False,
        )

        self.opacity_accum = torch.zeros(n, 1, device=device)
        self.offset_gradient_accum = torch.zeros(
            n * self.cfg.n_offsets, 1, device=device
        )
        self.offset_denom = torch.zeros(n * self.cfg.n_offsets, 1, device=device)
        self.anchor_demon = torch.zeros(n, 1, device=device)
        self.max_radii2D = torch.zeros(n, device=device)

    def generate_gaussians(
        self,
        camera,
        visible_mask: Optional[torch.Tensor] = None,
        is_training: bool = False,
        appearance_id: Optional[int] = None,
        step: int = 0,
    ) -> NeuralGaussians:
        del is_training  # Scaffold-GS decodes the same way in train/eval.
        del step
        center = camera.camera_center(self.device)
        return self.decoder.predict_gaussians(
            self.anchor_params,
            center,
            visible_mask=visible_mask,
            appearance_id=appearance_id,
        )

    def prefilter_anchors(self, camera) -> torch.Tensor:
        from .renderer import prefilter_anchors

        return prefilter_anchors(self, camera)

    def render(self, camera, background, **kwargs):
        from .renderer import render

        return render(self, camera, background, **kwargs)

    def create_optimizer(self, optim_cfg: OptimConfig) -> None:
        p = self.anchor_params
        groups: List[Dict[str, Any]] = []

        def add(name: str, params: List[nn.Parameter], lr: float) -> None:
            groups.append({"params": params, "lr": lr, "name": name})

        add("anchor", [p.anchor], optim_cfg.position_lr_init * self.spatial_lr_scale)
        add("offset", [p.offset], optim_cfg.offset_lr_init * self.spatial_lr_scale)
        add("anchor_feat", [p.anchor_feat], optim_cfg.feature_lr)
        add("scaling", [p.scaling], optim_cfg.scaling_lr)
        add(
            "mlp_opacity",
            list(self.decoder.mlp_opacity.parameters()),
            optim_cfg.mlp_opacity_lr_init,
        )
        add(
            "mlp_cov",
            list(self.decoder.mlp_cov.parameters()),
            optim_cfg.mlp_cov_lr_init,
        )
        add(
            "mlp_color",
            list(self.decoder.mlp_color.parameters()),
            optim_cfg.mlp_color_lr_init,
        )
        if self.decoder.use_feat_bank:
            add(
                "mlp_featurebank",
                list(self.decoder.mlp_feature_bank.parameters()),
                optim_cfg.mlp_featurebank_lr_init,
            )
        if self.decoder.embedding_appearance is not None:
            add(
                "embedding_appearance",
                list(self.decoder.embedding_appearance.parameters()),
                optim_cfg.appearance_lr_init,
            )

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

        self.lr_schedulers = {
            "anchor": get_expon_lr_func(
                optim_cfg.position_lr_init * self.spatial_lr_scale,
                optim_cfg.position_lr_final * self.spatial_lr_scale,
                lr_delay_mult=optim_cfg.position_lr_delay_mult,
                max_steps=optim_cfg.position_lr_max_steps,
            ),
            "offset": get_expon_lr_func(
                optim_cfg.offset_lr_init * self.spatial_lr_scale,
                optim_cfg.offset_lr_final * self.spatial_lr_scale,
                lr_delay_mult=optim_cfg.offset_lr_delay_mult,
                max_steps=optim_cfg.offset_lr_max_steps,
            ),
            "anchor_feat": lambda step: float(optim_cfg.feature_lr),
            "scaling": lambda step: float(optim_cfg.scaling_lr),
            "mlp_opacity": get_expon_lr_func(
                optim_cfg.mlp_opacity_lr_init,
                optim_cfg.mlp_opacity_lr_final,
                lr_delay_mult=optim_cfg.mlp_opacity_lr_delay_mult,
                max_steps=optim_cfg.mlp_opacity_lr_max_steps,
            ),
            "mlp_cov": get_expon_lr_func(
                optim_cfg.mlp_cov_lr_init,
                optim_cfg.mlp_cov_lr_final,
                lr_delay_mult=optim_cfg.mlp_cov_lr_delay_mult,
                max_steps=optim_cfg.mlp_cov_lr_max_steps,
            ),
            "mlp_color": get_expon_lr_func(
                optim_cfg.mlp_color_lr_init,
                optim_cfg.mlp_color_lr_final,
                lr_delay_mult=optim_cfg.mlp_color_lr_delay_mult,
                max_steps=optim_cfg.mlp_color_lr_max_steps,
            ),
        }
        if self.decoder.use_feat_bank:
            self.lr_schedulers["mlp_featurebank"] = get_expon_lr_func(
                optim_cfg.mlp_featurebank_lr_init,
                optim_cfg.mlp_featurebank_lr_final,
                lr_delay_mult=optim_cfg.mlp_featurebank_lr_delay_mult,
                max_steps=optim_cfg.mlp_featurebank_lr_max_steps,
            )
        if self.decoder.embedding_appearance is not None:
            self.lr_schedulers["embedding_appearance"] = get_expon_lr_func(
                optim_cfg.appearance_lr_init,
                optim_cfg.appearance_lr_final,
                lr_delay_mult=optim_cfg.appearance_lr_delay_mult,
                max_steps=optim_cfg.appearance_lr_max_steps,
            )

    def update_learning_rate(self, iteration: int) -> None:
        assert self.optimizer is not None
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_schedulers[group["name"]](iteration)

    def training_statis(
        self,
        means2d: torch.Tensor,
        visibility_filter: torch.Tensor,
        gaussians: NeuralGaussians,
    ) -> None:
        from .growth import training_statis

        training_statis(self, means2d, visibility_filter, gaussians)

    def adjust_anchor(
        self,
        check_interval: int,
        success_threshold: float,
        grad_threshold: float,
        min_opacity: float,
    ) -> None:
        from .growth import adjust_anchor

        adjust_anchor(
            self,
            check_interval=check_interval,
            success_threshold=success_threshold,
            grad_threshold=grad_threshold,
            min_opacity=min_opacity,
        )

    def export_attributes(self) -> Dict[str, Any]:
        p = self.anchor_params
        decoder_state = {}
        for key, module in [
            ("mlp_opacity", self.decoder.mlp_opacity),
            ("mlp_cov", self.decoder.mlp_cov),
            ("mlp_color", self.decoder.mlp_color),
        ]:
            decoder_state[key] = module.state_dict()
        if self.decoder.use_feat_bank:
            decoder_state["mlp_feature_bank"] = self.decoder.mlp_feature_bank.state_dict()
        if self.decoder.embedding_appearance is not None:
            decoder_state["embedding_appearance"] = (
                self.decoder.embedding_appearance.state_dict()
            )
        return {
            "anchor": p.anchor.detach().cpu().clone(),
            "offset": p.offset.detach().cpu().clone(),
            "anchor_feat": p.anchor_feat.detach().cpu().clone(),
            "scaling": p.scaling.detach().cpu().clone(),
            "rotation": p.rotation.detach().cpu().clone(),
            "opacity": p.opacity.detach().cpu().clone(),
            "decoder": decoder_state,
            "config": asdict(self.cfg),
            "voxel_size": float(self.voxel_size),
            "spatial_lr_scale": float(self.spatial_lr_scale),
            "num_cameras": (
                int(self.decoder.embedding_appearance.num_embeddings)
                if self.decoder.embedding_appearance is not None
                else 0
            ),
        }

    @classmethod
    def from_attributes(
        cls, attrs: Dict[str, Any], device: str
    ) -> "ScaffoldGSModel":
        cfg = ModelConfig(**attrs["config"])
        model = cls(cfg, device)
        model.voxel_size = attrs["voxel_size"]
        model.spatial_lr_scale = attrs["spatial_lr_scale"]
        model.set_appearance(attrs.get("num_cameras", 0) or 1)
        decoder_state = attrs["decoder"]
        model.decoder.mlp_opacity.load_state_dict(decoder_state["mlp_opacity"])
        model.decoder.mlp_cov.load_state_dict(decoder_state["mlp_cov"])
        model.decoder.mlp_color.load_state_dict(decoder_state["mlp_color"])
        if model.decoder.use_feat_bank:
            model.decoder.mlp_feature_bank.load_state_dict(
                decoder_state["mlp_feature_bank"]
            )
        if model.decoder.embedding_appearance is not None:
            model.decoder.embedding_appearance.load_state_dict(
                decoder_state["embedding_appearance"]
            )
        model.anchor_params.anchor = nn.Parameter(
            attrs["anchor"].to(device), requires_grad=True
        )
        model.anchor_params.offset = nn.Parameter(
            attrs["offset"].to(device), requires_grad=True
        )
        model.anchor_params.anchor_feat = nn.Parameter(
            attrs["anchor_feat"].to(device), requires_grad=True
        )
        model.anchor_params.scaling = nn.Parameter(
            attrs["scaling"].to(device), requires_grad=True
        )
        model.anchor_params.rotation = nn.Parameter(
            attrs["rotation"].to(device), requires_grad=False
        )
        model.anchor_params.opacity = nn.Parameter(
            attrs["opacity"].to(device), requires_grad=False
        )
        return model

    def save_ply(self, path: str | Path) -> None:
        from plyfile import PlyData, PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        p = self.anchor_params
        anchor = p.anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        offset = (
            p.offset.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        )
        anchor_feat = p.anchor_feat.detach().cpu().numpy()
        opacities = p.opacity.detach().cpu().numpy()
        scaling = p.scaling.detach().cpu().numpy()
        rotation = p.rotation.detach().cpu().numpy()

        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += [f"f_offset_{i}" for i in range(offset.shape[1])]
        attrs += [f"f_anchor_feat_{i}" for i in range(anchor_feat.shape[1])]
        attrs += ["opacity"]
        attrs += [f"scale_{i}" for i in range(scaling.shape[1])]
        attrs += [f"rot_{i}" for i in range(rotation.shape[1])]
        dtype = [(a, "f4") for a in attrs]
        data = np.concatenate(
            [anchor, normals, offset, anchor_feat, opacities, scaling, rotation],
            axis=1,
        )
        elements = np.empty(anchor.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))

    def load_ply(self, path: str | Path) -> None:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        el = ply.elements[0]
        anchor = np.stack(
            [np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1
        ).astype(np.float32)

        def collect(prefix: str, dims: int) -> np.ndarray:
            names = sorted(
                [p.name for p in el.properties if p.name.startswith(prefix)],
                key=lambda s: int(s.rsplit("_", 1)[-1]),
            )
            arr = np.stack([np.asarray(el[n]) for n in names], axis=1).astype(
                np.float32
            )
            return arr.reshape(-1, dims)

        offset_flat = collect("f_offset_", 3)
        offset = offset_flat.reshape(-1, 3, self.cfg.n_offsets).permute(
            0, 2, 1
        )  # [N, K, 3] (official PLY stores 3*K in j-major order)
        anchor_feat = np.stack(
            [
                np.asarray(el[f"f_anchor_feat_{i}"])
                for i in range(self.cfg.feat_dim)
            ],
            axis=1,
        ).astype(np.float32)
        opacity = np.asarray(el["opacity"], dtype=np.float32)[:, None]
        scaling = np.stack(
            [np.asarray(el[f"scale_{i}"]) for i in range(6)], axis=1
        ).astype(np.float32)
        rotation = np.stack(
            [np.asarray(el[f"rot_{i}"]) for i in range(4)], axis=1
        ).astype(np.float32)

        device = self.device
        n = anchor.shape[0]
        self.anchor_params.anchor = nn.Parameter(
            torch.from_numpy(anchor).to(device), requires_grad=True
        )
        self.anchor_params.offset = nn.Parameter(
            torch.from_numpy(offset).to(device), requires_grad=True
        )
        self.anchor_params.anchor_feat = nn.Parameter(
            torch.from_numpy(anchor_feat).to(device), requires_grad=True
        )
        self.anchor_params.scaling = nn.Parameter(
            torch.from_numpy(scaling).to(device), requires_grad=True
        )
        self.anchor_params.rotation = nn.Parameter(
            torch.from_numpy(rotation).to(device), requires_grad=False
        )
        self.anchor_params.opacity = nn.Parameter(
            torch.from_numpy(opacity).to(device), requires_grad=False
        )
        print(f"[Model] Loaded {n} anchors from PLY.")

    def save_mlp_checkpoints(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.decoder.mlp_opacity.state_dict(), path / "opacity_mlp.pt")
        torch.save(self.decoder.mlp_cov.state_dict(), path / "cov_mlp.pt")
        torch.save(self.decoder.mlp_color.state_dict(), path / "color_mlp.pt")
        if self.decoder.use_feat_bank:
            torch.save(
                self.decoder.mlp_feature_bank.state_dict(),
                path / "feature_bank_mlp.pt",
            )
        if self.decoder.embedding_appearance is not None:
            torch.save(
                self.decoder.embedding_appearance.state_dict(),
                path / "embedding_appearance.pt",
            )

    def load_mlp_checkpoints(self, path: str | Path) -> None:
        path = Path(path)
        self.decoder.mlp_opacity.load_state_dict(
            torch.load(path / "opacity_mlp.pt", weights_only=False)
        )
        self.decoder.mlp_cov.load_state_dict(
            torch.load(path / "cov_mlp.pt", weights_only=False)
        )
        self.decoder.mlp_color.load_state_dict(
            torch.load(path / "color_mlp.pt", weights_only=False)
        )
        if self.decoder.use_feat_bank:
            self.decoder.mlp_feature_bank.load_state_dict(
                torch.load(path / "feature_bank_mlp.pt", weights_only=False)
            )
        if self.decoder.embedding_appearance is not None:
            self.decoder.embedding_appearance.load_state_dict(
                torch.load(path / "embedding_appearance.pt", weights_only=False)
            )


MODELS: Dict[str, Type[BaseGaussianModel]] = {
    ScaffoldGSModel.model_name: ScaffoldGSModel,
}


def get_model_class(name: str) -> Type[BaseGaussianModel]:
    """Resolve a model class, lazily importing heavy/optional modules.

    ``hac_pp`` lives in :mod:`scaffold_gs.hacpp` and requires the HAC++
    CUDA extensions (``_gridencoder`` / ``arithmetic``), so it is only
    imported when actually requested.
    """
    if name in MODELS:
        return MODELS[name]
    if name == "hac_pp":
        from .hacpp import HACPlusModel

        MODELS[name] = HACPlusModel
        return HACPlusModel
    raise KeyError(f"Unknown model: {name!r}. Available: {sorted(MODELS)}")
