"""Stable public view over the vendored HAC++ core.

The official ``GaussianModel`` exposes rendering/growth through public methods,
but training/export/codec code historically reached into private
``core._anchor``-style attributes. This view is the only place allowed to touch
those privates, so upgrading the vendored core or swapping HAC variants does
not break the gsplat2hac adapter.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class HACCoreView:
    """Thin, explicit interface to the vendored HAC++ ``GaussianModel``."""

    def __init__(self, core: Any) -> None:
        self._core = core

    # ------------------------------------------------------------------
    # Anchor attributes
    # ------------------------------------------------------------------

    @property
    def anchor(self) -> nn.Parameter:
        return self._core._anchor

    @anchor.setter
    def anchor(self, value: nn.Parameter) -> None:
        self._core._anchor = value

    @property
    def offset(self) -> nn.Parameter:
        return self._core._offset

    @offset.setter
    def offset(self, value: nn.Parameter) -> None:
        self._core._offset = value

    @property
    def mask(self) -> nn.Parameter:
        return self._core._mask

    @mask.setter
    def mask(self, value: nn.Parameter) -> None:
        self._core._mask = value

    @property
    def anchor_feat(self) -> nn.Parameter:
        return self._core._anchor_feat

    @anchor_feat.setter
    def anchor_feat(self, value: nn.Parameter) -> None:
        self._core._anchor_feat = value

    @property
    def scaling(self) -> nn.Parameter:
        return self._core._scaling

    @scaling.setter
    def scaling(self, value: nn.Parameter) -> None:
        self._core._scaling = value

    @property
    def rotation(self) -> nn.Parameter:
        return self._core._rotation

    @rotation.setter
    def rotation(self, value: nn.Parameter) -> None:
        self._core._rotation = value

    @property
    def opacity(self) -> nn.Parameter:
        return self._core._opacity

    @opacity.setter
    def opacity(self, value: nn.Parameter) -> None:
        self._core._opacity = value

    # ------------------------------------------------------------------
    # Bounds / state flags
    # ------------------------------------------------------------------

    @property
    def x_bound_min(self) -> torch.Tensor:
        return self._core.x_bound_min

    @x_bound_min.setter
    def x_bound_min(self, value: torch.Tensor) -> None:
        self._core.x_bound_min = value

    @property
    def x_bound_max(self) -> torch.Tensor:
        return self._core.x_bound_max

    @x_bound_max.setter
    def x_bound_max(self, value: torch.Tensor) -> None:
        self._core.x_bound_max = value

    @property
    def decoded_version(self) -> bool:
        return self._core.decoded_version

    @decoded_version.setter
    def decoded_version(self, value: bool) -> None:
        self._core.decoded_version = value

    @property
    def num_anchors(self) -> int:
        return self._core.get_anchor.shape[0]

    # ------------------------------------------------------------------
    # Public core accessors (passthroughs for the adapter)
    # ------------------------------------------------------------------

    def get_anchor(self) -> torch.Tensor:
        return self._core.get_anchor

    def get_scaling(self) -> torch.Tensor:
        return self._core.get_scaling

    def get_mask(self) -> torch.Tensor:
        return self._core.get_mask

    def get_mask_anchor(self) -> torch.Tensor:
        return self._core.get_mask_anchor

    def get_rotation(self) -> torch.Tensor:
        return self._core.get_rotation

    def get_encoding_params(self) -> torch.Tensor:
        return self._core.get_encoding_params()

    # ------------------------------------------------------------------
    # Decoder state
    # ------------------------------------------------------------------

    def decoder_state(self) -> Dict[str, Any]:
        core = self._core
        state = {
            "mlp_opacity": core.mlp_opacity.state_dict(),
            "mlp_cov": core.mlp_cov.state_dict(),
            "mlp_color": core.mlp_color.state_dict(),
            "mlp_grid": core.mlp_grid.state_dict(),
            "mlp_deform": core.mlp_deform.state_dict(),
            "encoding_xyz": core.encoding_xyz.state_dict(),
        }
        if core.use_feat_bank:
            state["mlp_feature_bank"] = core.mlp_feature_bank.state_dict()
        return state

    def load_decoder_state(self, state: Dict[str, Any]) -> None:
        core = self._core
        core.mlp_opacity.load_state_dict(state["mlp_opacity"])
        core.mlp_cov.load_state_dict(state["mlp_cov"])
        core.mlp_color.load_state_dict(state["mlp_color"])
        core.mlp_grid.load_state_dict(state["mlp_grid"])
        core.mlp_deform.load_state_dict(state["mlp_deform"])
        core.encoding_xyz.load_state_dict(state["encoding_xyz"])
        if core.use_feat_bank:
            core.mlp_feature_bank.load_state_dict(state["mlp_feature_bank"])

    # ------------------------------------------------------------------
    # Hash-grid parameters (nested private modules live here only)
    # ------------------------------------------------------------------

    def set_hash_params(
        self, hash_decoded: torch.Tensor, requires_grad: bool = False
    ) -> None:
        core = self._core
        if core.use_2D:
            len_3d = core.encoding_xyz.encoding_xyz.params.shape[0]
            len_2d = core.encoding_xyz.encoding_xy.params.shape[0]
            core.encoding_xyz.encoding_xyz.params = nn.Parameter(
                hash_decoded[0:len_3d], requires_grad=requires_grad
            )
            core.encoding_xyz.encoding_xy.params = nn.Parameter(
                hash_decoded[len_3d : len_3d + len_2d],
                requires_grad=requires_grad,
            )
            core.encoding_xyz.encoding_xz.params = nn.Parameter(
                hash_decoded[len_3d + len_2d : len_3d + 2 * len_2d],
                requires_grad=requires_grad,
            )
            core.encoding_xyz.encoding_yz.params = nn.Parameter(
                hash_decoded[len_3d + 2 * len_2d : len_3d + 3 * len_2d],
                requires_grad=requires_grad,
            )
        else:
            core.encoding_xyz.params = nn.Parameter(
                hash_decoded, requires_grad=requires_grad
            )
