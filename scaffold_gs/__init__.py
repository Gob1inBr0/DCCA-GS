"""Scaffold-GS on gsplat.

A faithful re-implementation of Scaffold-GS
(CVPR 2024, "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering")
built on top of the ``gsplat`` rasterizer.

The package is designed to be extended with anchor-based compression methods
such as HAC / HAC++: see :mod:`scaffold_gs.codec` and
:meth:`scaffold_gs.model.BaseGaussianModel.export_attributes`.
"""

from .config import DataConfig, ModelConfig, OptimConfig, TrainConfig
from .model import BaseGaussianModel, MODELS, ScaffoldGSModel

__all__ = [
    "BaseGaussianModel",
    "DataConfig",
    "MODELS",
    "ModelConfig",
    "OptimConfig",
    "ScaffoldGSModel",
    "TrainConfig",
]
