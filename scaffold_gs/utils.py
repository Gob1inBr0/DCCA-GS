"""Small utilities shared across the Scaffold-GS implementation."""

from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed python, numpy and torch (CUDA included)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def knn_distances(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Euclidean k-NN distances via scipy's KD-tree (CPU).

    Args:
        points: ``[N, 3]`` float array.
        k: number of neighbors (including the point itself).

    Returns:
        ``[N, k]`` distances sorted ascending; column ``1`` is the 1-NN
        distance used by Scaffold-GS for scale initialization.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    n = max(1, len(points))
    distances, _ = tree.query(points, k=min(k, n))
    if distances.ndim == 1:
        distances = distances[:, None]
    if distances.shape[1] < k:
        distances = np.pad(
            distances,
            ((0, 0), (0, k - distances.shape[1])),
            constant_values=1e-6,
        )
    return distances.astype(np.float32)


def voxelize_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-downsample ``[N, 3]`` points.

    The SfM points are rounded to the nearest voxel center and de-duplicated,
    exactly like the official Scaffold-GS ``voxelize_sample``.
    """
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0 when calling voxelize_points")
    coords = np.unique(np.round(points / voxel_size), axis=0)
    return coords * voxel_size


def median_nn_distance(points: np.ndarray) -> float:
    """Median 1-NN distance (used when ``voxel_size <= 0``)."""
    dist = knn_distances(points, k=2)[:, 1]
    return float(np.median(dist))


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
) -> Callable[[int], float]:
    """Exponential LR schedule used by the official 3DGS / Scaffold-GS."""

    def helper(step: int) -> float:
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        step = max(0, step)
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult
            if step < lr_delay_steps:
                delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * step / lr_delay_steps
            step = max(0, step - lr_delay_steps)
        else:
            delay_rate = 1.0
        lr = lr_init * delay_rate * 0.5 ** (step / max(max_steps, 1))
        if lr_final > 0:
            lr = max(lr, lr_final)
        return float(lr)

    return helper


def build_rotation_matrix(
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix, ``[N, 3, 3]``."""
    w, x, y, z = rotations.unbind(dim=-1)
    N = rotations.shape[0]
    R = torch.zeros((N, 3, 3), device=rotations.device, dtype=rotations.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Return the upper-triangular 6 elements of symmetric matrices."""
    return sym[:, [0, 1, 2, 4, 5, 7]]


def build_scaling_rotation(
    scales: torch.Tensor, rotations: torch.Tensor
) -> torch.Tensor:
    """Covariance ``R @ S @ S^T @ R^T`` as ``[N, 3, 3]``."""
    R = build_rotation_matrix(rotations)
    L = R * scales[:, None, :]
    return L @ L.transpose(1, 2)


def camera_extent_radius(camtoworlds: np.ndarray) -> float:
    """NeRF-style scene radius used as ``spatial_lr_scale``."""
    centers = camtoworlds[:, :3, 3]
    mean = centers.mean(axis=0)
    dists = np.linalg.norm(centers - mean, axis=1)
    return float(np.percentile(dists, 90)) + 1e-6


def to_torch_image(image_np: np.ndarray) -> torch.Tensor:
    """HWC uint8/float image -> CHW float tensor in [0, 1]."""
    arr = np.asarray(image_np)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    return tensor.permute(2, 0, 1).contiguous()
