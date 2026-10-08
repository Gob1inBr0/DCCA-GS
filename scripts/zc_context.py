"""Decoder-recomputable contexts for UAV-DCCA-ZC entropy probes.

These helpers are intentionally pure NumPy utilities so they can be unit-tested
without CUDA, HAC++ extensions, or the constriction range coder.
"""

from __future__ import annotations

import numpy as np


def quantile_buckets(x, n=3):
    """Stable n-way buckets for scalar decoder-side contexts."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.int16)
    if np.all(x == x[0]):
        return np.zeros(x.shape, dtype=np.int16)
    qs = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])
    return np.digitize(x, qs).astype(np.int16)


def resolve_cell_size(anchor_xyz, requested=0.0, divisor=512.0):
    """Return a deterministic cell size.

    Positive requested values are used as-is. Non-positive values use the scene
    bounding-box diagonal divided by `divisor`, which is recomputable from the
    decoded geometry and avoids per-scene hand tuning.
    """
    requested = float(requested)
    if requested > 0:
        return requested, "fixed"
    xyz = np.asarray(anchor_xyz, dtype=np.float64)
    if xyz.size == 0:
        return 1.0, "auto_empty"
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    if diag <= 0:
        return 1.0, "auto_degenerate"
    return diag / float(divisor), f"bbox_diag/{float(divisor):g}"


def cell_ids(anchor_xyz, cell_size):
    """Deterministic coarse 3D cells from decoded anchor coordinates."""
    if cell_size <= 0:
        raise ValueError("cell size must be positive")
    grid = np.floor(np.asarray(anchor_xyz, dtype=np.float64) / cell_size)
    grid = grid.astype(np.int64)
    _, inv = np.unique(grid, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def build_zc_anchor_context(anchor_xyz, cell_size):
    """Build density buckets and deterministic cell roots.

    The root of a cell is the first anchor in decoded row order inside that
    cell. This is cheap, deterministic, and requires no transmitted links.
    """
    cell = cell_ids(anchor_xyz, cell_size)
    counts = np.bincount(cell)
    density_bucket = quantile_buckets(counts[cell], 3)
    root = np.empty(cell.shape[0], dtype=np.int64)
    first = {}
    for idx, cid in enumerate(cell.tolist()):
        first.setdefault(cid, idx)
    for idx, cid in enumerate(cell.tolist()):
        root[idx] = first[cid]
    return {
        "cell": cell,
        "density_bucket": density_bucket,
        "root_anchor": root,
        "num_cells": int(counts.size),
    }


def combine_zc_context(coarse_ctx, mode, density_bucket, root_bucket):
    """Compose the standard coarse-symbol context with optional ZC buckets."""
    coarse_ctx = np.asarray(coarse_ctx, dtype=np.int16)
    if mode == "off":
        return coarse_ctx, 1
    if mode == "density":
        zc = np.asarray(density_bucket, dtype=np.int16)
        card = 3
    elif mode == "root":
        zc = np.asarray(root_bucket, dtype=np.int16)
        card = 3
    elif mode == "density-root":
        zc = (np.asarray(density_bucket, dtype=np.int16) * 3
              + np.asarray(root_bucket, dtype=np.int16))
        card = 9
    else:
        raise ValueError(f"unknown zc context mode: {mode}")
    return (coarse_ctx * card + zc).astype(np.int16), card
