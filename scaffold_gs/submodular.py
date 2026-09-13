"""Submodular set-cover anchor selection (Phase 1, design doc §4.1).

Replaces the independent per-anchor top-k inside the ADMM projection with a
greedy maximization of the covered pixel-block mass:

    max_{|S|=kappa}  sum_b  w_b * max_{i in S} c_{i,b}      (blocks counted once)

v1: w_b = 1 (pure coverage).  v2: w_b = mean sensitivity of the anchors
contributing to block b (coverage x importance).

Engineering cost controls (design §4.1): 8x8 pixel binning, candidate pool of
the top-2kappa linear scores, lazy greedy with stale-bound pruning.
"""

from __future__ import annotations

import torch

BLOCK = 8  # pixel binning


def build_bipartite(
    meta: dict,
    gids: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (block_ids [M], anchor_ids [M]) edge list from packed meta.

    Reuses the same intersection bookkeeping as compute_contribution_areas:
    per-pixel contributing gaussians -> their anchor indices -> 8x8 blocks.
    """
    from gsplat import rasterize_to_indices_in_range

    means2d = meta["means2d"]
    conics = meta["conics"]
    opacities = meta["opacities"]
    if means2d.dim() == 2:
        means2d = means2d[None]
        conics = conics[None]
        opacities = opacities[None]
    n_gauss = int(means2d.shape[-2])
    if n_gauss == 0:
        return None, None
    transmittances = torch.ones(
        means2d.shape[:-2] + (height, width), device=device
    )
    gs_ids, pixel_ids, _ = rasterize_to_indices_in_range(
        0, 1_000_000_000, transmittances, means2d, conics, opacities,
        width, height, meta["tile_size"],
    )[:3]
    if gs_ids is None or gs_ids.numel() == 0:
        return None, None
    anchor_ids = gids.long()[gs_ids.long()]
    bh = max(1, height // BLOCK)
    bw = max(1, width // BLOCK)
    # Flat pixel index of image 0 -> (row, col) -> 8x8 block id.
    flat = pixel_ids[0].long()
    row = torch.div(flat, width, rounding_mode="floor")
    col = flat % width
    block_ids = (row // BLOCK) * bw + (col // BLOCK)
    return block_ids, anchor_ids


def view_edges(
    model: BaseGaussianModel,
    cam,
    background: torch.Tensor,
    device: torch.device,
    gids: torch.Tensor,
    meta: dict,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build (block_ids, anchor_ids) edges for one rendered view.

    Mirrors _intersection_weights' calling convention exactly: the packed
    intersection API on this gsplat requires isect_offsets + flatten_ids.
    """
    from gsplat import rasterize_to_indices_in_range

    means2d = meta["means2d"]
    conics = meta["conics"]
    opacities = meta["opacities"]
    if means2d.dim() == 2:
        means2d = means2d[None]
        conics = conics[None]
        opacities = opacities[None]
    width = int(meta["width"])
    height = int(meta["height"])
    tile_size = int(meta["tile_size"])
    if int(means2d.shape[-2]) == 0:
        return None
    transmittances = torch.ones(
        means2d.shape[:-2] + (height, width), device=device
    )
    gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        0, 1_000_000_000, transmittances, means2d, conics, opacities,
        width, height, tile_size,
        meta["isect_offsets"], meta["flatten_ids"],
    )[:3]
    if gs_ids is None or gs_ids.numel() == 0:
        return None
    anchor_ids = gids.long()[gs_ids.long()]
    bw = max(1, width // BLOCK)
    flat = pixel_ids[0].long()
    row = torch.div(flat, width, rounding_mode="floor")
    col = flat % width
    block_ids = (row // BLOCK) * bw + (col // BLOCK)
    return block_ids, anchor_ids


def submodular_greedy_select(
    edges_by_view: list,
    kappa: int,
    sens: torch.Tensor | None,
    pool_factor: int = 2,
    base_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Lazy-greedy coverage selection.

    edges_by_view: list of (block_ids, anchor_ids) per rendered view.
    kappa: budget. sens: optional [N] sensitivity for v2 weighting.
    base_scores: [N] linear scores used to build the top-2kappa candidate pool.
    Returns (keep_anchor_indices, elapsed_ms).
    """
    import time

    t0 = time.time()
    device = base_scores.device if base_scores is not None else edges_by_view[0][0].device
    N = int(base_scores.shape[0]) if base_scores is not None else int(
        max((a.max().item() + 1) for _, a in edges_by_view)
    )
    pool_size = min(N, max(kappa + 1, kappa * int(pool_factor)))
    pool = torch.topk(base_scores, pool_size).indices if base_scores is not None \
        else torch.arange(N, device=device)

    # Aggregate per (anchor, block) coverage mass across views.
    # Unique pairs first (a block may contain several pixels of one anchor).
    pair_parts = []
    for block_ids, anchor_ids in edges_by_view:
        pair_parts.append(torch.stack([anchor_ids.long(), block_ids.long()], dim=1))
    pairs = torch.cat(pair_parts, dim=0)
    uniq_pairs, mass = pairs.unique(dim=0, return_counts=True)
    ua = uniq_pairs[:, 0].long()   # anchor idx
    ub = uniq_pairs[:, 1].long()   # block idx
    if sens is not None:
        w = 1.0 + sens[ua].float()
    else:
        w = mass.float()

    # Compact block ids to [0, B) BEFORE filtering to the pool
    ub_comp, block_map = torch.unique(ub, return_inverse=True)
    ub_c = block_map
    B = int(ub_comp.numel())
    in_pool = torch.zeros(N, dtype=torch.bool, device=device)
    in_pool[pool] = True
    m = in_pool[ua]
    ua_p, w_p, ub_p = ua[m], w[m], ub_c[m]
    # Vectorized dense mapping: searchsorted over the sorted pool
    pool_sorted, pool_argsort = torch.sort(pool)
    pos_in_sorted = torch.searchsorted(pool_sorted, ua_p)
    pos_in_sorted = pos_in_sorted.clamp_max(pool_sorted.numel() - 1)
    ok = pool_sorted[pos_in_sorted] == ua_p
    # (all ua_p are in the pool by construction of m)
    dense = torch.zeros_like(pos_in_sorted)
    dense[ok] = pool_argsort[pos_in_sorted[ok]]
    ua_dense = dense
    P = int(pool.numel())

    # CSR: for each pool anchor, its (block, weight) edges
    order = torch.argsort(ua_dense, stable=True)
    ua_sorted = ua_dense[order]
    w_sorted = w_p[order].float()
    ub_sorted = ub_p[order]
    counts = torch.bincount(ua_sorted, minlength=P)
    starts = torch.zeros(P + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=starts[1:])
    flat_ub = ub_sorted
    flat_w = w_sorted

    # Vectorized batched greedy: each round re-evaluates the marginal gains
    # of a candidate batch (per-candidate GPU round-trips are infeasible at
    # kappa~1e5), takes the batch top, updates coverage, repeats.
    covered = torch.zeros(B, device=device)
    sel = torch.zeros(P, dtype=torch.bool, device=device)
    gains = torch.zeros(P, device=device)
    gains.scatter_add_(0, ua_dense, w_p)
    BATCH = 4096
    order_hint = torch.argsort(gains, descending=True)
    pos = 0
    remaining = kappa
    while remaining > 0 and pos < P:
        batch_idx = order_hint[pos: pos + max(BATCH, remaining)]
        pos += batch_idx.numel()
        bg = torch.zeros(batch_idx.numel(), device=device)
        for j in range(batch_idx.numel()):
            i = int(batch_idx[j])
            s, e = int(starts[i]), int(starts[i + 1])
            bg[j] = (flat_w[s:e] - covered[flat_ub[s:e]]).clamp_min(0).sum()
        take = torch.argsort(bg, descending=True)[:remaining]
        for j in take.tolist():
            i = int(batch_idx[j])
            if bg[j] <= 0:
                continue
            sel[i] = True
            remaining -= 1
            s, e = int(starts[i]), int(starts[i + 1])
            cov_idx = flat_ub[s:e]
            covered[cov_idx] = torch.maximum(covered[cov_idx], flat_w[s:e])

    keep_pool_idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
    keep = pool[keep_pool_idx]
    return keep, (time.time() - t0) * 1000.0
