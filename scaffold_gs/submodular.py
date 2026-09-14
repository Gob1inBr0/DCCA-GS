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

import heapq

import numpy as np
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
    bh = max(1, (height + BLOCK - 1) // BLOCK)
    flat = pixel_ids[0].long()
    row = torch.div(flat, width, rounding_mode="floor")
    col = flat % width
    block_ids = (row // BLOCK) * bw + (col // BLOCK)
    # Per-view reduce BEFORE returning: raw pixel pairs are ~133M per view
    # (~2 GB in int64) at 890k anchors / 1600 px — the training process has
    # no headroom for that. Reduce to unique (anchor, block) pairs with
    # their pixel counts via one int64 key sort; output is orders smaller
    # and the greedy consumes (anchor, block, count) triplets.
    nb = bw * bh
    key = anchor_ids * nb + block_ids
    uniq_keys, counts = torch.unique(key, return_counts=True)
    ua = torch.div(uniq_keys, nb, rounding_mode="floor")
    ub = uniq_keys % nb
    return ua, ub, counts.float()


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
        max((a.max().item() + 1) for _, a, _ in edges_by_view)
    )
    pool_size = min(N, max(kappa + 1, kappa * int(pool_factor)))
    pool = torch.topk(base_scores, pool_size).indices if base_scores is not None \
        else torch.arange(N, device=device)

    # Edges are per-view reduced (anchor, block, pixel-count) triplets
    # (see view_edges). Merge across views: the same (anchor, block) seen
    # in two views sums its counts — identical totals to the old
    # all-views-raw-pairs unique, at a fraction of the memory.
    triplets = []
    for ua_v, ub_v, cnt_v in edges_by_view:
        if ua_v.numel() == 0:
            continue
        triplets.append(torch.stack([ua_v.long(), ub_v.long(), cnt_v.float()], dim=1))
    if not triplets:
        return pool[:0], (time.time() - t0) * 1000.0
    t = torch.cat(triplets, dim=0)
    uniq_ab, inverse = t[:, :2].unique(dim=0, return_inverse=True)
    ua = uniq_ab[:, 0].long()      # anchor idx
    ub = uniq_ab[:, 1].long()      # block idx
    mass = torch.zeros(ua.shape[0], device=ua.device)
    mass.scatter_add_(0, inverse, t[:, 2])
    if sens is not None:
        w = 1.0 + sens[ua].float()
    else:
        w = mass

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

    # Exact lazy greedy with a stale-gain priority queue. The previous
    # batched loop spent 272 s per projection at N~890k (889k per-candidate
    # GPU round-trips); the heap version pops-and-accepts most candidates
    # without recomputation and runs in tens of seconds, while returning
    # the EXACT greedy order (a popped candidate is only accepted when its
    # refreshed gain is >= the next stale head — valid because gains only
    # decrease as coverage grows).
    covered_t = torch.zeros(B, device=device)
    sel = torch.zeros(P, dtype=torch.bool, device=device)
    gains = torch.zeros(P, device=device)
    gains.scatter_add_(0, ua_dense, w_p)

    import heapq

    ua_np = ua_dense.detach().cpu().numpy()
    starts_np = starts.detach().cpu().numpy()
    ub_np = flat_ub.detach().cpu().numpy()
    w_np = flat_w.detach().cpu().float().numpy()
    covered = np.zeros(B, dtype=np.float64)

    heap = [(-float(g), int(i)) for i, g in enumerate(gains.detach().cpu().tolist())]
    heapq.heapify(heap)
    remaining = int(kappa)
    while remaining > 0 and heap:
        neg_g, i = heapq.heappop(heap)
        if sel[i]:
            continue
        s, e = int(starts_np[i]), int(starts_np[i + 1])
        if e > s:
            ub_i = ub_np[s:e]
            cur = float(
                np.clip(w_np[s:e] - covered[ub_i], 0.0, None).sum()
            )
        else:
            cur = 0.0
        # Stale-gain check: another candidate with a larger claim remains —
        # push back with the refreshed gain instead of accepting blind.
        if heap and -heap[0][0] > cur:
            heapq.heappush(heap, (-cur, i))
            continue
        if cur <= 0:
            break  # every remaining candidate has zero marginal
        sel[i] = True
        remaining -= 1
        if e > s:
            covered[ub_i] = np.maximum(covered[ub_i], w_np[s:e])

    if remaining > 0:
        # Pool swept with budget left: every unselected candidate is
        # zero-marginal in block space. Fill by base score so the budget is
        # spent exactly — size parity with the topk baseline is what makes
        # the G1 comparison size-matched.
        rest = torch.nonzero(~sel, as_tuple=False).squeeze(-1)
        if rest.numel():
            pool_scores = base_scores[pool] if base_scores is not None else torch.zeros(P, device=device)
            fill = rest[
                torch.argsort(pool_scores[rest], descending=True)[:remaining]
            ]
            sel[fill] = True

    keep_pool_idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
    keep = pool[keep_pool_idx]
    return keep, (time.time() - t0) * 1000.0
