"""Background-feature codebook (方案 C, design doc
``docs/02-design/背景场与ADMM覆盖约束_设计.md`` §2.3).

Background anchors (top coverage-area quantile, encoder-side measurement)
share a small codebook of feature vectors: their 32-dim features are
replaced by k-means centroids and are NOT entropy-coded. The bitstream
carries (a) one index per background anchor, (b) the fp16 codebook, and
(c) the packed background flags — flags are encoder-measured, so they
travel in the payload rather than being recomputed decode-side.

Everything here is a pure tensor/numpy function: selection, clustering,
payload round-trip, and reconstruction. The codec integration lives in
``hacpp.encode_attributes`` / ``decode_attributes``; the size accounting
question the byte-account script answers (what share of feat bytes the
background actually owns) gates whether this is worth enabling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

BG_CODEBOOK_FILENAME = "bg_codebook.npz"
BG_CODEBOOK_HEADER_KEY = "bg_codebook"


def derive_flags(
    area: np.ndarray,
    seen: Optional[np.ndarray] = None,
    quantile: float = 0.9,
) -> np.ndarray:
    """Background flags: anchors above the (1-quantile) area cut, among
    anchors the sampled views actually saw. Unseen anchors are foreground
    by default (no coverage evidence)."""
    area = np.asarray(area, dtype=np.float64)
    if area.ndim != 1:
        raise ValueError(f"area must be 1-D, got shape {area.shape}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    flags = np.zeros(area.shape[0], dtype=bool)
    valid = np.isfinite(area)
    if seen is not None:
        valid = valid & np.asarray(seen, dtype=bool)
    if valid.sum() == 0:
        return flags
    cut = np.quantile(area[valid], quantile)
    flags[valid & (area >= cut)] = True
    return flags


def flags_from_npz(
    path: str | Path,
    n_expected: int,
    quantile: float,
) -> np.ndarray:
    """Load the analysis dump (``anchor_stats.npz``) and derive flags.
    Accepts either a precomputed ``flags`` key or ``area`` (+``seen``).

    NOTE on identity: the flags are only meaningful for the exact checkpoint
    the dump was produced from (row order = that model's anchor order). The
    codec writes the anchor hash into the payload at encode time; dumps
    produced by ``dump_anchor_area_opacity.py`` SHOULD also store an
    ``anchor_sha256`` — when it is absent the codec warns loudly.
    """
    with np.load(Path(path)) as data:
        if "flags" in data:
            flags = np.asarray(data["flags"], dtype=bool)
        elif "area" in data:
            flags = derive_flags(
                data["area"],
                data["seen"] if "seen" in data else None,
                quantile,
            )
        else:
            raise KeyError(f"{path} has neither 'flags' nor 'area'")
    if flags.shape[0] != n_expected:
        raise ValueError(
            f"flags length {flags.shape[0]} != anchor count {n_expected}"
        )
    return flags


def kmeans(
    X: torch.Tensor,
    K: int,
    iters: int = 25,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic Lloyd k-means on CPU-safe tensors.

    Returns (centroids [K, D] float32, assign [N] int64). Empty clusters
    are re-seeded from the points farthest from their centroid; K is
    clamped to N.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D [N, D], got {tuple(X.shape)}")
    n = X.shape[0]
    if n == 0:
        raise ValueError("kmeans needs at least one row")
    K = max(1, min(int(K), int(n)))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    Xc = X.detach().to("cpu", torch.float32)
    init = torch.randperm(n, generator=gen)[:K]
    centroids = Xc[init].clone()
    assign = torch.zeros(n, dtype=torch.int64)
    for it in range(int(iters)):
        dists = torch.cdist(Xc, centroids)
        new_assign = dists.argmin(dim=1)
        if it > 0 and torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        used = torch.zeros(n, dtype=torch.bool)
        for kk in range(K):
            member = assign == kk
            if member.any():
                centroids[kk] = Xc[member].mean(dim=0)
            else:
                # Re-seed at the worst-fit point; skip points already taken
                # as re-seeds this round (otherwise several empty clusters
                # all grab the same worst point and duplicate centroids).
                d = dists.min(dim=1).values.clone()
                d[used] = -1.0
                worst = int(d.argmax().item())
                used[worst] = True
                centroids[kk] = Xc[worst]
    # Final assignment against the final centroids (Lloyd invariant).
    assign = torch.cdist(Xc, centroids).argmin(dim=1)
    return centroids, assign


def build_codebook(
    feat: torch.Tensor,
    flags: torch.Tensor,
    codebook_size: int,
    iters: int = 25,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Replace flag-raised rows of ``feat`` [N, D] with k-means centroids.

    Returns (feat_replaced, payload_dict). The payload holds the fp16
    codebook, uint8/16 indices (row order of ``feat``), and packed flags —
    all numpy, ready for ``save_payload``.
    """
    if feat.dim() != 2:
        raise ValueError(f"feat must be 2-D, got {tuple(feat.shape)}")
    if flags.shape[0] != feat.shape[0] or flags.device != feat.device:
        raise ValueError("flags must be a same-length bool tensor on feat.device")
    bg = flags.bool()
    n_bg = int(bg.sum().item())
    feat_out = feat.detach().clone()
    if n_bg == 0:
        payload = {
            "codebook": np.zeros((0, feat.shape[1]), dtype=np.float16),
            "indices": np.zeros((0,), dtype=np.uint8),
            "flags_packed": np.packbits(bg.detach().to("cpu", torch.uint8).numpy()),
            "n_total": int(feat.shape[0]),
            "codebook_size": 0,
            "version": 1,
        }
        return feat_out, payload
    centroids, assign = kmeans(
        feat[bg], codebook_size, iters=iters, seed=seed
    )
    # fp16 is the payload's storage dtype: snap the centroids to it BEFORE
    # writing them into the features, so the encoder-side replacement and
    # the decoder-side reconstruction are bit-identical by construction.
    centroids = centroids.to(torch.float16).to(torch.float32)
    feat_out[bg] = centroids.to(device=feat.device, dtype=feat.dtype)[
        assign.to(feat.device)
    ]
    max_idx = int(centroids.shape[0] - 1)
    idx_dtype = np.uint8 if max_idx < 256 else np.uint16
    payload = {
        "codebook": centroids.numpy().astype(np.float16),
        "indices": assign.numpy().astype(idx_dtype),
        "flags_packed": np.packbits(bg.detach().to("cpu", torch.uint8).numpy()),
        "n_total": int(feat.shape[0]),
        "codebook_size": int(centroids.shape[0]),
        "version": 1,
    }
    return feat_out, payload


def save_payload(payload: Dict[str, Any], path: str | Path) -> int:
    """Write the payload npz; returns its size in bytes (for accounting)."""
    path = Path(path)
    np.savez_compressed(path, **payload)
    return path.stat().st_size


def load_payload(path: str | Path) -> Dict[str, Any]:
    with np.load(Path(path)) as data:
        payload = {k: data[k] for k in data.files}
    version = int(payload.get("version", -1))
    if version != 1:
        raise ValueError(f"unsupported bg_codebook payload version {version}")
    return payload


def unpack_flags(payload: Dict[str, Any], device: torch.device) -> torch.Tensor:
    n = int(payload["n_total"])
    flags = np.unpackbits(payload["flags_packed"], count=n).astype(bool)
    return torch.from_numpy(flags).to(device)


def reconstruct(
    feat_fg: torch.Tensor,
    payload: Dict[str, Any],
    device: torch.device,
    feat_dim: int,
) -> torch.Tensor:
    """Assemble the full feature matrix from fg-only decoded rows.

    ``feat_fg`` holds the non-background rows in row order (the same
    subset the encoder entropy-coded); background rows come from the
    codebook via the stored indices.
    """
    flags = unpack_flags(payload, device)
    n = flags.shape[0]
    codebook = torch.from_numpy(
        payload["codebook"].astype(np.float32)
    ).to(device)
    indices = torch.from_numpy(payload["indices"].astype(np.int64)).to(device)
    feat = torch.empty(n, feat_dim, device=device, dtype=feat_fg.dtype)
    if feat_fg.shape[0] != int((~flags).sum().item()):
        raise ValueError(
            f"fg row count mismatch: got {feat_fg.shape[0]}, "
            f"expected {(~flags).sum().item()}"
        )
    if (~flags).any():
        feat[~flags] = feat_fg
    if flags.any():
        feat[flags] = codebook[indices]
    return feat


def payload_file_bytes(path: str | Path) -> int:
    """Charged size of the payload (compressed file), in bytes."""
    return Path(path).stat().st_size
