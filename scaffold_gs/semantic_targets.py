"""One-time per-anchor DINO target refresh for semantic supervision.

Training starts from a fresh anchor set, so targets exported from another
model's anchors are misaligned by index. The correct Stage-B protocol is to
aggregate the per-view DINO features onto the training model's OWN anchors
once growth stops (``update_until``), then supervise from there on.

The aggregation reuses the renderer's ``visible_mask`` (prefilter) and
projects anchor centers with the gsplat camera (no dependence on the
per-Gaussian ``anchor_indices``/``gaussian_ids`` bookkeeping).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _sample_nearest(feat_map: torch.Tensor, xy: torch.Tensor, scale: float):
    hf, wf = feat_map.shape[:2]
    y = (xy[:, 1] / scale).round().long().clamp(0, hf - 1)
    x = (xy[:, 0] / scale).round().long().clamp(0, wf - 1)
    return feat_map[y, x]


def _project_anchors(cam, anchor: torch.Tensor, device):
    viewmats, Ks = cam.to_gsplat(device)
    view = viewmats[0]
    hom = torch.cat(
        [anchor, torch.ones(anchor.shape[0], 1, device=device)], dim=1
    )
    pv = hom @ view.T  # world -> camera (OpenGL convention, z < 0 in front)
    z = -pv[:, 2].clamp_min(1e-6)
    fx, fy = float(Ks[0, 0, 0]), float(Ks[0, 1, 1])
    cx, cy = float(Ks[0, 0, 2]), float(Ks[0, 1, 2])
    u = fx * pv[:, 0] / z + cx
    v = fy * pv[:, 1] / z + cy
    return torch.stack([u, v], dim=1)


def refresh_semantic_targets(
    model,
    dataset,
    cache_dir,
    *,
    pca_dims: int = 8,
    min_views: int = 3,
    device: str = "cuda",
) -> None:
    """Aggregate DINO caches onto the model's current anchors and install the
    per-anchor PCA target + coverage mask on ``model.core``."""
    cache_dir = Path(cache_dir)
    dev = torch.device(device)
    core = model.core
    n_total = int(core.get_anchor.shape[0])
    dino_sum = None
    counts = torch.zeros(n_total, device=dev)
    cams = dataset.train_cameras
    for vi, cam in enumerate(cams):
        p = cache_dir / "dino" / f"{vi:05d}.npz"
        if not p.exists():
            continue
        with torch.no_grad():
            out = model.render(cam, dataset.background)
        aidx = torch.nonzero(out.visible_mask).squeeze(-1)
        if aidx.numel() == 0:
            continue
        counts.index_add_(
            0, aidx, torch.ones_like(aidx, dtype=torch.float32)
        )
        px = _project_anchors(cam, core.get_anchor[aidx], dev)
        fm = torch.from_numpy(np.load(p)["feat"]).to(dev).float()
        v = _sample_nearest(fm, px, 14.0)
        if dino_sum is None:
            dino_sum = torch.zeros(n_total, v.shape[1], device=dev)
        dino_sum.index_add_(0, aidx, v)
        if vi % 50 == 0:
            print(f"[semantic_targets] refresh view {vi}/{len(cams)}",
                  flush=True)
    if dino_sum is None:
        raise RuntimeError("no DINO caches found under " + str(cache_dir))
    vals = dino_sum / counts.clamp_min(1).unsqueeze(-1)
    cov = counts >= min_views
    cv = vals[cov]
    z = (cv - cv.mean(0)) / cv.std(0).clamp_min(1e-6)
    _, _, v = torch.pca_lowrank(z, q=pca_dims, center=False)
    red = z @ v[:, :pca_dims]
    tgt = torch.zeros(n_total, pca_dims, device=dev)
    tgt[cov] = red
    core.semantic_target = tgt.contiguous()
    core.semantic_cov = cov.reshape(-1, 1).contiguous().float()
    core.semantic_refreshed = True
    print(
        f"[semantic_targets] refreshed {n_total} anchors, "
        f"covered={int(cov.sum())} ({100.0 * cov.float().mean():.1f}%)"
    )
