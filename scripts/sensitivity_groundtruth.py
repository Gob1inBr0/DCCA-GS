"""A1: sensitivity ground-truth calibration (design doc 99 §6.3 / A1).

On a trained checkpoint, compute the Niedermayr-style ground-truth
sensitivity — the L1 norm of the gradient of the rendering loss w.r.t. each
anchor's pre-quantization features, accumulated over N training views — and
correlate it with:
  (a) the four I2 formula features (decoder-recomputable), and
  (b) the complexity-multiplier logits produced by mlp_complexity
      (the network the I6 supervision shaped).

Also reports the zero-sensitivity rate among rendered anchors (evidence for
or against a clamp/zeroing path on the backward pass) and writes per-anchor
arrays for offline scatter plots.

Usage:
  python scripts/sensitivity_groundtruth.py --ckpt <ckpt> --data-dir <data> \
    --out-dir <out> --device cuda --views 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.trainer import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--views", type=int, default=16)
    args = ap.parse_args()

    import random

    from scaffold_gs.losses import l1_loss as l1_fn

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _optim, iteration, _meta = load_checkpoint(args.ckpt, args.device)
    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=1,
        test_every=8,
        white_background=False,
        preload_images=False,
        max_width=1600,
        device=args.device,
    )
    background = dataset.background
    train_cams = list(dataset.train_cameras)
    rng = random.Random(0)
    cams = rng.sample(train_cams, min(args.views, len(train_cams)))
    core = model.core
    n_total = int(core.get_anchor.shape[0])

    s_gt = np.zeros(n_total, dtype=np.float64)
    seen = np.zeros(n_total, dtype=bool)

    model.train()
    for vi, cam in enumerate(cams):
        out = model.render(
            cam,
            background,
            is_training=True,
            retain_grad=True,
            appearance_id=cam.appearance_id,
            step=int(iteration),
        )
        pred = out.image[0].permute(2, 0, 1)
        gt = dataset.get_image(cam)
        loss = l1_fn(pred, gt).mean()
        loss.backward()
        g = out.gaussians.pre_quant_feat.grad
        if g is None:
            print(f"[A1] view {vi}: no grad on pre_quant_feat, skipping")
            continue
        per_g = g.norm(dim=-1, keepdim=True)  # [M,1] per rendered gaussian
        idx = out.gaussians.anchor_indices.long()  # [M]
        seen_np = np.zeros(n_total, dtype=bool)
        seen_np[idx.detach().cpu().numpy()] = True
        acc = torch.zeros(n_total, 1, device=per_g.device)
        acc.index_add_(0, idx, per_g.detach())
        s_gt += acc.double().cpu().numpy()
        seen |= torch.from_numpy(seen_np)
        print(f"[A1] view {vi + 1}/{len(cams)} done, "
              f"touched {int(seen_np.sum())} anchors", flush=True)

    # I2 formula features + complexity logits (full population, chunked).
    feat4 = np.zeros((n_total, 4), dtype=np.float32)
    logits = np.zeros((n_total, 3), dtype=np.float32)
    k = model.cfg.n_offsets
    with torch.no_grad():
        for start in range(0, n_total, 200_000):
            end = min(start + 200_000, n_total)
            anchor = core.get_anchor[start:end]
            ctx = core.calc_context_feat(anchor, caller="sensitivity_gt")
            grid_out = core.get_grid_mlp(ctx)
            (_mean, _scale, _prob, mean_scaling, _scale_scaling,
             mean_offsets, _scale_offsets, _qa, _qs, _qo) = torch.split(
                grid_out,
                [model.cfg.feat_dim, model.cfg.feat_dim, model.cfg.feat_dim,
                 6, 6, 3 * k, 3 * k, 1, 1, 1],
                dim=-1,
            )
            f_in = core.build_formula_complexity_input(
                anchor,
                mean_scaling.view(-1, 6),
                mean_offsets.view(-1, 3 * k),
                core.get_mask[start:end],
            )
            feat4[start:end] = f_in.detach().float().cpu().numpy()
            logits[start:end] = core.get_complexity_mlp(
                f_in
            ).detach().float().cpu().numpy()

    np.savez_compressed(
        out_dir / "sensitivity_gt.npz",
        s_gt=s_gt, seen=seen, feat4=feat4, logits=logits,
    )

    vis = seen & (s_gt > 0)
    zero_rate = float((seen & (s_gt == 0)).sum()) / max(1, int(seen.sum()))

    def corr(x: np.ndarray, y: np.ndarray) -> dict:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return {"pearson": None, "spearman": None, "n": int(m.sum())}
        xs, ys = x[m], y[m]
        pear = float(np.corrcoef(xs, ys)[0, 1])
        rx = np.argsort(np.argsort(xs)).astype(np.float64)
        ry = np.argsort(np.argsort(ys)).astype(np.float64)
        spear = float(np.corrcoef(rx, ry)[0, 1])
        return {"pearson": pear, "spearman": spear, "n": int(m.sum())}

    summary: dict = {
        "n_total": n_total,
        "n_seen": int(seen.sum()),
        "n_visible_nonzero": int(vis.sum()),
        "zero_sensitivity_rate_among_seen": zero_rate,
        "views": len(cams),
        "corr_s_gt_vs": {},
    }
    names = ["density", "scale_aniso_a", "scale_aniso_b", "offset_energy"]
    for j, nm in enumerate(names):
        summary["corr_s_gt_vs"][f"feat4_{nm}"] = corr(feat4[:, j], s_gt)
    for j, nm in enumerate(["logit_0", "logit_1", "logit_2"]):
        summary["corr_s_gt_vs"][nm] = corr(logits[:, j], s_gt)
    # |logits| aggregate magnitude vs sensitivity (the multiplier's absolute
    # claim on "this anchor matters").
    summary["corr_s_gt_vs"]["logit_l1"] = corr(
        np.abs(logits).sum(axis=1).astype(np.float64), s_gt
    )
    with open(out_dir / "a1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
