"""Fit the R4 conditional entropy predictor (scaling/offsets) on a checkpoint.

The predictor is fit on the codec-order train split (Morton-first 80%) and
early-stopped on the last 20%, exactly like Stage-A R4. It is then saved as
``--out`` and consumed by ``train.py compress --attr-ctx <file>``.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/fit_attr_ctx.py \
      --ckpt runs/4-28_i6_90k_h32_l0p002/ckpts/ckpt_90000.pth \
      --out runs/attr_ctx_4_28_h32_l0p002.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scaffold_gs.attr_ctx import attr_ctx_meta, fit_attr_ctx_predictor
from scaffold_gs.hacpp import HACPlusModel, anchor_codec_order
from scaffold_gs.trainer import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/attr_ctx.pt")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fields",
        default="scaling",
        choices=("scaling", "both"),
        help="which entropy params to adjust (default scaling: reliable net gain)",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    assert isinstance(model, HACPlusModel)
    core = model.core
    k_offsets = model.cfg.n_offsets
    feat_dim = model.cfg.feat_dim
    from hacplus.scene.gaussian_model import MAX_batch_size

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]
    print(f"[fit_attr_ctx] {N} anchors, iteration={iteration}", flush=True)

    parts = {name: [] for name in (
        "ctx", "feat_q", "xs", "xo", "q_s", "q_o",
        "mean_s", "scale_s", "mean_o", "scale_o", "mask",
    )}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="fit_attr_ctx")
        out = core.get_grid_mlp(ctx)
        (
            mean, scale, prob, mean_scaling, scale_scaling,
            mean_offsets, scale_offsets, qa, qs, qo,
        ) = torch.split(
            out,
            [feat_dim, feat_dim, feat_dim, 6, 6, 3 * k_offsets, 3 * k_offsets, 1, 1, 1],
            dim=-1,
        )
        qa = qa.repeat(1, feat_dim)
        qs = qs.repeat(1, 6)
        qo = qo.repeat(1, 3 * k_offsets)
        Q_feat = 1.0 * (1 + torch.tanh(qa))
        Q_scaling = 0.001 * (1 + torch.tanh(qs))
        Q_offsets = 0.2 * (1 + torch.tanh(qo))
        if core.is_content_aware_quant_active():
            masks_slice = masks[start:end]
            (
                Q_feat, Q_scaling, Q_offsets, _, _, _, _,
            ) = core._codec_apply_content_aware_quant_params(
                "fit_attr_ctx", anchor_slice, masks_slice,
                Q_feat, Q_scaling, Q_offsets, None, None, None,
                mean_scaling.view(-1, 6),
                mean_offsets.view(-1, 3 * k_offsets),
            )
        feat_q = torch.round(feat[start:end] / Q_feat) * Q_feat
        xs = torch.round(scaling[start:end] / Q_scaling) * Q_scaling
        xo = torch.round(
            offsets[start:end].reshape(-1, 3 * k_offsets) / Q_offsets
        ) * Q_offsets
        msk = masks[start:end].reshape(-1, k_offsets)
        msk_flat = msk.repeat(1, 3).reshape(-1, 3 * k_offsets).bool()
        xo = torch.where(msk_flat, xo, torch.zeros_like(xo))
        parts["ctx"].append(ctx)
        parts["feat_q"].append(feat_q)
        parts["xs"].append(xs)
        parts["xo"].append(xo)
        parts["q_s"].append(Q_scaling)
        parts["q_o"].append(Q_offsets)
        parts["mean_s"].append(mean_scaling)
        parts["scale_s"].append(scale_scaling.clamp(min=1e-9))
        parts["mean_o"].append(mean_offsets)
        parts["scale_o"].append(scale_offsets.clamp(min=1e-9))
        parts["mask"].append(msk)

    g = {
        name: torch.cat(v, dim=0).contiguous().detach()
        for name, v in parts.items()
    }
    g["feat_dim"] = feat_dim
    g["n_offsets"] = k_offsets
    n_train = int(N * (1.0 - args.val_fraction))
    res = fit_attr_ctx_predictor(
        g,
        n_train,
        hidden=args.hidden,
        steps=args.steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        fields=args.fields,
    )
    pred = res["predictor"]
    stats = res["stats"]
    for key in (
        "base_scaling_bits", "adj_scaling_bits",
        "base_offsets_bits", "adj_offsets_bits",
        "scaling_gain_pct", "offsets_gain_pct", "n_params",
    ):
        v = stats[key]
        if isinstance(v, float):
            stats[key] = round(v, 4)
    stats["n_params"] = int(stats["n_params"])
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": pred.state_dict(),
            "meta": attr_ctx_meta(pred),
            "stats": stats,
            "iteration": int(iteration),
            "N_anchors": int(N),
        },
        out,
    )
    print(f"[fit_attr_ctx] saved {out}", flush=True)


if __name__ == "__main__":
    main()
