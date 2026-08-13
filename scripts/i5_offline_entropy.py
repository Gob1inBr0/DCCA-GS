"""I5 stage-0 offline feasibility: scalar vs lattice symbol entropy.

Loads a trained HAC++ checkpoint and estimates, with the learned Gaussian
entropy model (grid-MLP means/scales), the coded bits of feat/scaling/offsets
under (a) the current scalar quantization and (b) lattice vector quantization
(Z/D4/E8 with the doc's option-1 per-dimension Gaussian CDF).

Stop condition from the design doc: if the offline bit gain is < 5%, drop I5.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/i5_offline_entropy.py \
      --ckpt <ckpt_90000.pth> --lattice d4 --out runs/i5_offline.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.hacpp import HACPlusModel, anchor_codec_order
from scaffold_gs.lattice import vq_quantize
from scaffold_gs.trainer import load_checkpoint


def _model_bits(values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, q: torch.Tensor) -> float:
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-9))
    p = dist.cdf(values + 0.5 * q) - dist.cdf(values - 0.5 * q)
    return float((-torch.log2(p.clamp_min(1e-12))).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lattice", default="d4", choices=["z", "d4", "e8"])
    p.add_argument("--group-feat", type=int, default=4)
    p.add_argument("--group-scaling", type=int, default=4)
    p.add_argument("--group-offsets", type=int, default=4)
    p.add_argument("--content-aware", action="store_true")
    p.add_argument("--dither", action="store_true")
    p.add_argument("--out", default="runs/i5_offline.json")
    args = p.parse_args()

    model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    assert isinstance(model, HACPlusModel)
    core = model.core
    k = model.cfg.n_offsets
    device = model.device
    from hacplus.scene.gaussian_model import MAX_batch_size

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]
    print(f"[I5] {N} anchors, lattice={args.lattice}, "
          f"groups=({args.group_feat},{args.group_scaling},{args.group_offsets}), "
          f"content_aware={args.content_aware}, dither={args.dither}")

    # PCA cumulative variance of features (design doc stage 0, item 1).
    fc = feat.cpu().numpy()
    fc = fc - fc.mean(0)
    cov = np.cov(fc, rowvar=False)
    ev = np.linalg.eigvalsh(cov)[::-1]
    cum = np.cumsum(ev) / ev.sum()
    pca = {int(t): round(float(cum[min(t, len(cum)) - 1]), 4) for t in (1, 2, 5, 10, 25, 50)}
    print("[I5] feat PCA cumulative variance:", pca)

    scalar_bits = {"feat": 0.0, "scaling": 0.0, "offsets": 0.0}
    lattice_bits = {"feat": 0.0, "scaling": 0.0, "offsets": 0.0}

    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="i5_offline")
        out = core.get_grid_mlp(ctx)
        (
            mean, scale, prob, mean_scaling, scale_scaling,
            mean_offsets, scale_offsets, qa, qs, qo,
        ) = torch.split(
            out,
            [
                model.cfg.feat_dim, model.cfg.feat_dim, model.cfg.feat_dim,
                6, 6, 3 * k, 3 * k, 1, 1, 1,
            ],
            dim=-1,
        )
        qa = qa.repeat(1, model.cfg.feat_dim)
        qs = qs.repeat(1, 6)
        qo = qo.repeat(1, 3 * k)
        mean_offsets = mean_offsets.contiguous().view(-1)
        scale_offsets = scale_offsets.contiguous().view(-1)
        Q_feat = 1.0 * (1 + torch.tanh(qa))
        Q_scaling = 0.001 * (1 + torch.tanh(qs))
        Q_offsets = 0.2 * (1 + torch.tanh(qo))
        if args.content_aware and core.is_content_aware_quant_active():
            masks_slice = masks[start:end]
            (
                Q_feat, Q_scaling, Q_offsets, _, _, _, _,
            ) = core._codec_apply_content_aware_quant_params(
                "i5_offline", anchor_slice, masks_slice,
                Q_feat, Q_scaling, Q_offsets, None, None, None,
                mean_scaling.view(-1, 6), mean_offsets.view(-1, 3 * k),
            )

        arange_idx = torch.arange(start, end, device=device)
        n_num = end - start

        xf = feat[start:end]
        scalar_bits["feat"] += _model_bits(
            torch.round(xf / Q_feat) * Q_feat,
            mean, scale.clamp(min=1e-9), Q_feat,
        )
        step_f = Q_feat if args.content_aware else torch.full_like(Q_feat, 1.0)
        values_f, q_eff_f, _ = vq_quantize(
            xf, step_f, args.lattice, args.group_feat,
            0, arange_idx, 0, args.dither,
        )
        lattice_bits["feat"] += _model_bits(
            values_f, mean, scale.clamp(min=1e-9), q_eff_f
        )

        xs = scaling[start:end]
        scalar_bits["scaling"] += _model_bits(
            torch.round(xs / Q_scaling) * Q_scaling,
            mean_scaling, scale_scaling.clamp(min=1e-9), Q_scaling,
        )
        step_s = Q_scaling if args.content_aware else torch.full_like(Q_scaling, 0.001)
        values_s, q_eff_s, _ = vq_quantize(
            xs, step_s, args.lattice, args.group_scaling,
            0, arange_idx, 1, args.dither,
        )
        lattice_bits["scaling"] += _model_bits(
            values_s, mean_scaling, scale_scaling.clamp(min=1e-9), q_eff_s
        )

        xo = offsets[start:end].reshape(n_num, 3 * k)
        mask_flat = masks[start:end].repeat(1, 1, 3).reshape(-1, 3 * k).reshape(-1).bool()
        Q_off_flat = Q_offsets
        scalar_bits["offsets"] += _model_bits(
            (torch.round(xo / Q_off_flat) * Q_off_flat).reshape(-1)[mask_flat],
            mean_offsets[mask_flat], scale_offsets[mask_flat].clamp(min=1e-9),
            Q_off_flat.reshape(-1)[mask_flat],
        )
        step_o = Q_offsets if args.content_aware else torch.full_like(Q_offsets, 0.2)
        values_o, q_eff_o, _ = vq_quantize(
            xo, step_o, args.lattice, args.group_offsets,
            0, arange_idx, 2, args.dither,
        )
        lattice_bits["offsets"] += _model_bits(
            values_o.reshape(-1)[mask_flat],
            mean_offsets[mask_flat], scale_offsets[mask_flat].clamp(min=1e-9),
            q_eff_o.reshape(-1)[mask_flat],
        )

    tot_scalar = sum(scalar_bits.values())
    tot_lattice = sum(lattice_bits.values())
    gain = (tot_scalar - tot_lattice) / tot_scalar if tot_scalar > 0 else 0.0
    rows = {
        "iteration": int(iteration),
        "lattice": args.lattice,
        "groups": [args.group_feat, args.group_scaling, args.group_offsets],
        "content_aware": bool(args.content_aware),
        "dither": bool(args.dither),
        "pca_cumvar": pca,
        "scalar_bits": {k: round(v, 2) for k, v in scalar_bits.items()},
        "scalar_MB": round(tot_scalar / 8 / 1024 / 1024, 4),
        "lattice_bits": {k: round(v, 2) for k, v in lattice_bits.items()},
        "lattice_MB": round(tot_lattice / 8 / 1024 / 1024, 4),
        "total_gain": round(float(gain) * 100, 3),
        "pass_5pct": float(gain) >= 0.05,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(
        "[I5] OFFLINE GAIN: "
        f"{gain * 100:.2f}% -> {'PASS (>=5%)' if gain >= 0.05 else 'FAIL (<5%)'}"
    )


if __name__ == "__main__":
    main()
