"""Compute codec efficiency = actual bitstream bits / estimated cross-entropy bits.

The arithmetic coder is already near-optimal given its probability model, so
this number tells you whether the remaining gap comes from the coder itself
(efficiency < ~1.05) or from a mismatched probability model (larger gap).

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/codec_efficiency.py \
      --ckpt <ckpt_90000.pth> \
      --work-dir runs/codec_efficiency
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from scaffold_gs.hacpp import HACPlusCodec, anchor_codec_order
from scaffold_gs.trainer import load_checkpoint


def _gaussian_bits(x, mean, scale, q):
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-9))
    p = dist.cdf(x + 0.5 * q) - dist.cdf(x - 0.5 * q)
    return -torch.log2(p.clamp_min(1e-12))


@torch.no_grad()
def estimate_cross_entropy_bits(model) -> dict:
    """Replicate the codec's probability path without arithmetic coding."""
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

    bits_feat = 0.0
    bits_scaling = 0.0
    bits_offsets = 0.0

    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="codec_efficiency")
        out = core.get_grid_mlp(ctx)
        (
            mean,
            scale,
            prob,
            mean_scaling,
            scale_scaling,
            mean_offsets,
            scale_offsets,
            qa,
            qs,
            qo,
        ) = torch.split(
            out,
            [
                model.cfg.feat_dim,
                model.cfg.feat_dim,
                model.cfg.feat_dim,
                6,
                6,
                3 * k,
                3 * k,
                1,
                1,
                1,
            ],
            dim=-1,
        )
        qa = qa.repeat(1, model.cfg.feat_dim)
        qs = qs.repeat(1, 6)
        qo = qo.repeat(1, 3 * k)
        Q_feat = 1.0 * (1 + torch.tanh(qa))
        Q_scaling = 0.001 * (1 + torch.tanh(qs))
        Q_offsets = 0.2 * (1 + torch.tanh(qo))
        if core.is_content_aware_quant_active():
            masks_slice = masks[start:end]
            (
                Q_feat,
                Q_scaling,
                Q_offsets,
                _,
                _,
                _,
                _,
            ) = core._codec_apply_content_aware_quant_params(
                "codec_efficiency",
                anchor_slice,
                masks_slice,
                Q_feat,
                Q_scaling,
                Q_offsets,
                None,
                None,
                None,
                mean_scaling.view(-1, 6),
                mean_offsets.view(-1, 3 * k),
            )

        # ---- feat: mixed Gaussian + channel-context (replicate encode loop) ----
        feat_slice = feat[start:end]
        feat_q = torch.round(feat_slice / Q_feat) * Q_feat
        mean_scale = torch.cat([mean, scale, prob], dim=-1)
        feat_decoded = torch.zeros_like(feat_q)
        for cc in range(model.cfg.feat_dim // 10):
            mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
                feat_decoded, mean_scale, to_dec=cc
            )
            probs = torch.softmax(
                torch.stack(
                    [prob[:, cc * 10 : cc * 10 + 10], prob_adj], dim=-1
                ),
                dim=-1,
            )
            sl = slice(cc * 10, cc * 10 + 10)
            q_sl = Q_feat[:, sl]
            dist1 = torch.distributions.Normal(
                mean[:, sl], scale[:, sl].clamp_min(1e-9)
            )
            dist2 = torch.distributions.Normal(
                mean_adj, scale_adj.clamp_min(1e-9)
            )
            p = (
                probs[..., 0]
                * (dist1.cdf(feat_q[:, sl] + 0.5 * q_sl) - dist1.cdf(feat_q[:, sl] - 0.5 * q_sl))
                + probs[..., 1]
                * (dist2.cdf(feat_q[:, sl] + 0.5 * q_sl) - dist2.cdf(feat_q[:, sl] - 0.5 * q_sl))
            )
            bits_feat += (-torch.log2(p.clamp_min(1e-12))).sum().item()
            feat_decoded[:, sl] = feat_q[:, sl]

        # ---- scaling ----
        scaling_slice = scaling[start:end].reshape(-1)
        scaling_q = torch.round(scaling_slice / Q_scaling.reshape(-1)) * Q_scaling.reshape(-1)
        bits_scaling += (
            _gaussian_bits(
                scaling_q,
                mean_scaling.reshape(-1),
                scale_scaling.reshape(-1).clamp_min(1e-9),
                Q_scaling.reshape(-1),
            )
            .sum()
            .item()
        )

        # ---- offsets (masked) ----
        mask_bool = (
            masks[start:end]
            .repeat(1, 1, 3)
            .view(-1, 3 * k)
            .view(-1)
            .bool()
        )
        offsets_slice = offsets[start:end].reshape(-1, 3 * k).reshape(-1)
        offsets_q = (
            torch.round(offsets_slice / Q_offsets.reshape(-1)) * Q_offsets.reshape(-1)
        )
        offsets_q[~mask_bool] = 0.0
        bits_offsets += (
            _gaussian_bits(
                offsets_q[mask_bool],
                mean_offsets.reshape(-1)[mask_bool],
                scale_offsets.reshape(-1)[mask_bool].clamp_min(1e-9),
                Q_offsets.reshape(-1)[mask_bool],
            )
            .sum()
            .item()
        )

    # ---- masks / hash: binary Bernoulli estimates ----
    mask_flat = masks.reshape(-1).float()
    p1 = mask_flat.mean().clamp(1e-6, 1 - 1e-6)
    bits_masks = mask_flat.numel() * (
        -(p1 * torch.log2(p1) + (1 - p1) * torch.log2(1 - p1))
    ).item()

    from hacplus.utils.encodings import get_binary_vxl_size

    _, bit_hash, _, _ = get_binary_vxl_size((core.get_encoding_params() + 1) / 2)

    return {
        "num_anchors": int(N),
        "est_feat_bits": float(bits_feat),
        "est_scaling_bits": float(bits_scaling),
        "est_offsets_bits": float(bits_offsets),
        "est_masks_bits": float(bits_masks),
        "est_hash_bits": float(bit_hash),
        "est_attr_bits": float(bits_feat + bits_scaling + bits_offsets),
        "est_total_bits": float(
            bits_feat + bits_scaling + bits_offsets + bits_masks + bit_hash
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--work-dir", default="runs/codec_efficiency")
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    est = estimate_cross_entropy_bits(model)

    codec = HACPlusCodec()
    bitstream_dir = work_dir / "bitstreams"
    if bitstream_dir.exists():
        import shutil

        shutil.rmtree(bitstream_dir)
    meta = codec.encode(model, bitstream_dir)

    fields = {
        "feat": ("bit_feat", "est_feat_bits"),
        "scaling": ("bit_scaling", "est_scaling_bits"),
        "offsets": ("bit_offsets", "est_offsets_bits"),
        "masks": ("bit_masks", "est_masks_bits"),
        "hash": ("bit_hash", "est_hash_bits"),
    }
    rows = []
    total_actual = 0.0
    total_est = 0.0
    for name, (actual_key, est_key) in fields.items():
        actual = float(meta[actual_key])
        estimated = float(est[est_key])
        total_actual += actual
        total_est += estimated
        rows.append(
            {
                "field": name,
                "actual_bits": int(actual),
                "estimated_bits": round(estimated, 2),
                "efficiency": round(actual / max(estimated, 1.0), 4),
            }
        )
    rows.append(
        {
            "field": "total",
            "actual_bits": int(total_actual),
            "estimated_bits": round(total_est, 2),
            "efficiency": round(total_actual / max(total_est, 1.0), 4),
        }
    )

    summary = {
        "iteration": int(iteration),
        "estimated": est,
        "actual": {
            name: int(meta[actual_key]) for name, (actual_key, _) in fields.items()
        },
        "rows": rows,
    }
    out = work_dir / "codec_efficiency.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
