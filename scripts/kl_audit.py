"""KL redundancy audit: empirical entropy vs model cross-entropy per field.

Reads the codec_efficiency.json produced by scripts/codec_efficiency.py and
computes, per field:
  - number of symbols
  - model cross-entropy bits/symbol (from the codec-efficiency estimate)
  - empirical entropy H(q) bits/symbol (histogram of quantized symbols)
  - KL = model_CE - H(q) bits/symbol
  - actual bits/symbol (from the real bitstream)

Usage (5090):
    PYTHONPATH=$PWD python scripts/kl_audit.py \
      --ckpt <ckpt_90000.pth> \
      --efficiency-json runs/codec_efficiency/codec_efficiency.json \
      --work-dir runs/kl_audit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scaffold_gs.hacpp import HACPlusCodec, anchor_codec_order
from scaffold_gs.trainer import load_checkpoint


def _emp_entropy(t: torch.Tensor) -> float:
    t = t.reshape(-1).float()
    vals, counts = torch.unique(t, return_counts=True)
    p = counts.float() / t.numel()
    return float(-(p * torch.log2(p)).sum().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--efficiency-json", default="runs/codec_efficiency/codec_efficiency.json")
    p.add_argument("--work-dir", default="runs/kl_audit")
    args = p.parse_args()

    model, _, _, _ = load_checkpoint(args.ckpt, "cuda")
    core = model.core
    k = model.cfg.n_offsets
    from hacplus.scene.gaussian_model import MAX_batch_size

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]

    scaling_sym_parts = []
    offsets_sym_parts = []
    mask_parts = []
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="kl_audit")
        out = core.get_grid_mlp(ctx)
        (
            _mean,
            _scale,
            _prob,
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
        Q_feat = 1.0 * (1 + torch.tanh(qa.repeat(1, model.cfg.feat_dim)))
        Q_scaling = 0.001 * (1 + torch.tanh(qs.repeat(1, 6)))
        Q_offsets = 0.2 * (1 + torch.tanh(qo.repeat(1, 3 * k)))
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
                "kl_audit",
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

        scaling_slice = scaling[start:end].reshape(-1)
        Qs = Q_scaling.reshape(-1)
        scaling_sym_parts.append(torch.round(scaling_slice / Qs))

        mask_bool = (
            masks[start:end]
            .repeat(1, 1, 3)
            .view(-1, 3 * k)
            .reshape(-1)
            .bool()
        )
        offsets_slice = offsets[start:end].reshape(-1, 3 * k).reshape(-1)
        Qo = Q_offsets.reshape(-1)
        offsets_sym = torch.round(offsets_slice / Qo)
        offsets_sym[~mask_bool] = 0
        offsets_sym_parts.append(offsets_sym[mask_bool])
        mask_parts.append(masks[start:end].reshape(-1).float())

    scaling_sym = torch.cat(scaling_sym_parts)
    offsets_sym = torch.cat(offsets_sym_parts)
    mask_flat = torch.cat(mask_parts)

    est = json.load(open(args.efficiency_json))
    est_bits = est["estimated"]
    actual = est["actual"]
    rows = []
    for name, symbols, est_key, actual_key in [
        ("scaling", scaling_sym, "est_scaling_bits", "scaling"),
        ("offsets", offsets_sym, "est_offsets_bits", "offsets"),
        ("masks", mask_flat, "est_masks_bits", "masks"),
    ]:
        num = max(int(symbols.numel()), 1)
        h_emp = _emp_entropy(symbols)
        ce = float(est_bits[est_key]) / num
        kl = ce - h_emp
        rows.append(
            {
                "field": name,
                "num_symbols": num,
                "emp_H_bits": round(h_emp, 4),
                "model_CE_bits": round(ce, 4),
                "KL_bits_per_symbol": round(kl, 4),
                "actual_bits_per_symbol": round(float(actual[actual_key]) / num, 4),
            }
        )

    summary = {"num_anchors": int(N), "rows": rows}
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "kl_audit.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
