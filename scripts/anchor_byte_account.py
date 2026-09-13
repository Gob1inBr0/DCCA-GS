"""Per-anchor bitstream byte accounting by coverage area (方案 C/D 前置测量).

Answers the gating question of docs/02-design/背景场与ADMM覆盖约束_设计.md:
what share of the coded bytes do the large-coverage ("background") anchors
actually own?  The complexity multiplier already coarsens smooth background
quantization steps, so the naive "10-20% of anchors" prior can be far off —
this script measures the real number, which decides whether 方案 C (feature
codebook) / 方案 D (background field) are worth their complexity.

Per-anchor attribution (exact, deterministic — no 5% subsampling, no dither):
  feat      mixture likelihood, channel-autoregressive (same model the
            arithmetic coder uses; hard-round quantized values)
  scaling   diagonal Gaussian entropy model
  offsets   diagonal Gaussian entropy model, offset-masked
Global fields (xyz GPCC, hash grid, MLP weights, masks, header) cannot be
split per anchor and are reported as separate line items.

Usage (server):
  python scripts/anchor_byte_account.py \
    --ckpt runs/<run>/ckpts/ckpt_30000.pth \
    --area-npz analysis/anchor_stats/<run>/anchor_stats.npz \
    --out-dir analysis/anchor_stats/<run>/byte_account

Local testability: the aggregation half is pure numpy (see
``aggregate_account``); tests feed synthetic per-anchor tables plus the real
area dump without touching torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pure aggregation (no torch) — locally testable against the real area npz.
# ---------------------------------------------------------------------------

HEADLINE_QUANTILES = (0.05, 0.10, 0.20)  # top-x fraction of anchors by area


def _share_of_top(
    bits: np.ndarray,
    area: np.ndarray,
    frac: float,
) -> float:
    """Share of total bits owned by the top-``frac`` anchors by area."""
    n = area.shape[0]
    k = max(1, int(round(n * frac)))
    idx = np.argsort(area)[-k:]
    total = float(bits.sum())
    if total <= 0:
        return 0.0
    return float(bits[idx].sum()) / total


def aggregate_account(
    per_anchor_bits: Dict[str, np.ndarray],
    area: np.ndarray,
    seen: Optional[np.ndarray] = None,
    n_deciles: int = 10,
) -> Dict[str, Any]:
    """Decile table + headline shares, keyed by coverage area.

    per_anchor_bits: arrays of per-anchor bits (feat/scaling/offsets);
    global (non-splittable) fields go into ``global_bits`` instead.
    """
    area = np.asarray(area, dtype=np.float64)
    bits_sum = np.zeros_like(area)
    for v in per_anchor_bits.values():
        bits_sum = bits_sum + np.asarray(v, dtype=np.float64)
    valid = np.isfinite(area) & np.isfinite(bits_sum)
    if seen is not None:
        valid = valid & np.asarray(seen, dtype=bool)
    total_anchor_bits = float(bits_sum[valid].sum())
    total_all_bits = float(bits_sum.sum())

    order = np.argsort(area[valid])
    decile_rows = []
    n_v = int(valid.sum())
    edges = np.array_split(np.arange(n_v), n_deciles)
    sorted_area = area[valid][order]
    for di, idx in enumerate(edges, start=1):
        sel = np.zeros(n_v, dtype=bool)
        sel[idx] = True
        row_bits = float(bits_sum[valid][order][sel].sum())
        decile_rows.append(
            {
                "decile": f"D{di}",
                "n": int(sel.sum()),
                "area_median": float(np.median(sorted_area[idx])) if len(idx) else 0.0,
                "anchor_bits_share": row_bits / total_anchor_bits if total_anchor_bits else 0.0,
                "bits_per_anchor_median": float(
                    np.median(bits_sum[valid][order][sel])
                )
                if len(idx)
                else 0.0,
            }
        )

    headline = {
        f"top_{int(q * 100)}pct_area_bits_share": _share_of_top(
            bits_sum, np.where(valid, area, -np.inf), q
        )
        for q in HEADLINE_QUANTILES
    }
    return {
        "deciles": decile_rows,
        "headline": headline,
        "total_anchor_bits": total_anchor_bits,
        "total_all_bits_incl_unseen": total_all_bits,
        "n_valid": n_v,
        "n_total": int(area.shape[0]),
        "n_unseen_or_nan": int((~valid).sum()),
    }


def flags_share(
    area: np.ndarray,
    seen: Optional[np.ndarray],
    quantile: float,
    per_anchor_bits: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """What the 方案 C flag rule would capture: the bg set's population and
    bit share (the number that gates C/D investment)."""
    area = np.asarray(area, dtype=np.float64)
    valid = np.isfinite(area)
    if seen is not None:
        valid = valid & np.asarray(seen, dtype=bool)
    flags = np.zeros(area.shape[0], dtype=bool)
    if valid.any():
        cut = np.quantile(area[valid], quantile)
        flags[valid & (area >= cut)] = True
    bits_sum = np.zeros_like(area)
    for v in per_anchor_bits.values():
        bits_sum = bits_sum + np.asarray(v, dtype=np.float64)
    total = float(bits_sum[valid].sum())
    return {
        "quantile": quantile,
        "bg_population_share": float(flags[valid].sum()) / max(1, int(valid.sum())),
        "bg_bits_share": float(bits_sum[flags][valid[flags]].sum()) / total
        if total > 0
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Model-side per-anchor bits (server; mirrors the codec's entropy models).
# ---------------------------------------------------------------------------


def compute_per_anchor_bits(model, batch: int = 200_000) -> Dict[str, np.ndarray]:
    """Exact per-anchor bits for feat/scaling/offsets from a trained model.

    Mirrors ``HACPlusModel._estimate_rate_terms`` but (a) full population,
    (b) hard-round quantized values instead of uniform dither, (c) returns
    per-anchor rows instead of means. Global fields are NOT included.
    """
    import torch

    from scaffold_gs.model import get_model_class  # noqa: F401  (registration)

    core = model.core
    device = model.device
    cfg = model.cfg
    k = cfg.n_offsets
    cg = int(core.feat_channel_group)
    n_total = int(core.get_anchor.shape[0])

    mask_anchor = core.get_mask_anchor.detach()
    feat_all = model._view.anchor_feat.detach()
    offsets_all = model._view.offset.detach()
    scaling_all = core.get_scaling.detach()

    out_feat = np.zeros(n_total, dtype=np.float64)
    out_scaling = np.zeros(n_total, dtype=np.float64)
    out_offsets = np.zeros(n_total, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_total, batch):
            end = min(start + batch, n_total)
            anchor_slice = core.get_anchor[start:end]
            ctx = core.calc_context_feat(
                anchor_slice, caller="anchor_byte_account"
            )
            (
                mean, scale, prob, mean_scaling, scale_scaling,
                mean_offsets, scale_offsets, qa, qs, qo,
            ) = torch.split(
                core.get_grid_mlp(ctx),
                [cfg.feat_dim, cfg.feat_dim, cfg.feat_dim, 6, 6, 3 * k, 3 * k, 1, 1, 1],
                dim=-1,
            )
            qa = qa.repeat(1, cfg.feat_dim)
            qs = qs.repeat(1, 6)
            qo = qo.repeat(1, 3 * k)
            Q_feat = 1.0 * (1 + torch.tanh(qa))
            Q_scaling = 0.001 * (1 + torch.tanh(qs))
            Q_offsets = 0.2 * (1 + torch.tanh(qo))
            if core.is_content_aware_quant_active():
                (
                    Q_feat, Q_scaling, Q_offsets, _, _, _, _,
                ) = core._codec_apply_content_aware_quant_params(
                    "anchor_byte_account",
                    anchor_slice,
                    core.get_mask[start:end],
                    Q_feat, Q_scaling, Q_offsets, None, None, None,
                    mean_scaling, mean_offsets,
                )
            center_feat = feat_all.mean()
            feat_q = (
                torch.round((feat_all[start:end] - center_feat) / Q_feat) * Q_feat
                + center_feat
            )
            scaling_q = (
                torch.round(
                    (scaling_all[start:end] - core.get_scaling.mean()) / Q_scaling
                )
                * Q_scaling
                + core.get_scaling.mean()
            )
            offsets_q = (
                torch.round((offsets_all[start:end] - offsets_all.mean()) / Q_offsets)
                * Q_offsets
                + offsets_all.mean()
            )
            mask_flat = (
                core.get_mask[start:end]
                .repeat(1, 1, 3)
                .view(-1, 3 * k)
            )
            offsets_q = offsets_q * mask_flat

            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale_c = scale.clamp(min=1e-9)
            feat_bits = torch.zeros(end - start, cfg.feat_dim, device=device)
            for cc in range(cfg.feat_dim // cg):
                mean_adj, scale_adj, prob_adj = core.get_deform_mlp.forward(
                    feat_q, mean_scale, to_dec=cc
                )
                probs = torch.softmax(
                    torch.stack(
                        [prob[:, cc * cg : cc * cg + cg], prob_adj], dim=-1
                    ),
                    dim=-1,
                )
                feat_bits[:, cc * cg : cc * cg + cg] = core.EG_mix_prob_2.forward(
                    feat_q[:, cc * cg : cc * cg + cg],
                    mean[:, cc * cg : cc * cg + cg],
                    mean_adj,
                    scale_c[:, cc * cg : cc * cg + cg],
                    scale_adj,
                    probs[..., 0],
                    probs[..., 1],
                    Q=Q_feat[:, cc * cg : cc * cg + cg],
                    x_mean=center_feat,
                )
            scaling_bits = core.entropy_gaussian.forward(
                scaling_q,
                mean_scaling,
                scale_scaling.clamp(min=1e-9),
                Q_scaling,
                core.get_scaling.mean(),
            )
            offsets_bits = core.entropy_gaussian.forward(
                offsets_q,
                mean_offsets,
                scale_offsets.clamp(min=1e-9),
                Q_offsets,
                offsets_all.mean(),
            )
            out_feat[start:end] = feat_bits.sum(dim=-1).double().cpu().numpy()
            out_scaling[start:end] = scaling_bits.sum(dim=-1).double().cpu().numpy()
            out_offsets[start:end] = (
                (offsets_bits * mask_flat).sum(dim=-1).double().cpu().numpy()
            )
    return {"feat": out_feat, "scaling": out_scaling, "offsets": out_offsets}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--area-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--quantile", type=float, default=0.9)
    args = ap.parse_args()

    import torch  # noqa: F401  (env check: must import before the model)

    from scaffold_gs.trainer import load_checkpoint

    model, _optim, iteration, _meta = load_checkpoint(args.ckpt, args.device)
    print(f"[byte-account] loaded ckpt at iteration {iteration}")

    per_anchor = compute_per_anchor_bits(model, batch=args.batch)
    data = np.load(args.area_npz)
    area = data["area"]
    seen = data["seen"] if "seen" in data else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "per_anchor_bits.npz",
        **per_anchor,
        area=area,
        seen=seen if seen is not None else np.ones_like(area, dtype=bool),
    )
    account = aggregate_account(per_anchor, area, seen)
    account["bg_flags"] = flags_share(area, seen, args.quantile, per_anchor)
    account["global_fields_note"] = (
        "xyz/hash/mlp/masks/header are not per-anchor attributable; see"
        " hac_meta.json for their totals"
    )
    with open(out_dir / "byte_account.json", "w") as f:
        json.dump(account, f, indent=2)
    print(json.dumps(account["headline"], indent=2))
    print(json.dumps(account["bg_flags"], indent=2))
    for row in account["deciles"]:
        print(
            f"{row['decile']}: n={row['n']} area~{row['area_median']:.2e} "
            f"bits_share={row['anchor_bits_share']:.3f} "
            f"bits/anchor={row['bits_per_anchor_median']:.0f}"
        )


if __name__ == "__main__":
    main()
