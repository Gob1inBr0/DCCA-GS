"""Direction-4 offline gate: scale-level independent entropy model (Octree-GS style).

On a decoded bitstream (no retraining), splits anchors into ``--levels``
groups by anchor scale (quantiles of exp(scaling[:, :3]).mean), then measures
whether a per-level correction of the existing Gaussian entropy parameters
(mean bias + scale multiplier per level per dimension) reduces the
scaling+offsets cross-entropy by >= 3% over the global model.

This mirrors P0's offline entropy methodology (Gaussian CDF bins, val split =
last 20% of codec/Morton order). Stop condition (design doc 6.4): hierarchical
entropy gain >= 3% before investing in real per-level coding.

Usage (5090, HAC_5090_a100 env):

    python scripts/octree_level_entropy.py \
        --artifact-dir runs/mlp_quant_sens_cd8_rest16_110k/b8/bitstreams \
        --levels 3 --out runs/octree_level_entropy/results.json \
        --legacy-complexity-8dim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scaffold_gs.hacpp import HACPlusCodec


def patch_legacy_8dim_complexity() -> None:
    from scaffold_gs import hac_core

    orig = hac_core.HACCoreView.load_decoder_state

    def patched(self, state):
        cs = state.get("mlp_complexity")
        if cs is not None and cs.get("0.weight") is not None:
            w = cs["0.weight"]
            if w.shape[-1] == 8:
                cs = dict(cs)
                cs["0.weight"] = w[:, :4].contiguous()
                state = dict(state)
                state["mlp_complexity"] = cs
        return orig(self, state)

    hac_core.HACCoreView.load_decoder_state = patched
    import hacplus.utils.codec_consistency as cc

    cc.FORMULA_INPUT_VERSION = "formula_decoder_available_v1"


def _gaussian_bits(x, mean, scale, q):
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-6))
    p = dist.cdf(x + 0.5 * q) - dist.cdf(x - 0.5 * q)
    return -torch.log2(p.clamp_min(1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--out", default="runs/octree_level_entropy/results.json")
    p.add_argument("--levels", type=int, default=3, choices=[2, 3])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--legacy-complexity-8dim",
        action="store_true",
        help="Decode pre-4D bitstreams (8-dim complexity input, v1 header).",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    artifact_dir = Path(args.artifact_dir)
    if args.legacy_complexity_8dim:
        patch_legacy_8dim_complexity()

    codec = HACPlusCodec()
    model = codec.decode(artifact_dir)
    core = model.core
    k_offsets = model.cfg.n_offsets
    from hacplus.scene.gaussian_model import MAX_batch_size

    anchor = core.get_anchor.detach()
    scaling = core.get_scaling.detach()
    offsets = model._view.offset.detach()
    masks = core.get_mask.detach()
    N = anchor.shape[0]
    print(f"[octree_level] {N} anchors, levels={args.levels}", flush=True)

    parts = {name: [] for name in
             ("xs", "xo", "q_s", "q_o", "mean_s", "scale_s",
              "mean_o", "scale_o", "mask", "anchor_scale")}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="octree_level_entropy")
        out = core.get_grid_mlp(ctx)
        (
            mean, scale, prob, mean_scaling, scale_scaling,
            mean_offsets, scale_offsets, qa, qs, qo,
        ) = torch.split(
            out,
            [
                model.cfg.feat_dim, model.cfg.feat_dim, model.cfg.feat_dim,
                6, 6, 3 * k_offsets, 3 * k_offsets, 1, 1, 1,
            ],
            dim=-1,
        )
        qa = qa.repeat(1, model.cfg.feat_dim)
        qs = qs.repeat(1, 6)
        qo = qo.repeat(1, 3 * k_offsets)
        Q_scaling = 0.001 * (1 + torch.tanh(qs))
        Q_offsets = 0.2 * (1 + torch.tanh(qo))
        if core.is_content_aware_quant_active():
            masks_slice = masks[start:end]
            (
                _, Q_scaling, Q_offsets, _, _, _, _,
            ) = core._codec_apply_content_aware_quant_params(
                "octree_level_entropy", anchor_slice, masks_slice,
                Q_scaling, Q_scaling, Q_offsets, None, None, None,
                mean_scaling.view(-1, 6), mean_offsets.view(-1, 3 * k_offsets),
            )
        xs = torch.round(scaling[start:end] / Q_scaling) * Q_scaling
        xo = (
            torch.round(
                offsets[start:end].reshape(-1, 3 * k_offsets) / Q_offsets
            )
            * Q_offsets
        )
        mask = masks[start:end].reshape(-1, k_offsets, 1).repeat(1, 1, 3).reshape(
            -1, 3 * k_offsets
        )
        parts["xs"].append(xs)
        parts["xo"].append(xo)
        parts["q_s"].append(Q_scaling)
        parts["q_o"].append(Q_offsets)
        parts["mean_s"].append(mean_scaling)
        parts["scale_s"].append(scale_scaling.clamp(min=1e-9))
        parts["mean_o"].append(mean_offsets)
        parts["scale_o"].append(scale_offsets.clamp(min=1e-9))
        parts["mask"].append(mask)
        parts["anchor_scale"].append(
            torch.exp(scaling[start:end, :3]).mean(dim=1)
        )

    g = {
        name: torch.cat(v, dim=0).contiguous().detach()
        for name, v in parts.items()
    }
    mask_bool = g["mask"].bool()
    n_levels = args.levels

    # Level assignment by anchor-scale quantiles (equal-count bins).
    ascale = g["anchor_scale"]
    edges = [ascale.quantile(torch.tensor(i / n_levels, device=device))
             for i in range(1, n_levels)]
    level_id = torch.zeros(N, dtype=torch.long, device=device)
    for e in edges:
        level_id += (ascale > e).long()

    n_train = int(N * (1.0 - args.val_fraction))
    train_mask = torch.arange(N, device=device) < n_train
    val_mask = ~train_mask
    print(
        "[octree_level] per-level counts (train/val): "
        + ", ".join(
            f"L{l}: {int((level_id[train_mask] == l).sum())}/"
            f"{int((level_id[val_mask] == l).sum())}"
            for l in range(n_levels)
        )
    )

    with torch.no_grad():
        # ---- H_global (baseline, current mlp_grid model) ----
        base_s = _gaussian_bits(g["xs"], g["mean_s"], g["scale_s"], g["q_s"])
        base_o = _gaussian_bits(g["xo"], g["mean_o"], g["scale_o"], g["q_o"])
        base_s_val = base_s[val_mask].sum().item()
        base_o_val = base_o[val_mask][mask_bool[val_mask]].sum().item()
        base_all_s = base_s.sum().item()
        base_all_o = base_o[mask_bool].sum().item()

        # ---- Per-level residual fit (train) + eval (val) ----
        res_s = (g["xs"] - g["mean_s"]) / g["scale_s"]
        res_o = (g["xo"] - g["mean_o"]) / g["scale_o"]
        ds, do = 6, 3 * k_offsets
        mu_s = torch.zeros(n_levels, ds, device=device)
        sigma_s = torch.ones(n_levels, ds, device=device)
        mu_o = torch.zeros(n_levels, do, device=device)
        sigma_o = torch.ones(n_levels, do, device=device)
        for l in range(n_levels):
            tr = train_mask & (level_id == l)
            mu_s[l] = res_s[tr].mean(dim=0)
            sigma_s[l] = res_s[tr].std(dim=0, unbiased=False).clamp_min(1e-3)
            otr = tr.unsqueeze(-1).expand_as(g["xo"]) & mask_bool
            cnt = otr.sum(dim=0).clamp_min(1)
            mu_l = torch.where(otr, res_o, torch.zeros_like(res_o)).sum(dim=0) / cnt
            var_l = torch.where(
                otr, (res_o - mu_l) ** 2, torch.zeros_like(res_o)
            ).sum(dim=0) / cnt
            mu_o[l] = mu_l
            sigma_o[l] = var_l.clamp_min(1e-6).sqrt().clamp_min(1e-3)

        lev = level_id[val_mask]
        mean_s_adj = g["mean_s"][val_mask] + mu_s[lev] * g["scale_s"][val_mask]
        scale_s_adj = g["scale_s"][val_mask] * sigma_s[lev]
        mean_o_adj = g["mean_o"][val_mask] + mu_o[lev] * g["scale_o"][val_mask]
        scale_o_adj = g["scale_o"][val_mask] * sigma_o[lev]
        lvl_s = _gaussian_bits(
            g["xs"][val_mask], mean_s_adj, scale_s_adj, g["q_s"][val_mask]
        ).sum().item()
        lvl_o = _gaussian_bits(
            g["xo"][val_mask], mean_o_adj, scale_o_adj, g["q_o"][val_mask]
        )[mask_bool[val_mask]].sum().item()

    bit2mb = 8 * 1024 * 1024
    gain_s = 100.0 * (base_s_val - lvl_s) / max(base_s_val, 1e-9)
    gain_o = 100.0 * (base_o_val - lvl_o) / max(base_o_val, 1e-9)
    total_base = base_s_val + base_o_val
    total_lvl = lvl_s + lvl_o
    gain_total = 100.0 * (total_base - total_lvl) / max(total_base, 1e-9)

    results = {
        "bitstream": str(artifact_dir),
        "n_anchors": N,
        "n_levels": n_levels,
        "level_definition": "quantile(exp(scaling[:, :3]).mean)",
        "per_level_counts_train_val": [
            [int((level_id[train_mask] == l).sum()),
             int((level_id[val_mask] == l).sum())]
            for l in range(n_levels)
        ],
        "H_base_scaling_val_MB": round(base_s_val / bit2mb, 6),
        "H_level_scaling_val_MB": round(lvl_s / bit2mb, 6),
        "H_base_offsets_val_MB": round(base_o_val / bit2mb, 6),
        "H_level_offsets_val_MB": round(lvl_o / bit2mb, 6),
        "H_base_total_val_MB": round(total_base / bit2mb, 6),
        "H_level_total_val_MB": round(total_lvl / bit2mb, 6),
        "gain_scaling_pct": round(gain_s, 4),
        "gain_offsets_pct": round(gain_o, 4),
        "gain_total_pct": round(gain_total, 4),
        "decision": "pass" if gain_total >= 3.0 else "close",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print(
        f"[octree_level] total gain {gain_total:.2f}% -> "
        f"{results['decision']}"
    )


if __name__ == "__main__":
    main()
