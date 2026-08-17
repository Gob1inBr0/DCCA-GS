"""Stage-A offline feasibility for CompGS-style "anchor prediction + residual
coding" (innovation R) on the PHG codec.

Design doc: docs/CompGS残差编码实验设计.md (section 3).

Variants (all evaluated on the Morton-last 20% validation split):

    H_base      current codec bits:
                feat = mixture-of-Gaussians + Channel_CTX_fea autoregression;
                scaling/offsets = mlp_grid Gaussian bins; Q follows the I2
                formula path (mlp_complexity multipliers).
    R0          residual r = x_q - mlp_grid_mean, coded with zero-mean
                Laplace(0, b) or Gaussian(0, sigma); b/sigma MLE per field.
    R1          small zero-initialized MLP predictor on calc_context_feat(anchor);
                residual coded with Laplace(0, b).
    R2          cross-field predictor (closest to CompGS):
                pred_scaling = MLP_s(feat_q, ctx)
                pred_offsets = MLP_o(feat_q, scaling_q, mask, ctx)
                residual coded with Laplace(0, b) / Gaussian(0, sigma).
    R3          feat channel-delta coding: each 10-channel group coded against
                the previous decoded group (Laplace per group).
    R4          control: the SAME predictor as R2, but the prediction is added
                to the mlp_grid Gaussian mean (no residual subtraction).

Stop conditions (from the design doc):
    1. R2 field gain < 3% or net total_MB gain < 1%  -> close
    2. |R2_gauss - R4| < 0.1% of H_base              -> close
    3. R0 has no gain and R1/R2 have no gain         -> close
    4. predictor weights (16-bit) eat all savings    -> close

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/residual_feasibility.py \
      --ckpt runs/4-28_i6_90k_h32_l0p002/ckpts/ckpt_90000.pth \
      --out runs/residual_feasibility.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffold_gs.hacpp import HACPlusModel, anchor_codec_order
from scaffold_gs.trainer import load_checkpoint


# ---------------------------------------------------------------------------
# Entropy helpers (same probability paths as the real arithmetic coder)
# ---------------------------------------------------------------------------


def _gaussian_bin(x, mean, scale, q):
    """P(symbol) with a Gaussian bin of width q centered at x (x = s*q)."""
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-6))
    return (dist.cdf(x + 0.5 * q) - dist.cdf(x - 0.5 * q)).clamp_min(1e-12)


def _laplace_cdf(x, mean, b):
    b = b.clamp_min(1e-9)
    return torch.where(
        x <= mean,
        0.5 * torch.exp((x - mean) / b),
        1.0 - 0.5 * torch.exp(-(x - mean) / b),
    )


def _laplace_bin(x, mean, b, q):
    p = _laplace_cdf(x + 0.5 * q, mean, b) - _laplace_cdf(x - 0.5 * q, mean, b)
    return p.clamp_min(1e-12)


def _feature_mixed_bits(feat_q, mean, scale, prob, q, deform, feat_dim, chunk):
    """Feat bits under the real codec's mixed-Gaussian channel autoregression.

    Mirrors ``encode_attributes``: per ``chunk``-channel group, the deform MLP
    predicts adjusted mixture parameters from the partial decoded feat, the
    mixture is softmax-blended with the base Gaussian, and the coded channel
    is fed back as ``feat_q`` (STE simulation of the decoder).
    """
    n = feat_q.shape[0]
    feat_decoded = torch.zeros_like(feat_q)
    mean_scale = torch.cat([mean, scale, prob], dim=-1)
    total = torch.zeros((), device=feat_q.device)
    for cc in range(feat_dim // chunk):
        sl = slice(cc * chunk, (cc + 1) * chunk)
        mean_adj, scale_adj, prob_adj = deform.forward(
            feat_decoded, mean_scale, to_dec=cc
        )
        probs = F.softmax(
            torch.stack([prob[:, sl], prob_adj], dim=-1), dim=-1
        )
        p0 = _gaussian_bin(feat_q[:, sl], mean[:, sl], scale[:, sl], q[:, sl])
        p1 = _gaussian_bin(
            feat_q[:, sl], mean_adj, scale_adj.clamp_min(1e-9), q[:, sl]
        )
        p = probs[..., 0] * p0 + probs[..., 1] * p1
        total = total + (-torch.log2(p)).sum()
        feat_decoded[:, sl] = feat_q[:, sl]
    return total


def _residual_bits(y, pred, q, mask, param, dist):
    """Bits of the quantized residual ``round((y - pred)/q)*q``.

    ``param`` is the Laplace scale b or the Gaussian sigma, depending on
    ``dist``. Masked entries (offsets) contribute only where ``mask`` is True.
    """
    r = torch.round((y - pred) / q) * q
    if mask is not None:
        r = r[mask]
        q = q[mask]
    if dist == "laplace":
        p = _laplace_bin(r, torch.zeros_like(r), param, q)
    else:
        p = _gaussian_bin(r, torch.zeros_like(r), param, q)
    return (-torch.log2(p)).sum()


class _ResMLP(nn.Module):
    """One-hidden-layer predictor, last layer zero-initialized (identity start)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AdjMLP(nn.Module):
    """P0-1-style conditional mean/log-scale adjustment (zero-init start)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_predictor(
    x_tr,
    y_tr,
    q_tr,
    mask_tr,
    x_va,
    y_va,
    q_va,
    mask_va,
    out_dim: int,
    hidden: int,
    steps: int,
    weight_decay: float,
    dist: str,
    device: torch.device,
):
    """Fit a residual predictor on the train split, early-stop on val bits.

    Loss is the smooth L1 (Laplace) / L2 (Gaussian) fit; the entropy scale is
    re-estimated from train residuals at each checkpoint and at the end.
    Returns (val_bits, param, n_params, state_dict) with the best state loaded.
    """
    mlp = _ResMLP(x_tr.shape[-1], hidden, out_dim).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=weight_decay)
    n_tr = x_tr.shape[0]

    def _eval_bits() -> float:
        with torch.no_grad():
            pred_va = mlp(x_va)
            pred_tr = mlp(x_tr[:8192])
            r_tr = y_tr[:8192] - pred_tr
            if mask_tr is not None:
                r_tr = r_tr[mask_tr[:8192]]
            if dist == "laplace":
                param = r_tr.abs().mean().clamp_min(1e-9)
            else:
                param = r_tr.square().mean().sqrt().clamp_min(1e-9)
            bits = _residual_bits(
                y_va, pred_va, q_va, mask_va, param, dist
            )
        return float(bits.item()), param.item()

    with torch.no_grad():
        best_val, _ = _eval_bits()
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    for it in range(steps):
        idx = torch.randperm(n_tr, device=device)[: min(n_tr, 8192)]
        pred = mlp(x_tr[idx])
        r = y_tr[idx] - pred
        if mask_tr is not None:
            r = r[mask_tr[idx]]
        if dist == "laplace":
            loss = r.abs().mean()
        else:
            loss = r.square().mean()
        loss = loss + 0.01 * pred.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            val, _ = _eval_bits()
            if val < best_val:
                best_val = val
                best_state = {
                    k: v.clone() for k, v in mlp.state_dict().items()
                }

    mlp.load_state_dict(best_state)
    final_bits, final_param = _eval_bits()
    n_params = sum(p.numel() for p in mlp.parameters())
    return final_bits, final_param, n_params, best_state


def _fit_condmean(
    x_tr,
    base_mean_tr,
    base_scale_tr,
    y_tr,
    q_tr,
    mask_tr,
    x_va,
    base_mean_va,
    base_scale_va,
    y_va,
    q_va,
    mask_va,
    out_dim: int,
    hidden: int,
    steps: int,
    weight_decay: float,
    device: torch.device,
):
    """Fit a P0-1-style conditional mean/log-scale adjustment (R4 control).

    ``mean' = base_mean + adj_mean``, ``scale' = base_scale * exp(adj_log_scale)``,
    and the RAW value is coded with Gaussian(mean', scale'). Last layer is
    zero-initialized so the start reproduces H_base exactly.
    """
    mlp = _AdjMLP(x_tr.shape[-1], hidden, out_dim).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=weight_decay)
    n_tr = x_tr.shape[0]

    def _bits(x, base_mean, base_scale, y, q, mask):
        adj = mlp(x)
        mean = base_mean + adj[:, :out_dim]
        scale = base_scale * torch.exp(
            adj[:, out_dim : 2 * out_dim].clamp(-1.4, 1.4)
        )
        if mask is not None:
            mean = mean[mask]
            scale = scale[mask]
            y = y[mask]
            q = q[mask]
        p = _gaussian_bin(y, mean, scale, q)
        bits = (-torch.log2(p)).sum()
        return bits, adj

    with torch.no_grad():
        bits0, _ = _bits(x_va, base_mean_va, base_scale_va, y_va, q_va, mask_va)
        best_val = float(bits0.item())
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    for it in range(steps):
        idx = torch.randperm(n_tr, device=device)[: min(n_tr, 8192)]
        bits, adj = _bits(
            x_tr[idx], base_mean_tr[idx], base_scale_tr[idx],
            y_tr[idx], q_tr[idx], mask_tr[idx] if mask_tr is not None else None,
        )
        loss = bits / max(float(y_tr[idx].numel()), 1.0) + 0.1 * adj.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                val = float(
                    _bits(
                        x_va, base_mean_va, base_scale_va,
                        y_va, q_va, mask_va,
                    )[0].item()
                )
            if val < best_val:
                best_val = val
                best_state = {
                    k: v.clone() for k, v in mlp.state_dict().items()
                }

    mlp.load_state_dict(best_state)
    with torch.no_grad():
        bits, _ = _bits(x_va, base_mean_va, base_scale_va, y_va, q_va, mask_va)
    n_params = sum(p.numel() for p in mlp.parameters())
    return float(bits.item()), n_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _mb(bits: float) -> float:
    return bits / 8.0 / 1024.0 / 1024.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/residual_feasibility.json")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps-list", default="400,1500")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-mb", type=float, default=0.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    assert isinstance(model, HACPlusModel)
    core = model.core
    k_offsets = model.cfg.n_offsets
    feat_dim = model.cfg.feat_dim
    chunk = int(core.feat_channel_group)
    from hacplus.scene.gaussian_model import MAX_batch_size

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]
    print(f"[R] {N} anchors, iteration={iteration}, feat_dim={feat_dim}", flush=True)

    parts = {name: [] for name in (
        "anchor", "ctx", "mean", "scale", "prob", "q",
        "feat_q", "xs", "xo", "q_s", "q_o",
        "mean_s", "scale_s", "mean_o", "scale_o", "mask", "mask_flat",
    )}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="residual_feasibility")
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
                "residual_feasibility", anchor_slice, masks_slice,
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
        parts["anchor"].append(anchor_slice)
        parts["ctx"].append(ctx)
        parts["mean"].append(mean)
        parts["scale"].append(scale.clamp(min=1e-9))
        parts["prob"].append(prob)
        parts["q"].append(Q_feat)
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
        parts["mask_flat"].append(msk_flat)

    g = {
        name: torch.cat(v, dim=0).contiguous().detach()
        for name, v in parts.items()
    }
    deform = core.get_deform_mlp
    n_tr = int(N * (1.0 - args.val_fraction))
    tr = slice(0, n_tr)
    va = slice(n_tr, N)

    # --- H_base (validation split) ---------------------------------------
    with torch.no_grad():
        H_feat_base = float(_feature_mixed_bits(
            g["feat_q"][va], g["mean"][va], g["scale"][va], g["prob"][va],
            g["q"][va], deform, feat_dim, chunk,
        ).item())
        H_s_base = float(
            (-torch.log2(
                _gaussian_bin(g["xs"][va], g["mean_s"][va], g["scale_s"][va], g["q_s"][va])
            )).sum().item()
        )
        H_o_base = float(
            (-torch.log2(
                _gaussian_bin(g["xo"][va], g["mean_o"][va], g["scale_o"][va], g["q_o"][va])
            ))[g["mask_flat"][va]].sum().item()
        )
    H_base = H_feat_base + H_s_base + H_o_base
    print(
        f"H_base: feat={_mb(H_feat_base):.4f}MB scaling={_mb(H_s_base):.4f}MB "
        f"offsets={_mb(H_o_base):.4f}MB total={_mb(H_base):.4f}MB",
        flush=True,
    )

    ref_total_mb = args.total_mb
    if ref_total_mb <= 0.0:
        meta = Path(args.ckpt).resolve().parents[1] / "bitstreams" / "hac_meta.json"
        if meta.is_file():
            try:
                ref_total_mb = float(json.loads(meta.read_text()).get("total_MB", 0.0))
            except Exception:
                ref_total_mb = 0.0

    rows = {}

    def add_variant(name, feat_bits, scaling_bits, offsets_bits, n_params=0):
        total = feat_bits + scaling_bits + offsets_bits
        gain = (H_base - total) / H_base
        so_base = H_s_base + H_o_base
        so_bits = scaling_bits + offsets_bits
        gain_so = (so_base - so_bits) / so_base if so_base > 0 else 0.0
        save_mb = _mb(H_base - total)
        pred_mb = n_params * 2.0 / 1024.0 / 1024.0  # 16-bit flat payload
        net_mb = save_mb - pred_mb
        net_field_pct = net_mb / _mb(H_base) * 100.0 if _mb(H_base) > 0 else 0.0
        net_total_pct = net_mb / ref_total_mb * 100.0 if ref_total_mb > 0 else None
        rows[name] = {
            "feat_MB": round(_mb(feat_bits), 4),
            "scaling_MB": round(_mb(scaling_bits), 4),
            "offsets_MB": round(_mb(offsets_bits), 4),
            "total_MB": round(_mb(total), 4),
            "gain_pct": round(float(gain) * 100.0, 3),
            "gain_so_pct": round(float(gain_so) * 100.0, 3),
            "save_MB": round(save_mb, 5),
            "predictor_MB_16bit": round(pred_mb, 5),
            "net_save_MB": round(net_mb, 5),
            "net_field_pct": round(net_field_pct, 3),
            "net_total_pct": (
                round(net_total_pct, 3) if net_total_pct is not None else None
            ),
        }
        print(
            f"{name}: total={_mb(total):.4f}MB gain={gain * 100:.3f}% "
            f"net={net_mb:.5f}MB (pred {pred_mb:.5f}MB)",
            flush=True,
        )

    # --- R0: zero-parameter residual vs mlp_grid mean ----------------------
    with torch.no_grad():
        r_s_tr = g["xs"][tr] - g["mean_s"][tr]
        r_o_tr = g["xo"][tr] - g["mean_o"][tr]
        b_s = r_s_tr.abs().mean().clamp_min(1e-9)
        sigma_s = r_s_tr.square().mean().sqrt().clamp_min(1e-9)
        b_o = r_o_tr[g["mask_flat"][tr]].abs().mean().clamp_min(1e-9)
        sigma_o = r_o_tr[g["mask_flat"][tr]].square().mean().sqrt().clamp_min(1e-9)
        r_s_va = g["xs"][va] - g["mean_s"][va]
        r_o_va = g["xo"][va] - g["mean_o"][va]
        H_s_r0_lap = float(_residual_bits(
            g["xs"][va], g["mean_s"][va], g["q_s"][va], None, b_s, "laplace"
        ).item())
        H_o_r0_lap = float(_residual_bits(
            g["xo"][va], g["mean_o"][va], g["q_o"][va], g["mask_flat"][va],
            b_o, "laplace",
        ).item())
        H_s_r0_gau = float(_residual_bits(
            g["xs"][va], g["mean_s"][va], g["q_s"][va], None, sigma_s, "gaussian"
        ).item())
        H_o_r0_gau = float(_residual_bits(
            g["xo"][va], g["mean_o"][va], g["q_o"][va], g["mask_flat"][va],
            sigma_o, "gaussian",
        ).item())
    add_variant("R0_laplace", H_feat_base, H_s_r0_lap, H_o_r0_lap)
    add_variant("R0_gaussian", H_feat_base, H_s_r0_gau, H_o_r0_gau)

    # --- R1: small MLP predictor on hash context ---------------------------
    x_ctx = g["ctx"]
    for steps in [int(v) for v in args.steps_list.split(",")]:
        bits_s, _, nps, _ = _fit_predictor(
            x_ctx[tr], g["xs"][tr], g["q_s"][tr], None,
            x_ctx[va], g["xs"][va], g["q_s"][va], None,
            6, args.hidden, steps, args.weight_decay, "laplace", device,
        )
        bits_o, _, npo, _ = _fit_predictor(
            x_ctx[tr], g["xo"][tr], g["q_o"][tr], g["mask_flat"][tr],
            x_ctx[va], g["xo"][va], g["q_o"][va], g["mask_flat"][va],
            3 * k_offsets, args.hidden, steps, args.weight_decay, "laplace", device,
        )
        add_variant(f"R1_laplace_h{args.hidden}_s{steps}", H_feat_base, bits_s, bits_o, nps + npo)

    # --- R2: cross-field coupled prediction --------------------------------
    x_s_in = torch.cat([g["feat_q"], g["ctx"]], dim=-1)
    x_o_in = torch.cat(
        [g["feat_q"], g["xs"], g["mask"], g["mask_flat"], g["ctx"]], dim=-1
    )
    for steps in [int(v) for v in args.steps_list.split(",")]:
        bits_s, _, nps, _ = _fit_predictor(
            x_s_in[tr], g["xs"][tr], g["q_s"][tr], None,
            x_s_in[va], g["xs"][va], g["q_s"][va], None,
            6, args.hidden, steps, args.weight_decay, "laplace", device,
        )
        bits_o, _, npo, _ = _fit_predictor(
            x_o_in[tr], g["xo"][tr], g["q_o"][tr], g["mask_flat"][tr],
            x_o_in[va], g["xo"][va], g["q_o"][va], g["mask_flat"][va],
            3 * k_offsets, args.hidden, steps, args.weight_decay, "laplace", device,
        )
        add_variant(
            f"R2_laplace_h{args.hidden}_s{steps}", H_feat_base, bits_s, bits_o, nps + npo
        )

    # R2_gauss: cross-field predictor, residual coded with Gaussian(0, sigma).
    for steps in [int(v) for v in args.steps_list.split(",")]:
        bits_s_g, _, nps, _ = _fit_predictor(
            x_s_in[tr], g["xs"][tr], g["q_s"][tr], None,
            x_s_in[va], g["xs"][va], g["q_s"][va], None,
            6, args.hidden, steps, args.weight_decay, "gaussian", device,
        )
        bits_o_g, _, npo, _ = _fit_predictor(
            x_o_in[tr], g["xo"][tr], g["q_o"][tr], g["mask_flat"][tr],
            x_o_in[va], g["xo"][va], g["q_o"][va], g["mask_flat"][va],
            3 * k_offsets, args.hidden, steps, args.weight_decay, "gaussian", device,
        )
        add_variant(
            f"R2_gaussian_h{args.hidden}_s{steps}",
            H_feat_base, bits_s_g, bits_o_g, nps + npo,
        )

    # R4 control (P0-1 style): same cross-field inputs, but the prediction is
    # applied as a Gaussian mean/log-scale adjustment and the RAW value is
    # coded (no residual subtraction).
    x_s4 = torch.cat(
        [g["mean_s"], g["scale_s"], g["feat_q"], g["ctx"]], dim=-1
    )
    x_o4 = torch.cat(
        [
            g["mean_o"], g["scale_o"], g["feat_q"], g["xs"],
            g["mask"], g["mask_flat"], g["ctx"],
        ],
        dim=-1,
    )
    for steps in [int(v) for v in args.steps_list.split(",")]:
        bits_s4, nps4 = _fit_condmean(
            x_s4[tr], g["mean_s"][tr], g["scale_s"][tr],
            g["xs"][tr], g["q_s"][tr], None,
            x_s4[va], g["mean_s"][va], g["scale_s"][va],
            g["xs"][va], g["q_s"][va], None,
            6, args.hidden, steps, args.weight_decay, device,
        )
        bits_o4, npo4 = _fit_condmean(
            x_o4[tr], g["mean_o"][tr], g["scale_o"][tr],
            g["xo"][tr], g["q_o"][tr], g["mask_flat"][tr],
            x_o4[va], g["mean_o"][va], g["scale_o"][va],
            g["xo"][va], g["q_o"][va], g["mask_flat"][va],
            3 * k_offsets, args.hidden, steps, args.weight_decay, device,
        )
        add_variant(
            f"R4_condmean_h{args.hidden}_s{steps}",
            H_feat_base,
            bits_s4,
            bits_o4,
            nps4 + npo4,
        )

    # --- R3: feat channel-delta ---------------------------------------------
    n_groups = feat_dim // chunk
    b_delta = []
    with torch.no_grad():
        prev = torch.zeros_like(g["feat_q"][tr][:, :chunk])
        for cc in range(n_groups):
            sl = slice(cc * chunk, (cc + 1) * chunk)
            d = g["feat_q"][tr][:, sl] - prev
            b_delta.append(d.abs().mean().clamp_min(1e-9))
            prev = g["feat_q"][tr][:, sl].clone()
        H_f_delta = 0.0
        prev = torch.zeros_like(g["feat_q"][va][:, :chunk])
        for cc in range(n_groups):
            sl = slice(cc * chunk, (cc + 1) * chunk)
            d = g["feat_q"][va][:, sl] - prev
            H_f_delta += float(
                (-torch.log2(
                    _laplace_bin(
                        torch.round(d / g["q"][va][:, sl]) * g["q"][va][:, sl],
                        torch.zeros_like(d),
                        b_delta[cc],
                        g["q"][va][:, sl],
                    )
                )).sum().item()
            )
            prev = g["feat_q"][va][:, sl].clone()
    add_variant("R3_feat_delta", H_f_delta, H_s_base, H_o_base)

    # --- Decisions ----------------------------------------------------------
    best_r2 = max(
        (k for k in rows if k.startswith("R2_laplace")),
        key=lambda k: rows[k]["gain_pct"],
    )
    best_r2g = max(
        (k for k in rows if k.startswith("R2_gaussian")),
        key=lambda k: rows[k]["gain_pct"],
    )
    best_r4 = max(
        (k for k in rows if k.startswith("R4_")),
        key=lambda k: rows[k]["gain_pct"],
    )
    best_r1 = max(
        (k for k in rows if k.startswith("R1_")),
        key=lambda k: rows[k]["gain_pct"],
    )
    r0_gain = rows["R0_laplace"]["gain_pct"]
    r2_row = rows[best_r2]
    stop1 = r2_row["gain_so_pct"] < 3.0 or (
        r2_row["net_total_pct"] is not None and r2_row["net_total_pct"] < 1.0
    ) or r2_row["net_field_pct"] < 1.0
    diff_r2g_r4 = rows[best_r2g]["gain_pct"] - rows[best_r4]["gain_pct"]
    stop2 = abs(diff_r2g_r4) < 0.1
    stop3 = r0_gain <= 0.0 and rows[best_r1]["gain_pct"] <= 0.0 and r2_row["gain_pct"] <= 0.0
    stop4 = r2_row["net_save_MB"] <= 0.0
    reasons = []
    if stop1:
        reasons.append(
            f"R2 gain/net below threshold (scaling+offsets gain="
            f"{r2_row['gain_so_pct']:.2f}%, "
            f"net_field={r2_row['net_field_pct']:.2f}%)"
        )
    if stop2:
        reasons.append(
            f"R2_gauss vs R4 diff {diff_r2g_r4:.3f}% < 0.1% (no residual-structure gain)"
        )
    if stop3:
        reasons.append("R0/R1/R2 all have no gain")
    if stop4:
        reasons.append("predictor weights eat all savings")
    decision = "stage_b" if not reasons else "close"

    summary = {
        "iteration": int(iteration),
        "N_anchors": int(N),
        "N_train": int(n_tr),
        "N_val": int(N - n_tr),
        "feat_dim": int(feat_dim),
        "chunk": int(chunk),
        "reference_total_MB": round(ref_total_mb, 4) if ref_total_mb > 0 else None,
        "H_base": {
            "feat_MB": round(_mb(H_feat_base), 4),
            "scaling_MB": round(_mb(H_s_base), 4),
            "offsets_MB": round(_mb(H_o_base), 4),
            "total_MB": round(_mb(H_base), 4),
            "total_bits": round(H_base, 2),
        },
        "variants": rows,
        "best_R2": best_r2,
        "best_R2_gaussian": best_r2g,
        "best_R4": best_r4,
        "diff_R2gauss_minus_R4_pct": round(diff_r2g_r4, 3),
        "decisions": {"stop1": stop1, "stop2": stop2, "stop3": stop3, "stop4": stop4},
        "reasons": reasons,
        "decision": decision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[R] DECISION: " + ("stage_b" if decision == "stage_b" else "close"))
    if reasons:
        for r in reasons:
            print("  - " + r)


if __name__ == "__main__":
    main()
