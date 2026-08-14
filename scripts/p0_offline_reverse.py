"""P0 reverse progressive coding offline experiment (scaling+offsets -> feat).

Question: does conditioning the feat entropy model on decoded scaling, offsets
and masks reduce feat bits, and is the absolute saving larger than the forward
P0-1 (feat -> scaling/offsets)?

The feat entropy replicates the real codec: mixture-of-Gaussians plus the
channel autoregressive loop (``Channel_CTX_fea``), one 10-channel group at a
time, feeding back the STE-quantized decoded channels.

V1 conditioning: a residual ``mlp_feat_ctx`` (last layer zero-initialized)
adjusts the initial feat mean/scale from
``concat(mean, scale, scaling_q, offsets_q, mask)``; the original
channel-autoregressive loop then runs unchanged.

Decision rules:
  1. gain < 2% or Delta_reverse <= Delta_forward -> close the reverse route;
  2. gain >= 2% and Delta_reverse > Delta_forward -> stage B is justified.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/p0_offline_reverse.py \
      --ckpt runs/4-28_i6_90k_h32/ckpts/ckpt_90000.pth \
      --out runs/p0_offline_reverse.json \
      --hidden-list 64,128 --steps-list 400,1500
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


def _gaussian_bin(x, mean, scale, q):
    """P(symbol) with a Gaussian bin of width q centered at x (x = s*q)."""
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-6))
    p = dist.cdf(x + 0.5 * q) - dist.cdf(x - 0.5 * q)
    return p.clamp_min(1e-12)


def _feature_mixed_bits(
    feat_q,
    mean,
    scale,
    prob,
    q,
    deform,
    feat_dim,
    chunk,
):
    """Feat bits under the real codec's mixed-Gaussian channel autoregression.

    Mirrors ``encode_attributes``: per 10-channel group, the deform MLP
    predicts adjusted mixture parameters from the partial decoded feat, the
    mixture is softmax-blended with the base Gaussian, and the coded channel
    is fed back as ``feat_q`` (STE simulation of the decoder).
    """
    b = feat_q.shape[0]
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


class _FeatCtxMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _fit_reverse(
    x_tr,
    x_va,
    *,
    feat_q_tr, mean_tr, scale_tr, prob_tr, q_tr,
    feat_q_va, mean_va, scale_va, prob_va, q_va,
    deform, feat_dim, chunk,
    hidden,
    steps,
    weight_decay,
    device,
):
    mlp = _FeatCtxMLP(x_tr.shape[-1], hidden, 2 * feat_dim).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=weight_decay)

    def _cond(feat_q, mean, scale, prob, q, x):
        adj = mlp(x)
        mean_c = mean + adj[:, :feat_dim]
        scale_c = scale * torch.exp(adj[:, feat_dim : 2 * feat_dim].clamp(-1.4, 1.4))
        return _feature_mixed_bits(
            feat_q, mean_c, scale_c, prob, q, deform, feat_dim, chunk
        )

    with torch.no_grad():
        best_val = float(_cond(feat_q_va, mean_va, scale_va, prob_va, q_va, x_va))
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    n = x_tr.shape[0]
    for it in range(steps):
        idx = torch.randperm(n, device=device)[: min(n, 8192)]
        adj = mlp(x_tr[idx])
        mean_c = mean_tr[idx] + adj[:, :feat_dim]
        scale_c = scale_tr[idx] * torch.exp(
            adj[:, feat_dim : 2 * feat_dim].clamp(-1.4, 1.4)
        )
        bits = _feature_mixed_bits(
            feat_q_tr[idx], mean_c, scale_c, prob_tr[idx], q_tr[idx],
            deform, feat_dim, chunk,
        )
        loss = bits / (feat_q_tr[idx].numel()) + 0.1 * adj.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                val = float(_cond(feat_q_va, mean_va, scale_va, prob_va, q_va, x_va))
            if val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    mlp.load_state_dict(best_state)
    with torch.no_grad():
        bits = float(_cond(feat_q_va, mean_va, scale_va, prob_va, q_va, x_va))
    return bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/p0_offline_reverse.json")
    p.add_argument("--hidden-list", default="64,128")
    p.add_argument("--steps-list", default="400,1500")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--forward-json", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    assert isinstance(model, HACPlusModel)
    core = model.core
    k_offsets = model.cfg.n_offsets
    feat_dim = model.cfg.feat_dim
    chunk = 10
    from hacplus.scene.gaussian_model import MAX_batch_size

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]
    print(f"[P0-rev] {N} anchors, iteration={iteration}", flush=True)

    parts = {name: [] for name in
             ("mean", "scale", "prob", "q", "feat_q",
              "scaling_q", "offsets_q", "mask",
              "q_s", "q_o", "mean_s", "scale_s", "mean_o", "scale_o",
              "xs", "xo")}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="p0_reverse")
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
                "p0_reverse", anchor_slice, masks_slice,
                Q_feat, Q_scaling, Q_offsets, None, None, None,
                mean_scaling.view(-1, 6), mean_offsets.view(-1, 3 * k_offsets),
            )
        feat_q = torch.round(feat[start:end] / Q_feat) * Q_feat
        xs = torch.round(scaling[start:end] / Q_scaling) * Q_scaling
        xo = torch.round(
            offsets[start:end].reshape(-1, 3 * k_offsets) / Q_offsets
        ) * Q_offsets
        msk = masks[start:end].reshape(-1, k_offsets)
        msk_flat = msk.repeat(1, 3).reshape(-1, 3 * k_offsets).bool()
        xo = torch.where(msk_flat, xo, torch.zeros_like(xo))
        parts["mean"].append(mean)
        parts["scale"].append(scale.clamp(min=1e-9))
        parts["prob"].append(prob)
        parts["q"].append(Q_feat)
        parts["feat_q"].append(feat_q)
        parts["scaling_q"].append(xs)
        parts["offsets_q"].append(xo)
        parts["mask"].append(msk)
        parts["q_s"].append(Q_scaling)
        parts["q_o"].append(Q_offsets)
        parts["mean_s"].append(mean_scaling)
        parts["scale_s"].append(scale_scaling.clamp(min=1e-9))
        parts["mean_o"].append(mean_offsets)
        parts["scale_o"].append(scale_offsets.clamp(min=1e-9))
        parts["xs"].append(xs)
        parts["xo"].append(xo)

    g = {name: torch.cat(v, dim=0).contiguous().detach() for name, v in parts.items()}
    mask_bool = g["mask"].repeat(1, 3).reshape(-1, 3 * k_offsets).bool()
    N_train = int(N * (1.0 - args.val_fraction))
    deform = core.get_deform_mlp

    with torch.no_grad():
        base_feat = _feature_mixed_bits(
            g["feat_q"][N_train:], g["mean"][N_train:], g["scale"][N_train:],
            g["prob"][N_train:], g["q"][N_train:], deform, feat_dim, chunk,
        )
        base_so = (
            _gaussian_bin(
                g["xs"][N_train:], g["mean_s"][N_train:], g["scale_s"][N_train:],
                g["q_s"][N_train:],
            ).log2().sum()
            + _gaussian_bin(
                g["xo"][N_train:], g["mean_o"][N_train:], g["scale_o"][N_train:],
                g["q_o"][N_train:],
            ).log2()[mask_bool[N_train:]].sum()
        )
    H_feat_base = float(base_feat.item())
    H_so_base = float(-base_so.item())
    print(
        f"H_feat_base={H_feat_base / 8 / 1024 / 1024:.4f}MB "
        f"H_so_base={H_so_base / 8 / 1024 / 1024:.4f}MB",
        flush=True,
    )

    delta_forward = None
    if args.forward_json:
        fj = Path(args.forward_json)
        if fj.is_file():
            fwd = json.loads(fj.read_text())
            delta_forward = fwd["H_base_MB"] * fwd["P0_1"]["gain"] / 100.0
    if delta_forward is None:
        delta_forward = H_so_base / 8 / 1024 / 1024 * 0.02298

    x_cond = torch.cat(
        [g["scaling_q"], g["offsets_q"], g["mask"]], dim=-1
    )
    x_feat = torch.cat([g["mean"], g["scale"], x_cond], dim=-1)

    rows = {}
    for hidden in [int(v) for v in args.hidden_list.split(",")]:
        for steps in [int(v) for v in args.steps_list.split(",")]:
            print(f"--- hidden={hidden} steps={steps} ---", flush=True)
            bits = _fit_reverse(
                x_feat[:N_train], x_feat[N_train:],
                feat_q_tr=g["feat_q"][:N_train], mean_tr=g["mean"][:N_train],
                scale_tr=g["scale"][:N_train], prob_tr=g["prob"][:N_train],
                q_tr=g["q"][:N_train],
                feat_q_va=g["feat_q"][N_train:], mean_va=g["mean"][N_train:],
                scale_va=g["scale"][N_train:], prob_va=g["prob"][N_train:],
                q_va=g["q"][N_train:],
                deform=deform, feat_dim=feat_dim, chunk=chunk,
                hidden=hidden, steps=steps,
                weight_decay=args.weight_decay, device=device,
            )
            gain = (H_feat_base - bits) / H_feat_base
            delta_rev = H_feat_base - bits
            rows[f"h{hidden}_s{steps}"] = {
                "MB": round(bits / 8 / 1024 / 1024, 4),
                "gain": round(float(gain) * 100, 3),
                "delta_reverse_MB": round(delta_rev / 8 / 1024 / 1024, 4),
            }
            print(
                f"h{hidden}_s{steps}: {bits / 8 / 1024 / 1024:.4f}MB "
                f"gain={gain * 100:.2f}% delta_rev={delta_rev / 8 / 1024 / 1024:.5f}MB",
                flush=True,
            )

    best = max(rows.values(), key=lambda r: r["gain"])
    decision = (
        "close_below_2pct"
        if best["gain"] < 2.0
        else (
            "close_not_better_than_forward"
            if best["delta_reverse_MB"] <= delta_forward
            else "stage_b"
        )
    )
    summary = {
        "iteration": int(iteration),
        "N_anchors": int(N),
        "H_feat_base_MB": round(H_feat_base / 8 / 1024 / 1024, 4),
        "H_so_base_MB": round(H_so_base / 8 / 1024 / 1024, 4),
        "delta_forward_MB": round(delta_forward, 4),
        "configs": rows,
        "best": max(rows.items(), key=lambda kv: kv[1]["gain"])[0],
        "decision": decision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
