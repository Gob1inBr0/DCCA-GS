"""P0 stage-A offline conditional-entropy validation (P0-1 / P0-2).

On a trained checkpoint (codec order = mask + Morton), measures:

- ``H_base`` : scaling/offsets symbol entropy under the current mlp_grid
  Gaussian entropy parameters (unconditional baseline);
- ``H_prog`` : entropy when a small MLP (the ``mlp_attr_ctx`` spec) conditions
  on the STE-quantized decoded feat (P0-1);
- ``H_ctx(k)``: entropy when a small MLP conditions on mean/max pooling over
  the previous ``k`` Morton neighbors' decoded symbols (P0-2), k scan.

Design-doc stop condition: continue only if at least one of P0-1 / P0-2 gives
an offline gain >= 3% over ``H_base``.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/p0_offline_entropy.py \
      --ckpt runs/4-28_i6_90k_h32/ckpts/ckpt_90000.pth \
      --out runs/p0_offline.json --k-list 8,16,32
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from scaffold_gs.hacpp import HACPlusModel, anchor_codec_order
from scaffold_gs.trainer import load_checkpoint


def _gaussian_bits(
    x_hat: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Per-element -log2 p(symbol) with Gaussian CDF bins of width q."""
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-6))
    p = dist.cdf(x_hat + 0.5 * q) - dist.cdf(x_hat - 0.5 * q)
    return -torch.log2(p.clamp_min(1e-12))


class _AdjMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        # Residual adjustment: start at identity (zero output) so the initial
        # state reproduces the base entropy model exactly.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _apply_adj(
    base_mean_s, base_scale_s, base_mean_o, base_scale_o, adj
) -> tuple:
    """Add mean / log-scale adjustments from a fitted MLP output [..., 72]."""
    ds = base_mean_s.shape[-1]
    do = base_mean_o.shape[-1]
    mean_s = base_mean_s + adj[..., :ds]
    scale_s = base_scale_s * torch.exp(adj[..., ds : 2 * ds].clamp(-1.4, 1.4))
    mean_o = base_mean_o + adj[..., 2 * ds : 2 * ds + do]
    scale_o = base_scale_o * torch.exp(
        adj[..., 2 * ds + do : 2 * ds + 2 * do].clamp(-1.4, 1.4)
    )
    return mean_s, scale_s, mean_o, scale_o


def _fit_and_eval(
    x_train,
    x_val,
    *,
    mean_s_tr, scale_s_tr, mean_o_tr, scale_o_tr, q_s_tr, q_o_tr, mask_tr,
    mean_s_va, scale_s_va, mean_o_va, scale_o_va, q_s_va, q_o_va, mask_va,
    xs_tr, xo_tr, xs_va, xo_va,
    hidden: int,
    steps: int,
    weight_decay: float,
    device: torch.device,
):
    """Fit a small adjustment MLP on the train split, eval on the val split."""
    ds = mean_s_tr.shape[-1]
    do = mean_o_tr.shape[-1]
    out_dim = 2 * ds + 2 * do
    mlp = _AdjMLP(x_train.shape[-1], hidden, out_dim).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=weight_decay)
    with torch.no_grad():
        adj0 = mlp(x_val)
        ms0, ss0, mo0, so0 = _apply_adj(
            mean_s_va, scale_s_va, mean_o_va, scale_o_va, adj0
        )
        best_val = float(
            _gaussian_bits(xs_va, ms0, ss0, q_s_va).sum()
            + _gaussian_bits(xo_va, mo0, so0, q_o_va)[mask_va].sum()
        )
        best = (best_val, {k: v.clone() for k, v in mlp.state_dict().items()})
    n = x_train.shape[0]
    for it in range(steps):
        idx = torch.randperm(n, device=device)[: min(n, 8192)]
        adj = mlp(x_train[idx])
        ms, ss, mo, so = _apply_adj(
            mean_s_tr[idx], scale_s_tr[idx], mean_o_tr[idx], scale_o_tr[idx], adj
        )
        loss_s = _gaussian_bits(xs_tr[idx], ms, ss, q_s_tr[idx]).mean()
        loss_o = _gaussian_bits(
            xo_tr[idx][mask_tr[idx]], mo[mask_tr[idx]], so[mask_tr[idx]], q_o_tr[idx][mask_tr[idx]]
        ).mean()
        loss = loss_s + loss_o + 0.1 * adj.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                adj = mlp(x_val)
                ms, ss, mo, so = _apply_adj(
                    mean_s_va, scale_s_va, mean_o_va, scale_o_va, adj
                )
                val_bits = (
                    _gaussian_bits(xs_va, ms, ss, q_s_va).sum()
                    + _gaussian_bits(xo_va, mo, so, q_o_va)[mask_va].sum()
                )
            if float(val_bits) < best[0]:
                best = (float(val_bits), {k: v.clone() for k, v in mlp.state_dict().items()})

    if best[1] is not None:
        mlp.load_state_dict(best[1])

    with torch.no_grad():
        adj = mlp(x_val)
        ms, ss, mo, so = _apply_adj(mean_s_va, scale_s_va, mean_o_va, scale_o_va, adj)
        bits_s = _gaussian_bits(xs_va, ms, ss, q_s_va).sum()
        bits_o = _gaussian_bits(xo_va, mo, so, q_o_va)[mask_va].sum()
        adjt = mlp(x_train[:8192])
        mst, sst, mot, sot = _apply_adj(
            mean_s_tr[:8192], scale_s_tr[:8192], mean_o_tr[:8192], scale_o_tr[:8192], adjt
        )
        train_bits = (
            _gaussian_bits(xs_tr[:8192], mst, sst, q_s_tr[:8192]).sum()
            + _gaussian_bits(xo_tr[:8192], mot, sot, q_o_tr[:8192])[mask_tr[:8192]].sum()
        )
        base_train_bits = (
            _gaussian_bits(xs_tr[:8192], mean_s_tr[:8192], scale_s_tr[:8192], q_s_tr[:8192]).sum()
            + _gaussian_bits(xo_tr[:8192], mean_o_tr[:8192], scale_o_tr[:8192], q_o_tr[:8192])[mask_tr[:8192]].sum()
        )
        print(
            f"  [fit] train {train_bits/8/1024/1024:.4f}MB vs base {base_train_bits/8/1024/1024:.4f}MB | "
            f"val {(bits_s + bits_o) / 8 / 1024 / 1024:.4f}MB",
            flush=True,
        )
    return float(bits_s.item()), float(bits_o.item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/p0_offline.json")
    p.add_argument("--k-list", default="8,16,32")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, _, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    assert isinstance(model, HACPlusModel)
    core = model.core
    k_offsets = model.cfg.n_offsets
    from hacplus.scene.gaussian_model import MAX_batch_size

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]
    print(f"[P0] {N} anchors, iteration={iteration}", flush=True)

    parts = {name: [] for name in
             ("feat_q", "xs", "xo", "q_s", "q_o", "mean_s", "scale_s",
              "mean_o", "scale_o", "mask", "anchor")}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="p0_offline")
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
        Q_feat = 1.0 * (1 + torch.tanh(qa))
        Q_scaling = 0.001 * (1 + torch.tanh(qs))
        Q_offsets = 0.2 * (1 + torch.tanh(qo))
        if core.is_content_aware_quant_active():
            masks_slice = masks[start:end]
            (
                Q_feat, Q_scaling, Q_offsets, _, _, _, _,
            ) = core._codec_apply_content_aware_quant_params(
                "p0_offline", anchor_slice, masks_slice,
                Q_feat, Q_scaling, Q_offsets, None, None, None,
                mean_scaling.view(-1, 6), mean_offsets.view(-1, 3 * k_offsets),
            )
        feat_q = torch.round(feat[start:end] / Q_feat) * Q_feat
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
        parts["feat_q"].append(feat_q)
        parts["xs"].append(xs)
        parts["xo"].append(xo)
        parts["q_s"].append(Q_scaling)
        parts["q_o"].append(Q_offsets)
        parts["mean_s"].append(mean_scaling)
        parts["scale_s"].append(scale_scaling.clamp(min=1e-9))
        parts["mean_o"].append(mean_offsets)
        parts["scale_o"].append(scale_offsets.clamp(min=1e-9))
        parts["mask"].append(mask)
        parts["anchor"].append(anchor_slice)

    # The model outputs carry autograd graphs; the entropy fits must treat
    # them as constants (detach) or the first backward frees the shared graph.
    g = {
        name: torch.cat(v, dim=0).contiguous().detach()
        for name, v in parts.items()
    }
    mask_bool = g["mask"].bool()
    N_train = int(N * (1.0 - args.val_fraction))

    # --- H_base on the validation split ---
    with torch.no_grad():
        base_s = _gaussian_bits(
            g["xs"][N_train:], g["mean_s"][N_train:], g["scale_s"][N_train:],
            g["q_s"][N_train:],
        ).sum()
        base_o = _gaussian_bits(
            g["xo"][N_train:], g["mean_o"][N_train:], g["scale_o"][N_train:],
            g["q_o"][N_train:],
        )[mask_bool[N_train:]].sum()
    H_base = float(base_s.item() + base_o.item())
    print(f"H_base: scaling={base_s.item()/8/1024/1024:.4f}MB "
          f"offsets={base_o.item()/8/1024/1024:.4f}MB total={H_base/8/1024/1024:.4f}MB",
          flush=True)

    # --- P0-1: condition on decoded feat ---
    x_prog = torch.cat(
        [
            g["mean_s"], g["scale_s"], g["mean_o"], g["scale_o"], g["feat_q"],
        ],
        dim=-1,
    )
    bs, bo = _fit_and_eval(
        x_prog[:N_train], x_prog[N_train:],
        mean_s_tr=g["mean_s"][:N_train], scale_s_tr=g["scale_s"][:N_train],
        mean_o_tr=g["mean_o"][:N_train], scale_o_tr=g["scale_o"][:N_train],
        q_s_tr=g["q_s"][:N_train], q_o_tr=g["q_o"][:N_train],
        mask_tr=mask_bool[:N_train],
        mean_s_va=g["mean_s"][N_train:], scale_s_va=g["scale_s"][N_train:],
        mean_o_va=g["mean_o"][N_train:], scale_o_va=g["scale_o"][N_train:],
        q_s_va=g["q_s"][N_train:], q_o_va=g["q_o"][N_train:],
        mask_va=mask_bool[N_train:],
        xs_tr=g["xs"][:N_train], xo_tr=g["xo"][:N_train],
        xs_va=g["xs"][N_train:], xo_va=g["xo"][N_train:],
        hidden=args.hidden, steps=args.steps, device=device,
        weight_decay=args.weight_decay,
    )
    H_prog = bs + bo
    gain_prog = (H_base - H_prog) / H_base
    print(f"P0-1 H_prog: total={H_prog/8/1024/1024:.4f}MB gain={gain_prog*100:.2f}%",
          flush=True)

    # --- P0-2: condition on mean/max over previous k Morton neighbors ---
    neighbor_feat = torch.cat(
        [g["feat_q"], g["xs"], g["xo"], g["mask"], g["anchor"]], dim=-1
    )  # [N, 50+6+30+10+3]
    rows_ctx = {}
    for k in [int(x) for x in args.k_list.split(",")]:
        mean_acc = torch.zeros_like(neighbor_feat)
        max_acc = torch.full_like(neighbor_feat, float("-inf"))
        for d in range(1, k + 1):
            prev = torch.zeros_like(neighbor_feat)
            prev[d:] = neighbor_feat[:-d]
            mean_acc += prev
            max_acc = torch.maximum(max_acc, prev)
        mean_acc /= float(k)
        ctx_feat = torch.cat([mean_acc, max_acc], dim=-1)
        x_ctx = torch.cat(
            [
                g["mean_s"], g["scale_s"], g["mean_o"], g["scale_o"], ctx_feat,
            ],
            dim=-1,
        )
        tr = slice(k, N_train)
        va = slice(N_train, N)
        bs, bo = _fit_and_eval(
            x_ctx[tr], x_ctx[va],
            mean_s_tr=g["mean_s"][tr], scale_s_tr=g["scale_s"][tr],
            mean_o_tr=g["mean_o"][tr], scale_o_tr=g["scale_o"][tr],
            q_s_tr=g["q_s"][tr], q_o_tr=g["q_o"][tr], mask_tr=mask_bool[tr],
            mean_s_va=g["mean_s"][va], scale_s_va=g["scale_s"][va],
            mean_o_va=g["mean_o"][va], scale_o_va=g["scale_o"][va],
            q_s_va=g["q_s"][va], q_o_va=g["q_o"][va], mask_va=mask_bool[va],
            xs_tr=g["xs"][tr], xo_tr=g["xo"][tr],
            xs_va=g["xs"][va], xo_va=g["xo"][va],
            hidden=args.hidden, steps=args.steps, device=device,
            weight_decay=args.weight_decay,
        )
        H_ctx = bs + bo
        gain = (H_base - H_ctx) / H_base
        rows_ctx[k] = {
            "bits": round(H_ctx, 2),
            "MB": round(H_ctx / 8 / 1024 / 1024, 4),
            "gain": round(float(gain) * 100, 3),
        }
        print(f"P0-2 k={k}: total={H_ctx/8/1024/1024:.4f}MB gain={gain*100:.2f}%",
              flush=True)

    summary = {
        "iteration": int(iteration),
        "N_anchors": int(N),
        "hidden": args.hidden,
        "H_base_bits": round(H_base, 2),
        "H_base_MB": round(H_base / 8 / 1024 / 1024, 4),
        "P0_1": {
            "bits": round(H_prog, 2),
            "MB": round(H_prog / 8 / 1024 / 1024, 4),
            "gain": round(float(gain_prog) * 100, 3),
            "pass": float(gain_prog) >= 0.03,
        },
        "P0_2": {str(k): v for k, v in rows_ctx.items()},
        "pass_any_3pct": float(gain_prog) >= 0.03
        or any(v["gain"] >= 3.0 for v in rows_ctx.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "[P0] ACCEPTANCE: "
        + ("PASS (>=3%)" if summary["pass_any_3pct"] else "FAIL (<3%)")
    )


if __name__ == "__main__":
    main()
