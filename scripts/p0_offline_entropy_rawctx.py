"""P0-2 raw-neighbor context experiment (no mean/max pooling of raw features).

Question: did the previous mean/max pooling over raw neighbor features lose
the neighbor information?  Here each of the previous ``k`` Morton neighbors is
mapped through a shared ``Linear(D,32)+ReLU``, the k embeddings are mean
pooled to a 32-dim context, concatenated after the base entropy parameters
``[mean_s, scale_s, mean_o, scale_o]``, and fed to the residual ``_AdjMLP``
(last layer zero-initialized so the start equals the baseline).

Decision rule (from the experiment plan):
  gain < 1%  -> P0-2 is formally closed;
  gain >= 3% -> raw neighbor info is useful, consider full reproduction;
  1%..3%     -> run a real RD probe before deciding.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/p0_offline_entropy_rawctx.py \
      --ckpt runs/4-28_i6_90k_h32/ckpts/ckpt_90000.pth \
      --out runs/p0_offline_rawctx.json --k-list 8,16,32 --steps 400
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


def _gaussian_bits(x_hat, mean, scale, q):
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-6))
    p = dist.cdf(x_hat + 0.5 * q) - dist.cdf(x_hat - 0.5 * q)
    return -torch.log2(p.clamp_min(1e-12))


class _AdjMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        # Residual adjustment: start at identity so the initial state equals
        # the baseline entropy model.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class _NeighborEmbed(nn.Module):
    """Shared per-neighbor Linear(D -> 32) + ReLU (no attention)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(inplace=True))

    def forward(self, nbr):  # [B, k, D] -> [B, k, 32]
        return self.net(nbr)


def _apply_adj(base_mean_s, base_scale_s, base_mean_o, base_scale_o, adj):
    ds = base_mean_s.shape[-1]
    do = base_mean_o.shape[-1]
    mean_s = base_mean_s + adj[..., :ds]
    scale_s = base_scale_s * torch.exp(adj[..., ds : 2 * ds].clamp(-1.4, 1.4))
    mean_o = base_mean_o + adj[..., 2 * ds : 2 * ds + do]
    scale_o = base_scale_o * torch.exp(
        adj[..., 2 * ds + do : 2 * ds + 2 * do].clamp(-1.4, 1.4)
    )
    return mean_s, scale_s, mean_o, scale_o


def _build_neighbors(neighbor_feat, k):
    """Stack previous k Morton neighbors: [N, k, D], zero-padded."""
    n = neighbor_feat.shape[0]
    nbr = torch.zeros(n, k, neighbor_feat.shape[1], device=neighbor_feat.device)
    for d in range(1, k + 1):
        nbr[d:, d - 1] = neighbor_feat[:-d]
    return nbr


def _fit_rawctx(
    nbr_tr,
    nbr_va,
    *,
    mean_s_tr, scale_s_tr, mean_o_tr, scale_o_tr, q_s_tr, q_o_tr, mask_tr,
    mean_s_va, scale_s_va, mean_o_va, scale_o_va, q_s_va, q_o_va, mask_va,
    xs_tr, xo_tr, xs_va, xo_va,
    hidden,
    steps,
    weight_decay,
    device,
):
    embed = _NeighborEmbed(nbr_tr.shape[-1]).to(device)
    ds = mean_s_tr.shape[-1]
    do = mean_o_tr.shape[-1]
    base_dim = 2 * ds + 2 * do
    mlp = _AdjMLP(base_dim + 32, hidden, 2 * ds + 2 * do).to(device)
    opt = torch.optim.Adam(
        list(embed.parameters()) + list(mlp.parameters()),
        lr=1e-3,
        weight_decay=weight_decay,
    )

    def _bits(mean_s, scale_s, mean_o, scale_o, xs, xo, q_s, q_o, mask):
        return (
            _gaussian_bits(xs, mean_s, scale_s, q_s).sum()
            + _gaussian_bits(xo, mean_o, scale_o, q_o)[mask].sum()
        )

    with torch.no_grad():
        pooled = embed(nbr_va).mean(dim=1)
        x0 = torch.cat([mean_s_va, scale_s_va, mean_o_va, scale_o_va, pooled], dim=-1)
        adj0 = mlp(x0)
        ms0, ss0, mo0, so0 = _apply_adj(mean_s_va, scale_s_va, mean_o_va, scale_o_va, adj0)
        best_val = float(_bits(ms0, ss0, mo0, so0, xs_va, xo_va, q_s_va, q_o_va, mask_va))
        best_state = {
            "embed": {k: v.clone() for k, v in embed.state_dict().items()},
            "mlp": {k: v.clone() for k, v in mlp.state_dict().items()},
        }

    n = nbr_tr.shape[0]
    for it in range(steps):
        idx = torch.randperm(n, device=device)[: min(n, 8192)]
        pooled = embed(nbr_tr[idx]).mean(dim=1)
        x = torch.cat(
            [mean_s_tr[idx], scale_s_tr[idx], mean_o_tr[idx], scale_o_tr[idx], pooled],
            dim=-1,
        )
        adj = mlp(x)
        ms, ss, mo, so = _apply_adj(
            mean_s_tr[idx], scale_s_tr[idx], mean_o_tr[idx], scale_o_tr[idx], adj
        )
        loss = (
            _gaussian_bits(xs_tr[idx], ms, ss, q_s_tr[idx]).mean()
            + _gaussian_bits(
                xo_tr[idx][mask_tr[idx]],
                mo[mask_tr[idx]],
                so[mask_tr[idx]],
                q_o_tr[idx][mask_tr[idx]],
            ).mean()
            + 0.1 * adj.abs().mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                pooled = embed(nbr_va).mean(dim=1)
                xv = torch.cat(
                    [mean_s_va, scale_s_va, mean_o_va, scale_o_va, pooled], dim=-1
                )
                adjv = mlp(xv)
                msv, ssv, mov, sov = _apply_adj(
                    mean_s_va, scale_s_va, mean_o_va, scale_o_va, adjv
                )
                val_bits = float(_bits(msv, ssv, mov, sov, xs_va, xo_va, q_s_va, q_o_va, mask_va))
            if val_bits < best_val:
                best_val = val_bits
                best_state = {
                    "embed": {k: v.clone() for k, v in embed.state_dict().items()},
                    "mlp": {k: v.clone() for k, v in mlp.state_dict().items()},
                }

    embed.load_state_dict(best_state["embed"])
    mlp.load_state_dict(best_state["mlp"])
    with torch.no_grad():
        pooled = embed(nbr_va).mean(dim=1)
        xv = torch.cat([mean_s_va, scale_s_va, mean_o_va, scale_o_va, pooled], dim=-1)
        adjv = mlp(xv)
        msv, ssv, mov, sov = _apply_adj(mean_s_va, scale_s_va, mean_o_va, scale_o_va, adjv)
        bits_s = _gaussian_bits(xs_va, msv, ssv, q_s_va).sum()
        bits_o = _gaussian_bits(xo_va, mov, sov, q_o_va)[mask_va].sum()
    return float(bits_s.item()), float(bits_o.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/p0_offline_rawctx.json")
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
    print(f"[P0-raw] {N} anchors, iteration={iteration}", flush=True)

    parts = {name: [] for name in
             ("feat_q", "xs", "xo", "q_s", "q_o", "mean_s", "scale_s",
              "mean_o", "scale_o", "mask", "anchor")}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="p0_rawctx")
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
                "p0_rawctx", anchor_slice, masks_slice,
                Q_feat, Q_scaling, Q_offsets, None, None, None,
                mean_scaling.view(-1, 6), mean_offsets.view(-1, 3 * k_offsets),
            )
        parts["feat_q"].append(torch.round(feat[start:end] / Q_feat) * Q_feat)
        parts["xs"].append(torch.round(scaling[start:end] / Q_scaling) * Q_scaling)
        xo = torch.round(
            offsets[start:end].reshape(-1, 3 * k_offsets) / Q_offsets
        ) * Q_offsets
        parts["xo"].append(xo)
        parts["q_s"].append(Q_scaling)
        parts["q_o"].append(Q_offsets)
        parts["mean_s"].append(mean_scaling)
        parts["scale_s"].append(scale_scaling.clamp(min=1e-9))
        parts["mean_o"].append(mean_offsets)
        parts["scale_o"].append(scale_offsets.clamp(min=1e-9))
        parts["mask"].append(
            masks[start:end].reshape(-1, k_offsets, 1).repeat(1, 1, 3).reshape(
                -1, 3 * k_offsets
            )
        )
        parts["anchor"].append(anchor_slice)

    g = {name: torch.cat(v, dim=0).contiguous().detach() for name, v in parts.items()}
    mask_bool = g["mask"].bool()
    N_train = int(N * (1.0 - args.val_fraction))

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
    print(f"H_base: {H_base / 8 / 1024 / 1024:.4f}MB", flush=True)

    neighbor_feat = torch.cat(
        [g["feat_q"], g["xs"], g["xo"], g["mask"], g["anchor"]], dim=-1
    )
    rows = {}
    for k in [int(x) for x in args.k_list.split(",")]:
        print(f"--- k={k} (steps={args.steps}) ---", flush=True)
        nbr = _build_neighbors(neighbor_feat, k)
        tr = slice(k, N_train)
        va = slice(N_train, N)
        bs, bo = _fit_rawctx(
            nbr[tr], nbr[va],
            mean_s_tr=g["mean_s"][tr], scale_s_tr=g["scale_s"][tr],
            mean_o_tr=g["mean_o"][tr], scale_o_tr=g["scale_o"][tr],
            q_s_tr=g["q_s"][tr], q_o_tr=g["q_o"][tr], mask_tr=mask_bool[tr],
            mean_s_va=g["mean_s"][va], scale_s_va=g["scale_s"][va],
            mean_o_va=g["mean_o"][va], scale_o_va=g["scale_o"][va],
            q_s_va=g["q_s"][va], q_o_va=g["q_o"][va], mask_va=mask_bool[va],
            xs_tr=g["xs"][tr], xo_tr=g["xo"][tr],
            xs_va=g["xs"][va], xo_va=g["xo"][va],
            hidden=args.hidden, steps=args.steps,
            weight_decay=args.weight_decay, device=device,
        )
        H_ctx = bs + bo
        gain = (H_base - H_ctx) / H_base
        rows[k] = {
            "MB": round(H_ctx / 8 / 1024 / 1024, 4),
            "gain": round(float(gain) * 100, 3),
            "verdict": (
                "close" if gain < 0.01
                else "rd_probe" if gain < 0.03
                else "useful"
            ),
        }
        print(
            f"k={k}: {H_ctx / 8 / 1024 / 1024:.4f}MB gain={gain * 100:.2f}% "
            f"verdict={rows[k]['verdict']}",
            flush=True,
        )
        del nbr
        torch.cuda.empty_cache()

    summary = {
        "iteration": int(iteration),
        "N_anchors": int(N),
        "steps": args.steps,
        "hidden": args.hidden,
        "H_base_MB": round(H_base / 8 / 1024 / 1024, 4),
        "k": {str(k): v for k, v in rows.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
