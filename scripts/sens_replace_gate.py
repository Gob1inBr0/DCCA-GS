"""I6 replace-scheme Phase-1 gate: offline predictor input ablation.

Question: can a decoder-recomputable ``mlp_sens`` (independent of the I2
formula MLP) predict the render-sensitivity multiplier better than the current
formula features? Candidate input sets (all computable before attribute
decode, i.e. zero side info):

- A: formula features (8-dim, current I2 input) -- baseline (~0 correlation)
- B: A + hash context + mlp_grid mean/scale/prob + base Q hints
- C: B + mean over the previous k Morton neighbors' decoded attributes

The script accumulates the I6 sensitivity EMA on the checkpoint, builds the
target multiplier ``1 + strength*tanh(-z_rel)`` per field, fits a small
``mlp_sens`` per input set on the first 80% of Morton order and reports the
validation Pearson correlation between predicted and target multipliers.

Kill condition (design): if B/C correlations stay < 0.3, the replacement
scheme is closed.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/sens_replace_gate.py \
      --ckpt runs/4-28_i6_90k_h32/ckpts/ckpt_90000.pth \
      --data-dir <4-28> --steps 200 --k-list 8,16,32 --out runs/sens_replace_gate.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusModel, anchor_codec_order
from scaffold_gs.trainer import load_checkpoint


def _collect_sensitivity(model, dataset, steps):
    core = model.core
    model.train()
    cams = list(dataset.train_cameras)
    for i in range(steps):
        cam = cams[i % len(cams)]
        out = model.render(
            cam,
            dataset.background,
            is_training=True,
            retain_grad=True,
            appearance_id=cam.appearance_id,
            step=30_000 + i,
        )
        gt = dataset.get_image(cam)
        loss = ((out.image[0].permute(2, 0, 1) - gt) ** 2).mean()
        loss.backward()
        model.accumulate_sensitivity(out.gaussians)
        model.optimizer.zero_grad(set_to_none=True)


class _SensMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        # Start at a constant output (logits 0 -> multiplier 1) so the initial
        # state is a valid neutral baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _fit_corr(x_tr, y_tr, x_va, y_va, hidden, steps, wd, device):
    mlp = _SensMLP(x_tr.shape[-1], hidden, y_tr.shape[-1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=wd)

    def _pred(x):
        return 1.0 + torch.tanh(mlp(x))

    with torch.no_grad():
        best_loss = float(torch.nn.functional.mse_loss(_pred(x_va), y_va))
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    n = x_tr.shape[0]
    for it in range(steps):
        idx = torch.randperm(n, device=device)[: min(n, 8192)]
        loss = torch.nn.functional.mse_loss(_pred(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                vl = float(torch.nn.functional.mse_loss(_pred(x_va), y_va))
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    mlp.load_state_dict(best_state)
    with torch.no_grad():
        pred = _pred(x_va)
        corr = {}
        for j, name in enumerate(["feat", "scaling", "offsets"]):
            a = pred[:, j]
            b = y_va[:, j]
            if a.std() == 0 or b.std() == 0:
                corr[name] = float("nan")
            else:
                corr[name] = round(
                    float(torch.corrcoef(torch.stack([a, b]))[0, 1]), 4
                )
    return corr, best_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--k-list", default="8,16,32")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fit-steps", type=int, default=400)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/sens_replace_gate.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, optim_cfg, iteration, _ = load_checkpoint(args.ckpt, "cuda")
    assert isinstance(model, HACPlusModel)
    model.cfg.sensitivity_enabled = True
    model.cfg.sensitivity_weight = 1e-3
    model.cfg.sensitivity_start_iter = 0
    model.create_optimizer(optim_cfg)

    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=1,
        max_width=1600,
        test_every=8,
        white_background=False,
        preload_images=False,
        device="cuda",
    )
    _collect_sensitivity(model, dataset, args.steps)
    core = model.core
    k_offsets = model.cfg.n_offsets
    feat_dim = model.cfg.feat_dim

    order = anchor_codec_order(model)
    anchor = core.get_anchor.detach()[order]
    feat = model._view.anchor_feat.detach()[order]
    scaling = core.get_scaling.detach()[order]
    offsets = model._view.offset.detach()[order]
    masks = core.get_mask.detach()[order]
    N = anchor.shape[0]
    print(f"[sens_replace] {N} anchors, iteration={iteration}", flush=True)

    parts = {name: [] for name in
             ("ctx", "mean", "scale", "prob", "q", "feat_q",
              "scaling_q", "offsets_q", "mask",
              "mean_s", "scale_s", "mean_o", "scale_o", "qa", "qs", "qo",
              "formula", "sens")}
    from hacplus.scene.gaussian_model import MAX_batch_size
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="sens_replace")
        out = core.get_grid_mlp(ctx)
        (
            mean, scale, prob, mean_scaling, scale_scaling,
            mean_offsets, scale_offsets, qa, qs, qo,
        ) = torch.split(
            out,
            [feat_dim, feat_dim, feat_dim, 6, 6, 3 * k_offsets, 3 * k_offsets, 1, 1, 1],
            dim=-1,
        )
        qa_r = qa.repeat(1, feat_dim)
        qs_r = qs.repeat(1, 6)
        qo_r = qo.repeat(1, 3 * k_offsets)
        Q_feat = 1.0 * (1 + torch.tanh(qa_r))
        Q_scaling = 0.001 * (1 + torch.tanh(qs_r))
        Q_offsets = 0.2 * (1 + torch.tanh(qo_r))
        if core.is_content_aware_quant_active():
            masks_slice = masks[start:end]
            (
                Q_feat, Q_scaling, Q_offsets, _, _, _, _,
            ) = core._codec_apply_content_aware_quant_params(
                "sens_replace", anchor_slice, masks_slice,
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
        formula = core.build_formula_complexity_input(
            anchor_slice,
            mean_scaling.view(-1, 6),
            mean_offsets.view(-1, 3 * k_offsets),
            masks[start:end],
        )
        idx = torch.arange(start, end, device=device)
        ema = torch.stack(
            [
                core.sensitivity_feat[idx].squeeze(-1),
                core.sensitivity_scaling[idx].squeeze(-1),
                core.sensitivity_offsets[idx].squeeze(-1),
            ],
            dim=-1,
        )
        z = (ema - core.sensitivity_mean) / core.sensitivity_mean.clamp_min(1e-12)
        sens = 1.0 + torch.tanh(-z)
        parts["ctx"].append(ctx)
        parts["mean"].append(mean)
        parts["scale"].append(scale.clamp(min=1e-9))
        parts["prob"].append(prob)
        parts["q"].append(Q_feat)
        parts["feat_q"].append(feat_q)
        parts["scaling_q"].append(xs)
        parts["offsets_q"].append(xo)
        parts["mask"].append(msk)
        parts["mean_s"].append(mean_scaling)
        parts["scale_s"].append(scale_scaling.clamp(min=1e-9))
        parts["mean_o"].append(mean_offsets)
        parts["scale_o"].append(scale_offsets.clamp(min=1e-9))
        parts["qa"].append(qa_r)
        parts["qs"].append(qs_r)
        parts["qo"].append(qo_r)
        parts["formula"].append(formula)
        parts["sens"].append(sens)

    g = {name: torch.cat(v, dim=0).contiguous().detach() for name, v in parts.items()}
    N_train = int(N * (1.0 - args.val_fraction))
    tr = slice(0, N_train)
    va = slice(N_train, N)

    feats = {
        "A": g["formula"],
        "B": torch.cat(
            [g["formula"], g["ctx"], g["mean"], g["scale"], g["prob"],
             g["mean_s"], g["scale_s"], g["mean_o"], g["scale_o"],
             g["qa"], g["qs"], g["qo"]],
            dim=-1,
        ),
    }
    neighbor = torch.cat(
        [g["feat_q"], g["scaling_q"], g["offsets_q"], g["mask"]], dim=-1
    )
    rows = {}
    for name, x in feats.items():
        corr, loss = _fit_corr(
            x[tr], g["sens"][tr], x[va], g["sens"][va],
            args.hidden, args.fit_steps, args.weight_decay, device,
        )
        rows[name] = {"corr": corr, "mse": round(loss, 6)}
        print(f"set {name}: corr={corr} mse={loss:.6f}", flush=True)

    for k in [int(v) for v in args.k_list.split(",")]:
        mean_acc = torch.zeros_like(neighbor)
        for d in range(1, k + 1):
            prev = torch.zeros_like(neighbor)
            prev[d:] = neighbor[:-d]
            mean_acc += prev
        mean_acc /= float(k)
        xc = torch.cat([feats["B"], mean_acc], dim=-1)
        corr, loss = _fit_corr(
            xc[tr], g["sens"][tr], xc[va], g["sens"][va],
            args.hidden, args.fit_steps, args.weight_decay, device,
        )
        rows[f"C_k{k}"] = {"corr": corr, "mse": round(loss, 6)}
        print(f"set C_k{k}: corr={corr} mse={loss:.6f}", flush=True)

    best_corr = max(
        (
            v["corr"].get("feat", 0.0)
            + v["corr"].get("scaling", 0.0)
            + v["corr"].get("offsets", 0.0)
        ) / 3.0
        for name, v in rows.items()
        if name != "A"
    )
    summary = {
        "iteration": int(iteration),
        "N_anchors": int(N),
        "steps": args.steps,
        "input_sets": rows,
        "best_mean_corr_non_A": round(best_corr, 4),
        "pass": best_corr >= 0.3,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[sens_replace] GATE:", "PASS" if summary["pass"] else "FAIL (<0.3)")


if __name__ == "__main__":
    main()
