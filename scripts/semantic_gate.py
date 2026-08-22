"""Semantic-prior Stage-A gate (design: docs/语义先验实验设计.md §3).

Aggregates per-view semantic caches (dino/depth/sam2, produced by
scripts/extract_semantic_priors.py) onto anchors through the renderer's
projected 2D positions, then checks whether decoder-recomputable input sets
can predict the semantic target:

  A: formula features (4-d I2 input)
  B: A + hash context + mlp_grid outputs + base Q hints
  C: B + mean over previous k Morton neighbors' decoded attributes (k=8/16/32)

Decision (design §3.2): any signal best Pearson r > 0.3 -> pass;
0.15..0.3 -> keep best signal; all < 0.15 -> close direction.

Usage (5090, HAC_5090_a100 env):

    python scripts/semantic_gate.py \
      --ckpt /home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_l0p002/ckpts/ckpt_90000.pth \
      --data-dir /home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 \
      --cache-dir runs/semantic_cache/4-28 \
      --out runs/semantic_gate/results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scaffold_gs.config import ModelConfig
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusModel, anchor_codec_order


def load_checkpoint_legacy8(path: str, device: torch.device):
    """Load a pre-4D checkpoint (8-dim complexity input, possibly h25/h32).

    The last four image-side statistics were constant zeros in the legacy
    formula mode, so trimming the first four columns is bit-exact; the hidden
    width is rebuilt from the checkpoint state.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = HACPlusModel(cfg, device)
    model.voxel_size = ckpt["voxel_size"]
    model.spatial_lr_scale = ckpt["spatial_lr_scale"]
    sd = dict(ckpt["model_state"])

    # Rebuild mlp_complexity with the checkpoint's hidden width / input 4-d.
    keys = [k for k in sd if "mlp_complexity" in k and k.endswith(".weight")]
    if keys:
        w0 = sd["mlp_complexity.0.weight"]
        hidden = int(w0.shape[0])
        n_linear = len(keys)  # one .weight key per Linear module
        seq = [nn.Linear(4, hidden), nn.ReLU(True)]
        for _ in range(max(n_linear - 2, 0)):
            seq += [nn.Linear(hidden, hidden), nn.ReLU(True)]
        seq.append(nn.Linear(hidden, 3))
        model.core.mlp_complexity = nn.Sequential(*seq).to(device)
        if w0.shape[-1] == 8:
            sd["mlp_complexity.0.weight"] = w0[:, :4].contiguous()
        print(
            f"[semantic_gate] mlp_complexity rebuilt hidden={hidden} "
            f"input=4 (legacy 8->4 trim)"
        )
    model.load_state_dict(sd)
    return model, ckpt


def _collect_inputs(model, device):
    """Decoder-recomputable inputs + quantized attributes in codec order."""
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
    print(f"[semantic_gate] {N} anchors", flush=True)

    from hacplus.scene.gaussian_model import MAX_batch_size

    parts = {name: [] for name in
             ("ctx", "mean", "scale", "prob", "feat_q", "scaling_q",
              "offsets_q", "mask", "mean_s", "scale_s", "mean_o", "scale_o",
              "qa", "qs", "qo", "formula")}
    for s in range(math.ceil(N / MAX_batch_size)):
        start = s * MAX_batch_size
        end = min((s + 1) * MAX_batch_size, N)
        anchor_slice = anchor[start:end]
        ctx = core.calc_context_feat(anchor_slice, caller="semantic_gate")
        out = core.get_grid_mlp(ctx)
        (
            mean, scale, prob, mean_scaling, scale_scaling,
            mean_offsets, scale_offsets, qa, qs, qo,
        ) = torch.split(
            out,
            [feat_dim, feat_dim, feat_dim, 6, 6, 3 * k_offsets, 3 * k_offsets,
             1, 1, 1],
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
                "semantic_gate", anchor_slice, masks_slice,
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
        parts["ctx"].append(ctx)
        parts["mean"].append(mean)
        parts["scale"].append(scale.clamp(min=1e-9))
        parts["prob"].append(prob)
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

    return {
        name: torch.cat(v, dim=0).contiguous().detach()
        for name, v in parts.items()
    }, N


def _sample_nearest(feat_map: torch.Tensor, xy: torch.Tensor, scale: float):
    """feat_map [Hf, Wf, D]; xy [n, 2] pixel coords (x,y) -> [n, D]."""
    hf, wf = feat_map.shape[:2]
    y = (xy[:, 1] / scale).round().long().clamp(0, hf - 1)
    x = (xy[:, 0] / scale).round().long().clamp(0, wf - 1)
    return feat_map[y, x]


def _project_anchors(cam, anchor: torch.Tensor, device):
    """Project anchor centers to pixel coords (x, y) with the gsplat camera."""
    viewmats, Ks = cam.to_gsplat(device)
    view = viewmats[0]
    hom = torch.cat(
        [anchor, torch.ones(anchor.shape[0], 1, device=device)], dim=1
    )
    pv = hom @ view.T  # world -> camera (OpenGL convention, z < 0 in front)
    z = -pv[:, 2].clamp_min(1e-6)
    fx, fy = float(Ks[0, 0, 0]), float(Ks[0, 1, 1])
    cx, cy = float(Ks[0, 0, 2]), float(Ks[0, 1, 2])
    u = fx * pv[:, 0] / z + cx
    v = fy * pv[:, 1] / z + cy
    return torch.stack([u, v], dim=1)


def _aggregate_semantics(model, dataset, cache_dir, signals, device, N):
    """Per-anchor mean semantic target from per-view caches via renderer."""
    cache_dir = Path(cache_dir)
    n_total = int(model.core.get_anchor.shape[0])
    order = anchor_codec_order(model)  # full -> coded order
    sums = {
        sig: torch.zeros(n_total, 1, device=device) for sig in ("depth", "sam2")
    }
    dino_sum = None
    counts = torch.zeros(n_total, device=device)
    sam2_counts = torch.zeros(n_total, device=device)
    cam_order = dataset.train_cameras

    for vi, cam in enumerate(cam_order):
        with torch.no_grad():
            out = model.render(cam, dataset.background)
        aidx = torch.nonzero(out.visible_mask).squeeze(-1)  # global anchor idx
        if aidx.numel() == 0:
            continue
        counts.index_add_(0, aidx, torch.ones_like(aidx, dtype=torch.float32))
        px = _project_anchors(cam, model.core.get_anchor[aidx], device)

        if "dino" in signals:
            p = cache_dir / "dino" / f"{vi:05d}.npz"
            fm = torch.from_numpy(np.load(p)["feat"]).to(device).float()  # [Hp,Wp,768]
            v = _sample_nearest(fm, px, 14.0)
            if dino_sum is None:
                dino_sum = torch.zeros(n_total, v.shape[1], device=device)
            dino_sum.index_add_(0, aidx, v)
        if "depth" in signals:
            p = cache_dir / "depth" / f"{vi:05d}.npz"
            fm = torch.from_numpy(np.load(p)["depth"]).to(device).float()
            v = _sample_nearest(fm.unsqueeze(-1), px, 4.0).squeeze(-1)
            sums["depth"].index_add_(0, aidx, v.unsqueeze(-1))
        if "sam2" in signals:
            p = cache_dir / "sam2" / f"{vi:05d}.npz"
            if not p.exists():
                continue
            sam2_counts.index_add_(0, aidx, torch.ones_like(aidx, dtype=torch.float32))
            aj = json.loads((cache_dir / "sam2" / f"{vi:05d}_area.json").read_text())
            rm = torch.from_numpy(np.load(p)["region"]).to(device).long()
            ids = _sample_nearest(rm.unsqueeze(-1), px, 4.0).squeeze(-1)
            vals = torch.tensor(
                [aj.get(str(int(i)), 0.0) for i in ids.cpu().tolist()],
                device=device,
            )
            sums["sam2"].index_add_(0, aidx, vals.unsqueeze(-1))
        if vi % 100 == 0:
            print(f"[aggregate] view {vi}/{len(cam_order)}", flush=True)

    cov_full = counts >= 3
    out = {}
    if dino_sum is not None:
        out["dino"] = (
            (dino_sum / counts.clamp_min(1).unsqueeze(-1))[order],
            cov_full[order],
        )
    for sig in ("depth", "sam2"):
        if sig in sums and sums[sig].sum() != 0:
            cnt = sam2_counts if sig == "sam2" else counts
            cov_sig = cnt >= 3
            out[sig] = (
                (sums[sig] / cnt.clamp_min(1).unsqueeze(-1))[order],
                cov_sig[order],
            )
    full_out = {}
    if dino_sum is not None:
        full_out["dino"] = (
            dino_sum / counts.clamp_min(1).unsqueeze(-1),
            cov_full,
        )
    for sig in ("depth", "sam2"):
        if sig in sums and sums[sig].sum() != 0:
            cnt = sam2_counts if sig == "sam2" else counts
            full_out[sig] = (
                sums[sig] / cnt.clamp_min(1).unsqueeze(-1),
                cnt >= 3,
            )
    return out, full_out


def _pca_reduce(x: torch.Tensor, q: int) -> torch.Tensor:
    """Randomized PCA to q components on covered anchors (z-scored)."""
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True).clamp_min(1e-6)
    z = (x - mu) / sd
    _, _, v = torch.pca_lowrank(z, q=q, center=False)
    return (z @ v[:, :q])


class _PredMLP(nn.Module):
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


def _fit_corr(x_tr, y_tr, x_va, y_va, hidden, steps, wd, device):
    mlp = _PredMLP(x_tr.shape[-1], hidden, y_tr.shape[-1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=wd)
    with torch.no_grad():
        best_loss = float(torch.nn.functional.mse_loss(mlp(x_va), y_va))
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    n = x_tr.shape[0]
    for it in range(steps):
        idx = torch.randperm(n, device=device)[: min(n, 8192)]
        loss = torch.nn.functional.mse_loss(mlp(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                vl = float(torch.nn.functional.mse_loss(mlp(x_va), y_va))
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    mlp.load_state_dict(best_state)
    with torch.no_grad():
        pred = mlp(x_va)
        rs = []
        for j in range(y_va.shape[1]):
            a, b = pred[:, j], y_va[:, j]
            if a.std() == 0 or b.std() == 0:
                rs.append(float("nan"))
            else:
                rs.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    return rs, best_loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--cache-dir",
        default="/home/fansonglin/data_space/DCCA-GS/semantic_cache/4-28",
    )
    p.add_argument("--signals", default="dino,depth,sam2")
    p.add_argument("--k-list", default="8,16,32")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fit-steps", type=int, default=400)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--pca-dims", type=int, default=8)
    p.add_argument(
        "--shuffle-split",
        action="store_true",
        help="Stage A.5: random 80/20 anchor split instead of contiguous "
        "Morton split (guards against spatial-autocorrelation inflation).",
    )
    p.add_argument(
        "--export-targets",
        default=None,
        help="Also export the full-anchor-space DINO PCA target to an npz "
        "for Stage-B training (T-A/T-A2).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        default="/home/fansonglin/data_space/DCCA-GS/semantic_gate/results.json",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    signals = [s.strip() for s in args.signals.split(",")]

    model, ckpt = load_checkpoint_legacy8(args.ckpt, device)
    model.eval()
    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=1,
        max_width=1600,
        test_every=8,
        white_background=False,
        preload_images=False,
        cache_images_cpu=False,
        device="cuda",
    )
    g, N = _collect_inputs(model, device)
    targets, full_targets = _aggregate_semantics(
        model, dataset, args.cache_dir, signals, device, N
    )

    if args.export_targets and "dino" in full_targets:
        vals, covf = full_targets["dino"]
        cv = vals[covf]
        z = (cv - cv.mean(0)) / cv.std(0).clamp_min(1e-6)
        _, _, v = torch.pca_lowrank(z, q=args.pca_dims, center=False)
        red = z @ v[:, : args.pca_dims]
        tgt = torch.zeros(vals.shape[0], args.pca_dims, device=device)
        tgt[covf] = red
        np.savez(
            args.export_targets,
            target=tgt.cpu().numpy(),
            cov=covf.cpu().numpy(),
        )
        print(f"[semantic_gate] exported targets -> {args.export_targets}")

    covered = None
    target_meta = {}
    prepared = {}
    for sig, (vals, cov) in targets.items():
        covered = cov if covered is None else covered
        cv = vals[cov]
        n_cov = int(cov.sum().item())
        stds = cv.std(0).float()
        print(
            f"[semantic_gate] {sig}: covered={n_cov}/{N} "
            f"dim={vals.shape[1]} std={stds.cpu().tolist()[:8]}",
            flush=True,
        )
        target_meta[sig] = {
            "coverage": round(n_cov / N, 4),
            "n_covered": n_cov,
            "std": [round(float(s), 5) for s in stds.cpu().tolist()],
            "dim": int(vals.shape[1]),
        }
        if sig == "dino":
            red = _pca_reduce(vals[cov], args.pca_dims)
            if not torch.isfinite(red).all():
                print(f"[semantic_gate] {sig} PCA produced non-finite values; skip")
                continue
            prepared[sig] = red
            target_meta[sig]["dim"] = int(red.shape[1])
            target_meta[sig]["std"] = [
                round(float(s), 5) for s in red.std(0).cpu().tolist()
            ]
        else:
            if not torch.isfinite(cv).all() or cv.std(0).clamp_min(1e-6).min() < 1e-9:
                print(f"[semantic_gate] {sig} target degenerate; skip")
                continue
            z = (vals[cov] - vals[cov].mean(0)) / vals[cov].std(0).clamp_min(1e-6)
            prepared[sig] = z

    # Align inputs to covered anchors and split train/val.
    inputs = {
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
    for sig, y in prepared.items():
        n_tr = int(y.shape[0] * (1.0 - args.val_fraction))
        if args.shuffle_split:
            gen = torch.Generator(device=device).manual_seed(args.seed)
            perm = torch.randperm(y.shape[0], generator=gen, device=device)
            tr_idx = perm[:n_tr]
            va_idx = perm[n_tr:]
        else:
            tr_idx = slice(0, n_tr)
            va_idx = slice(n_tr, y.shape[0])
        for name, x in inputs.items():
            xc = x[covered]
            rs, loss = _fit_corr(
                xc[tr_idx], y[tr_idx], xc[va_idx], y[va_idx],
                args.hidden, args.fit_steps, args.weight_decay, device,
            )
            rows[f"{sig}|{name}"] = {
                "corr_per_dim": rs,
                "corr_best": max((v for v in rs if not math.isnan(v)), default=float("nan")),
                "corr_mean": sum(v for v in rs if not math.isnan(v)) / max(
                    sum(1 for v in rs if not math.isnan(v)), 1
                ),
                "mse": round(loss, 6),
            }
            print(f"{sig}|{name}: best_r={rows[f'{sig}|{name}']['corr_best']:.4f}",
                  flush=True)
        for k in [int(v) for v in args.k_list.split(",")]:
            mean_acc = torch.zeros_like(neighbor)
            for d in range(1, k + 1):
                prev = torch.zeros_like(neighbor)
                prev[d:] = neighbor[:-d]
                mean_acc += prev
            mean_acc /= float(k)
            xc = torch.cat([inputs["B"], mean_acc], dim=-1)[covered]
            rs, loss = _fit_corr(
                xc[tr_idx], y[tr_idx], xc[va_idx], y[va_idx],
                args.hidden, args.fit_steps, args.weight_decay, device,
            )
            rows[f"{sig}|C_k{k}"] = {
                "corr_per_dim": rs,
                "corr_best": max((v for v in rs if not math.isnan(v)), default=float("nan")),
                "corr_mean": sum(v for v in rs if not math.isnan(v)) / max(
                    sum(1 for v in rs if not math.isnan(v)), 1
                ),
                "mse": round(loss, 6),
            }
            print(f"{sig}|C_k{k}: best_r={rows[f'{sig}|C_k{k}']['corr_best']:.4f}",
                  flush=True)

    best_per_signal = {
        sig: max(
            (
                rows[f"{sig}|{name}"]["corr_best"]
                for name in list(inputs)
                + [f"C_k{k}" for k in [int(v) for v in args.k_list.split(",")]]
                if not math.isnan(rows[f"{sig}|{name}"]["corr_best"])
            ),
            default=float("nan"),
        )
        for sig in prepared
    }
    best_per_signal = {
        sig: r for sig, r in best_per_signal.items() if not math.isnan(r)
    }
    overall_best = max(best_per_signal.values(), default=float("nan"))
    decision = (
        "pass" if overall_best > 0.3
        else "keep-best" if overall_best >= 0.15
        else "close"
    )
    summary = {
        "ckpt": args.ckpt,
        "N_anchors": int(N),
        "signals": signals,
        "target_meta": target_meta,
        "rows": rows,
        "best_per_signal": best_per_signal,
        "overall_best_r": round(float(overall_best), 4),
        "decision": decision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    out.write_text(json.dumps(_clean(summary), indent=2, sort_keys=True))
    print(json.dumps(_clean(summary), indent=2, sort_keys=True))
    print("[semantic_gate] DECISION:", decision.upper())


if __name__ == "__main__":
    main()
