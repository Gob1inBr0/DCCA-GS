"""I6 route-B experiment: quantized per-anchor sensitivity side info + RD.

The decoder cannot recompute the sensitivity EMA, so route B transmits the
per-anchor multipliers themselves. This script:

1. accumulates the I6 sensitivity EMA on a checkpoint;
2. builds continuous multipliers ``1 + strength*tanh(-z_rel)`` per field;
3. percentile-quantizes them to N levels (N in {2,4,8} -> 1/2/3 bit);
4. estimates side-info cost three ways: raw bits, per-field empirical entropy,
   and a Morton prev-k residual (context) entropy;
5. encodes/decodes/evaluates with the multipliers as ``q_override_*``
   (continuous = upper bound, quantized = practical) and reports
   ``total_MB = codec total + side info`` with PSNR/SSIM/LPIPS.

Kill conditions (design): 2-bit net BD-rate < 2% -> close; 1-bit PSNR drop
> 0.3 dB vs the continuous upper bound -> quantization too coarse.

Usage (5090, HAC_5090_a100 env):

    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/sensitivity_side_info.py \
      --ckpt runs/4-28_i6_90k_h32/ckpts/ckpt_90000.pth \
      --data-dir <4-28> --steps 200 --out runs/sensitivity_side_info.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec, anchor_codec_order
from scaffold_gs.trainer import evaluate, load_checkpoint

from sens_replace_gate import _collect_sensitivity


def _percentile_quantize(mult, n_levels):
    """Percentile binning -> (quantized_mult, bin_index, representatives)."""
    edges = torch.quantile(
        mult, torch.linspace(0.0, 1.0, n_levels + 1, device=mult.device)[1:-1]
    )
    idx = torch.searchsorted(edges, mult)
    reps = torch.empty(n_levels, device=mult.device)
    for i in range(n_levels):
        sel = mult[idx == i]
        if sel.numel() > 0:
            reps[i] = sel.mean()
        else:
            lo = edges[i - 1] if i > 0 else mult.min()
            hi = edges[i] if i < n_levels - 1 else mult.max()
            reps[i] = (lo + hi) / 2.0
    return reps[idx], idx


def _hist_entropy(idx, n_levels):
    counts = torch.bincount(idx.long(), minlength=n_levels).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def _ctx_entropy(idx, k):
    """Empirical entropy of the residual vs the mean of the previous k."""
    n = idx.shape[0]
    idx = idx.float()
    pred = torch.zeros(n, device=idx.device)
    for d in range(1, k + 1):
        prev = torch.zeros(n, device=idx.device)
        prev[d:] = idx[:-d]
        pred += prev
    pred /= float(k)
    res = (idx - pred.round())[k:]
    lo = int(res.min().item())
    hi = int(res.max().item())
    counts = torch.bincount((res - lo).long(), minlength=hi - lo + 1).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum()), n - k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--strengths", default="0.5,1.0")
    p.add_argument("--n-list", default="2,4,8")
    p.add_argument("--ctx-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/sensitivity_side_info.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, optim_cfg, iteration, _ = load_checkpoint(args.ckpt, "cuda")
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
    print("[side_info] accumulating sensitivity EMA ...", flush=True)
    _collect_sensitivity(model, dataset, args.steps)

    core = model.core
    order = anchor_codec_order(model)
    N = order.shape[0]
    fields = ["feat", "scaling", "offsets"]
    ema = torch.stack(
        [
            core.sensitivity_feat[order].squeeze(-1),
            core.sensitivity_scaling[order].squeeze(-1),
            core.sensitivity_offsets[order].squeeze(-1),
        ],
        dim=-1,
    )
    z = (ema - core.sensitivity_mean) / core.sensitivity_mean.clamp_min(1e-12)

    codec = HACPlusCodec()
    rows = {}
    base_dir = Path(args.out).parent / "base"
    meta_base = codec.encode(model, base_dir)
    dec_base = codec.decode(base_dir)
    m_base = evaluate(dec_base, dataset, base_dir / "eval", iteration)
    rows["baseline_formula"] = {
        "total_MB": meta_base["total_MB"],
        "psnr": round(m_base["psnr"], 4),
        "ssim": round(m_base["ssim"], 4),
        "lpips": round(m_base["lpips"], 4),
    }
    print("baseline:", rows["baseline_formula"], flush=True)
    del dec_base
    torch.cuda.empty_cache()

    for strength in [float(v) for v in args.strengths.split(",")]:
        mult = 1.0 + strength * torch.tanh(-z)
        tag = f"s{strength:g}"
        true_dir = Path(args.out).parent / f"{tag}_true"
        true_dir.mkdir(parents=True, exist_ok=True)
        ov = {
            f"q_override_{f}": str(true_dir / f"q_override_{f}.npy")
            for f in fields
        }
        for j, f in enumerate(fields):
            np.save(
                true_dir / f"q_override_{f}.npy",
                mult[:, j].cpu().numpy().astype(np.float32)[:, None],
            )
        meta = codec.encode(model, true_dir, **ov)
        dec = codec.decode(true_dir, **ov)
        m = evaluate(dec, dataset, true_dir / "eval", iteration)
        rows[f"{tag}_true"] = {
            "total_MB": meta["total_MB"],
            "psnr": round(m["psnr"], 4),
            "ssim": round(m["ssim"], 4),
            "lpips": round(m["lpips"], 4),
            "side_info_MB": 0.0,
        }
        print(f"{tag}_true:", rows[f"{tag}_true"], flush=True)
        del dec
        torch.cuda.empty_cache()

        for n in [int(v) for v in args.n_list.split(",")]:
            quant = []
            idxs = []
            for j in range(3):
                q, ix = _percentile_quantize(mult[:, j], n)
                quant.append(q)
                idxs.append(ix)
            bits_raw = N * 3 * int(math.ceil(math.log2(n)))
            bits_ent = sum(
                N * _hist_entropy(ix, n) for ix in idxs
            )
            bits_ctx = 0.0
            for ix in idxs:
                h, cnt = _ctx_entropy(ix, args.ctx_k)
                bits_ctx += h * cnt + args.ctx_k * int(math.ceil(math.log2(n)))
            out_dir = Path(args.out).parent / f"{tag}_N{n}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ov = {}
            for j, f in enumerate(fields):
                path = out_dir / f"q_override_{f}.npy"
                np.save(path, quant[j].cpu().numpy().astype(np.float32)[:, None])
                ov[f"q_override_{f}"] = str(path)
            meta = codec.encode(model, out_dir, **ov)
            dec = codec.decode(out_dir, **ov)
            m = evaluate(dec, dataset, out_dir / "eval", iteration)
            rows[f"{tag}_N{n}"] = {
                "codec_MB": meta["total_MB"],
                "side_raw_MB": round(bits_raw / 8 / 1024 / 1024, 5),
                "side_entropy_MB": round(bits_ent / 8 / 1024 / 1024, 5),
                "side_ctx_MB": round(bits_ctx / 8 / 1024 / 1024, 5),
                "total_raw_MB": round(
                    meta["total_MB"] + bits_raw / 8 / 1024 / 1024, 4
                ),
                "total_entropy_MB": round(
                    meta["total_MB"] + bits_ent / 8 / 1024 / 1024, 4
                ),
                "total_ctx_MB": round(
                    meta["total_MB"] + bits_ctx / 8 / 1024 / 1024, 4
                ),
                "psnr": round(m["psnr"], 4),
                "ssim": round(m["ssim"], 4),
                "lpips": round(m["lpips"], 4),
            }
            print(f"{tag}_N{n}:", rows[f"{tag}_N{n}"], flush=True)
            del dec
            torch.cuda.empty_cache()

    summary = {"iteration": int(iteration), "N_anchors": int(N), "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
