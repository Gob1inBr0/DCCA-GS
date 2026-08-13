"""Sweep the I2 complexity MLP architecture (hidden width x layers).

For every ``hidden:layers`` config the script runs the full pipeline
(train -> compress -> decode-eval) and writes a JSON report with
PSNR/SSIM/LPIPS/total_MB/anchors per config.

Usage (5090, HAC_5090_a100 env):

    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$PWD PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
    python scripts/sweep_mlp_complexity.py \
      --data-dir /home/fansonglin/data_space/web_scan/第一组数据 \
      --result-root runs/mlp_sweep \
      --max-steps 12000 --update-until 6000 \
      --configs "25:1;32:1;64:1;64:2;128:1" --sensitivity
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], log_path: Path) -> int:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def _parse_eval(log: Path):
    text = log.read_text(errors="ignore")
    m = re.search(
        r"\[Eval @\d+\] PSNR ([\d.]+) \| SSIM ([\d.]+) \| LPIPS ([\d.]+)", text
    )
    if not m:
        raise RuntimeError(f"no eval metrics in {log}")
    return {
        "psnr": float(m.group(1)),
        "ssim": float(m.group(2)),
        "lpips": float(m.group(3)),
    }


def _parse_meta(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    return {
        "total_MB": float(meta["total_MB"]),
        "num_anchors": int(meta["num_anchors"]),
        "num_anchors_total": int(meta["num_anchors_total"]),
    }


def run_one(hidden: int, layers: int, args, root: Path, results, lock) -> None:
    tag = f"h{hidden}_l{layers}"
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.log"
    row = {"tag": tag, "hidden": hidden, "layers": layers}

    train_cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "train",
        "--cfg.model.model-name", "hac_pp",
        "--cfg.data.data-dir", args.data_dir,
        "--cfg.data.result-dir", str(out_dir),
        "--cfg.data.data-factor", str(args.data_factor),
        "--cfg.data.max-width", str(args.max_width),
        "--cfg.data.test-every", "8",
        "--cfg.data.no-preload-images",
        "--cfg.model.voxel-size", "0.001",
        "--cfg.model.feat-dim", "50",
        "--cfg.model.n-offsets", "10",
        "--cfg.model.appearance-dim", "0",
        "--cfg.model.ratio", "1",
        "--cfg.model.tile-size", str(args.tile_size),
        "--cfg.model.mlp-complexity-hidden", str(hidden),
        "--cfg.model.mlp-complexity-layers", str(layers),
        "--cfg.model.content-aware-start-iter", str(args.content_aware_start),
        "--cfg.model.content-aware-ramp-iters", str(args.content_aware_ramp),
        "--cfg.optim.max-steps", str(args.max_steps),
        "--cfg.optim.eval-steps", str(args.max_steps),
        "--cfg.optim.save-steps", str(args.max_steps),
        "--cfg.optim.lambda-rate", "0.004",
        "--cfg.optim.mask-lr-final", "0.002",
        "--cfg.optim.start-stat", "500",
        "--cfg.optim.update-from", "1500",
        "--cfg.optim.update-until", str(args.update_until),
        "--cfg.optim.update-interval", "100",
        "--cfg.device", "cuda",
    ]
    if args.sensitivity:
        train_cmd += [
            "--cfg.model.sensitivity-enabled",
            "--cfg.model.sensitivity-start-iter", str(args.sensitivity_start),
            "--cfg.model.sensitivity-weight", "0.001",
        ]

    print(f"=== {tag}: start hidden={hidden} layers={layers} ===", flush=True)
    rc = _run(train_cmd, log)
    train_ok = rc == 0 and "Training finished" in log.read_text(errors="ignore")
    if not train_ok:
        row["error"] = "train_failed"
        print(f"  {tag} TRAIN FAILED (rc={rc})", flush=True)
    else:
        ckpt = out_dir / "ckpts" / f"ckpt_{args.max_steps}.pth"
        rc = _run(
            [
                sys.executable, str(REPO / "train.py"), "compress",
                "--cfg.ckpt", str(ckpt),
                "--cfg.out-dir", str(out_dir / "bitstreams"),
                "--cfg.codec", "hac_pp",
            ],
            out_dir / "compress.log",
        )
        if rc != 0:
            row["error"] = "compress_failed"
        else:
            rc = _run(
                [
                    sys.executable, str(REPO / "scripts" / "eval_decoded.py"),
                    "--artifact-dir", str(out_dir / "bitstreams"),
                    "--data-dir", args.data_dir,
                    "--result-dir", str(out_dir / "decoded_eval"),
                    "--max-width", str(args.max_width),
                    "--no-preload-images",
                ],
                out_dir / "eval.log",
            )
            if rc != 0:
                row["error"] = "eval_failed"
            else:
                row.update(_parse_eval(out_dir / "eval.log"))
                row.update(_parse_meta(out_dir / "bitstreams" / "hac_meta.json"))

    with lock:
        results.append(row)
        print(
            f"  {tag}: "
            + (
                row.get("error", "")
                or (
                    f"PSNR {row['psnr']:.3f} | SSIM {row['ssim']:.4f} | "
                    f"LPIPS {row['lpips']:.4f} | {row['total_MB']:.4f}MB "
                    f"| anchors {row['num_anchors']}"
                )
            ),
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-root", default="runs/mlp_sweep")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=12_000)
    p.add_argument("--update-until", type=int, default=6_000)
    p.add_argument("--data-factor", type=int, default=1)
    p.add_argument("--max-width", type=int, default=1600)
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--content-aware-start", type=int, default=500)
    p.add_argument("--content-aware-ramp", type=int, default=300)
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--sensitivity-start", type=int, default=600)
    p.add_argument(
        "--configs",
        default="25:1;32:1;64:1;64:2;128:1",
        help="Semicolon-separated hidden:layers pairs, e.g. '25:1;64:2'.",
    )
    p.add_argument("--parallel", type=int, default=1, help="Configs in parallel.")
    args = p.parse_args()

    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(";")]
    base_env = dict(os.environ)

    summary = {"configs": [], "best": None}
    results: list = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(run_one, hidden, layers, args, root, results, lock)
            for hidden, layers in configs
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # keep other configs alive
                print(f"config worker failed: {exc}", flush=True)

    summary["configs"] = sorted(results, key=lambda c: c["tag"])

    ok = [c for c in summary["configs"] if "error" not in c]
    if ok:
        # Rank by PSNR, breaking ties by smaller bitstream.
        best = max(ok, key=lambda c: (c["psnr"], -c["total_MB"]))
        summary["best"] = best["tag"]

    with open(root / "mlp_sweep.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("\n=== SUMMARY ===")
    for c in summary["configs"]:
        if "error" in c:
            print(f"{c['tag']}: {c['error']}")
        else:
            print(
                f"{c['tag']}: PSNR {c['psnr']:.3f} | SSIM {c['ssim']:.4f} | "
                f"LPIPS {c['lpips']:.4f} | {c['total_MB']:.4f}MB"
            )
    print("BEST:", summary.get("best"))


if __name__ == "__main__":
    main()
