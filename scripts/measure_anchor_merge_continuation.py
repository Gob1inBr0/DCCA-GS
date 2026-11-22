"""Continue matched room checkpoints beyond 30k and independently decode them."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.config import DataConfig, ModelConfig, OptimConfig, TrainConfig
from scaffold_gs.trainer import run_training
from scripts.compare_anchor_merge_response import audit_payload
from scripts.run_in_training_anchor_merge import MERGE_STEP, SEED, measure


START = 30_000
DEFAULT_END = 35_000
DEFAULT_INTERVAL = 1_000


def checkpoint(root, step):
    return root / "train" / "ckpts" / f"ckpt_{step}.pth"


def train(source, previous_schedule, root, end_step=DEFAULT_END, eval_interval=DEFAULT_INTERVAL):
    if end_step <= START or eval_interval <= 0:
        raise ValueError("Continuation must end after 30k with a positive evaluation interval")
    steps = list(range(START + eval_interval, end_step, eval_interval)) + [end_step]
    archive = torch.load(source, map_location="cpu", weights_only=False)
    merge_step = archive["core_state"].get("anchor_merge_step")
    if archive["iteration"] != START or merge_step not in (None, MERGE_STEP):
        raise ValueError("Expected a 30k control or completed in-training merge checkpoint")
    if archive["optimizer_state"] is None:
        raise ValueError("Continuation requires the original optimizer state")
    model = ModelConfig(**archive["model_config"])
    original_optim = OptimConfig(**archive["optim_config"])
    data = replace(DataConfig(**archive["data_config"]), result_dir=str(root / "train"))
    # Reproduce the exact schedule prefix, then draw the next training cameras
    # from the same generator. The original room training split has 272 views.
    original_cameras = np.load(previous_schedule, allow_pickle=False)
    schedule = np.random.default_rng(SEED).integers(0, 272, size=end_step)
    if original_cameras.shape != (START,) or not np.array_equal(schedule[:START], original_cameras):
        raise ValueError("The extended camera sequence does not reproduce the original prefix")
    root.mkdir(parents=True, exist_ok=True)
    schedule_path = root / "camera_schedule.npy"
    np.save(schedule_path, schedule, allow_pickle=False)

    info = {
        "iteration": START,
        "source_checkpoint": str(source),
        "end_step": end_step,
        "checkpoint_steps": steps,
        "anchor_merge_step": merge_step,
        "learning_rate_policy": "retain original schedule lengths and terminal rates",
        "lambda_rate": original_optim.lambda_rate,
        "feature_dim": model.feat_dim,
        "update_until": original_optim.update_until,
        "data_dir": data.data_dir,
        "learning_rates": {
            group["name"]: group["lr"] for group in archive["optimizer_state"]["param_groups"]
        },
        "schedule_steps": {
            name: value for name, value in vars(original_optim).items()
            if name.endswith("_lr_max_steps")
        },
    }
    (root / "resume_config.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[Continuation] checkpoint and schedule verified: {json.dumps(info)}", flush=True)
    optim = replace(
        original_optim, max_steps=end_step, save_steps=steps,
        eval_steps=[],
    )
    del archive
    run_training(TrainConfig(
        model=model, data=data, optim=optim, seed=SEED, resume=str(source),
        camera_schedule=str(schedule_path), anchor_merge_step=merge_step,
        anchor_merge_response_rms_limit=0.10,
    ))


def decode(root, step):
    path = checkpoint(root, step)
    archive = torch.load(path, map_location="cpu", weights_only=False)
    assert archive["iteration"] == step
    assert archive["core_state"].get("anchor_merge_step") in (None, MERGE_STEP)
    data = DataConfig(**archive["data_config"])
    rates = {group["name"]: group["lr"] for group in archive["optimizer_state"]["param_groups"]}
    del archive
    out = root / str(step)
    out.mkdir()
    measure(path, data, out)
    payload = audit_payload(out / "bitstreams", out / "payload")
    metrics = json.loads((out / "decoded_eval/metrics.jsonl").read_text().splitlines()[-1])
    files = {path.name: path.stat().st_size for path in (out / "payload").iterdir()}
    result = {
        **metrics, **payload, "files": files, "learning_rates": rates,
        "meta": json.loads((out / "bitstreams/hac_meta.json").read_text()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[ContinuationDecode @{step}] PSNR {metrics['psnr']:.4f}, {payload['payload_bytes']} bytes",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("source", type=Path)
    train_parser.add_argument("previous_schedule", type=Path)
    train_parser.add_argument("--end-step", type=int, default=DEFAULT_END)
    train_parser.add_argument("--eval-interval", type=int, default=DEFAULT_INTERVAL)
    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("step", type=int)
    args = parser.parse_args()
    if args.stage == "train":
        train(args.source, args.previous_schedule, args.root, args.end_step, args.eval_interval)
    else:
        decode(args.root, args.step)


if __name__ == "__main__":
    main()
