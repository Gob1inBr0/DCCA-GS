"""Train matched room arms with a new-position anchor merge inside 30k steps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scaffold_gs.config import DataConfig, ModelConfig, OptimConfig, TrainConfig
from scaffold_gs.anchor_merge import EXPECTED_SAVING_BYTES
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.trainer import evaluate, load_checkpoint, run_training

MERGE_STEP = 18_001
END_STEP = 30_000
SEED = 42


def run_arm(archive, data, root, arm, resume, max_steps, merge_step, schedule, *,
            response_rms_limit=0.05, save_merge_initial=False,
            min_saving_bytes=EXPECTED_SAVING_BYTES, candidate_neighbors=None):
    result_dir = root / arm
    result_dir.mkdir(exist_ok=False)
    arm_data = DataConfig(**{**vars(data), "result_dir": str(result_dir)})
    model = ModelConfig(**archive["model_config"])
    optim = OptimConfig(**archive["optim_config"])
    optim.max_steps = max_steps
    optim.eval_steps = [max_steps]
    optim.save_steps = [max_steps]
    cfg = TrainConfig(
        model=model, data=arm_data, optim=optim, seed=SEED,
        resume=str(resume) if resume else None, anchor_merge_step=merge_step,
        anchor_merge_response_rms_limit=response_rms_limit,
        anchor_merge_min_saving_bytes=min_saving_bytes,
        anchor_merge_candidate_neighbors=candidate_neighbors,
        save_anchor_merge_initial=save_merge_initial,
        camera_schedule=str(schedule) if schedule else None,
    )
    run_training(cfg)
    final = result_dir / "ckpts" / f"ckpt_{max_steps}.pth"
    if merge_step is not None:
        report = json.loads((result_dir / f"anchor_merge_{merge_step}.json").read_text())
        if not report["merges"]:
            print(f"[Experiment] No eligible merges at step {merge_step}", flush=True)
    return final


def measure(path, data, result_dir):
    bitstream = result_dir / "bitstreams"
    codec = HACPlusCodec()
    original, _, step, _ = load_checkpoint(path, "cuda")
    codec.encode(original, bitstream)
    del original
    decoded = codec.decode(bitstream)
    dataset = ColmapDataset(
        data_dir=data.data_dir, data_factor=data.data_factor,
        max_width=data.max_width, test_every=data.test_every,
        white_background=data.white_background, preload_images=False,
    )
    evaluate(decoded, dataset, result_dir / "decoded_eval", step)


def smoke_reference(archive, root):
    # Use a spatial subset of a known checkpoint to exercise the real codec
    # and subsequent CUDA optimizer steps at low cost.
    state = archive["model_state"]
    xyz = state["_anchor"]
    original_rows = len(xyz)
    offsets = state["_offset"].shape[1]
    distance = (xyz - xyz.median(dim=0).values).square().sum(dim=-1)
    rows = distance.argsort()[:4096].sort().values

    def slice_rows(obj):
        if isinstance(obj, dict):
            return {key: slice_rows(value) for key, value in obj.items()}
        if isinstance(obj, torch.Tensor) and obj.ndim:
            if obj.shape[0] == original_rows:
                return obj[rows].clone()
            if obj.shape[0] == original_rows * offsets:
                return obj.reshape(original_rows, offsets, *obj.shape[1:])[rows].flatten(0, 1)
        return obj

    trimmed = dict(archive)
    trimmed["model_state"] = slice_rows(state)
    trimmed["stats"] = slice_rows(archive["stats"])
    if archive.get("optimizer_state") is not None:
        trimmed["optimizer_state"] = {
            **archive["optimizer_state"],
            "state": slice_rows(archive["optimizer_state"]["state"]),
        }
    path = root / "smoke_reference.pth"
    torch.save(trimmed, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "common", "control", "merged"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    archive = torch.load(args.reference, map_location="cpu", weights_only=False)
    data = DataConfig(**{**archive["data_config"], "data_dir": str(args.data),
                        "preload_images": False})
    root = args.root
    stage = args.stage
    if stage == "smoke":
        resume = smoke_reference(archive, root)
        start = int(archive["iteration"])
        end = start + 400
        final = run_arm(archive, data, root, stage, resume, end, start + 1, None)
        report = json.loads((root / stage / f"anchor_merge_{start + 1}.json").read_text())
        if not report["merges"]:
            raise RuntimeError("Smoke scene had no qualifying pairs to exercise the optimizer update")
        measure(final, data, root / stage)
        return

    schedule = root / "camera_schedule.npy"
    if stage == "common":
        dataset = ColmapDataset(
            data_dir=data.data_dir, data_factor=data.data_factor, max_width=data.max_width,
            test_every=data.test_every, white_background=data.white_background,
            preload_images=False, device="cpu",
        )
        rng = np.random.default_rng(SEED)
        np.save(schedule, rng.integers(0, len(dataset.train_cameras), size=END_STEP))
        run_arm(archive, data, root, stage, None, MERGE_STEP - 1, None, schedule)
        return

    resume = root / "common" / "ckpts" / f"ckpt_{MERGE_STEP - 1}.pth"
    merge = MERGE_STEP if stage == "merged" else None
    final = run_arm(archive, data, root, stage, resume, END_STEP, merge, schedule)
    measure(final, data, root / stage)


if __name__ == "__main__":
    main()
