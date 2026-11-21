"""Prepare matched DCCA-GS checkpoints for a new-position anchor merge experiment."""

import argparse
import json
from pathlib import Path

import torch

from scaffold_gs.anchor_merge import plan_anchor_merge, select_camera_centers
from scaffold_gs.datasets import ColmapDataset


def remap_buffers(value, rows, anchors, capacity):
    if isinstance(value, dict):
        return {key: remap_buffers(tensor, rows, anchors, capacity)
                for key, tensor in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim:
        if value.shape[0] == anchors:
            return value[rows].clone()
        if value.shape[0] == anchors * capacity:
            shaped = value.reshape(anchors, capacity, *value.shape[1:])
            return shaped[rows].reshape(-1, *value.shape[1:]).clone()
    return value


def merge_checkpoint(checkpoint, meta, camera_centers):
    state = checkpoint["model_state"]
    arrays, rows, _, report = plan_anchor_merge(
        state, checkpoint["voxel_size"], meta, camera_centers
    )
    if not report["merges"]:
        raise RuntimeError("No qualifying anchor pairs in the checkpoint")
    for key, array in arrays.items():
        state[key] = torch.from_numpy(array[rows])
    old_count, capacity = arrays["_offset"].shape[:2]
    for key in ("_spa_state", "_mini_splat_state", "_sensitivity_state"):
        if key in state:
            state[key] = remap_buffers(state[key], rows, old_count, capacity)
    checkpoint["stats"] = remap_buffers(checkpoint["stats"], rows, old_count, capacity)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.reference, map_location="cpu", weights_only=False)
    checkpoint["optimizer_state"] = None
    checkpoint["model_config"].update(spa_enabled=False, mini_splat_enabled=False)
    checkpoint["data_config"]["data_dir"] = str(args.data)
    data = checkpoint["data_config"]
    dataset = ColmapDataset(data_dir=data["data_dir"], data_factor=data["data_factor"],
                            test_every=data["test_every"], max_width=data["max_width"],
                            preload_images=False, device="cpu")
    centers, camera_names = select_camera_centers(dataset.train_cameras)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output / "control.pth")
    report = merge_checkpoint(checkpoint, json.loads(args.metadata.read_text()), centers)
    report["selection_cameras"] = camera_names
    report["reference"] = str(args.reference)
    torch.save(checkpoint, args.output / "merged.pth")
    (args.output / "merge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
