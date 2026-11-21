"""Measure a response tolerance sweep from one DCCA-GS training checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from hacplus.utils.encodings_cuda import decoder
from scaffold_gs.anchor_merge import merge_training_anchors
from scaffold_gs.config import DataConfig, ModelConfig
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec, HACPlusModel
from scaffold_gs.trainer import load_checkpoint, save_checkpoint
from scripts.run_in_training_anchor_merge import END_STEP, MERGE_STEP, measure, run_arm

LIMITS = {"r0625": 0.0625, "r075": 0.075, "r0875": 0.0875, "r100": 0.1}


def make_initial(shared, root, data, group):
    out = root / group
    out.mkdir()
    if group == "control_initial":
        measure(shared, data, out)
        return

    original, optim, step, _ = load_checkpoint(shared, "cuda")
    original.create_optimizer(optim)
    dataset = ColmapDataset(
        data_dir=data.data_dir, data_factor=data.data_factor,
        max_width=data.max_width, test_every=data.test_every,
        white_background=data.white_background, preload_images=False,
    )
    report = merge_training_anchors(original, dataset.train_cameras, out)
    report["iteration"] = step + 1
    original.core.anchor_merge_step = step + 1
    (out / f"anchor_merge_{step + 1}.json").write_text(json.dumps(report, indent=2) + "\n")
    initial = out / "after_anchor_merge.pth"
    save_checkpoint(original, optim, data, step, initial)
    del original, dataset
    measure(initial, data, out)


def audit_payload(source, dest):
    dest.mkdir()
    archive = torch.load(source / "attributes.pth", map_location="cpu", weights_only=False)
    capsule = {name: archive[name] for name in ("config", "voxel_size", "spatial_lr_scale")}
    capsule["networks"] = {name: state for name, state in archive["decoder"].items()
                           if name != "encoding_xyz"}
    del archive
    side_files = ("codec_header.json", "content_aware_q_meta.json",
                  "xyz_gpcc.npz", "x_bound_min.pkl", "x_bound_max.pkl")
    package = dest / "decoder_config_and_weights.pth"
    torch.save(capsule, package)
    for path in source.iterdir():
        if path.suffix == ".b" or path.name in side_files:
            (dest / path.name).symlink_to(path.resolve())

    capsule = torch.load(package, map_location="cpu", weights_only=False)
    model = HACPlusModel(ModelConfig(**capsule["config"]), "cuda")
    model.voxel_size = capsule["voxel_size"]
    model.spatial_lr_scale = capsule["spatial_lr_scale"]
    model._view.load_decoder_state({
        **capsule["networks"], "encoding_xyz": model.core.encoding_xyz.state_dict(),
    })
    # The production decoder predicts attributes before installing the binary grid.
    # Set up its context exclusively from the received binary grid up front.
    hash_bits = decoder(model.core.get_encoding_params().numel(), str(dest / "hash.b"), "cuda")
    model._view.set_hash_params((2 * hash_bits.float() - 1).view(
        -1, model.core.n_features_per_level
    ))
    model.decode_attributes(dest)
    standard = HACPlusCodec().decode(source)
    for field in ("_anchor", "_offset", "_anchor_feat", "_mask", "_scaling"):
        if not torch.equal(getattr(model.core, field), getattr(standard.core, field)):
            raise RuntimeError(f"Compact payload changed {field}: {source}")
    if not torch.equal(model.core.get_encoding_params(), standard.core.get_encoding_params()):
        raise RuntimeError(f"Compact payload changed the hash grid: {source}")
    return {
        "payload_bytes": sum(path.stat().st_size for path in dest.iterdir()),
        "active_offsets": int(model.core.get_mask.sum().item()),
        "coded_anchors": model.num_anchors,
        "matches_standard": True,
    }


def collect(original_root, root):
    decoded = {
        "control": original_root / "control",
        "r050": original_root / "merged",
        "control_initial": root / "control_initial",
        "r050_initial": root / "r050_initial",
    }
    for arm in LIMITS:
        decoded[arm] = root / arm
        decoded[f"{arm}_initial"] = root / f"{arm}_initial"

    original = json.loads((original_root / "merged" / f"anchor_merge_{MERGE_STEP}.json").read_text())
    replay = json.loads((root / "r050_initial" / f"anchor_merge_{MERGE_STEP}.json").read_text())
    exact_fields = ("initial_coded_anchors", "final_coded_anchors", "merges",
                    "active_offsets", "selection_cameras")
    if any(original[key] != replay[key] for key in exact_fields):
        raise RuntimeError("0.05 replay differs from the original arm")
    round_counts = lambda report: [(row["candidates"], row["accepted"]) for row in report["rounds"]]
    if round_counts(original) != round_counts(replay):
        raise RuntimeError("0.05 replay differs in the per-round counts")
    for key in ("max_position_error_before_quantization", "mean_response_rms",
                "mean_estimated_saving_bytes"):
        if abs(original[key] - replay[key]) > 1e-8:
            raise RuntimeError(f"0.05 replay differs in {key}")

    payload_dir = root / "payload"
    payload_dir.mkdir()
    results = {}
    for arm, path in decoded.items():
        metrics = json.loads((path / "decoded_eval" / "metrics.jsonl").read_text().splitlines()[-1])
        payload = audit_payload(path / "bitstreams", payload_dir / arm)
        results[arm] = {**metrics, **payload}
    merges = {"r050": replay}
    for arm in LIMITS:
        merges[arm] = json.loads((root / arm / f"anchor_merge_{MERGE_STEP}.json").read_text())
    (root / "results.json").write_text(json.dumps({"decoded": results, "merge": merges}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("control_initial", "r050_initial", "collect", *LIMITS))
    parser.add_argument("original_root", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    original_root = args.original_root
    root = args.root
    if args.stage == "collect":
        collect(original_root, root)
        return
    shared = original_root / "common" / "ckpts" / f"ckpt_{MERGE_STEP - 1}.pth"
    archive = torch.load(shared, map_location="cpu", weights_only=False)
    data = DataConfig(**{**archive["data_config"], "data_dir": str(args.data),
                        "preload_images": False})
    if args.stage in ("control_initial", "r050_initial"):
        make_initial(shared, root, data, args.stage)
        return
    group = args.stage
    final = run_arm(archive, data, root, group, shared, END_STEP, MERGE_STEP,
                    original_root / "camera_schedule.npy", response_rms_limit=LIMITS[group],
                    save_merge_initial=True)
    initial_dir = root / f"{group}_initial"
    initial_dir.mkdir()
    measure(root / group / "after_anchor_merge.pth", data, initial_dir)
    measure(final, data, root / group)


if __name__ == "__main__":
    main()
