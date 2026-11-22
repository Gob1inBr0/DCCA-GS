"""Compare a byte filter and no byte filter from the same room checkpoint."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from scaffold_gs.config import DataConfig
from scripts.compare_anchor_merge_response import audit_payload
from scripts.run_in_training_anchor_merge import (
    END_STEP,
    MERGE_STEP,
    measure,
    run_arm,
    smoke_reference,
)


NEIGHBORS = 16
LIMITS = {"minimum_one_byte": 1.0, "no_minimum": None, "smoke": None}


def decode(path, data, dest):
    dest.mkdir()
    measure(path, data, dest)
    payload = audit_payload(dest / "bitstreams", dest / "payload")
    metrics = json.loads((dest / "decoded_eval" / "metrics.jsonl").read_text().splitlines()[-1])
    meta = json.loads((dest / "bitstreams" / "hac_meta.json").read_text())
    files = {path.name: path.stat().st_size for path in (dest / "payload").iterdir()}
    return {**metrics, **payload, "meta": meta, "payload_files": files}


def verify_updates(initial_path, final_path):
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    rates = {group["name"]: group["lr"] for group in final["optimizer_state"]["param_groups"]}
    assert rates["anchor"] == 0 and rates["offset"] > 0 and rates["mask"] > 0, rates
    changes = {
        key: int(torch.count_nonzero(initial["model_state"][key] != final["model_state"][key]))
        for key in ("_anchor", "_offset", "_mask")
    }
    assert changes["_anchor"] == 0 and changes["_offset"] > 0 and changes["_mask"] > 0, changes
    print(f"[SpatialUpdates] rates={rates} changed_values={changes}", flush=True)
    return {"learning_rates": rates, "changed_values": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("control", *LIMITS))
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--candidate-neighbors", type=int, default=NEIGHBORS)
    args = parser.parse_args()
    if args.candidate_neighbors < 1:
        parser.error("--candidate-neighbors must be positive")
    stage, root = args.stage, args.root
    archive = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert archive["iteration"] == MERGE_STEP - 1
    data = replace(DataConfig(**archive["data_config"]), preload_images=False)
    if stage == "control":
        final = run_arm(archive, data, root, stage, args.checkpoint, END_STEP, None, args.schedule)
        result = {
            "initial": decode(args.checkpoint, data, root / stage / "initial"),
            "final": decode(final, data, root / stage / "final"),
            "updates": verify_updates(args.checkpoint, final),
        }
    else:
        smoke = stage == "smoke"
        start = smoke_reference(archive, root) if smoke else args.checkpoint
        final_step = MERGE_STEP - 1 + 400 if smoke else END_STEP
        final = run_arm(
            archive, data, root, stage, start, final_step, MERGE_STEP, args.schedule,
            response_rms_limit=0.10, save_merge_initial=True, min_saving_bytes=LIMITS[stage],
            candidate_neighbors=args.candidate_neighbors,
        )
        out = root / stage
        initial = out / "after_anchor_merge.pth"
        report = json.loads((out / f"anchor_merge_{MERGE_STEP}.json").read_text())
        assert report["expected_saving_min_bytes"] == LIMITS[stage]
        assert report["candidate_neighbors"] == args.candidate_neighbors and report["merges"] > 0
        assert bool(report["accepted_below_one_estimated_byte"]) == (LIMITS[stage] is None), report
        result = {
            "merge": report,
            "updates": verify_updates(initial, final),
            "initial": decode(initial, data, out / "initial"),
            "final": decode(final, data, out / "final"),
        }
    (root / stage / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
