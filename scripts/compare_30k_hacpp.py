"""Compare prior 30k MiniSplat runs with official HAC++ high/low points."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUN_ROOT = Path("/dev/shm/dcca_runs/1-78")
OLD_RUN_ROOT = Path("/home/project2/runs")
HAC_ROOT = Path("/home/project2/HAC-plus/results")

SCENES = {
    "playroom": (
        "DeepBlending/playroom.csv",
        ["lyh_full_playroom", "lyh_depth_playroom"],
    ),
    "drjohnson": (
        "DeepBlending/drjohnson.csv",
        ["lyh_full_drjohnson", "lyh_depth_drjohnson"],
    ),
    "tandt_train": (
        "TanksAndTemples/train.csv",
        ["lyh_full_tandt_train", "lyh_depth_tandt_train"],
    ),
    "tandt_truck": (
        "TanksAndTemples/truck.csv",
        ["lyh_full_tandt_truck", "lyh_depth_tandt_truck"],
    ),
    "mip_garden": (
        "MipNeRF360/garden.csv",
        ["lyh_full_mip_garden", "lyh_depth_mip_garden"],
    ),
    "mip_bicycle": (
        "MipNeRF360/bicycle.csv",
        ["lyh_full_mip_bicycle", "lyh_depth_mip_bicycle"],
    ),
    "mip_stump": (
        "MipNeRF360/stump.csv",
        ["lyh_full_mip_stump", "lyh_depth_mip_stump"],
    ),
}


def hac_rows(scene: str) -> dict[str, dict]:
    path = HAC_ROOT / SCENES[scene][0]
    with path.open() as f:
        rows = {
            row["Submethod"]: row
            for row in csv.DictReader(f)
        }
    return rows


def dcca_row(scene: str, run: str) -> dict | None:
    root = RUN_ROOT / run if (RUN_ROOT / run).exists() else OLD_RUN_ROOT / run
    quant = root / "mlp_quant_cd8_rest16" / "results.json"
    if quant.exists():
        data = json.loads(quant.read_text())
        row = next((r for r in data["rows"] if r.get("bits") == "mixed"), None)
        if row is not None:
            return row
    meta = root / "bitstreams" / "hac_meta.json"
    metrics = root / "decoded_eval" / "metrics.jsonl"
    if meta.exists() and metrics.exists():
        m = json.loads(metrics.read_text().strip().splitlines()[-1])
        return {
            "psnr": float(m["psnr"]),
            "ssim": float(m["ssim"]),
            "lpips": float(m["lpips"]),
            "total_MB": float(json.loads(meta.read_text())["total_MB"]),
        }
    return None


def main() -> None:
    print("scene | hac high PSNR/MB | hac low PSNR/MB | full PSNR/MB | depth PSNR/MB")
    for scene, (_, runs) in SCENES.items():
        h = hac_rows(scene)
        high, low = h.get("HAC++-highrate"), h.get("HAC++-lowrate")
        full = dcca_row(scene, runs[0])
        depth = dcca_row(scene, runs[1])
        fmt = lambda r, key="psnr": f"{float(r[key]):.3f}" if r else "NA"
        print(
            f"{scene} | "
            f"{fmt(high, 'PSNR')}/{float(high['Size [Bytes]']) / 1024 / 1024:.3f} | "
            f"{fmt(low, 'PSNR')}/{float(low['Size [Bytes]']) / 1024 / 1024:.3f} | "
            f"{fmt(full)}/{float(full['total_MB']):.3f} | "
            f"{fmt(depth)}/{float(depth['total_MB']):.3f}"
        )


if __name__ == "__main__":
    main()
