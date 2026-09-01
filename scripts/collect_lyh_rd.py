"""Collect 1-78/2-06/4-10 RD results from LYH runs into one JSON.

Run on the remote host:
    python scripts/collect_lyh_rd.py --scene 1-78 \
      --run-root /dev/shm/dcca_runs/1-78 \
      --out /home/project2/runs/rd_summary_1-78.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _latest_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    lines = metrics_path.read_text().strip().splitlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def _hacpp_size_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Encoded sizes in MB:.*?Total ([\d.]+)", text, re.S)
    return float(m.group(1)) if m else None


def collect_hacpp(run_dir: Path, tag: str) -> dict | None:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    data = json.loads(results_path.read_text())
    row = None
    for value in data.values():
        if isinstance(value, dict) and "PSNR" in value:
            row = value
            break
    if row is None:
        return None
    # The survey's "highrate / lowrate" labels are based on resulting model
    # size, not lambda. For HAC++ a smaller lambda means a larger model, so
    # lambda=0.0005 is highrate and lambda=0.004 is lowrate.
    rate_label = "highrate" if "_l0005" in tag else "lowrate"
    return {
        "method": "hacpp",
        "tag": tag,
        "rate_label": rate_label,
        "psnr": float(row["PSNR"]),
        "ssim": float(row["SSIM"]),
        "lpips": float(row["LPIPS"]),
        "total_MB": _hacpp_size_from_log(run_dir.parent / f"{tag}.log"),
        "anchors_coded": None,
    }


def collect_dcca(run_dir: Path, tag: str) -> dict | None:
    quant_path = run_dir / "mlp_quant_cd8_rest16" / "results.json"
    if quant_path.exists():
        data = json.loads(quant_path.read_text())
        row = next((r for r in data["rows"] if r.get("bits") == "mixed"), None)
        if row is not None:
            return {
                "method": "dcca_mlp_quant",
                "tag": tag,
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
                "total_MB": float(row["total_MB"]),
                "anchors_coded": int(row["num_anchors"]),
            }

    meta_path = run_dir / "bitstreams" / "hac_meta.json"
    metrics_path = run_dir / "decoded_eval" / "metrics.jsonl"
    metrics = _latest_metrics(metrics_path)
    if metrics is None or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return {
        "method": "dcca_baseline",
        "tag": tag,
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "lpips": float(metrics["lpips"]),
        "total_MB": float(meta["total_MB"]),
        "anchors_coded": int(meta["num_anchors"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.run_root)
    rows = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        tag = directory.name
        if not (
            tag.startswith("hacpp_")
            or tag.startswith("dcca_")
            or tag.startswith("lyh_")
        ):
            continue
        if tag.startswith("hacpp_"):
            row = collect_hacpp(directory, tag)
        else:
            row = collect_dcca(directory, tag)
        if row is not None:
            rows.append(row)

    summary = {"scene": args.scene, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
