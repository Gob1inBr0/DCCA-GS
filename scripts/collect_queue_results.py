"""Collect queued DB-ablation / Mip360 / T&T cells into JSON + the unified CSV.

Run after queue_after_db_rd.sh finishes both GPUs (GPU0 invokes it):
    python scripts/collect_queue_results.py

Scans result dirs by tag prefix, reads baseline (bitstreams + decoded_eval) and
MLP-quantized (mlp_quant_cd8_rest16) rows, and appends rows to
docs/PHG_experiments.csv (skipping run_id+variant pairs already present).
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

RUNS = Path("/home/fansonglin/data_space/web_scan/runs")
PHG = Path("/home/fansonglin/xieliang/chentong/PHG")
CSV = PHG / "docs" / "PHG_experiments.csv"

PATTERNS = {
    "phg_db_ablation": ("db_playroom_i6_110k_h32_l0p002_ablation_", "DB-playroom"),
    "phg_mip360_rd": ("mip360_", "Mip360"),
    "phg_tandt_rd": ("tandt_", "TNT"),
}


def scene_label(group: str, tag: str) -> str:
    if group == "phg_db_ablation":
        return "DB-playroom"
    if group == "phg_mip360_rd":
        return "Mip360-" + tag.split("_", 2)[1]
    if group == "phg_tandt_rd":
        return "TNT-" + tag.split("_", 2)[1]
    return tag


def existing_keys() -> set:
    if not CSV.is_file():
        return set()
    with open(CSV, encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return set()
        ri = header.index("run_id")
        vi = header.index("variant")
        return {(row[ri], row[vi]) for row in r if len(row) > vi}


def main() -> None:
    keys = existing_keys()
    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": {}}
    new_rows = []
    for group, (prefix, label_prefix) in PATTERNS.items():
        for d in sorted(RUNS.iterdir()):
            if not d.is_dir() or not d.name.startswith(prefix):
                continue
            meta_p = d / "bitstreams" / "hac_meta.json"
            metrics_p = d / "decoded_eval" / "metrics.jsonl"
            quant_p = d / "mlp_quant_cd8_rest16" / "results.json"
            if not (meta_p.is_file() and metrics_p.is_file() and quant_p.is_file()):
                continue
            meta = json.loads(meta_p.read_text())
            met = json.loads(metrics_p.read_text())
            quant = json.loads(quant_p.read_text())["rows"][0]
            tag = d.name
            m = re.search(r"_l0p(\d+)", tag)
            lam = float(m.group(1)) / 1000.0 if m else 0.0
            if tag.endswith("_ablation_i6off"):
                variant_note = "ablation I6-off"
            elif tag.endswith("_ablation_i2off"):
                variant_note = "ablation I2-off (I6 off too)"
            else:
                variant_note = "lambda-RD 30k"
            for variant, dd in (
                ("baseline float32", {
                    "psnr": met.get("psnr"), "ssim": met.get("ssim"),
                    "lpips": met.get("lpips"), "total_MB": meta.get("total_MB"),
                    "anchors": meta.get("num_anchors"),
                }),
                ("MLP quant cd8_rest16", {
                    "psnr": quant.get("psnr"), "ssim": quant.get("ssim"),
                    "lpips": quant.get("lpips"), "total_MB": quant.get("total_MB"),
                    "anchors": quant.get("num_anchors"),
                }),
            ):
                key = (tag, variant)
                if key in keys:
                    continue
                new_rows.append([
                    group,
                    scene_label(group, tag),
                    tag,
                    f"lambda={lam} {variant_note} {variant}",
                    110000 if "110k" in tag else 30000,
                    lam,
                    50,
                    "cd8_rest16" if variant.startswith("MLP") else "float32",
                    dd.get("psnr"),
                    dd.get("ssim"),
                    dd.get("lpips"),
                    dd.get("total_MB"),
                    "",
                    dd.get("anchors"),
                    "render_metrics",
                    "",
                    "queued DB-ablation/Mip360/T&T (new optimal config)",
                    "collect_queue_results.py",
                ])
                keys.add(key)
            summary["cells"][tag] = {
                "lambda": lam,
                "baseline": {
                    "psnr": met.get("psnr"), "total_MB": meta.get("total_MB"),
                },
                "quant": {
                    "psnr": quant.get("psnr"), "total_MB": quant.get("total_MB"),
                },
            }
            b, q = summary["cells"][tag]["baseline"], summary["cells"][tag]["quant"]
            print(
                f"{tag}: base {b['psnr']}/{b['total_MB']} "
                f"| quant {q['psnr']}/{q['total_MB']}",
                flush=True,
            )

    out = RUNS / "queue_results.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[collect_queue] wrote {out}", flush=True)

    if new_rows:
        with open(CSV, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(new_rows)
        print(f"[collect_queue] appended {len(new_rows)} rows to {CSV}", flush=True)
    else:
        print("[collect_queue] no new rows", flush=True)
    print("[collect_queue] COLLECT_QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
