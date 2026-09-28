"""Build one consolidated experiments CSV from existing docs + LYH run dirs.

Run on the remote LYH host:
    python scripts/build_experiments_all_csv.py \
      --out docs/data/experiments_all.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


COLUMNS = [
    "group",
    "scene",
    "run_id",
    "variant",
    "iteration",
    "lambda",
    "feat_dim",
    "mlp_quant",
    "psnr",
    "ssim",
    "lpips",
    "total_mb",
    "anchors_trained",
    "anchors_coded",
    "metric_type",
    "metric_value",
    "notes",
    "source",
    "method_rank",
]


def read_existing(path: str) -> tuple[list[dict[str, str]], set[tuple[str, str, str, str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict((k, r.get(k, "")) for k in COLUMNS) for r in reader]
    keys = {
        (r["run_id"], r["variant"], r["iteration"], r["lambda"], r["mlp_quant"])
        for r in rows
    }
    return rows, keys


def infer_lambda(tag: str) -> str:
    low = tag.lower()
    m = re.search(r"_(?:l|lambda)(\d{4})", low)
    if m:
        return {
            "0001": "0.001",
            "0002": "0.002",
            "0003": "0.003",
            "0004": "0.004",
            "0005": "0.0005",
        }.get(m.group(1), "")
    if "baseline_110k" in low:
        return "0.004"
    m = re.search(r"l0p(\d{3})", low)
    if m:
        return f"0.{m.group(1)}"
    return ""


def infer_iteration(tag: str) -> str:
    if "30k" in tag or "3_0" in tag or "30000" in tag:
        return "30000"
    if "110k" in tag or "110000" in tag:
        return "110000"
    if "hacpp_1-78" in tag or tag.startswith("hacpp_"):
        return "30000"
    if tag.startswith("dcca_1-78"):
        return "110000"
    return ""


def scene_from_tag(tag: str) -> str:
    for name in ("playroom", "drjohnson", "tandt_train", "tandt_truck",
                 "mip_garden", "mip_bicycle", "mip_stump"):
        if name in tag:
            return name.replace("mip_", "Mip-").replace("tandt_", "T&T-")
    if "1-78" in tag or "178" in tag:
        return "1-78"
    if "2-06" in tag or "2-6" in tag:
        return "2-06"
    if "4-10" in tag:
        return "4-10"
    return ""


def row_from_dcca(tag: str, data: dict, existing_keys: set) -> tuple[dict, bool]:
    run_id = tag
    variant = "DCCA"
    if "nospa" in tag:
        variant = "DCCA no-SPA"
    elif "depth" in tag:
        variant = "DCCA MiniSplat depth"
    elif "full" in tag:
        variant = "DCCA MiniSplat full"
    elif "baseline" in tag:
        variant = "DCCA baseline"
    iteration = infer_iteration(tag)
    lam = infer_lambda(tag)
    mlp = "cd8_rest16" if data.get("method") == "dcca_mlp_quant" else "float32"
    key = (run_id, variant, iteration, lam, mlp)
    if key in existing_keys:
        return {}, False
    return {
        "group": "dcca_lyh",
        "scene": scene_from_tag(tag),
        "run_id": run_id,
        "variant": variant,
        "iteration": iteration,
        "lambda": lam,
        "feat_dim": "50",
        "mlp_quant": mlp,
        "psnr": data.get("psnr", ""),
        "ssim": data.get("ssim", ""),
        "lpips": data.get("lpips", ""),
        "total_mb": data.get("total_MB", ""),
        "anchors_trained": data.get("anchors_trained", ""),
        "anchors_coded": data.get("anchors_coded", ""),
        "metric_type": "render_metrics",
        "metric_value": "",
        "notes": "LYH run dir",
        "source": "lyh_runs",
        "method_rank": "",
    }, True


def row_from_hacpp(tag: str, data: dict, existing_keys: set) -> tuple[dict, bool]:
    run_id = tag
    variant = "HAC++"
    iteration = infer_iteration(tag)
    lam = infer_lambda(tag)
    key = (run_id, variant, iteration, lam, "float32")
    if key in existing_keys:
        return {}, False
    return {
        "group": "baseline",
        "scene": scene_from_tag(tag),
        "run_id": run_id,
        "variant": variant,
        "iteration": iteration,
        "lambda": lam,
        "feat_dim": "",
        "mlp_quant": "float32",
        "psnr": data.get("psnr", ""),
        "ssim": data.get("ssim", ""),
        "lpips": data.get("lpips", ""),
        "total_mb": data.get("total_MB", ""),
        "anchors_trained": "",
        "anchors_coded": data.get("anchors_coded", ""),
        "metric_type": "render_metrics",
        "metric_value": "",
        "notes": "official HAC++ run",
        "source": "hacpp",
        "method_rank": "",
    }, True


def normalize_allmethod(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        readers = list(csv.reader(f))
    if len(readers) < 3:
        return []
    headers = readers[0]
    rows = []
    for method_row in readers[2:]:
        method = method_row[0]
        rank = method_row[1] if len(method_row) > 1 else ""
        for col in range(2, len(headers), 4):
            scene = headers[col] if col < len(headers) else ""
            if not scene or col + 3 >= len(method_row):
                continue
            vals = method_row[col : col + 4]
            if not vals or vals[0] in ("", "N/A"):
                continue
            rows.append({
                "group": "survey",
                "scene": scene,
                "run_id": f"survey_{method}_{scene}",
                "variant": method,
                "iteration": "",
                "lambda": "",
                "feat_dim": "",
                "mlp_quant": "",
                "psnr": vals[0],
                "ssim": vals[1] if len(vals) > 1 else "",
                "lpips": vals[2] if len(vals) > 2 else "",
                "total_mb": vals[3] if len(vals) > 3 else "",
                "anchors_trained": "",
                "anchors_coded": "",
                "metric_type": "render_metrics",
                "metric_value": "",
                "notes": "3DGS compression survey table",
                "source": "allMothod",
                "method_rank": rank,
            })
    return rows


def scan_lyh_runs(root: str, existing_keys: set) -> list[dict]:
    from collect_lyh_rd import collect_dcca, collect_hacpp

    out = []
    root_path = Path(root)
    for d in sorted(root_path.iterdir()):
        if not d.is_dir():
            continue
        tag = d.name
        if tag.startswith("hacpp_"):
            data = collect_hacpp(d, tag)
            if data:
                row, added = row_from_hacpp(tag, data, existing_keys)
                if added:
                    out.append(row)
                    existing_keys.add((row["run_id"], row["variant"], row["iteration"],
                                       row["lambda"], row["mlp_quant"]))
        elif tag.startswith("lyh_") or tag.startswith("dcca_") or tag.startswith("p0_"):
            data = collect_dcca(d, tag)
            if data:
                row, added = row_from_dcca(tag, data, existing_keys)
                if added:
                    out.append(row)
                    existing_keys.add((row["run_id"], row["variant"], row["iteration"],
                                       row["lambda"], row["mlp_quant"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/data/experiments_all.csv")
    args = parser.parse_args()

    existing_rows, keys = read_existing("docs/data/experiments.csv")
    all_rows = existing_rows
    all_rows.extend(normalize_allmethod("docs/data/allMothod.csv"))
    all_rows.extend(scan_lyh_runs("/home/project2/runs", keys))
    all_rows.extend(scan_lyh_runs("/dev/shm/dcca_runs/1-78", keys))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
