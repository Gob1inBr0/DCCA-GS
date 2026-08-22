"""Wait for the DB lambda-RD cells, then aggregate results into JSON + CSV.

Run on 5090 after launching runner_db_lambda_rd.sh on both GPUs:
    python scripts/collect_db_lambda_rd.py

Waits until each GPU log contains two ALL_DONE lines (playroom + drjohnson),
then reads baseline (bitstreams + decoded_eval) and MLP-quantized
(mlp_quant_cd8_rest16) results for lambda in {0.001, 0.002, 0.004} and appends
rows to docs/data/experiments.csv.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

RUNS = Path("/home/fansonglin/data_space/DCCA-GS/runs")
PHG = Path("/home/fansonglin/data_space/DCCA-GS/PHG")
CSV = PHG / "docs" / "data" / "experiments.csv"
GPU_LOGS = [RUNS / "db_rd_lambda_gpu0.log", RUNS / "db_rd_lambda_gpu1.log"]
SCENES = {"playroom": "DB-playroom", "drjohnson": "DB-drjohnson"}
SCENE_DIRS = {
    "playroom": "/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db/playroom",
    "drjohnson": "/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db/drjohnson",
}
LAMBDAS = (0.001, 0.002, 0.004)


def tag(scene: str, lam: float) -> str:
    return f"db_{scene}_i6_110k_h32_l{format(lam, '.3f').replace('.', 'p')}"


def wait_done() -> None:
    print("[collect] waiting for ALL_DONE in both GPU logs ...", flush=True)
    while True:
        counts = []
        for log in GPU_LOGS:
            try:
                counts.append(log.read_text().count("ALL_DONE"))
            except FileNotFoundError:
                counts.append(0)
        if all(c >= 2 for c in counts):
            print(f"[collect] all cells done {counts}", flush=True)
            return
        print(
            f"[collect] gpu0={counts[0]}/2 gpu1={counts[1]}/2 "
            f"{time.strftime('%H:%M:%S')}",
            flush=True,
        )
        time.sleep(600)


def main() -> None:
    wait_done()
    summary = {"scenes": {}, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    rows = []
    for scene, scene_label in SCENES.items():
        scene_rows = []
        for lam in LAMBDAS:
            t = tag(scene, lam)
            base = RUNS / t
            meta_p = base / "bitstreams" / "hac_meta.json"
            metrics_p = base / "decoded_eval" / "metrics.jsonl"
            quant_p = base / "mlp_quant_cd8_rest16" / "results.json"
            meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
            met = json.loads(metrics_p.read_text()) if metrics_p.is_file() else {}
            quant = (
                json.loads(quant_p.read_text())["rows"][0]
                if quant_p.is_file()
                else {}
            )
            point = {
                "lambda": lam,
                "tag": t,
                "baseline": {
                    "psnr": met.get("psnr"),
                    "ssim": met.get("ssim"),
                    "lpips": met.get("lpips"),
                    "total_MB": meta.get("total_MB"),
                    "num_anchors": meta.get("num_anchors"),
                },
                "quant_cd8_rest16": {
                    "psnr": quant.get("psnr"),
                    "ssim": quant.get("ssim"),
                    "lpips": quant.get("lpips"),
                    "total_MB": quant.get("total_MB"),
                    "num_anchors": quant.get("num_anchors"),
                    "attr_MB": quant.get("attr_MB"),
                    "mlp_payload_MB": quant.get("mlp_payload_MB"),
                },
            }
            scene_rows.append(point)
            # CSV rows: baseline + quantized per lambda.
            for variant, d in (
                ("baseline float32", point["baseline"]),
                ("MLP quant cd8_rest16", point["quant_cd8_rest16"]),
            ):
                rows.append(
                    [
                        "phg_db_lambda_rd",
                        scene_label,
                        t,
                        f"lambda={lam} {variant}",
                        110000,
                        lam,
                        50,
                        "cd8_rest16" if variant.startswith("MLP") else "float32",
                        d.get("psnr"),
                        d.get("ssim"),
                        d.get("lpips"),
                        d.get("total_MB"),
                        "",
                        d.get("num_anchors"),
                        "render_metrics",
                        "",
                        "DB lambda-RD (new optimal config 110k h32 dim50 I2+I6)",
                        "collect_db_lambda_rd.py",
                    ]
                )
        summary["scenes"][scene] = scene_rows
        for r in scene_rows:
            b, q = r["baseline"], r["quant_cd8_rest16"]
            print(
                f"{scene} lambda={r['lambda']}: base {b['psnr']:.3f}/{b['total_MB']:.4f} "
                f"| quant {q['psnr']:.3f}/{q['total_MB']:.4f}",
                flush=True,
            )
    out = RUNS / "db_rd_110k.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[collect] wrote {out}", flush=True)

    if CSV.is_file():
        with open(CSV, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[collect] appended {len(rows)} rows to {CSV}", flush=True)
    else:
        print(f"[collect] CSV missing: {CSV}", flush=True)
    print("[collect] COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
