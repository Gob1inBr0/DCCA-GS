"""BD-rate comparison: DCCA-GS 1-78 30k (32-dim) vs HAC++ 1-78 30k.

HAC++ 1-78 30k reference points come from docs/06-planning/HANDOVER_20260902.md
section 6.1 (hacpp_1-78_high/mid/low). Ordering note: lamb=0.004 is the low-rate
point and lamb=0.0005 is the high-rate point (the tag names high/low are about
model SIZE, not lambda). We compare by the target metric on the DECODED bitstream.

Usage:
    python scripts/bdrate_compare_1-78.py \
      --run-root /dev/shm/dcca_runs/1-78 \
      --runs dcca_1-78_30k_32dim_l0004:0.004,dcca_1-78_30k_32dim_l0002:0.002,dcca_1-78_30k_32dim_l0005:0.0005 \
      --out /home/project2/runs/bdrate_1-78.json

--source controls which volume is used for the rate axis:
  fp32  -> decoded_eval/metrics_summary.json metrics.codec_total_mb (MLP float32)
  mlp   -> mlp_quant_cd8_rest16/results.json total_MB (post-MLP-quant, final size)
Use the SAME source for both methods for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# HAC++ 1-78 30k reference (HANDOVER_20260902 section 6.1). These are the
# method's as-published sizes/metrics; we compare DCCA to these on the shared
# lambda set. lambda -> (PSNR, SSIM, LPIPS, total_MB).
HACPP_178_30K = {
    0.004: (27.8406, 0.8829, 0.1597, 20.0265),
    0.002: (28.2105, 0.8903, 0.1514, 26.2734),
    0.0005: (28.6780, 0.9001, 0.1396, 40.5589),
}


def _read_run(root: Path, tag: str, source: str) -> dict | None:
    run = root / tag
    if source == "mlp":
        q = run / "mlp_quant_cd8_rest16" / "results.json"
        if q.exists():
            d = json.loads(q.read_text())
            # results.json may nest metrics under "metrics" or be flat.
            m = d.get("metrics", d)
            if "total_MB" in m:
                return {
                    "psnr": float(m["psnr"]),
                    "ssim": float(m.get("ssim", 0.0)),
                    "lpips": float(m.get("lpips", 0.0)),
                    "mb": float(m["total_MB"]),
                }
    # Fallback / fp32 source: decoded_eval metrics_summary.json.
    ms = run / "decoded_eval" / "metrics_summary.json"
    if ms.exists():
        d = json.loads(ms.read_text())
        met = d["metrics"]
        mb = float(
            met.get("codec_total_mb", met.get("method_total_mb", float("nan")))
        )
        return {
            "psnr": float(met["psnr"]),
            "ssim": float(met["ssim"]),
            "lpips": float(met["lpips"]),
            "mb": mb,
        }
    return None


def bd_rate(rate_a, psnr_a, rate_b, psnr_b):
    """Bjontegaard delta: average % rate change of A vs B over shared PSNR."""
    pts_a = sorted(zip(psnr_a, rate_a))
    pts_b = sorted(zip(psnr_b, rate_b))
    if len(pts_a) < 2 or len(pts_b) < 2:
        return None
    lo = max(min(p for p, _ in pts_a), min(p for p, _ in pts_b))
    hi = min(max(p for p, _ in pts_a), max(p for p, _ in pts_b))
    if hi <= lo:
        return None
    xs = [lo + (hi - lo) * i / 199.0 for i in range(200)]

    def interp(pts):
        ps = [p for p, _ in pts]
        rs = [math.log10(r) for _, r in pts]
        out = []
        for x in xs:
            if x <= ps[0]:
                out.append(rs[0])
            elif x >= ps[-1]:
                out.append(rs[-1])
            else:
                for j in range(len(ps) - 1):
                    if ps[j] <= x <= ps[j + 1]:
                        t = (x - ps[j]) / (ps[j + 1] - ps[j])
                        out.append(rs[j] + t * (rs[j + 1] - rs[j]))
                        break
        return out

    la = interp(pts_a)
    lb = interp(pts_b)
    mean_diff = sum(a - b for a, b in zip(la, lb)) / len(la)
    return 100.0 * (10 ** mean_diff - 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="/dev/shm/dcca_runs/1-78")
    p.add_argument("--runs", default=(
        "dcca_1-78_30k_32dim_l0004:0.004,"
        "dcca_1-78_30k_32dim_l0002:0.002,"
        "dcca_1-78_30k_32dim_l0005:0.0005"))
    p.add_argument("--source", choices=["fp32", "mlp"], default="fp32")
    p.add_argument("--method-name", default="DCCA-GS(32dim,30k)")
    p.add_argument("--out", default="runs/bdrate_1-78.json")
    args = p.parse_args()

    root = Path(args.run_root)
    dcca = []
    for item in args.runs.split(","):
        tag, lam_s = item.split(":")
        lam = float(lam_s)
        row = _read_run(root, tag, args.source)
        if row is None:
            print(f"WARN missing result for {tag} ({args.source})", flush=True)
            continue
        row["lambda"] = lam
        dcca.append(row)

    hac = [
        {"lambda": lam, "psnr": p[0], "ssim": p[1], "lpips": p[2], "mb": p[3]}
        for lam, p in sorted(HACPP_178_30K.items())
    ]

    # sanity: self comparison should be ~0
    sanity = bd_rate(
        [h["mb"] for h in hac], [h["psnr"] for h in hac],
        [h["mb"] for h in hac], [h["psnr"] for h in hac],
    )

    dcca_sorted = sorted(dcca, key=lambda r: r["lambda"])
    bdr = bd_rate(
        [d["mb"] for d in dcca_sorted], [d["psnr"] for d in dcca_sorted],
        [h["mb"] for h in hac], [h["psnr"] for h in hac],
    ) if len(dcca_sorted) >= 2 else None
    # negative BD-rate = DCCA needs FEWER bits at same PSNR = better.

    summary = {
        "method": args.method_name,
        "source": args.source,
        "hacpp_refs": ["HANDOVER_20260902 6.1"],
        "dcca_points": [
            {"lambda": d["lambda"], "psnr": round(d["psnr"], 4),
             "ssim": round(d["ssim"], 4), "lpips": round(d["lpips"], 4),
             "mb": round(d["mb"], 4)}
            for d in dcca_sorted
        ],
        "hacpp_points": [
            {"lambda": h["lambda"], "psnr": h["psnr"], "ssim": h["ssim"],
             "lpips": h["lpips"], "mb": h["mb"]}
            for h in hac
        ],
        "sanity_self_bdrate_pct": round(sanity, 4) if sanity is not None else None,
        "bdrate_pct_vs_hacpp": round(bdr, 4) if bdr is not None else None,
        "interpretation": (
            "BD-rate<0 => DCCA-GS uses fewer MB at equal PSNR (better); "
            "BD-rate>0 => worse; None if PSNR ranges do not overlap."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
