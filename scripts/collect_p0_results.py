#!/usr/bin/env python3
"""Collect + analyze P0 (E1/E4) results from 5090 run dirs.

Reads decoded metrics / bitstream metadata / final anchor counts and prints a
comparison table plus BD-PSNR/BD-rate for the E1 MiniSplat-vs-baseline budget
curves once all runs have finished. Missing runs are listed as PENDING.

Run on 5090:  python3 scripts/collect_p0_results.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RUNS = Path("/home/fansonglin/data_space/DCCA-GS/runs")
WEB = Path("/home/fansonglin/data_space/web_scan/runs")

# tag -> dict(ratio, mini, spa, scene, semantic)
EXPECTED = {
    # E1: baseline SPA (done previously)
    "spa_fixed_low_baseline": dict(ratio=0.52, mini=0, spa=1, scene="playroom"),
    "spa_fixed_high_baseline": dict(ratio=0.85, mini=0, spa=1, scene="playroom"),
    "spa_fixed_r092_baseline": dict(ratio=0.92, mini=0, spa=1, scene="playroom"),
    "spa_fixed_r097_baseline": dict(ratio=0.97, mini=0, spa=1, scene="playroom"),
    # E1: MiniSplat + SPA (0.85 done; 0.52/0.92/0.97 are P0)
    "spa_minisplat_cell2": dict(ratio=0.85, mini=1, spa=1, scene="playroom"),
    "p0_e1_playroom_r052_mini": dict(ratio=0.52, mini=1, spa=1, scene="playroom"),
    "p0_e1_playroom_r092_mini": dict(ratio=0.92, mini=1, spa=1, scene="playroom"),
    "p0_e1_playroom_r097_mini": dict(ratio=0.97, mini=1, spa=1, scene="playroom"),
    # E4: 4-28 2x2 (non-SPA baseline exists; three runs are P0)
    "run428_baseline": dict(ratio=None, mini=0, spa=0, scene="4-28"),
    # reused 110k baselines (old base, authoritative decoded results)
    "4-28_lxdim_110k_dim50_l0p004": dict(ratio=None, mini=0, spa=0, scene="4-28", path=WEB),
    "4-28_i6_110k_h32_l0p004_spa0p5": dict(ratio=0.5, mini=0, spa=1, scene="4-28", path=WEB),
    "p0_e4_428_nospa_mini": dict(ratio=None, mini=1, spa=0, scene="4-28"),
    "p0_e4_428_spa_base": dict(ratio=0.85, mini=0, spa=1, scene="4-28"),
    "p0_e4_428_spa_mini": dict(ratio=0.85, mini=1, spa=1, scene="4-28"),
}


def load_metrics(tag: str, base: Path = RUNS) -> dict | None:
    p = base / tag / "decoded_eval" / "metrics.jsonl"
    if not p.exists():
        # 4-28 non-SPA runs have no bitstream; fall back to training eval.
        p = base / tag / "metrics.jsonl"
        if not p.exists():
            return None
        src = "train_eval"
    else:
        src = "decoded"
    line = p.read_text().strip().splitlines()[-1]
    m = json.loads(line)
    m["_source"] = src
    return m


def load_meta(tag: str, base: Path = RUNS) -> dict | None:
    p = base / tag / "bitstreams" / "hac_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def final_anchors(tag: str, base: Path = RUNS) -> int | None:
    p = base / tag / "train.log"
    if not p.exists():
        return None
    txt = p.read_text(errors="replace")
    m = re.findall(r"Final anchors:\s*(\d+)", txt)
    return int(m[-1]) if m else None


def pchip_psnr(lr, psnr, x):
    try:
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(lr, psnr)(x)
    except Exception:
        return np.interp(x, lr, psnr)


def pchip_rate(psnr, rate, x):
    try:
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(psnr, np.log10(rate))(x)
    except Exception:
        return np.interp(x, psnr, np.log10(rate))


def bd(points_a, points_b) -> tuple[float, float] | None:
    """points = list[(rate_MB, psnr)]; returns (BD-PSNR dB, BD-rate %) for a->b."""
    if len(points_a) < 2 or len(points_b) < 2:
        return None
    a = np.array([(np.log10(r), p) for r, p in sorted(points_a)])
    b = np.array([(np.log10(r), p) for r, p in sorted(points_b)])
    lo = max(a[:, 0].min(), b[:, 0].min())
    hi = min(a[:, 0].max(), b[:, 0].max())
    if hi <= lo:
        return None
    x = np.linspace(lo, hi, 500)
    bd_psnr = float(np.trapezoid(pchip_psnr(b[:, 0], b[:, 1], x)
                                 - pchip_psnr(a[:, 0], a[:, 1], x), x) / (hi - lo))
    pa, pb = np.sort(a[:, 1]), np.sort(b[:, 1])
    plo, phi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
    if phi <= plo:
        return bd_psnr, float("nan")
    y = np.linspace(plo, phi, 500)
    bd_rate = 10 ** (np.trapezoid(pchip_rate(b[:, 1], 10**b[:, 0], y)
                                  - pchip_rate(a[:, 1], 10**a[:, 0], y), y) / (phi - plo)) - 1
    return bd_psnr, float(bd_rate) * 100


def main() -> None:
    rows = []
    for tag, cfg in EXPECTED.items():
        base = cfg.pop("path", RUNS)
        m = load_metrics(tag, base)
        meta = load_meta(tag, base)
        tr = final_anchors(tag, base)
        rows.append(dict(
            tag=tag, **cfg,
            psnr=m["psnr"] if m else None,
            ssim=m["ssim"] if m else None,
            lpips=m["lpips"] if m else None,
            total_mb=meta.get("total_MB") if meta else None,
            coded=meta.get("num_anchors") if meta else None,
            trained=tr,
            source=m.get("_source") if m else None,
        ))

    print("=== TABLE ===")
    hdr = f"{'tag':34s} {'scene':8s} {'spa':3s} {'mini':4s} {'ratio':5s} {'PSNR':8s} {'SSIM':7s} {'LPIPS':7s} {'MB':8s} {'coded':>8s} {'trained':>8s} src"
    print(hdr)
    for r in rows:
        print(f"{r['tag']:34s} {r['scene']:8s} {r['spa']:<3d} {r['mini']:<4d} "
              f"{str(r['ratio']):5s} "
              f"{r['psnr'] if r['psnr'] is not None else 'PENDING':>8} "
              f"{r['ssim'] if r['ssim'] is not None else '':>7} "
              f"{r['lpips'] if r['lpips'] is not None else '':>7} "
              f"{r['total_mb'] if r['total_mb'] is not None else '':>8} "
              f"{r['coded'] if r['coded'] is not None else '':>8} "
              f"{r['trained'] if r['trained'] is not None else '':>8} {r['source'] or ''}")

    def curve(name):
        pts = [(r["total_mb"], r["psnr"]) for r in rows
               if r["tag"].startswith(name) and r["total_mb"] is not None and r["psnr"] is not None]
        return sorted(pts)

    b_base = curve("spa_fixed_low_baseline") + curve("spa_fixed_high_baseline") + \
             curve("spa_fixed_r092_baseline") + curve("spa_fixed_r097_baseline")
    m = curve("spa_minisplat_cell2") + curve("p0_e1_playroom_r0")
    print("\n=== E1 BD (baseline SPA -> MiniSplat SPA) ===")
    res = bd(b_base, m)
    print("BD-PSNR:", None if res is None else round(res[0], 4), "dB | BD-rate:",
          None if res is None else round(res[1], 3), "%")
    if res is None or len(b_base) < 4 or len(m) < 4:
        print("(pending: needs all 4 budget-point MiniSplat runs)")

    print("\n=== E4 4-28 (training-eval or decoded; no-SPA rows use train_eval) ===")
    for r in rows:
        if r["scene"] == "4-28":
            print(f"{r['tag']}: PSNR {r['psnr']} SSIM {r['ssim']} LPIPS {r['lpips']} "
                  f"{r['total_mb'] or ''}MB coded={r['coded']} trained={r['trained']} src={r['source']}")


if __name__ == "__main__":
    main()
