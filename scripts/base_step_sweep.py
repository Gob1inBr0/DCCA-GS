#!/usr/bin/env python3
"""Base-layer coarsening sweep: push the progressive curve's starting point
(P0) into the sub-7 MB territory by starting feat/offset ladders at 16x
instead of 8x (and optionally scaling at 4x).

For each ladder config this runs c25_real_bitstream.py (--ladder) on the
same run directory and collects the per-prefix (bytes, PSNR) curve. The
ladder changes only WHERE the quality levels sit, not the final symbols —
so full-precision PSNR must match the baseline config to float noise, and
full-precision byte differences come purely from the conditional-table
context (different residual distributions per level).

Pre-registered gates for adopting a 16x ladder (defaults, override via
flags):
  P0 bytes      <= --p0-target-mb (7.0)
  P0 PSNR drop  <= --p0-loss-db  (1.5) vs the 8x baseline config
  no inversion  : PSNR non-decreasing across prefixes (tol 0.05 dB)

Usage (training machine, c25 python + PYTHONPATH=/home/project2/c25_pylib2):
  python scripts/base_step_sweep.py \
      --run /mnt/newproject2/dcca_runs/<run> \
      --data-dir /mnt/newproject2/data/1-78 \
      --stats <run>/anchor_stats.npz --out-dir analysis/base16_sweep
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = [
    ("baseline_8x", "feat=8,4,2,1 offset=8,4,2,1 scaling=2,1"),
    ("base16x", "feat=16,8,4,2,1 offset=16,8,4,2,1 scaling=2,1"),
    ("base16x_sc4x", "feat=16,8,4,2,1 offset=16,8,4,2,1 scaling=4,2,1"),
    ("offset16x", "feat=8,4,2,1 offset=16,8,4,2,1 scaling=2,1"),
]


def run_encoder(args, ladder, out_json, out_bin):
    cmd = [args.python, str(REPO / "scripts" / "c25_real_bitstream.py"),
           "--run", args.run, "--data-dir", args.data_dir,
           "--device", args.device,
           "--data-factor", str(args.data_factor),
           "--field-aware", "--ladder", ladder,
           "--out", str(out_json), "--bin", str(out_bin)]
    if args.stats:
        cmd += ["--stats", args.stats]
    if args.max_width:
        cmd += ["--max-width", str(args.max_width)]
    print(f"[SW] {ladder}\n[SW]   -> {out_json}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"[SW] encoder failed for ladder {ladder!r}, "
                         f"rc={proc.returncode}")
    return time.time() - t0


def curve(payload):
    return [{"prefix": r["prefix"], "real_bytes": r["real_bytes"],
             "psnr_mean": r["psnr_mean"]} for r in payload["results"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default="analysis/base16_sweep")
    ap.add_argument("--configs", default=None,
                    help="semicolon-separated 'tag:ladder' pairs overriding "
                         "the default grid")
    ap.add_argument("--p0-target-mb", type=float, default=7.0)
    ap.add_argument("--p0-loss-db", type=float, default=1.5)
    args = ap.parse_args()

    if args.configs:
        configs = []
        for part in args.configs.split(";"):
            tag, _, ladder = part.partition(":")
            configs.append((tag.strip(), ladder.strip()))
    else:
        configs = DEFAULT_CONFIGS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    base_p0_psnr = None
    for i, (tag, ladder) in enumerate(configs):
        out_json = out_dir / f"{tag}.json"
        out_bin = out_dir / f"{tag}.bin"
        t = run_encoder(args, ladder, out_json, out_bin)
        payload = json.loads(out_json.read_text())
        cv = curve(payload)
        p0, full = cv[0], cv[-1]
        if i == 0:
            base_p0_psnr = p0["psnr_mean"]
        inversions = [(a["prefix"], a["psnr_mean"], b["psnr_mean"])
                      for a, b in zip(cv[:-1], cv[1:])
                      if b["psnr_mean"] < a["psnr_mean"] - 0.05]
        entries.append({
            "tag": tag, "ladder": ladder, "coding_s": round(t, 1),
            "bin": str(out_bin), "curve": cv,
            "p0_bytes": p0["real_bytes"], "p0_psnr": p0["psnr_mean"],
            "full_bytes": full["real_bytes"], "full_psnr": full["psnr_mean"],
            "inversions": inversions,
        })
        print(f"[SW] {tag}: P0 {p0['real_bytes']/1e6:.2f} MB @"
              f"{p0['psnr_mean']:.3f}  full {full['real_bytes']/1e6:.2f} MB @"
              f"{full['psnr_mean']:.3f}  inversions={len(inversions)}",
              flush=True)

    base = entries[0]
    for e in entries[1:]:
        e["p0_bytes_delta_mb"] = round((e["p0_bytes"] - base["p0_bytes"]) / 1e6,
                                       3)
        e["p0_psnr_delta"] = round(e["p0_psnr"] - base["p0_psnr"], 3)
        e["full_psnr_delta"] = round(e["full_psnr"] - base["full_psnr"], 4)

    candidates = [e for e in entries[1:]
                  if e["p0_bytes"] <= args.p0_target_mb * 1e6
                  and e["p0_psnr"] >= base_p0_psnr - args.p0_loss_db
                  and not e["inversions"]
                  and abs(e["full_psnr_delta"]) < 0.05]
    winner = min(candidates, key=lambda e: e["p0_bytes"]) if candidates else None

    summary = {
        "run": args.run, "data_dir": args.data_dir, "stats": args.stats,
        "data_factor": args.data_factor, "max_width": args.max_width,
        "gates": {"p0_target_mb": args.p0_target_mb,
                  "p0_loss_db": args.p0_loss_db, "inversion_tol_db": 0.05},
        "baseline_tag": base["tag"],
        "entries": entries,
        "winner": winner["tag"] if winner else None,
        "verdict": ("PASS" if winner else "FAIL"),
    }
    out = out_dir / "sweep_summary.json"
    out.write_text(json.dumps(summary, indent=1))

    print("\n[SW] === sweep summary ===", flush=True)
    print(f"[SW] {'config':<16} {'P0 MB':>7} {'P0 dB':>7} {'dP0 dB':>7} "
          f"{'full MB':>8} {'full dB':>7} {'inv':>4}", flush=True)
    for e in entries:
        print(f"[SW] {e['tag']:<16} {e['p0_bytes']/1e6:7.2f} "
              f"{e['p0_psnr']:7.3f} "
              f"{e.get('p0_psnr_delta', 0.0):7.3f} "
              f"{e['full_bytes']/1e6:8.2f} {e['full_psnr']:7.3f} "
              f"{len(e['inversions']):4d}", flush=True)
    if winner:
        print(f"[SW] VERDICT: PASS — winner '{winner['tag']}' "
              f"(P0 {winner['p0_bytes']/1e6:.2f} MB @"
              f"{winner['p0_psnr']:.3f}, full-precision PSNR delta "
              f"{winner['full_psnr_delta']:+.4f})", flush=True)
    else:
        print("[SW] VERDICT: FAIL — no config meets all gates "
              "(P0 target, P0 loss budget, no inversion, full-precision "
              "PSNR unchanged)", flush=True)
    print(f"[SW] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
