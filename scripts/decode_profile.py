#!/usr/bin/env python3
"""Time the per-chunk range decode of a DCCA layered bitstream.

This is a thin wrapper around ``full_view_eval.py``: it monkeypatches
``c25_real_bitstream.decode_chunk`` so each chunk's decode wall time is
recorded, then runs the normal full-view evaluation.  It does not change
the bitstream or the model; the only output is a JSON with per-chunk times
plus the normal prefix JSON from ``full_view_eval``.

Usage (server):
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/home/project2/DCCA-GS \
  python scripts/decode_profile.py \
    --run /home/project2/dcca_runs/<tag> \
    --data-dir /home/project2/data/<scene> \
    --mapped /home/project2/dcca_runs/<tag>/<tag>.mapped.npy \
    --data-factor 2 --max-width 3200 \
    --out /home/project2/dcca_runs/<tag>/decode_profile.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import c25_real_bitstream as rb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--mapped", default=None)
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-out", default=None)
    args = ap.parse_args()

    stats = []
    orig_decode = rb.decode_chunk

    def timed_decode(data, g, params, ctx=None, s4=None, s5_flag=None):
        t0 = time.perf_counter()
        out = orig_decode(data, g, params, ctx, s4, s5_flag)
        dt = time.perf_counter() - t0
        stats.append({
            "bytes": int(len(data)),
            "sec": float(dt),
            "n_symbols": int(len(g)),
            "ctx": bool(ctx is not None),
        })
        return out

    rb.decode_chunk = timed_decode

    prefix_out = args.prefix_out or str(Path(args.out).with_name(
        Path(args.out).stem + "_prefixes.json"))
    fve_argv = [
        "full_view_eval.py",
        "--run", args.run,
        "--data-dir", args.data_dir,
        "--out", prefix_out,
        "--data-factor", str(args.data_factor),
    ]
    if args.mapped:
        fve_argv += ["--mapped", args.mapped]
    if args.max_width:
        fve_argv += ["--max-width", str(args.max_width)]
    sys.argv = fve_argv

    import full_view_eval as fve  # noqa: E402
    fve.main()

    total = float(sum(s["sec"] for s in stats))
    payload = {
        "run": args.run,
        "n_chunks": len(stats),
        "total_decode_sec": total,
        "per_chunk": stats,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[decode_profile] chunks={len(stats)} total={total:.3f}s -> {args.out}")


if __name__ == "__main__":
    main()
