"""Generate a fixed camera schedule for the fork framework.

The schedule is one integer per training step (1-indexed steps read
schedule[i-1]); every consumer (spine and all fork arms) must load the same
schedule so step i's camera is a deterministic function of i, not of a fresh
random draw. Stratified sampling instead of uniform random: shuffling the
train-camera list in repeating blocks keeps view coverage balanced across any
window (a purely random draw can starve views for thousands of steps, which
matters more when arms are compared pairwise at 150-view eval).

Usage:
  python scripts/gen_camera_schedule.py --n-cams 1050 --steps 30000 \
      --seed 42 --out camera_schedule_42.npy
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cams", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    blocks = []
    while sum(len(b) for b in blocks) < args.steps:
        block = np.arange(args.n_cams)
        rng.shuffle(block)
        blocks.append(block)
    schedule = np.concatenate(blocks)[: args.steps].astype(np.int64)
    np.save(args.out, schedule)
    # Self-check: every camera appears in the first full block, coverage per
    # window of len(cams) steps is exactly one (stratified property).
    first_window = schedule[: args.n_cams]
    assert np.unique(first_window).size == args.n_cams
    print(
        f"[schedule] saved {args.out}: steps={args.steps} cams={args.n_cams} "
        f"seed={args.seed} (stratified: 1 visit per camera per {args.n_cams}-step window)"
    )


if __name__ == "__main__":
    main()
