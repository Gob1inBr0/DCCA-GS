#!/usr/bin/env python3
"""S2: bytes-vs-PSNR prefix rendering sweep (progressive-stream design doc, stage S2).

Renders the scene using only the first k% of anchors under a given ordering,
for k in {5,10,25,50,100}, and reports PSNR per prefix — the measured
bytes-quality curve that validates the offline coverage surrogate (S1).

Also reports the A3 gate statistics from real render coverage: anchors-per-pixel-
block multiplicity and blocks-per-anchor (overlap strength decides whether the
submodular greedy order is worth running on real edges).

Run AFTER the r3 queue frees a whitelisted card (GPU-light: ~minutes).
Usage (server):
  python scripts/render_prefix_sweep.py --run <run_dir with bitstreams/attributes.pth> \
      --data-dir /dev/shm/dcca_data/1-78/data --out /tmp/prefix_sweep.json

Implementation notes:
- Decoded attributes come from bitstreams/attributes.pth (anchor, offset,
  scaling, mask, anchor_feat, decoder weights) — same source as eval_decoded.py.
- Prefix = subset of anchor rows; gaussians are generated ONLY for kept
  anchors (generate_gaussians on the subset), so missing anchors are skipped
  (design doc D2 pre-registered strategy).
- Orderings: contribution-desc (anchor_stats area), random, and (optional)
  greedy-on-real-edges if an edge dump exists.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ORDERINGS = ["contribution", "random"]
PREFIXES = [0.05, 0.10, 0.25, 0.50, 1.00]
N_VIEWS = 16  # fixed camera subset for comparability across prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", default=str(REPO / "analysis/anchor_stats/ph0c_base_r085_lam0005_s42/anchor_stats.npz"),
                    help="anchor_stats.npz with per-anchor area (contribution order)")
    ap.add_argument("--out", default="/tmp/prefix_sweep.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    art = torch.load(Path(args.run) / "bitstreams/attributes.pth", map_location="cpu", weights_only=False)
    N = art["anchor"].shape[0]

    area = np.load(args.stats)["area"].astype(np.float64)
    assert area.shape[0] == N, "stats row count mismatch with this run"
    rng = np.random.default_rng(0)
    orders = {
        "contribution": np.argsort(-area, kind="stable"),
        "random": rng.permutation(N),
    }

    # Model rebuild mirrors eval_decoded.py: construct HACPlusModel from the
    # bitstream header/attributes and load decoded attributes, then render
    # through the repo renderer. (TODO at execution time: reuse
    # eval_decoded.py's loader verbatim to avoid drift; the loader function
    # is the only piece that needs the repo runtime.)
    raise SystemExit(
        "Skeleton only: import eval_decoded's model loader here, then for each "
        "ordering x prefix: keep = order[:int(N*k)]; model.generate_gaussians("
        "anchor_indices=keep) -> render N_VIEWS cams -> PSNR; record A3 overlap "
        "stats (blocks-per-anchor / anchors-per-block multiplicity histograms) "
        "from the render meta at full prefix. See design doc S2."
    )


if __name__ == "__main__":
    main()
