"""Decode a HAC++ bitstream and evaluate PSNR/SSIM/LPIPS on held-out views.

Usage (5090, HAC_5090_a100 env):

    python scripts/eval_decoded.py \
        --artifact-dir runs/ours_hacpp_30k_full/bitstreams \
        --data-dir "$HOME/data_space/web_scan/第一组数据" \
        --result-dir runs/ours_hacpp_30k_full/decoded_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.trainer import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--data-factor", type=int, default=2)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    codec = HACPlusCodec()
    model = codec.decode(Path(args.artifact_dir))
    print(
        f"[Decode] anchors={model.num_anchors} "
        f"decoded_version={model.core.decoded_version}"
    )

    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=args.data_factor,
        test_every=args.test_every,
        white_background=False,
        preload_images=True,
        device=args.device,
    )
    evaluate(model, dataset, args.result_dir, iteration=30_000)


if __name__ == "__main__":
    main()
