"""Offline semantic-prior extraction (design: docs/语义先验实验设计.md Stage A).

For every TRAINING view of a scene, extracts and caches:
  - dino : DINOv2-base patch features at (H/14, W/14, 768)  -> fp16 npz
  - depth: Depth Anything V2 small dense depth at (H/4, W/4) -> fp16 npz
  - sam2 : SAM2.1 tiny automatic masks -> region id at (H/4, W/4) uint16
           plus region area fractions (json)

Pretrained weights are downloaded through HF_ENDPOINT (set to a reachable
mirror, e.g. https://hf-mirror.com). Usage (5090, HAC_5090_a100 env):

    HF_ENDPOINT=https://hf-mirror.com \
    python scripts/extract_semantic_priors.py \
      --scene 4-28 --data-dir /home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 \
      --out-dir runs/semantic_cache/4-28 --signals dino,depth,sam2 \
      --max-width 1600 --data-factor 1 --test-every 8

For SAM2 use --sam2-stride N to process only every N-th training view
(automatic mask generation is expensive); default 4.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _load_dataset(args):
    from scaffold_gs.datasets import ColmapDataset

    return ColmapDataset(
        data_dir=args.data_dir,
        data_factor=args.data_factor,
        test_every=args.test_every,
        white_background=False,
        preload_images=False,
        max_width=args.max_width,
        cache_images_cpu=False,
        device="cuda",
    )


def _make_dino(device, model_dir):
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(model_dir).to(device).eval()
    return model


def _make_depth(device, model_dir):
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    model = AutoModelForDepthEstimation.from_pretrained(model_dir).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_dir)
    return model, proc


def _make_sam2(device, ckpt):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(ckpt), device=device
    )
    return SAM2AutomaticMaskGenerator(model, points_per_side=16)


def _dino_feat(model, img_np, device):
    """img_np: float32 CHW [0,1] -> patch features (Hp, Wp, 768) fp16."""
    import torchvision.transforms.functional as F

    c, h, w = img_np.shape
    hp, wp = h // 14, w // 14
    t = torch.from_numpy(img_np).to(device)
    t = F.resize(t.unsqueeze(0), (hp * 14, wp * 14), antialias=True)[0]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        out = model(pixel_values=t.unsqueeze(0))
    feat = out.last_hidden_state[0, 1:, :]  # drop CLS
    feat = feat.reshape(hp, wp, -1).float().cpu().numpy().astype(np.float16)
    return feat


def _depth_map(model, proc, img_np, device):
    c, h, w = img_np.shape
    img_uint8 = (np.transpose(img_np, (1, 2, 0)) * 255.0).astype(np.uint8)
    inputs = proc(images=img_uint8, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    pred = out.predicted_depth if hasattr(out, "predicted_depth") else out.depth
    depth = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h // 4 * 4, w // 4 * 4), mode="bilinear",
        align_corners=False,
    )[0, 0]
    return depth[::4, ::4].float().cpu().numpy().astype(np.float16)


def _sam2_regions(gen, img_np):
    """Return (region_id[H//4, W//4] uint16, area_frac dict id->float)."""
    c, h, w = img_np.shape
    img = (np.transpose(img_np, (1, 2, 0)) * 255.0).astype(np.uint8)
    masks = gen.generate(img)
    masks.sort(key=lambda m: m["area"], reverse=True)
    rid = np.zeros((h, w), dtype=np.uint16)
    area = {}
    for i, m in enumerate(masks[:32], start=1):
        rid[m["segmentation"]] = i
        area[i] = float(m["area"])
    total = float(h * w)
    area_frac = {k: v / total for k, v in area.items()}
    return rid[::4, ::4], area_frac


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--signals", default="dino,depth,sam2")
    p.add_argument("--max-width", type=int, default=1600)
    p.add_argument("--data-factor", type=int, default=1)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--sam2-stride", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dino-model-dir",
        default="/home/fansonglin/data_space/DCCA-GS/models/dinov2-base",
    )
    p.add_argument(
        "--depth-model-dir",
        default="/home/fansonglin/data_space/DCCA-GS/models/dav2-small",
    )
    p.add_argument(
        "--sam2-ckpt",
        default="/home/fansonglin/data_space/DCCA-GS/models/sam2.1_hiera_tiny.pt",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    signals = [s.strip() for s in args.signals.split(",")]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    cams = dataset.train_cameras
    print(f"[extract] {args.scene}: {len(cams)} training views; signals={signals}")

    dino = _make_dino(device, args.dino_model_dir) if "dino" in signals else None
    depth_m, depth_p = (
        _make_depth(device, args.depth_model_dir) if "depth" in signals else (None, None)
    )
    sam2 = _make_sam2(device, args.sam2_ckpt) if "sam2" in signals else None

    stats = {"scene": args.scene, "views": len(cams), "per_signal": {}}
    for sig in signals:
        (out_root / sig).mkdir(parents=True, exist_ok=True)
        stats["per_signal"][sig] = {"count": 0, "seconds": 0.0, "peak_mib": 0.0}

    for vi, cam in enumerate(cams):
        if "sam2" in signals and vi % args.sam2_stride != 0:
            continue
        img = cam.load_image_cpu_uint8().numpy().astype(np.float32) / 255.0
        for sig in signals:
            t0 = time.perf_counter()
            try:
                if sig == "dino":
                    feat = _dino_feat(dino, img, device)
                    np.savez(out_root / "dino" / f"{vi:05d}.npz", feat=feat)
                elif sig == "depth":
                    dep = _depth_map(depth_m, depth_p, img, device)
                    np.savez(out_root / "depth" / f"{vi:05d}.npz", depth=dep)
                elif sig == "sam2":
                    rid, area = _sam2_regions(sam2, img)
                    np.savez(out_root / "sam2" / f"{vi:05d}.npz", region=rid)
                    (out_root / "sam2" / f"{vi:05d}_area.json").write_text(
                        json.dumps(area)
                    )
                dt = time.perf_counter() - t0
                s = stats["per_signal"][sig]
                s["count"] += 1
                s["seconds"] += dt
                s["peak_mib"] = max(
                    s["peak_mib"],
                    torch.cuda.max_memory_allocated(device) / 1024**2,
                )
            except Exception as exc:  # per-view robustness
                print(
                    f"[extract] view {vi} signal {sig} FAILED: {exc}",
                    flush=True,
                )
        if vi % 50 == 0:
            print(f"[extract] view {vi}/{len(cams)} done", flush=True)

    (out_root / "extract_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True)
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
