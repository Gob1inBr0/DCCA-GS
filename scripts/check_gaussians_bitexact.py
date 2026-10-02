#!/usr/bin/env python3
"""generate_gaussians + prefilter 的逐位一致性检查。

用法：在旧代码与新代码的工作树各跑一次（PYTHONPATH 指向各自工作树），
同一 checkpoint、固定相机、eval 路径（no_grad、STE 量化路径，无 CUDA
原子累加，结果确定），把全部输出张量存成 .pt；再用 --compare 模式对两份
文件逐张量 torch.equal 比对。任何一行不一致都会退出码非零。
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_step_tax import build  # noqa: E402  复用同一套模型重建流程


def dump(args):
    device = torch.device("cuda")
    model, dataset, _optim = build(
        args.ckpt, args.data_dir, args.data_factor, args.max_width, device
    )
    model.eval()
    cams = dataset.train_cameras[: args.cams]
    payload = {"num_anchors": torch.tensor(model.num_anchors)}
    with torch.no_grad():
        for i, cam in enumerate(cams):
            vis = model.prefilter_anchors(cam)
            g = model.generate_gaussians(
                cam,
                visible_mask=vis,
                is_training=False,
                step=args.step,
            )
            for field in ("xyz", "colors", "opacities", "scales", "quats",
                          "neural_opacity", "gaussian_anchor_indices"):
                payload[f"cam{i}_{field}"] = getattr(g, field).detach().cpu()
            payload[f"cam{i}_visible"] = vis.detach().cpu()
            payload[f"cam{i}_n_rows"] = torch.tensor(g.xyz.shape[0])
    torch.save(payload, args.out)
    print(f"dumped {len(payload)} tensors for {len(cams)} cameras -> {args.out}")


def compare(args):
    a = torch.load(args.a, map_location="cpu", weights_only=False)
    b = torch.load(args.b, map_location="cpu", weights_only=False)
    assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
    bad = []
    for key in sorted(a):
        ta, tb = a[key], b[key]
        if ta.shape != tb.shape:
            bad.append(f"{key}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
        elif not torch.equal(ta, tb):
            n_diff = int((ta != tb).sum()) if ta.is_floating_point() else -1
            bad.append(f"{key}: values differ ({n_diff} elements)")
    if bad:
        print("BITEXACT FAIL:")
        for line in bad:
            print(" ", line)
        sys.exit(1)
    print(f"BITEXACT OK: {len(a)} tensors equal (old vs new worktree)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--data_dir", required=True)
    d.add_argument("--data_factor", type=int, default=4)
    d.add_argument("--max_width", type=int, default=1600)
    d.add_argument("--step", type=int, default=30000)
    d.add_argument("--cams", type=int, default=3)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    args = p.parse_args()
    if args.cmd == "dump":
        dump(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
