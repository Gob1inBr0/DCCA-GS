"""基层渲染损失梯度通路冒烟测试（发车前阻塞项）。

设计文档：docs/02-design/基层渲染损失_B3满精度代价判定_实验设计_20261011.md §6。
更正后配方（见设计文档更正注记）：p0 去量化分支只在 step > 10000 激活
（generate_gaussians 内硬编码），200 步短训走不到该路径；本脚本绕开
trainer，直接以 step > 10000 调用 model.render(quant_mode="p0")，
与生产每 8 步的额外渲染走完全相同的代码路径。

同一模型、同一相机、同一步做两次完整前向+反向：
  probe A（w=0）：只有主渲染损失（0.8*L1 + 0.2*SSIM，与 trainer 同式）
  probe B（w>0）：主损失 + w * 基层渲染损失（quant_mode="p0"）

通过判据（退出码 0）：
  1. anchor_feat 梯度范数比值（B/A）可见地偏离 1（|log2| > 0.5，
     远超浮点噪声量级）——证明 p0 项梯度真的回到锚点特征；
  2. p0 渲染图与主渲染图平均绝对差 > 1e-4——证明两次取整路径真的
     走到（p0 图确实被粗化，而非与主图相同）。
任一不满足 → 退出码 1，阻塞发车。

用法（服务器）：
  cd /home/project2/DCCA-GS && \
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/home/T0ng/gsplat-main:. \
  /home/project2/miniconda3/envs/DCCA/bin/python scripts/p0r_smoke.py \
    --data-dir /dev/shm/octree_178
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from scaffold_gs.config import DataConfig, ModelConfig, TrainConfig
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.losses import l1_loss, ssim_loss
from scaffold_gs.model import get_model_class
from scaffold_gs.utils import set_random_seed


def _probe(model, cam, gt, background, step, weight, target_names):
    """One full forward+backward; returns grad norms / losses / image diff."""
    for p in model.parameters():
        p.grad = None
    out = model.render(
        cam, background, is_training=True, retain_grad=False,
        appearance_id=cam.appearance_id, step=step,
    )
    pred = out.image[0].permute(2, 0, 1)
    loss = 0.8 * l1_loss(pred, gt).mean() + 0.2 * ssim_loss(pred[None], gt[None])
    p0_loss_val = float("nan")
    img_diff = float("nan")
    if weight > 0:
        p0_out = model.render(
            cam, background, is_training=True, retain_grad=False,
            appearance_id=cam.appearance_id, step=step, quant_mode="p0",
        )
        p0_pred = p0_out.image[0].permute(2, 0, 1)
        p0_loss = 0.8 * l1_loss(p0_pred, gt).mean() \
            + 0.2 * ssim_loss(p0_pred[None], gt[None])
        p0_loss_val = float(p0_loss)
        img_diff = float((p0_pred - pred).abs().mean())
        loss = loss + weight * p0_loss
    loss.backward()
    named = dict(model.named_parameters())
    norms = {}
    for n in target_names:
        p = named.get(n)
        if p is None or not p.requires_grad:
            continue
        norms[n] = float(p.grad.norm()) if p.grad is not None else None
    return norms, float(loss), p0_loss_val, img_diff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step", type=int, default=10500,
                    help="must be > 10000 so the p0 dequant path engages")
    ap.add_argument("--weight", type=float, default=5.0,
                    help="p0 loss weight for probe B (large on purpose)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert args.step > 10000, "step must exceed 10000 (hardcoded gate)"

    set_random_seed(args.seed)
    tcfg = TrainConfig()
    tcfg.data = DataConfig(
        data_dir=args.data_dir, data_factor=1, max_width=1600
    )
    tcfg.model = ModelConfig(model_name="hac_pp")
    tcfg.device = args.device

    model = get_model_class(tcfg.model.model_name)(tcfg.model, args.device)
    dataset = ColmapDataset(
        data_dir=tcfg.data.data_dir,
        data_factor=tcfg.data.data_factor,
        test_every=tcfg.data.test_every,
        white_background=tcfg.data.white_background,
        preload_images=tcfg.data.preload_images,
        max_width=tcfg.data.max_width,
        cache_images_cpu=tcfg.data.cache_images_cpu,
        device=args.device,
    )
    model.init_from_pcd(dataset.points, dataset.points_rgb, dataset.scene_scale)
    model.set_appearance(dataset.num_cameras)
    core = model.core
    if callable(getattr(core, "update_anchor_bound", None)):
        # Real training calls this once at step==10000; probe runs past that.
        core.update_anchor_bound()

    cam = list(dataset.train_cameras)[0]
    gt = dataset.get_image(cam)
    background = dataset.background

    # Grad-probe targets: whatever the parameter registry exposes for the
    # three coded fields (names differ by wrapper; match by substring).
    target_names = [
        n for n, p in model.named_parameters()
        if any(k in n.lower() for k in ("anchor_feat", "offset", "scaling"))
        and p.requires_grad
    ]
    print(f"[smoke] anchors={model.num_anchors} cam={cam.uid} "
          f"step={args.step} targets={target_names}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    norms_a, loss_a, _, _ = _probe(
        model, cam, gt, background, args.step, 0.0, target_names
    )
    norms_b, loss_b, p0_loss_b, img_diff = _probe(
        model, cam, gt, background, args.step, args.weight, target_names
    )
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else float("nan"))

    print(f"[smoke] probeA(w=0)   loss={loss_a:.5f}", flush=True)
    print(f"[smoke] probeB(w={args.weight:g}) loss={loss_b:.5f} "
          f"p0_loss={p0_loss_b:.5f} img_diff={img_diff:.6f}", flush=True)

    ok_grad = None
    for n in target_names:
        a, b = norms_a.get(n), norms_b.get(n)
        if a is None or b is None or a <= 0.0:
            print(f"[smoke] {n}: grad missing/zero (A={a} B={b}) — "
                  "cannot judge this param", flush=True)
            continue
        ratio = b / a
        log2r = abs(math.log2(ratio))
        verdict = "DIFF" if log2r > 0.5 else "same"
        print(f"[smoke] {n}: ||g||A={a:.6e} ||g||B={b:.6e} "
              f"ratio={ratio:.4f} ({verdict})", flush=True)
        if "anchor_feat" in n:
            ok_grad = log2r > 0.5
    ok_img = img_diff > 1e-4
    print(f"[smoke] peak_gpu_mem={peak:.2f} GB "
          f"(main+p0 graphs coexist, as in production)", flush=True)
    print(f"[smoke] VERDICT grad_flow={'PASS' if ok_grad else 'FAIL'} "
          f"p0_path={'PASS' if ok_img else 'FAIL'}", flush=True)
    return 0 if (ok_grad and ok_img) else 1


if __name__ == "__main__":
    sys.exit(main())
