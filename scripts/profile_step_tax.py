#!/usr/bin/env python3
"""测量一个训练步里 CPU-GPU 同步税的占比（GPU 审查报告假设一）。

方法：加载一个 30000 步的 checkpoint（稳态，码率损失已激活，锚点增删
窗口已结束），按 run_training 的每步主体（渲染 → 损失 → 反向 → 优化器）
连续跑若干步，分两个口径输出：

1. 不挂分析器的墙钟：warmup 后连续 N 步，只在整个窗口末尾同步一次，
   得到每步真实耗时；
2. torch.profiler 窗口：统计阻塞类 CUDA API（设备同步、流同步、事件
   查询、DtoH 拷贝）的 CPU 侧驻留时间与调用次数，以及 CUDA kernel 的
   设备侧总占用时间，从而得到"CPU 被同步卡住的时间占比"与"GPU 实际
   忙碌占比"。

旧代码与新代码各跑一次本脚本（同一 checkpoint / 同一场景 / 同一随机
种子 / 同一张卡），两个 JSON 即可直接对比。
"""
import argparse
import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path

import torch

from scaffold_gs.config import ModelConfig, OptimConfig
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.losses import l1_loss, ssim_loss
from scaffold_gs.model import get_model_class
from scaffold_gs.utils import set_random_seed

# 阻塞类 CUDA 运行时 API：CPU 线程会停在里面的调用（torch.profiler 以
# cuda* Runtime 事件呈现，self_cpu_time 即 CPU 驻留时间）。
SYNC_APIS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaEventQuery",
    "cudaMemcpy",  # 非 async 版本天然阻塞
)
# 训练热路径里已知的强制同步来源（按审查报告的定位，用于交叉核对）
EXPECTED_SYNC_PRODUCERS = ("aten::nonzero", "aten::item", "aten::_local_scalar_dense")


def _filter_cfg(cls, d):
    allowed = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in dict(d).items() if k in allowed})


def build(ckpt_path, data_dir, data_factor, max_width, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = _filter_cfg(ModelConfig, ck["model_config"])
    optim_cfg = _filter_cfg(OptimConfig, ck["optim_config"])
    model = get_model_class(ck["model_name"])(model_cfg, str(device))
    sd = dict(ck["model_state"])
    if "_x_bound_min" in sd:
        # trainer.load_checkpoint 的等价流程：先建模型再喂状态
        model.load_state_dict(sd)
    else:
        model.load_state_dict(sd)
    model.voxel_size = ck.get("voxel_size", model_cfg.voxel_size)
    model.spatial_lr_scale = float(ck.get("spatial_lr_scale", 1.0))
    dataset = ColmapDataset(
        data_dir=data_dir,
        data_factor=data_factor,
        test_every=8,
        white_background=False,
        preload_images=False,
        max_width=max_width,
        cache_images_cpu=True,
        device=str(device),
    )
    model.set_appearance(dataset.num_cameras)
    model.create_optimizer(optim_cfg)
    if ck.get("optimizer_state") is not None:
        model.optimizer.load_state_dict(ck["optimizer_state"])
    stats = ck.get("stats")
    if stats is not None:
        for key in ("opacity_accum", "offset_gradient_accum", "offset_denom",
                    "anchor_demon", "max_radii2D"):
            if isinstance(stats.get(key), torch.Tensor):
                setattr(model, key, stats[key].to(device))
    model.train()
    return model, dataset, optim_cfg


def run_steps(model, dataset, optim_cfg, args, device, step_iter, record=None):
    """run_training 在 update 窗口之外的每步主体（30000 步 = 稳态）。"""
    background = dataset.background
    train_cams = list(dataset.train_cameras)
    for _ in range(step_iter):
        i = random.randrange(len(train_cams))
        cam = train_cams[i]
        model.update_learning_rate(args.step)
        out = model.render(
            cam, background, is_training=True, retain_grad=False,
            appearance_id=cam.appearance_id, step=args.step,
        )
        gt = dataset.get_image(cam)
        pred = out.image[0].permute(2, 0, 1)
        ll1 = l1_loss(pred, gt).mean()
        ssim = ssim_loss(pred[None], gt[None])
        if out.gaussians.xyz.shape[0] > 0:
            scale_reg = out.gaussians.scales.prod(dim=1).mean()
        else:
            scale_reg = torch.zeros((), device=device)
        loss = (
            (1.0 - optim_cfg.lambda_dssim) * ll1
            + optim_cfg.lambda_dssim * ssim
            + optim_cfg.scale_reg_lambda * scale_reg
        )
        rate_term = getattr(model, "rate_loss_term", None)
        if rate_term is not None:
            loss = loss + rate_term(out.gaussians, args.step)
        for name in ("sensitivity_supervision", "semantic_supervision",
                     "spa_loss_term"):
            fn = getattr(model, name, None)
            if fn is not None:
                loss = loss + fn(out.gaussians) if name != "spa_loss_term" \
                    else loss + fn()
        ladder_pen = getattr(out.gaussians, "ladder_penalty", None)
        if ladder_pen is not None:
            loss = loss + ladder_pen
        loss.backward()
        sens = getattr(model, "accumulate_sensitivity", None)
        if sens is not None and args.step >= getattr(
                model.cfg, "sensitivity_start_iter", 10**12):
            sens(out.gaussians)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        if record is not None:
            record.append(float(loss.detach()))
        del out, loss, pred, gt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--max_width", type=int, default=1600)
    p.add_argument("--step", type=int, default=30000,
                   help="传入渲染/码率路径的步号（>10000 才有完整码率热路径）")
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--timed", type=int, default=60)
    p.add_argument("--profile_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    args = p.parse_args()

    device = torch.device("cuda")
    set_random_seed(args.seed)
    model, dataset, optim_cfg = build(
        args.ckpt, args.data_dir, args.data_factor, args.max_width, device
    )
    n_anchors = model.num_anchors
    h, w = dataset.train_cameras[0].height, dataset.train_cameras[0].width

    # 阶段 1：warmup（图像缓存、分配器、惰性初始化）
    run_steps(model, dataset, optim_cfg, args, device, args.warmup)
    torch.cuda.synchronize()

    # 阶段 2：不挂分析器的墙钟
    t0 = time.perf_counter()
    run_steps(model, dataset, optim_cfg, args, device, args.timed)
    torch.cuda.synchronize()
    wall_per_step_ms = (time.perf_counter() - t0) * 1000.0 / args.timed

    # 阶段 3：profiler 窗口
    set_random_seed(args.seed)  # 与阶段 2 相同的相机序列，窗口内可复现
    random.randrange(len(dataset.train_cameras))  # 消耗掉阶段 1 尾迹不重要：
    # 这里重新播种后直接进入窗口，两臂脚本一致即可
    from torch.profiler import ProfilerActivity, profile, schedule

    losses = []
    background = dataset.background
    train_cams = list(dataset.train_cameras)
    sched = schedule(wait=3, warmup=3, active=args.profile_steps, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
    ) as prof:
        for _ in range(args.profile_steps + 6):
            i = random.randrange(len(train_cams))
            cam = train_cams[i]
            model.update_learning_rate(args.step)
            out = model.render(
                cam, background, is_training=True, retain_grad=False,
                appearance_id=cam.appearance_id, step=args.step,
            )
            gt = dataset.get_image(cam)
            pred = out.image[0].permute(2, 0, 1)
            ll1 = l1_loss(pred, gt).mean()
            ssim_l = ssim_loss(pred[None], gt[None])
            if out.gaussians.xyz.shape[0] > 0:
                scale_reg = out.gaussians.scales.prod(dim=1).mean()
            else:
                scale_reg = torch.zeros((), device=device)
            loss = (
                (1.0 - optim_cfg.lambda_dssim) * ll1
                + optim_cfg.lambda_dssim * ssim_l
                + optim_cfg.scale_reg_lambda * scale_reg
            )
            rate_term = getattr(model, "rate_loss_term", None)
            if rate_term is not None:
                loss = loss + rate_term(out.gaussians, args.step)
            spa = getattr(model, "spa_loss_term", None)
            if spa is not None:
                loss = loss + spa()
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            del out, loss, pred, gt
            prof.step()
    torch.cuda.synchronize()

    key_avgs = prof.key_averages()
    sync_rows, sync_us, sync_calls = [], 0.0, 0
    device_us = 0.0
    producers = {}
    for ev in key_avgs:
        name = ev.key
        self_cpu = getattr(ev, "self_cpu_time_total", 0) or 0
        self_dev = (
            getattr(ev, "self_device_time_total", None)
            if getattr(ev, "self_device_time_total", None) is not None
            else getattr(ev, "self_cuda_time_total", 0) or 0
        )
        device_us += self_dev
        if name in SYNC_APIS:
            sync_us += self_cpu
            sync_calls += ev.count
            sync_rows.append({
                "api": name, "count": ev.count,
                "cpu_ms": self_cpu / 1000.0,
                "per_step_cpu_ms": self_cpu / 1000.0 / args.profile_steps,
            })
        if name in EXPECTED_SYNC_PRODUCERS:
            producers[name] = {"count": ev.count,
                               "count_per_step": ev.count / args.profile_steps}
    sync_rows.sort(key=lambda r: -r["cpu_ms"])

    def _attr(ev, attr):
        return getattr(ev, attr, None)

    # 阶段 3 的窗口墙钟：从 profiler 事件里取 CPU 总跨度不可靠，改用
    # 阶段 2 的无分析器墙钟作为分母（保守：分析器下同步占比只会更高）。
    result = {
        "git_rev": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent
        ).stdout.strip() or "unknown",
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "step_arg": args.step,
        "num_anchors": int(n_anchors),
        "render_size": [int(w), int(h)],
        "gpu": torch.cuda.get_device_name(0),
        "wall_per_step_ms": round(wall_per_step_ms, 3),
        "timed_steps": args.timed,
        "profile_steps": args.profile_steps,
        "blocking_api_cpu_ms_total": round(sync_us / 1000.0, 3),
        "blocking_api_cpu_ms_per_step": round(sync_us / 1000.0 / args.profile_steps, 4),
        "blocking_api_calls_per_step": round(sync_calls / args.profile_steps, 2),
        "sync_share_of_wall": round(
            (sync_us / 1000.0 / args.profile_steps) / wall_per_step_ms, 4),
        "cuda_kernel_busy_ms_per_step": round(device_us / 1000.0 / args.profile_steps, 4),
        "kernel_busy_share_of_wall": round(
            (device_us / 1000.0 / args.profile_steps) / wall_per_step_ms, 4),
        "sync_apis": sync_rows,
        "known_sync_producers": producers,
        "last_losses": losses[-5:],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("sync_apis", "last_losses")},
                     indent=2, ensure_ascii=False))
    print("\n-- blocking APIs --")
    for row in sync_rows:
        print(f"  {row['api']:<28} calls/step={row['count'] / args.profile_steps:7.2f} "
              f"cpu_ms/step={row['per_step_cpu_ms']:8.4f}")
    print("\n-- known producer ops (per step) --")
    for name, s in producers.items():
        print(f"  {name:<28} calls/step={s['count_per_step']:7.2f}")


if __name__ == "__main__":
    main()
