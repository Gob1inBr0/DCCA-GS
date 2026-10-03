#!/usr/bin/env python3
"""把稳态训练步的 GPU 设备时间拆到内核类别（回答"还有多少优化空间"）。

复用 profile_step_tax 的模型重建与稳态步循环；跑一个 profiler 活跃窗口
（默认 20 步），导出 chrome trace，把所有 GPU kernel/memcpy 事件按名字
归入类别桶，输出每类的设备侧总毫秒与占比。
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_step_tax import build, run_steps  # noqa: E402

BUCKETS = [
    ("光栅化主内核(rasterize)", re.compile(r"rasterize", re.I)),
    ("投影(projection)", re.compile(r"projection|project_gauss", re.I)),
    ("tile求交+排序(cub/sort)", re.compile(r"intersect|cub::|radix|scan", re.I)),
    ("卷积(SSIM)", re.compile(r"conv|winograd|implicit", re.I)),
    ("GEMM(全部MLP)", re.compile(r"gemm|cutlass|matmul|s16816|ampere_|sm90|sgemm|hmma", re.I)),
    ("正态CDF(熵模型erf)", re.compile(r"erf|ndtr|normal_cdf", re.I)),
    ("逐元素算子", re.compile(r"elementwise|vectorized_elementwise|unrolled_elementwise", re.I)),
    ("归约(index_add/sum)", re.compile(r"reduce|index_add|scatter|cumsum", re.I)),
    ("索引/排序(unique/nonzero)", re.compile(r"unique|nonzero|gather|sort", re.I)),
    ("Adam更新", re.compile(r"adam", re.I)),
    ("显存拷贝", re.compile(r"memcpy|memset", re.I)),
]


def bucket_of(name: str) -> str:
    for label, pat in BUCKETS:
        if pat.search(name):
            return label
    return "其他"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--max_width", type=int, default=1600)
    p.add_argument("--step", type=int, default=30000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--active", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    from scaffold_gs.utils import set_random_seed
    set_random_seed(args.seed)
    model, dataset, optim_cfg = build(
        args.ckpt, args.data_dir, args.data_factor, args.max_width, device
    )

    run_steps(model, dataset, optim_cfg, args, device, args.warmup)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile, schedule
    set_random_seed(args.seed)
    sched = schedule(wait=2, warmup=2, active=args.active, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
    ) as prof:
        for _ in range(args.active + 4):
            run_steps(model, dataset, optim_cfg, args, device, 1)
            prof.step()
    torch.cuda.synchronize()

    trace_path = Path(args.out + ".trace.json")
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    fam_ms = defaultdict(float)
    fam_cnt = defaultdict(int)
    raw_ms = defaultdict(float)
    for ev in events:
        if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(ev.get("dur", 0.0)) / 1000.0  # us -> ms
        name = ev.get("name", "?")
        fam_ms[bucket_of(name)] += dur
        fam_cnt[bucket_of(name)] += 1
        short = re.sub(r"<[^>]*>|\(.*?\)|void |at::native::", "", name)[:70].strip()
        raw_ms[short] += dur
    total = sum(fam_ms.values())
    rows = sorted(fam_ms.items(), key=lambda kv: -kv[1])
    print(f"总设备时间 {total:.1f} ms / {args.active} 步 = {total/args.active:.1f} ms/步\n")
    print(f"{'类别':<26}{'ms/步':>10}{'占比':>8}{'次数/步':>9}")
    for label, ms in rows:
        print(f"{label:<26}{ms/args.active:>10.1f}{ms/total*100:>7.1f}%{fam_cnt[label]/args.active:>9.0f}")
    print("\n-- 设备时间 Top 15 内核（清洗后名字）--")
    for name, ms in sorted(raw_ms.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {ms/args.active:8.2f} ms/步  {name}")
    Path(args.out).write_text(json.dumps(
        {label: {"ms_per_step": round(ms / args.active, 2), "count_per_step": fam_cnt[label] / args.active}
         for label, ms in rows} | {"_total_ms_per_step": round(total / args.active, 2)},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
