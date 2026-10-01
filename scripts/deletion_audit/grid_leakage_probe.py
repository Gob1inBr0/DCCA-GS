"""网格泄漏探针（概念删除 MVP 第 2 步）：锚点删了，网格里还剩什么。

设计文档：docs/02-design/概念删除_压缩域最小验证设计.md（判据见其中 P2 节）。
原理：删除锚点后，hash grid 的参数原样保留。若被删区域的网格特征仍与存活锚点
特征一样"有信息"（范数相当、且与存活特征分布连续），说明网格仍在编码被删内容，
即存在泄漏；残留信息审计必须覆盖网格，而不只是锚点。

两个后端：
  dense（本地 CPU）：稠密网格 (G, G, G, C)，三线性采样用 F.grid_sample，
                     冒烟测试与本文件 CLI 调试走这条路；
  hacpp（服务器 CUDA）：对 HAC++ 的 mix_3D2D_encoding 前向取特征，逻辑相同，
                        但 _grid_encode 是 CUDA 自定义算子，只能在 5090 上跑。
                        接入点见 probe_features_hacpp 的说明。

本文件只依赖 torch/numpy，可在本地 CPU 运行 dense 后端。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def dense_sample_features(
    grid_volume: torch.Tensor,
    grid_min: tuple[float, float, float],
    grid_max: tuple[float, float, float],
    points: torch.Tensor,
) -> torch.Tensor:
    """在稠密网格上三线性采样特征。

    grid_volume: (C, Gx, Gy, Gz)，轴 [x, y, z]；points: (N, 3) 世界坐标，
    按 grid_min/grid_max 线性映射到 [-1, 1]，align_corners=True。
    """
    C = grid_volume.shape[0]
    lo = torch.tensor(grid_min, dtype=points.dtype)
    hi = torch.tensor(grid_max, dtype=points.dtype)
    coords = 2.0 * (points - lo) / (hi - lo).clamp_min(1e-9) - 1.0  # (N, 3)
    # grid_sample 坐标分量依次对应输入的最后一、倒数第二、倒数第三维，
    # 因此把 (C, Gx, Gy, Gz) 转成 (C, Gz, Gy, Gx) 后 x 分量才采样到 Gx 轴
    vol = grid_volume.permute(0, 3, 2, 1).unsqueeze(0)
    sampled = F.grid_sample(vol, coords.reshape(1, -1, 1, 1, 3), mode="bilinear", align_corners=True)
    return sampled.reshape(C, -1).T  # (N, C)


def probe_features_hacpp(model, points: torch.Tensor) -> torch.Tensor:
    """服务器端后端：用已加载 HAC++ 模型的 mix_3D2D_encoding 取特征。

    用法（5090，CUDA 环境）：
        features = probe_features_hacpp(model, points)
    model 为 scaffold_gs.hacpp.HACPlusModel 实例（attributes_deleted.pth 经标准
    加载路径重建，见 scaffold_gs/codec.py 的 attributes -> model 装载函数）；
    内部调用 model.encoding_xyz（mix_3D2D_encoding）前向。
    首次在服务器接入时核对返回维度 = 各编码 output_dim 之和（gaussian_model.py
    L102-L105），并在设计文档登记实测值。
    """
    device = next(model.parameters()).device
    feats = model.encoding_xyz(points.to(device))
    return feats.detach().cpu()


def _max_cos_to_set(features: torch.Tensor, reference: torch.Tensor, sample_cap: int) -> torch.Tensor:
    """逐点与参考集（随机子采样，上限 sample_cap 个）的最大余弦相似度。"""
    if reference.shape[0] > sample_cap:
        idx = torch.randperm(reference.shape[0])[:sample_cap]
        reference = reference[idx]
    a = F.normalize(features, dim=-1)
    b = F.normalize(reference, dim=-1)
    return (a @ b.T).amax(dim=1)


def leakage_statistics(
    features_deleted: torch.Tensor,
    features_surviving: torch.Tensor,
    features_baseline: torch.Tensor | None = None,
    cos_sample_cap: int = 512,
) -> dict:
    """泄漏统计量（判据见设计文档 P2）：
      norm_deleted / norm_surviving —— 被删区域特征能量相对存活区域的比例；
      max_cos_deleted_to_surviving  —— 被删特征与存活特征的最大余弦相似度分布；
      baseline（可选）              —— 远离任何锚点的随机位置的同一统计量，
                                        用来区分"网格在此处本来就有特征"和"泄漏"。
    判定不做硬编码阈值，由设计文档判据在报告解读时应用。
    """
    stats = {
        "n_deleted": int(features_deleted.shape[0]),
        "n_surviving": int(features_surviving.shape[0]),
        "norm_deleted_mean": float(features_deleted.norm(dim=-1).mean()),
        "norm_surviving_mean": float(features_surviving.norm(dim=-1).mean()),
        "max_cos_deleted_to_surviving_mean": float(
            _max_cos_to_set(features_deleted, features_surviving, cos_sample_cap).mean()
        ),
    }
    if features_baseline is not None and features_baseline.shape[0] > 0:
        stats["n_baseline"] = int(features_baseline.shape[0])
        stats["norm_baseline_mean"] = float(features_baseline.norm(dim=-1).mean())
        stats["max_cos_baseline_to_surviving_mean"] = float(
            _max_cos_to_set(features_baseline, features_surviving, cos_sample_cap).mean()
        )
    if stats["norm_surviving_mean"] > 0:
        stats["norm_ratio_deleted_over_surviving"] = stats["norm_deleted_mean"] / stats["norm_surviving_mean"]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-deleted", required=True, help="(N,C) .npy，被删锚点位置的网格特征")
    parser.add_argument("--features-surviving", required=True, help="(M,C) .npy，存活锚点位置的网格特征")
    parser.add_argument("--features-baseline", help="(K,C) .npy，可选：远离锚点的随机位置特征")
    parser.add_argument("--out", required=True, help="统计 JSON 输出路径")
    parser.add_argument("--cos-sample-cap", type=int, default=512)
    args = parser.parse_args()

    feats_del = torch.from_numpy(np.load(args.features_deleted)).float()
    feats_sur = torch.from_numpy(np.load(args.features_surviving)).float()
    feats_base = None
    if args.features_baseline:
        feats_base = torch.from_numpy(np.load(args.features_baseline)).float()
    stats = leakage_statistics(feats_del, feats_sur, feats_base, args.cos_sample_cap)
    Path(args.out).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
