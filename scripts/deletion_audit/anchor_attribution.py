"""锚点级概念归属与删除（概念删除 MVP 第 1 步）。

设计文档：docs/02-design/概念删除_压缩域最小验证设计.md
输入 train.py cmd_export 导出的 attributes.pth，输出删除目标锚点后的
attributes_deleted.pth 与归属报告 attribution_report.json。
网格编码（attrs["decoder"]）原样保留——保留正是为了让 grid_leakage_probe.py
测量"锚点删了、网格里还剩什么"。

选择方式（MVP）：
  --ids-file F     显式锚点下标文件，每行一个下标（服务器端由 SAM2 掩码投影生成）
  --sphere x,y,z,r 空间球选择（人工探针/冒烟测试用）
  --box x0,y0,z0,x1,y1,z1

支持度投影（mask_project_support_scores）为库函数：给定相机、逐视角区域 id 图
（runs/semantic_cache/<scene> 的 SAM2 产物）与目标区域 id，按投影最近邻投票得到
逐锚点支持度分数。相机约定为 OpenCV 惯例（x 右、y 下、z 向前）；接入
scaffold_gs 数据集时须先核对其 c2w 约定并在必要时转换（设计文档检查项 S1）。

本模块只依赖 torch/numpy，可在本地 CPU 运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# 逐锚点张量键：删除时按第一维过滤；其余键（config/voxel_size/decoder 等）原样保留
PER_ANCHOR_KEYS = (
    "anchor",
    "offset",
    "mask",
    "anchor_feat",
    "scaling",
    "rotation",
    "opacity",
)


def load_attrs(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def per_anchor_keys(attrs: dict, n: int) -> list[str]:
    """返回 attrs 中第一维等于锚点数 n 的已知逐锚点键（校验过的才删）。"""
    found = []
    for key in PER_ANCHOR_KEYS:
        value = attrs.get(key)
        if torch.is_tensor(value) and value.shape[:1] == (n,):
            found.append(key)
    return found


def sphere_mask(positions: torch.Tensor, center_xyz: tuple[float, float, float], radius: float) -> torch.Tensor:
    center = torch.tensor(center_xyz, dtype=positions.dtype)
    return (positions - center).norm(dim=-1) < radius


def box_mask(positions: torch.Tensor, box: tuple[float, float, float, float, float, float]) -> torch.Tensor:
    lo = torch.tensor(box[:3], dtype=positions.dtype)
    hi = torch.tensor(box[3:], dtype=positions.dtype)
    return ((positions >= lo) & (positions <= hi)).all(dim=-1)


def mask_project_support_scores(
    anchor_xyz: torch.Tensor,
    cameras: list[dict],
    region_maps: list[np.ndarray],
    target_region_id: int,
) -> dict:
    """把逐视角区域 id 图投影到锚点上，得到逐锚点支持度投票。

    cameras: 每个元素含 c2w (4,4)、K (3,3)、width、height；
    region_maps: 与 cameras 一一对应的 (h, w) uint16 数组，0 表示不属于任何区域，
                 分辨率可以低于原视图（内部按 map_w/width 缩放采样坐标）。
    返回 dict：scores (N,) = 命中目标区域的视角占比；votes (N,) 命中次数；
               votes_any (N,) 投进任意区域的次数（分母参考）。
    """
    n_views = len(cameras)
    assert len(region_maps) == n_views, "cameras 与 region_maps 数量须一致"
    n = anchor_xyz.shape[0]
    votes = torch.zeros(n, dtype=torch.int32)
    votes_any = torch.zeros(n, dtype=torch.int32)
    for cam, region in zip(cameras, region_maps):
        R = torch.tensor(np.asarray(cam["c2w"])[:3, :3], dtype=anchor_xyz.dtype)
        t = torch.tensor(np.asarray(cam["c2w"])[:3, 3], dtype=anchor_xyz.dtype)
        K = torch.tensor(np.asarray(cam["K"]), dtype=anchor_xyz.dtype)
        p_cam = (anchor_xyz - t) @ R  # (R^T (p - t))^T == (p - t) @ R
        z = p_cam[:, 2]
        valid = z > 1e-6
        u = K[0, 0] * p_cam[:, 0] / z.clamp_min(1e-6) + K[0, 2]
        v = K[1, 1] * p_cam[:, 1] / z.clamp_min(1e-6) + K[1, 2]
        map_h, map_w = region.shape[:2]
        scale_x = map_w / float(cam["width"])
        scale_y = map_h / float(cam["height"])
        mu = torch.round(u * scale_x).long()
        mv = torch.round(v * scale_y).long()
        in_bounds = valid & (mu >= 0) & (mu < map_w) & (mv >= 0) & (mv < map_h)
        rid = torch.from_numpy(
            region[np.clip(mv.numpy(), 0, map_h - 1), np.clip(mu.numpy(), 0, map_w - 1)]
        ).int()  # CPU torch 不支持 uint16 比较，转 int32
        votes += ((rid == target_region_id) & in_bounds).int()
        votes_any += ((rid > 0) & in_bounds).int()
    scores = votes.float() / max(1, n_views)
    return {"scores": scores, "votes": votes, "votes_any": votes_any}


def delete_anchors(attrs: dict, remove_mask: torch.Tensor) -> tuple[dict, dict]:
    """从 attrs 中删除 remove_mask 为真的锚点；网格/解码器参数原样保留。

    返回 (new_attrs, info)。逐锚点键按第一维过滤并断言形状一致；
    其余键（含 decoder 状态字典）原样引用保留，不做任何修改。
    """
    n = attrs["anchor"].shape[0]
    assert remove_mask.shape == (n,), "remove_mask 长度须等于锚点数"
    keep = ~remove_mask
    filtered, skipped = [], []
    out = {}
    for key, value in attrs.items():
        if key in PER_ANCHOR_KEYS and torch.is_tensor(value) and value.shape[:1] == (n,):
            out[key] = value[keep].clone()
            filtered.append(key)
        else:
            out[key] = value
            skipped.append(key)
    info = {
        "n_total": int(n),
        "n_removed": int(remove_mask.sum()),
        "n_kept": int(keep.sum()),
        "filtered_keys": filtered,
        "untouched_keys": skipped,
    }
    return out, info


def select_mask(attrs: dict, args: argparse.Namespace) -> torch.Tensor:
    n = attrs["anchor"].shape[0]
    positions = attrs["anchor"].float()
    if args.ids_file:
        ids = np.loadtxt(args.ids_file, dtype=int).reshape(-1)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.from_numpy(ids)] = True
        return mask
    if args.sphere:
        x, y, z, r = (float(v) for v in args.sphere.split(","))
        return sphere_mask(positions, (x, y, z), r)
    if args.box:
        box = tuple(float(v) for v in args.box.split(","))
        return box_mask(positions, box)
    raise SystemExit("须指定 --ids-file / --sphere / --box 之一")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", required=True, help="attributes.pth 路径")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids-file")
    parser.add_argument("--sphere", help="x,y,z,r")
    parser.add_argument("--box", help="x0,y0,z0,x1,y1,z1")
    parser.add_argument("--tag", default="", help="写入报告的概念标签（如 person/table）")
    args = parser.parse_args()

    attrs = load_attrs(args.attributes)
    remove_mask = select_mask(attrs, args)
    new_attrs, info = delete_anchors(attrs, remove_mask)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_attrs, out_dir / "attributes_deleted.pth")
    report = {
        "concept_tag": args.tag,
        "selection": {k: v for k, v in vars(args).items() if k in ("ids_file", "sphere", "box")},
        **info,
    }
    (out_dir / "attribution_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"[deletion] anchors {info['n_total']} -> {info['n_kept']} "
        f"(removed {info['n_removed']}); filtered={info['filtered_keys']}"
    )


if __name__ == "__main__":
    main()
