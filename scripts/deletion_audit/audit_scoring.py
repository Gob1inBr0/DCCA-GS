"""残留审计打分（概念删除 MVP 第 3 步，本地 CPU）。

设计文档：docs/02-design/概念删除_压缩域最小验证设计.md（判据见其中 P3 节）。
渲染本身复用现有评估路径（服务器上对原始/删除后 checkpoint 各跑一次
train.py 的 eval，得到两张同视角渲染图目录），本脚本只做打分：
  - 掩码内 PSNR 下降（原始渲染 vs 删除后渲染，仅统计概念区域）——删除对主体质量的影响；
  - 全图 PSNR 下降——删除的全局副作用；
  - --detector 检出打分为扩展位（重识别/检测器审计，MVP 未实现，接口预留）。

图像用 PIL 读取；三目录按文件名一一配对，缺失即报错（避免静默漏算）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    """掩码读取：单通道区域图，0 为背景、>0 为概念区域。"""
    from PIL import Image

    mask = np.asarray(Image.open(path), dtype=np.float64)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def masked_psnr(img: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """mask>0 的像素上的 PSNR（dB）。mask 与图像已对齐到同分辨率。"""
    diff2 = ((img - ref) ** 2).sum(axis=-1)[mask > 0]
    if diff2.size == 0:
        return float("nan")
    mse = float(diff2.mean())
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(1.0 / mse))


def score_pair_set(dir_original: Path, dir_deleted: Path, dir_masks: Path) -> list[dict]:
    originals = {p.name: p for p in dir_original.iterdir() if p.suffix.lower() in (".png", ".jpg")}
    results = []
    for name, orig_path in sorted(originals.items()):
        del_path = dir_deleted / name
        mask_path = dir_masks / name
        if not del_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"配对缺失：{name}（deleted={del_path.exists()}, mask={mask_path.exists()}）")
        img_o = load_image(orig_path)
        img_d = load_image(del_path)
        mask = _load_mask(mask_path)
        if mask.shape != img_o.shape[:2]:
            # 掩码允许是低分辨率区域图：最近邻放大到渲染分辨率
            from PIL import Image

            h, w = img_o.shape[:2]
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST),
                dtype=np.float64,
            )
        results.append(
            {
                "name": name,
                "psnr_masked_original_vs_deleted": masked_psnr(img_d, img_o, mask),
                "psnr_full_original_vs_deleted": masked_psnr(img_d, img_o, np.ones_like(mask)),
                "mask_fraction": float((mask > 0).mean()),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", required=True, help="原始模型渲染图目录（train.py eval 输出）")
    parser.add_argument("--deleted-dir", required=True, help="删除后模型渲染图目录")
    parser.add_argument("--masks-dir", required=True, help="逐视角概念区域掩码目录（与渲染图同名）")
    parser.add_argument("--out", required=True, help="审计报告 JSON 输出路径")
    parser.add_argument("--detector", default="", help="扩展位：检出打分器名称（MVP 未实现）")
    args = parser.parse_args()

    if args.detector:
        print(f"[audit] --detector {args.detector}：MVP 未实现检出打分，扩展位预留（设计文档 P3）")
    results = score_pair_set(Path(args.original_dir), Path(args.deleted_dir), Path(args.masks_dir))
    masked = [r["psnr_masked_original_vs_deleted"] for r in results if np.isfinite(r["psnr_masked_original_vs_deleted"])]
    summary = {
        "n_views": len(results),
        "masked_psnr_drop_mean": float(np.mean(masked)) if masked else float("nan"),
        "per_view": results,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit] n_views={summary['n_views']} masked_psnr_drop_mean={summary['masked_psnr_drop_mean']:.3f} dB")


if __name__ == "__main__":
    main()
