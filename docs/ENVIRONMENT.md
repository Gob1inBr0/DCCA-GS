# DCCA-GS 环境配置说明（原 PHG）

更新日期：2026-08-17。本文件是 5090 训练/压缩/评估环境的唯一权威说明；
`scripts/env_5090.sh` 负责把环境变量一次性配好，`environment.yml` 记录关键包版本。

## 1. 硬件与驱动（5090）

- GPU：两张卡（`nvidia-smi` 索引 0/1）；`tile_size` 用 **32**（当前架构上限，
  换机器需按目标卡架构调整，见 §7 坑 2）。
- 驱动：580.159.03；CUDA 版本：13.0。
- 显存：单卡 32 GB；长训练建议 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 2. conda 环境

环境名：`HAC_5090_a100`，Python 3.10.20。

```bash
conda activate HAC_5090_a100
source scripts/env_5090.sh
```

`scripts/env_5090.sh` 设置：

| 变量 | 值 | 作用 |
| --- | --- | --- |
| `PHG_ROOT` | 仓库根目录 | 定位脚本/数据 |
| `PYTHONPATH` | `$PHG_ROOT` | 直接 import `scaffold_gs` / `hacplus` |
| `PYTHONNOUSERSITE` | `1` | 忽略 `~/.local` site-packages，避免污染 |
| `PATH` | conda env bin 在前 | 让 `tmc3`（GPCC）可执行 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 长训显存碎片控制 |

## 3. 关键包版本（5090 实测，2026-08-17）

| 包 | 版本 | 备注 |
| --- | --- | --- |
| torch | 2.7.1+cu128 | CUDA 12.8 构建；`~/.local` 另有 2.12.1，见坑 1 |
| torchvision | 0.22.1+cu128 | |
| numpy | 1.23.5 | |
| scipy | 1.15.3 | |
| gsplat | 1.5.3 | 开发用本地 `pip install -e <gsplat-main>` |
| pycolmap | 4.1.1 | 读 COLMAP sparse |
| tyro | 1.0.15 | train.py CLI |
| opencv-python | 4.13.0.92 | 图像读取 |
| lpips | 0.1.4 | 评估指标 |
| torchmetrics | 1.9.0 | SSIM/PSNR |
| tensorboard | 2.20.0 | 可选 |
| pytest | 9.1.1 | 测试（5090 上 19 passed） |

完整依赖见 [environment.yml](../environment.yml)。

## 4. CUDA 扩展（必须手动构建）

`scaffold_gs/hacpp.py` + `hacplus/` 需要官方 HAC-plus 仓库的 CUDA 扩展：

- `gridencoder`（哈希网格编码）
- `arithmetic`（算术编码）
- `simple_knn`（官方 HAC 初始化用；PHG 已内置 `knn_distances` 替代，环境里保留）
- `torch_scatter`（**可选**：PHG 用 `torch.scatter_reduce` 替代，未使用）

在 5090 上这些扩展已编译进 `HAC_5090_a100`（pip 版本显示 `0.0.0`）。
在新机器上需要从官方 HAC-plus 源码构建，不能用 pip 直接安装。

## 5. GPCC 几何编码

anchor 坐标用 GPCC（`tmc3`）压缩：

- 二进制位置：`$HOME/miniconda3/envs/HAC_5090_a100/bin/tmc3`；
- 调用方式：`hacplus/utils/gpcc_utils.py` 通过 PATH 找 `tmc3`；
- 所以 PATH 必须包含 conda env 的 `bin/`（`scripts/env_5090.sh` 已处理）。

## 6. 数据集路径（5090）

| 数据集 | 路径 |
| --- | --- |
| 4-28 | `/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28` |
| DB playroom | `/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db/playroom` |
| DB drjohnson | `/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db/drjohnson` |
| Mip360（9 场景） | `/home/fansonglin/xieliang/Chenzhenxin/dataset/360_v2/<scene>` |
| T&T（train/truck） | `/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/tandt/<scene>` |

## 7. 常见坑（必读）

1. **必须 `PYTHONNOUSERSITE=1`**：本机 `~/.local` 里有一个 torch 2.12.1，
   不设该变量时会被优先加载，导致 torchvision 导入崩溃
   （`operator torchvision::nms does not exist`）。实测 conda 环境 torch 为
   2.7.1+cu128。
2. **`tile_size` 用 32**：5090 当前架构上限；64 会超共享内存，16 在老配置可用。
3. **评估口径固定**：`--data-factor 1 --max-width 1600 --test-every 8`，
   全分辨率评估会低约 0.36 dB，数字不可比。
4. **4-28 必须 `--no-preload-images`**：全量预载 float32 会爆显存。
5. **不要覆盖正在运行的 runner 脚本**：bash 按文件偏移增量读取，运行中覆盖会导致
   训练后的 compress/eval 阶段 `unexpected EOF`。
6. **长训 OOM**：growth 统计已 detach（约 10MB/步泄漏已修）；packed 光栅化峰值
   用 `tile_size=32` + `expandable_segments` 控制。
7. **5090 纪律**：不杀其他用户进程；长训练用带“空卡检测+自动重试”的 runner
   （`scripts/runner_*`）。
8. **git 同步**：本地无法直连 GitHub 22 端口时，用 `git bundle` 经 5090 中转推送。

## 8. 快速开始

```bash
conda activate HAC_5090_a100
source scripts/env_5090.sh
pytest tests/ -q                      # 5090 预期 19 passed
python train.py train --help          # 训练参数
python train.py compress --help       # 压缩参数（hac_pp / attr_ctx / mask_keep_ratio）
```

训练/压缩/评估的标准命令见 [README.md](../README.md)。
