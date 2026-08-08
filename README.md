# Scaffold-GS on gsplat

基于 [gsplat](https://github.com/nerfstudio-project/gsplat) 的
[Scaffold-GS](https://github.com/city-super/Scaffold-GS)
忠实核心实现（CVPR 2024），并预留 HAC / HAC++ 压缩方法接入点。

## 特性

- anchor 体素初始化（`voxel_size`，0 时自动取 1-NN 距离中位数）
- 每个 anchor 解码 K=10 个神经高斯：opacity / covariance / color 三个 MLP
- gsplat 光栅化（视锥预过滤 + 预计算颜色渲染）
- anchor 层级生长与剪枝（官方 `update_depth=3`、`update_init_factor=16` 等默认参数）
- COLMAP 训练 / 评估：PSNR、SSIM、LPIPS
- 官方兼容 PLY + MLP checkpoint 保存格式
- `BaseGaussianModel` + `MODELS` 注册表 + `CompressionCodec` 接口，
  后续 HAC / HAC++ 只需新增模型/编解码器，无需改训练器

## 环境安装

```bash
cd /Users/chen/Documents/scaffold-gs
python -m venv .venv && source .venv/bin/activate
pip install -e /Users/chen/Documents/gsplat-main
pip install -r requirements.txt
```

## 数据格式

COLMAP 场景：

```
data/garden/
├── images/          # 原始图像（或 images_4/ 已降采样）
└── sparse/0/
    ├── cameras.bin  # 或 .txt
    ├── images.bin
    └── points3D.bin
```

## 使用

```bash
# 训练（默认 30k 步，每 8 张留 1 张做评估）
python train.py train --data.data-dir data/garden --data.result-dir results/garden \
  --model.voxel-size 0.001 --model.appearance-dim 0

# 评估已有 checkpoint
python train.py eval --ckpt results/garden/ckpts/ckpt_30000.pth \
  --data.data-dir data/garden

# 导出稳定属性（HAC/HAC++ 输入格式 + 官方 PLY）
python train.py export --ckpt results/garden/ckpts/ckpt_30000.pth \
  --out-dir results/garden/export

# 压缩基线（v1 只导出未压缩属性；hac / hac_pp 预留）
python train.py compress --ckpt results/garden/ckpts/ckpt_30000.pth \
  --out-dir results/garden/bitstreams --codec none
```

复现 MipNeRF360 官方脚本时使用 `--model.appearance-dim 0`；
官方代码默认（BungeeNeRF 风格）为 `--model.appearance-dim 32`。

## 扩展 HAC / HAC++

1. 在 `scaffold_gs/model.py` 中新增 `HACModel` / `HACPlusModel` 并注册进 `MODELS`。
2. 复用 `renderer`、`growth`、`trainer`；只替换 `AnchorDecoder.predict_gaussians()`
   为哈希网格上下文解码。
3. 在 `scaffold_gs/codec.py` 注册 `hac` / `hac_pp` 编解码器，`compress` 子命令即可使用。

## 测试

```bash
pytest tests/ -x -q
```

`test_render_smoke.py` 需要 CUDA GPU；模型/生长单测可在 CPU 上运行。
