# PHG (PKUGS-HAC-Gsplat)

基于 [gsplat](https://github.com/nerfstudio-project/gsplat) 的
Scaffold-GS / HAC++ 神经高斯压缩框架。当前主线为 **V2**：
I1（层级上下文）可选、I2（内容感知量化）默认开、I6（渲染敏感度监督）可选。

## 特性

- anchor 体素初始化（`voxel_size`，0 时自动取 1-NN 距离中位数）
- 每个 anchor 解码 K=10 个神经高斯：opacity / covariance / color 三个 MLP
- gsplat 光栅化：视锥预过滤 + 预计算颜色渲染，packed 模式省显存
- anchor 层级生长与剪枝（生长统计使用 `torch.scatter_reduce` 实现）
- COLMAP 训练 / 评估：PSNR、SSIM、LPIPS

### 三个创新点

| 创新点 | 内容 | 默认 |
| --- | --- | --- |
| I1 | 尺度感知层级 anchor-hash 上下文：`concat(base, parent, level)`，解码端可重算 | 关 |
| I2 | 内容感知公式量化：`Q = Q0 * (1 + tanh(z) * α)`，complexity MLP 预测 Q | 开 |
| I6 | 渲染敏感度加权监督：梯度 EMA → 相对归一化 → 监督 complexity MLP | 关 |

### 压缩管线

- 几何：anchor 坐标经 GPCC（`tmc3`）压缩
- 属性：feat / scaling / offsets / masks / hash 网格走官方 `arithmetic` 熵编码
- 体积口径与官方 HAC++ 一致：属性流 + 几何 + hash + masks + header
  + **MLP 权重（32 bit/参数）** + xyz 边界（192 bit），bit-exact roundtrip 校验

### 工程要点

- `BaseGaussianModel` + `MODELS` 注册表 + `CompressionCodec` 接口，
  新增模型/编解码器无需改训练器
- 修复了 growth 统计的显存泄漏（约 10MB/步，导致 4-28 长训 OOM 的根因）
- `tile_size` 可配：5090（sm_120）用 32，64 会超过 gsplat 内核共享内存上限
- `max_width` 复刻官方 `resolution=-1` 评估口径（3795×2134 → 1600×899）
- I5（格点矢量量化 VQ）实验保留在 `i5-vq` 分支，未合并

## 目录结构

```
PHG/
├── train.py                 # CLI：train / eval / export / compress
├── scaffold_gs/
│   ├── config.py            # Data/Model/Optim 配置（tyro 可解析）
│   ├── datasets.py          # COLMAP 加载、max_width 缩放
│   ├── model.py             # BaseGaussianModel + ScaffoldGSModel
│   ├── hacpp.py             # HAC++ 模型适配 + 熵编码 codec
│   ├── hac_core.py          # HACCoreView（唯一访问核心私有属性的入口）
│   ├── renderer.py          # gsplat 光栅化封装
│   ├── growth.py            # 生长/剪枝 + 统计
│   └── trainer.py           # 训练循环 / 评估 / checkpoint
├── hacplus/                 # vendored 官方 HAC++ 核心
└── scripts/
    ├── eval_decoded.py      # 解码 bitstream 并评估
    ├── rd_sweep.py          # 后处理 RD 消融（q_scale / mask 比例）
    ├── sensitivity_gate.py  # I6 相关性 + 离线 RD 上界
    ├── sensitivity_rd_sweep.py
    ├── sweep_mlp_complexity.py  # complexity MLP 架构扫描（并行）
    └── volume_breakdown.py
```

## 环境

```bash
# 本地（仅开发/阅读，训练在 5090）
cd /Users/chen/Documents/PHG
python -m venv .venv && source .venv/bin/activate
pip install -e /Users/chen/Documents/gsplat-main
pip install -r requirements.txt

# 5090（HAC++ CUDA 扩展环境）
conda activate HAC_5090_a100
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH   # tmc3/GPCC
```

## 使用

### 训练（4-28 对比协议）

```bash
python train.py train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir <COLMAP场景> --cfg.data.result-dir runs/<tag> \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 50 --cfg.model.n-offsets 10 \
  --cfg.model.appearance-dim 0 --cfg.model.ratio 1 --cfg.model.tile-size 32 \
  --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
  --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
  --cfg.model.sensitivity-weight 0.001 \
  --cfg.optim.max-steps 90000 --cfg.optim.update-until 45000 \
  --cfg.optim.eval-steps 30000 60000 90000 --cfg.optim.save-steps 30000 60000 90000 \
  --cfg.optim.lambda-rate 0.004 --cfg.optim.mask-lr-final 0.002 \
  --cfg.optim.start-stat 500 --cfg.optim.update-from 1500 --cfg.optim.update-interval 100
```

开启 I1（注意 `start_iter` 必须早于 grid MLP 首次使用，短实验用 300）：

```bash
--cfg.model.hierarchical-context --cfg.model.hierarchical-context-start-iter 300
```

complexity MLP 架构（`8→hidden→3`，扫描出的最优为 hidden=32、1 层）：

```bash
--cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1
```

### 压缩与评估

```bash
# 熵编码
python train.py compress --cfg.ckpt runs/<tag>/ckpts/ckpt_90000.pth \
  --cfg.out-dir runs/<tag>/bitstreams --cfg.codec hac_pp

# 解码 bitstream 并评估（大场景必须 --no-preload-images）
python scripts/eval_decoded.py --artifact-dir runs/<tag>/bitstreams \
  --data-dir <COLMAP场景> --result-dir runs/<tag>/decoded_eval \
  --max-width 1600 --no-preload-images
```

### 其它工具

```bash
# complexity MLP 架构扫描（并行跑多个 hidden:layers 配置）
python scripts/sweep_mlp_complexity.py --data-dir <场景> --result-root runs/mlp_sweep \
  --max-steps 15000 --update-until 7500 --sensitivity \
  --configs "25:1;32:1;64:1;64:2;128:1" --parallel 2

# I6 离线分析（相关性 + 离线 RD 上界）
python scripts/sensitivity_gate.py --ckpt <ckpt> --data-dir <场景> --result-dir runs/sens_gate

# 后处理 RD 消融（q_scale / mask 比例）
python scripts/rd_sweep.py --ckpt <ckpt> --data-dir <场景> --result-dir runs/rd_sweep
```

## 实验结果（4-28，1600 宽，官方体积口径含 MLP）

| 方案 | PSNR | SSIM | LPIPS | 体积 |
| --- | --- | --- | --- | --- |
| h25 90k（I2+I6） | 28.637 | 0.8922 | 0.2767 | 5.524 MB |
| h25 30k | 27.866 | 0.8866 | 0.2762 | ~5.91 MB |
| h32 30k | 27.879 | 0.8867 | 0.2758 | 6.013 MB |
| Web_Scan 30k（I2+I6） | 25.743 | 0.8526 | 0.1391 | ~3.83 MB |

参考行：之前 ours 90k = 28.563 / 0.8882 / 0.2982 / 6.355 MB；
HAC++ 论文 = 28.311 / 0.8900 / 0.2932 / 6.946 MB。

## 测试

```bash
pytest tests/ -q
```

5090 `HAC_5090_a100` 环境下当前 **19 passed**。`test_hacpp_smoke.py`
需要 HAC++ CUDA 扩展；`test_render_smoke.py` 需要 GPU。

## 已知注意点

- `tile_size=64` 在 sm_120 上报共享内存超限，用 32
- 4-28 1200 张图预载约 28GB，评估/训练务必 `--no-preload-images`
- I1 若用默认 `start_iter=12000`，会在训练分支 step>10000 出现输入维度不匹配；
  要么配 `hierarchical_context_start_iter < 10000`（如 300），要么先修训练分支
- 5090 上不要占用/杀死其他用户的进程；长训练建议用带“空卡检测+自动重试”的
  runner 脚本
