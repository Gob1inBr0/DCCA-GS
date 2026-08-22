# DCCA-GS

**Decoder-Reproducible Content-Adaptive Compression for Anchor-Based 3D Gaussian Splatting**

一个基于 [gsplat](https://github.com/nerfstudio-project/gsplat) +
[Scaffold-GS](https://arxiv.org/abs/2312.00109) + [HAC++](https://arxiv.org/abs/2501.12255)
的锚点式三维高斯泼溅压缩框架。

> 核心观点：**锚点压缩的瓶颈不是数量，而是位置；比特分配不是静态的，而是内容自适应的。**
> DCCA-GS 用「解码端可重算」的内容自适应量化（I2）、渲染敏感度监督（I6）、
> 训练侧 ADMM 锚点预算（SPA）、Mini-Splatting 式表面增密（depth-reinit）四条主线，
> 在**不新增任何侧信息、不改变码流契约**的前提下提升率失真（RD）曲线。

---

## ✨ 特性

| 模块 | 状态 | 一句话 |
| --- | --- | --- |
| HAC++ 主链路 | ✅ 必需 | 锚点 + 哈希上下文 + 条件熵模型 + G-PCC + 算术编码，bit-exact roundtrip |
| **I2 内容自适应量化** | ✅ 默认开 | `Q = Q0 × (1 + tanh(z)·α)`，`mlp_complexity` 按内容预测量化步长 |
| **I6 渲染敏感度监督** | ✅ 推荐开 | 训练期用渲染梯度 EMA 监督复杂度网络，零侧信息 |
| **SPA 训练侧稀疏** | ✅ 默认开 | ADMM 硬投影 + 显式锚点预算，训练期而非编码期做剪枝 |
| **MiniSplat depth-reinit** | ✅ 默认开 | 生长停止时深度反投影增密，把锚点「铺」到场景表面，预算钉在增密前 |
| 语义先验（T-A2） | ⏸ 可选/默认关 | DINOv2 目标 + 8 维投影头，15k 自我锚点刷新；已证实是「比特换质量」 |
| MLP 权重量化 | ✅ 推荐 | per-channel PTQ + 静态算术编码，推荐 complexity/deform 8-bit、其余 16-bit |
| R4 attr-ctx | ⏸ 可选/默认关 | 训练后条件熵预测器（scaling），4-28 上约 -0.4% 体积 |
| I1 层级上下文 | ⚠️ 默认关 | 尺度感知 anchor-hash 上下文；增益有限，需注意 start_iter |

## 🧪 实验亮点

### playroom 30k（HAC++ 压缩后真实解码，1600 宽，λ=0.004）

| 配置 | total_MB | 解码 PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| 基线 SPA（I2+I6+SPA r0.85） | 1.859 | 30.221 | 0.9068 | 0.2809 |
| **MiniSplat+SPA（主路径）** | **1.905** | **30.420** | 0.9070 | 0.2787 |
| 语义 SPA | 2.197 | 30.338 | 0.9074 | 0.2784 |
| MiniSplat+语义+SPA | 2.257 | 30.458 | 0.9075 | 0.2784 |

**读法**：同总锚点预算下，MiniSplat 把锚点重排到表面 → **+0.199 dB、体积仅 +2.5%**
（「位置 > 预算」）；而语义监督 +0.117 dB 却要多花 +18.2% 体积（同比特 BD-rate +8.2%）。

### 4-28 大场景（110k，1600 宽）

| 方案 | PSNR | SSIM | LPIPS | total_MB |
| --- | ---: | ---: | ---: | ---: |
| **DCCA-GS（I2+I6）** | **28.823** | **0.8926** | **0.2771** | **5.485** |
| HAC++ 论文参考 | 28.311 | 0.8900 | 0.2932 | 6.946 |

完整实验数字见 [docs/data/experiments.csv](docs/data/experiments.csv)；分析见
[MiniSplat×SPA_实验报告](docs/03-reports/MiniSplat×SPA_实验报告.md)。

## 🚀 快速开始

### 环境

```bash
# 本地（仅开发/阅读；训练在 5090）
python -m venv .venv && source .venv/bin/activate
pip install -e /path/to/gsplat
pip install -r requirements.txt

# 5090（HAC++ CUDA 扩展环境）
conda activate HAC_5090_a100
source scripts/env_5090.sh   # PYTHONPATH / tmc3(GPCC) / expandable_segments 等
```

完整环境（驱动/CUDA、包版本、GPCC、数据路径、常见坑）见
[docs/04-guides/环境说明.md](docs/04-guides/环境说明.md)，conda 清单见 [environment.yml](environment.yml)。

### 训练

```bash
python train.py train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir <COLMAP场景> --cfg.data.result-dir runs/<tag> \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 50 --cfg.model.n-offsets 10 \
  --cfg.model.appearance-dim 0 --cfg.model.tile-size 32 \
  --cfg.model.content-aware-quant \
  --cfg.model.sensitivity-enabled \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
  --cfg.model.mini-splat-enabled \
  --cfg.optim.max-steps 30000 --cfg.optim.update-until 15000 \
  --cfg.optim.eval-steps 30000 --cfg.optim.save-steps 30000 \
  --cfg.optim.lambda-rate 0.004
```

> 默认主路径即为「I2+I6+SPA(0.85)+MiniSplat」（[config.py](scaffold_gs/config.py) 中
> `spa_enabled=True`、`mini_splat_enabled=True`）。关布尔开关用 `--no-*`（tyro 不认 `False`）。

### 压缩与评估

```bash
python train.py compress --cfg.ckpt runs/<tag>/ckpts/ckpt_30000.pth \
  --cfg.out-dir runs/<tag>/bitstreams --cfg.codec hac_pp

python scripts/eval_decoded.py --artifact-dir runs/<tag>/bitstreams \
  --data-dir <COLMAP场景> --result-dir runs/<tag>/decoded_eval \
  --max-width 1600 --no-preload-images
```

### 其它工具

```bash
pytest tests/ -q                                  # 单元/冒烟测试
python scripts/sweep_mlp_complexity.py ...        # complexity MLP 架构扫描
python scripts/mlp_quant_sweep.py ...             # MLP 量化位宽扫描
python scripts/rd_sweep.py ...                    # 后处理 RD 消融
python scripts/extract_semantic_priors.py ...     # DINOv2/深度/SAM2 离线提取
python scripts/semantic_gate.py ...               # 语义相关性门 + 目标导出
```

## 🧱 目录结构

```text
DCCA-GS/
├── train.py                     # CLI：train / eval / export / compress
├── scaffold_gs/                 # 本项目逻辑
│   ├── config.py                # Data/Model/Optim/Compress 配置（tyro）
│   ├── datasets.py              # COLMAP 加载、max_width、CPU 缓存
│   ├── model.py                 # BaseGaussianModel + MODELS 注册表
│   ├── hacpp.py                 # HAC++ 适配 + 熵编码 codec + 监督接入
│   ├── hac_core.py              # HACCoreView（核心私有属性唯一入口）
│   ├── mini_splat.py            # Mini-Splatting depth-reinit（新）
│   ├── semantic_targets.py      # DINO 逐锚点目标构建/刷新（新）
│   ├── mlp_quant.py             # MLP per-channel 量化 + 算术编码
│   ├── attr_ctx.py              # R4 条件熵预测器
│   ├── renderer.py / growth.py / trainer.py / losses.py / codec.py
├── hacplus/                     # vendored 官方 HAC++ 核心
│   └── scene/gaussian_model.py  # 锚点/MLP/SPA/语义状态增密与同步
├── scripts/                     # 扫描/收集/绘图/审计/门控脚本
├── tests/                       # pytest
└── docs/                        # 文档中心（见 docs/README.md）
```

## 📚 文档

- [docs/README.md](docs/README.md) — 文档中心索引（分类、命名、维护规则）
- [docs/03-reports/MiniSplat×SPA_实验报告.md](docs/03-reports/MiniSplat×SPA_实验报告.md) — 本方向核心报告 + 后续实验设计
- [docs/03-reports/消融实验汇总.md](docs/03-reports/消融实验汇总.md) — 全方向消融结论
- [docs/02-design/语义先验实验设计.md](docs/02-design/语义先验实验设计.md) — 语义先验设计（含 Stage A/B 门控）
- [docs/02-design/SPA训练侧实验设计.md](docs/02-design/SPA训练侧实验设计.md) — SPA 训练侧稀疏设计
- [docs/06-planning/HANDOVER.md](docs/06-planning/HANDOVER.md) — 新 Agent 交接（环境/命令/坑/状态）

## 🔬 复现口径

- 评估：`--data-factor 1 --max-width 1600 --test-every 8`；全分辨率数字不可比。
- 体积：HAC++ 解码必需载荷 = G-PCC 几何 + 属性流(feat/scaling/offsets/masks/hash) +
  header + MLP 权重(32 bit/参数) + xyz 边界(192 bit)；bit-exact roundtrip 校验。
- 数字来源：`docs/data/experiments.csv` + 5090 `runs/*/metrics.jsonl`、`bitstreams/hac_meta.json`。

## 🧪 测试

```bash
conda activate HAC_5090_a100 && source scripts/env_5090.sh
pytest tests/ -q
```

`test_hacpp_smoke.py` 需要 HAC++ CUDA 扩展；`test_render_smoke.py` 需要 GPU。

## 🗺️ Roadmap / 当前状态

- ✅ MiniSplat depth-reinit 已合入并成为默认路径；语义先验方向暂停（同比特无净增益）
- 🔜 MiniSplat × 预算曲线、大场景 4-28、多 seed 泛化（见报告 §7 E1/E4/E7）
- 🔜 Mini-Splatting 完整版（blur split / contribution simplification）评估可行性

## 📄 许可与引用

本项目基于 HAC++ / Scaffold-GS / gsplat 的开源实现扩展，遵循上游许可；数据集与
预训练模型权重版权归各自作者。引用请标注本项目仓库与所基于的上游论文。
