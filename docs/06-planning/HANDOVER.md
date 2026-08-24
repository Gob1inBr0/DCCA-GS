# DCCA-GS 项目交接文档（另一个 Agent 打开即用版）

更新：2026-08-22；分支：`main`；HEAD：`1b65191`
（5 组件方向：I2 + I6 + SPA + MiniSplat + MLP 量化；语义先验已暂停）

> 本文件的目标：一个没有上下文的新 agent 读完本文件后，能直接连上 5090、
> 复现已有实验、查看结果、继续排队实验，并知道所有已踩过的坑。

## 0. 30 秒上手

```bash
# 在 5090 上
cd /home/fansonglin/data_space/DCCA-GS/PHG
conda activate HAC_5090_a100
source scripts/env_5090.sh          # 设置 PYTHONPATH / PATH(tmc3) / PYTHONNOUSERSITE / CUDA_ALLOC_CONF
pytest tests/ -q                    # 数据/增长/语义/MiniSplat 单测在 5090 环境跑
python train.py train --help        # 训练参数
```

当前正在跑什么、看哪里：

```bash
ps -ef | grep -E 'train.py train|queue_|runner_' | grep -v grep
nvidia-smi
tail -f /home/fansonglin/data_space/DCCA-GS/runs/spa_fixed_launch.sh
tail -f /home/fansonglin/data_space/DCCA-GS/runs/spa_minisplat_launch.sh
```

所有实验数字在 `docs/data/experiments.csv`（2026-08-22 已含 MiniSplat/语义行），
5090 结果目录在 `/home/fansonglin/data_space/DCCA-GS/runs/`。

## 1. 项目是什么

DCCA-GS（原 PHG / PKUGS-HAC-Gsplat）是基于 gsplat 的 Scaffold-GS / HAC++ 神经高斯压缩框架，
目标是“模型可替换 + 稳定属性导出 + 编解码器接口化”的低耦合架构，在 HAC++ 主链路上
叠加创新点并给出量化分析。

### 1.1 三个仓库位置

| 位置 | 路径/地址 | 说明 |
| --- | --- | --- |
| GitHub | `git@github.com:Gob1inBr0/DCCA-GS.git`（迁移前为 `goblinIBigBro/PHG`，旧地址仍可推） | 唯一权威远端 |
| 本地 | `/Users/chen/Documents/DCCA-GS` | 开发/文档/画图 |
| 5090 | `/home/fansonglin/data_space/DCCA-GS/PHG` | 训练/压缩/评估 |

注意：本地 SSH(22) 被代理挡，但 **HTTPS 可直推**
（`git push https://github.com/Gob1inBr0/DCCA-GS.git main`，osxkeychain 存有凭据）；
bundle 经 5090 中转仅作备用（见 §8）。

### 1.2 创新点状态总表

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| HAC++ 核心 codec（`hac_pp`） | ✅ 必需 | GPCC 几何 + 哈希上下文 + 条件熵模型 + 算术编码；bit-exact roundtrip |
| I2 内容感知公式量化 | ✅ 默认开 | `Q = Q0×(1+tanh(z)×α)`，`mlp_complexity` 预测 Q；**输入已改为 4 维**（4 个结构量，不再带 4 个恒零） |
| I6 渲染敏感度监督 | ✅ 推荐开 | 仅训练期 EMA 监督，零侧信息；DB playroom 110k 贡献约 +0.064 dB，4-28 约 +0.1 dB |
| SPA 训练侧稀疏（ADMM 预算） | ✅ 默认开 | 固定预算 r=0.85 为默认；同锚点比编码端 top-k 高 5.5 dB（30k playroom） |
| MiniSplat depth-reinit | ✅ 默认开（新） | 15k 深度反投影增密；cell2 = +0.199 dB / +2.5% 体积（同预算）→ 主路径 |
| 语义先验 T-A2（DINO） | ⏸ 默认关 | 15k 自我锚点目标刷新；同比特 BD-rate +8.2%，大场景 4-28 −0.041 dB → 方向暂停 |
| MLP 权重量化 | ✅ 推荐 cd8/rest16 | complexity/deform 8-bit，其余 16-bit；16-bit 全量零损失，8-bit 全量会劣化 |
| R4 attr-ctx | ✅ 可选、默认关 | 训练后条件熵调整（scaling），4-28 90k λ0.002 上 -0.41% 体积 |
| I1 层级上下文 | ⚠️ 默认关 | 有 start_iter 坑；gains 小 |
| P0 条件熵 | ❌ 已关闭 | 离线增益 ≤2.3% < 3% 阈值 |
| CompGS 式残差编码（R） | ❌ 已关闭 | R0~R3 无增益/负增益；R4 以 attr-ctx 保留 |
| I5（VQ） | ❌ 已关闭 | 存档分支 `i5-vq` |
| I6 替换/侧信息 | ❌ 已关闭 | 相关性 ≤0.0086，真值侧信息反而更大 |

## 2. 环境（打开即用）

### 2.1 5090 环境

- conda 环境：`HAC_5090_a100`（Python 3.10.20）
- 关键版本：torch 2.7.1+cu128、torchvision 0.22.1+cu128、gsplat 1.5.3、
  pycolmap 4.1.1、tyro 1.0.15、lpips 0.1.4、pytest 9.1.1
- CUDA 扩展（源码构建，pip 显示 0.0.0）：`gridencoder`、`arithmetic`、`simple_knn`
- GPCC：`tmc3` 放在 conda env `bin/`，由 `hacplus/utils/gpcc_utils.py` 按 PATH 调用

环境文件：

- `scripts/env_5090.sh`：一次性设置 `PHG_ROOT`/`PYTHONPATH`/`PYTHONNOUSERSITE=1`/
  `PATH`（含 tmc3）/`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `environment.yml`：conda 依赖清单（含 torch 版本坑的注释）
- `../04-guides/环境说明.md`：完整环境说明（数据路径、坑、快速开始）

### 2.2 数据集路径（5090）

| 数据集 | 路径 |
| --- | --- |
| 4-28 | `/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28` |
| DB playroom | `$BASE/data/playroom`（`/home/fansonglin/data_space/DCCA-GS/data/playroom`）|
| DB drjohnson | 以 5090 `$BASE/data/` 实际为准（旧路径 `/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db/drjohnson` 备用）|
| Mip360（9 场景） | `/home/fansonglin/xieliang/Chenzhenxin/dataset/360_v2/<scene>` |
| T&T | `/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/tandt/{train,truck}` |

## 3. 代码结构

```text
PHG/
├── train.py                      # tyro 子命令：train / eval / export / compress
├── environment.yml
├── scaffold_gs/
│   ├── config.py                 # Data/Model/Optim/Compress 四组配置
│   ├── datasets.py               # COLMAP 读取、max_width 缩放、CPU 缓存
│   ├── model.py                  # BaseGaussianModel / MODELS 注册表
│   ├── hac_core.py               # HACCoreView（vendored core 的唯一稳定视图）
│   ├── hacpp.py                  # HACPlusModel + HACPlusCodec（encode/decode/迁移）
│   ├── renderer.py               # prefilter + gsplat rasterization
│   ├── growth.py                 # 生长/剪枝/统计（torch.scatter_reduce）
│   ├── trainer.py                # 训练循环（含 SPA/rate/sensitivity loss）
│   ├── losses.py
│   ├── mlp_quant.py              # 逐通道量化 + 静态算术编码
│   ├── attr_ctx.py               # R4 预测器（8-bit payload）
│   └── codec.py                  # CompressionCodec 接口
├── hacplus/                      # vendored 官方 HAC++ 核心
│   ├── scene/gaussian_model.py   # 锚点/MLP/SPA 状态/ADMM 剪枝
│   └── utils/                    # entropy_models / codec_consistency / gpcc_utils ...
├── scripts/                      # runner / 扫描 / 收集 / 绘图 / 审计（见 §4）
├── tests/                        # pytest（5090 上 19 passed）
└── docs/                         # 设计文档、实验报告、HANDOVER（本文件）
```

关键接口（改代码前先看）：

- `BaseGaussianModel`：`init_from_pcd / render / training_statis / adjust_anchor /
  optimizer_groups / state_dict / export_attributes`
- `HACCoreView`：唯一允许触碰 `core._anchor` 等私有属性的地方
- `HACPlusCodec(CompressionCodec)`：`encode(model, out_dir, **kw)` /
  `decode(artifact_dir, **kw)`
- `MODELS` / `CODECS` 注册表：新增模型/编解码器只需注册

## 4. 常用命令

### 4.1 训练（推荐配置 = I2+I6，dim50，h32，110k，λ0.002/0.004）

```bash
cd /home/fansonglin/data_space/DCCA-GS/PHG
source scripts/env_5090.sh

python train.py train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir <DATA_DIR> \
  --cfg.data.result-dir runs/<tag> \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 50 --cfg.model.n-offsets 10 \
  --cfg.model.appearance-dim 0 --cfg.model.ratio 1 --cfg.model.tile-size 32 \
  --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
  --cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1 \
  --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
  --cfg.model.sensitivity-weight 0.001 \
  --cfg.optim.max-steps 110000 --cfg.optim.eval-steps 110000 --cfg.optim.save-steps 110000 \
  --cfg.optim.lambda-rate 0.002 --cfg.optim.mask-lr-final 0.002 \
  --cfg.optim.start-stat 500 --cfg.optim.update-from 1500 \
  --cfg.optim.update-until 45000 --cfg.optim.update-interval 100
```

开 SPA 时追加：

```bash
--cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
```

关布尔开关用 `--no-*` 形式（tyro 不认“空格+False”）：

```bash
--cfg.model.no-sensitivity-enabled
--cfg.model.no-content-aware-quant --cfg.model.no-sensitivity-enabled
```

通用 cell runner（训练+compress+eval+MLP 量化一条龙）：

```bash
bash scripts/runner_phg_cell.sh <gpu> <scene> <data_dir> <lambda> <tag> \
  <max_steps> <update_until> [extra flags...]
```

### 4.2 压缩与评估

```bash
python train.py compress --cfg.ckpt <ckpt.pth> --cfg.out-dir <out>/bitstreams \
  --cfg.codec hac_pp [--cfg.mask-keep-ratio 0.5] [--cfg.attr-ctx <attr_ctx.pt>]

python scripts/eval_decoded.py --artifact-dir <out>/bitstreams \
  --data-dir <DATA_DIR> --result-dir <out>/decoded_eval \
  --data-factor 1 --max-width 1600 --no-preload-images
```

### 4.3 MLP 量化（推荐 cd8/rest16）

```bash
python scripts/mlp_quant_sweep.py --ckpt <ckpt.pth> --data-dir <DATA_DIR> \
  --result-dir <out>/mlp_quant_cd8_rest16 --data-factor 1 --max-width 1600 \
  --no-preload-images --skip-baseline \
  --group-bits mlp_complexity:8 mlp_deform:8 mlp_opacity:16 mlp_cov:16 \
               mlp_color:16 mlp_grid:16
```

### 4.4 RD 与审计脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/rd_sweep.py` | 固定 checkpoint 的 q-scale RD 扫描 |
| `scripts/plot_db_rd.py` | DB RD 双线图（无 SPA vs +SPA，线性轴） |
| `scripts/codec_efficiency.py` | 实际/估计比特效率（≈1 说明编码器贴合模型） |
| `scripts/kl_audit.py` | 模型交叉熵 vs 经验熵（KL 代理） |
| `scripts/residual_feasibility.py` | CompGS 式残差编码离线验证（已关闭） |
| `scripts/fit_attr_ctx.py` | 拟合 R4 预测器（`--fields scaling`） |
| `scripts/collect_queue_results.py` | 扫描结果目录 → 追加统一 CSV + 生成 SPA-RD JSON |

### 4.5 测试

```bash
pytest tests/ -q     # 5090 预期 19 passed
```

## 5. 实验结果（截至 2026-08-19）

所有数字都在 `../data/experiments.csv`（约 158 行，18 列：
`group,scene,run_id,variant,iteration,lambda,feat_dim,mlp_quant,psnr,ssim,lpips,
total_mb,anchors_trained,anchors_coded,metric_type,metric_value,notes,source`）。

### 5.1 4-28（1600 宽，解码必需 MiB）

| 方案 | PSNR | SSIM | LPIPS | total_MB |
| --- | --- | --- | --- | --- |
| PHG h32 90k（I2+I6） | 28.655 | 0.8921 | 0.2767 | 5.5604 |
| PHG 110k dim50 λ0.004 | 28.825 | 0.8926 | 0.2771 | 5.6471 |
| PHG 110k + MLP 量化 | 28.8235 | 0.8926 | 0.2771 | 5.4852 |
| PHG 90k λ0.002 | 28.988 | 0.8974 | 0.2708 | 8.3069 |
| HAC++（论文参考） | 28.311 | 0.8900 | 0.2932 | 6.9462 |

### 5.2 DB λ-RD（110k，I2+I6）

| 场景 | λ | 基线 PSNR / total_MB | 量化 PSNR / total_MB |
| --- | --- | --- | --- |
| playroom | 0.001 | 30.665 / 5.9653 | 30.6624 / 5.8024 |
| playroom | 0.002 | 30.731 / 4.3819 | 30.7288 / 4.2231 |
| playroom | 0.004 | 30.621 / 3.1710 | 30.6190 / 3.0072 |
| drjohnson | 0.001 | 30.090 / 9.7574 | 30.0934 / 9.6294 |
| drjohnson | 0.002 | 30.038 / 7.0156 | 30.0376 / 6.8549 |
| drjohnson | 0.004 | 30.048 / 4.9209 | 30.0443 / 4.7625 |

### 5.3 SPA 110k（playroom λ0.002，核心创新）

| 方案 | PSNR | SSIM | LPIPS | total_MB（基线/量化） |
| --- | --- | --- | --- | --- |
| Full（I2+I6+SPA 0.5） | 29.8908 | 0.8963 | 0.3072 | 1.1292 / 0.9639 |
| w/o SPA（I2+I6） | 30.7310 | 0.9130 | 0.2575 | 4.3819 / 4.2231 |
| w/o I6（I2+SPA） | 已完成，见 CSV | | | |
| w/o I2（SPA only） | 已完成，见 CSV | | | |

### 5.4 消融结论（110k playroom）

- I6：+0.064 dB（30.6671→30.7310），体积略降；
- I2：在 DB 上贡献很小（w/o I2 仅差 0.002 dB），主要价值在大尺度场景（4-28）；
- SPA：体积 -74%（4.38→1.13 MiB，量化后 0.96 MiB），PSNR -0.84 dB，适合低码率档；
- MLP 量化：每组约 -0.16 MiB、PSNR 变化 <0.004 dB。

### 5.5 关键分析结论

- 编码效率（实际/估计比特）：feat 1.0042、scaling 0.9634、offsets 1.0018、
  masks 1.0066、hash 1.0314、total 0.9942 → 算术编码器已贴紧模型；
- KL 审计（4-28 90k）：scaling 的模型交叉熵比无条件经验熵高 0.494 bit/符号
  （真实 KL 的下界，约 0.092 MiB / 1.65% total）；offsets/masks 接近；
- P0 条件熵 ≤2.3%、P0-2 邻居 ~0.1%、反向 ~0.4% → 关闭；
- 残差编码 R0~R3 无增益/负增益，R4 条件均值/对数尺度以 attr-ctx 保留（默认关）；
- I6 侧信息相关性 ≤0.0086 → 关闭；
- MLP 量化：16-bit 全量零损失；8-bit 全量 -0.079 dB 且体积反增（mlp_grid 8-bit
  使属性码率 +0.25 MiB）；opacity 对 8-bit 最敏感；推荐 cd8/rest16。

## 6. 码流契约与版本

- 体积口径：解码必需载荷 = G-PCC 锚点 + feat/scaling/offsets 流 + masks + hash +
  压缩后 MLP 载荷 + 边界/头部；不含 checkpoint/调试文件；
- bit-exact：编码/解码用同一条件模型与量化路径；`codec_roundtrip_diagnostics.json`
  记录符号/量化步长/整数 CDF 不匹配数（全 0）；
- `FORMULA_INPUT_VERSION = "formula_decoder_available_v2_4d"`（2026-08-19 由 v1 升到
  v2，因为 `mlp_complexity` 输入从 8 维（含 4 个恒零）改为 4 维）；
- **旧 v1 码流与新版不兼容**（版本强校验）；旧 checkpoint 自动迁移：
  `load_state_dict` 检测 `mlp_complexity.0.weight` 为 `[hidden, 8]` 时取前 4 列
  （数学上完全等价，因为后 4 列是死权重）；
- 评估协议：`--data-factor 1 --max-width 1600 --test-every 8`；全分辨率数字不可比。

## 7. 踩坑清单（必读）

1. `PYTHONNOUSERSITE=1` 必须设：`~/.local` 有 torch 2.12.1，会盖掉 conda 的
   2.7.1+cu128 并导致 torchvision 崩溃；
2. `tile_size=32`（当前卡架构上限），64 超共享内存；
3. 4-28 必须 `--no-preload-images`（全量预载 ~28GB 显存）；
4. 长训用 `expandable_segments:True`；growth 统计已 detach（历史 OOM 根因）；
5. **不要覆盖正在运行的 runner 脚本**（bash 按偏移读取，会 `unexpected EOF`）；
6. tyro 布尔开关用 `--cfg.model.no-xxx`，不能用 `--cfg.model.xxx False`；
7. SPA 预算必须锚定 `max(N_ref)`（`κ=ratio×N_t` 会几何塌缩到 80 个锚点）；
8. I6 依赖 I2（配置校验强制），所以“w/o I2”= I2 与 I6 一起关；
9. deform hidden 加载：`load_state_dict` 已按 checkpoint 形状自适应重建；
10. 体积口径差异：旧数字若没有 `bit_mlp`/MLP 载荷，不可直接比；
11. 5090 纪律：不杀其他用户进程；长训练用带空卡检测+自动重试的 runner；
12. 本地 git push 被代理挡（22 端口），用 bundle 经 5090 中转（见 §8）。

## 8. Git 同步（本地 ↔ GitHub ↔ 5090）

```bash
# 方式一（推荐）：HTTPS 直推（SSH 22 被代理挡）
cd /Users/chen/Documents/DCCA-GS
git push https://github.com/Gob1inBr0/DCCA-GS.git main

# 方式二（备用）：bundle 经 5090 中转
git bundle create /tmp/phg-main.bundle main
# scp 到 5090（可用 expect：scp /tmp/phg-main.bundle small5090:/tmp/phg-main.bundle）
# 在 5090 上
cd /home/fansonglin/data_space/DCCA-GS/PHG
git fetch /tmp/phg-main.bundle main:refs/heads/bundle-main
git push origin bundle-main:main
# 本地再 git fetch/pull 同步
```

5090 工作区保持 `i6-sens-replace` 或 `main` 均可（我们一直以文件同步+远端 main 为准，
不依赖 5090 本地分支与远端一致）。

## 9. 当前状态（2026-08-22，实时）

已完成并自动收集：

- DB 110k λ-RD（playroom/drjohnson × λ0.001/0.002/0.004，基线+量化）✅
- DB SPA 消融 30k 与 110k（含修正后的 I2+SPA、SPA-only）✅
- Mip360（garden/flowers/stump）与 T&T（train/truck）30k × λ0.002/0.004 ✅
- MLP 量化全位宽扫描 + 逐 MLP 8-bit 消融 ✅
- SPA 110k playroom λ0.002（29.8908 / 1.1292）✅
- **MiniSplat × SPA 固定预算 4 格**（playroom 30k，HAC++ 解码后）✅
  （cell1 30.221/1.859 → cell2 30.420/1.905 → cell3 30.338/2.197 → cell4b 30.458/2.257）
- **SPA 预算曲线 r={0.52,0.85,0.92,0.97} × baseline/语义**（playroom，解码后）✅
- **E1 MiniSplat × SPA 预算曲线** ✅（BD-PSNR +0.124 dB，BD-rate −8.6%）
- **E4 4-28 110k** ✅（SPA+Mini 28.736/5.449，SPA base 28.748/5.514，
  非SPA+Mini 28.759/5.433；大场景未见明显 Mini 增益）
- **4-28 B 组（非 SPA，30k）baseline vs 语义** ✅（28.308 vs 28.267，语义无净增益）
- **语义先验方向：暂停**（同比特 BD-rate +8.2%；大场景 −0.041 dB）
- **drjohnson 30k cell1/cell2 泛化** ✅（cell1 29.3899/2.0379，cell2 29.3146/2.0021；
  MiniSplat −0.073 dB / −1.8% 体积 → 场景依赖，非普适）

接下来（按优先级，详见 `../03-reports/MiniSplat×SPA_实验报告.md` §7）：

| 任务 | 优先级 |
| --- | --- |
| E1 MiniSplat × 预算曲线补齐 | P0 |
| E4 MiniSplat × 大场景 4-28 | P0 |
| E3 MiniSplat 参数敏感性 / E5 与 SPA 解耦 / E7 多场景多 seed | P1 |

监控：

```bash
ls /home/fansonglin/data_space/DCCA-GS/runs/                 # 最新 run 目录
tail -f /home/fansonglin/data_space/DCCA-GS/runs/*/train.log
```

## 10. 给下一个 Agent 的行动清单

1. 读本文件 + `../04-guides/环境说明.md` + `../04-guides/环境配置与交接.md` +
   `../06-planning/DCCA-GS_项目变更文档.md`；
2. 5090：`conda activate HAC_5090_a100 && source scripts/env_5090.sh`，
   `pytest tests/ -q` 跑单测；
3. 看 `../data/experiments.csv` 和 `runs/db_rd_110k.json` /
   `runs/db_spa_rd_110k.json` 了解已有结果；
4. 看 `ps`/日志确认队列状态，不要重复启动已在跑的任务；
5. 需要新实验时用 `runner_phg_cell.sh` 或 `queue_*` 脚本，遵循 §7 的坑；
6. 所有新结果写回统一 CSV（`collect_queue_results.py` 或手动追加），并 commit+push。

## 11. 相关文档索引

- `../04-guides/环境说明.md`：环境
- `../04-guides/环境配置与交接.md`：另一台 Linux+NVIDIA 机器的完整配置步骤
- `../06-planning/DCCA-GS_项目变更文档.md`：状态/坑/结果历史
- `../01-architecture/DCCA-GS_架构说明.md`：架构与创新点详解
- `../03-reports/P0_阶段A报告.md`、`../03-reports/R_阶段A报告.md`、`../03-reports/R4_attr上下文_报告.md`、
  `../03-reports/SPA_阶段A报告.md`：各方向报告
- `../03-reports/MiniSplat×SPA_实验报告.md`：MiniSplat + 语义 + SPA 主报告（含后续实验设计）
- `../data/experiments.csv`：统一实验数据
- `HAC-plus-main-v1/陈曈提案_DCCA-GS论文版.docx`（及中文版）：IEEE 提案（论文级内容；
  旧 `陈曈提案_PHG论文版.docx` 保留归档）

## 12. 改名说明（重要）

项目显示名已统一为 **DCCA-GS（Decoder-Reproducible Content-Adaptive Compression
for Anchor-Based 3D Gaussian Splatting）**，原名为 PHG（PKUGS-HAC-Gsplat）。

当前已改名：

- 仓库显示名、README、HANDOVER、环境说明、提案标题；
- 对外文案统一用 DCCA-GS。

**为兼容性暂时保留、后续统一改名的清单**（不要在本版本改动，否则会破坏运行中任务
与旧码流）：

| 保留项 | 原因 | 后续目标名 |
| --- | --- | --- |
| 码流 `format = "phg_v1"` 与 `FORMULA_INPUT_VERSION` 中的 phg 前缀 | 旧码流/运行中任务的版本校验 | `dcca_v1`（需版本迁移） |
| 编解码器名 `hac_pp` | `MODELS`/`CODECS` 注册键、CLI `--cfg.model.model-name hac_pp` | `dcca_gs`（需改配置与 runner） |
| 文件名 `test_phg_*.py`、`runner_phg_cell.sh`、`finish_*_phg*` | 运行中队列/日志引用 | `test_dcca_*`、`runner_dcca_cell.sh` 等 |
| CSV `group` 名（`phg_*`） | 收集器去重与历史数据 | `dcca_*`（需迁移 CSV） |
| 提案/文档文件名中的 PHG | 链接与归档稳定性 | 随版本发布统一迁移 |

改名原则：显示名与对外文案立即生效；内部标识只在“不影响运行中任务、不破坏旧码流
bit-exact 校验”时改动，否则留到一次显式的版本迁移提交。
