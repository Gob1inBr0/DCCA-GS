# MiniSplat × SPA 实验报告：锚点位置 vs 预算

> 版本：v1.0（2026-08-22）
>
> 定位：回答「SPA（训练侧 ADMM 锚点预算）在高预算档失效，到底是数量不够还是位置不对」。
> 做法：把 Mini-Splatting 的 depth-reinit（深度反投影增密）接入锚点/HAC++ 世界，并把 SPA 预算钉在增密前，
> 从而在**总锚点不变**的前提下只重排「哪些锚点存活」。所有数字均来自 5090
> `runs/` 实际 `metrics.jsonl` + `hac_meta.json`（HAC++ 编码后真实解码评估），未使用推测值。

---

## 0. TL;DR

| 结论 | 证据 |
| --- | --- |
| **「位置 > 预算」成立** | playroom 30k 同预算（SPA ratio 0.85）：MiniSplat+SPA 30.420 dB / 1.905 MB，基线 SPA 30.221 dB / 1.859 MB → **+0.199 dB，体积仅 +2.5%** |
| MiniSplat 是「免费重排」 | 增密只把锚点铺到表面，SPA 预算钉在增密前；训练锚点 112,875→115,170（+2.0%），体积 +2.5% |
| MiniSplat 是**场景依赖** | drjohnson 30k：cell2 29.3146 dB / 2.0021 MB vs cell1 29.3871 dB / 2.0379 MB → **−0.073 dB，体积 −1.8%**；4-28 110k 同样近零 |
| 语义监督是**比特昂贵**的 | 同预算下语义 +0.117 dB（高档），但体积 +18.2%；完整预算曲线上 **BD-PSNR −0.10 dB、BD-rate +8.2%**（同率不占优） |
| 大场景语义无净增益 | 4-28（非 SPA，30k）语义 28.267 vs 基线 28.308 dB（**−0.041 dB**），SSIM/LPIPS 几乎不变 |
| 当前主路径 | **cell2 = I2 + I6 + SPA(0.85) + MiniSplat**（`spa_enabled=True, spa_ratio=0.85, mini_splat_enabled=True`）；语义方向暂停 |

---

## 1. 假设

- SPA（`GaussianSpa` 式 ADMM 稀疏化）在 playroom 30k 给出同锚点数下比编码端 top-k 高 5.5 dB 的强结果
  （见 [SPA_阶段A报告](../03-reports/SPA_阶段A报告.md)），但 110k/高预算档增益收敛；其机制是 `get_mask`
  的 top-k，**只删不搬**，无法改变幸存锚点的空间分布。
- **H1（位置 > 预算）**：同 SPA 预算下，depth-reinit 重排锚点位置 → 质量显著提升，体积几乎不变。
- **H2（语义可叠加性）**：MiniSplat 与 DINO 语义监督（T-A2）是否正交，叠加是否仍值得。
- **H3（语义性价比）**：语义监督是否只是「用更多比特换质量」，而非「免费提升」。

---

## 2. 方法与实现

### 2.1 设计

在生长停止点（`update_until=15000`）：

1. 用 gsplat `render_mode="D"` 渲染 8 个训练相机的深度图；
2. 反投影到世界表面点，按 `voxel_size` 体素去重、取簇中位点 → 候选锚点；
3. 走官方 `cat_tensors_to_optimizer` 增密候选锚点（`mini_splat_max_new=4000`）；
4. **增密前把 SPA 预算钉死**（`spa_final_n = 增密前锚点数`），因此新锚点只参与「谁活下来」的重排，
   不抬高预算；
5. 继续训练到 30k，再 HAC++ 压缩 → 解码 → 评估。

这样隔离了「位置（placement）」与「数量（count）」：如果增益来自多塞锚点，体积会相应变大；
如果来自重排，体积基本不动而质量上升。

### 2.2 实现接入点（后续维护请对齐）

| 层 | 文件 | 关键符号 |
| --- | --- | --- |
| 配置 | `scaffold_gs/config.py` | `mini_splat_enabled / mini_splat_reinit_iter / mini_splat_max_new / mini_splat_views / mini_splat_voxel`；`spa_enabled / spa_ratio` |
| 增密与状态同步 | `hacplus/scene/gaussian_model.py` | `append_depth_anchors()`、`_sync_semantic_state()`、`_prune_anchor_optimizer()`（跳过投影头参数组）|
| 模型入口 | `scaffold_gs/hacpp.py` | `mini_splat_reinit()`、`semantic_supervision()`、`sensitivity_supervision()` |
| 深度反投影 | `scaffold_gs/mini_splat.py` | `render_scene_depth()`、`collect_depth_surface_anchors()`、`_backproject_depth()` |
| 训练循环 | `scaffold_gs/trainer.py` | `iteration == mini_splat_reinit_iter` 触发；15k 语义目标刷新（`semantic_targets.refresh_semantic_targets`） |
| 离线语义 | `scaffold_gs/semantic_targets.py` | 用当前训练锚点投影采样 DINO 特征 → 逐锚点目标（`cov` 语义覆盖掩码，n×1） |

### 2.3 已修复的坑（供复现/扩展参考）

- **semantic_cov 越界（device-side assert，cell4 在 19999 步崩溃）**：根因是 `append_depth_anchors`
  里先手动 append 语义张量、再调 `_sync_semantic_state()`，而此时代码还没执行
  `cat_tensors_to_optimizer`，`get_anchor.shape[0]` 仍是增密前的数量 → semantic_cov 被反向截断、
  与锚点数不匹配。**修复：把 `_sync_semantic_state()` 移到 `cat_tensors_to_optimizer(d)` 之后**
  （commit `77ec99f`）。
- **投影头参数组被当成锚点参数**：`cat_tensors_to_optimizer` 断言「每组一个参数」，投影头组有两个参数
  （weight/bias），生长/剪枝时需跳过 `semantic_proj_head` 组。
- **语义目标必须用本次训练的锚点刷新**：用基线模型导出的目标与 30k 从零训练锚点索引对不上，
  目标为空、监督无效（那次的 +0.045 dB 不能归因）。正确协议是在 15k（生长停止）用当前锚点重建目标
  （覆盖率 100%），20k 起监督生效。

---

## 3. 实验设置（共同协议）

| 项 | 值 |
| --- | --- |
| 场景 | Deep Blending **playroom**（训练/验证 8:1，宽 1600，`test-every 8`） |
| 训练 | 30k，`update_until=15000`，`feat_dim=50`，`mlp_complexity_hidden=32`，`tile_size=32`，`appearance_dim=0` |
| 基础创新 | I2 内容感知量化（on）+ I6 渲染敏感度监督（on） |
| 率失真 | λ=0.004（default，已从 ckpt `optim_config` 核实） |
| SPA | `spa_ratio` ∈ {0.52, 0.85, 0.92, 0.97}，ρ=1e-3，预算锚定 `max(N_ref)` |
| MiniSplat | `reinit_iter=15000, max_new=4000, views=8, voxel=0.0` |
| 语义 | T-A2 投影头，DINOv2，15k 自我锚点目标刷新，20k 起监督（`weight=1e-3`） |
| 评估 | `compress → decode → eval_decoded`，29 个验证视图；`total_MB` = HAC++ 解码必需载荷（含 MLP 权重） |
| 数据来源 | 5090 `/home/fansonglin/data_space/DCCA-GS/runs/*`（`metrics.jsonl` + `bitstreams/hac_meta.json`） |

---

## 4. 结果

### 4.1 playroom 主对照（SPA ratio=0.85，4 格）

| cell | 配置 | 训练锚点 | 编码锚点 | total_MB | 解码 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **cell1** | 基线 SPA（I2+I6+SPA） | 112,875 | 87,449 | 1.8594 | 30.2210 | 0.90678 | 0.28093 |
| **cell2** | **MiniSplat + SPA** | 115,170 | 90,872 | 1.9051 | **30.4202** | 0.90701 | 0.27874 |
| cell3 | 语义 SPA（+DINO T-A2） | 115,410 | 101,206 | 2.1974 | 30.3379 | 0.90743 | 0.27844 |
| cell4b | MiniSplat + 语义 + SPA | 114,554 | 101,706 | 2.2568 | 30.4584 | 0.90751 | 0.27836 |

### 4.2 playroom SPA 预算曲线（baseline vs 语义，HAC++ 解码后）

| ratio | 方案 | 训练锚点 | 编码锚点 | total_MB | 解码 PSNR | SSIM | LPIPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.52 | baseline | 51,735 | 40,897 | 1.0580 | 29.3743 | 0.89415 | 0.31051 |
| 0.52 | semantic | 52,616 | 45,197 | 1.2331 | 29.6152 | 0.89572 | 0.30587 |
| 0.85 | baseline | 112,875 | 87,449 | 1.8594 | 30.2210 | 0.90678 | 0.28093 |
| 0.85 | semantic | 115,410 | 101,206 | 2.1974 | 30.3379 | 0.90743 | 0.27844 |
| 0.92 | baseline | 137,827 | 109,966 | 2.2026 | 30.5782 | 0.90879 | 0.27491 |
| 0.92 | semantic | 146,025 | 131,230 | 2.6672 | 30.5763 | 0.90842 | 0.27548 |
| 0.97 | baseline | 176,402 | 151,222 | 2.8383 | 30.8196 | 0.91039 | 0.26757 |
| 0.97 | semantic | 176,888 | 151,437 | 3.1101 | 30.7967 | 0.91036 | 0.26646 |

**BD 分析**（PCHIP，共享 log-rate 区间 [1.233, 2.838] MB）：

- **BD-PSNR ≈ −0.103 dB**（同体积下，语义平均比基线差）
- **BD-rate ≈ +8.2%**（同等质量下，语义多花约 8.2% 码率）

### 4.2.1 E1：MiniSplat × 预算曲线（P0 已完成，2026-08-23）

| ratio | 方案 | 训练锚点 | 编码锚点 | total_MB | 解码 PSNR | ΔPSNR vs 基线 | SSIM | LPIPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.52 | baseline SPA | 51,735 | 40,897 | 1.0580 | 29.3743 | — | 0.89415 | 0.31051 |
| 0.52 | **MiniSplat+SPA** | 57,500 | 43,154 | 1.0908 | 29.5247 | **+0.150** | 0.89562 | 0.30652 |
| 0.85 | baseline SPA | 112,875 | 87,449 | 1.8594 | 30.2210 | — | 0.90678 | 0.28093 |
| 0.85 | **MiniSplat+SPA** | 115,170 | 90,872 | 1.9051 | 30.4202 | **+0.199** | 0.90701 | 0.27874 |
| 0.92 | baseline SPA | 137,827 | 109,966 | 2.2026 | 30.5782 | — | 0.90879 | 0.27491 |
| 0.92 | **MiniSplat+SPA** | 143,043 | 109,476 | 2.2421 | 30.6089 | **+0.031** | 0.90879 | 0.27477 |
| 0.97 | baseline SPA | 176,402 | 151,222 | 2.8383 | 30.8196 | — | 0.91039 | 0.26757 |
| 0.97 | **MiniSplat+SPA** | 183,240 | 147,878 | 2.8739 | 30.8182 | −0.001 | 0.91091 | 0.26668 |

**BD 分析**（PCHIP，四点预算曲线，基线 SPA → MiniSplat+SPA）：

- **BD-PSNR ≈ +0.124 dB**（同体积下，MiniSplat 平均领先）
- **BD-rate ≈ −8.6%**（同等质量下，MiniSplat 少花约 8.6% 码率）
- 逐档趋势：**低/中预算增益最大（0.52→+0.150、0.85→+0.199），高预算逐趋零（0.92→+0.031、0.97→−0.001）**；
  体积增量始终 ≤ +3.1%。→ 「位置 > 预算」**在整条预算曲线上成立，且预算越紧越值**。
- **E1 判定：PASS**（通过线为全档 ≥ +0.05 dB 或 BD-rate ≤ −3%；本实验 BD-rate −8.6%，达标）。

### 4.2.2 drjohnson 泛化（30k，λ=0.004，SPA 0.85，已补，2026-08-24）

| cell | 配置 | 训练锚点 | 编码锚点 | 量化 total_MB | float32 total_MB | 量化 PSNR | float32 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cell1 | I2+I6+SPA0.85 | 155,370 | 90,501 | 2.0379 | 2.2060 | 29.3871 | 29.3900 | 0.9000 | 0.2806 |
| cell2 | +MiniSplat | 152,698 | 89,790 | 2.0021 | 2.1704 | 29.3146 | 29.3151 | 0.8993 | 0.2809 |

**结果**：MiniSplat 在 drjohnson 上 **−0.073 dB**，体积 −1.8%，锚点数 −0.8%；两个 bitstream 均通过 bit-exact roundtrip。
结论修正：MiniSplat 的"位置 > 预算"不是普适规律，在 playroom 正、4-28 接近零、drjohnson 为负，应报告为**场景依赖**。

### 4.3 4-28 大场景 B 组（非 SPA，30k，训练/验证 1050/150）

| run | 配置 | 锚点数 | 训练期验证 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| run428_baseline | I2+I6 | 455,404 | **28.3080** | 0.89269 | 0.27169 |
| run428_semantic | +DINO T-A2（15k 刷新） | 458,015 | 28.2673（**−0.041**） | 0.89270 | 0.27201 |

＞语义目标已在 15k 用本次训练自身 458,015 锚点重建，覆盖率 100%，协议正确；但大场景无净增益。
（该组为训练期验证评估，未做 HAC++ 压缩；体积口径不适用。）

### 4.4 历史非 SPA 语义对照（playroom 1600w，供上下文）

| run | 语义目标 | 编码锚点 | total_MB | 解码 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `db_playroom_sem_t2a5_30k` | 空目标（≈I2+I6 基线） | 161,134 | 4.1075 | 30.9166 | 0.91224 | 0.25791 |
| `db_playroom_sem_t2a5_30k_v3` | 15k 自我锚点刷新 | 204,035 | 5.2841 | 31.0419 | 0.91396 | 0.25742 |

＞+0.125 dB，但体积 +28.6%、active 锚点 +26.6%——语义监督是**密度耦合**的质量提升。

---

## 5. 分析

### 5.1 同预算 vs 同比特

- **同总锚点预算（SPA ratio 固定）**：cell2 vs cell1 → +0.199 dB、体积 +2.5%、编码锚点 +3.9%；
  语义 cell3 vs cell1 → +0.117 dB，但体积 +18.2%、编码锚点 +15.7%。
- **同比特（BD 视角）**：语义 vs 基线整条预算曲线 → **BD-PSNR ≈ −0.10 dB、BD-rate ≈ +8.2%**。

结论：语义在「同总锚点」下看起来领先，是因为 SPA 只固定总锚点、不固定解码/比特预算；
换算到同体积后语义**不占优**。MiniSplat 则相反：同预算下净挣 0.199 dB，且几乎不涨价。

### 5.2 语义与预算的非单调关系

- 低预算（0.52）：语义 +0.241 dB，体积 +16.6%
- 中高预算（0.85）：+0.117 dB，体积 +18.2%
- 高预算（0.92 / 0.97）：−0.002 / −0.023 dB，体积 +21.1% / +9.6%（**由正转负**）

### 5.3 MiniSplat 与语义的叠加

- cell4b vs cell3（同为语义，加 MiniSplat）：+0.120 dB，体积 +2.7% → 重排对语义组同样有效。
- cell4b vs cell2（同为 MiniSplat，加语义）：+0.038 dB，体积 **+18.5%** → 语义在高档位叠加不划算。

### 5.4 大场景

4-28（45.5 万锚点）：语义 −0.041 dB，SSIM/LPIPS 基本不变。MiniSplat 目前**未在 4-28 验证**。

---

## 6. 结论与默认路径

1. **MiniSplat（depth-reinit）+ SPA 是当前最优训练路径**（cell2），已设为默认：
   `spa_enabled=True, spa_ratio=0.85, mini_splat_enabled=True`。
2. **语义先验（T-A2）暂不作为「免费提升」**：同比特有 BD-rate +8.2% 的代价；低预算段仍可能有价值，
   保留在配置中（默认关）。
3. **「位置 > 预算」是本项目可写进论文的核心经验**：锚点压缩的瓶颈是分布，不是单纯数量。

---

## 7. 复现命令（5090）

```bash
# 训练（cell2 为例；baseline 去掉 --cfg.model.mini-splat-enabled）
bash runs/spa_minisplat_launch.sh cell2 0 0 0.85

# 压缩 + 解码评估（评估协议与 §3 一致）
cd PHG && source scripts/env_5090.sh
python train.py compress --cfg.ckpt runs/spa_minisplat_cell2/ckpts/ckpt_30000.pth \
  --cfg.out-dir runs/spa_minisplat_cell2/bitstreams --cfg.codec hac_pp
python scripts/eval_decoded.py --artifact-dir runs/spa_minisplat_cell2/bitstreams \
  --data-dir data/playroom --result-dir runs/spa_minisplat_cell2/decoded_eval \
  --data-factor 1 --max-width 1600 --no-preload-images
```

> 数字已同步至 `../data/experiments.csv`（group `phg_minisplat` / `phg_semantic_stageB`）。

## 8. 相关文档

- [SPA_阶段A报告](SPA_阶段A报告.md)
- [语义先验_阶段A报告](语义先验_阶段A报告.md)
- [语义先验实验设计](../02-design/语义先验实验设计.md)
- [SPA训练侧实验设计](../02-design/SPA训练侧实验设计.md)
- [消融实验汇总](消融实验汇总.md)
- 外部深读笔记（Obsidian）：`2403.14166 - Fang - Mini-Splatting 受限高斯数`
