# DCCA-GS 项目变更与状态文档（原 PHG）

更新日期：2026-08-16

本文档面向后来接手的人：看完这一份，就能知道 PHG 当前做到哪、哪些创新点有效/已关闭、
哪些坑踩过、结果是什么、接下来怎么跑。

## 1. 项目是什么

DCCA-GS（原 PHG，PKUGS-HAC-Gsplat）是基于 gsplat 的 Scaffold-GS / HAC++ 神经高斯压缩框架。
目标是：以 HAC++ 压缩管线为基线，叠加创新点（I1/I2/I6 等），同时保持
“模型可替换 + 稳定属性导出 + 编解码器接口化”的低耦合架构，方便后续灵活改动。

仓库位置：

- GitHub：`goblinIBigBro/PHG`
- 本地：`/Users/chen/Documents/PHG`
- 5090：`/home/fansonglin/xieliang/chentong/PHG`
- 旧项目（HAC-plus-main-v1）：`/Users/chen/Documents/HAC-plus-main-v1`（论文/旧实验数据都在这里）

## 2. 当前代码状态

| 分支 | HEAD | 状态 |
| --- | --- | --- |
| `main` | `61e60d5` | V2 主线：I2 默认开、I6 可选、I1 关；feat_dim 泛化、训练加速、速度型 runner 均已合入 |
| `i6-sens-replace` | 本地 `5351e8a`；5090 `1c85b6c` | I6 替换/侧信息实验 + deform hidden 自适应修复；实验结论为“关闭”，修复代码尚未并入 main |
| `i5-vq` | `9550d68` | I5（VQ）存档分支，已关闭 |

注意：本地和 5090 的 `i6-sens-replace` 历史不完全一致（本地多两个 `load_checkpoint`
修复提交，5090 通过 scp 同步了脚本文件）。**deform hidden 自适应修复未并入 main**，
用 main 加载低维（dim16/32）checkpoint 仍会失败。

## 3. 模块（创新点）状态总表

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| HAC++ 核心 codec（`hac_pp`） | ✅ 必需 | GPCC 几何 + 哈希上下文 + 条件熵模型 + 算术编码；bit-exact roundtrip |
| I2 内容感知公式量化 | ✅ 默认开 | `Q = Q0 * (1 + tanh(z) * α)`，complexity MLP 预测 Q；start 20000、ramp 10000 |
| I6 渲染敏感度监督（监督式） | ✅ 可选开 | 训练期用渲染梯度 EMA 监督 complexity MLP；不改变码流、零旁路；约 +0.1 dB |
| I1 层级上下文 | ⚠️ 可选、默认关 | `concat(base, parent, level)`；有 start_iter 坑，见 §4 |
| I6 替换方案 / side-info | ❌ 已关闭 | 解码端可重算输入与敏感度相关≈0；真值侧信息重排反而更大 |
| I5（VQ） | ❌ 已关闭 | 存档在 `i5-vq` |
| P0 渐进式编码 | ❌ 已关闭 | 阶段 A 离线增益不足 3%，按设计文档停止 |
| feat_dim 泛化 | ✅ 已合入 | `Channel_CTX_fea(feat_dim, channel_group)`；16/32/50 均 bit-exact |

**当前有效组合（论文对照用）**：`HAC++ core + I2 + I6`，feat_dim=50、hidden=32、
90k、λ=0.004，即 4-28 的 h32 90k 配置。

## 4. 踩过的坑（必读）

1. **growth 统计显存泄漏**：统计张量未 detach，约 10MB/步，是 4-28 长训 OOM 的根因；已修复。
2. **gsplat packed 光栅化无上限分配**：大场景交点估算可达 2.5~4B，单步瞬态 ~30GB。
   用 `tile_size=32`（5090 sm_120 上限；64 会超共享内存）。
3. **4-28 图像预载**：1200 张 float32 全上 GPU ≈ 28GB，会爆 32GB 显存；必须
   `--no-preload-images`（CPU uint8 缓存约 5GB，每步只传当前 batch）。
4. **评估口径**：必须 `data-factor 1 + max-width 1600`（官方 resolution=-1 规则）。
   全分辨率评估会低 ~0.36dB（如 h32 90k：28.655 vs 28.298），不能混用。
5. **I1 start_iter**：默认 12000 会在训练分支 step>10000 时维度不匹配；
   必须配 `< 10000`（短实验用 300）。
6. **I6 曾全零失效**：renderer 必须传 `retain_grad`；方差归一化会被压平，
   改为相对归一化 `(ema - mean) / mean` + clamp `[0.1, 2.0]`。
7. **deform hidden 加载**：dim16/32 训练时是旧 `2×g` 隐藏宽度、压缩时按 `4×g` 加载会失败；
   `load_checkpoint` 已按 checkpoint 形状自适应重建（在 `i6-sens-replace`，未并入 main）。
8. **bitstream 体积口径**：官方口径 = 属性流 + 几何 + hash + masks + header +
   **MLP 权重（32 bit/参数）** + xyz 边界（192 bit）。旧数字没有 `bit_mlp` 是旧口径，不可直接比。
9. **I6 替换/侧信息结论**：预测器输入与敏感度 EMA 相关性最佳仅 0.0086（杀阈 0.3）；
   真值敏感度重排在训练好的 90k 上反而变大（5.5604 → 5.7143MB）。不要重复投入。
10. **P0 停止条件**：P0-1 离线增益 1.86~2.30%、P0-2 约 0.1%，均 < 3% 阈值，已关闭。
11. **5090 使用纪律**：不杀其他用户进程；长训练用带“空卡检测+自动重试”的 runner。
12. **HAC++ 环境**：`conda activate HAC_5090_a100`；`PYTHONNOUSERSITE=1`；
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；PATH 要含 tmc3/GPCC。
13. **不要修改正在运行的 runner 脚本**：bash 是按文件偏移增量读取的，运行中覆盖脚本
    会导致训练结束后的压缩/评估阶段报 `unexpected EOF`（2026-08-16 120k 运行踩过）；
    需要改动时先等该轮跑完，或用独立的新脚本。

## 5. 结果汇总（4-28，1600 宽，官方体积口径）

### 5.1 主对照

| 方案 | PSNR | SSIM | LPIPS | total_MB |
| --- | --- | --- | --- | --- |
| PHG h32 90k（I2+I6） | 28.655 | 0.8921 | 0.2767 | 5.5604 |
| PHG h25 90k（I2+I6） | 28.637 | 0.8922 | 0.2767 | 5.5240 |
| PHG h32 30k（I2+I6） | 27.879 | 0.8867 | 0.2758 | 6.0130 |
| PHG dim16 90k | 28.273 | 0.8877 | 0.2815 | 4.0279 |
| PHG dim32 90k | 28.455 | 0.8901 | 0.2796 | 4.5630 |
| 旧 ours 90k（CT_HAC） | 28.563 | 0.8882 | 0.2982 | 6.3547 |
| HAC++ 论文 | 28.311 | 0.8900 | 0.2932 | 6.9462 |

### 5.2 RD 曲线（h32 90k，joint q_scale，已完成）

| q_scale | total_MB | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- |
| 0.75 | 6.0881 | 28.707 | 0.8928 | 0.2761 |
| 0.875 | 5.7970 | 28.685 | 0.8925 | 0.2764 |
| 1.0（基线） | 5.5604 | 28.655 | 0.8921 | 0.2767 |
| 1.125 | 5.3637 | 28.626 | 0.8917 | 0.2771 |
| 1.25 | 5.1977 | 28.587 | 0.8913 | 0.2775 |
| 1.5 | 4.9329 | 28.502 | 0.8902 | 0.2785 |
| 2.0 | 4.5751 | 28.286 | 0.8874 | 0.2810 |

### 5.3 step 扫描（老 h32 90k 训练，已完成 30k/60k/90k）

| 训练步数 | total_MB | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- |
| 30000 | 6.9730 | 28.193 | 0.8912 | 0.2744 |
| 60000 | 6.1723 | 28.570 | 0.8935 | 0.2742 |
| 90000 | 5.5604 | 28.655 | 0.8921 | 0.2767 |

趋势：训练越久体积越小、PSNR 越高但边际递减；90k 是否到顶待 120k 数据。

### 5.4 Web_Scan feat_dim 粗扫（30k，I2+I6）

详见 `docs/feat_dim_sweep.md`。要点：16 维是甜点（BD-rate 相对 50 维 −24.5%），
32 维质量最高（BD-rate −34%）；4-28 上 90k 全维度表见上（dim16/32/50）。

## 6. 常用命令（5090）

```bash
conda activate HAC_5090_a100
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH
cd /home/fansonglin/xieliang/chentong/PHG

# 训练（runner：<gpu> <tag> <dim> <max_steps> <update_until> <save> [eval] [hidden] [lambda]）
bash scripts/runner_4_28_90k.sh 1 i6_90k_h32_l0p002 50 90000 45000 \
  "30000 60000 90000" "90000" 32 0.002

# 压缩 + 解码评估
python train.py compress --cfg.ckpt <ckpt.pth> --cfg.out-dir <out> --cfg.codec hac_pp
python scripts/eval_decoded.py --artifact-dir <out> --data-dir <4-28> \
  --result-dir <eval> --data-factor 1 --max-width 1600 --no-preload-images

# 后处理 RD 扫描（q_scale）
python scripts/rd_sweep.py --ckpt <ckpt> --data-dir <4-28> \
  --result-dir runs/rd_xxx --data-factor 1 --max-width 1600 --no-preload-images \
  --q-scale-joint 0.75 0.875 1.0 1.125 1.25 1.5 2.0

# step 扫描（压缩+评估已存在检查点）
bash scripts/step_sweep_4_28.sh <gpu> <run_dir> "30000 60000 90000"

# 绘图
python scripts/plot_rd_step_4_28.py
```

## 7. 当前任务与下一步

进行中（2026-08-16）：

- GPU0：`4-28_i6_90k_h32_120k`，120k 训练（60k~120k 每 10k 存 ckpt，eval 只在 120k）
- GPU1：`4-28_i6_90k_h32_l0p002`，λ=0.002 的 90k 训练（第二个 λ 点）
- 已完成：q_scale RD 7 点；30k/60k/90k step 评估

下一步：

1. λ=0.002 训练完 → compress + eval → 与 λ=0.004 组成 2 点 λ-RD 曲线
2. 120k 训练完 → 评估 70k/80k/100k/110k/120k → 确定 90k 附近最优轮数
3. 把 PHG q_scale 曲线、PHG λ 曲线、旧 HAC λ 曲线（HAC 文件夹）和 HAC++ 官方点画在同一张图
4. 把 deform hidden 修复并入 main，同步三处代码

## 8. 旧项目数据位置（HAC 文件夹）

- `result/rd_curve_260708_three_scene/rd_main_curves_3scene_260708.csv`：
  4-28/1-100/3-07 的 `ct_formula_i1_hybrid_90k`、`ct_shared_all_i1_hybrid_90k`、
  `official_hacpp_60k` 在 λ∈{0.001,0.002,0.004,0.006} 的完整 RD 点
- `paper/data/main_table_operating_points.csv`：HAC++ / PC-GS / CAT / ContextGS / HAC
  等算法在三个场景的操作点
- `docs/Codex_当前状态快照.md`、`docs/创新点解释文档.md`：旧项目状态快照与创新点概念
