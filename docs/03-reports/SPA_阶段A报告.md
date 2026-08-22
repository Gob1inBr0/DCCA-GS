# 创新点 S（GaussianSpa 式训练侧稀疏化）阶段 A 报告

日期：2026-08-17；场景：Deep Blending **playroom**；配置：30k、I2+I6、dim50、
h32、λ=0.002、voxel 0.001、1600 宽、`update_until=15000`；SPA ratio=0.5、
ρ=1e-3、u clamp ±1、预算按 `max(N_ref)×ratio_t` 线性斜坡（1500→15000）。

## 1. 结果（compress → decode → eval，29 个验证视图）

| 方案 | 训练 anchors | 编码 anchors | total_MB | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- |
| 基线（无 SPA，编码端不裁剪） | 338,560 | 168,626 | 4.1831 | 30.872 | 0.9134 | 0.2594 |
| MaskTopk-only 0.5（基线 ckpt + 编码端 topk） | 338,560 | 169,280 | 4.1958 | 30.872 | 0.9134 | 0.2594 |
| MaskTopk-only 0.1026（对齐 SPA 编码锚点数） | 338,560 | 34,736 | 1.4740 | 24.120 | 0.8399 | 0.3714 |
| **SPA-anchor 0.5（训练侧 ADMM）** | 50,741 | 34,722 | **1.0835** | 29.639 | 0.8948 | 0.3106 |

## 2. 关键观察

1. **基线编码端 mask 已剪掉一半**：训练 338,560 anchors，`mask_anchor` 只编
   168,626（约 49.8%）。所以 MaskTopk-0.5 与基线几乎无差别（4.1958 vs 4.1831 MB，
   PSNR 完全相同）——编码端 50% topk 没有额外作用。
2. **SPA 大幅压缩体积**：1.08 MB vs 基线 4.18 MB（-74%），代价是 PSNR -1.23 dB
   （30.87→29.64）、SSIM -0.019、LPIPS +0.051。
3. **同锚点数对照（topk≈0.103）是决定性的**：同样约 34.7k 编码锚点，SPA 1.08 MB /
   29.64 PSNR，而编码端 topk 1.47 MB / 24.12 PSNR——**SPA 高 5.52 dB 且体积还小
   27%**。说明“剪掉就忘”的编码端 topk 无法替代训练侧 ADMM：只有让幸存锚点在预算
   下继续训练，稀疏后的表示才有效。

## 3. 实现要点

- `spa_z/spa_u` 为纯训练态张量，随生长/剪枝同步（复用 sensitivity 同步模式），
  不写入码流；
- ADMM 每 100 步在 `adjust_anchor` 内执行：`z=TopK(a+u,κ)`、`u=clamp(u+a-z,±1)`、
  `κ=round(max(N_ref)×ratio_t)`；
- 预算斜坡避免“每轮减半”塌缩（第一版 `κ=ratio×N_t` 会把锚点数几何衰减到 80）；
- 增广损失 `ρ/2·‖a−z+u‖²` 接入 trainer；
- `--cfg.model.spa-enabled/--spa-ratio/--spa-rho/--spa-u-clamp`；
- 控制组 `--cfg.mask-keep-ratio`（编码端 topk）已透传到 compress。

## 4. 判定（按设计文档 §3.5）

- SPA 相对基线单点：体积 -74%、PSNR -1.23 dB，是一个有效的低码率操作点；
- SPA vs 同锚点数 MaskTopk：**+5.52 dB 且体积更小，远超设计文档“≥1% 增量”阈值**，
  ADMM 乘子反馈 + 预算训练被证明必要；
- 阶段 A 判定：**通过**。下一步（阶段 B）：110k 全量 + 多 ratio（0.5/0.3/0.2）+
  多 λ 画 RD/BD-rate 曲线，并在 drjohnson 上复现方向。

## 5. 体积构成（SPA vs 基线，bits）

| 字段 | 基线 | SPA | 下降 |
| --- | --- | --- | --- |
| 几何 bit_anchor | 2,752,744 | 676,640 | -75% |
| feat | 14,916,904 | 2,645,176 | -82% |
| scaling | 6,388,568 | 1,153,968 | -82% |
| offsets | 6,500,528 | 1,268,640 | -80% |
| masks | 1,307,672 | 292,840 | -78% |
| hash | 430,664 | 258,728 | -40% |
| MLP（32bit 口径） | 2,784,704 | 2,784,704 | 0% |
| total_MB | 4.1831 | 1.0835 | -74% |

MLP 权重成为 SPA 低码率点的最大占比项（2.78MB 中的 2.57MB），后续可叠加 MLP
量化（16/8-bit）进一步压缩。

## 6. 产物

- 训练：`runs/spa_db_playroom_30k_base/`、`runs/spa_db_playroom_30k_spa/`
- 压缩/评估：`runs/spa_db_playroom_30k_{base,topk05,topk010,spa}_bit/`
- 代码：`hacplus/scene/gaussian_model.py`（SPA 状态 + ADMM 剪枝）、
  `scaffold_gs/hacpp.py`（spa_loss_term、状态存取）、`scaffold_gs/config.py`、
  `scaffold_gs/trainer.py`、`train.py`
