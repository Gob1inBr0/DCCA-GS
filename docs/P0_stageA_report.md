# 创新点 P0 阶段 A 报告：离线条件熵验证

日期：2026-08-15；模型：PHG V2 h32 90k（`4-28_i6_90k_h32/ckpts/ckpt_90000.pth`）。

## 方法

- 按 codec 顺序（mask + Morton）取 259,061 个锚点；
- 符号与基线熵参数与真实 codec 一致：`mlp_grid` 的 mean/scale、自适应 Q（含 I2 内容感知乘子）、offsets 只统计 mask 有效位；
- 基线 `H_base` = 现有高斯熵模型下的 scaling+offsets 符号熵；
- P0-1：以 STE 量化后的已解码 feat 为条件，拟合残差 MLP（`mlp_attr_ctx` 规格）输出 mean/log-scale 调整量；
- P0-2：以“前 k 个 Morton 邻居的 feat/scaling/offsets/mask/坐标差”的 mean/max 池化为条件，拟合同规格残差 MLP；
- 拟合细节：最后一层零初始化（恒等起点）、Adam + weight decay、按验证集早停、验证集为 Morton 序最后 20%。

## 结果（验证集 20%，总熵 0.3826 MB）

| 方案 | 熵 (MB) | 增益 |
| --- | --- | --- |
| H_base（现有基线） | 0.3826 | — |
| P0-1：以 feat 为条件（hidden 64 / 400 步） | 0.3755 | +1.86% |
| P0-1：以 feat 为条件（hidden 128 / 1500 步） | 0.3738 | +2.30% |
| P0-2：前 8 个 Morton 邻居 | 0.3821 | +0.13% |
| P0-2：前 16 个 Morton 邻居 | 0.3822 | +0.09% |
| P0-2：前 32 个 Morton 邻居 | 0.3824 | +0.06% |

## 结论

- P0-1 离线增益约 2%（1.86%~2.30%），低于设计文档 3% 停止阈值；
- P0-2 离线增益约 0.1%，基本无信息；
- **阶段 A 验收不通过（FAIL <3%），按设计文档停止 P0，不再进入阶段 B/C。**

## 产物

- 脚本：`scripts/p0_offline_entropy.py`
- 5090 结果：`~/data_space/web_scan/runs/p0_offline.json`（hidden 64 / 400 步）、
  `p0_offline_big.json`（hidden 128 / 1500 步）
