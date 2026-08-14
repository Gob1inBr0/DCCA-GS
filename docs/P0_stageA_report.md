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

## 补充实验（2026-08-15）：P0-2 原始邻居特征版

验证上一轮 mean/max 池化是否丢失邻居信息：每个前 k 个 Morton 邻居先过共享
`Linear(99→32)+ReLU`，再对 k 个邻居做 mean 池化得到 32 维上下文，拼到熵参数后
进残差 `_AdjMLP`（最后一层零初始化）。Adam、lr 1e-3、wd 1e-4、早停，验证集为
Morton 序最后 20%。

| k | 400 步增益 | 1500 步增益 |
| --- | --- | --- |
| 8 | +0.148% | +0.257% |
| 16 | +0.149% | +0.247% |
| 32 | +0.135% | +0.177% |

结论：原始邻居特征版与 mean/max 池化版同量级（0.06%~0.26%），池化没有丢失
信息；按实验判定规则（<1%）**P0-2 正式关闭**。

产物：`scripts/p0_offline_entropy_rawctx.py`；5090 结果
`runs/p0_offline_rawctx_s400.json`、`p0_offline_rawctx_s1500.json`。

## 补充实验（2026-08-15）：反向渐进式编码（scaling+offsets → feat）

回答“用已解码的 scaling/offsets/mask 作条件能否显著降低 feat 熵”。feat 熵严格
复刻真实 codec（混合高斯 + 通道自回归，`Channel_CTX_fea` 逐 10 通道），V1 只调整
feat 初始熵参数（残差 `mlp_feat_ctx`，最后一层零初始化）。Adam、lr 1e-3、
wd 1e-4、早停，验证集为 Morton 序最后 20%。

基线：`H_feat_base` = 0.6038 MB（验证集 20%）；`H_so_base` = 0.3924 MB。
正向 P0-1 绝对节省 `Δ_forward` ≈ 0.0088 MB。

| 配置 | H_feat_cond (MB) | 相对增益 | Δ_reverse (MB) |
| --- | --- | --- | --- |
| h64 / 400 | 0.6014 | +0.406% | 0.0025 |
| h64 / 1500 | 0.6013 | +0.421% | 0.0025 |
| h128 / 400 | 0.6014 | +0.406% | 0.0025 |
| h128 / 1500 | 0.6013 | +0.421% | 0.0025 |

结论：反向条件增益约 0.4%（<2%），且 `Δ_reverse`（0.0025MB）远小于 `Δ_forward`
（0.0088MB），按决策规则**反向路线关闭**，不做 Stage B。

产物：`scripts/p0_offline_reverse.py`；5090 结果
`runs/p0_offline_reverse.json`。
