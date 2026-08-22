# 创新点 R：CompGS 式“锚点预测 + 残差编码”可行性实验设计

## 0. 文档说明

| 项目 | 内容 |
| --- | --- |
| 版本 | v0.1 |
| 日期 | 2026-08-17 |
| 状态 | 设计稿，未开始实验 |
| 适用范围 | PHG（`scaffold_gs` / `hacplus`）压缩管线 |
| 关联代码 | `scaffold_gs/hacpp.py`、`hacplus/scene/gaussian_model.py`、`hacplus/utils/codec_consistency.py` |
| 关联文档 | [创新点P0设计文档.md](创新点P0设计文档.md)、[P0_阶段A报告](../03-reports/P0_阶段A报告.md) |
| 论文参照 | CompGS（ACM MM 2024）、CompGS++（arXiv 2504.13022） |

本文档回答一个问题：**把“锚点预测 + 残差”的编码结构引入 PHG，能否在真实码率口径下带来可用的压缩收益？**

它只验证可行性，不预设结论。实验分三阶段：A 离线熵验证（本阶段先做）、B 最小编解码实现、C 联合训练验证。按停止条件逐级决定是否继续。

## 1. 问题定义

### 1.1 CompGS 做了什么

CompGS 提出混合基元结构：少量锚点基元（anchor primitive，完整编码、空间稀疏）作为参照，其余耦合基元（coupled primitive）的属性由锚点基元预测，再以“紧凑残差”形式保存；训练时用率约束优化（rate-constrained optimization）同时最小化渲染失真和码率，熵估计同时覆盖锚点基元与耦合基元。

关键点：**CompGS 的耦合基元属性是“预测值 + 残差”结构，残差本身进熵编码。**

### 1.2 PHG 与 CompGS 的结构差异

PHG（Scaffold-GS + HAC++）已经是锚点结构：每个锚点解码 K=10 个神经高斯（基元），这些基元的属性由 `mlp_opacity` / `mlp_cov` / `mlp_color` 从锚点特征现场生成，**不直接存入码流**。真正进熵编码的是锚点属性：

- feat（局部特征，50 维，占体积最大）
- scaling（缩放，6 维）
- offsets（偏移，10×3 维，受 mask 控制）
- masks / hash / 几何 / MLP 权重

因此不能照搬 CompGS 的“给耦合基元存残差”——PHG 的耦合基元本来就不存。**可迁移的等价形式是：对真正进熵编码的锚点字段，用解码端可重算的预测器生成预测值，对“预测残差”做熵编码。**

### 1.3 与 P0 系列实验的关系（重要）

P0 已经测过“用已解码字段做条件”：

| 实验 | 内容 | 结果 |
| --- | --- | --- |
| P0-1 | 用已解码 feat 调整 scaling/offsets 熵参数 | 相对增益约 2.3%（绝对约 0.0088 MB） |
| P0-2 | 前 k 个 Morton 邻居做上下文 | 0.06%–0.26%，关闭 |
| 反向 | scaling/offsets/mask 调整 feat 初始熵参数 | 约 0.4%，关闭 |

这些是“条件熵”实验，即：给同一个高斯熵模型换更好的 mean/scale。

残差编码和条件熵在信息论上是同一件事的下界：**给定同一个预测器，无论“把预测值作为高斯均值”还是“先减预测值再编码残差”，极限码率都是条件熵，两者相等**。残差结构可能赢的地方只有三点：

1. 残差分布用更简单的模型（如拉普拉斯）比当前高斯假设更贴合真实分布，缩小 KL 散度；
2. 预测器可以是非线性、跨字段的（如 feat_q → scaling），比当前 `mlp_grid` 只吃哈希上下文的均值预测更强；
3. 通道间差分（delta coding）能利用同一锚点内相邻通道的相关性。

所以本实验的核心不是“再验证一次条件熵”，而是验证：**在 P0-1 已经接近条件熵上限的前提下，残差编码结构能否通过上述三点之一，带来净收益。**

## 2. 输入清单

### 2.1 事实

| 编号 | 内容 | 验证方式 |
| --- | --- | --- |
| F1 | 当前编码顺序：feat → scaling → offsets；解码端解码 scaling/offsets 前已有 `feat_q` | 代码事实 |
| F2 | `mlp_grid` 已为三个字段预测高斯 mean/scale，编码符号为 `round(x/Q)` | 代码事实 |
| F3 | P0-1（feat 条件化 scaling/offsets）相对增益约 2.3%，绝对约 0.0088 MB | P0 报告 |
| F4 | P0-2（邻居条件）与反向（scaling/offsets → feat）均低于阈值关闭 | P0 报告 |
| F5 | 新增 MLP 权重按 16-bit + 算术编码计入 `total_MB`（当前 MLP 量化口径） | 代码/实验记录 |
| F6 | 解码端可用的输入只有：锚点坐标、哈希特征、已解码符号、模型权重 | 硬约束 |
| F7 | 对同一预测器，条件熵编码与残差编码的极限码率相等 | 信息论定义 |

### 2.2 假设（待实验证伪/证实）

| 编号 | 假设 | 风险 |
| --- | --- | --- |
| A1 | 残差分布用拉普拉斯/高斯零均值模型，比当前高斯(mean, scale)更贴真实分布 | 若当前模型已接近真实熵，收益为零 |
| A2 | 跨字段预测器（feat_q → scaling/offsets）能给出比 `mlp_grid` 均值更强的预测 | P0-1 显示条件增益只有 2% |
| A3 | 通道差分（delta coding）能压缩 feat 通道间冗余 | 未验证，可能被通道自回归已覆盖 |
| A4 | 预测器新增权重（按 16-bit 计入）不会吃掉残差收益 | 需要显式计算净收益 |

### 2.3 被剥离的不可靠假设

1. “CompGS 的收益 = 残差结构本身的收益”——CompGS 同时改了基元结构、率约束训练和熵编码，收益来源不可分离；
2. “残差一定比条件均值编码省”——数学上对同一预测器二者极限相同；
3. “预测器不需要计入体积”——所有新增权重必须计入；
4. “可以用训练期统计量做预测器输入”——解码端拿不到，除非写侧信息（本实验默认零侧信息）。

## 3. 阶段 A：离线残差熵验证（先做）

### 3.1 目标

在训练好的 checkpoint 上，用与真实 codec 一致的符号与量化步长 Q，比较：

```text
H_base：当前 codec 的逐字段码率（feat / scaling / offsets 分开）
H_res ：预测残差 r = x − pred 的熵编码码率（同一 Q 下）
```

只比较熵，不写真实码流；结论用“相对字段增益 + 绝对 total_MB 净收益”两个口径。

### 3.2 数据与模型

- 首选：4-28 90k h32（`runs/4-28_i6_90k_h32_l0p002/ckpts/ckpt_90000.pth`），与 P0 同源，方便直接对比；
- 备选：Deep Blending playroom 110k（`runs/db_playroom_i6_110k_h32_l0p002`，训练中），用于验证跨场景稳定性；
- 验证集：Morton 排序后 20% 锚点（与 P0 一致），前 80% 用于拟合预测器参数。

### 3.3 基线（H_base）

逐字段复用真实 codec 的熵公式：

- feat：混合高斯 + `Channel_CTX_fea` 通道自回归（复用 `p0_offline_reverse.py::_feature_mixed_bits` 的严格口径）；
- scaling / offsets：`mlp_grid` 输出的 mean/scale + 高斯 bin 概率；
- Q 全部走当前 I2 formula 路径（含 `mlp_complexity` 乘子），保证与真实编解码一致。

### 3.4 残差变体（按顺序做，每个都是独立开关）

**R0（零参数对照，必做）**：预测器就用 `mlp_grid` 自己的均值：

```text
r = x_q − mean_pred
符号 s = round(r / Q)
熵模型：拉普拉斯(0, b)，b 在训练集上按字段 MLE 估计（b = mean|r|）
```

这个变体回答：**仅把“高斯均值建模”换成“零均值拉普拉斯残差建模”，KL 散度能不能变小。** 不新增任何参数。

**R1（小预测器）**：新增零初始化残差 MLP（输入为 `calc_context_feat(anchor)`，8→hidden 64→1 层），对每个字段预测 x，再编码残差；拉普拉斯尺度 b 仍按字段估计。回答：**更强的预测器能否显著降低条件熵。**

**R2（跨字段耦合预测，最接近 CompGS）**：

```text
pred_scaling = MLP_s(feat_q, ctx)
pred_offsets = MLP_o(feat_q, scaling_q, ctx)
```

编码顺序保持 feat → scaling → offsets，所以 feat_q / scaling_q 解码端可得。回答：**“已解码属性 → 下一字段残差”是否比 P0-1 的“已解码属性 → 调整高斯均值”更好。**

**R3（通道差分）**：feat 的第 c 个 10 通道组，残差 = 当前组 − 上一组已解码重建（在 `Channel_CTX_fea` 自回归之外再叠一层 delta）。回答：**通道间线性冗余是否还存在。**

**R4（控制组）**：用 R2 的同一个预测器，但不减残差，而是把预测值作为高斯均值调整量（即 P0-1 的做法）。**如果 H(R4) ≈ H(R2)，说明残差结构相对条件均值编码没有额外收益，理论判断成立，R 整体关闭。**

### 3.5 实现方式

新增 `scripts/residual_feasibility.py`（在 5090 `HAC_5090_a100` 环境跑）：

1. 复用 `p0_offline_entropy.py` / `p0_offline_reverse.py` 的数据加载、Morton 排序、验证集切分、I2 Q 计算；
2. 基线 `H_base` 按字段输出；
3. 对每个变体：训练集拟合参数 → 验证集计算 `H_res`；
4. 输出表格：字段、H_base、H_res、相对增益、绝对 ΔMB；
5. 净收益 = ΔH − 新增预测器权重体积（16-bit + 算术编码估算）。

预计单场景运行时间与 P0 脚本同量级（1–2 小时），可在 5090 上与现有训练叠加。

### 3.6 阶段 A 停止条件

满足以下任意一条，整个 R 方向关闭，不进入阶段 B：

1. R2 相对 `H_base` 的字段增益 < 3%，或换算到 `total_MB` 的净收益 < 1%；
2. R2 与 R4 的码率差 < 0.1%（残差结构相对条件均值编码无额外收益）；
3. R0 无增益且 R1/R2 也无增益（问题不在分布假设，而在预测器表达力）；
4. 新增预测器权重吃掉全部残差收益。

## 4. 阶段 B：最小编解码实现（仅当 A 通过）

1. `hacplus/utils/codec_consistency.py` 增加 `residual` 模式：符号定义改为 `round((x − pred)/Q)`，解码端同序重建 `pred`；
2. `encode_attributes` / `decode_attributes` 在 Morton 排序后逐字段走“预测 → 残差 → 算术编码”；
3. 编码顺序仍为 feat → scaling → offsets，R2 的预测器输入用已解码符号；
4. 训练路径 `_estimate_rate_terms` 同步改为估计残差熵（用 STE 量化模拟已解码符号）；
5. 预测器权重导出进 MLP payload（16-bit + 算术编码）；
6. 验收：bit-exact roundtrip（符号、Q、整数 CDF 不匹配数 = 0），PSNR/SSIM/LPIPS 与编码前一致，体积按新口径下降。

## 5. 阶段 C：联合训练验证（仅当 B 通过）

- 在 30k 短训配置（`update_until=15000`）上做 I2+I6+残差 vs I2+I6 对照；
- 再在 110k 全量上验证一个场景；
- 报告 RD 曲线与 BD-rate，用与 P0 相同的成功标准（解码后评估、bit-exact、所有权重计入码率）。

## 6. 成功标准（汇总）

| 阶段 | 通过条件 |
| --- | --- |
| A | R2 字段增益 ≥3% 且 `total_MB` 净收益 ≥1%，且优于 R4 控制组 |
| B | 真实编解码 bit-exact，体积下降与离线估计一致（±10%） |
| C | 110k 场景 BD-rate ≤ −3%，且渲染指标不劣化 |

## 7. 风险与应对

| 风险 | 应对 |
| --- | --- |
| P0-1 已接近条件熵上限，残差结构无增量 | 阶段 A 的 R4 控制组直接给出结论，成本低 |
| 预测器权重吃掉收益 | 净收益口径强制计入 16-bit 权重体积；预测器限定 ≤1 层、hidden ≤64 |
| 拉普拉斯假设也不贴分布 | 增加高斯(0, σ) 变体对比；若两者接近，说明问题不在分布族 |
| 通道差分与现有通道自回归重复 | R3 只作为补充信息，不作为主判据 |
| 跨字段预测器在训练中分布失配 | 阶段 B 必须同步改 `_estimate_rate_terms`，用 STE 模拟解码符号 |

## 8. 参考文献

- CompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting（ACM MM 2024，[arXiv:2404.09458](https://arxiv.org/abs/2404.09458)）
- CompGS++: Compressed Gaussian Splatting for Static and Dynamic Scene Representation（[arXiv:2504.13022](https://arxiv.org/abs/2504.13022)）
- PHG P0 阶段 A 报告（`../03-reports/P0_阶段A报告.md`）
