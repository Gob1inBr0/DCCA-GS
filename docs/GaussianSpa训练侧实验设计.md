# 创新点 S：GaussianSpa 式训练侧稀疏化实验设计

## 0. 文档说明

| 项目 | 内容 |
| --- | --- |
| 版本 | v0.1 |
| 日期 | 2026-08-17 |
| 状态 | 设计稿，未开始实验 |
| 适用范围 | PHG（`scaffold_gs` / `hacplus`）训练与压缩管线 |
| 关联代码 | `scaffold_gs/trainer.py`、`scaffold_gs/hacpp.py`、`hacplus/scene/gaussian_model.py` |
| 关联文档 | [CompGS残差编码实验设计.md](CompGS残差编码实验设计.md)、[创新点P0设计文档.md](创新点P0设计文档.md) |
| 论文参照 | GaussianSpa（CVPR 2025，[arXiv:2411.06019](https://arxiv.org/abs/2411.06019)） |

本文档回答一个问题：**把 GaussianSpa 的训练侧稀疏化（ADMM + 硬稀疏预算）引入 PHG，能否在真实码率口径下带来可用的压缩收益？**

GaussianSpa 是训练侧（compaction）方法，不涉及熵编码；它与 PHG 已有的 mask 机制高度重合，所以本文档的重点不是“从零实现稀疏化”，而是验证一个更窄的问题：**在 PHG 已有的 STE mask + topk 剪枝之上，显式稀疏预算与 ADMM 乘子反馈是否带来增量。**

## 1. 问题定义

### 1.1 GaussianSpa 做了什么

GaussianSpa 把训练写成带硬稀疏约束的优化问题：

```text
min_{a, Θ} L(a, Θ)   s.t.   ‖a‖₀ ≤ κ
```

其中 `a` 是每个高斯的保留系数（0/1 稀疏掩码），`Θ` 是其余参数。求解用 ADMM 在训练循环内交替三步（每若干步一次）：

1. **优化步**：对 `L + ρ/2·‖a − z + u‖²` 做梯度下降，其中 `z` 是上一轮的硬掩码，`u` 是乘子；
2. **稀疏步**：`z ← TopK(a + u, κ)`，只保留系数最大的前 κ 个；
3. **乘子更新**：`u ← u + a − z`。

论文报告相对 vanilla 3DGS：高斯数降到约 1/6–1/10，PSNR 提升约 0.4–0.9 dB。

**注意**：它的基线是 vanilla 3DGS（没有任何 anchor/熵编码结构），所以这个数字不能直接迁移到 PHG。

### 1.2 PHG 现状：已经有“半个 GaussianSpa”

PHG 已经天然具备大部分组件：

| 组件 | PHG 现状 |
| --- | --- |
| soft 稀疏系数 | `get_mask`（[N,10,1]，sigmoid + STE 二值化）、`get_mask_anchor`（[N,1]，anchor 级 soft score） |
| 硬投影 | 训练中 `adjust_anchor` 按 mask 阈值剪枝；编码端另有 `mask_keep_ratio` 按 `mask_rate` topk 保留 |
| 稀疏反馈 | STE 让 mask 的梯度回传，`mask` 参数组有自己的 lr/scheduler |

PHG **缺的是**：

1. 显式稀疏预算 `κ`：现有阈值/topk 是启发式，最终稀疏度不可精确控制，也不按预算调度；
2. ADMM 乘子 `u`：剪枝只发生在 `adjust_anchor` 的瞬间，稀疏约束没有持续的对偶反馈；
3. 周期性硬投影与乘子更新：现有剪枝是“剪掉就忘”，GaussianSpa 是“每 D 步重新分配预算”。

所以实验的核心假设是：**在相同的最终存活比例下，ADMM 的预算调度 + 乘子反馈能比“启发式剪枝 / 编码端 topk”找到更好的稀疏-质量权衡。**

### 1.3 对 PHG 的三个可行约束点

| 约束点 | 约束对象 | 直接影响 | 建议 |
| --- | --- | --- | --- |
| SPA-anchor | `get_mask_anchor` 的零范数（进码流 anchor 数） | GPCC 几何 + feat + scaling + offsets + masks + hash 上下文，几乎整条码流 | **先做** |
| SPA-offset | `get_mask`（每个 anchor 保留的 offset 数） | offsets 字段 + 部分 anchor 存活 | 第二步 |
| SPA-opacity | 渲染基元 opacity | 只减渲染计算，opacity 不进码流 | **不做** |

SPA-anchor 是最直接的：编码端 `mask_anchor == True` 的 anchor 才写几何与属性，约束它的零范数就是约束码流体积上限。

### 1.4 与现有创新点/实验的关系

| 方向 | 层次 | 与 SPA 的关系 |
| --- | --- | --- |
| I2 公式量化、I6 敏感度监督 | 每个符号的量化步长 | 正交：SPA 减少符号数量，I2/I6 改变符号质量 |
| CompGS 残差编码（R 实验） | 熵模型/符号编码 | 正交：SPA 在表示侧，R 在编码侧，可叠加 |
| P0 系列（条件熵） | 熵模型上下文 | 正交，且 P0 已关闭 |
| 编码端 `mask_keep_ratio` | 压缩时后处理 | **不是替代品，是控制组**：SPA 要证明训练侧乘子反馈比它更有价值 |

体积口径：SPA 新增的 `a/z/u` 是纯训练态张量，不写入码流（与 I6 的 sensitivity 张量同理）；它们的成本只在训练显存，不进 `total_MB`。

## 2. 输入清单

### 2.1 事实

| 编号 | 内容 | 验证方式 |
| --- | --- | --- |
| F1 | `get_mask_anchor` 返回 [N,1] 的 STE 二值化 soft score，由 `get_mask`（[N,10,1]）平均而来 | 代码（`gaussian_model.py:526`） |
| F2 | 编码端只编码 `mask_anchor == True` 的 anchor（几何 + 属性 + masks） | 代码（`hacpp.py` `encode_attributes`） |
| F3 | `encode_attributes` 已有 `mask_keep_ratio`：编码时按 `mask_rate` topk 保留前 `keep_n` 个 | 代码（`hacpp.py:1082`） |
| F4 | 训练循环每 100 步调一次 `adjust_anchor`（含生长与剪枝），`update_from=1500`、`update_until` 可配 | 代码（`trainer.py:237`、`config.py:242`） |
| F5 | `prune_anchor` 已处理生长/剪枝后张量形状同步（sensitivity 张量有现成先例） | 代码（`gaussian_model.py:1183`） |
| F6 | 训练态张量（如 sensitivity）不写入码流、旧 checkpoint 缺字段可默认初始化 | I6 先例 |
| F7 | mask 参数组已有独立 lr/scheduler（`mask_lr_*`） | 代码（`config.py:209`） |
| F8 | GaussianSpa 的效果基线是 vanilla 3DGS；PHG 已做 anchor 稀疏与掩码，预期增量更小 | 论文 |

### 2.2 假设（待实验证伪/证实）

| 编号 | 假设 | 风险 |
| --- | --- | --- |
| A1 | 显式 κ 预算 + 乘子反馈比启发式剪枝/编码端 topk 更优 | 若 PHG 的 STE + 剪枝已经逼近最优，SPA 无增量 |
| A2 | 约束 mask 稀疏度会等比例减少 total_MB（几何+属性），渲染损失可接受 | 需要全流程 compress→decode→eval 验证 |
| A3 | ADMM 的 κ schedule 能与现有 densify 共存 | κ 固定会抑制生长，必须按比例调度 |
| A4 | SPA-anchor 比 SPA-offset 更划算 | 一个 anchor 对应 1 组几何 + 50 维 feat + 6 维 scaling + 10×3 offsets |

### 2.3 被剥离的不可靠假设

1. “GaussianSpa 的 +0.4~0.9 dB 会直接迁移”——基线不同（vanilla vs HAC++/Scaffold-GS）；
2. “mask 稀疏不需要训练侧改动，编码端 topk 就够了”——这正是 SPA 要回答的问题，不能预设；
3. “κ 固定一个值即可”——与生长机制冲突，需要随迭代调度的 budget schedule；
4. “ADMM 每一步都要执行”——需要周期性更新（对齐 `adjust_anchor` 的 100 步间隔），否则训练成本不可接受；
5. “约束 opacity 也能省码流”——opacity 不进码流，约束它只影响渲染，不做。

## 3. 阶段 A：30k 短训消融（先做）

### 3.1 目标

三组对照，回答两个问题：

1. SPA-anchor 相对基线（现有 mask 剪枝 + 编码端不额外裁剪）是否带来 BD-rate 增益；
2. SPA 相对控制组 MaskTopk-only（编码端 topk 达到相同稀疏度）是否还有增量——即 ADMM 乘子反馈是否必要。

### 3.2 数据与配置

- 场景：4-28（与 P0、I6 同源），30k 短训，`update_until=15000`；
- 模型：h32、dim50、I2+I6 开启（当前最佳配置，lambda 0.002）；
- 三组：
  - **基线**：现有训练（STE mask + 启发式剪枝），编码端 `mask_keep_ratio=1.0`；
  - **SPA-anchor**：训练侧 ADMM 约束 anchor 稀疏度，编码端 `mask_keep_ratio=1.0`；
  - **MaskTopk-only（控制组）**：基线训练，编码端 `mask_keep_ratio = 最终目标比例`（0.5 / 0.3 / 0.2 三档）。
- SPA-offset 不在阶段 A 第一轮跑，等 SPA-anchor 结果出来后再决定。

### 3.3 SPA-anchor 最小实现

**新增训练态张量（不进码流）：**

```text
a : [N, 1]   软系数，直接复用 mask_rate = get_mask.mean(dim=1)
z : [N, 1]   二值硬掩码，z = TopK(a + u, κ)
u : [N, 1]   ADMM 乘子，初始化全 0，clamp 到 [-1, 1]
```

**每 100 步（与 `adjust_anchor` 同频）执行一次 ADMM 更新，放在 `adjust_anchor` 之后：**

1. 稀疏步：`z ← TopK(a.detach() + u, κ_t)`；
2. 乘子更新：`u ← clamp(u + a.detach() − z, −1, 1)`；
3. 用 `z` 作为本次 `prune_anchor` 的剪枝 mask（替换现有阈值剪枝），同时把 `z` 用于 `mask_rate` 的 STE 前向（保持 `get_mask_anchor` 语义不变）。

**κ schedule（关键设计）：**

```text
κ_t = max(1, round(N_t × ratio))         # 1500 ≤ t ≤ update_until（与生长并行，按当前锚点数比例）
κ_t = max(1, round(N_final × ratio))     # t > update_until（固定预算，停止生长后让剩余锚点重新分配）
```

其中 `N_t` 是当前 anchor 数，`N_final` 是 `update_until` 时的 anchor 数，`ratio ∈ {0.5, 0.3, 0.2}` 三档各跑一组。按比例而不是绝对值，是为了不与 densify 冲突：生长仍然发生，但每次投影都会把超出预算的 anchor 淘汰。

**优化步中的增广项：**

```text
L_total = L_render + λ_rate·L_rate + λ_dssim·L_dssim
        + ρ/2·‖a − z + u‖²
```

`ρ` 从 `1e-3` 起，若训练震荡则调大（`1e-2`）或对 `u` 加更紧的 clamp。`a` 的梯度走现有 STE 路径（`get_mask` 的 `sigmoid + 直通`），不额外引入参数组。

**与现有逻辑的交互：**

- `adjust_anchor` 内的生长逻辑（梯度阈值 + offset_mask）保持不变；
- 剪枝逻辑在 SPA 开启时改用 `z`，关闭时保持原样；
- `mask_keep_ratio=1.0`，确保阶段 A 对比的是训练侧效果，而不是编码端后处理；
- 新增张量 `a/z/u` 随生长/剪枝同步扩展/裁剪（复用 `prune_anchor` 里 sensitivity 张量的同步模式，并在 `anchor_growing` 里对新 anchor 补零）。

### 3.4 SPA-offset 最小实现（第二轮，仅当 SPA-anchor 通过）

把约束下放到 offset 级：

```text
a : [N, 10, 1]   软系数，复用 get_mask 的 sigmoid 输出
z : [N, 10, 1]   每 anchor 保留前 k_cap 个 offset（k_cap ∈ {6, 4, 3}）
u : [N, 10, 1]   ADMM 乘子
```

其余机制与 SPA-anchor 相同。它回答：**在 anchor 数量之外，进一步减少每个 anchor 的 offset 数是否还有净收益**（offsets 是 [N,30] 的字段，仅次于 feat）。

### 3.5 评估与判定

每个 κ 档位跑完后做全流程：

```text
compress → decode → eval（PSNR / SSIM / LPIPS / total_MB / bitstream 组成）
```

主判据（任一档位满足即视为值得继续）：

1. SPA-anchor 相对基线 BD-rate ≤ −3%；
2. 且相对同 κ 的 MaskTopk-only 控制组，BD-rate 增益 ≥1%（证明 ADMM 乘子反馈有增量）。

停止条件（满足任意一条，S 方向关闭）：

1. 所有档位相对基线 BD-rate > −3%；
2. 与 MaskTopk-only 差距 <1%——结论改为“编码端 topk 即可，训练侧 ADMM 无必要”（这本身是有效结论）；
3. 训练不稳定（loss 发散、anchor 数震荡 >20%）且调参后仍无法收敛；
4. 稀疏预算达标但 total_MB 下降比例明显低于 κ 比例（说明剩余码流被其他字段占据，mask 稀疏不是瓶颈）。

### 3.6 预计成本

- 30k 短训：约 1–2 小时/组（5090 A100），4 组（基线 + SPA-anchor 三档）约 4–8 小时，可叠加 GPU 跑；
- 每组 compress + decode + eval 与常规实验同量级；
- 总成本一晚上可完成，与 P0/CompGS 实验量级一致。

## 4. 阶段 B：110k 验证与联合（仅当阶段 A 通过）

1. 在 4-28 上跑 110k 全量：SPA-anchor（最佳 ratio）+ I2 + I6 vs 同配置无 SPA，报告完整 RD 曲线（3 个 lambda）；
2. 跨场景泛化：Deep Blending playroom 110k（`runs/db_playroom_i6_110k_h32_l0p002` 完成后可作为基线），验证 SPA 的收益不是单场景过拟合；
3. 若 CompGS 残差编码（R 实验）阶段 A 通过，则在最小编解码实现之后做一次 SPA+R 联合实验，验证正交性；
4. 报告 bitstream 组成变化（几何 / feat / scaling / offsets / masks / hash / MLP），确认省下的体积来自预期字段。

## 5. 成功标准（汇总）

| 阶段 | 通过条件 |
| --- | --- |
| A | 至少一个 κ 档位 BD-rate ≤ −3%，且相对同 κ 控制组增益 ≥1% |
| B | 110k 场景 BD-rate ≤ −3%，playroom 复现相同方向（允许幅度不同） |
| 联合 | SPA + R 的 BD-rate 不劣于单独 SPA 或单独 R 的更好者（无负向交互） |

## 6. 风险与应对

| 风险 | 应对 |
| --- | --- |
| PHG 已有 STE mask + topk 剪枝，SPA 增量小 | MaskTopk-only 控制组直接给出答案，成本低 |
| κ 固定抑制 anchor 生长 | 按比例调度 `κ_t = ratio × N_t`，生长与淘汰并行 |
| ADMM 乘子震荡导致训练不稳定 | `u` clamp ±1、`ρ` 从 1e-3 起、必要时加大 `ρ`；每 100 步才更新 |
| 新增张量形状随生长/剪枝失配 | 复用 `prune_anchor` / `anchor_growing` 的 sensitivity 同步模式，单元测试覆盖 |
| mask 稀疏只省码流、不省渲染 | PHG 的 `get_mask` STE 前向已经过滤渲染基元，两者同时受益；用 eval 时间验证 |
| 30k 结论与 110k 不一致 | 阶段 B 必须用 110k 复核，以 110k 为准 |

## 7. 关联候选方向（不在本文档范围）

- **CompGS 残差编码**：见 [CompGS残差编码实验设计.md](CompGS残差编码实验设计.md)。它与 SPA 正交，实验顺序可并行；
- **“先提取特征再进概率模型”（VAE/hyperprior 路线）**：尚未立项。若继续，先跑两个 Gate：
  - Gate 0：`scripts/codec_efficiency.py` 算实际 bitstream 与 `−log₂ p(symbol)` 的比值，确认概率模型还有多少冗余；
  - Gate 1：离线 latent 熵对比（变换后 latent 的条件熵 vs 当前字段熵）。
  Gate 0/1 不通过则不做，避免重蹈 P0 覆辙。

## 8. 参考文献

- GaussianSpa: An Optimizing-Sparsity Framework for 3D Gaussian Splatting Compression（CVPR 2025，[arXiv:2411.06019](https://arxiv.org/abs/2411.06019)，[项目页](https://noodle-lab.github.io/gaussianspa/)）
- 3D Gaussian Splatting as Markov Chain Monte Carlo（[arXiv:2404.09591](https://arxiv.org/abs/2404.09591)）——拓扑/稀疏剪枝类参照
- A Survey on 3D Gaussian Splatting Compression（[arXiv:2407.09510](https://arxiv.org/abs/2407.09510)）——compaction 类方法综述
- PHG P0 阶段 A 报告（`/Users/chen/Documents/PHG/docs/P0_stageA_report.md`）
