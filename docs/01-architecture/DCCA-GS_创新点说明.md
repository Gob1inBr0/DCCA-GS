# PHG 创新点说明

> 版本：v2.2（2026-09-02）
>
> 创新点三层定位：
>
> - **主创新（唯一保留）**：渲染敏感度损失的复杂度乘子（＝ 解码端可重算的内容复杂度
>   量化 I2 × 训练期渲染敏感度监督 I6，共享同一个 MLP，零侧信息）。
> - **工程实现（归一起，不主 claim）**：feat_dim 泛化、哈希/熵模型/G-PCC/算术编码工程等。
>   作为"实现/工程贡献"汇总，不单列创新点。
>   ⚠️ **MLP 权重量化已移除**（只省 ~0.5%、口径不自洽），不再作为创新点或贡献点。
> - **基础措施（不作主创新提及）**：SPA（GaussianSpa 式训练侧 ADMM 剪枝）、
>   Mini-Splatting（depth-reinit / blur-split / 贡献面积简化）。二者是既有方法的组件，
>   按**基础措施**使用；除非后期对 SPA 做出能显著提升其效果的改动，才单独作为创新点。

---

## 0. 背景与基线

### 0.1 HAC++ 压缩管线（五个环节）

HAC++ 的压缩管线可以抽象成五个环节：

```text
锚点坐标 → 哈希网格上下文 → mlp_grid 预测熵参数 → 量化 round(x/Q) → 算术编码
```

1. **锚点（anchor）**：场景被体素化为锚点，每个锚点解码 K=10 个神经高斯；
   锚点坐标本身用 GPCC 压缩（不进熵编码器）；
2. **哈希网格上下文**：`encoding_xyz`（1 个 3D 网格 + 3 个 2D 投影面）对锚点
   坐标插值出上下文特征；哈希参数进码流，但输出可由坐标重算（零侧信息）；
3. **`mlp_grid` 预测熵参数**：从上下文预测每个字段的均值（mean）、尺度（scale）、
   混合概率（prob），以及基础量化步长（Q）；
4. **量化**：`round(x/Q)`；训练时用加噪 + 直通梯度（STE）模拟；
5. **算术编码**：把符号的高斯概率离散成整数 CDF，用 integer-CDF 算术编码器
   写码流。feat 额外经过 `Channel_CTX_fea` 通道自回归（逐 10 维组条件解码），
   所以解码必须按固定顺序（feat → scaling → offsets）。

### 0.2 体积口径（共同参照系）

```text
total_MB = (bits_xyz + bits_feat + bits_scaling + bits_offsets
            + bits_masks + bits_hash + bit_mlp + bit_bounds
            + header_bits) / (8 × 1024 × 1024)
```

- `bit_mlp` 官方口径 = Σ(MLP 参数个数) × 32 bit；MLP 量化实验用真实压缩载荷替换；
- `bit_bounds` = 32 × 3 × 2（x_bound 边界，float32）；
- 贡献对应：**主创新（复杂度乘子）**省 `bits_feat/scaling/offsets`（更细的 Q 给重要锚点）；
  **工程实现（MLP 量化）**省 `bit_mlp`；**基础措施（SPA）**省几乎所有与锚点数量成比例的项
  （几何 + 属性 + masks + hash）；**基础措施（MiniSplat）**在固定预算下调整锚点位置。

### 0.3 四个创新点"动在哪里"的速览

| 层 | 内容 | 维度 | 原版基线 | PHG 改动 |
| --- | --- | --- | --- | --- |
| **主创新** | 渲染敏感度损失的复杂度乘子 | 量化步长 Q | `mlp_grid` 输出每锚点 Q 调整量 | 再乘一个**内容复杂度乘子**，并用**渲染敏感度监督**该乘子（I2×I6） |
| **工程实现** | MLP 权重量化 + 算术编码 | 解码器模型体积 | 权重按 float32 计入体积 | 逐通道 PTQ + 静态区间编码 |
| **基础措施** | SPA 剪枝 | 锚点数量 | 训练后一次 topk | 训练中 ADMM 交替"优化-稀疏化" |
| **基础措施** | MiniSplat | 锚点位置 | 固定预算下重排锚点 | 生长停止点做深度/表面重采样，SPA 预算钉在增密前 |

---

## 1. 主创新：渲染敏感度损失的复杂度乘子（I2 + I6 合并）

硬约束：**解码端必须能原样重算出每个锚点的 Q**，否则就要写侧信息
（side information），反而吃掉收益。两个协作机制：

- **机制 A（I2）**：解码端可重算的内容复杂度公式 → Q 乘子（零侧信息）；
- **机制 B（I6）**：训练期用渲染敏感度监督同一个 MLP（只改训练目标，不改码流）。

### 1.2 原版 HAC++ 自适应量化（AQM）：基线机制与伪代码

原版 HAC++ 的自适应量化（adaptive quantization module，AQM）把量化步长从全局
超参变成"每个锚点一个网络输出"。公式与代码一致，按官方 HAC-plus 仓库
`HAC-origin/scene/gaussian_model.py` 核对（论文 arXiv:2501.12255）。

**符号表：**

```text
x        某字段量化前的数值（feat / scaling / offsets）
Q0       基础量化步长：feat=1.0，scaling=0.001，offsets=0.2
q_adj    每锚点量化步长调整量（mlp_grid 输出，[N,1]，repeat 到字段维度）
Q        最终量化步长
x_hat    量化重建值
```

**原版 AQM 伪代码（encode/decode 共用同一路径）：**

```text
# ── 原版 HAC++ AQM ──
feat_context = hash_grid_interp(anchor)            # 解码端可由坐标重算

[mean, scale, prob,
 mean_scaling, scale_scaling,
 mean_offsets, scale_offsets,
 qa, qs, qo] = mlp_grid(feat_context)              # 熵参数 + 3 个 Q 调整量

Q_feat    = 1.0   × (1 + tanh(qa))
Q_scaling = 0.001 × (1 + tanh(qs))
Q_offsets = 0.2   × (1 + tanh(qo))

x_hat = round(x / Q)      # 训练：STE_multistep（加噪/直通）；编码：精确 round
```

**机制要点：**

1. `mlp_grid` 的输入只依赖锚点坐标与哈希网格参数，编码端与解码端看到同一个
   `feat_context`、同一个 `q_adj`；"训练时算出来、解码时拿不到"的输入
   （例如渲染梯度）不能进这条路径，否则必须写侧信息。
2. `1 + tanh(·) ∈ (0, 2)`，保证 Q 恒正且有上界；Q→0 会让整数 CDF 出 NaN，
   AQM 用 tanh 从结构上排除该风险。
3. `qa/qs/qo` 各是 `[N,1]`，repeat 到字段全部通道——一个锚点一个量化粒度。
4. 训练用加噪/直通的 STE 模拟量化，编码用精确 round，两侧 Q 数值完全一致。
5. 原版训练目标只有渲染损失 + 码率项，没有显式的"每个锚点对画面贡献多大"的
   信号——这是 I6 要补的监督信号。

### 1.3 机制 A（I2）：解码端可重算的内容复杂度量化

I2 在 AQM 之外再乘一个乘子：

```text
Q_field = Q0_field × (1 + tanh(q_AQM_field)) × m_field

m_field = 1 + tanh(z_field) × α
z       = mlp_complexity(公式输入)        # [N, 3]，每字段一个
α       = complexity_scale × ramp_progress
ramp_progress = clamp((step − start_iter) / ramp_iters, 0, 1)
```

默认 `complexity_scale=0.35`、`start_iter=20000`、`ramp_iters=10000`——Q 乘子在
训练后期逐步从 1.0 放大到目标幅度，避免训练初期扰动。

**`mlp_complexity` 结构：** `Linear(4 → hidden) + ReLU + Linear(hidden → 3)`
（PHG v2 起 4 维输入，删去了旧 8 维输入中 4 个恒零的照片统计），
hidden 默认 `feat_dim//2`；架构扫描（在旧 8 维输入上进行，即 8→32→3）确定
**hidden=32、1 层最优**。

**4 维公式输入（全部解码端可重算）：**

```text
1. local_density     = exp(−NN_dist / voxel_size)        # 局部密度
2. scale_anisotropy  = std(mean_scaling[:, :3])          # scale 前 3 维各向异性
3. offset_energy     = mean(|mean_offsets|)              # offset 平均能量
4. active_mask_ratio = mean(masks)                       # 掩码激活比例
```

其中 `mean_scaling` / `mean_offsets` 是 `mlp_grid` 预测的熵参数均值，
`masks` 是锚点的掩码——这三样在解码端都有，所以 Q 不需要写码流，只写全局参数
`content_aware_q_meta.json`。四个量都是"内容复杂度"的代理特征，不需要梯度就能算，
编码端与解码端一致。

**局部密度的确定性采样（保证解码端一致）：**

```text
N ≤ 4096：全对距离 torch.cdist(anchor, anchor)，最近邻距离
N > 4096：sample_idx = round(linspace(0, N−1, 4096))，确定性采样 4096 个锚点
          （禁止随机采样），分 chunk 计算 cdist(anchor_chunk, sample_anchors).min
```

**接入点：** `_codec_apply_content_aware_quant_params` 是唯一入口，训练
（`_estimate_rate_terms`）、编码（`encode_attributes`）、解码
（`decode_attributes`）、率估计四端共用，保证 Q 完全一致。

### 1.4 机制 B（I6）：渲染敏感度监督

**敏感度采集（训练期，量化前）：**

```text
g_a = ‖∂L_render / ∂a‖₂          # a ∈ {feat [N,50], scaling [N,6], offsets [N,30]}
                                  # 对量化前属性 retain_grad 得到
EMA_a ← α·EMA_a + (1−α)·g_a      # α = sensitivity_ema = 0.99，逐锚点
全局统计：mean ← α·mean + (1−α)·batch_mean(g)（var 同步维护）
```

**目标映射（相对归一化）：**

```text
z_score = (EMA − mean) / clamp(mean, 1e-12)
target  = clamp(1 + strength × tanh(−z_score), 0.1, 2.0)
```

两点事实依据：

1. 梯度范数跨数量级，方差 EMA z-score 会把信号压平，故用相对归一化
   `(EMA−mean)/mean`，把"比平均敏感多少倍"变成无量纲分数；
2. 符号方向：`z_score > 0` 表示更敏感 → 乘子 < 1 → Q 变小 → 量化更细；
   乘子必须 clamp 到 `[0.1, 2.0]`，否则 Q→0 会让算术编码的 CDF 出现 NaN。

**监督损失：**

```text
pred   = 1 + strength × tanh(complexity_logits)   # 可导
target 已 detach
L_sens = sensitivity_weight × MSE(pred, target)
```

默认 `sensitivity_weight=1e-3`、`strength=1.0`、`start_iter=20000`。梯度只流向
`mlp_complexity`；编码/解码时 I6 完全不参与，bitstream 里没有任何 I6 字段。

### 1.5 合并后的算法伪代码

```text
# ── 训练每步 ──
if step ≥ content_aware_start_iter:                    # I2 生效
    z = mlp_complexity(formula_input)                  # 4 维公式输入
    m = 1 + tanh(z) × α                                # 内容复杂度乘子
    Q = Q0 × (1 + tanh(q_AQM)) × m                     # 在 AQM 之外再乘
    x_hat = STE_round(x / Q)                           # 量化（训练模拟）

if sensitivity_enabled and step ≥ sens_start_iter:     # I6 生效
    g = |∂L_render / ∂x|                               # 量化前属性梯度范数
    EMA_x ← β·EMA_x + (1−β)·g                          # β = 0.99
    mean ← β·mean + (1−β)·batch_mean(g)
    z_score = (EMA − mean) / clamp(mean, 1e-12)
    target = clamp(1 + γ·tanh(−z_score), 0.1, 2.0)
    pred   = 1 + γ·tanh(complexity_logits)             # 同一 mlp_complexity
    L_sens = w·MSE(pred, target.detach())

loss = L_render + λ·L_rate + L_sens
backward(); optimizer.step()

# ── 编码 / 解码（I6 不参与）──
z = mlp_complexity(formula_input)                      # 与训练同一公式路径
m = 1 + tanh(z) × α
Q = Q0 × (1 + tanh(q_AQM)) × m
x_hat = round(x / Q)   →  算术编码 / 解码
```

### 1.6 与 HAC++ AQM 的关系

| 层 | HAC++ 原版 AQM | PHG I2 | PHG I6 |
| --- | --- | --- | --- |
| Q 来源 | `mlp_grid` 输出 `qa/qs/qo` | 在 AQM 之外再乘 `mlp_complexity` 乘子 | 不产生 Q，只监督 |
| 输入 | 哈希上下文 | 4 维公式特征 | 渲染损失梯度 EMA |
| 参与阶段 | 训练/编码/解码 | 训练/编码/解码 | 仅训练 |
| 码流 | 无侧信息 | 无侧信息（Q 可重算） | 无任何字段 |

I2 是完整的量化机制（编码和解码时真正改变 Q 的就是这条公式路径）；I6 只是
训练目标，把同一个 `mlp_complexity` 训练成"敏感度最优 Q"的近似。合并后的
创新点表述：**解码端可重算的内容复杂度量化 × 训练期渲染敏感度监督，
共享同一个 MLP，零侧信息**。

### 1.7 实验证据与消融

| 项目 | 结果 | 结论 |
| --- | --- | --- |
| I6 监督式（30k/90k 消融） | 约 +0.1dB，零侧信息 | 保留 |
| 复杂度 MLP 架构扫描 | 8→32→3（1 层）最优 | 采用 hidden=32 |
| 公式 logits 与敏感度 EMA 相关性 | feat −0.0003 / scaling 0.0042 / offsets 0.0006 | 公式输入不含敏感度信息 |


### 1.8 边界与理论局限

- I6 是"让公式 MLP 逼近敏感度"，解码端得到的仍是 MLP 的猜测，不是真值；
- 不写侧信息前提下这是当前约束下的解；侧信息成本已验证不划算，解码端可重算的
  敏感度代理特征经相关性门验证目前没有。

---

## 2. 创新点②：MLP 权重量化 + 算术编码（已移出创新点，见卷首说明）

### 2.2 原版 HAC++：MLP 权重的存储方式

原版 HAC++ 对 MLP 权重不量化、不熵编码：

```text
# ── 原版体积口径（bit_mlp）──
bit_mlp = Σ_mlp params(mlp) × 32        # 每个参数固定 32 bit

# ── 原版码流/导出 ──
MLP 权重 → 直接写 float32（原样 .pth / 二进制段）
```

### 2.3 PHG 方法：对称逐通道后训练量化（PTQ）+ 静态区间编码

**对称逐输出通道后训练量化（post-training quantization，PTQ）：**

```text
q     = round(w / scale_c)
scale_c = max_i |w_{c,i}| / (2^(b−1) − 1)     # 每个输出通道一个 scale
w_hat = q × scale_c
```

量化误差上界：

```text
|w − w_hat| ≤ scale_c / 2
            = max_i |w_{c,i}| / (2·(2^(b−1) − 1))
            ≈ 2^(−b) · max_i |w_{c,i}|

b=16：相对误差 ≈ 3.05e-5
b=8 ：相对误差 ≈ 0.78%
```

选择 **per-channel** 而不是 per-tensor：不同输出通道的权重幅度可以差几个数量级，
逐通道 scale 避免大通道把全局 scale 拉大、小通道整体被量化成 0。

**传输格式（`scaffold_gs/mlp_quant.py`）：**

```text
bits ≥ 16：raw int16 直接写入（频率表比符号本身还贵，不做熵编码）
bits < 16：静态 32-bit range coder
           符号频率归一化到 2^16（AC_FREQ_TOTAL），逐值存 v/count 表
载荷 = 熵编码后的整数索引 + 每通道 scale（float32）
```

**整数区间编码（integer range coder）：** 设符号的频率归一化为
`freq`，累计分布为 `cum`，编码器维护区间 `[low, high]`：

```text
r = high − low + 1
low  ← low  + (r × cum)  >> 16
high ← low  + (r × freq) >> 16 − 1

解码：target = ((code − low + 1) × 2^16 − 1) // r
      找到符号使 cum ≤ target < cum + freq，再用同一组公式更新区间
```

这是算术编码的整数实现，避免浮点精度导致的编解码不一致。

支持逐 MLP 混合位宽（`--group-bits mlp_complexity:8 mlp_deform:8 ...`），
可以只对不敏感的 MLP 压到 8-bit，敏感的保持 16-bit。

### 2.4 误差传播分析（结论）

- **16-bit**：误差约 3e-5，远小于训练/推理噪声，质量和熵参数完全不变
  （实测 PSNR 一位不差）；
- **`mlp_grid` 8-bit**：它直接预测熵参数和基础 Q，权重误差被放大成符号概率误差
  → 属性码率上升（attr 5.2284 → 5.4908MB，约 +4.9%），total 反而比 16-bit 更大；
- **`mlp_opacity` 8-bit**：直接改变渲染不透明度 → 最明显的质量损失（−0.054dB）；
- **`mlp_deform` / `mlp_complexity` 8-bit**：前者只修正通道自回归的熵参数，
  后者只调制 Q 乘子（经过 tanh 压缩）→ 对质量和码率几乎无影响。

### 2.5 实验证据

**全位宽扫描（4-28 90k h32 基线，259,061 锚点）：**

| bits | PSNR | SSIM | LPIPS | attr_MB | mlp_payload_MB | total_MB |
| --- | --- | --- | --- | --- | --- | --- |
| 32（基线） | 28.6551 | 0.8921 | 0.2767 | 5.2284 | 0.3320 | 5.5604 |
| 16（全量） | 28.6551 | 0.8921 | 0.2767 | 5.2284 | 0.1926 | 5.4210 |
| 8（全量） | 28.5765 | 0.8917 | 0.2771 | 5.4908 | 0.1273 | 5.6181 |
| 6（全量） | 27.2459 | 0.8830 | 0.2856 | 6.4120 | 0.0882 | 6.5003 |
| 4（全量） | 20.3864 | 0.7898 | 0.3837 | 9.0623 | 0.0591 | 9.1215 |

**逐 MLP 单独 8-bit 消融（90k 基线）：**

| 8-bit 对象 | PSNR | SSIM | LPIPS | total_MB | ΔPSNR |
| --- | --- | --- | --- | --- | --- |
| mlp_opacity | 28.6011 | 0.8919 | 0.2769 | 5.2350 | −0.054 |
| mlp_cov | 28.6376 | 0.8920 | 0.2769 | 5.2390 | −0.017 |
| mlp_color | 28.6458 | 0.8921 | 0.2769 | 5.2368 | −0.009 |
| mlp_grid | 28.6584 | 0.8922 | 0.2767 | 5.5180 | +0.003（attr +0.25MB） |
| mlp_deform | 28.6551 | 0.8921 | 0.2767 | 5.2963 | −0.000 |
| mlp_complexity | 28.6553 | 0.8921 | 0.2767 | 5.2341 | −0.000 |
| 除 grid 外全 8-bit | 28.5722 | 0.8917 | 0.2772 | 5.3276 | −0.083 |

**推荐配置**：`mlp_complexity` + `mlp_deform` 8-bit，其余 16-bit（避免误差叠加）：

| 场景 | PSNR | SSIM | LPIPS | total_MB | mlp_payload_MB |
| --- | --- | --- | --- | --- | --- |
| 4-28 90k | 28.6559 | 0.8921 | 0.2767 | 5.4052 | 0.165 |
| 4-28 110k（当前最优操作点） | 28.8235 | 0.8926 | 0.2771 | 5.4852 | 0.165 |
| Deep Blending playroom 110k | 30.7288 | 0.9130 | 0.2575 | 4.2231 | 0.1653 |
| Deep Blending drjohnson 110k | 30.0376 | 0.9112 | 0.2459 | 6.8549 | 0.1647 |

对应未量化基线：4-28 90k 5.5604 → 5.4052（−0.155MB，质量一致）；playroom
4.3819 → 4.2231；drjohnson 7.0156 → 6.8549。

### 2.6 纪律与边界

- **量化后必须重新 encode → decode → eval**：`mlp_grid / mlp_deform /
  mlp_complexity` 参与熵模型与 Q，不能只换体积行；
- 8-bit 不能无差别套用：`mlp_grid` 会劣化熵模型（属性码率上升），`mlp_opacity`
  直接掉质量；全 8-bit 是负优化；
- 这是**无训练 PTQ**；要压到 6-bit 以下需要量化感知训练（QAT）补偿，当前没有
  证据支持直接压更低。

---

## 3. 创新点③：GaussianSpa 式训练侧 ADMM 剪枝

原版的编码端 topk 是"剪掉就忘"：训练时不考虑最终锚点预算，被剪锚点负责的区域
没有机会在预算约束下再适应（3.7 控制组：同锚点数下比 SPA 低 5.52dB）。

### 3.2 原版 GaussianSpa：问题形式化

以下公式与伪代码按 GaussianSpa 原文（arXiv:2411.06019 v3，CVPR 2025）
第 3.2–3.3 节整理，公式编号沿用原文。

**式 (5)：把 3DGS 训练改写成带预算的约束优化：**

```text
min  L(Θ)
s.t. N(G) ≤ κ
```

约束要求高斯总数不超过目标数 κ。高斯"个数"不可导。

**式 (6)：把"高斯个数约束"转成"不透明度向量的 ℓ0 约束"：**

```text
min_{a,Θ}  L(a,Θ)
s.t.       ‖a‖₀ ≤ κ
```

其中 a ∈ R^N 是全部高斯的不透明度向量，Θ 是其余 3DGS 变量。

### 3.3 原版 Algorithm 1：完整伪代码

```text
Algorithm 1  Procedure of "Optimizing-Sparsifying"

Input:  Gaussian opacity a, 3DGS variables Θ,
        target number of Gaussians κ,
        penalty parameter δ, feasibility tolerance ε, maximum iterations T
Output: Optimized a and Θ

1:  z ← a,  λ ← 0
2:  t ← 0
3:  while ‖a − z‖² > ε  and  t ≤ T do
4:      Update a and Θ with Eq. (14);                 ▷ "Optimizing" Step
5:      Update z with Eq. (16);                       ▷ "Sparsifying" Step
6:      Update λ with Eq. (17);                       ▷ Multiplier Update
7:      t ← t + 1
8:  end while
```

### 3.4 原版公式逐条推导与解释

**式 (7)–(9)：把硬约束从目标里"拆"出来。**

引入指示函数：

```text
h(a) = { 0,      ‖a‖₀ ≤ κ
       { +∞,     otherwise
```

式 (6) 变成无约束形式 `min L(a,Θ) + h(a)`（式 (8)）。引入与 a 同形的辅助变量 z：

```text
min_{a,z,Θ}  L(a,Θ) + h(z)
s.t.         a = z
```

可导的 L 和不可导的 h 各管一个变量，二者之间只有等式约束 a = z。

**式 (10)：增广拉格朗日。**

```text
L(a,z,Θ,λ;δ) = L(a,Θ) + h(z)
               + δ/2·‖a − z + λ‖² + δ/2·‖λ‖²
```

λ 是对偶乘子，δ 是惩罚参数。最后一项 `δ/2·‖λ‖²` 只依赖 λ，PHG 实现中省略。

**式 (11)–(14)："Optimizing" 步。**

固定 z、λ，只对 a 和 Θ 优化：

```text
min_{a,Θ}  L(a,Θ) + δ/2·‖a − z + λ‖²
```

梯度与更新：

```text
∂L/∂a = ∂L(a,Θ)/∂a + δ·(a − z + λ)
∂L/∂Θ = ∂L(a,Θ)/∂Θ

a ← a − η·∂L/∂a
Θ ← Θ − η·∂L/∂Θ
```

只有 a 多一项"被拉向 z 的软惩罚"；Θ 没有直接惩罚，只通过渲染损失学习——
这是"剪枝后再适应"的数学来源。

**式 (15)–(16)："Sparsifying" 步。**

固定 a、λ，对 z 优化：

```text
min_z  h(z) + δ/2·‖a − z + λ‖²
```

这是 h 的**近端算子（proximal operator）**：

```text
z ← prox_h(a + λ)
```

解析解：**把 a + λ 中最大的 κ 个元素保留为 1，其余置 0（TopK）**。

为什么是 TopK：对投影问题

```text
min_z  ‖s − z‖²   s.t.  z ∈ {0,1},  ‖z‖₀ ≤ κ
```

逐坐标展开损失：保留坐标 i 付出 `(s_i − 1)²`，剪掉付出 `s_i²`；保留更优当且仅当
`s_i > 1/2`；数量超过 κ 时按 s 从大到小取前 κ 个。**硬稀疏投影一步 TopK 就是精确解**。

**式 (17)：乘子更新。**

```text
λ ← λ + a − z
```

`a − z` 是当前约束违反量；λ 累积残差，下一轮 `δ(a − z + λ)` 把历史违约变成持续、
方向正确的压力。没有 λ（λ≡0）时每次剪枝后 a 可能重新涨回去；有 λ 则预算约束
有"记忆"。

**收敛判据：** `‖a − z‖² ≤ ε`。

### 3.5 理论性质

1. ℓ0 约束是 NP-hard 的组合选择，但此处投影是逐坐标标量投影，有闭式解
   （原文 5.3 节：逐坐标 TopK 是特例，一步精确）；
2. 整体结构与**交替方向乘子法（ADMM）**同构：优化步 = 原始变量下降，稀疏步 =
   近端投影，乘子步 = 对偶上升；3DGS 损失非凸，论文把它作为启发式交替优化，
   靠实验验证；
3. 与**迭代硬阈值（iterative hard thresholding，IHT）**的区别：IHT 直接对原始
   变量做硬阈值，GaussianSpa 先把 a 与 z 解耦，用乘子记录历史；
4. 原文实现节奏：15k 步开始"优化-稀疏化"，25k 步删除零高斯，之后轻量微调
   （light tuning）。

### 3.6 PHG 适配与实现

#### 3.6.1 原版 → PHG 逐项映射

| 原版 GaussianSpa | PHG SPA | 说明 |
| --- | --- | --- |
| a：逐高斯不透明度 | `a = mean(get_mask)` | PHG 的剪枝单位是锚点（anchor），每个锚点解码 K 个高斯；取掩码均值得到每锚点一个可导软分数 |
| z：硬二值辅助变量 | `spa_z`（0/1 张量） | 由 TopK 更新 |
| λ：对偶乘子 | `spa_u` | 更新式相同，另加 `clamp(±1)` 防发散 |
| δ：惩罚参数 | `spa_rho = 1e-3` | 原版 δ |
| κ：固定目标数 | 预算调度 `κ_t` | 见 3.6.3 |
| ε 收敛判据 | 不判 ε，跑到训练结束 | PHG 与 densify 共存，按固定步数执行 |
| 每步交替 | 每步增广 loss + 每 100 步投影 | 投影挂在 `adjust_anchor` 上 |

#### 3.6.2 训练集成方式

```text
# ── 每步：增广 loss（hacpp.spa_loss_term）──
a = mean(get_mask)                 # [N,1]，STE 可导
loss = L_render + λ·L_rate + ρ/2·mean((a − z + u)²)
backward(); optimizer.step()

# ── 每 100 步（adjust_anchor 内）：稀疏投影 + 乘子更新 + 剪枝 ──
a = mean(get_mask).detach()
κ = 预算调度(step)
scores = a + u
z = TopK(scores, κ)                # 0/1 硬掩码
u = clamp(u + a − z, ±1)
prune_mask = ¬z                    # 剪掉未入选锚点
# spa_z / spa_u 随生长/剪枝同步扩展/裁剪（补零或截断）
```

`spa_z` / `spa_u` 是纯训练态张量，**不写入码流**，体积不变。

#### 3.6.3 预算调度：打破正反馈，防止几何塌缩

第一版用"当前锚点数 × 比例"（`κ_t = ratio × N_t`）存在正反馈环：剪掉一批 →
`N_t` 变小 → `κ_t` 变小 → 下一轮剪更多 → 锚点数指数收缩（实验中衰减到约 80 个
锚点，完全失去表示能力）。

正确版本把预算锚定在**历史最大锚点数**上：

```text
生长期（[spa_start_iter, spa_update_until]）：
  N_ref = max(N_ref, 当前锚点数)
  ratio_t = 1 − (1 − spa_ratio) × progress      # progress ∈ [0,1] 线性推进
  κ = round(N_ref × ratio_t)

停止生长后：
  κ = round(N_final × spa_ratio)
```

剪枝不会自动缩小预算，预算只由时间表 `ratio_t` 决定。生长期内锚点仍可生长
（densify 照常按梯度分裂），每次投影淘汰超出 `κ_t` 的部分；停止生长后 `κ` 固定，
幸存锚点继续训练，把预算重新分配到真正需要的区域。

#### 3.6.4 为什么用 STE 软掩码 a 而不是直接优化 0/1

`z` 是 0/1，不可导；`a` 复用 PHG 已有的 `get_mask` 均值：

```text
a = mean(get_mask)          # sigmoid 输出，STE 直通：前向近似二值、反向有梯度
```

优化步在可导的 `a` 上做梯度下降，投影在 `a + u` 上做 topk，两者通过增广项耦合；
0/1 硬约束语义由 `z` 和最终 `prune_mask` 保证。

### 3.7 实验证据（阶段 A，DB playroom 30k）

配置：I2+I6、dim50、h32、λ=0.002、1600 宽、`update_until=15000`、SPA
ratio=0.5、ρ=1e-3、u clamp ±1。29 个验证视图，compress → decode → eval：

| 方案 | 训练锚点数 | 编码锚点数 | total_MB | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- |
| 基线（无 SPA） | 338,560 | 168,626 | 4.1831 | 30.872 | 0.9134 | 0.2594 |
| MaskTopk-only 0.5 | 338,560 | 169,280 | 4.1958 | 30.872 | 0.9134 | 0.2594 |
| MaskTopk-only 0.1026（对齐 SPA 锚点数） | 338,560 | 34,736 | 1.4740 | 24.120 | 0.8399 | 0.3714 |
| **SPA-anchor 0.5（训练侧 ADMM）** | 50,741 | 34,722 | **1.0835** | **29.639** | 0.8948 | 0.3106 |

**体积构成（bits，SPA vs 基线）：**

| 字段 | 基线 | SPA | 下降 |
| --- | --- | --- | --- |
| 几何 bit_anchor | 2,752,744 | 676,640 | −75% |
| feat | 14,916,904 | 2,645,176 | −82% |
| scaling | 6,388,568 | 1,153,968 | −82% |
| offsets | 6,500,528 | 1,268,640 | −80% |
| masks | 1,307,672 | 292,840 | −78% |
| hash | 430,664 | 258,728 | −40% |
| MLP（32-bit 口径） | 2,784,704 | 2,784,704 | 0% |
| total_MB | 4.1831 | 1.0835 | −74% |

**结论：**

1. SPA 相对基线体积 −74%（4.18 → 1.08MB），PSNR −1.23dB（30.87 → 29.64），
   构成有效的低码率操作点；
2. 同锚点数对照：约 34.7k 编码锚点下，编码端 topk 24.12dB / 1.47MB，SPA
   29.64dB / 1.08MB——SPA 高 5.52dB 且体积小 27%，编码端 topk 无法替代训练侧
   ADMM；
3. 基线编码端 `mask_anchor` 本来就会滤掉约一半锚点（168,626 / 338,560），所以
   MaskTopk-0.5 与基线几乎无差别；
4. SPA 低码率点体积大头变成 MLP 权重（2.78MB 中的 2.57MB）。

### 3.8 状态

**创新点③目前只做了阶段 A 对比实验（单场景、单 ratio、30k），完整实验尚未完成。**
阶段 A 已通过：同锚点数下相对 topk 的增益远超设计阈值（≥1%）。
尚未验证：SPA 在不同 λ 下的行为、训练时长增加的成本、与 I2+I6 的长期稳定性。

---

## 4. 创新点④：Mini-Splatting anchor spatial re-organization

SPA 解决"保留多少锚点"；MiniSplat 解决"哪些锚点值得保留"——只压数量不改变
幸存锚点的空间分布。沿用 Fang & Wang 的观点（count is not the bottleneck,
placement is），把 Mini-Splatting 的 depth-reinit 移植到锚点/HAC++ 世界，
不增加码流侧信息，只改变训练中的锚点位置。

### 4.2 与 SPA 的关系

`depth-reinit + SPA` 的关键设计是：

1. 在生长停止点（30k 协议为 15000，110k 协议为 45000）渲染若干训练相机的深度图；
2. 把有效深度反投影到世界表面，体素去重后生成候选锚点；
3. 调用 `append_depth_anchors` 加入候选锚点；
4. **增密前把 SPA 预算钉死**（`spa_final_n = 增密前锚点数`），
   因此新增锚点只参与"谁活下来"，不会抬高预算；
5. SPA 的 ADMM top-k 在重排后的锚点集上继续训练。

这样隔离了"位置（placement）"与"数量（count）"：增益若来自多塞锚点，体积会
相应变大；若来自重排，体积几乎不动而质量上升。

### 4.3 训练触发点与整体数据流

训练器在每次迭代的尾部、优化器更新之前检查：

```text
if iteration == mini_splat_reinit_iter and not core.mini_splat_done:
    # 1) 钉死 SPA 预算：增密前的锚点数
    core.spa_final_n = n_before
    core.spa_ref_n = max(core.spa_ref_n, n_before)

    # 2) depth-reinit：从训练相机得到表面候选锚点
    candidates = collect_depth_surface_anchors(model, cams, ...)
    depth_added = core.append_depth_anchors(candidates, voxel)

    # 3) 完整版（可选）：贡献面积 + blur-split + 简化
    if mini_splat_full:
        area = compute_contribution_areas(model, cams, ...)
        blur_candidates = collect_blur_split_anchors(...)
        blur_added = core.append_depth_anchors(blur_candidates, voxel)
        keep = topk(area, kappa)
        core.prune_anchor(~keep)
        core.mini_splat_full_selected = True
        core.mini_splat_importance = area[keep]

    core.mini_splat_done = True
```

"生长停止点"是 `optim.update_until`（30k 协议为 15000，110k 协议为 45000）。
在该迭代上，正常的 `adjust_anchor` 不再执行，因此深度重采样不会与"同一轮
生长/剪枝"互相干扰；重采样后，SPA 继续在重排后的锚点集上工作。

### 4.4 depth-reinit 的具体实现

#### 4.4.1 相机选择

固定抽样以保持确定性和控制开销：

```text
views = min(mini_splat_views, len(train_cameras))
stride = len(train_cameras) / float(views)
idx = sorted(set(int(i * stride) for i in range(views)))
cams = [train_cameras[i] for i in idx if i < len(train_cameras)]
```

默认 `views=8`。该选择顺序与随机种子无关，后续解码/复现不需要存相机索引。

#### 4.4.2 深度渲染

`render_scene_depth()` 不使用 RGB 渲染，而是直接用 gsplat 深度模式：

```text
visible_mask = prefilter_anchors(model, cam)
gaussians = generate_gaussians(cam, visible_mask, is_training=False)
depths, alphas, meta = rasterization(
    means=gaussians.xyz,
    quats=gaussians.quats,
    scales=gaussians.scales,
    opacities=gaussians.opacities,
    viewmats=viewmats,
    Ks=Ks,
    render_mode="D",
    packed=True,
)
```

返回的：

- `depths`：`[1, H, W, 1]`，相机坐标系深度；
- `alphas`：`[1, H, W, 1]` 累积不透明度；
- `meta`：包含 `gaussian_ids`、`means2d`、`conics`、`opacities`、
  `isect_offsets`、`flatten_ids` 等，供完整版计算贡献面积；
- `gaussians.gaussian_anchor_indices`：每个被光栅化的高斯回到其全局锚点索引。

这一层在训练时也是"训练后的重采样"，因此不参与当前步的 autograd。

#### 4.4.3 有效深度过滤与反投影

gsplat 背景/无表面位置深度为 0，必须过滤：

```text
valid = isfinite(depth) and depth > 0 and alpha > alpha_min
```

默认 `alpha_min = 0.05`。随后把像素坐标反投影到世界坐标：

```text
x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
[x, y, z, 1] -> world = c2w @ [x, y, z, 1]
```

`fx/fy/cx/cy` 来自 `SceneCamera.K`，且已经由 `max_width` 缩放；`c2w` 是
COLMAP 相机到世界矩阵。反投影结果为 `[N_pixel, 3]`。

#### 4.4.4 体素去重与确定性上限

候选点先按世界坐标体素化：

```text
cell = round(world / voxel_size)
uniq, inverse = unique(cell)
mean_world = scatter_mean(world, inverse)
```

如果去重后的候选数超过 `mini_splat_max_new`，用固定 seed 的 CPU
`torch.randperm` 选择子集，避免 GPU 随机数与解码/复现不一致。

保证：表面点不重复；候选锚点稳定；不会因为一个相机恰好投影大量近景像素而
生成失控数量的锚点。

#### 4.4.5 追加锚点

`append_depth_anchors(candidates, voxel)` 对每个候选锚点初始化：

```text
scaling = log(voxel * ones(6))
rotation = identity quaternion
anchor_feat = zeros(feat_dim)
offset = zeros(K, 3)
mask = ones(K+1)
opacity = inverse_sigmoid(0.1)
```

然后走官方的 `cat_tensors_to_optimizer`，让新锚点进入与旧锚点相同的优化器参数组。
注意：优化器中 MLP/编码器/语义投影头参数组没有逐锚点扩展，必须跳过；
只有 `anchor / offset / mask / anchor_feat / scaling / rotation / opacity`
逐锚点组合并扩展。

#### 4.4.6 关键状态同步

只扩展模型参数是不够的，训练统计张量也必须同步：

```text
anchor_demon              [N, 1]                   per-anchor
opacity_accum             [N, 1]                   per-anchor
offset_gradient_accum     [N*K, 1]                 per Gaussian fan-in
offset_denom              [N*K, 1]                 per Gaussian fan-in
sensitivity_feat/scaling/offsets  [N, 1]           per-anchor
spa_z / spa_u             [N, 1]                   per-anchor
mini_splat_importance     [N, 1]                   per-anchor
semantic_target / cov     [N, dim] / [N, 1]        per-anchor
```

当前实现：

1. `append_depth_anchors` 中追加 `n*K` 长度的 `offset_gradient_accum` /
   `offset_denom`，而不是只追加 `n`；
2. 完整版直接调用 `prune_anchor`（绕过 `adjust_anchor` 的预裁剪）时，统一裁剪
   上述统计张量；
3. 用"统计张量长度是否等于当前实际锚点数 `self._anchor.shape[0]`"判断是否
   已经预裁剪，避免 `adjust_anchor` 路径的二次裁剪。

这一步缺失会导致下一步 `training_statis` 或 `anchor_growing` 出现
`device-side assert` / `IndexError`，是 1-78 大场景培训期最常见的故障源。

### 4.5 完整版实现：贡献面积、blur split 与 intersection-preserving simplification

#### 4.5.1 贡献面积

完整版先用低层 gsplat 相交接口计算每个锚点的"最大贡献像素面积"：

```text
gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
    transmittances, means2d, conics, opacities, ...
)
```

对每个相交像素，用 alpha 合成公式恢复该像素的"argmax contributor"权重，再通过：

```text
global_anchor = gaussian_anchor_indices[gaussian_ids]
area.scatter_add_(0, global_anchor, per_gaussian_count / (H*W))
```

把像素面积累加回全局锚点。`gaussian_ids` 是当前光栅化视图中可见高斯的
局部索引，必须先映射回全局锚点索引，否则面积会统计到错误锚点。

#### 4.5.2 blur split

贡献面积超过 `mini_splat_blur_threshold` 的锚点被认为覆盖了过多像素。
对每个被判断为 blur 的锚点，取其光栅化高斯的世界坐标作为子锚点候选，再做
体素去重并限制在 blur 预算内。默认：

```text
blur_threshold = 0.01
max_new = 4000
depth_budget = max_new // 2
blur_budget = max_new - depth_added
```

#### 4.5.3 简化

增密后重新计算贡献面积，按面积分数保留：

```text
if spa_enabled:
    kappa = n_before * spa_ratio
else:
    kappa = n_before

keep = topk(score, kappa)
core.prune_anchor(~keep)
```

之后把幸存锚点的贡献面积存为 `mini_splat_importance`。后续 SPA 分数为：

```text
scores = importance_normalized + 0.25 * spa_soft_score
```

即把"哪些位置对渲染贡献大"的记忆传给 SPA。

### 4.6 码流与训练端边界

上述所有过程都是训练端行为：

- 不新增逐锚点 side information；
- 不改变 `codec_header`、`FORMULA_INPUT_VERSION`、算术编码顺序；
- 不改变 `total_MB` 的字段定义；
- 完整版的 `mini_splat_importance` 只参与训练，不写入码流；
- 增密会提高"训练锚点数"，但 SPA 预算钉在增密前，因此编码锚点数和码流
  体积不会按增密数量线性上涨。

工程上的额外约束：

- HAC++ 读取 1200 张大图时设置 `ulimit -n 65536`；
- 官方 `capture()` 引用未初始化 `denom` 的开关不要打开；
- 深度模式渲染和贡献面积计算均使用 packed gsplat，避免大场景显存爆炸；
- 大场景图像用 `--no-preload-images` + CPU uint8 缓存；
- 远程训练调度使用 `WAIT_VRAM_MB` 显存门槛，避免和其他任务叠爆。

### 4.7 关键模块

| 模块 | 职责 |
| --- | --- |
| `scaffold_gs/config.py` | `mini_splat_enabled / reinit_iter / max_new / views / voxel / full` |
| `scaffold_gs/hacpp.py` | `mini_splat_reinit()`，固定 SPA 预算并调用重采样 |
| `scaffold_gs/mini_splat.py` | 深度渲染、反投影、体素化、贡献面积、blur split |
| `hacplus/scene/gaussian_model.py` | `append_depth_anchors()`、统计张量同步、`prune_anchor()` |
| `scaffold_gs/trainer.py` | 在 `iteration == mini_splat_reinit_iter` 触发 |

### 4.8 实验结果

**playroom 30k、SPA ratio=0.85、λ=0.004、HAC++ 解码后：**

| 配置 | PSNR | total_MB | 说明 |
| --- | ---: | ---: | --- |
| SPA baseline | 30.2210 | 1.8594 | 无 MiniSplat |
| depth-reinit + SPA | 30.4202 | 1.9051 | +0.199 dB，体积 +2.5% |
| SPA 预算曲线（r=0.52/0.85/0.92/0.97） | — | — | BD-PSNR +0.124 dB，BD-rate −8.6% |

**完整版 vs depth-reinit 跨场景（30k、SPA 0.85、λ=0.004，官方体积口径）：**

| 场景 | full PSNR/MB | depth PSNR/MB | ΔPSNR | Δ体积 |
| --- | ---: | ---: | ---: | ---: |
| playroom | 30.525 / 1.356 | 30.273 / 1.422 | +0.252 | −4.6% |
| drjohnson | 29.416 / 2.082 | 29.504 / 2.225 | −0.088 | −6.4% |
| T&T train | 22.164 / 3.239 | 22.284 / 3.631 | −0.120 | −10.8% |
| T&T truck | 25.391 / 3.248 | 25.380 / 3.434 | +0.011 | −5.4% |
| Mip garden | 26.181 / 5.529 | 26.291 / 6.060 | −0.110 | −8.8% |
| Mip bicycle | 24.171 / 3.824 | 24.255 / 4.270 | −0.085 | −10.4% |
| Mip stump | 25.665 / 2.648 | 25.781 / 2.980 | −0.116 | −11.1% |

结论：

1. depth-reinit 在 playroom 上是"免费重排"，在 SPA 预算曲线上 BD-rate −8.6%；
2. 完整版跨场景的优势是体积更紧凑（约 5–11%），不是普适 PSNR 提升；
3. 完整版只在 playroom/T&T truck 同时改善质量，playroom 换 seed 后增益不稳定；
4. 论文报告 depth-reinit 为主要实现，完整版作为可选/消融版本，
   不能写成"泛化有效"。

---

## 5. 四个创新点的正交性

| 创新点 | 维度 | 省什么 | 与其余三个的关系 |
| --- | --- | --- | --- |
| ① 渲染敏感性复杂度量化 | 量化步长 | 属性字段的码率 | 独立于锚点数量和 MLP 体积 |
| ② MLP 权重量化 | 模型体积 | bit_mlp 固定开销 | SPA 后占比最大，必须组合 |
| ③ SPA 剪枝 | 锚点数量 | 所有与锚点数成比例的项 | 与 ① 正交；低码率点依赖 ② 收尾 |
| ④ MiniSplat | 锚点位置 | 固定预算下重新分配锚点 | 与 ③ 联合，改变"哪些锚点活"而不是增加预算 |

---

## 6. 汇总对照表

**4-28 场景（1600 宽，官方体积口径，含 MLP 权重）：**

| 方案 | PSNR | SSIM | LPIPS | total_MB |
| --- | --- | --- | --- | --- |
| HAC++ 论文 | 28.311 | 0.8900 | 0.2932 | 6.9462 |
| PHG 90k（I2+I6，float32） | 28.655 | 0.8921 | 0.2767 | 5.5604 |
| PHG 90k + MLP 量化（推荐配置） | 28.6559 | 0.8921 | 0.2767 | 5.4052 |
| PHG 110k + MLP 量化（当前最优） | 28.8235 | 0.8926 | 0.2771 | 5.4852 |

**Deep Blending playroom（110k，survey 对照来自 3DGS 压缩综述 repo，场景相同）：**

| 方案 | PSNR | SSIM | LPIPS | total_MB |
| --- | --- | --- | --- | --- |
| HAC++ highrate（survey） | 30.9313 | 0.91297 | 0.25488 | 4.5635 |
| PHG 110k（I2+I6，float32） | 30.7310 | 0.9130 | 0.2575 | 4.3819 |
| PHG 110k + MLP 量化 | 30.7288 | 0.9130 | 0.2575 | 4.2231 |
| PHG 30k + SPA（低码率点，训练步数不同） | 29.639 | 0.8948 | 0.3106 | 1.0835 |

注意：不同场景的 PSNR 绝对值不可直接比较；SPA 行训练步数不同（30k vs 110k），
只表示低码率操作点，不参与同口径排序。

---

## 7. 术语表

| 术语 | 含义 |
| --- | --- |
| anchor | 锚点：体素化后的代表点，每个锚点解码 K=10 个神经高斯 |
| Q / Q0 | 量化步长 / 基础量化步长（feat=1.0、scaling=0.001、offsets=0.2） |
| AQM | 自适应量化（adaptive quantization module）：mlp_grid 输出每锚点 Q 调整量 |
| STE | Straight-Through Estimator：训练期量化噪声直通梯度 |
| PTQ | 后训练量化（Post-Training Quantization） |
| GPCC | 点云几何压缩标准，PHG 用 `tmc3` 压缩锚点整数坐标 |
| Morton 序 | Z 序：codec 固定用它给锚点排序（编码/解码一致） |
| hash-grid context | 哈希网格插值出的上下文特征（`calc_interp_feat`，48 维） |
| mlp_grid | 上下文 → 熵参数（mean/scale/prob）+ 基础 Q 的网络 |
| Channel_CTX_fea | feat 通道自回归熵模型（逐 channel_group 条件解码） |
| bit_mlp | MLP 权重体积（官方口径 32 bit/参数） |
| total_MB | 官方体积口径汇总值 |
| BD-rate | 相对基线在同质量下的码率节省百分比 |
| range coder | 区间编码器（算术编码的整数实现），PHG 静态表用 32-bit range coder |
| ADMM | 交替方向乘子法：优化步 + 硬投影 + 乘子更新的交替优化框架 |
| δ（惩罚参数） | 增广拉格朗日里的二次惩罚权重；PHG 里是 `spa_rho=1e-3` |
| λ / u（对偶乘子） | 记录约束违约历史的拉格朗日乘子；原版 GaussianSpa 用 λ，PHG 用 `spa_u` |
| z | 硬二值辅助变量，TopK 投影的结果；PHG 里是 `spa_z` |
| κ | 锚点/高斯数量预算；PHG 用 `max(N_ref)×ratio_t` 调度 |
| 近端算子 | proximal operator：`prox_h(v) = argmin_z h(z) + ½‖z−v‖²`，此处解析解为 TopK |
| ℓ0 球 | 非零元素数不超过 κ 的向量集合 `{z : ‖z‖₀ ≤ κ}` |
| 侧信息 | side information：解码端无法重算、必须写入码流的额外数据 |

---

## 8. 数据来源

本文档所有数字均来自以下已确认来源，未新增任何未实测数字：

- `../data/experiments.csv`：主对照、MLP 量化、SPA、I6 侧信息/替换、P0、R、
  codec efficiency 全部行；
- `../03-reports/SPA_阶段A报告.md`：SPA 阶段 A 结果与体积构成；
- `PHG_改动说明_框架图版.md`（第 7.8、8、9 节）：MLP 量化表格、体积口径、
  实验汇总；
- 代码：`hacplus/utils/codec_consistency.py`、`scaffold_gs/hacpp.py`、
  `hacplus/scene/gaussian_model.py`、`scaffold_gs/mlp_quant.py`。

原版论文与代码：

- GaussianSpa：Zhang et al., "GaussianSpa: An 'Optimizing-Sparsifying' Simplification
  Framework for Compact and High-Quality 3D Gaussian Splatting," CVPR 2025,
  arXiv:2411.06019。Algorithm 1 与式 (5)–(17) 按 arXiv v3 HTML 原文整理；
- HAC++：Chen et al., "HAC++: Towards 100X Compression of 3D Gaussian Splatting,"
  arXiv:2501.12255。AQM 公式按官方 HAC-plus 代码
  `HAC-origin/scene/gaussian_model.py` 核对。

---

## 9. 明确不属于创新点的部分（供论文叙述时排除）

- **基础措施（非主创新，除非后期大改）**：SPA（GaussianSpa 式训练侧 ADMM 剪枝）、
  Mini-Splatting（depth-reinit / blur-split / 贡献面积简化）。二者是既有方法的组件，
  按**基础措施**使用；若后期对 SPA 做出能显著提升其效果的改动，再单独作为创新点。
- feat_dim 泛化、训练加速、图像缓存、tile 尺寸、deform 加载修复：工程实现；
- I1 层级上下文：消融为中性偏负（+0.003dB、体积 +0.10MB），已默认关闭并删除
  侧信息；
- I5 矢量量化、P0 渐进式编码、I6 替换与侧信息路线、R 残差编码：全部按决策规则
  关闭，可作为"负面结果/消融对照"写入论文但不算创新点。
