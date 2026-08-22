# PHG 改动计划：I1+I2 最小实施与 I5/I6 后续阶段

> 版本：v1.0
>
> 日期：2026-08-12
>
> 状态：待实施
>
> 适用范围：`scaffold_gs` 项目（目录与 GitHub 仓库改名为 PHG，Python 包名保持 `scaffold_gs`）
>
> 关联文档：[创新点5-6设计文档.md](创新点5-6设计文档.md)、[未来改动方向.md](未来改动方向.md)

## 0. 摘要

在现有 `scaffold-gs` 的 `hac_pp` 上实现 I1（纯坐标层级 Anchor-Hash 上下文）与 I2（内容感知公式量化）的完整训练/编码/解码闭环，并落地统一评估体系（BD-rate、体积分解、bit-exact 回环测试）。I6（渲染敏感度加权）与 I5（矢量量化与抖动量化）作为本计划的阶段 2/3 完整实现，阶段 4 做组合实验；v1 代码中只放默认关闭的配置占位。

本版相对原 PHG v1 方案的关键修正：

1. I1 重定义为“解码端可重算的层级上下文”，上下文 = base + parent + level，不再保存视角上下文（`cam_radius`、`high_weight`），不写 `i1_context_u8.bin` / `i1_context_header.json`，不移植 `quantize/decode/write/read_i1_context`。
2. I1 默认关闭、I2 formula 默认开启；v1 交付前跑 I1 on/off 消融，再决定是否翻转 I1 默认值。
3. 编码端与解码端统一使用网格化坐标（`round(x / voxel_size) * voxel_size`）计算全部上下文与 Q，保证 bit-exact。
4. I5/I6 完整纳入本计划：阶段 2 实现 I6（训练期敏感度监督、解码零改动），阶段 3 实现 I5（A3* 格点量化 + 共享抖动），阶段 4 做组合实验；每个阶段都有实现规格、测试与杀条件。

## 1. 范围

### 1.1 本版（v1）交付

- I1 纯坐标层级上下文（默认关，独立开关）。
- I2 formula 内容感知量化（默认开）。
- 新增 `hacplus/utils/codec_consistency.py` 最小集（公式 Q、整数符号、文件分类）。
- 评估基础设施：BD-rate 脚本、体积分解脚本、bit-exact 回环测试。
- I5/I6 配置占位与版本字段（默认关；v1 期间置 True 抛 `NotImplementedError`）。
- 项目与仓库改名、清理未用文件。

### 1.2 明确不移植

- `q_param_mode`、`qmeta_*` 全套、I2 `exact` / `codebook` 模式。
- `coverage_prune`、`hybrid_prune`、`codec_opacity_importance`、`anchor_grad_importance`。
- `progressive_stream`、level-wise 编解码、ROI 解码。
- I1 视角上下文全部状态与存储：`anchor_obs_distance_accum`、`anchor_viewgroup_accum`、`codec_reuse_context`、`codec_bitstream_context`、`codec_bitstream_level_ids`、`_derive_codec_reuse_context`、`_activate_i1_bitstream_context`、`_codec_write/load_i1_context`、`quantize/decode/write/read_i1_context`。
- 训练期 obs-distance 百分位分层（level 改用空间距离 fallback）。

## 2. 关键决策

以下决策直接确定实现方式，实施者不再自行选择。

| 编号 | 决策 | 理由 |
| --- | --- | --- |
| D1 | I1 上下文 = `concat(base, parent, level)`，维度 `grid_context_dim = base_output_dim * 2 + 3` | 移除视角上下文后不再有 4 维 view 特征 |
| D2 | 训练、渲染、编码、解码四条路径使用完全相同的 `calc_context_feat` 输入构成，不传 camera 作为 I1 输入 | 避免训练/编码分布失配；视角相关外观由 feature bank 与下游 MLP 处理 |
| D3 | level 只保留空间 fallback：`norm(anchor - center) / diag`，阈值 0.33 / 0.66 | 解码端可由网格坐标重算，零存储 |
| D4 | 阈值写入 `codec_header.json`，字段 `level_threshold_low` / `level_threshold_high` | 场景可调且不破坏格式 |
| D5 | 编码端先做 `_anchor_int = round(anchor / voxel_size)`，Morton 排序后用 `_anchor = _anchor_int * voxel_size` 计算一切上下文与 Q | 与解码端 GPCC 重建坐标一致 |
| D6 | I2 只实现 formula 模式，`Q = Q0 * (1 + tanh(z) * α)`，`Q0 = (1.0, 0.001, 0.2)` | exact 模式旁路信息过大，历史审查已否定 |
| D7 | 默认配置 `hierarchical_context=False`、`content_aware_quant=True` | I1 历史单独消融为负收益/中性，I2 formula 有正向证据 |
| D8 | v1 交付前必须跑 I1 on/off 消融；若 I1 on 的 BD-rate 优于 off，则把默认翻转为 True 并更新配置注释 | 用数据决定默认值，不靠假设 |
| D9 | codec 相关 checkpoint 字段严格校验，缺字段报错；纯训练字段允许默认兼容 | 防止静默默认导致解码错误 |
| D10 | v1 代码中 I5/I6 配置为 True 时抛 `NotImplementedError`；阶段 2/3 落地时替换为真实实现并删除占位 | 防止 v1 期间误开半实现，阶段落地后不允许保留占位 |

## 3. 配置改动（`scaffold_gs/config.py`）

### 3.1 `ModelConfig` 新增

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `hierarchical_context` | bool | `False` | I1 开关 |
| `hierarchical_context_start_iter` | int | `12000` | I1 生效起始迭代 |
| `content_aware_quant` | bool | `True` | I2 开关 |
| `content_aware_q_mode` | str | `"formula"` | 只允许 `"formula"` |
| `complexity_scale` | float | `0.35` | 复杂度调制强度 α |
| `content_aware_start_iter` | int | `20000` | I2 生效起始迭代 |
| `content_aware_ramp_iters` | int | `10000` | I2 强度 ramp 迭代数 |
| `mlp_complexity_hidden` | int \| None | `None` | 默认 `feat_dim // 2` |
| `mlp_complexity_layers` | int | `1` | 复杂度 MLP 层数 |
| `level_threshold_low` | float | `0.33` | 空间距离分层低阈值 |
| `level_threshold_high` | float | `0.66` | 空间距离分层高阈值 |
| `vq_enabled` | bool | `False` | 矢量量化开关（阶段 3） |
| `vq_lattice` | str | `"a3_star"` | 只支持 A3* 格点 |
| `vq_fields` | str | `"scaling,offsets"` | 参与 VQ 的字段，feat 除外 |
| `vq_group_scale_mode` | str | `"mean"` | 组标度 = 组内逐元素 Q 的均值 |
| `dither_enabled` | bool | `False` | 抖动量化开关（阶段 3） |
| `dither_seed` | int | `0` | 场景级种子，写入 `codec_header.json` |
| `sensitivity_enabled` | bool | `False` | 渲染敏感度加权开关（阶段 2） |
| `sensitivity_weight` | float | `0.1` | 监督损失权重（消融 0.03/0.1/0.3） |
| `sensitivity_strength` | float | `1.0` | 敏感度 Q 映射强度 α_sens |
| `sensitivity_ema_decay` | float | `0.99` | 梯度 EMA 衰减 |
| `sensitivity_start_iter` | int | `20000` | 与 `content_aware_start_iter` 对齐 |

### 3.2 `OptimConfig` 新增

| 字段 | 默认值 |
| --- | --- |
| `mlp_complexity_lr_init` | `0.005` |
| `mlp_complexity_lr_final` | `0.0005` |
| `mlp_complexity_lr_delay_mult` | `0.01` |
| `mlp_complexity_lr_max_steps` | `30000` |

### 3.3 CLI

- `--cfg.model.hierarchical-context False` / `True`
- `--cfg.model.content-aware-quant False` / `True`
- 其余新配置沿用 `--cfg.model.<kebab-case>` 规则。

## 4. I1 核心实现（`hacplus/scene/gaussian_model.py`）

### 4.1 移植最小集

- `compute_anchor_level_ids(anchor)`：只保留空间 fallback 一条路径；输入必须为网格化坐标；阈值从 `codec_header` 或模型配置读取；删除 obs-distance、百分位、bitstream 全部分支。
- `compute_anchor_level_onehot(anchor)`：`one_hot(level_id, 3)`。
- `calc_context_feat(x, anchor_indices=None, caller="")`：
  - `base = calc_interp_feat(x)`；
  - `parent_anchor = round(x / parent_stride) * parent_stride`，`parent_stride = max(voxel_size * update_hierachy_factor, 1e-6)`；
  - `parent = calc_interp_feat(parent_anchor)`；
  - `level = compute_anchor_level_onehot(x)`；
  - 拼接 `concat(base, parent, level)`。
- `is_hierarchical_context_active()`：`hierarchical_context and current_iter >= hierarchical_context_start_iter`；删除 `codec_bitstream_context_active` 分支。

### 4.2 删除内容

- `get_viewpoint_context`、`get_encoding_viewpoint_context`、`classify_view_group`（codec 路径）。
- `anchor_obs_distance_accum`、`anchor_viewgroup_accum`、`codec_reuse_context`、`codec_bitstream_context`、`codec_bitstream_level_ids`、`codec_bitstream_context_active`。
- `_derive_codec_reuse_context`、`prepare_codec_reuse_context`、`require_codec_reuse_context`、`_activate_i1_bitstream_context`、`_codec_write_i1_context`、`_codec_load_i1_context`。
- `eval_grid_mlp` 中针对旧 I1 checkpoint 的“截断 first.weight”兼容分支。
- `grid_context_dim = base_output_dim * 2 + 3`，`mlp_grid` 第一层输入宽度同步修改。

### 4.3 保留状态

- `current_step` / `current_iter`（I2 ramp 与训练切换用）。
- anchor 生长/剪枝时只同步 `current_step` 等纯训练状态，不再同步任何 I1 累积器。

## 5. I2 formula 核心实现（同文件）

### 5.1 复杂度网络

- `mlp_complexity`：输入 `build_formula_complexity_input` 输出，隐藏层 `feat_dim // 2`（可配），输出 3 维（feat/scaling/offsets 各一）。
- `build_formula_complexity_input(anchor, predicted_mean_scaling, predicted_mean_offsets, masks)`：局部密度、尺度各向异性、偏移能量、mask 激活比例；不含任何训练统计量。
- `_estimate_formula_local_density(anchor)`：`N <= 4096` 用全对距离；否则用 `linspace(0, N-1, 4096)` 确定性采样，禁止随机采样。

### 5.2 Q 计算

- `Q0 = (1.0, 0.001, 0.2)`。
- `adj = 1 + tanh(z)`，`z` 为复杂度网络输出。
- `Q = Q0 * adj`，按字段（feat/scaling/offsets）分别应用。
- 强度 `strength = complexity_scale * ramp_progress`，ramp 从 `content_aware_start_iter` 到 `start_iter + ramp_iters`。
- `formula_input_version = "formula_decoder_available_v1"`，encode/decode 两侧校验一致。

### 5.3 导出

- `mlp_complexity` 权重随 MLP payload 导出，文件名与现有 MLP 文件同规则（如 `mlp_codec_*.pth`）。
- `codec_header.json` 记录 `formula_input_version`、`content_aware_q_mode`、`complexity_scale`。

## 6. Codec 改动

### 6.1 新增 `hacplus/utils/codec_consistency.py`（最小集）

只包含：

- `quantization_integer_symbols`（`q_step` / `inv_scale` 两种模式）。
- `classify_codec_file`：删除 `i1_context_payload` / `i1_context_header` 分类；`i1_context_*` 文件视为 unknown 并报错。
- `build_formula_complexity_features`、`formula_complexity_multiplier`。
- `stable_lowest_indices`（如仍需剪枝排序）。
- 版本常量与 header 字段名。

不移植：`quantize_i1_context`、`decode_i1_context`、`encode_i1_context`、`write_i1_context`、`read_i1_context`、`pack_two_bit`、`unpack_two_bit`。

### 6.2 `encode_attributes`

1. 用 `mask_anchor` 筛选 selected anchors（不含 hybrid 剪枝）。
2. `_anchor_int = round(anchor / voxel_size)`，`calculate_morton_order(_anchor_int)` 排序，`_anchor = _anchor_int * voxel_size`。
3. 写 `xyz_gpcc.npz`；保留 `_codec_debug_gpcc_roundtrip` 检查，断言 round-trip 后整数完全一致；若 GPCC 有损，则改为两遍法（先压坐标、解回、再算上下文与 Q）。
4. 对网格坐标分 chunk 计算 `calc_context_feat` → `grid_mlp` → 熵参数与 Q 预测。
5. `content_aware_quant` 时走 formula Q 调制，写 `content_aware_q_meta.json`（只写公式版本与全局参数，不写逐 anchor Q）。
6. 写 `feat/scaling/offsets/masks/hash/MLP/x_bound`；不写任何 `i1_context_*` 文件。
7. 体积统计：`decode_required` 不含 i1 文件，`total_MB` 按新口径计算。

### 6.3 `decode_attributes`

1. 解 `xyz_gpcc.npz` 得到 `_anchor_int_dec`；用与 encode 相同的 Morton 排序（或 `codec_header` 中保存的 `anchor_order_int`）。
2. `anchor_decoded = _anchor_int_dec * voxel_size`，与 encode 使用同一网格坐标。
3. 按同一 chunk 划分计算 `calc_context_feat` → `grid_mlp` → 同一 Q 路径，解码 feat/scaling/offsets。
4. 回环校验：`anchor_int`、`masks`、`context`、熵参数、`q_params`、解码符号逐项哈希一致；任一项不一致即报错。
5. 渲染解码结果时不再调用 `calc_context_feat`（解码后的属性直接使用），与现状一致。

## 7. Adapter 改动（`scaffold_gs/hacpp.py`、`scaffold_gs/hac_core.py`）

### 7.1 `generate_gaussians`

- 训练/渲染路径调用 `calc_context_feat(anchor, anchor_indices=visible_anchor_indices)`，不传 `viewpoint_camera` 作为 I1 输入。
- 每次渲染把 `core.current_step` / `current_iter` 设为当前 step。
- `step >= content_aware_start_iter` 且 `content_aware_quant` 时应用 formula Q 调制（训练噪声与 STE 路径保留）。

### 7.2 `training_statis`

- 签名去掉 `camera` 相关 I1 累积；删除 `anchor_obs_distance_accum` / `anchor_viewgroup_accum` 更新。
- 保留 `growth.accumulate_growth_stats`。

### 7.3 `HACCoreView`

- `decoder_state()` 增加 `complexity_mlp` 权重导出。
- `state_tensors()` / `load_state_tensors()` 只管理 `current_step` / `current_iter` 等纯训练状态。
- `state_dict` / `load_state_dict`：codec 必需字段（`mlp_complexity`、`grid_mlp` 维度）缺失时报错；纯训练字段缺失时按默认初始化。

## 8. 评估基础设施（v1 交付物）

- `scripts/rd_sweep.py` 保留并补 BD-rate 计算（与现有基线 RD 曲线对比）。
- 新增体积分解脚本：按几何（xyz_gpcc）、属性（feat/scaling/offsets/masks）、模型（MLP/hash）、旁路信息（header、content_aware_q_meta）分行输出 MB。
- 统一 bit-exact 回环测试：encode → decode → 哈希核对 → 渲染评估。
- 验收：两个数据集上能稳定复现当前基线 RD 曲线；体积分解口径统一。

## 9. 阶段 2：I6 渲染敏感度加权（完整实现规格）

### 9.1 目标与原理

- 把量化步长（Q）的分配依据从内容复杂度代理改为渲染损失敏感度：`敏感度 = |∂L_render/∂属性|`。
- 解码端约束：敏感度是训练统计量，不能进码流；采用监督路线，让解码端可重算的公式 Q（`1 + tanh(z)`）逼近敏感度最优的 Q。

### 9.2 配置

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sensitivity_enabled` | bool | `False` | 总开关 |
| `sensitivity_weight` | float | `0.1` | 监督损失权重（消融 0.03/0.1/0.3） |
| `sensitivity_strength` | float | `1.0` | Q 映射强度 α_sens |
| `sensitivity_ema_decay` | float | `0.99` | 梯度 EMA 衰减 |
| `sensitivity_start_iter` | int | `20000` | 与 `content_aware_start_iter` 对齐 |

### 9.3 训练实现

- 位置：`scaffold_gs/trainer.py` 累积梯度，`scaffold_gs/losses.py` 计算监督损失。
- 条件：`sensitivity_enabled and step >= sensitivity_start_iter` 时，在 `loss.backward()` 后、优化器步进前累积三个属性的绝对梯度（量化前、训练分支）：
  - `grad_feat = |∂L_render/∂_anchor_feat|`
  - `grad_scaling = |∂L_render/∂_scaling|`（pre-activation 存储值）
  - `grad_offsets = |∂L_render/∂_offset|`
- EMA 更新：`ema ← sensitivity_ema_decay * ema + (1 - sensitivity_ema_decay) * grad`，每步更新。
- 归一化：`s_norm = ema / (ema.mean() + 1e-8)`，三个字段独立计算。
- 监督损失：目标乘数 `m_target = 1 / (1 + sensitivity_strength * s_norm)`；实际乘数 `m_pred = 1 + tanh(z)`（z 为 `mlp_complexity` 输出）；`L_sens = MSE(m_pred, m_target)`，三字段取平均。
- 总损失：`L = L_render + sensitivity_weight * L_sens`；`mlp_complexity` 使用既有 `mlp_complexity_lr_*` 优化器组。

### 9.4 编码/解码

- 零改动：公式输入（局部密度、mean scaling/offsets、masks）、Q 公式、码流文件列表与阶段 1 相同。
- `content_aware_q_meta.json` 增加 `sensitivity_enabled` 记录（训练期元信息，解码不依赖）。

### 9.5 实验与杀条件

- 相关性分析：敏感度 EMA vs 现有复杂度特征，Pearson 相关系数 > 0.9 → 放弃 I6。
- 离线 RD 上界：用真实敏感度重排 Q 离线量化编码，无改善 → 放弃 I6。
- 监督训练三档消融：`sensitivity_weight ∈ {0.03, 0.1, 0.3}`，固定其余配置。
- 验收：BD-rate 优于阶段 1 同配置基线；bit-exact 回环通过；码流文件列表不变。

### 9.6 测试

- 单测：EMA 形状与更新；`L_sens` 梯度回传至 `mlp_complexity`；`sensitivity_enabled=False` 时总损失与阶段 1 逐位一致。
- 冒烟：`sensitivity_enabled=True` 短训 3k，无 OOM、无 NaN。
- 端到端：Web_Scan 30k 一组（最佳 weight），bit-exact 回环 + BD-rate 对比。

## 10. 阶段 3：I5 矢量量化与抖动量化（完整实现规格）

### 10.1 离线可行性（阶段 0）

- 用阶段 1 训练好的模型与真实属性，比较标量量化与 A3* 格点量化的条件符号熵；概率近似必须与最终编码完全一致（10.3）。
- 杀条件：BD-rate 增益 < 2% 且 LPIPS 无改善 → 放弃 VQ；抖动单独评估：LPIPS 无改善或码率上升 → 放弃抖动。

### 10.2 配置

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `vq_enabled` | bool | `False` | VQ 开关 |
| `vq_lattice` | str | `"a3_star"` | 只支持 A3*（三维最优格点之一） |
| `vq_fields` | str | `"scaling,offsets"` | 参与 VQ 的字段；feat 保持标量 |
| `vq_group_scale_mode` | str | `"mean"` | 组标度 = 组内逐元素 Q 的均值 |
| `dither_enabled` | bool | `False` | 抖动量化开关 |
| `dither_seed` | int | `0` | 场景级种子，写入 `codec_header.json` |

### 10.3 量化与概率规则

- 分组：scaling 6 维 → 2×3；offsets 30 维 → 10×3；feat 50 维不做。
- 组标度：`s_g = mean(Q_group)`，Q 为熵模型 + I2 formula 调制后的逐元素 Q；s_g 解码端可重算，不进码流。
- 最近格点：`v = nearest_lattice_point_a3_star(x / s_g)`，符号为整数坐标；反量化 `x_hat = v * s_g`。
- 抖动（subtractive dither）：`u_g = PRNG(dither_seed, anchor_index, field, group)`，`u_g ∈ [0,1)^3`；编码 `v = Q_lattice(x / s_g + u_g)`；解码 `x_hat = (v - u_g) * s_g`。
- 概率模型（因子化平移 bin）：`P(v) = ∏ [Φ((v_i + 0.5 - u_i - μ'_i)/σ'_i) - Φ((v_i - 0.5 - u_i - μ'_i)/σ'_i)]`，其中 `μ'_g = mean(μ_group)/s_g`、`σ'_g = sqrt(mean(σ_group²))/s_g`。
- 该近似在离线阶段与 Voronoi 精确概率对比；若符号概率偏差 > 5%，改用 Voronoi 概率方案。
- mask=0 的 offsets 组不产生符号，与现状一致。

### 10.4 确定性随机源

- 新增 `hacplus/utils/dither.py`：SplitMix64 + 均匀采样，numpy 实现；训练侧 torch 包装逐位对齐（同一种子输出完全一致）。
- 种子组合：`(dither_seed, anchor_index, field_id, group_id)`；写入 `codec_header.json` 的是场景级 `dither_seed`。

### 10.5 训练与编解码改动

- 训练：`vq_enabled` 时 scaling/offsets 走 `LatticeQuantSTE`（前向格点值、反向直通）；熵损失用 10.3 概率；`dither_enabled` 时叠加抖动。
- encode：scaling/offsets 按组算格点符号并算术编码；feat 路径与阶段 1 一致；`codec_header.json` 记录 `quant_mode`、`vq_lattice`、`vq_fields`、`dither_enabled`、`dither_seed`。
- decode：读 header → 重算 Q → s_g → 重生成 u_g → 解码格点符号 → 反量化；bit-exact 哈希核对扩展至格点符号。

### 10.6 测试

- 单测：A3* 最近点与已知点集对照；numpy/torch 抖动逐位一致；无抖动 round-trip 符号一致；有抖动 round-trip 反量化与参考一致；mask=0 组无符号。
- 冒烟：`vq_enabled=True`、`dither_enabled=True` 各短训 3k。
- 端到端：VQ 开/关 × 抖动开/关 共 4 组 Web_Scan 30k，输出 2×2 表。

### 10.7 验收

- 离线门槛通过；bit-exact 回环通过；BD-rate 或 LPIPS 至少一项优于阶段 1 基线；否则按杀条件关闭对应开关，保留实现但不进默认配置。

## 11. 阶段 4：组合与论文实验

- 组合矩阵：阶段 1 为基线，I6（最佳 weight）× I5（VQ ± 抖动）做 2×2；I1 按阶段 1 消融锁定的默认值固定。
- 验收：组合 BD-rate 优于任一单点最优；bit-exact 回环通过；体积分解与视觉对比（平滑区、遮挡区）附在报告中。
- 与 v3 衔接：组合通过后，把 I5/I6 并入 QAT + 熵编码联合训练（`未来改动方向.md` 的 v3 路线），作为最终管线。

## 12. 改名与整理

1. 本地 `/Users/chen/Documents/scaffold-gs` → `/Users/chen/Documents/PHG`。
2. 5090 服务器 `~/gsplat2hac` → `~/PHG`。
3. GitHub 仓库 `gsplat2hac` → `PHG`（需用户执行或授权），更新 remote URL。
4. 删除未用 vendored 文件：`hacplus/arguments/`、`hacplus/environment.yml`、`hacplus/submodules/*.zip`。
5. 更新 README、requirements；清理 `**/__pycache__` 与临时产物。
6. 保留 `tests/`、`scripts/`、`LICENSE`。

## 13. 测试计划

### 13.1 单元测试

- 新配置默认值：`hierarchical_context=False`、`content_aware_quant=True`、`content_aware_q_mode="formula"`。
- `mlp_complexity` 形状随 `mlp_complexity_hidden` / `mlp_complexity_layers` 变化。
- `grid_context_dim == base_output_dim * 2 + 3`。
- `compute_anchor_level_ids` 只依赖（网格坐标、场景边界、阈值），相同输入两次调用结果一致。
- I2 formula：encode 侧 Q 与 decode 侧 Q 逐元素一致（float32 全等或 `allclose(rtol=1e-6)`）。
- round-trip：encode → decode 后 `feat/scaling/offsets` 与 encode 前符号完全一致；bitstream 中不存在 `i1_context_*` 文件。
- export → `from_attributes` 后 `mlp_complexity` 权重一致。
- 阶段 2/3 专项单测见 9.6 / 10.6，全部并入最终回归矩阵。

### 13.2 短训冒烟

- 降低 start_iter（I1 `500`、I2 `1000`），3k 内验证 I1/I2 实际生效、无 OOM、anchors 正常增长。
- 跑一组 I1/I2 全关对照，确认不劣化。

### 13.3 回归

- `pytest tests` 全过（原有用例 + 新增 I1/I2 用例）。

### 13.4 端到端

- Web_Scan 30k（`update_until=6000` 封顶）→ compress → decode → eval。
- bitstream 断言：无 `i1_context_u8.bin` / `i1_context_header.json`；含 `content_aware_q_meta.json`、`codec_header.json`。
- decode 后 PSNR 与 encode 前差 < 0.01 dB。
- 质量下限：PSNR ≥ 25.68、SSIM ≥ 0.851、LPIPS ≤ 0.142；bitstream ≤ 3.59 MB（去掉 i1 后应更小，作为上限）。
- I1 on/off 消融：同一数据集两档配置各跑一遍，比较 BD-rate；结果决定 I1 默认值（D8）。

### 13.5 耦合度审计

- `core._` 私有访问只出现在 `hac_core.py`。
- 无平铺 `scene/utils` 导入。
- 无 `sys.path` 修改。

## 14. 假设与默认

1. v1 默认 `hierarchical_context=False`、`content_aware_quant=True`；最终默认由 I1 消融决定。
2. 只支持新 checkpoint 做 codec；旧 checkpoint 兼容仅限纯训练字段。
3. GPCC round-trip 无损；若发现误差，必须采用两遍法（先压坐标、解回、再算上下文与 Q）。
4. 继续使用 gsplat 渲染，不引入 diff-gaussian-rasterization。
5. 30k 训练继续使用 `update_until=6000` 封顶（32GB 显存约束）。
6. GitHub 仓库重命名需用户执行或授权。
7. v1 代码中 I5/I6 配置为 True 时抛 `NotImplementedError`；阶段 2/3 落地时替换为真实实现并删除占位。

## 15. 验收清单

- [ ] 全部单元/冒烟/回归/端到端测试通过。
- [ ] bitstream 不含 `i1_context_*`，round-trip bit-exact 通过。
- [ ] Web_Scan 端到端质量与体积达到 13.4 节门槛。
- [ ] BD-rate 与体积分解脚本可复现基线。
- [ ] I1 on/off 消融结果记录在案，I1 默认值据此锁定。
- [ ] 阶段 2：I6 相关性/离线上界杀条件执行，监督训练 BD-rate 优于阶段 1 基线。
- [ ] 阶段 3：I5 离线门槛通过，VQ/抖动 2×2 端到端完成。
- [ ] 阶段 4：组合消融矩阵完成，bit-exact 通过。
- [ ] 项目改名与清理完成，仓库 remote 指向 PHG。
