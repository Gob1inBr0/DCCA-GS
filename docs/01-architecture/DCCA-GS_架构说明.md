# DCCA-GS 改动说明文档（框架图版）（原 PHG）

> 更新日期：2026-08-16
> 适用代码：本地 `mlp-quant`（HEAD `4833b98`）；主线 `main`（HEAD `61e60d5`）
> 用途：一份文档讲清 PHG 从 `gsplat2hac` 改名重建以来的全部改动、当前架构、
> 创新点接入方式、实验结果与踩坑记录；既可以作为项目交接文档，也可以直接
> 交给另一 AI 画框架图（本文用“模块职责 + 调用关系 + 数据流”的纯叙述方式
> 描述，不包含显式的节点/连线清单）。

---

## 1. 项目概览

### 1.1 PHG 是什么

PHG（PKUGS-HAC-Gsplat）是基于 gsplat 的 Scaffold-GS / HAC++ 神经高斯压缩框架。
它的定位是：**以官方 HAC++ 压缩管线为基线，叠加 PKU-GS 创新点（I1/I2/I6 等），
同时保持“模型可替换 + 稳定属性导出 + 编解码器接口化”的低耦合架构**，方便后续
把 HAC、HAC++、以及新的压缩思路以插件方式接入，而不用重写训练器。

核心组成：

- 渲染底层：gsplat 的 `rasterization`（packed 模式），替代官方
  diff-gaussian-rasterization，负责 anchor 视锥预过滤和神经高斯渲染。
- 压缩核心：vendored 官方 HAC++ `GaussianModel`（哈希网格上下文 + 条件熵模型 +
  GPCC 几何压缩 + 官方 arithmetic 算术编码），通过 `scaffold_gs/hacpp.py` 适配。
- 训练/生长：移植官方 Scaffold-GS 的 anchor 体素初始化、K=10 神经高斯解码、
  层级生长与剪枝，训练循环在 `scaffold_gs/trainer.py`。
- 创新层：I1（层级上下文，默认关）、I2（内容感知公式量化，默认开）、
  I6（渲染敏感度监督，可选开），以及已关闭的 I5（VQ）、P0（渐进式编码）、
  I6 替换/侧信息路线；最新增加 MLP 权重量化与码流效率审计。

### 1.2 三个代码库的位置与关系

| 仓库 | 位置 | 作用 |
| --- | --- | --- |
| PHG（当前项目） | 本地 `/Users/chen/Documents/PHG`；5090 `/home/fansonglin/xieliang/chentong/PHG`；GitHub `goblinIBigBro/PHG` | 主项目：训练、编解码、实验脚本、文档 |
| gsplat-main | 本地 `/Users/chen/Documents/gsplat-main`；5090 以 `pip install -e` 方式依赖 | 渲染库，PHG 通过 `pip install -e` 使用本地源码 |
| HAC-plus-main-v1（旧项目） | 本地 `/Users/chen/Documents/HAC-plus-main-v1` | 论文/旧实验数据、创新点解释文档、旧 HAC 结果 CSV 都在这里 |

关系：PHG 把旧项目（HAC-plus-main-v1）里的创新点设计与实验结果作为依据，
把官方 HAC++ 核心 `hacplus/` 以 vendored 方式搬进 PHG，渲染换成 gsplat，
训练器、编解码器、生长逻辑重新以低耦合方式组织。

### 1.3 历史改名

- 项目原名 `scaffold-gs` / `gsplat2hac`，2026-08 中旬按 `../06-planning/DCCA-GS_改动计划.md` 改名为
  `PHG`（Python 包名仍为 `scaffold_gs`，避免大量 import 改动）。
- GitHub 仓库名同步改为 `PHG`；本地目录 `/Users/chen/Documents/PHG`；
  5090 上从 `~/PHG` 移到 `/home/fansonglin/xieliang/chentong/PHG`。

---

## 2. 版本与分支状态

### 2.1 分支总览（以 2026-08-16 为准）

| 分支 | 状态 | HEAD | 内容 |
| --- | --- | --- | --- |
| `main` | ✅ 主线 V2 | `61e60d5` | I2 默认开、I6 可选、I1 默认关；feat_dim 泛化、训练加速、速度型 runner 已合入；共 27 个提交 |
| `mlp-quant`（当前） | ✅ 实验/分析线 | `4833b98` | MLP 权重量化（per-channel PTQ + 算术编码）、codec efficiency、KL audit、λ-RD 与质量层级 RD 绘图、110k 最优操作点 |
| `i6-sens-replace` | ⚠️ 已关闭但保留 | 本地 `5351e8a`；5090 `1c85b6c` | I6 替换方案/侧信息实验；含 deform hidden 自适应加载修复（未并入 main） |
| `i5-vq` | ❌ 存档 | `9550d68` | I5 矢量量化实验，已关闭 |

注意：

- 本地与 5090 的 `i6-sens-replace` 历史不一致（本地多两个 `load_checkpoint`
  修复提交，5090 通过 scp 同步脚本文件）。
- **deform hidden 自适应修复未并入 main**：用 main 加载 dim16/dim32 checkpoint
  仍可能失败，需要用 `i6-sens-replace` 上的修复或等合并。
- 本地 PHG 工作区当前干净（`git status` 无改动），无 torch/GPU，只做开发与文档；
  实际训练、压缩、评估都在 5090 的 `HAC_5090_a100` conda 环境执行。

### 2.2 main 主线的关键演进（git log 摘要）

1. `0b62b32` Scaffold-GS on gsplat 忠实核心（anchor 体素初始化、K=10 神经高斯、
   opacity/cov/color MLP、视锥预过滤、生长/剪枝、COLMAP 训练与 PSNR/SSIM/LPIPS）。
2. `290d9c0` 修 gsplat 1.5.3 兼容、CLI 与数据集解析，在 RTX 5090 验证。
3. `9a21d0e` 加入 HAC++ 模块：vendored 官方核心、gsplat 渲染、RD 训练、熵编码 codec；
   修 Scaffold means2d 梯度统计与优化器别名问题。
4. `072633f` 用 GPCC/tmc3 + Morton 序压缩 anchor 坐标。
5. `b479b11` 对齐官方 hash-grid 配置（n_features=4、log2=13、2D=15）。
6. `22834c3` 修 HAC++ 训练保真度（>10k 步的量化噪声）、pycolmap 延迟导入、
   记录 PYTHONNOUSERSITE。
7. `1103684` 生长梯度换算到像素单位、packed 光栅化 + chunk 化 prefilter/decode、
   内存日志；Web_Scan 与官方 ~2.3MB 对齐。
8. `463c3b4` 修 HAC++ anchor 生长去重与逐高斯梯度统计、O(G+U) 内存去重。
9. `d73b7d2` 后处理 RD 旋钮：量化步长缩放（q_scale_*）与 anchor-mask 剪枝消融。
10. `2eb11e6` PHG v1：I1 纯坐标层级上下文 + I2 公式量化，项目改名为 PHG。
11. `fb99beb` 把 hacplus 收成规范包：HACCoreView、共享 growth 统计、去 sys.path hack。
12. `e8fb371` I6 渲染敏感度加权监督（阶段 2）。
13. `1046b0a` I6 修复：retain_grad 透传、相对归一化、Q clamp、评估预载。
14. `792c32f` 4-28 显存修复：growth 统计 detach（泄漏根因）、max_width=1600、tile_size。
15. `3ae836b` complexity MLP 架构扫描：**hidden=32、1 层（8→32→3）最优**。
16. `d8290ab` 体积口径把解码器 MLP 权重（32bit/参数）与 xyz 边界计入 total_MB。
17. `8479973` feat_dim 泛化：`Channel_CTX_fea(feat_dim, channel_group)` + codec 通道组。
18. `7718baa` 4-28 训练加速：CPU uint8 图像缓存、Adam foreach、更大 decode chunk。
19. `61e60d5` 速度型 4-28 90k runner：eval 只留最终步，30k/60k 只存 ckpt。

### 2.3 mlp-quant 分支新增内容（相对 main）

- `scaffold_gs/mlp_quant.py`：MLP 权重量化 + 静态算术编码。
- `scripts/mlp_quant_sweep.py`：逐位宽 / 逐 MLP / 混合位宽扫描。
- `scripts/codec_efficiency.py`：实际码流 bits vs 概率模型交叉熵。
- `scripts/kl_audit.py`：经验熵 vs 模型交叉熵（KL 冗余审计）。
- `scripts/plot_lambda_rd_4_28.py` / `plot_rd_envelope_4_28.py` / `plot_rd_step_4_28.py`：
  λ-RD、质量层级包络、step 扫描绘图。
- `scripts/rd_compare_dims.py`：dim16/dim32 的 RD 对比。
- `scripts/sens_replace_gate.py` / `sensitivity_side_info.py`：I6 替换/侧信息结论固化。
- `scripts/step_sweep_4_28.sh`、`step_sweep_wait_4_28.sh`、`finish_120k_wait.sh`：
  轮数扫描与等待脚本。

---

## 3. 总体架构（画图用的叙述版）

PHG 是三层结构，从下到上依次是：

```text
┌─────────────────────────────────────────────────────────┐
│  scripts/（实验层）：训练 runner、压缩、评估、RD、绘图   │
├─────────────────────────────────────────────────────────┤
│  scaffold_gs/（适配层）：配置/数据/模型/渲染/生长/训练/   │
│  编解码器接口/MLP量化 —— 一切与框架相关的代码            │
├─────────────────────────────────────────────────────────┤
│  hacplus/（vendored 官方核心）：GaussianModel、哈希网格、 │
│  Channel_CTX_fea、熵模型、GPCC、arithmetic 编解码         │
└─────────────────────────────────────────────────────────┘
```

### 3.1 分层职责

**第一层：`hacplus/`（官方核心，尽量少改）**

- `hacplus/scene/gaussian_model.py`：官方 HAC++ `GaussianModel`，拥有全部锚点
  参数（anchor/offset/mask/anchor_feat/scaling/rotation/opacity）、哈希网格
  `encoding_xyz`、六个 MLP（opacity/cov/color/grid/deform/complexity）、
  熵模型与生长/剪枝方法；PHG 只往里加 I1/I2/I6 的最小集合与
  `feat_channel_group` 泛化。
- `hacplus/utils/`：`codec_consistency.py`（公式 Q 纯函数、文件分类、版本常量）、
  `encodings.py`/`encodings_cuda.py`（哈希编码、GPCC 封装）、`entropy_models.py`
  （高斯/混合高斯概率与算术编码的 encoder/decoder）、`gpcc_utils.py` 等。
- 依赖：`_gridencoder`、`arithmetic`、`simple_knn`、`tmc3`/GPCC 等 CUDA 扩展，
  只在 5090 的 `HAC_5090_a100` 环境可用。

**第二层：`scaffold_gs/`（适配层，PHG 自己的代码）**

- `config.py`：tyro 可解析的 dataclass 配置（Data/Model/Optim/Train/Eval/Export/
  Compress）。
- `datasets.py`：COLMAP 场景加载、train/val 划分、data_factor 缩放、max_width
  官方口径缩放、CPU uint8 图像缓存。
- `model.py`：`NeuralGaussians` 数据结构、`AnchorParams`、`AnchorDecoder`、
  `BaseGaussianModel` 抽象基类、`ScaffoldGSModel`、模型注册表 `MODELS`。
- `hac_core.py`：`HACCoreView`，全项目唯一允许访问核心私有属性
  （`core._anchor` 等）的入口，其它代码不得直接碰私有字段。
- `hacpp.py`：`HACPlusModel`（把 `BaseGaussianModel` 接口接到官方核心）、
  `HACPlusCodec`（encode/decode）、I2/I6 的生成/监督逻辑、rate loss。
- `renderer.py`：`prefilter_anchors`（深度模式光栅化做视锥过滤）与 `render`
  （packed RGB 光栅化，`retain_grad` 支持）。
- `growth.py`：共享的生长/剪枝统计与参数/优化器状态拼接裁剪。
- `trainer.py`：`run_training` 主循环、`save_checkpoint`/`load_checkpoint`、`evaluate`。
- `codec.py`：`CompressionCodec` 抽象接口 + `RawAttributeCodec` + 注册表。
- `losses.py`：L1/SSIM（本地实现，不依赖 gsplat.losses 版本）。
- `mlp_quant.py`：MLP 权重量化与算术编码（mlp-quant 分支新增）。

**第三层：`scripts/`（实验层）**

- 训练 runner：`runner_4_28_90k.sh`、`step_sweep_4_28.sh`、`step_sweep_wait_4_28.sh`、
  `finish_120k_wait.sh`。
- 压缩/评估：`eval_decoded.py`、`rd_sweep.py`、`volume_breakdown.py`、
  `codec_efficiency.py`、`kl_audit.py`、`mlp_quant_sweep.py`。
- 创新点分析：`sensitivity_gate.py`、`sens_replace_gate.py`、
  `sensitivity_side_info.py`、`sweep_mlp_complexity.py`、
  `p0_offline_entropy.py`、`p0_offline_entropy_rawctx.py`、`p0_offline_reverse.py`。
- 绘图：`plot_lambda_rd_4_28.py`、`plot_rd_envelope_4_28.py`、`plot_rd_step_4_28.py`、
  `rd_compare_dims.py`。

### 3.2 关键调用链（叙述版）

- **训练**：`train.py train` → `run_training` → `ColmapDataset`（加载场景）
  → `HACPlusModel`（`init_from_pcd` 体素化）→ 每步 `model.render`（内部
  `prefilter_anchors` → `generate_gaussians` → gsplat `rasterization`）→ 损失
  （L1+SSIM+scale_reg+rate+I6）→ `backward` → `accumulate_sensitivity`
  → `training_statis`/`adjust_anchor` → `optimizer.step` → eval/save。
- **压缩**：`train.py compress --codec hac_pp` → `HACPlusCodec.encode` →
  `export_attributes`（落 `attributes.pth`）→ `encode_attributes`
  （GPCC 几何 + 熵编码属性 + header/q_meta/hac_meta）。
- **解码**：`HACPlusCodec.decode` → `from_attributes` 重建模型 → `decode_attributes`
  （GPCC 解回坐标 → 重算上下文与 Q → 算术解码 → 写回参数）→ `eval_decoded.py`
  渲染评估。

---

## 4. 核心模块详解

### 4.1 `scaffold_gs/config.py`（配置）

六个 dataclass，全部可被 tyro 从命令行解析、存进 checkpoint、再被 codec 复用：

| 配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `DataConfig` | `data_dir`、`result_dir`、`data_factor`、`test_every`、`white_background`、`preload_images`、`cache_images_cpu`、`max_width`、`near_plane`、`far_plane` | `test_every=8` 划分评估集；`max_width` 复刻官方 `resolution=-1`；`cache_images_cpu` 是 4-28 加速的关键 |
| `ModelConfig` | `model_name`、`feat_dim`、`n_offsets`、`voxel_size`、`tile_size`、`appearance_dim`、`ratio`、哈希网格参数、I1/I2/I6 全部开关与超参 | 默认 `feat_dim=32`、`n_offsets=10`、`voxel_size=0.001`、`tile_size=16`（5090 用 32）、`content_aware_quant=True`、`sensitivity_enabled=False`；`vq_enabled/dither_enabled=True` 会抛 NotImplementedError（占位） |
| `OptimConfig` | 全部 LR 计划、`lambda_rate=0.004`、`lambda_dssim=0.2`、`scale_reg_lambda=0.01`、`start_stat=500`、`update_from=1500`、`update_interval=100`、`update_until=15000`、`min_opacity=0.005`、`success_threshold=0.8`、`densify_grad_threshold=0.0002` | 与官方 Scaffold-GS/HAC++ 对齐；含 `mlp_complexity_lr_*` |
| `TrainConfig` / `EvalConfig` / `ExportConfig` / `CompressConfig` | 组合上述配置 + `device`/`seed`/`ckpt`/`out_dir`/`codec` | `compress` 的 `codec` 可选 `none`/`hac_pp` |

### 4.2 `scaffold_gs/datasets.py`（数据集）

- `ColmapDataset`：pycolmap 读 `sparse/0`（自动回退 `sparse/`），按图像名排序，
  每隔 `test_every` 张取一张做验证集；世界坐标保持 COLMAP 原始坐标系，不做归一化
  （与官方一致）。
- `max_width_size`：官方 `resolution=-1` 规则，宽超过 `max_width` 时缩到
  `(max_width, int(h * max_width / w))`，并同步缩放 K；`max_width` 优先于
  `data_factor` 决定最终尺寸（`data_factor` 仍决定图像目录 `images_<factor>`）。
- 图像加载：`preload_images=True` 全部搬 GPU（小场景）；4-28 必须
  `--no-preload-images` 并开 `cache_images_cpu`（CPU uint8 缓存约 5GB RAM，
  每步只把当前 batch 的 uint8 搬 GPU 再 /255）。
- `SceneCamera`：封装 c2w/K/宽高/训练外观 id，`to_gsplat` 输出
  `viewmats [1,4,4]` 与 `Ks [1,3,3]`。
- 延迟导入 pycolmap：必须先建模型（触发 torch_scatter 加载），再导入 pycolmap，
  否则 HAC++ 环境 dlopen 会段错误。

### 4.3 `scaffold_gs/model.py`（模型基座）

- `NeuralGaussians`：一次渲染解码出的高斯集合（xyz/colors/opacities/scales/quats +
  锚点索引 + 训练用 bit-per-param + I6 用 pre-quant 属性 + complexity_logits）。
- `AnchorParams`：锚点参数容器，提供 `cat_`/`prune_` 供生长/剪枝复用。
- `AnchorDecoder`：`predict_gaussians` 是 Scaffold 的“锚点 → 神经高斯”解码入口；
  `mlp_opacity/cov/color` 输入 `concat(feat, view_dir, dist)`，输出 K 个高斯属性；
  appearance embedding 可选。设计上它是 HAC/HAC++ 的替代点（hash 上下文可以
  替换或预处理它）。
- `BaseGaussianModel`（抽象基类）：定义 `init_from_pcd`、`prefilter_anchors`、
  `generate_gaussians`、`render`、`create_optimizer`、`update_learning_rate`、
  `export_attributes`、`training_statis`、`adjust_anchor`。**任何新模型只要实现
  这个接口就能复用 trainer/renderer/growth/codec**。
- `ScaffoldGSModel`：纯 Scaffold-GS 实现（可脱离 HAC++ CUDA 扩展在 CPU/GPU 跑）。
- 注册表：`MODELS = {"scaffold_gs": ..., "hac_pp": HACPlusModel}`，
  `get_model_class(name)` 取类。

### 4.4 `scaffold_gs/hac_core.py`（核心访问的唯一入口）

`HACCoreView` 用显式属性/方法包装官方核心的私有状态：

- 锚点参数：`anchor/offset/mask/anchor_feat/scaling/rotation/opacity`（读+写）。
- 边界与标志：`x_bound_min/max`、`decoded_version`。
- 只读访问器：`get_anchor/get_scaling/get_mask/get_mask_anchor/get_rotation/
  get_encoding_params`。
- 解码器状态：`decoder_state()` 导出 6 个 MLP + `encoding_xyz` 的 state_dict
  （可选 feature bank）；`load_decoder_state()` 反方向加载，并在 `mlp_deform`
  隐藏宽度与 checkpoint 不一致时用 `Channel_CTX_fea(feat_dim, channel_group,
  hidden=ckpt_hidden)` 自动重建（该修复目前在 `i6-sens-replace`，未并入 main）。
- 纯训练状态：`state_tensors()` 只存 `current_step/current_iter`。
- I6 状态：`sensitivity_state()` 管理 per-anchor EMA 与全局 mean/var。
- 哈希参数：`set_hash_params()` 按 use_2D 布局切分 3D/2D 参数。

耦合纪律：**全项目只有 `hac_core.py` 允许访问 `core._xxx` 私有字段**；
`scaffold_gs` 其它文件与 scripts 都通过 `HACCoreView` 或公开方法访问。

### 4.5 `scaffold_gs/hacpp.py`（HAC++ 适配 + 编解码器）

`HACPlusModel`：

- 构造：用 `ModelConfig` 创建官方 `GaussianModel`（传 I1/I2/I6 全部参数），
  包一层 `HACCoreView`。
- `init_from_pcd`：官方 `create_from_pcd`（ratio 采样 → voxelize → 初始 anchor/
  offsets/masks/feat/scaling/rotation/opacity）。
- `create_optimizer`/`update_learning_rate`：单 Adam（eps=1e-15），按参数组
  （anchor lr=0、offset/feature/MLP/encoding_xyz/mask/complexity 各带指数 LR 计划）。
- `generate_gaussians`：见 §5/§7 的量化噪声、I2 Q 调制、I6 retain_grad 逻辑；
  chunk=16384 解码神经高斯控显存。
- `rate_loss_term`：`λ_rate × (bit_per_param + bit_hash/denom)`，`bit_per_param`
  来自 5% 锚点子采样的熵估计（`_estimate_rate_terms`，与官方一致）。
- `sensitivity_supervision` / `accumulate_sensitivity`：I6，见 §7.3。
- `export_attributes`：命名张量（anchor/offset/mask/anchor_feat/scaling/rotation/
  opacity）+ decoder state + config + voxel_size + x_bound，可 `from_attributes`
  round-trip。
- `encode_attributes` / `decode_attributes`：见 §6。

`HACPlusCodec`：

- `encode`：先写 `attributes.pth`（未压缩导出），再 `encode_attributes` 熵编码，
  返回 `hac_meta.json` 的逐字段 bit 统计。
- `decode`：读 `attributes.pth` → `from_attributes` → `decode_attributes` →
  返回可渲染模型。
- 支持后处理旋钮：`q_scale_feat/scaling/offsets`（量化步长乘子）、
  `mask_keep_ratio`（按 mask 分数剪掉低分锚点）、`q_override_*`（per-anchor
  量化乘子覆盖，I6 侧信息实验用）。

### 4.6 `scaffold_gs/renderer.py`（gsplat 光栅化封装）

- `prefilter_anchors(model, camera)`：用 anchor 自身 scale、单位四元数、单位
  opacity，以 `render_mode="D"`（深度模式）+ `packed=True` 光栅化，
  取 `meta["radii"] > 0` 的行作为可见锚点；chunk=65536 防止大场景 OOM。
- `render(...)`：`generate_gaussians` 解码可见锚点 → gsplat `rasterization`
  （`render_mode="RGB"`、`packed=True`、`sparse_grad=False`、`sh_degree=None`，
  颜色为预计算的 RGB）。`retain_grad=True` 时对 `meta["means2d"]` 调用
  `retain_grad()`（生长统计与 I6 都需要）。
- `RenderOutput`：image/alpha/meta/gaussians/visible_mask。
- `tile_size` 可配：5090（sm_120）用 32（64 超共享内存）；tile 越大，packed
  交点内存越小（约按 1/tile² 下降）。

### 4.7 `scaffold_gs/growth.py`（生长/剪枝）

- `accumulate_growth_stats`：每步累加可见锚点的 opacity/访问计数，以及
  2D 梯度范数（packed 模式下先把每个高斯在多个 tile 的梯度 index_add 聚合回
  每个高斯，再按官方语义把像素梯度乘以 (0.5W, 0.5H)）。
  **统计全部 detach**，否则前一步 autograd 图会被 in-place 写入保活
  （~10MB/步的显存泄漏，4-28 长训 OOM 根因）。
- `grow_anchors`：官方层级生长，`torch.unique` 去重候选格点；去重比较用
  chunk 化（4096/块），避免 O(U×G×3) 广播爆显存（之前 20k anchors 尝试申请
  23.93GB 的教训）；新锚点父特征用 `torch.scatter_reduce("amax")` 聚合。
- `prune_params_and_optimizer` / `cat_params_and_optimizer`：生长/剪枝时同步
  拼接/裁剪模型参数与优化器状态（按参数组名匹配 `_anchor/_offset/...`）。
- `adjust_anchor`：先按梯度阈值生长，再按低 opacity + 足够访问次数剪枝；
  每次调用后重置统计并 `torch.cuda.empty_cache()`。

### 4.8 `scaffold_gs/trainer.py`（训练/评估）

- `run_training`：主循环见 §5。
- `save_checkpoint`：`checkpoint.pth` = {model/optimizer/iteration/config} +
  PLY（官方兼容字段）+ MLP state dict 目录。
- `load_checkpoint`：重建模型与优化器，支持 flat state-dict 里读取
  `mlp_deform` 隐藏宽度并自动重建（在 `i6-sens-replace` 上修复）。
- `evaluate`：对验证集渲染，计算 PSNR/SSIM/LPIPS，结果写 `metrics.jsonl`。

### 4.9 `scaffold_gs/codec.py`（编解码器接口）

- `CompressionCodec`（ABC）：`encode(model, out_dir) -> meta`、
  `decode(artifact_dir) -> model`、可选 `rate(meta)`。
- `RawAttributeCodec`（`name="none"`）：只写未压缩 `attributes.pth`。
- 注册表 `CODECS` + `register_codec`：未来 HAC/HAC++ 变体注册后，
  trainer/export/compress 无需改动。

### 4.10 `scaffold_gs/losses.py` / `utils.py`

- 本地 L1/SSIM 实现（版本无关，兼容 gsplat 1.5.3 与新版）。
- `utils.py`：种子、体素化、kNN、指数 LR、四元数/协方差、`camera_extent_radius`
  （作为 spatial_lr_scale）。

### 4.11 `scaffold_gs/mlp_quant.py`（MLP 权重量化，mlp-quant 分支）

- 6 个 MLP 组：`mlp_opacity/mlp_cov/mlp_color/mlp_grid/mlp_deform/
  mlp_complexity`。
- per-channel 对称 PTQ：`q = round(w / scale)`，`scale = per-output-channel
  max|w| / (2^(bits-1)-1)`，反量化 `w_hat = q*scale`。
- 码流：`bits>=16` 直接写 int16 索引；`bits<16` 用自实现的 32-bit 静态算术编码
  （range coder）+ 值/频次表；scale 与形状写 `mlp_quant_meta.json`。
- **关键纪律：量化后必须用反量化权重重新 encode 属性再 decode/eval**
  （因为 `mlp_grid/mlp_deform/mlp_complexity` 参与熵模型与 Q，不能只换体积行）。
- total_MB 口径：量化场景下把 `bit_mlp = params*32` 替换为
  `mlp_quant.bin + mlp_quant_meta.json` 的真实字节。

---

## 5. 训练管线

### 5.1 流程（一个迭代）

```text
1. model.update_learning_rate(iteration)
2. 随机选一个训练相机
3. model.render(cam, background, is_training=True,
                retain_grad=(iteration < update_until), step=iteration)
   ├─ prefilter_anchors：深度模式光栅化 → 可见锚点 mask
   ├─ generate_gaussians：量化噪声/Q 调制 → 神经高斯
   └─ gsplat rasterization：RGB packed
4. loss = (1-λ_dssim)*L1 + λ_dssim*SSIM + λ_scale_reg*scale_reg
        + λ_rate * rate_term
        + (I6 开时) sensitivity_weight * L_sens
5. loss.backward()
6. (I6 开时) accumulate_sensitivity：EMA 更新
7. (start_stat < iter < update_until) training_statis：生长统计
8. (update_from < iter < update_until, 每 100 步) adjust_anchor：生长+剪枝
9. optimizer.step(); optimizer.zero_grad()
10. 每 100 步打印显存并 empty_cache
11. 到 eval/save 步：evaluate / save_checkpoint / save_ply / save_mlp_checkpoints
```

### 5.2 量化噪声（官方保真度关键）

- `3000 < step <= 10000`：feat/scaling/offsets 加均匀量化噪声（Q=1/0.001/0.2）。
- `step == 10000`：`update_anchor_bound()` 固定场景边界。
- `step > 10000`：用 `calc_context_feat → mlp_grid` 预测熵参数与基础 Q，
  I2 生效后叠加公式 Q 调制，再加噪声；`_estimate_rate_terms` 用 5% 锚点
  子采样估计 bit-per-param 进损失。这一步让模型学会承受编码时的硬量化。

### 5.3 显存与加速要点

- **growth 统计 detach**：修掉 ~10MB/步泄漏。
- **tile_size=32**（5090 sm_120 上限；64 超共享内存），packed 交点内存降 ~1/4~1/16。
- **`--no-preload-images` + CPU uint8 缓存**：4-28 1200 张全分辨率 float32
  预载约 28GB 会爆 32GB 显存；CPU 缓存约 5GB RAM，每步只搬当前 batch。
- **Adam `foreach=True`、decode chunk=16384、prefilter chunk=65536**。
- **评估口径**：`data-factor 1 + max-width 1600`（3795×2134 → 1600×899），
  全分辨率评估会低约 0.36dB。
- 每 100 步 `empty_cache()` + 显式 `del out/loss/pred/gt` 释放上一步 autograd 图。

---

## 6. 编解码管线

### 6.1 编码（`encode_attributes`）

```text
1. mask_anchor 筛选（或 mask_keep_ratio 后处理剪枝）
2. anchor_int = round(anchor / voxel_size)
3. calculate_morton_order(anchor_int) 排序；之后所有上下文/Q 用
   anchor = anchor_int * voxel_size（网格坐标）计算
4. compress_gpcc(anchor_int) → xyz_gpcc.npz（含 voxel_size）；写 x_bound_min/max
5. 写 codec_header.json（format=phg_v1、codec、num_anchors、模型配置、
   formula_input_version、anchor_int/masks 的 SHA-256）
6. I2 开时写 content_aware_q_meta.json（mode=formula、version、complexity_scale、
   start/ramp）
7. 按 MAX_batch_size=3000 分 chunk，每 chunk：
   ├─ calc_context_feat(网格坐标) → mlp_grid → mean/scale/prob + 基础 Q(qa/qs/qo)
   ├─ Q = q_scale * Q0 * (1 + tanh(q_*))，I2 生效时再乘复杂度乘子
   ├─ feat：STE 量化 → Channel_CTX_fea 通道自回归（每 channel_group=10/8/... 一组，
   │  混合高斯概率）→ encoder_gaussian_mixed_chunk 写 feat_{s}_{cc}.b
   ├─ scaling：STE 量化 → 高斯熵编码 → scaling_{s}.b
   └─ offsets：STE 量化，mask 为 0 的位清零不编码 → offsets_{s}.b
8. hash.b：encoding_xyz 参数 (params+1)/2 伯努利熵编码
9. masks.b：伯努利熵编码
10. bit_mlp = 所有 mlp 参数 × 32；bit_bounds = 32*3*2（x_bound）
11. total_bits = xyz + feat + scaling + offsets + hash + masks + mlp + bounds
    + header/q_meta 字节*8 → hac_meta.json（含 total_MB）
```

### 6.2 解码（`decode_attributes`）

```text
1. 读 codec_header.json（校验 format/formula_input_version/complexity_scale）
2. 解 xyz_gpcc.npz → anchor_int（校验 SHA-256）→ Morton 排序 → 网格坐标
3. 解码 masks.b（校验 SHA-256）、hash.b
4. 与编码完全相同的 chunk/上下文/Q 路径：
   ├─ feat：通道自回归逐组解码（解码端用已解码的前组做条件）
   ├─ scaling / offsets（offsets 只解 mask 有效位）
5. 写回 _anchor/_anchor_feat/_offset/_scaling/_mask；_rotation 置单位四元数、
   _opacity 置 0.1 反 sigmoid；_decoded_version=True
6. set_hash_params 写回 hash
7. 写 codec_roundtrip_diagnostics.json（bit_exact_roundtrip=True + 哈希）
```

### 6.3 bit-exact 保证

- 编码/解码共用同一份 `calc_context_feat → mlp_grid → Q` 路径与同一网格坐标，
  所以公式 Q 两侧逐元素一致。
- 回环断言：`anchor_int`、`masks` 哈希一致；GPCC round-trip 后整数完全一致
  （若 GPCC 有损则必须两遍法：先压坐标、解回、再算上下文与 Q）。
- `attributes.pth` 中保存的是训练后（未量化）属性，仅用于重建模型结构；
  最终渲染用解码后的量化属性。

---

## 7. 创新点详解

### 7.1 I1：尺度感知层级 Anchor-Hash 条件熵建模（默认关）

- **动机**：把哈希网格上下文从单一分辨率扩展为“基础 + 父级 + 层级”多尺度，
  帮助熵模型表达不同尺度结构。
- **上下文**：`context = concat(base, parent, level)`，其中
  `base = calc_interp_feat(anchor)`（hash 插值，维度 4×12=48）；
  `parent_anchor = round(anchor / parent_stride) * parent_stride`，
  `parent_stride = max(voxel_size * update_hierachy_factor, 1e-6)`；
  `level = one_hot(compute_anchor_level_ids(anchor), 3)`。
  `grid_context_dim = 48*2 + 3 = 99`（I1 关时是 48）。
- **层级**：纯空间距离：`spatial = ||anchor - scene_center|| / ||extent||`，
  阈值 0.33/0.66 → level 0/1/2；解码端可由网格坐标重算，**零旁路信息**。
- **接入点**：`hacplus/scene/gaussian_model.py` 的 `calc_context_feat`/
  `is_hierarchical_context_active`/`compute_anchor_level_ids`；
  `mlp_grid` 第一层输入宽度按 `grid_context_dim` 调整。
- **配置**：`hierarchical_context`、`hierarchical_context_start_iter`（默认
  12000，有坑见 §10）、`level_threshold_low/high`。
- **状态**：⚠️ 可选、默认关。4-28 30k h32 内部消融（5090 实测）：

  | 方案 | PSNR | SSIM | LPIPS | total_MB | anchors |
  | --- | --- | --- | --- | --- | --- |
  | 30k h32（无 I1，I2+I6） | 28.2122 | 0.8902 | 0.2751 | 6.0125 | 317,806 |
  | 30k h32 + I1 | 28.2155 | 0.8901 | 0.2754 | 6.1163 | 317,445 |

  结论：I1 只带来 +0.003dB，SSIM/LPIPS 基本持平，体积反而 +0.10MB，
  属于“中性偏负”，维持默认关。

### 7.2 I2：内容感知公式量化（默认开）

- **动机**：Q 不应全局固定，而应按每个锚点的局部内容复杂度分配
  （平滑区可以粗量化，复杂区需要细量化）。
- **公式**：

  ```text
  Q0 = (feat: 1.0, scaling: 0.001, offsets: 0.2)
  Q  = Q0 * (1 + tanh(z) * strength)
  strength = complexity_scale * ramp_progress
  ramp_progress = clamp((step - start_iter) / ramp_iters, 0, 1)
  ```

  默认 `complexity_scale=0.35`、`start_iter=20000`、`ramp_iters=10000`。
- **复杂度网络**：`mlp_complexity`，输入 4 维公式特征
  `concat(局部密度, scale 各向异性, offset 能量, mask 激活比例, 4 个零 photo 统计)`，
  隐藏层默认 `feat_dim//2`（可配），输出 3 维（feat/scaling/offsets 各一）；
  **扫描出的最优配置是 hidden=32、1 层（8→32→3）**。
- **局部密度**：`exp(-NN_dist / voxel_size)`；N≤4096 用全对距离，否则用
  `linspace(0, N-1, 4096)` 确定性采样（禁随机采样，保证解码端一致）。
- **解码端可重算**：所有输入都来自网格坐标、`mlp_grid` 预测、masks，
  不写逐锚点 Q，只有 `content_aware_q_meta.json` 存全局参数；
  `formula_input_version = "formula_decoder_available_v2_4d"` 两侧校验。
- **接入点**：`_codec_apply_content_aware_quant_params`（训练/编码/解码/率估计
  四路径共用）；`is_content_aware_quant_active` 用 `current_step` 门控。
- **状态**：✅ 默认开，I6 依赖它（I6 监督的就是 `mlp_complexity`）。

### 7.3 I6：渲染敏感度监督（可选开）

- **动机**：I2 用“内容复杂度代理”分配 Q，但真正决定量化损失的是
  **渲染损失对属性的敏感度** `|∂L_render/∂a|`；被遮挡/低影响锚点应粗量化，
  画面中心/高影响锚点应细量化。
- **训练时**：

  ```text
  grad_feat     = |∂L_render/∂feat|   （pre-quant 值 retain_grad）
  grad_scaling  = |∂L_render/∂scaling|
  grad_offsets  = |∂L_render/∂offsets|
  EMA ← α*EMA + (1-α)*grad          （α = sensitivity_ema = 0.99，逐锚点）
  全局 EMA 同时维护 mean/var
  z_score = (EMA - mean) / max(mean, 1e-12)      （相对归一化）
  target  = clamp(1 + strength * tanh(-z_score), 0.1, 2.0)
  pred    = 1 + strength * tanh(complexity_logits)
  L_sens  = sensitivity_weight * MSE(pred, target.detach())
  ```

  `complexity_logits` 保持可导，梯度流向 `mlp_complexity`；target 侧 detach。
  默认 `sensitivity_weight=1e-3`、`strength=1.0`、`start_iter=20000`。
- **编码/解码**：**零改动、零旁路**——I6 只是训练期监督，码流文件与 Q 公式
  不变；`content_aware_q_meta.json` 可记录 `sensitivity_enabled`（训练元信息）。
- **坑**：早期用方差 EMA z-score 归一化会把信号压平（grad 范数跨数量级），
  改为相对归一化；Q 乘子必须 clamp 到 [0.1,2.0]，否则 Q→0 会让算术编码 CDF 出 NaN。
- **状态**：✅ 可用（约 +0.1dB），默认关；论文对照组合为 I2+I6。

### 7.4 I6 替换方案与侧信息（已关闭）

- **替换方案**：新增独立 `mlp_sens`，用解码端可重算特征（哈希上下文、mlp_grid
  输出、masks、公式特征、前 k 个邻居属性）预测敏感度乘子，替代 I2 公式路径。
- **结论（关闭）**：解码端可重算输入与敏感度 EMA 的相关性最佳仅 0.0086
  （杀阈 0.3），预测器学不到敏感度信息；用真值敏感度重排 Q 反而变大
  （5.5604 → 5.7143MB）。
- **侧信息路线**：把 per-anchor 敏感度乘子量化（1/2/3 bit）写入码流。
  理论净收益被 side info 成本吃掉（2-bit ≈ 0.19MB，基线体积 5.5MB 时毛收益
  ≈ 0.42MB）；按决策规则关闭。
- 产物：`scripts/sens_replace_gate.py`、`sensitivity_side_info.py`，
  支持 `q_override_feat/scaling/offsets` 的 encode/decode round-trip 测试。

### 7.5 I5：矢量量化（VQ，已关闭）

- 原设计：scaling/offsets 用 A3* 格点量化 + 可选抖动，组标度取组内 Q 均值，
  概率模型用平移 bin 高斯近似；`vq_enabled/dither_enabled` 配置占位。
- 状态：❌ 存档在 `i5-vq`（HEAD `9550d68`），未合入主线；v1 配置置 True 会抛
  NotImplementedError。

### 7.6 P0：渐进式跨属性编码与空间自适应上下文（已关闭）

- P0-1：以已解码 feat 为条件，拟合残差 MLP 调整 scaling/offsets 的熵参数
  （mean/log-scale），最后一层零初始化保证起点等于基线。
- P0-2：以 Morton 前 k 个邻居（feat/scaling/offsets/mask/坐标差）池化为条件；
  原始邻居特征版（先 Linear+ReLU 再 mean 池化）与 mean/max 池化版同量级。
- P0-reverse：以已解码 scaling/offsets/mask 为条件降低 feat 条件熵。
- **结果**：P0-1 离线增益 1.86~2.30%（<3% 阈值）、P0-2 约 0.06~0.26%
  （<1%）、反向约 0.4%（<2% 且 Δ_reverse=0.0025MB 远小于 Δ_forward=0.0088MB）。
  全部按决策规则关闭，不进入 Stage B/C。
- 产物：`scripts/p0_offline_entropy.py`、`p0_offline_entropy_rawctx.py`、
  `p0_offline_reverse.py`；报告 `../03-reports/P0_阶段A报告.md`。

### 7.7 feat_dim 泛化（已合入 main）

- `Channel_CTX_fea(feat_dim, channel_group)`：组大小自动取 10/8/5/4/2/1 中最大
  可整除者（50→10、32→8、16→8、24→8...）；隐藏宽度默认 `4*channel_group`。
- codec 的 feat 通道自回归循环改用 `core.feat_channel_group`；
  feat_dim ∈ {8,16,24,32,50} 均通过 bit-exact roundtrip。
- 兼容坑：老 checkpoint 的 `mlp_deform` 隐藏宽度是旧 `2×g`，压缩时按 `4×g`
  加载会失败；`load_checkpoint`/`HACCoreView.load_decoder_state` 按
  checkpoint 形状自适应重建（**在 `i6-sens-replace`，未并入 main**）。

### 7.8 MLP 权重量化（mlp-quant 分支，实验结论）

- 目标：total_MB 里 `bit_mlp = params×32` 是大头之一；量化 + 熵编码后可以
  显著降低模型体积。
- 方案：per-channel 对称 PTQ；`bits≥16` 原始 int16，`bits<16` 静态算术编码；
  支持逐 MLP 混合位宽（`--group-bits mlp_complexity:8 mlp_deform:8 ...`）。
- **全位宽扫描（90k h32 基线，259,061 anchors）**：

  | bits | PSNR | SSIM | LPIPS | attr_MB | mlp_payload_MB | total_MB |
  | --- | --- | --- | --- | --- | --- | --- |
  | 32（基线） | 28.6551 | 0.8921 | 0.2767 | 5.2284 | 0.3320 | 5.5604 |
  | 16（全量） | 28.6551 | 0.8921 | 0.2767 | 5.2284 | 0.1926 | 5.4210 |
  | 8（全量） | 28.5765 | 0.8917 | 0.2771 | 5.4908 | 0.1273 | 5.6181 |
  | 6（全量） | 27.2459 | 0.8830 | 0.2856 | 6.4120 | 0.0882 | 6.5003 |
  | 4（全量） | 20.3864 | 0.7898 | 0.3837 | 9.0623 | 0.0591 | 9.1215 |

  关键观察：16-bit 全量**质量完全不变**（-0.139MB）；8-bit 全量反而把
  属性流推大（attr 5.2284→5.4908，`mlp_grid` 低比特让熵模型变差），total 只降
  一点点；6/4-bit 质量崩坏。

- **逐 MLP 单独 8-bit 消融（90k 基线）**：

  | 8-bit 对象 | PSNR | SSIM | LPIPS | total_MB | ΔPSNR |
  | --- | --- | --- | --- | --- | --- |
  | mlp_opacity | 28.6011 | 0.8919 | 0.2769 | 5.2350 | −0.054 |
  | mlp_cov | 28.6376 | 0.8920 | 0.2769 | 5.2390 | −0.017 |
  | mlp_color | 28.6458 | 0.8921 | 0.2769 | 5.2368 | −0.009 |
  | mlp_grid | 28.6584 | 0.8922 | 0.2767 | 5.5180 | +0.003（attr 涨 0.25MB） |
  | mlp_deform | 28.6551 | 0.8921 | 0.2767 | 5.2963 | −0.000 |
  | mlp_complexity | 28.6553 | 0.8921 | 0.2767 | 5.2341 | −0.000 |
  | 除 grid 外全 8-bit | 28.5722 | 0.8917 | 0.2772 | 5.3276 | −0.083 |

  结论：opacity 最敏感（−0.054dB）、grid 会让属性码率上升，deform/complexity
  几乎无损。

- **推荐配置（保守组合）**：complexity + deform 8-bit、其余 16-bit，
  避免误差叠加：
  - 90k 基线：28.6559 / 0.8921 / 0.2767 / **5.4052 MB**（mlp_payload 0.165MB，
    anchors 259,061），质量与 32-bit 基线一致，体积 −0.155MB。
  - 110k 最优操作点：**28.8235 / 0.8926 / 0.2771 / 5.4852 MB**
    （anchors 254,921，mlp_payload 0.165MB）。

---

## 8. 体积口径与构成

### 8.1 官方口径（必须一致）

```text
total_MB = (bits_xyz + bits_feat + bits_scaling + bits_offsets
            + bits_hash + bits_masks + bit_mlp + bit_bounds
            + header_bytes*8) / (8 * 1024 * 1024)
```

- `bit_mlp = Σ(mlp 参数个数) × 32`（官方口径；MLP 量化实验用真实压缩载荷替换）。
- `bit_bounds = 32 × 3 × 2`（x_bound_min/max 各 [1,3] float32）。
- `bit2MB_scale = 8*1024*1024`。

### 8.2 构成分类（volume_breakdown）

`hacplus/utils/codec_consistency.py::classify_codec_file` 把 bitstream 目录文件
分为：`raw_attributes`（不参与体积）、`codec_header`、`formula_header`、
`anchor_gpcc`（xyz_gpcc.npz）、`core_codec`（feat/scaling/offsets .b、bounds）、
`hash`、`mlp`、`masks`、`aux`（不计入）。

### 8.3 4-28 h32 90k 的码流效率（codec_efficiency，5090 结果）

259,061 anchors，实际 bits vs 概率模型交叉熵估计：

| 字段 | 实际 bits | 估计交叉熵 bits | efficiency |
| --- | --- | --- | --- |
| feat | 23,065,952 | 22,970,156.88 | 1.0042 |
| scaling | 9,409,976 | 9,767,245.48 | 0.9634 |
| offsets | 5,871,984 | 5,861,708.90 | 1.0018 |
| masks | 1,458,712 | 1,449,125.83 | 1.0066 |
| hash | 214,864 | 208,325.03 | 1.0314 |
| total | 40,021,488 | 40,256,562.13 | 0.9942 |

**结论**：efficiency ≈ 1.00~1.03，**算术编码器**已经逼近概率模型下界，


KL audit（同模型，`runs/kl_audit/kl_audit.json`）补充“模型本身冗余”：

| 字段 | 模型交叉熵 bits/符号 | 经验熵 bits/符号 | KL bits/符号 |
| --- | --- | --- | --- |
| scaling | 6.2837 | 5.7897 | +0.4941 |
| offsets | 5.7703 | 5.9343 | −0.164 |
| masks | 0.5594 | 0.5594 | 0.0000 |

解读：masks 概率模型完全贴合；offsets 已优于经验直方图参考；**scaling 是唯一
有可测模型冗余的字段（约 0.49 bit/符号，1.55M 符号 ≈ 0.09MB）**，但这是熵模型
层面的改进空间（例如混合高斯/Laplace），不是编码器问题。feat 因通道自回归
结构无法用简单直方图做严格 KL，未列入。

---

## 9. 实验结果汇总

统一口径：4-28 场景、1600 宽评估（data-factor 1 + max-width 1600）、
官方体积口径（含 MLP 32bit/参数）。

### 9.1 主对照

| 方案 | PSNR | SSIM | LPIPS | total_MB | 备注 |
| --- | --- | --- | --- | --- | --- |
| PHG h32 90k（I2+I6） | 28.655 | 0.8921 | 0.2767 | 5.5604 | λ=0.004，259,061 anchors |
| PHG h25 90k（I2+I6） | 28.637 | 0.8922 | 0.2767 | 5.5240 | hidden=25 |
| PHG h32 30k（I2+I6） | 27.879 | 0.8867 | 0.2758 | 6.0130 | |
| PHG h25 30k（I2+I6） | 27.866 | 0.8866 | 0.2762 | ~5.91 | README 口径 |
| PHG dim32 90k | 28.455 | 0.8901 | 0.2796 | 4.5630 | 体积 −17.9%，PSNR −0.20dB |
| PHG dim16 90k | 28.273 | 0.8877 | 0.2815 | 4.0279 | 体积 −27.6%，PSNR −0.38dB |
| PHG 110k + MLP 量化（推荐配置） | 28.8235 | 0.8926 | 0.2771 | 5.4852 | anchors 254,921，当前最优操作点 |
| 旧 ours 90k（CT_HAC 前版本） | 28.563 | 0.8882 | 0.2982 | 6.3547 | 参考基线 |
| HAC++ 论文 | 28.311 | 0.8900 | 0.2932 | 6.9462 | 论文对照 |

锚点数：dim32 90k = 247,513；dim16 90k = 251,056；h32 90k（λ=0.004）=
259,061；h32 110k = 254,921。

### 9.2 λ-RD（两个 λ 点，h32 90k dim50）

| λ | PSNR | SSIM | LPIPS | total_MB | anchors |
| --- | --- | --- | --- | --- | --- |
| 0.004（基线） | 28.655 | 0.8921 | 0.2767 | 5.5604 | 259,061 |
| 0.002（新） | 28.988 | 0.8974 | 0.2708 | 8.3069 | 338,130 |

λ 越小体积越大、质量越高（更多锚点、更细量化）；与旧 HAC 的
`official_hacpp_60k`、`ct_formula_i1_hybrid_90k`、`ct_shared_all_i1_hybrid_90k`
在 λ∈{0.001,0.002,0.004,0.006} 的曲线画在一起
（旧数据：`HAC-plus-main-v1/result/rd_curve_260708_three_scene/
rd_main_curves_3scene_260708.csv`）。

### 9.3 q_scale RD（h32 90k 基线，后处理量化步长缩放）

| q_scale | total_MB | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- |
| 0.75 | 6.0881 | 28.707 | 0.8928 | 0.2761 |
| 0.875 | 5.7970 | 28.685 | 0.8925 | 0.2764 |
| 1.0（基线） | 5.5604 | 28.655 | 0.8921 | 0.2767 |
| 1.125 | 5.3637 | 28.626 | 0.8917 | 0.2771 |
| 1.25 | 5.1977 | 28.587 | 0.8913 | 0.2775 |
| 1.5 | 4.9329 | 28.502 | 0.8902 | 0.2785 |
| 2.0 | 4.5751 | 28.286 | 0.8874 | 0.2810 |

### 9.4 step 扫描（老 h32 90k 训练，30k/60k/90k）

| 训练步数 | total_MB | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- |
| 30000 | 6.9730 | 28.193 | 0.8912 | 0.2744 |
| 60000 | 6.1723 | 28.570 | 0.8935 | 0.2742 |
| 90000 | 5.5604 | 28.655 | 0.8921 | 0.2767 |

趋势：训练越久体积越小、PSNR 越高但边际递减；110k 附近出现当前最优
（28.8235dB/5.4852MB，含 MLP 量化）；120k 训练自身的逐 10k 扫描见 §9.5，
PSNR 峰值确认为 110k。

### 9.5 120k 训练逐 10k 扫描（同一 run 的 60k~120k，decode 后评估）

| 训练步数 | total_MB | PSNR | SSIM | LPIPS | anchors |
| --- | --- | --- | --- | --- | --- |
| 60000 | 6.9627 | 28.5454 | 0.8938 | 0.2739 | 349,384 |
| 70000 | 6.3974 | 28.6095 | 0.8939 | 0.2747 | 312,688 |
| 80000 | 6.1106 | 28.6806 | 0.8937 | 0.2756 | 293,465 |
| 90000 | 5.9427 | 28.6824 | 0.8933 | 0.2763 | 278,490 |
| 100000 | 5.7769 | 28.7705 | 0.8929 | 0.2761 | 265,869 |
| 110000 | 5.6471 | **28.8250** | 0.8926 | 0.2771 | 254,921 |
| 120000 | 5.4911 | 28.8138 | 0.8922 | 0.2779 | 245,500 |

**结论：PSNR 峰值在 110k**（28.8250dB）；120k 体积继续降到 5.4911MB 但 PSNR
回落 0.011dB。110k + MLP 量化即当前最优操作点（28.8235 / 5.4852MB）。
注：该 run 的 90k checkpoint（5.9427MB）与老 h32 90k run（5.5604MB）不是同一
次训练，数字不可直接混用。

### 9.6 Web_Scan feat_dim 粗扫（30k，I2+I6）

| feat_dim | PSNR | SSIM | LPIPS | total_MB |
| --- | --- | --- | --- | --- |
| 8 | 25.163 | 0.8394 | 0.1650 | 2.3275 |
| 16 | 25.735 | 0.8482 | 0.1467 | 2.7067 |
| 24 | 25.756 | 0.8485 | 0.1453 | 3.2309 |
| 32 | 25.856 | 0.8517 | 0.1403 | 3.4440 |
| 50 | 25.752 | 0.8507 | 0.1403 | 3.7859 |

BD-rate（相对 dim50）：**dim16 −24.5%、dim32 −34.0%**；性价比拐点在 8→16。
注意：Web_Scan 30k 与 4-28 90k 结论不完全一致（4-28 90k 对容量更敏感，
dim16/dim32 都是“更小、质量略降”）。

### 9.7 I6 / I1 / P0 / 关闭路线结论

- I6 监督式：约 +0.1dB，零旁路，保留。
- I1（30k h32）：+0.003dB、体积 +0.10MB，中性偏负，默认关。
- I6 替换：相关性 0.0086（杀阈 0.3），关闭。
- I6 真值敏感度重排：5.5604 → 5.7143MB（变大），关闭。
- P0-1/P0-2/反向：增益均低于阈值，关闭。
- I5 VQ：存档关闭。
- MLP 量化：保留并推荐（complexity/deform 8-bit + 其余 16-bit）。

---

## 10. 踩坑清单（必读）

1. **growth 统计显存泄漏**：统计张量未 detach，约 10MB/步，是 4-28 长训 OOM
   的根因；已修复。
2. **gsplat packed 光栅化无上限分配**：大场景交点估算可达 2.5~4B，单步瞬态
   ~30GB；用 `tile_size=32`（5090 sm_120 上限；64 超共享内存）。
3. **4-28 图像预载**：1200 张 float32 全上 GPU ≈ 28GB，会爆 32GB 显存；
   必须 `--no-preload-images`（CPU uint8 缓存约 5GB RAM，每步只传当前 batch）。
4. **评估口径**：必须 `data-factor 1 + max-width 1600`；全分辨率评估低约
   0.36dB（h32 90k：28.655 vs 28.298），不能混用。
5. **I1 start_iter**：默认 12000 会在训练分支 step>10000 时输入维度不匹配；
   必须配 `< 10000`（短实验用 300）。
6. **I6 曾全零失效**：renderer 必须传 `retain_grad`；方差 EMA z-score 归一化
   会被压平，改为相对归一化 `(ema - mean)/mean` + clamp `[0.1, 2.0]`。
7. **deform hidden 加载**：dim16/32 老 checkpoint 是旧 `2×g` 隐藏宽度，压缩时
   按 `4×g` 加载会失败；`load_checkpoint`/`HACCoreView` 已按形状自适应重建
   （在 `i6-sens-replace`，未并入 main）。
8. **体积口径**：官方口径 = 属性流 + 几何 + hash + masks + header + MLP 权重
   （32bit/参数）+ xyz 边界（192 bit）；旧数字没有 `bit_mlp` 是旧口径，不可直接比。
9. **I6 替换/侧信息结论**：预测器输入与敏感度 EMA 相关性最佳仅 0.0086；
   真值敏感度重排反而变大（5.5604 → 5.7143MB），不要重复投入。
10. **P0 停止条件**：P0-1 增益 1.86~2.30%、P0-2 约 0.1%、反向约 0.4%，
    均低于阈值，已关闭。
11. **5090 使用纪律**：不杀其他用户进程；长训练用带“空卡检测+自动重试”的
    runner。
12. **环境**：`conda activate HAC_5090_a100`；`PYTHONNOUSERSITE=1`；
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；PATH 含 tmc3/GPCC；
    延迟导入 pycolmap（先建模型后导 pycolmap，否则 dlopen 段错误）。
13. **不要修改正在运行的 runner 脚本**：bash 按文件偏移增量读取，运行中覆盖
    脚本会让训练结束后的压缩/评估阶段报 `unexpected EOF`（2026-08-16 120k
    运行踩过）；要改就等跑完或用独立新脚本。
14. **MLP 量化必须重编码**：量化 `mlp_grid/mlp_deform/mlp_complexity` 会改变
    熵模型与 Q，必须用反量化权重重新 encode → decode → eval，不能只换体积行。
15. **q_override 一致性**：encode/decode 必须使用同一 override 文件与同一
    排序（codec 的 Morton 顺序），否则符号哈希不一致。

---

## 11. 测试与运行

### 11.1 测试

```bash
pytest tests/ -q
```

- 5090 `HAC_5090_a100` 环境：**19 passed**。
- `tests/test_hacpp_smoke.py`、`tests/test_render_smoke.py` 需要 HAC++ CUDA
  扩展/GPU；其余 CPU 单测（模型/生长/数据集/配置）本地可跑。
- 测试文件：`test_datasets.py`（max_width 规则）、`test_model.py`（体素化/
  形状/解码/export round-trip/注册表）、`test_growth.py`、`test_hacpp_smoke.py`
  （训练步+codec round-trip、channel group、feat_dim round-trip）、
  `test_phg_i1i2.py`（配置默认值、I1 维度与确定性、complexity 形状、I2 Q
  确定性、roundtrip 无 i1 文件）、`test_phg_i6.py`（配置校验、梯度到
  mlp_complexity、生长窗口后生效、q_override roundtrip）、`test_render_smoke.py`。

### 11.2 常用命令（5090）

```bash
conda activate HAC_5090_a100
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH
cd /home/fansonglin/data_space/DCCA-GS/PHG

# 训练（runner：<gpu> <tag> <dim> <max_steps> <update_until> <save> [eval] [hidden] [lambda]）
bash scripts/runner_4_28_90k.sh 1 i6_90k_h32_l0p002 50 90000 45000 \
  "30000 60000 90000" "90000" 32 0.002

# 压缩 + 解码评估
python train.py compress --cfg.ckpt <ckpt.pth> --cfg.out-dir <out> --cfg.codec hac_pp
python scripts/eval_decoded.py --artifact-dir <out> --data-dir <4-28> \
  --result-dir <eval> --data-factor 1 --max-width 1600 --no-preload-images

# 后处理 RD（q_scale）与 step 扫描
python scripts/rd_sweep.py --ckpt <ckpt> --data-dir <4-28> \
  --result-dir runs/rd_xxx --data-factor 1 --max-width 1600 --no-preload-images \
  --q-scale-joint 0.75 0.875 1.0 1.125 1.25 1.5 2.0
bash scripts/step_sweep_4_28.sh <gpu> <run_dir> "30000 60000 90000"

# MLP 量化扫描（混合位宽）
python scripts/mlp_quant_sweep.py --ckpt <ckpt_90000.pth> --data-dir <4-28> \
  --result-dir runs/mlp_quant_sweep --max-width 1600 --data-factor 1 \
  --no-preload-images \
  --group-bits mlp_complexity:8 mlp_deform:8 mlp_opacity:16 mlp_cov:16 \
               mlp_color:16 mlp_grid:16

# 码流效率与体积分解
python scripts/codec_efficiency.py --ckpt <ckpt> --work-dir runs/codec_efficiency
python scripts/volume_breakdown.py --bitstream-dir <bitstream目录>

# 绘图（5090 上有结果 JSON 时）
python scripts/plot_lambda_rd_4_28.py
python scripts/plot_rd_envelope_4_28.py
python scripts/plot_rd_step_4_28.py
```

### 11.3 画图 AI 参考信息

如果要画框架图，建议至少包含以下层次的节点关系（叙述式）：

- 数据层：COLMAP 场景（images + sparse/0）→ `ColmapDataset`（train/val、
  data_factor、max_width、CPU 缓存）。
- 模型层：`HACPlusModel`（包装官方 `GaussianModel`）↔ `HACCoreView`；
  锚点参数、6 个 MLP、哈希网格 `encoding_xyz`。
- 渲染层：`prefilter_anchors`（深度光栅化）→ `generate_gaussians`（神经高斯
  解码）→ gsplat `rasterization`（packed RGB）。
- 训练层：`trainer.run_training` 主循环，挂 L1/SSIM/scale_reg/rate/I6 四个损失
  支路，`growth` 生长/剪枝，`optimizer` 步进。
- 编解码层：`HACPlusCodec` ↔ `encode_attributes`/`decode_attributes`，
  中间件 GPCC（几何）、哈希上下文、`mlp_grid` 熵参数、`Channel_CTX_fea`
  通道自回归、arithmetic 熵编码；产物 `codec_header.json`、
  `content_aware_q_meta.json`、`feat/scaling/offsets/masks/hash .b`、
  `attributes.pth`、`hac_meta.json`。
- 创新点：I1（calc_context_feat 前置）、I2/I6（Q 调制与监督，围绕
  `mlp_complexity`）、MLP 量化（压缩后处理，替换 bit_mlp 体积行）。
- 工具层：scripts 的 runner/评估/RD/绘图脚本。

---

## 12. 当前状态与下一步

### 12.1 进行中/最新（2026-08-16）

- 已完成：q_scale RD 7 点、step 扫描 30k/60k/90k、λ=0.002 第二个 λ 点
  （28.988/8.3069MB）、120k 训练逐 10k 扫描（**PSNR 峰值 110k**，28.8250dB；
  120k 为 28.8138dB/5.4911MB）、110k + MLP 量化最优操作点
  （28.8235/5.4852MB）、MLP 量化全位宽与逐 MLP 消融（推荐 complexity/deform
  8-bit + 其余 16-bit）、codec efficiency（≈1.0）、KL audit（scaling 有约
  0.49 bit/符号模型冗余）、RD 图 v4（PHG λ 曲线 + dim16/dim32 + 110k 星标 +
  旧 HAC λ 曲线）。
- 数据：4-28 已用于全部实验；1-100、3-07 两个大场景数据集已安排从阿里云盘
  下载并自动解压到 5090 `CT_HAC_v1/data/`（下载/解压进行中），用于后续
  多场景验证。

### 12.2 下一步建议

1. 把 `i6-sens-replace` 上的 deform hidden 自适应修复并入 main，三处
   （GitHub/5090/本机）同步。
2. 用 1-100、3-07 复跑当前最优配置（I2+I6、dim50 h32、110k、MLP 量化
   complexity/deform 8-bit + 其余 16-bit），
   补齐多场景论文对照。
3. 如果需要“质量层级”RD 包络：在高 λ 用 dim16、低 λ 用 dim32/50，画出
   所有曲线左上方的包络。

---

## 附录：脚本与文件索引

### scripts/

| 脚本 | 作用 |
| --- | --- |
| `runner_4_28_90k.sh` | 4-28 90k 训练 runner（GPU/tag/dim/steps/update_until/save/eval/hidden/λ） |
| `step_sweep_4_28.sh` / `step_sweep_wait_4_28.sh` | 轮数扫描（30k/60k/90k...）与等待型 |
| `finish_120k_wait.sh` | 120k 训练等待/收尾 |
| `eval_decoded.py` | 解码 bitstream 后评估 PSNR/SSIM/LPIPS |
| `rd_sweep.py` | 后处理 RD：q_scale_* / mask_keep_ratio |
| `volume_breakdown.py` | bitstream 体积分类 |
| `codec_efficiency.py` | 实际 bits / 交叉熵 |
| `kl_audit.py` | 经验熵 vs 模型交叉熵（KL） |
| `mlp_quant_sweep.py` | MLP 量化扫描（位宽/分组/混合） |
| `sensitivity_gate.py` | I6 相关性 + 离线 RD 上界 |
| `sens_replace_gate.py` | I6 替换方案输入消融（A/B/C） |
| `sensitivity_side_info.py` | I6 侧信息 1/2/3-bit 量化 + RD |
| `sweep_mlp_complexity.py` | complexity MLP 架构扫描（并行） |
| `p0_offline_entropy.py` / `_rawctx.py` / `p0_offline_reverse.py` | P0 离线实验 |
| `plot_lambda_rd_4_28.py` / `plot_rd_envelope_4_28.py` / `plot_rd_step_4_28.py` / `rd_compare_dims.py` | RD/step/包络绘图 |

### 术语表

| 术语 | 含义 |
| --- | --- |
| anchor | 锚点（体素化后的代表点），每个锚点解码 K=10 个神经高斯 |
| Q / Q0 | 量化步长 / 基础量化步长（feat=1.0、scaling=0.001、offsets=0.2） |
| STE | Straight-Through Estimator，训练期量化噪声直通梯度 |
| GPCC | 点云几何压缩标准，PHG 用 `tmc3` 压缩 anchor 整数坐标 |
| Morton 序 | Z 序，codec 固定用它对锚点排序（编码/解码一致） |
| hash-grid context | 哈希网格插值出的上下文特征（`calc_interp_feat`，48 维） |
| mlp_grid | 上下文 → 熵参数（mean/scale/prob）+ 基础 Q 的网络 |
| Channel_CTX_fea | feat 通道自回归熵模型（逐 channel_group 条件解码） |
| bit_mlp | MLP 权重体积（官方口径 32bit/参数） |
| total_MB | 官方体积口径汇总值 |
| BD-rate | 相对基线在同质量下的码率节省百分比 |

GaussianSpa 训练侧实验（阶段 A）完成，用 DB playroom 30k（I2+I6、dim50、λ=0.002、1600 宽、SPA ratio=0.5）。

**结果（compress → decode → eval，29 验证视图）**

| 方案                                    | 训练 anchors | 编码 anchors | total_MB   | PSNR       | SSIM   | LPIPS  |
| --------------------------------------- | ------------ | ------------ | ---------- | ---------- | ------ | ------ |
| 基线（无 SPA）                          | 338,560      | 168,626      | 4.1831     | 30.872     | 0.9134 | 0.2594 |
| MaskTopk-only 0.5                       | 338,560      | 169,280      | 4.1958     | 30.872     | 0.9134 | 0.2594 |
| MaskTopk-only 0.1026（对齐 SPA 锚点数） | 338,560      | 34,736       | 1.4740     | 24.120     | 0.8399 | 0.3714 |
| **SPA-anchor 0.5（训练侧 ADMM）**       | 50,741       | 34,722       | **1.0835** | **29.639** | 0.8948 | 0.3106 |

**结论：阶段 A 通过，SPA 机制被证明必要**

- SPA 相对基线：体积 -74%（4.18→1.08 MB），PSNR -1.23 dB，是一个有效的低码率操作点；
- **同锚点数对照是决定性的**：同样是约 34.7k 编码锚点，SPA 1.08 MB / 29.64 PSNR，编码端 topk 1.47 MB / 24.12 PSNR——**SPA 高 5.52 dB 且体积还小 27%**。说明“剪掉就忘”的 topk 无法替代训练侧 ADMM：预算下继续训练才能让幸存锚点真正适应；
- 额外发现：基线编码端 `mask_anchor` 本来就会滤掉约一半 anchors，所以 MaskTopk-0.5 与基线几乎无差别；SPA 低码率点的最大体积项变成 MLP 权重（32bit 口径 2.78MB 中的 2.57MB），下一步可叠加 MLP 量化。

- 
