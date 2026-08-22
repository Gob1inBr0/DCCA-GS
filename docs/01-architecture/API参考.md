# gsplat API 参考文档（中文）

> 本文档根据本仓库源码（`gsplat/` 目录）逐函数整理，覆盖所有公开 API 的
> 签名、参数含义、返回值和典型用法。建议配合 `../04-guides/上手指南.md` 一起阅读。
>
> 版本：本仓库 main 分支（v1.5.3 之后，含 2026 年新增特性）

---

## 目录

1. [包结构与导入方式](#1-包结构与导入方式)
2. [核心渲染 API](#2-核心渲染-api)
   - 2.1 `rasterization()` —— 3DGS 唯一核心入口
   - 2.2 `rasterization_2dgs()` —— 2D 高斯（表面重建）
   - 2.3 `meta` 字典字段说明
   - 2.4 `rasterization_inria_wrapper()`（对比用）
3. [策略 API（稠密化）](#3-策略-apidensification)
   - 3.1 `Strategy` 基类
   - 3.2 `DefaultStrategy`
   - 3.3 `MCMCStrategy`
   - 3.4 底层算子 `gsplat/strategy/ops.py`
4. [损失函数](#4-损失函数)
5. [优化器](#5-优化器)
6. [导出与压缩](#6-导出与压缩)
7. [颜色校正](#7-颜色校正)
8. [底层 CUDA 算子（进阶）](#8-底层-cuda-算子进阶)
9. [分布式训练](#9-分布式训练)
10. [工具函数](#10-工具函数)
11. [新模块：sensors / scene / experimental](#11-新模块sensors--scene--experimental)
12. [能力检测开关](#12-能力检测开关)

---

## 1. 包结构与导入方式

所有公开 API 都可以直接从 `gsplat` 顶层导入：

```python
import gsplat
gsplat.__version__          # 版本号
gsplat.has_3dgs             # 是否编译了 3DGS CUDA 内核
gsplat.has_2dgs             # 是否编译了 2DGS 内核
gsplat.has_3dgut            # 是否编译了 3DGUT（无迹变换）内核
gsplat.has_adam             # 是否编译了 fused Adam 内核
gsplat.has_reloc            # 是否编译了 relocation 内核
gsplat.has_losses           # 是否编译了 fused losses 内核
gsplat.has_camera_wrappers  # 是否编译了相机 wrapper
```

顶层导出清单（`gsplat/__init__.py` 的 `__all__`）分类如下：

| 类别 | 符号 |
|---|---|
| 渲染 | `rasterization`, `rasterization_2dgs`, `rasterization_inria_wrapper`, `rasterization_2dgs_inria_wrapper`, `RenderMode`, `RasterizeMode`, `RendererConfig`, `RendererConfig_MixedBatch`, `RendererConfig_ParallelBatch` |
| 策略 | `Strategy`, `DefaultStrategy`, `MCMCStrategy` |
| 损失 | `l1_loss`, `mse_loss`, `ssim_loss`, `torch_ssim_loss`, `create_ssim_window`, `depth_l1_loss`, `gaussian_density_reg`, `gaussian_scale_reg`, `gaussian_z_scale_reg`, `opacity_reg_loss`, `scale_reg_loss`, `out_of_bound_loss`, `total_variation_loss`, `lidar_distance_loss`, `lidar_intensity_loss`, `lidar_raydrop_loss`, `lidar_background_loss`, `FusedGaussianLosses` |
| 优化器 | `SelectiveAdam` |
| 导出/压缩 | `export_splats`, `PngCompression` |
| 颜色校正 | `color_correct_affine`, `color_correct_quadratic` |
| 低层算子 | `proj`, `fully_fused_projection`, `fully_fused_projection_2dgs`, `fully_fused_projection_with_ut`, `isect_tiles`, `isect_tiles_lidar`, `isect_tiles_sparse`, `isect_offset_encode`, `build_sparse_tile_layout`, `quat_scale_to_covar_preci`, `rasterize_to_pixels`, `rasterize_to_pixels_2dgs`, `rasterize_to_pixels_sparse`, `rasterize_to_pixels_eval3d`, `rasterize_num_contributing_gaussians*`, `rasterize_contributing_gaussian_ids*`, `rasterize_top_contributing_gaussian_ids*`, `spherical_harmonics`, `world_to_cam`, `accumulate`, `accumulate_2dgs`, `rasterize_to_indices_in_range*`, `CameraModel`, `RollingShutterType`, 各种 Lidar 参数类 |
| 其他 | `compute_lidar_angles_to_columns_map`, `compute_lidar_tiling`, `LidarTiling`, `SpinningDirection` |

---

## 2. 核心渲染 API

### 2.1 `rasterization()` —— 3DGS 唯一核心入口

**位置**：`gsplat/rendering.py:231`

**一句话**：把 N 个 3D 高斯光栅化到 C 张图像上，支持批量、SH 颜色、深度、
抗锯齿、畸变相机、卷帘快门、分布式渲染。

```python
from gsplat import rasterization

render_colors, render_alphas, meta = rasterization(
    means, quats, scales, opacities, colors,
    viewmats, Ks, width, height,
    **可选参数,
)
```

#### 必选参数（形状约定）

| 参数 | 形状 | 说明 |
|---|---|---|
| `means` | `[..., N, 3]` | 高斯 3D 中心位置 |
| `quats` | `[..., N, 4]` | 旋转四元数，**wxyz 约定**，不要求归一化 |
| `scales` | `[..., N, 3]` | 高斯尺度 |
| `opacities` | `[..., N]` | 不透明度 |
| `colors` | 两种模式（见下） | 颜色 |
| `viewmats` | `[..., C, 4, 4]` | 世界→相机矩阵 |
| `Ks` | `[..., C, 3, 3]` | 相机内参（**不可微**，警告） |
| `width` / `height` | int | 图像尺寸（LiDAR 时被 lidar_coeffs 覆盖） |

`colors` 的两种模式：

```python
# 模式 A：后激活颜色（sh_degree=None），直接 alpha 混合
colors = torch.rand(B, C, N, D)      # [..., (C,) N, D]
# 模式 B：SH 系数（sh_degree=3 等），按视线方向求值（在 batch/camera 间共享）
colors = torch.rand(N, K, D)         # [N, K, D]，要求 (sh_degree+1)^2 <= K
```

#### 常用可选参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `near_plane` / `far_plane` | 0.01 / 1e10 | 裁剪平面 |
| `radius_clip` | 0.0 | 2D 半径 ≤ 该值（像素）的高斯被跳过，加速大场景 |
| `eps2d` | 0.3 | 加到 2D 协方差特征值的 epsilon（0.3 ≈ 最小 3 像素） |
| `sh_degree` | None | 使用的 SH 阶数，设了就按 SH 模式解读 colors |
| `packed` | True | packed 模式（稀疏张量，省显存）；False 更快更耗内存 |
| `tile_size` | None（自动） | tile 大小，16；3DGUT 路径可 8/16 |
| `backgrounds` | None | 背景色 `[..., C, D]` |
| `render_mode` | "RGB" | `"RGB"`, `"d"`, `"Ed"`, `"D"`, `"ED"`, `"RGB-d"`, `"RGB-Ed"`, `"RGB+D"`, `"RGB+ED"` |
| `sparse_grad` | False | means/quats/scales 梯度存 COO 稀疏格式（省显存，需 packed=True） |
| `absgrad` | False | 计算 2D 均值绝对梯度（AbsGS），配合策略 `absgrad=True` |
| `rasterize_mode` | "classic" | `"classic"` 或 `"antialiased"`（Mip-Splatting） |
| `channel_chunk` | 32 | 一次渲染的通道数，超过则分块 |
| `distributed` | False | 多 GPU 分布式渲染（需 torch.distributed 已初始化） |
| `camera_model` | "pinhole" | `"pinhole"`, `"ortho"`, `"fisheye"`, `"ftheta"` |
| `covars` | None | 直接给协方差 `[..., N, 3, 3]`，给了则忽略 quats/scales |
| `with_ut` | False | 无迹变换投影（3DGUT） |
| `with_eval3d` | False | 3D 世界空间求值（3DGUT），需 `renderer_config` |
| `global_z_order` | True | True 按 z 深度排序，False 按欧氏距离 |
| `radial_coeffs` / `tangential_coeffs` / `thin_prism_coeffs` | None | OpenCV 畸变系数（pinhole: 6/2/4；fisheye: 4） |
| `rolling_shutter` | GLOBAL | 卷帘快门类型，配 `viewmats_rs` 使用 |
| `rays` | None | 每条光线 `[..., C, H, W, 6]`（ox,oy,oz, dx*spread,...），3DGUT 次级光线 |
| `extra_signals` | None | 额外信号通道 `[..., (C,) N, E]`，渲染结果在 `meta["render_extra_signals"]` |
| `renderer_config` | None | `RendererConfig_MixedBatch()` 或 `RendererConfig_ParallelBatch()` |

#### 返回值

```python
render_colors  # [..., C, height, width, X]  X = D（RGB）、1（深度）或 D+1（RGB+深度）
render_alphas  # [..., C, height, width, 1]
meta           # dict，中间结果（见 2.3 节）
```

#### 最小示例

```python
import torch
from gsplat import rasterization

device = "cuda"
N, C, H, W = 100, 1, 200, 300
means   = torch.randn(N, 3, device=device)
quats   = torch.randn(N, 4, device=device)
scales  = torch.rand(N, 3, device=device) * 0.1
opacities = torch.rand(N, device=device)
colors  = torch.rand(N, 3, device=device)
viewmats = torch.eye(4, device=device)[None]                     # [1,4,4]
Ks       = torch.tensor([[300.,0,150.],[0.,300.,100.],[0.,0.,1.]], device=device)[None]  # [1,3,3]

render_colors, render_alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, W, H)
# render_colors: [1, 200, 300, 3]
```

---

### 2.2 `rasterization_2dgs()` —— 2D 高斯（表面重建）

**位置**：`gsplat/rendering.py:1319`

用于 2D Gaussian Splatting（2DGS），把 3D 场景表示为 2D 圆盘（surfel），
适合表面重建（几何更锐利）。参数与 `rasterization()` 基本一致，额外有：

| 参数 | 默认 | 说明 |
|---|---|---|
| `distloss` | False | 是否计算畸变正则（几何细节更好） |
| `depth_mode` | "expected" | `"expected"` 或 `"median"` 深度 |

**注意**：返回 **7 个值**（比 3DGS 多法线、畸变、中值深度）：

```python
(render_colors, render_alphas, render_normals, surf_normals,
 render_distort, render_median, meta) = rasterization_2dgs(...)

# render_normals   [..., C, H, W, 3]  逐像素累积法线
# surf_normals     [..., C, H, W, 3]  由深度反推的表面法线
# render_distort   [..., C, H, W, 1]  畸变图（L1 版本）
# render_median    [..., C, H, W, 1]  中值深度
```

示例见 `examples/simple_trainer_2dgs.py`。

---

### 2.3 `meta` 字典字段说明

`rasterization()` 返回的 `meta` 是训练策略的数据来源，常用字段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `means2d` | `[nnz, 2]`（packed） | 投影后 2D 均值，**含 `.grad` 和 `.absgrad`**，稠密化靠它 |
| `radii` | `[nnz]` | 高斯像素半径（2DGS 为 `[nnz, 2]`） |
| `gaussian_ids` | `[nnz]` | packed 模式下被激活的高斯索引 |
| `camera_ids` | `[nnz]` | 每个激活高斯对应的相机索引 |
| `depths` | `[nnz]` | 深度 |
| `conics` | `[nnz, 3]` | 投影协方差的逆（上三角） |
| `opacities` | `[nnz]` | 逐高斯不透明度 |
| `isect_offsets` | `[tile_h, tile_w]` | tile 相交偏移 |
| `isect_ids` / `flatten_ids` / `tiles_per_gauss` | — | 排序中间结果 |
| `width` / `height` / `tile_size` | int | 图像与 tile 尺寸 |
| `render_extra_signals` | `[..., C, H, W, E]` | 仅当传了 `extra_signals` 时存在 |

2DGS 额外字段：`ray_transforms`, `normals`, `n_cameras`, `render_distort`,
`gradient_2dgs`（2DGS 稠密化用的梯度，配合策略 `key_for_gradient="gradient_2dgs"`）。

---

### 2.4 `rasterization_inria_wrapper()`（对比用）

包装 Inria 原版 `diff-gaussian-rasterization` 后端，仅用于性能/精度对比。
需要额外安装原版库（有独立 LICENSE），`eps2d` 被硬编码为 0.3，只返回渲染图。

---

## 3. 策略 API（Densification）

### 3.1 `Strategy` 基类

**位置**：`gsplat/strategy/base.py`

```python
from gsplat import Strategy

@dataclass
class Strategy:
    def check_sanity(self, params, optimizers): ...   # 校验参数/优化器键一致
    def step_pre_backward(self, *args, **kwargs): ...  # loss.backward() 之前回调
    def step_post_backward(self, *args, **kwargs): ... # loss.backward() 之后回调
    def initialize_state(self, ...): ...               # 建运行状态（DefaultStrategy/MCMCStrategy 才有）
```

**自定义策略 = 继承此类**，实现三个 `step_*` 回调即可接入训练循环。

### 3.2 `DefaultStrategy`

**位置**：`gsplat/strategy/default.py:32`

遵循原版 3DGS 论文的自适应稠密化：周期性 **复制**（小高斯补细节）、**分裂**
（大高斯补覆盖）、**剪枝**（低不透明度）、**重置不透明度**。

```python
from gsplat import DefaultStrategy

strategy = DefaultStrategy(
    prune_opa=0.005,          # 不透明度 < 该值 → 剪枝
    grow_grad2d=0.0002,       # 2D 梯度 > 该值 → 复制/分裂
    grow_scale3d=0.01,        # 3D 尺度 < 该值 → 复制；> 该值 → 分裂
    grow_scale2d=0.05,        # 2D 尺度 > 该值 → 分裂
    prune_scale3d=0.1,        # 3D 尺度 > 该值 → 剪枝
    prune_scale2d=0.15,       # 2D 尺度 > 该值 → 剪枝
    refine_start_iter=500,    # 该步之后开始稠密化
    refine_stop_iter=15_000,  # 该步之后停止稠密化
    refine_every=100,         # 每多少步稠密化一次
    reset_every=3000,         # 每多少步重置不透明度
    absgrad=False,            # AbsGS 绝对梯度（需 rasterization(absgrad=True)）
    revised_opacity=False,    # 修正不透明度启发式（实验性）
    verbose=False,
    key_for_gradient="means2d",  # 2DGS 用 "gradient_2dgs"
)

# 三个核心方法
state = strategy.initialize_state(scene_scale=1.0)   # 建运行状态
strategy.check_sanity(params, optimizers)            # 可选但推荐

for step in range(max_steps):
    render, alpha, info = rasterization(...)
    strategy.step_pre_backward(params, optimizers, state, step, info)  # 保留 2D 均值梯度
    loss.backward()
    strategy.step_post_backward(params, optimizers, state, step, info) # 复制/分裂/剪枝/重置
    for opt in optimizers.values():
        opt.step()
```

**参数约定**：`params` 必须是 `Dict[str, nn.Parameter]`，且至少含
`{"means", "scales", "quats", "opacities"}`；`optimizers` 与 `params` 键一一对应，
每个优化器恰好一个 param_group。

### 3.3 `MCMCStrategy`

**位置**：`gsplat/strategy/mcmc.py:40`

遵循 MCMC 论文（`3D Gaussian Splatting as Markov Chain Monte Carlo`）：
**传送**（低不透明度高斯搬到高不透明度处）+ **按不透明度分布采样新增** + **位置扰动**。

```python
from gsplat import MCMCStrategy

strategy = MCMCStrategy(
    cap_max=1_000_000,        # 高斯数量上限
    noise_lr=5e5,             # 采样噪声学习率
    refine_start_iter=500,
    refine_stop_iter=25_000,
    refine_every=100,
    min_opacity=0.005,        # 低于该值 → 剪枝
)

state = strategy.initialize_state()
# 注意：MCMC 没有 step_pre_backward（被注释掉）
loss.backward()
strategy.step_post_backward(params, optimizers, state, step, info, lr=1e-3)
```

训练入口：`python simple_trainer.py mcmc --data_dir ...`。

### 3.4 底层算子 `gsplat/strategy/ops.py`

策略内部使用的张量操作，也可单独调用（都返回更新后的 params 和 optimizers）：

```python
from gsplat.strategy import ops

ops.duplicate(params, optimizers, state, mask)          # 复制选中的高斯
ops.split(params, optimizers, state, mask)              # 分裂选中的高斯
ops.remove(params, optimizers, state, mask)             # 删除选中的高斯
ops.reset_opa(params, optimizers, state, value)         # 把不透明度重置为 value
ops.relocate(params, optimizers, state, mask)           # 传送（MCMC）
ops.sample_add(params, optimizers, state, n, ...)       # 按分布采样新增（MCMC）
ops.inject_noise_to_position(params, state, ...)        # 位置注入噪声（MCMC，CUDA 融合）
```

每个 op 都会同步处理对应优化器中的一阶/二阶动量，保证状态一致。

---

## 4. 损失函数

**位置**：`gsplat/losses.py`，全部支持反向传播。

### 4.1 图像损失（训练主损失）

| 函数 | 签名 | 说明 |
|---|---|---|
| `l1_loss(pred, target)` | 任意形状 | 逐元素 L1，返回与输入同形状（不 reduce） |
| `mse_loss(pred, target)` | 任意形状 | 逐元素 MSE |
| `ssim_loss(img1, img2, window_size=11)` | `(B,C,H,W)`，值域 [0,1] | `1 - SSIM`，标量；有 CUDA 时自动用 fused_ssim |
| `torch_ssim_loss(img1, img2, window, window_size, channel)` | — | 参考实现（Wang et al. 2004），返回 SSIM 图 |
| `create_ssim_window(window_size, channel, device)` | — | 预计算高斯窗口 `(C,1,W,W)` |
| `depth_l1_loss(disp, disp_gt, scene_scale=1.0)` | — | 视差空间（逆深度）L1，更偏重近处 |
| `huber_loss(pred, target, delta=1.0)` | 任意形状 | 逐元素 Huber |
| `smooth_l1_loss` / `bce_loss` / `bce_with_logits_loss` / `cross_entropy_loss` | — | 常用损失直通 PyTorch |

### 4.2 正则化损失

| 函数 | 签名 | 说明 |
|---|---|---|
| `opacity_reg_loss(opacities)` | 原始 logits | `sigmoid(opacities).mean()`，抑制透明飘絮 |
| `scale_reg_loss(log_scales)` | 原始 log 尺度 | `exp(log_scales).mean()`，惩罚大高斯 |
| `gaussian_scale_reg(scales, visibility=None)` | 后激活尺度 `[N,3]` | 逐元素绝对值，可用 visibility 加权 |
| `gaussian_density_reg(densities, visibility=None)` | 后激活不透明度 `[N]` | 惩罚大不透明度 |
| `gaussian_z_scale_reg(z_scales, threshold)` | 后激活 `[N]` | 惩罚 z 尺度超过阈值（道路层约束） |
| `out_of_bound_loss(positions, cuboid_dims)` | `[N,3]` / `[N,3]` | `relu(|pos| - dim/2)`，惩罚出边界高斯 |
| `total_variation_loss(x)` | 任意形状 | TV 平滑损失 |
| `normal_cosine_loss(pred_normal, gt_normal)` | — | 法线余弦损失 |
| `weights_reg(weights_list, dim=1)` | — | 权重分布正则 |

> 前四个正则损失（gaussian_scale/density/z_scale/out_of_bound）有 CUDA 融合版本：
> `FusedGaussianLosses(z_scale_threshold=0.0)`（`gsplat/losses_fused.py`），
> 一次性算完四个损失，只支持 fp32/fp64。

### 4.3 LiDAR 损失（3DGS+LiDAR，2026 新增）

| 函数 | 说明 |
|---|---|
| `lidar_distance_loss(pred, target, normalize=True)` | 距离损失（近处权重更高） |
| `lidar_intensity_loss(pred, target, mode="l1")` | 强度损失 |
| `lidar_raydrop_loss(pred, target)` | 射线丢弃（raydrop）损失 |
| `lidar_background_loss(...)` | 背景损失 |

### 4.4 典型用法（抄作业）

```python
from gsplat.losses import l1_loss, ssim_loss
from gsplat.losses_fused import FusedGaussianLosses

loss = l1_loss(render, gt_image).mean() + 0.2 * ssim_loss(render, gt_image)
# render: [C,H,W,3]（注意 gsplat 是 HWC 布局，需要 permute 到 CHW 再喂 ssim_loss）
```

---

## 5. 优化器

### `SelectiveAdam`

**位置**：`gsplat/optimizers/selective_adam.py:21`

稀疏版 Adam：只更新 `visibility` 掩码选中的参数子集，CUDA 融合单内核。
来自 Taming3DGS 论文。

```python
from gsplat import SelectiveAdam

optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
loss.backward()
optimizer.step(visibility=mask)   # mask: [N] 的 0/1 张量，1 = 该高斯被更新
```

> 注意：`step()` 必须显式传入 `visibility`；每个 param_group 只能有一个参数。

---

## 6. 导出与压缩

### `export_splats()` —— 导出模型

**位置**：`gsplat/exporter.py:588`

```python
from gsplat import export_splats

data = export_splats(
    means,       # [N, 3]
    scales,      # [N, 3]
    quats,       # [N, 4]
    opacities,   # [N]
    sh0,         # [N, 1, 3]  SH 第 0 阶（RGB 基色）
    shN,         # [N, K, 3]  其余 SH 阶
    format="ply",              # "ply" | "splat" | "ply_compressed"
    save_to="model.ply",       # 可选，同时写文件
)  # -> bytes
```

- `ply`：标准 PLY，大多数查看器支持
- `splat`：antimatter15 查看器的自定义格式
- `ply_compressed`：Supersplat 查看器的压缩格式

自动过滤 NaN/Inf 的高斯。另有 `load_ply_to_splats(path)` 可读回 PLY。

### `PngCompression` —— 模型压缩

**位置**：`gsplat/compression/png_compression.py:31`

把高斯参数量化排序后压成 PNG（SH 用 K-means 聚类），压缩比极高。

```python
from gsplat import PngCompression

compression = PngCompression(use_sort=True, verbose=True)
compression.compress("compress_dir", splats)          # splats: dict，含
                                                      # "means"/"scales"/"quats"
                                                      # "opacities"/"sh0"/"shN"
                                                      #（后激活值）
restored = compression.decompress("compress_dir")     # -> dict
```

> 需要额外安装 `imageio`、`plas`、`torchpq`。要求 splats 数量为平方数
> （否则会丢弃少量最低不透明度的高斯）。

### `save_ply()`

**位置**：`gsplat/utils.py:26`。训练循环里常用的简易保存：

```python
from gsplat.utils import save_ply
save_ply(splats, "results/model.ply", colors=colors)  # splats: ParameterDict
```

---

## 7. 颜色校正

**位置**：`gsplat/color_correct.py`

```python
from gsplat import color_correct_affine, color_correct_quadratic

# 仿射校正：逐通道 a*ref + b = img 的逆映射
corrected = color_correct_affine(img, ref)          # [..., C]（通道在最后）

# 二次校正：迭代最小二乘，处理非线性变换和过曝
corrected = color_correct_quadratic(img, ref, num_iters=5, eps=0.5/255)
```

---

## 8. 底层 CUDA 算子（进阶）

> 这些是 `rasterization()` 内部的四个流水线阶段，一般不需要直接用，
> 但理解它们 = 理解 gsplat 的整个设计。全部位于 `gsplat/cuda/_wrapper.py`。

### 8.1 光栅化流水线四阶段

```
① fully_fused_projection   高斯投影（协方差计算 + 世界→相机 + 透视投影）
② isect_tiles              计算高斯与哪些 tile 相交 + 按深度排序
③ isect_offset_encode      编码偏移量
④ rasterize_to_pixels      逐像素 alpha 混合
```

### 8.2 各算子签名

| 函数 | 签名要点 | 返回 |
|---|---|---|
| `fully_fused_projection(means, covars\|quats+scales, viewmats, Ks, width, height, eps2d, near_plane, far_plane, radius_clip, packed, sparse_grad, calc_compensations, camera_model, opacities)` | 融合投影 | packed 时返回 `(batch_ids, camera_ids, gaussian_ids, indptr, radii, means2d, depths, conics, compensations)`；非 packed 返回 `(radii, means2d, depths, conics, compensations)` |
| `isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height, sort, segmented, packed, n_images, image_ids, gaussian_ids, conics, opacities)` | 相交测试；给 conics+opacities 时启用 AccuTile（椭圆相交） | `(tiles_per_gauss, isect_ids, flatten_ids)` |
| `isect_offset_encode(...)` | 编码 tile 偏移 | `isect_offsets` |
| `rasterize_to_pixels(means2d, conics, colors, opacities, image_width, image_height, tile_size, isect_offsets, flatten_ids, backgrounds, masks, packed, absgrad)` | 逐像素合成 | `(render_colors, render_alphas)` |
| `proj(means, covars, Ks, width, height, camera_model)` | 纯投影（不进相机系） | `(means2d, covars2d)` |
| `quat_scale_to_covar_preci(quats, scales)` | 四元数+尺度 → 协方差矩阵 | `(covars, precisions)` |
| `world_to_cam(means, covars, viewmats)` | 世界→相机（已弃用，PyTorch 实现） | `(means_cam, covars_cam)` |
| `spherical_harmonics(degrees_to_use, dirs, coeffs, masks)` | SH 求值 | `[..., N, D]` |
| `accumulate(colors, alphas, transmittance, final_T)` | packed 模式下的累积缓冲 | `(render_colors, render_alphas)` |
| `rasterize_num_contributing_gaussians(...)` | 每像素贡献高斯数量 | — |
| `rasterize_contributing_gaussian_ids(...)` | 每像素贡献高斯 ID | — |
| `rasterize_top_contributing_gaussian_ids(...)` | 每像素前 K 个贡献高斯 ID | — |

### 8.3 其他常量/枚举

```python
from gsplat import CameraModel, RollingShutterType
CameraModel              # "pinhole" | "ortho" | "fisheye" | "ftheta"（str 字面量）
RollingShutterType.GLOBAL / RollingShutterType.ROLLING  # 卷帘快门类型
```

---

## 9. 分布式训练

### `gsplat.distributed.cli`

把单卡训练函数一键改造成多卡启动器：

```python
from gsplat.distributed import cli

def train_fn(local_rank: int, world_rank: int, world_size: int, cfg):
    ...  # 原单卡逻辑，注意 distributed=True 传给 rasterization

if __name__ == "__main__":
    cli(train_fn, (cfg,), verbose=True)
```

配套的低层集合通信：`all_gather_int32`、`all_to_all_int32`、
`all_gather_tensor_list`、`all_to_all_tensor_list`（见 `gsplat/distributed.py`）。

多卡启动方式：`simple_trainer.py` 的 `main()` 中示范了
`torch.multiprocessing.spawn` 用法。

---

## 10. 工具函数

**位置**：`gsplat/utils.py`、`gsplat/init_utils.py`、`gsplat/relocation.py`

| 函数 | 说明 |
|---|---|
| `save_ply(splats, dir, colors=None)` | 保存 ParameterDict 为 PLY |
| `log_transform(x)` / `inverse_log_transform(y)` | 尺度参数激活（exp）及其逆（log） |
| `normalized_quat_to_rotmat(quat)` | 归一化四元数 → 旋转矩阵 |
| `depth_to_points(depth, Ks, viewmats)` | 深度图 → 3D 点云 |
| `depth_to_normal(depth, viewmats, Ks)` | 深度图 → 法线图 |
| `get_projection_matrix(znear, zfar, fovX, fovY, device)` | 构造投影矩阵 |
| `multi_frame_depth_unprojection(depths, Ks, c2ws, ...)` | 多帧深度反投影（高斯初始化用） |
| `knn_scale_init(means, K=4)` | 用 KNN 距离初始化尺度（`init_utils.py`） |
| `compute_relocation(opacities, scales, mask, ...)` | 复制/分裂后重新分配不透明度/尺度（`relocation.py`） |

---

## 11. 新模块：sensors / scene / experimental

> 这些是 2025-2026 年重构出的新架构（详见 `../01-architecture/modules-design.md`），
> 目前**正在逐步迁移**，日常使用仍以 `rasterization()` 为主。

### 11.1 `gsplat.sensors` —— 相机/传感器库

`gsplat/sensors/functional/cameras.py` 提供射线与投影的纯函数：

```python
from gsplat.sensors.functional import (
    camera_rays_to_image_points,        # 相机射线 → 像素坐标
    image_points_to_camera_rays,        # 像素 → 相机射线
    project_world_points_mean_pose,     # 世界点 → 像素（静态位姿）
    project_world_points_shutter_pose,  # 世界点 → 像素（卷帘快门）
    image_points_to_world_rays_static_pose,
    image_points_to_world_rays_shutter_pose,
    pixel_grid_to_world_rays_shutter_pose,
    generate_image_points,              # 生成像素网格
)
```

`gsplat/sensors/functional/lidars.py`（旋转激光雷达）：

```python
from gsplat.sensors.functional import (
    generate_spinning_lidar_rays,       # 生成激光雷达射线
    inverse_project_spinning_lidar,     # 反投影
    sensor_rays_to_sensor_angles, sensor_angles_to_sensor_rays, ...
)
```

模型侧：`gsplat/sensors/models` 提供有状态（nn.Module）的相机模型基类
`CameraModel`（pinhole / fisheye / ftheta / LiDAR），支持 TorchScript 部署。

### 11.2 `gsplat.scene` —— 场景对象

```python
from gsplat.scene import Scene, GaussianScene, GaussianInferenceScene
```

- `Scene`：场景抽象基类（ABC）
- `GaussianScene`：常规训练场景（参数持有者，配合 strategy/optimizer 使用）
- `GaussianInferenceScene`：**推理专用**场景，把高斯打包成 fp16 紧凑布局，
  供 HiGS 推理光栅化器使用

### 11.3 `gsplat.experimental` —— HiGS 推理渲染（2026 新增）

低延迟推理路径（宏 tile 融合 + fp16 打包），只做前向：

```python
from gsplat.experimental import render_scene, GaussianInferenceScene

scene = GaussianInferenceScene(...)          # 从训练好的高斯构建
ret = render_scene(scene, out=None, **request)
# ret.render_image / ret.render_alpha / ret.metadata
```

`examples/simple_viewer.py --use_gaussian_render_inference_scene` 可体验。

---

## 12. 能力检测开关

```python
gsplat.has_3dgs / has_2dgs / has_3dgut / has_adam / has_reloc / has_losses / has_camera_wrappers
```

这些是布尔常量，反映当前安装是否编译了对应 CUDA 内核（JIT 模式下首次调用才编译）。
写跨环境代码时用它们做特性检测：

```python
if gsplat.has_3dgs:
    ...  # 使用 CUDA 光栅化
else:
    ...  # CPU 回退（如 accumulate 的 torch 实现）
```

---

## 附：与 examples 的对照表

| 想学什么 | 看哪里 |
|---|---|
| 最小渲染流程 | `examples/image_fitting.py` |
| 完整训练 + 策略接入 | `examples/simple_trainer.py` 的 `Runner.train()` |
| 2DGS 训练 | `examples/simple_trainer_2dgs.py` |
| 查看器 | `examples/simple_viewer.py` |
| 动态场景（G-SHARP） | `examples/dynamic_surgical_trainer.py` + `examples/AV_TRAINER.md` |
| 汽车场景（3DGUT + LiDAR） | `examples/av_trainer.py` |
| 批量/多场景 | `../01-architecture/gsplat批量渲染说明.md` |
| 性能基准 | `examples/benchmarks/`（basic.sh、mcmc.sh 等） |
