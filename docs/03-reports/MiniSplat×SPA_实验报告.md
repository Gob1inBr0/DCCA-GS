# MiniSplat × SPA 实验报告：锚点位置 vs 预算

> 版本：v1.0（2026-08-22）
>
> 定位：回答「SPA（训练侧 ADMM 锚点预算）在高预算档失效，到底是数量不够还是位置不对」。
> 做法：把 Mini-Splatting 的 depth-reinit（深度反投影增密）接入锚点/HAC++ 世界，并把 SPA 预算钉在增密前，
> 从而在**总锚点不变**的前提下只重排「哪些锚点存活」。所有数字均来自 5090
> `runs/` 实际 `metrics.jsonl` + `hac_meta.json`（HAC++ 编码后真实解码评估），未使用推测值。

---

## 0. TL;DR

| 结论 | 证据 |
| --- | --- |
| **「位置 > 预算」成立** | playroom 30k 同预算（SPA ratio 0.85）：MiniSplat+SPA 30.420 dB / 1.905 MB，基线 SPA 30.221 dB / 1.859 MB → **+0.199 dB，体积仅 +2.5%** |
| MiniSplat 是「免费重排」 | 增密只把锚点铺到表面，SPA 预算钉在增密前；训练锚点 112,875→115,170（+2.0%），体积 +2.5% |
| 语义监督是**比特昂贵**的 | 同预算下语义 +0.117 dB（高档），但体积 +18.2%；完整预算曲线上 **BD-PSNR −0.10 dB、BD-rate +8.2%**（同率不占优） |
| 大场景语义无净增益 | 4-28（非 SPA，30k）语义 28.267 vs 基线 28.308 dB（**−0.041 dB**），SSIM/LPIPS 几乎不变 |
| 当前主路径 | **cell2 = I2 + I6 + SPA(0.85) + MiniSplat**（`spa_enabled=True, spa_ratio=0.85, mini_splat_enabled=True`）；语义方向暂停 |

---

## 1. 背景与假设

### 1.1 为什么做这个实验

- SPA（`GaussianSpa` 式 ADMM 稀疏化）在 playroom 30k 给出同锚点下比编码端 top-k 高 5.5 dB 的强结果
  （见 [SPA_阶段A报告](../03-reports/SPA_阶段A报告.md)），但 110k/高预算档增益收敛。
- SPA 的机制是 `get_mask` 的 top-k：**只删不搬**。它无法改变幸存锚点的空间分布——
  重叠（clustered）区域仍重叠，未覆盖（gappy）区域仍缺锚点。
- Mini-Splatting（Fang & Wang, ECCV 2024）的核心观点是 *count is not the bottleneck —
  placement is*：先按表面增密（blur split / depth reinit），再按交集保持/采样做简化（simplify），
  把「位置」重新铺开。本工作把它移植到锚点/HAC++ 世界，只先做最高保真、可在 GPU 上测试的一半：
  **depth reinitialization**。

### 1.2 要验证的命题

1. **H1（位置 > 预算）**：同 SPA 预算下，depth-reinit 重排锚点位置 → 质量显著提升，体积几乎不变。
2. **H2（语义可叠加性）**：MiniSplat 与 DINO 语义监督（T-A2）是否正交，叠加是否仍值得。
3. **H3（语义性价比）**：语义监督是否只是「用更多比特换质量」，而非「免费提升」。

---

## 2. 方法与实现

### 2.1 设计

在生长停止点（`update_until=15000`）：

1. 用 gsplat `render_mode="D"` 渲染 8 个训练相机的深度图；
2. 反投影到世界表面点，按 `voxel_size` 体素去重、取簇中位点 → 候选锚点；
3. 走官方 `cat_tensors_to_optimizer` 增密候选锚点（`mini_splat_max_new=4000`）；
4. **增密前把 SPA 预算钉死**（`spa_final_n = 增密前锚点数`），因此新锚点只参与「谁活下来」的重排，
   不抬高预算；
5. 继续训练到 30k，再 HAC++ 压缩 → 解码 → 评估。

这样隔离了「位置（placement）」与「数量（count）」：如果增益来自多塞锚点，体积会相应变大；
如果来自重排，体积基本不动而质量上升。

### 2.2 实现接入点（后续维护请对齐）

| 层 | 文件 | 关键符号 |
| --- | --- | --- |
| 配置 | `scaffold_gs/config.py` | `mini_splat_enabled / mini_splat_reinit_iter / mini_splat_max_new / mini_splat_views / mini_splat_voxel`；`spa_enabled / spa_ratio` |
| 增密与状态同步 | `hacplus/scene/gaussian_model.py` | `append_depth_anchors()`、`_sync_semantic_state()`、`_prune_anchor_optimizer()`（跳过投影头参数组）|
| 模型入口 | `scaffold_gs/hacpp.py` | `mini_splat_reinit()`、`semantic_supervision()`、`sensitivity_supervision()` |
| 深度反投影 | `scaffold_gs/mini_splat.py` | `render_scene_depth()`、`collect_depth_surface_anchors()`、`_backproject_depth()` |
| 训练循环 | `scaffold_gs/trainer.py` | `iteration == mini_splat_reinit_iter` 触发；15k 语义目标刷新（`semantic_targets.refresh_semantic_targets`） |
| 离线语义 | `scaffold_gs/semantic_targets.py` | 用当前训练锚点投影采样 DINO 特征 → 逐锚点目标（`cov` 语义覆盖掩码，n×1） |

### 2.3 已修复的坑（供复现/扩展参考）

- **semantic_cov 越界（device-side assert）**：cell4 在 19999 步崩溃。根因是
  `append_depth_anchors` 里先手动 append 语义张量（长度 n_before+4000），再调
  `_sync_semantic_state()`，而此时代码还没执行 `cat_tensors_to_optimizer`，
  `get_anchor.shape[0]` 仍是增密前的 n_before → `_sync_semantic_state` 把 semantic_cov
  反向截断回 n_before，锚点却已经 +4000 → 越界。
  **修复：把 `_sync_semantic_state()` 移到 `cat_tensors_to_optimizer(d)` 之后**（commit `77ec99f`）。
- **投影头参数组被当成锚点参数**：`cat_tensors_to_optimizer` 断言「每组一个参数」，投影头组有两个参数
  （weight/bias）。生长/剪枝时需跳过 `semantic_proj_head` 组。
- **语义目标必须用本次训练的锚点刷新**：早期用基线模型导出的目标与 30k 从零训练锚点索引对不上，
  目标为空、监督无效（那次的 +0.045 dB 不能归因）。正确协议是在 15k（生长停止）用当前锚点重建目标
  （覆盖率 100%），20k 起监督生效。

---

## 3. 实验设置（共同协议）

| 项 | 值 |
| --- | --- |
| 场景 | Deep Blending **playroom**（训练/验证 8:1，宽 1600，`test-every 8`） |
| 训练 | 30k，`update_until=15000`，`feat_dim=50`，`mlp_complexity_hidden=32`，`tile_size=32`，`appearance_dim=0` |
| 基础创新 | I2 内容感知量化（on）+ I6 渲染敏感度监督（on） |
| 率失真 | λ=0.004（default，已从 ckpt `optim_config` 核实） |
| SPA | `spa_ratio` ∈ {0.52, 0.85, 0.92, 0.97}，ρ=1e-3，预算锚定 `max(N_ref)` |
| MiniSplat | `reinit_iter=15000, max_new=4000, views=8, voxel=0.0` |
| 语义 | T-A2 投影头，DINOv2，15k 自我锚点目标刷新，20k 起监督（`weight=1e-3`） |
| 评估 | `compress → decode → eval_decoded`，29 个验证视图；`total_MB` = HAC++ 解码必需载荷（含 MLP 权重） |
| 数据来源 | 5090 `/home/fansonglin/data_space/DCCA-GS/runs/*`（`metrics.jsonl` + `bitstreams/hac_meta.json`） |

---

## 4. 结果

### 4.1 playroom 主对照（SPA ratio=0.85，4 格）

| cell | 配置 | 训练锚点 | 编码锚点 | total_MB | 解码 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **cell1** | 基线 SPA（I2+I6+SPA） | 112,875 | 87,449 | 1.8594 | 30.2210 | 0.90678 | 0.28093 |
| **cell2** | **MiniSplat + SPA** | 115,170 | 90,872 | 1.9051 | **30.4202** | 0.90701 | 0.27874 |
| cell3 | 语义 SPA（+DINO T-A2） | 115,410 | 101,206 | 2.1974 | 30.3379 | 0.90743 | 0.27844 |
| cell4b | MiniSplat + 语义 + SPA | 114,554 | 101,706 | 2.2568 | 30.4584 | 0.90751 | 0.27836 |

### 4.2 playroom SPA 预算曲线（baseline vs 语义，HAC++ 解码后）

| ratio | 方案 | 训练锚点 | 编码锚点 | total_MB | 解码 PSNR | SSIM | LPIPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.52 | baseline | 51,735 | 40,897 | 1.0580 | 29.3743 | 0.89415 | 0.31051 |
| 0.52 | semantic | 52,616 | 45,197 | 1.2331 | 29.6152 | 0.89572 | 0.30587 |
| 0.85 | baseline | 112,875 | 87,449 | 1.8594 | 30.2210 | 0.90678 | 0.28093 |
| 0.85 | semantic | 115,410 | 101,206 | 2.1974 | 30.3379 | 0.90743 | 0.27844 |
| 0.92 | baseline | 137,827 | 109,966 | 2.2026 | 30.5782 | 0.90879 | 0.27491 |
| 0.92 | semantic | 146,025 | 131,230 | 2.6672 | 30.5763 | 0.90842 | 0.27548 |
| 0.97 | baseline | 176,402 | 151,222 | 2.8383 | 30.8196 | 0.91039 | 0.26757 |
| 0.97 | semantic | 176,888 | 151,437 | 3.1101 | 30.7967 | 0.91036 | 0.26646 |

**BD 分析**（PCHIP，共享 log-rate 区间 [1.233, 2.838] MB）：

- **BD-PSNR ≈ −0.103 dB**（同体积下，语义平均比基线差）
- **BD-rate ≈ +8.2%**（同等质量下，语义多花约 8.2% 码率）

### 4.3 4-28 大场景 B 组（非 SPA，30k，训练/验证 1050/150）

| run | 配置 | 锚点数 | 训练期验证 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| run428_baseline | I2+I6 | 455,404 | **28.3080** | 0.89269 | 0.27169 |
| run428_semantic | +DINO T-A2（15k 刷新） | 458,015 | 28.2673（**−0.041**） | 0.89270 | 0.27201 |

＞语义目标已在 15k 用本次训练自身 458,015 锚点重建，覆盖率 100%，协议正确；但大场景无净增益。
（该组为训练期验证评估，未做 HAC++ 压缩；体积口径不适用。）

### 4.4 历史非 SPA 语义对照（playroom 1600w，供上下文）

| run | 语义目标 | 编码锚点 | total_MB | 解码 PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `db_playroom_sem_t2a5_30k` | 空目标（≈I2+I6 基线） | 161,134 | 4.1075 | 30.9166 | 0.91224 | 0.25791 |
| `db_playroom_sem_t2a5_30k_v3` | 15k 自我锚点刷新 | 204,035 | 5.2841 | 31.0419 | 0.91396 | 0.25742 |

＞+0.125 dB，但体积 +28.6%、active 锚点 +26.6%——语义监督是**密度耦合**的质量提升。

---

## 5. 分析

### 5.1 同预算 vs 同比特

- **同总锚点预算（SPA ratio 固定）**：cell2 vs cell1 → +0.199 dB、体积 +2.5%、编码锚点 +3.9%；
  语义 cell3 vs cell1 → +0.117 dB，但体积 +18.2%、编码锚点 +15.7%。
- **同比特（BD 视角）**：语义 vs 基线整条预算曲线 → **BD-PSNR ≈ −0.10 dB、BD-rate ≈ +8.2%**。

结论：语义在「同总锚点」下看起来领先，是因为 SPA 只固定总锚点、不固定解码/比特预算；
换算到同体积后语义**不占优**。MiniSplat 则相反：同预算下净挣 0.199 dB，且几乎不涨价。

### 5.2 语义与预算的非单调关系

- 低预算（0.52）：语义 +0.241 dB，体积 +16.6%（值得一试）
- 中高预算（0.85）：+0.117 dB，体积 +18.2%
- 高预算（0.92 / 0.97）：−0.002 / −0.023 dB，体积 +21.1% / +9.6%（**由正转负**）

解释：语义监督让复杂度 MLP 学到「内容相关」的分配，但在锚点已充足的档位，它主要增加
active 锚点与每锚点熵，质量饱和；低档位它反而能定向地把稀缺预算放到重要内容上。

### 5.3 MiniSplat 与语义的叠加

- cell4b vs cell3（同为语义，加 MiniSplat）：+0.120 dB，体积 +2.7% → 重排对语义组同样有效。
- cell4b vs cell2（同为 MiniSplat，加语义）：+0.038 dB，体积 **+18.5%** → 语义在高档位叠加不划算。

### 5.4 大场景

4-28（45.5 万锚点）：语义 −0.041 dB，SSIM/LPIPS 基本不变。与 playroom 不同：锚点基数大、
视觉内容更杂、DINO 目标对「复杂度分配」的指示在超大场景里被稀释；MiniSplat 目前**未在 4-28 验证**，
是需要补的关键实验（见 §7 E4）。

---

## 6. 结论与默认路径

1. **MiniSplat（depth-reinit）+ SPA 是当前最优训练路径**（cell2），已设为默认：
   `spa_enabled=True, spa_ratio=0.85, mini_splat_enabled=True`。
2. **语义先验（T-A2）暂不作为「免费提升」**：同比特有 BD-rate +8.2% 的代价；低预算段仍可能有价值，
   保留在配置中（默认关）。
3. **「位置 > 预算」是本项目可写进论文的核心经验**：锚点压缩的瓶颈是分布，不是单纯数量。

---

## 7. 后续实验设计（按优先级）

> 设计原则：每个实验先给**假说 → 协议 → 判据 → 停止条件**；未过判据就关，避免无限堆叠。
> 共同约束：单 seed / 单场景结论一律先当成「抽样」，多用 2–3 个 seed 复核；体积口径统一为
> HAC++ 解码必需载荷。

### E1（P0）MiniSplat × 预算曲线补齐

- **假说**：depth-reinit 在任意预算档都有效，尤其低档（锚点少时「位置」更重要）。
- **协议**：r ∈ {0.52, 0.85, 0.92, 0.97} × MiniSplat±（0/1），playroom 30k，其余与 §3 相同；
  全部 compress→decode→eval。
- **判据**：与对应 baseline SPA 同档比 PSNR；与整条曲线比 BD-PSNR / BD-rate。
- **停止**：若某档 MiniSplat < baseline（负增益）且只在高档出现 → 记录非单调，不加参数；
  若全档稳定 ≥ +0.05 dB 且体积 < +3% → 转 E4/E7 泛化。
- **同时可做**：cell2 与 cell4b 在低/高档重跑，补充「MiniSplat × 语义」的完整 2×2。

### E2（P1）Mini-Splatting 完整版（未移植的另一半）

- **假说**：blur split（每像素 argmax 贡献区域）+ intersection-preserving simplification 比
  depth-reinit-only 更能提升「位置」质量；depth-reinit 只是次优近似。
- **协议**：packed gsplat 需暴露每像素 contributor 索引（现 renderer 不提供），先做
  `contribution_area` 的 GPU 实现；再实现简化；与 depth-reinit-only、MiniSplat+SPA 三组对照。
- **判据**：完整版相对 depth-reinit-only 的 BD-PSNR ≥ +0.05 dB 才值得复杂化。
- **停止**：若贡献面积计算不可行（packed 光栅化限制）→ 记录为工程限制，不硬做。

### E3（P1）MiniSplat 参数敏感性

- **假说**：`max_new` / `voxel` / `views` / reinit 轮次（1 次 vs 多次）影响「铺开」质量。
- **协议**：单变量扫描（max_new ∈ {2k,4k,8k}；voxel ∈ {0, model_voxel, 2×}；views ∈ {4,8,16}；
  reinit 在 15k 一次 vs 10k/15k 两次），playroom 30k，固定 r=0.85。
- **判据**：选择 PSNR/体积最优组合；报告边界（过大 max_new 是否会破坏 SPA 预算收敛）。
- **停止**：若所有参数都在 ±0.03 dB 内 → 说明实现鲁棒，写「无需调参」结论，只保留 1 组默认。

### E4（P0）MiniSplat × 大场景（4-28）

- **假说**：4-28（45.5 万锚点）目前**没做过 MiniSplat**；若「位置>预算」普适，则应显著正增益；
  若负/零，说明机理与大场景（相机尺度/遮挡/多源）耦合。
- **协议**：4-28 最优配置下 110k × MiniSplat± × SPA（r=0.85）与非 SPA 两档（可复用之前最优配置的数据）；λ ∈ {0.002, 0.004}；训练预算/时间按 5090 空卡排期（更新状态）。评估 1600w、test-every 8。
- **判据**：同 λ/预算下 PSNR ≥ +0.05 dB 且总锚点/体积增量 < +5%；并与 playroom 单点行为对比。
- **停止**：若大场景负增益 → 写「MiniSplat 场景依赖」，不推广；这与语义在大场景的无增益共同
  构成「大场景锚点分配饱和」的证据链。

### E5（P1）MiniSplat 与 SPA 的解耦消融（MiniSplat-only vs SPA-only）

- **假说**：MiniSplat 的增益到底来自「SPA 预算下的重排」，还是单纯「增密」本身？
- **协议**：(a) MiniSplat on / SPA off（允许预算自然增长）；(b) MiniSplat off / SPA on（cell1）；
  (c) MiniSplat on / SPA on（cell2）；(d) 都不开 = 基线。全部 HAC++ 编码+解码。
- **判据**：若 (a) 相对 (d) 的 PSNR 增益 ≈ (c) 相对 (b) 的增益 → 说明是「增密位置」而非 SPA 交互；
  若 (c) 显著高于两者 → 说明需要 SPA 预算 + 重排的组合。
- **停止**：若 (a) 体积大幅上涨而 PSNR 不涨 → 说明没有预算约束的增密只是浪费，直接保留 (c)。

### E6（P1）语义先验重新定位（低预算 / 换目标）

- **假说**：语义只在低预算段有价值（§5.2），或需要一个「与位置/几何更相关」的目标才能成为免费增益。
- **协议**：
  1. 低预算语义（r=0.52 已有）→ 补 MiniSplat+语义低档（E1 顺带），用 BD 判优劣；
  2. 换目标：`mlp_complexity` 直接监督「渲染敏感度 × 语义」混合目标，或用 MiniSplat 派生目标
     （表面覆盖率 / 局部密度）替代 DINO；
  3. 若换目标 r（相关性门）< 0.3 → 关闭，不投入训练。
- **判据**：同体积/同率下 BD-PSNR > 0；或者低预算段 PSNR/体积较 baseline 有 ≥ 2% 优势。
- **停止**：若换目标 r 不过门，或高/低预算均无净增益 → 语义方向正式关闭（与 Stage A 门一致）。

### E7（P1）泛化与统计稳健性

- **假说**：cell2 的 +0.199 dB 在 drjohnson / T&T / Mip360 及多 seed 下仍成立。
- **协议**：drjohnson（对标 playroom）、T&T train/truck、Mip360 garden/flowers/stump；each ×
  seed ∈ {42, 2026} × {cell1, cell2}；30k，r=0.85。
- **判据**：至少 3/4 场景 PSNR 增益 > 0，且跨 seed 标准差 < 增益的 1/2；记录 BD。
- **停止**：若只在 playroom 成立 → 结论降级为「场景依赖」，论文只报 playroom + 说明。

### E8（P2）码流与工程验证

- 对最佳组合补 `codec_roundtrip_diagnostics.json`（bit-exact 全 0 不匹配）确认；
  测量 encode/decode 时间与内存；确认 `total_MB` 口径不因 MiniSplat 增密变化（几何/掩码占比）。
- **判据**：bit-exact 通过；体积口径与官方 HAC++ 可比；无新增侧信息。

### E9（P2）机制验证（为什么位置更值钱）

- 用渲染统计定量比较增密前后：局部锚点密度、均值/最邻近间距、per-pixel contributor 数、
  SPA mask active 率；画出「锚点分布热图 vs 误差热图」。
- **判据**：若增密后的 error hotspot 与锚点稀疏区对应关系改善 → 为「位置>预算」提供直接证据，
  而非仅靠 PSNR 间接推断。

---

## 8. 复现命令（5090）

```bash
# 训练（cell2 为例；baseline 去掉 --cfg.model.mini-splat-enabled）
bash runs/spa_minisplat_launch.sh cell2 0 0 0.85

# 压缩 + 解码评估（评估协议与 §3 一致）
cd PHG && source scripts/env_5090.sh
python train.py compress --cfg.ckpt runs/spa_minisplat_cell2/ckpts/ckpt_30000.pth \
  --cfg.out-dir runs/spa_minisplat_cell2/bitstreams --cfg.codec hac_pp
python scripts/eval_decoded.py --artifact-dir runs/spa_minisplat_cell2/bitstreams \
  --data-dir data/playroom --result-dir runs/spa_minisplat_cell2/decoded_eval \
  --data-factor 1 --max-width 1600 --no-preload-images
```

> 数字已同步至 `../data/experiments.csv`（group `phg_minisplat` / `phg_semantic_stageB`）。

## 9. 相关文档

- [SPA_阶段A报告](SPA_阶段A报告.md)
- [语义先验_阶段A报告](语义先验_阶段A报告.md)
- [语义先验实验设计](../02-design/语义先验实验设计.md)
- [SPA训练侧实验设计](../02-design/SPA训练侧实验设计.md)
- [消融实验汇总](消融实验汇总.md)
- 外部深读笔记（Obsidian）：`2403.14166 - Fang - Mini-Splatting 受限高斯数`
