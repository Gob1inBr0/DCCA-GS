# MiniSplat 完整版：blur-split + contribution-area 验证设计

> 分支：`codex/minisplat-full`（从 `main 48bb98a` 建出）
> 状态：实现完成，完整版单场景冒烟通过，LYH 泛化矩阵已启动

## 1. 目标

当前主路径只实现了 Mini-Splatting 的 depth-reinit 一半：

- playroom E1 四点预算曲线 BD-PSNR **+0.124 dB**、BD-rate **−8.6%**；
- T&T train 正、T&T truck / drjohnson / 4-28 负。

本分支加入论文的另一半机制，验证「完整 Mini-Splatting」能否把当前“场景依赖”的结论变成泛化正增益：

1. **blur split**：渲染逐像素最大贡献面积，`S_i > θ_blur·H·W` 的锚点分裂；
2. **intersection-preserving simplification**：用贡献面积对锚点做 top-k 保留；
3. **sampling**：本轮暂以 depth-reinit 的体素去重替代论文世界空间采样，作为工程近似；
4. 叠加 SPA 固定预算，最终按同协议评估 PSNR/SSIM/LPIPS、体积和锚点数。

## 2. 实现入口

| 层 | 文件 | 内容 |
| --- | --- | --- |
| 配置 | `scaffold_gs/config.py` | `mini_splat_full`、`mini_splat_blur_threshold`、`mini_splat_importance_weight` |
| 高斯映射 | `scaffold_gs/model.py` | `NeuralGaussians.gaussian_anchor_indices` |
| 生成路径 | `scaffold_gs/hacpp.py` | 输出逐高斯 anchor 映射；`mini_splat_reinit` 增加 full 分支 |
| 核心算法 | `scaffold_gs/mini_splat.py` | 深度反投影、逐像素 argmax 贡献面积、blur 分裂候选 |
| SPA 融合 | `hacplus/scene/gaussian_model.py` | `mini_splat_importance` 状态 + 绝对预算 + 贡献面积加权 score |

关键实现点：gsplat 的 `meta["gaussian_ids"]` 是“投影后活跃高斯 → 原生成高斯索引”的映射，贡献面积必须先经过它再换算到 anchor；否则会出现 `index > num_anchors` 或 `src` 长度不匹配。

## 3. 验证协议

共同设置（与 cell2 对齐）：

- 30k，`update_until=15000`，λ=0.004；
- I2+I6+SPA(0.85)+MiniSplat，`spa_final_n ≈ 0.85·N_before`；
- `mini_splat_views=8`（完整版贡献面积计算的开销控制，后续可再验证 16 views）；
- `max_new=4000`，`voxel=0`，`blur_threshold=0.01`；
- 训练 1600w，`test-every=8`；
- 压缩 → 解码 → `eval_decoded`，记录 `total_MB` 和解码 PSNR/SSIM/LPIPS。

## 4. 当前启动矩阵（LYH）

```
GPU0 drjohnson      GPU1 tandt_train
GPU2 tandt_truck    GPU3 mip_garden
GPU4 playroom       GPU5 mip_bicycle
GPU6 mip_stump      GPU7 保留（后续 4-28/多 seed）
```

所有 run 位于验证 worktree：

```
/home/T0ng/DCCA-GS-minifull
```

结果目录：

```
/home/T0ng/runs/lyh_full_*
```

## 5. 判定标准

- 若 ≥4/7 场景相对 depth-reinit 版 PSNR 增益 > 0，且 4-28 不显著为负，则接受“完整版泛化有效”；
- 若仍只 playroom/T&T train 为正，则判定完整版仍为场景依赖，写入负结论并关闭推广；
- 完整版相对 depth-reinit-only 的 BD-PSNR ≥ +0.05 dB 才值得作为论文主路径。

## 6. 后续

- 等当前矩阵跑完后收集 `decoded_eval/metrics.jsonl`、`bitstreams/hac_meta.json`；
- 补齐 seed=2026 与 4-28 110k 对照；
- 回写 `experiments.csv` 和 MiniSplat 报告，再决定是否合入 main。
