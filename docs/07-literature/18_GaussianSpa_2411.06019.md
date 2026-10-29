# 《GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting》精读笔记

- arXiv：https://arxiv.org/abs/2411.06019
- 发表：CVPR 2025（txt 为 arXiv v3，2025-04-10，正文未出现会议标注）

> 一句话定位：把"剪掉多少高斯"从手工准则改成训练目标里的硬约束（高斯数 ≤ κ），用 ADMM 三步交替（梯度优化 / TopK 硬投影 / 乘子更新）在训练中渐进施加稀疏，让信息从被剪高斯平滑转移给幸存高斯——同更少的高斯数下渲染质量反超原版 3DGS。

## 问题与支撑观察

第 1 节，两层：(1) 3DGS 存储大头是高斯本身，"the sheer volume of Gaussians leads to memory usage that exceeds the capacity of typical hardware"；(2) 现有剪枝都靠手工准则（opacity [22,50]、importance score/hit count [16,17,37]、dominant primitives [28]、binary mask [26,51]），"such single-perspective heuristic criteria may lack robustness"；更要命的是"sudden one-shot removal may cause permanent loss of Gaussians crucial to visual synthesis, making it challenging to recover the original performance after even long-term training"。"Mask-based pruning methods also suffer from this issue due to weak sparsity enforcement"。

- 图 2（第 1 节）失败实验：迭代 25K 一次性剪掉 85% 高斯，Importance-based 掉 0.5dB、Opacity-based 掉 0.2dB，且 15K–35K 长期训练都回不来；GaussianSpa 无此损失。
- 图 5（第 3.4 节）机制可视化：Room 场景上不透明度分布随训练出现"零高斯/幸存高斯"两极化，同时 PSNR 持续上升——稀疏约束下信息确实在向幸存高斯转移。
- 图 4/图 8（第 4.3 节）点云可视化：剪枝后高斯自动集中到高频区域、低频区域（天空）用更少更大的高斯。

第 5.3 节对 ℓ1/Lasso 的说明：Lasso 是"the tightest convex approximation but sacrifices exact sparsity"——要精确的"个数 ≤ κ"必须上 ℓ0，而 ℓ0 是 NP-hard 的。它的解法：不优化 ℓ0 目标本身，而是把约束交给辅助变量的硬投影（闭式解存在于 unitary matrix 情形，sparsifying step 对应该情形）。

## 方法

只有训练期一件事，编码期/解码期与标准 3DGS 无异（剪枝结果直接体现在最终模型里，无任何码流侧改动；打分向量是高斯不透明度 a 本身，a 的梯度就是渲染损失对不透明度的梯度，不是手工代理量；spa 状态 z、λ 不进码流）。

流程：正常训练到 15K（实现细节第 4.1 节：与 Mini-Splatting 用相同起始 checkpoint 保证公平）；15K 起，把"优化-稀疏化"三步交替融进训练（式 11/16/17）；25K 删除全部"零高斯"（不透明度已压到≈0 的高斯），之后做一段 light tuning（第 3.4 节）。

问题形式化（式 5）：把训练损失加预算约束 min L, s.t. N(G) ≤ κ。高斯个数不可导，没法直接梯度下降；借渲染公式（式 3）中"每个高斯对像素的贡献最终由其不透明度 a_i 决定、每高斯恰一个不透明度"，把个数约束换成不透明度向量的 ℓ0 约束（式 6）：min_{a,Θ} L(a,Θ), s.t. ‖a‖₀ ≤ κ，其中 a ∈ R^N 是全体不透明度，Θ 是其余 3DGS 变量。引入指示函数（式 7）h(a) = 0（若 ‖a‖₀ ≤ κ）/+∞（否则），变无约束形式（式 8）min_{a,Θ} L(a,Θ) + h(a)；h 仍不可导，于是引入与 a 同形的辅助变量 z，把硬约束挪进等式（式 9）：min_{a,z,Θ} L(a,Θ) + h(z), s.t. a = z。对偶增广拉格朗日（式 10）：L(a,z,Θ,λ;δ) = L(a,Θ) + h(z) + (δ/2)‖a−z+λ‖² + (δ/2)‖λ‖²，λ 是对偶乘子，δ 是惩罚参数。

三步交替：

1. **"Optimizing"（式 11、13、14）**——固定 z、λ，只优化 a 和 Θ：min_{a,Θ} L(a,Θ) + (δ/2)‖a−z+λ‖²，两项都可导，直接梯度下降。梯度（式 13，txt 里式 (12) 未出现、编号可能错位，以 PDF 为准）：∂L/∂a = ∂L(a,Θ)/∂a + δ(a−z+λ)；∂L/∂Θ = ∂L(a,Θ)/∂Θ。更新（式 14）：a ← a − η·∂L/∂a；Θ ← Θ − η·∂L/∂Θ。只有 a 多出一项"被拉向稀疏解 z"的软惩罚；Θ 无直接惩罚，只通过渲染损失重排参数——这是"剪枝后再适应"的数学来源。
2. **"Sparsifying"（式 15、16）**——固定 a、λ，只优化 z：min_z h(z) + (δ/2)‖a−z+λ‖²，是指示函数的近端算子，解析解 z ← prox_h(a+λ)。原文解释（第 3.3 节）："The proximal operator maps the opacity a to the auxiliary variable z, which contains at most κ non-zeros… the projection keeps top-κ elements and sets the rest to zeros."——把 a+λ 中前 κ 大的元素保留、其余清零，一步精确，无需迭代。z 保留元素的取值（置 1 还是保留原值）原文未明说，z 被描述为 "the exactly sparse version of a"，原文公式以 PDF 为准。
3. **Multiplier Update（式 17）**：λ ← λ + a − z。a−z 是约束违反量，λ 把历史违约累积成下一步的持续压力：z 剪掉而 a 涨回来 → 惩罚项往下压；反之往上推。

整体节奏（Algorithm 1）：输入 a、Θ、κ、δ、容差 ϵ、最大迭代 T；`z←a, λ←0`；循环条件 `‖a−z‖₂ > ϵ 且 t ≤ T`；循环体内依次：式 14（Optimizing）→ 式 16（Sparsifying）→ 式 17（Multiplier Update）。收敛即"该保留的高斯稳定下来"（要求不透明度两极化，图 5 可证）。

超参参考（第 8 节、图 9）：δ 扫了 {0.0001, 0.0003, 0.0005} 与 {0.0009, 0.0013, 0.0017} 两组，稀疏投影间隔 interval 扫了 {30, 40, 50, 60, 70, 80}，各档收敛行为一致；正文未给最终取值。

## 实验口径

- 数据集：Mip-NeRF 360、Tanks&Temples、Deep Blending；指标 PSNR/SSIM/LPIPS + 高斯个数（#G/M）。主口径是**高斯个数**而非 MB。
- 对比对象（Table 1）：3DGS（数字引 EAGLES [19]）、CompactGaussian、LP-3DGS-R/M、EAGLES、Mini-Splatting（官方代码复现）、Taming 3DGS、CompGS。
- 训练配置：与原版 3DGS 同环境；**与 Mini-Splatting 用相同起始 checkpoint** 再跑"优化-稀疏化"（第 4.1 节）——起点已含 Mini-Splatting 的初始化红利；15K 开始、25K 删零高斯。
- 存储实验单独放补充材料：加 LightGaussian 的 SH 蒸馏 + 向量量化后报 MB（Table 2）。

## 主结果

- Table 1 平均：Mip-NeRF 360 上 27.85dB / 0.547M 高斯，对 3DGS 的 27.45dB / 3.110M（高斯 6×少、PSNR +0.4dB）；Tanks&Temples 23.98dB / 0.269M 对 23.63dB / 1.830M；Deep Blending 30.37dB / 0.335M 对 29.42dB / 2.780M（约 10×少、+0.9dB，摘要口径）。同表对 LP-3DGS：+0.7dB 且高斯再少 3×（第 4.2 节）。
- Table 2（补充材料，Mip-NeRF 360 存储叠加压缩）：GaussianSpa 27.85dB / 25MB，对 LightGaussian 27.28dB / 42MB、EfficientGS 27.38dB / 98MB。
- Table 5（Deep Blending 逐场景）：GaussianSpa 出现两行（Drjohnson 29.89dB/0.450M 与 29.82dB/0.293M；Playroom 30.84dB/0.219M 两行）——txt 未保留两行的档位标注（应为两个 κ 目标数），以 PDF 为准。

## 消融（方法本身是单一机制，无逐组件开关式消融，证明力在对照实验设计）

- 同剪枝率对照曲线：Figure 7——Kitchen、Room 两场景，横轴是高斯削减率（reduction rate），每个削减率下比 PSNR，GaussianSpa 对 Mini-Splatting 平均 +0.5dB；Figure 14 前 13 子图——多场景 Quality-#G 曲线，"consistently outperforms Mini-Splatting with the same #G"；Figure 14 后 2 子图——对 LightGaussian 的削减率曲线（Kitchen、Playroom）。即"固定预算比质量"画成曲线而不是单点。
- 准则替换实验（第 5.1 节）：把 LightGaussian 的重要性准则塞进式 (15) 的稀疏投影里，其余不动，仍对 LightGaussian 平均 +0.4dB（Playroom、Kitchen，Figure 14）——证明增益来自"训练中稀疏"的框架而非某个分数。
- 失败实验对照：Figure 2——手工准则一次性剪 85% 的 PSNR 曲线（Importance-based −0.5dB、Opacity-based −0.2dB 且不恢复）作反衬。
- 机制可视化：Figure 5（不透明度两极化 + PSNR 同涨）、Figure 6/8（椭球与点云分布：高频密、低频疏）。
- 超参稳健性：Figure 9——δ 两组、interval 六档的 loss 曲线收敛行为一致。
