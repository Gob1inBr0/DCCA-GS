# 《GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2411.06019 |
| 发表 | CVPR 2025（任务指定；txt 为 arXiv v3，2025-04-10，正文未出现会议标注） |
| 阅读材料 | 全文（.lit-cache/txt/2411.06019.txt） |
| 与 DCCA-GS 的关系 | 基础措施来源：我们的 SPA 就是它的"优化-稀疏化"ADMM 剪枝的锚点版改造（创新点③） |

> 一句话定位：把"剪掉多少高斯"从手工准则改成训练目标里的硬约束（高斯数 ≤ κ），用 ADMM 三步交替（梯度优化 / TopK 硬投影 / 乘子更新）在训练中渐进施加稀疏，让信息从被剪高斯平滑转移给幸存高斯——同更少的高斯数下渲染质量反超原版 3DGS。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

第 1 节，两层：

1. 3DGS 存储大头是高斯本身，"the sheer volume of Gaussians leads to memory usage that exceeds the capacity of typical hardware"。
2. 现有剪枝都靠手工准则（opacity [22,50]、importance score/hit count [16,17,37]、dominant primitives [28]、binary mask [26,51]），"such single-perspective heuristic criteria may lack robustness"；更要命的是"sudden one-shot removal may cause permanent loss of Gaussians crucial to visual synthesis, making it challenging to recover the original performance after even long-term training"。掩码类方法也一样："Mask-based pruning methods also suffer from this issue due to weak sparsity enforcement"。

一句话：不是"剪多少"难，是"一次性剪、按单一分数剪"不可恢复——和我们创新点③动机里"剪掉就忘"的判断完全一致。

### 支撑观察（直觉 / 数据 / 失败实验）

- 图 2（第 1 节）：失败实验——迭代 25K 一次性剪掉 85% 高斯，Importance-based 掉 0.5dB、Opacity-based 掉 0.2dB，且 15K–35K 长期训练都回不来；GaussianSpa 无此损失。
- 图 5（第 3.4 节）：机制可视化——Room 场景上不透明度分布随训练出现"零高斯/幸存高斯"两极化，同时 PSNR 持续上升，证明稀疏约束下信息确实在向幸存高斯转移。
- 图 4/图 8（第 4.3 节）：点云可视化——剪枝后高斯自动集中到高频区域、低频区域（天空）用更少更大的高斯，这是"训练中稀疏"才有的自适应分布。

### 显然的路为什么没走

- 显然路 A：更好的手工重要性分数（LightGaussian 的 hit count、RadSplat 的 ray contribution）。它不否认有用，但认为是单视角代理量，鲁棒性差——干脆不要分数，让渲染损失的梯度自己决定谁重要（a 的梯度来自 L 本身）。
- 显然路 B：可学习掩码（Compact3DGS、LP-3DGS）。它批评掩码方法"weak sparsity enforcement on the multitude of Gaussians"——掩码损失是软正则，压不出干净的可剪集合。
- 显然路 C：ℓ1/Lasso 正则（稀疏学习标准做法）。第 5.3 节明确说 Lasso 是"the tightest convex approximation but sacrifices exact sparsity"——要精确的"个数 ≤ κ"必须上 ℓ0，而 ℓ0 是 NP-hard 的。它的解法：不优化 ℓ0 目标本身，而是把约束交给辅助变量的硬投影（TopK 是特例下的精确解）。

## 二学：机制（洞察怎么变成方法）

### 核心流程

只有训练期一件事，编码期/解码期与标准 3DGS 无异（剪枝结果直接体现在最终模型里，无任何码流侧改动）：

1. 正常训练到 15K（实现细节第 4.1 节：与 Mini-Splatting 用相同起始 checkpoint 保证公平）；
2. 15K 起，把"优化-稀疏化"三步交替融进训练（式 11/16/17，见下）；
3. 25K 删除全部"零高斯"（不透明度已压到≈0 的高斯），之后做一段 light tuning（第 3.4 节），"even slightly higher than the original 3DGS"。

### 关键公式（按原文抄，注明编号）

问题形式化（式 5）：把训练损失加预算约束

> min L,  s.t. N(G) ≤ κ

高斯个数不可导，没法直接梯度下降。借渲染公式（式 3）中"每个高斯对像素的贡献最终由其不透明度 a_i 决定、每高斯恰一个不透明度"，把个数约束换成不透明度向量的 ℓ0 约束（式 6）：

> min_{a,Θ} L(a,Θ),  s.t. ‖a‖₀ ≤ κ

其中 a ∈ R^N 是全体不透明度，Θ 是其余 3DGS 变量。引入指示函数（式 7）

> h(a) = 0, 若 ‖a‖₀ ≤ κ； +∞, 否则

把式 6 变无约束形式（式 8）`min_{a,Θ} L(a,Θ) + h(a)`；h 仍不可导，于是引入与 a 同形的辅助变量 z，把硬约束挪进等式（式 9）：

> min_{a,z,Θ} L(a,Θ) + h(z),  s.t. a = z

到此"可导的 L"和"不可导的 h"各管一个变量，只剩一条等式约束。对偶增广拉格朗日（式 10）：

> L(a,z,Θ,λ;δ) = L(a,Θ) + h(z) + (δ/2)‖a−z+λ‖² + (δ/2)‖λ‖²

λ 是对偶乘子，δ 是惩罚参数。之后三步交替（这正是我们 SPA 的骨架，每步编号照抄原文）：

**第一步 "Optimizing"（式 11、13、14）**——固定 z、λ，只优化 a 和 Θ：

> min_{a,Θ} L(a,Θ) + (δ/2)‖a−z+λ‖²   (11)

两项都可导，直接梯度下降。梯度（式 13，txt 中该式标号为 (13)，且 txt 里式 (12) 未出现，编号可能有错位，以 PDF 为准）：

> ∂L/∂a = ∂L(a,Θ)/∂a + δ(a−z+λ)；  ∂L/∂Θ = ∂L(a,Θ)/∂Θ

更新（式 14）：

> a ← a − η·∂L/∂a；  Θ ← Θ − η·∂L/∂Θ

机制要点：只有 a 多出一项"被拉向稀疏解 z"的软惩罚；Θ 无直接惩罚，只通过渲染损失重排参数——这就是"剪枝后再适应"的数学来源。

**第二步 "Sparsifying"（式 15、16）**——固定 a、λ，只优化 z：

> min_z h(z) + (δ/2)‖a−z+λ‖²   (15)

这是指示函数的近端算子，解析解（式 16）：

> z ← prox_h(a+λ)

原文解释（第 3.3 节）："The proximal operator maps the opacity a to the auxiliary variable z, which contains at most κ non-zeros… the projection keeps top-κ elements and sets the rest to zeros."——把 a+λ 中前 κ 大的元素保留、其余清零，一步精确，无需迭代。注意：原文只说"保留前 κ 个、其余置零"，没有明说保留元素是置 1 还是保留原值；z 被描述为 "the exactly sparse version of a"。我们文档按 0/1 二值理解（TopK 推导 `(s_i−1)²<s_i² ⇔ s_i>1/2` 也是二值口径），与收敛条件 ‖a−z‖→0（要求不透明度两极化，图 5 可证）相容，但严格说这是我们的解释，以 PDF 为准。

**第三步 Multiplier Update（式 17）**：

> λ ← λ + a − z

a−z 是约束违反量，λ 把历史违约累积成下一步的持续压力：z 剪掉而 a 涨回来 → 惩罚项往下压；反之往上推。没有 λ，每次投影后 a 可能"忘记"预算——λ 给了预算约束记忆。

**整体节奏（Algorithm 1，txt 与我们文档逐字一致）**：输入 a、Θ、κ、δ、容差 ϵ、最大迭代 T；`z←a, λ←0`；循环条件 `‖a−z‖₂ > ϵ 且 t ≤ T`；循环体内依次：式 14（Optimizing）→ 式 16（Sparsifying）→ 式 17（Multiplier Update）。收敛即"该保留的高斯稳定下来"。

超参参考（第 8 节、图 9）：δ 扫了 {0.0001, 0.0003, 0.0005} 与 {0.0009, 0.0013, 0.0017} 两组，稀疏投影间隔 interval 扫了 {30, 40, 50, 60, 70, 80}，各档收敛行为一致；正文未给最终取值（论文未报告具体数值档位）。

### 与我们文档（创新点说明 §3）的核对（任务特别要求）

逐条对照 txt：**式 (5)(6)(7)(8)(9)(10)(11)(14)(15)(16)(17) 与 Algorithm 1 的 8 个步骤、循环条件、三步公式编号全部一致，无不一致项**。需注意的细节：

1. 式 (5) 原文目标是 `min L`（泛指训练损失），我们文档写成 `min L(Θ)`——记法差异，含义一致；
2. 梯度式在原文有独立编号 (13)，我们文档把梯度式并进"(11)–(14)"区间未单独标号，内容一致；txt 中式 (12) 缺失、(8) 的标号错位到 (13) 附近，属抽取错位；
3. z 保留元素的取值（置 1 vs 保留原值）原文未明说，见上文式 (16) 处的分析；我们按二值实现（z 就是 0/1 的 prune 掩码），自洽；
4. 我们文档 3.5 节"一般矩阵下稀疏投影无解析解、逐坐标 TopK 是特例"对应原文 5.3 节的说法是"闭式解存在于 unitary matrix 情形，sparsifying step 对应该情形"——角度不同，结论一致；
5. 实现节奏差异：原文的稀疏投影间隔是超参（图 9 扫 30–80），我们的实现是每 100 步投影，在该范围之外；δ=1e-3（我们的 spa_rho）落在原文扫描范围内；15K 开始 / 25K 删零 / light tuning 与我们文档 3.5 节第 4 点的转述一致。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | 高斯不透明度 a 本身（渲染公式里该高斯贡献的最终乘子）；重要性排序发生在 TopK 投影时，打分向量是 a+λ（λ 含历史违约记忆） |
| 哪一端算得出 | 仅训练期；解码端与码流完全无关 |
| 有无侧信息 | 无——剪枝结果就是最终模型本身，spa 状态（z、λ）不进码流（与我们 PHG 实现同口径） |
| 训练期还是编码期 | 训练期（15K–25K 内渐进施加），编码期无操作 |
| 和渲染质量的距离 | 直接：a 的梯度就是渲染损失对不透明度的梯度，不是任何手工代理量；这是它与所有准则类剪枝的本质区别 |

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 数据集：Mip-NeRF 360、Tanks&Temples、Deep Blending；指标 PSNR/SSIM/LPIPS + 高斯个数（#G/M）。注意它的主口径是**高斯个数**而非 MB。
- 对比对象（Table 1）：3DGS（数字引 EAGLES [19]）、CompactGaussian、LP-3DGS-R/M、EAGLES、Mini-Splatting（官方代码复现）、Taming 3DGS、CompGS。
- 训练配置：与原版 3DGS 同环境；**与 Mini-Splatting 用相同起始 checkpoint** 再跑"优化-稀疏化"（第 4.1 节）——即它的起点已含 Mini-Splatting 的初始化红利；15K 开始、25K 删零高斯。
- 存储实验单独放补充材料：加 LightGaussian 的 SH 蒸馏 + 向量量化后报 MB（Table 2）。

### 主结果（注明表号）

- Table 1 平均：Mip-NeRF 360 上 27.85dB / 0.547M 高斯，对 3DGS 的 27.45dB / 3.110M（高斯 6×少、PSNR +0.4dB）；Tanks&Temples 23.98dB / 0.269M 对 23.63dB / 1.830M；Deep Blending 30.37dB / 0.335M 对 29.42dB / 2.780M（约 10×少、+0.9dB，摘要口径）。同表对 LP-3DGS：+0.7dB 且高斯再少 3×（第 4.2 节）。
- Table 2（补充材料，Mip-NeRF 360 存储叠加压缩）：GaussianSpa 27.85dB / 25MB，对 LightGaussian 27.28dB / 42MB、EfficientGS 27.38dB / 98MB。
- Table 5（Deep Blending 逐场景）：GaussianSpa 出现两行（Drjohnson 29.89dB/0.450M 与 29.82dB/0.293M；Playroom 30.84dB/0.219M 两行）——txt 未保留两行的档位标注（应为两个 κ 目标数），以 PDF 为准。

### 消融设计（注明表号）

它没有"逐组件开关"式消融（方法本身就是单一机制），证明力放在**对照实验设计**上，这正是要学的地方：

1. **同剪枝率对照曲线（核心设计）**：Figure 7——Kitchen、Room 两场景，横轴是高斯削减率（reduction rate），每个削减率下比 PSNR，GaussianSpa 对 Mini-Splatting 平均 +0.5dB；Figure 14 前 13 子图——多场景 Quality-#G 曲线，"consistently outperforms Mini-Splatting with the same #G"；Figure 14 后 2 子图——对 LightGaussian 的削减率曲线（Kitchen、Playroom）。即"固定预算比质量"画成曲线而不是单点，一张图封死"你只是挑了好看的操作点"的质疑。
2. **准则替换实验（框架通用性）**：第 5.1 节把 LightGaussian 的重要性准则塞进式 (15) 的稀疏投影里，其余不动，仍对 LightGaussian 平均 +0.4dB（Playroom、Kitchen，Figure 14）——证明增益来自"训练中稀疏"的框架而非某个分数。
3. **失败实验对照**：Figure 2——手工准则一次性剪 85% 的 PSNR 曲线（Importance-based −0.5dB、Opacity-based −0.2dB 且不恢复）作反衬。
4. **机制可视化**：Figure 5（不透明度两极化 + PSNR 同涨）、Figure 6/8（椭球与点云分布：高频密、低频疏）。
5. **超参稳健性**：Figure 9——δ 两组、interval 六档的 loss 曲线收敛行为一致。

### 贡献列表（原文照抄 + 逐条标注）

1. "We propose a general 3DGS simplification framework that formulates the simplification objective as an optimization problem and solves it in the 3DGS training process. … Hence, GaussianSpa can maximally maintain and smoothly transfer the information to the sparse Gaussians from the original model." —— [机制] 剪枝问题化：约束写进训练目标，信息平滑转移而非一次性删除。
2. "We propose an efficient 'optimizing-sparsifying' solution for the formulated problem, which can be integrated into the 3DGS training with negligible costs, separately solving two sub-problems." —— [机制] ADMM 三步求解：梯度步 + TopK 解析投影 + 乘子更新，训练开销可忽略。
3. "We comprehensively evaluate GaussianSpa via extensive experiments on various complex scenes, demonstrating improved rendering quality compared to existing state-of-the-art approaches." —— [实证] 三个数据集 + 同预算曲线 + 准则替换，数字见上文主结果。

三条均实，无凑数。

### 讲故事方式

主线一句话："别在训练结束后按分数砍，把'高斯数 ≤ κ'写进训练目标，让梯度把信息搬到幸存高斯上。"最有力的是 Figure 2 + Figure 5 组合：前者是失败实验（一次性剪枝永久掉点、训练救不回来），后者是机制可视化（不透明度两极化的同时 PSNR 上涨）——先立"现有路走不通"，再给"我的路在正常工作"的直接证据；结论由 Table 1 的"+0.4~0.9dB 且高斯少一个量级"收束。

## 对 DCCA-GS 的可借鉴点

1. **light tuning 阶段（第 3.4 节、Figure 5）**：原文在 25K 删净零高斯后专门跑一段轻量微调，且最终质量反超原版。我们目前 SPA 投影持续到训练结束，没有"删净→冻结预算→集中微调"的收尾动作。可试：最后一次投影后把 z=0 锚点整体删除并停止投影，再微调若干步，验证能否补回阶段 A 中 −1.23dB 的一部分。实验：DB playroom 30k，删后微调 0/3k/5k 三档。这与创新点③"待办：SPA 后继续训练"的验证可以直接合并。
2. **分数扩容有原文背书（第 5.1 节）**：原文打分向量只有不透明度 a（无位置/尺度信息），且证明了投影步骤可插任意准则（换 LightGaussian 准则仍 +0.4dB）。这给我们的改造留了口子：锚点分数从"掩码均值"扩成"掩码均值 + 可重算静态项"（复用 I2 的局部密度/尺度各向异性特征）不破坏 ADMM 结构——SPA 状态不进码流，不受零侧信息约束。实验：同预算下 s=mean(mask) 对照 s=mean(mask)+β·f(I2 特征)，看同锚点数 PSNR。
3. **剪枝粒度的两级化**：原文逐高斯投影；我们升到锚点级（原文框架对粒度不敏感，5.1 节为证）。反向可再借一步：幸存锚点内部的 K 个高斯掩码还可以跑一轮锚点内稀疏化——masks 字段本身按锚点计费，内部稀疏化能直接省 bit_masks 且不改变锚点数。属于"锚点级 + 高斯级"两段式投影，工程上是同一套 TopK 复用。
4. **同预算对照曲线（三学设计）**：我们创新点③的验证目前是单 ratio=0.5 的同锚点数对照（§3.7）。应照它的 Figure 7/14 画 PSNR–剪枝率曲线（ratio 扫 0.2–0.8，每点同预算对编码端 topk），再加一个准则替换消融（ADMM 框架不变、分数换成纯 opacity 或 mask-topk）证明增益来自框架本身——这组设计搬过来就能撑起我们的剪枝章节。
