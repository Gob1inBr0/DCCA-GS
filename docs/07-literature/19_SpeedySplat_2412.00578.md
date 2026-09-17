# 《Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2412.00578 |
| 发表 | CVPR 2025（本 txt 为 arXiv v3，2025-08-14） |
| 阅读材料 | 全文（.lit-cache/txt/2412.00578.txt，含附录 A.1–A.5） |
| 与 DCCA-GS 的关系 | 信号设计参考 + 剪枝基线：它的剪枝分数与我们 I6 一样来自反向传播梯度（零侧信息、仅训练期）；Table 4 的"同预算只换分数"对照是我们 SPA+MiniSplat 实验该学的口径 |

> 一句话定位：主攻渲染加速，两头省钱：把高斯在图像上的占位从"忽略不透明度的圆外接方块"收紧为按 alpha 阈值精确求交的 tile（SnugBox/AccuTile，不改任何画质，平均 1.99×）；把 PUP 3D-GS 的 Hessian 剪枝分数重参数化成反向传播里现成的逐像素梯度平方，配 Soft/Hard 两段剪枝砍掉 10.6× 高斯；合计渲染加速 6.71×、训练加速 1.47×、模型缩小 10.6×（Table 2）。

> 任务说明核对：任务提示中的"segment-average 的光栅化精度""min blending weight"在本 txt（v3）中不存在；v3 的剪枝分数是重参数化 Hessian 的逐像素梯度平方（4.2.1 节，式 20–21），且原文明说该梯度"already computed in the backward pass of render"——是反向梯度，不是前向光栅化贡献。另外"高斯被部分遮挡时贡献被高估/低估"的统计或例子在 v3 中也没有，属于论文未报告。以下全部以 v3 实际内容为准。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

两个无效开销，4.1 节开头原话："First, Gaussian Splatting grossly overestimates the extent of Gaussians in the image. Second, ... 3D-GS models are heavily overparameterized." 前者是渲染管线的浪费：原版给高斯分配 tile 时，用投影 2D 协方差最大特征值算半径 r = ⌈3√λmax⌉（式 8），取圆的外接方块，"neglects opacity σi in its calculation and generally overestimates the Gaussian extent"——不透明度低到根本不参与渲染的高斯也被塞进一堆 tile。后者是模型本身塞多了，引三篇压缩工作作证（第 1 节）。

### 支撑观察（直觉 / 数据 / 失败实验）

- Table 1（第 3.2 节）：渲染耗时分解。基线单帧 7.478ms，其中 Render 4.483、Radix Sort 1.551、duplicateWithKeys 0.570、Preprocess 0.665——这四项的耗时全部随"高斯-tile 对"的数量增长，所以把占位算准一项就牵动全部下游函数。这是把"占位高估"换算成时间的第一个数据。
- Fig 2（第 4.1 节）：同一高斯三种 tile 分配的示意（圆外接方块 / 紧致包围盒 / 精确命中椭圆的 tile），占位高估的直观证据。
- 引用 PUP 3D-GS 的结论（4.2.1 节）："Using this score, up to 90% of Gaussians can be robustly pruned from the model while retaining high visual quality"——过度参数化量级的参考。
- Fig 7（附录 A.2）：Soft 0/50–95% × Hard 0–40% 全网格扫描，(80%, 30%) 附近画质几乎不掉——高比例剪枝的可行性是扫出来的，不是假设的。
- 关于"逐高斯独立打分 vs 光栅化实际像素贡献"的批判性论述、"部分遮挡导致贡献高估/低估"的统计：v3 文本未提供，论文未报告；v3 对 PUP 分数的批评只有工程性两点（见下）。

### 显然的路为什么没走

- 直接把 PUP 的 Hessian 分数搬进训练期剪枝：没选，第 4.2 节列了两个缺点——其一，存储：Hessian 要为每高斯存 μ、s 的 6×6 块（正比于 N×36），而 3DGS 模型本身每高斯 59 个参数，训练期背这份账不现实；其二，计算：PUP 需要逐像素反传再逐高斯聚合，"breaks the efficient flow of gradients in 3D-GS"。新分数把存储压到 N×1，省 36×（4.2.1 节）。
- 用现成的 StopThePop Tile-Based Culling（已有的精确 tile 求交法）：附录 A.4 正面对比——AccuTile overall 3.660ms（2.038×）对 Tile-Based Culling 4.051ms（1.841×，Table 7），且后者需要 padded alpha threshold，否则会少算高斯-tile 映射。
- 训后一次性剪枝：没选，改成嵌进训练调度的 Soft（密化中）+ Hard（密化后）两段。时机依据：6K 步后 L1 损失已很小、只在 opacity reset 后回升（4.2.2 节）；密化结束时的模型表现已接近训满模型，后续步数正好当剪后微调用（4.2.3 节）。

## 二学：机制（洞察怎么变成方法）

### 核心流程

渲染管线改造（第 4.1 节，无损，不改渲染结果）：SnugBox 把 alpha 阈值代入 2D 高斯表达式反解出真实像素范围（椭圆），算紧致包围盒；AccuTile 在包围盒内逐行/列解析求交，只保留真正碰到的 tile，计 tile 数只需遍历包围盒短边。两者都在 preprocess 和 duplicateWithKeys 各调一次，自身开销常数级。训练期剪枝（第 4.2 节）：Soft Pruning 在第 6000/9000/12000 步的三次 opacity reset 之前各剪 80%；Hard Pruning 从 15000 步起每 3000 步剪 30%，两段合计把高斯数压到 1/10.6（第 4.2.2/4.2.3 节）。量化与熵编码：本文不涉及——它是加速/剪枝工作，不压字段比特，体积收益全部来自基元数量减少。

### 关键公式（按原文抄，注明编号）

渲染（式 5–7）：α_i(p) = σ_i g_i(p)；g_i = e^q，q = −½ (p − μ_i2D) Σ_i2D^{-1} (p − μ_i2D)^T；C(p) = Σ_{i∈N} c_i α_i(p) Π_{j=1..i−1} (1 − α_j(p))。参与混合的条件是 α_i > 1/255。

原版 tile 半径（式 8）：r = ⌈3√λmax⌉，λmax 为 Σ_i2D 的最大特征值。

alpha 阈值椭圆与 SnugBox：txt 的公式编号在此处错乱（(6) 重复出现、(10)/(13)/(14)/(15) 顺序颠倒），以下按 \label 名与内容引用，原文编号以 PDF 为准。把 α_i = 1/255 代入式 5 整理得 2 log(255σ_i) = (p − μ_i2D) Σ_i2D^{-1} (p − μ_i2D)^T；记 t = 2 log(255σ_i)、x_d = p_x − μ_x、y_d = p_y − μ_y，椭圆方程为 t = a x_d² + 2b x_d y_d + c y_d²（txt 标注 eq:simplified_ellipse），{a, b, c} 为 Σ_i2D^{-1} 的元素。对 y_d 求极值解出 x_dargs = ±√(−b²t / ((b²−ac)a))（txt 标注 eq:bbox_args），回代得 y_d = (−b x_d ± √((b²−ac) x_d² + t c))/c（txt 标注 eq:ellipse_intersect），x 方向的边界由对称性交换变量得到。

剪枝分数链条（式 17–21）：PUP 的出发点是 L2 损失的 Hessian H = Σ_{φ∈P_gt} ∇_G I_G(φ) ∇_G I_G(φ)^T（式 17），逐高斯取块 H_i = Σ_φ ∇_{G_i} I_G(φ) ∇_{G_i} I_G(φ)^T（式 18），取 log 行列式得标量分数 U_i = log|∇_{μ_i,s_i} I_G ∇_{μ_i,s_i} I_G^T|（式 19，txt 标注 eq:pup_score）。Speedy-Splat 的重参数化：把求导对象从 3D 几何参数换成逐像素的 2D 响应标量 g_i(p)，Ũ_i = log|∇_{g_i} I_G ∇_{g_i} I_G^T|（式 20）；因 g_i 是标量、log 单调，化为 Ũ_i = (∇_{g_i} I_G)²（式 21）。原文强调"Gradient ∇gi IG is already computed in the backward pass of render and can be efficiently squared and aggregated across all pixels"，存储从 N×36 降到 N。H 的精确性条件照抄 PUP：L1 残差为零时 Hessian 近似精确（第 4.2 节）。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 每高斯剪枝敏感度 Ũ_i = (∇_{g_i} I_G)²：渲染结果对"该高斯逐像素 2D 响应"的梯度平方、跨像素聚合，是 L2 损失 Hessian 的重参数化标量近似 |
| 哪一端算得出（编码端/解码端/仅训练期） | 仅训练期：渲染 backward 里现成的逐像素梯度，平方聚合即可，无额外前向开销 |
| 有无侧信息 | 无（分数只服务于训练期剪枝决策，不进码流） |
| 训练期还是编码期 | 训练期：Soft/Hard 剪枝都嵌在训练调度里 |
| 和渲染质量的距离（直接测质量还是代理量） | 代理量：由 L2 损失的二阶敏感度推出（L1 残差为零时精确），不直接测渲染质量 |

与我们 I6（渲染梯度 EMA，机制 B）在信号来源上的异同：

| 维度 | Speedy-Splat 的 Ũ_i | 我们的 I6 |
| --- | --- | --- |
| 梯度来源 | 反向传播，对渲染中间量 g_i(p) 求导 | 反向传播，对可学习属性（feat/scaling/offsets）的渲染损失梯度 |
| 聚合方式 | 逐像素平方后跨像素求和（空间聚合，二阶幅度量，负梯度也算正贡献） | 时间上取 EMA（带符号一阶量，平滑训练波动） |
| 空间/属性分辨 | 逐高斯、逐像素——能说出"哪些像素在乎这个高斯" | 逐锚点、逐属性——能说出"哪个属性持续被关注" |
| 用途 | 排序后删基元（剪枝分数） | 训练共享 MLP 调量化步长（基元留下、改精度） |
| 侧信息/码流 | 均不进码流 | 均不进码流 |

需要纠正的一个印象：提示里说它用"前向光栅化贡献"，按 v3 原文不是——真正的"前向贡献"打分是 LightGaussian、Mini-Splatting importance、RadSplat 那一类（累计 blending 权重）。"前向贡献 vs 反向梯度"是剪枝分数的一条分类轴，写相关工作时可以沿用。

## 三学：证明与包装（创新怎么立住）

### 实验口径

场景与 3DGS 官方完全一致：Mip-NeRF360 九景 + T&T 的 train/truck + Deep Blending 的 drjohnson/playroom，用数据集作者发布的 COLMAP 位姿与稀疏点云（5.1 节）。硬件 RTX A5000，CUDA events 计时前向渲染，每场景 3 次独立训练取平均（5.2 节）。体积口径注意：Table 3/5/6 的 Comp 列全部是"高斯数压缩比"，不是 MB——与我们的 total_MB 字段求和口径不同，引用时需要说明。与剪枝基线的对比"we report the published results of each method"（Table 3 说明），即汇总各方发表数字，不是统一重跑。

### 主结果（注明表号）

- Table 2（累加行，全部场景平均）：基线 2.93M / 134 FPS / 23.2min → +SnugBox 1.82× FPS → +AccuTile 1.99× → +Soft Pruning 3.14×（1.79× 压缩）→ +Hard Pruning 0.28M（10.6×）/ 898 FPS（6.71×）/ 15.7min（1.47×）。
- Table 3（Mip-NeRF360）：本文全家桶 10.6× 压缩 / 6.51× FPS / 26.94 dB；对比 PUP 8.65× / 3.20× / 26.83 dB，Mini-Splat 6.84× / 27.34 dB，LightGaussian 2.94× / 2.69× / 27.28 dB，3D-GS 27.55 dB。txt 中此表缺项以"-"补位且列被拉平，逐格对应按行序推断（FPS 列仅 8 值、Train 列仅 5 值），个别格位可能错位，引用建议核对 PDF。
- Table 5/6：T&T 全家桶 23.45 dB（3D-GS 23.70）、压缩 10.1×；Deep Blending 29.32 dB 反超 3D-GS 的 29.09 dB、压缩 11.1×。
- Fig 1（truck 单场景）：2.6M → 0.26M 高斯，25.39 → 25.34 dB，184.87 → 1148.77 FPS。

### 消融设计（注明表号）

- 渐进叠加，三表联动：Table 1（逐函数耗时）、Table 2（高斯数/FPS/训练时长）、Table 3 下半（质量指标）共用同一套行结构（+SnugBox → +AccuTile → +Soft Pruning → +Hard Pruning），每个组件的时间、数量、画质贡献都能分离。这种"一个行结构、三张表各看一个侧面"的呈现方式很值得抄。
- 同预算分数替换（本文最值得学的对照，Table 4）：固定 PUP 的 post-hoc 管线（两轮"剪 66% + 微调 5000 步"，共剪 88.44%），最终都是 0.34M 高斯，只换分数——PUP 分数 26.2136 dB / SSIM 0.8044 / LPIPS 0.2731 / 378.57 FPS，本文分数 26.8658 dB / SSIM 0.8022 / LPIPS 0.2840 / 345.52 FPS：PSNR 高 0.65 dB，LPIPS 和 FPS 略差，全部照实报。这是"信号本身有没有价值"最干净的证明形态。
- 参数扫描（Fig 7，附录 A.2）：Soft (0, 50–95%) × Hard (0–40%) 每 5% 一档、每场景 3 次、六个指标的热图，据此选 (80%, 30%)——把"比例是拍脑袋"变成"比例是扫出来的"。
- 替换实验（Table 7/8）：对 StopThePop Tile-Based Culling 的逐函数、逐场景耗时对比，确认精确求交这条路里自己最快。
- 诚实披露（第 6 节 Limitations）：画质比 3D-GS 略降；自家分数与 PUP 分数单独对比仍有"a slight, yet noticeable, gap in performance"。

### 贡献列表（原文照抄 + 逐条标注）

1. "SnugBox: A precise algorithm for computing Gaussian-tile bounding box intersections." —— [机制] 占位收紧第一步，无损，平均 1.82×（Table 2）。
2. "AccuTile: An extension of SnugBox for computing exact Gaussian-tile intersections." —— [机制] 精确 tile 求交，平均 1.99×（Table 2）。
3. "Soft Pruning: An augmentation for pruning Gaussians during densification." —— [机制] 密化期内嵌剪枝，单次 80%。
4. "Hard Pruning: An augmentation for pruning Gaussians post-densification." —— [工程] 密化后定期剪 30%，两段合计 10.6×（Table 2）。
（四条全是机制/工程条目，无观察类条目；动机性论证放在引言里。）

### 讲故事方式

主线一句话："像素算得准一点、基元删得狠一点，两头一起省。"最有力的是 Table 4：同一条管线、同一个最终数量、只换一个分数，0.65 dB 的差距把"分数设计本身有价值"独立立住了——全篇主结果（6.71×）混杂了渲染加速和剪枝两个来源，只有 Table 4 是纯分数对照。Fig 2 的三格示意图负责让"占位高估"一眼看懂。

## 对 DCCA-GS 的可借鉴点

1. 信号设计互验（学它的重参数化 → 用到我们 SPA 打分与 I6 监督）：它证明"对渲染中间量的逐像素梯度取平方聚合"能当剪枝分数，且几乎零成本（backward 里现成）。我们的梯度本来就在算，两个低成本的交叉实验：(a) 把 I6 的属性级梯度 EMA 拿来当 SPA 的剪枝排序分数试一次；(b) 反过来，在 I6 的监督信号里加一路 (∂I/∂g)² 式的逐像素平方梯度，与纯属性 EMA 对照。都满足硬约束：训练期信号，零侧信息，不进码流。实验设计照 Table 4：同一底座、同一剪除率（或同一 total_MB），只换信号。
2. 同预算对照口径（直接搬）：Table 4 的"固定管线、固定最终基元数、只换分数"是我们 SPA+MiniSplat 实验该有的对照形态；再配 Mini-Splatting Fig 7 式的预算扫描曲线，一个证明"信号好"，一个证明"全预算段都好"。
3. 剪枝调度（学它的时机选择 → 用到 SPA 调度）：它把大比例剪枝安排在训练动态的稳定点（opacity reset 前 6K/9K/12K 各一次），而不是均匀撒。我们 SPA 的 ADMM 收缩目前按固定间隔走，可以改成在 HAC++ 训练的关键节点（如增密结束前后分两段）做 Soft/Hard 式调度，实验：改 SPA 调度后的 RD 曲线对比。
4. 不值得跟进：SnugBox/AccuTile 是渲染引擎优化，不改画质也不动码流契约，与我们正交，相关工作里归入加速线即可。另注意它的压缩比口径是高斯数而非 MB，写对比时需换算或注明。
