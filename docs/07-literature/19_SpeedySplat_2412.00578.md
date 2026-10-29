# 《Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives》精读笔记

- arXiv：https://arxiv.org/abs/2412.00578
- 发表：CVPR 2025（本 txt 为 arXiv v3，2025-08-14，含附录 A.1–A.5）

> 一句话定位：主攻渲染加速，两头省钱：把高斯在图像上的占位从"忽略不透明度的圆外接方块"收紧为按 alpha 阈值精确求交的 tile（SnugBox/AccuTile，不改任何画质，平均 1.99×）；把 PUP 3D-GS 的 Hessian 剪枝分数重参数化成反向传播里现成的逐像素梯度平方，配 Soft/Hard 两段剪枝砍掉 10.6× 高斯；合计渲染加速 6.71×、训练加速 1.47×、模型缩小 10.6×（Table 2）。

## 问题与支撑观察

4.1 节开头原话："First, Gaussian Splatting grossly overestimates the extent of Gaussians in the image. Second, ... 3D-GS models are heavily overparameterized." 前者是渲染管线的浪费：原版给高斯分配 tile 时，用投影 2D 协方差最大特征值算半径 r = ⌈3√λmax⌉（式 8），取圆的外接方块，"neglects opacity σi in its calculation and generally overestimates the Gaussian extent"——不透明度低到根本不参与渲染的高斯也被塞进一堆 tile。后者是模型本身塞多了，引三篇压缩工作作证（第 1 节）；并引用 PUP 3D-GS 的结论（4.2.1 节）："Using this score, up to 90% of Gaussians can be robustly pruned from the model while retaining high visual quality"。

- Table 1（第 3.2 节）渲染耗时分解：基线单帧 7.478ms，其中 Render 4.483、Radix Sort 1.551、duplicateWithKeys 0.570、Preprocess 0.665——这四项的耗时全部随"高斯-tile 对"的数量增长，所以把占位算准一项就牵动全部下游函数。
- Fig 2（第 4.1 节）：同一高斯三种 tile 分配的示意（圆外接方块 / 紧致包围盒 / 精确命中椭圆的 tile）。
- Fig 7（附录 A.2）：Soft 0/50–95% × Hard 0–40% 全网格扫描，(80%, 30%) 附近画质几乎不掉。
- 对 PUP 分数的批评只有工程性两点（第 4.2 节）：其一，存储——Hessian 要为每高斯存 μ、s 的 6×6 块（正比于 N×36），而 3DGS 模型本身每高斯 59 个参数，训练期背这份账不现实；其二，计算——PUP 需要逐像素反传再逐高斯聚合，"breaks the efficient flow of gradients in 3D-GS"。

## 方法

渲染管线改造（第 4.1 节，无损，不改渲染结果）：SnugBox 把 alpha 阈值代入 2D 高斯表达式反解出真实像素范围（椭圆），算紧致包围盒；AccuTile 在包围盒内逐行/列解析求交，只保留真正碰到的 tile，计 tile 数只需遍历包围盒短边。两者都在 preprocess 和 duplicateWithKeys 各调一次，自身开销常数级。渲染与求交的式子：α_i(p) = σ_i g_i(p)，g_i = e^q，q = −½ (p − μ_i2D) Σ_i2D^{-1} (p − μ_i2D)^T，参与混合的条件是 α_i > 1/255（式 5–7）；把 α_i = 1/255 代入整理得 2 log(255σ_i) = (p − μ_i2D) Σ_i2D^{-1} (p − μ_i2D)^T，记 t = 2 log(255σ_i)，椭圆方程为 t = a x_d² + 2b x_d y_d + c y_d²（{a, b, c} 为 Σ_i2D^{-1} 的元素），对 y_d 求极值解出 x_dargs = ±√(−b²t / ((b²−ac)a))，回代得 y_d = (−b x_d ± √((b²−ac) x_d² + t c))/c，x 方向的边界由对称性交换变量得到（txt 此处公式编号错乱：((6) 重复出现、(10)/(13)/(14)/(15) 顺序颠倒），原文编号以 PDF 为准）。

剪枝分数（式 17–21，第 4.2.1 节）：PUP 的出发点是 L2 损失的 Hessian H = Σ_{φ∈P_gt} ∇_G I_G(φ) ∇_G I_G(φ)^T（式 17），逐高斯取块 H_i = Σ_φ ∇_{G_i} I_G(φ) ∇_{G_i} I_G(φ)^T（式 18），取 log 行列式得标量分数 U_i = log|∇_{μ_i,s_i} I_G ∇_{μ_i,s_i} I_G^T|（式 19）。Speedy-Splat 的重参数化：把求导对象从 3D 几何参数换成逐像素的 2D 响应标量 g_i(p)，Ũ_i = log|∇_{g_i} I_G ∇_{g_i} I_G^T|（式 20）；因 g_i 是标量、log 单调，化为 Ũ_i = (∇_{g_i} I_G)²（式 21）。原文强调"Gradient ∇gi IG is already computed in the backward pass of render and can be efficiently squared and aggregated across all pixels"，存储从 N×36 降到 N。H 的精确性条件照抄 PUP：L1 残差为零时 Hessian 近似精确（第 4.2 节）。分数只服务于训练期剪枝决策，不进码流。

剪枝调度（第 4.2.2/4.2.3 节）：Soft Pruning 在第 6000/9000/12000 步的三次 opacity reset 之前各剪 80%；Hard Pruning 从 15000 步起每 3000 步剪 30%，两段合计把高斯数压到 1/10.6。时机依据：6K 步后 L1 损失已很小、只在 opacity reset 后回升；密化结束时的模型表现已接近训满模型，后续步数正好当剪后微调用。量化与熵编码：本文不涉及——它是加速/剪枝工作，不压字段比特，体积收益全部来自基元数量减少。

对 StopThePop Tile-Based Culling 的替换对比（附录 A.4）：AccuTile overall 3.660ms（2.038×）对 Tile-Based Culling 4.051ms（1.841×，Table 7），且后者需要 padded alpha threshold，否则会少算高斯-tile 映射。

## 实验口径

场景与 3DGS 官方完全一致：Mip-NeRF360 九景 + T&T 的 train/truck + Deep Blending 的 drjohnson/playroom，用数据集作者发布的 COLMAP 位姿与稀疏点云（5.1 节）。硬件 RTX A5000，CUDA events 计时前向渲染，每场景 3 次独立训练取平均（5.2 节）。体积口径注意：Table 3/5/6 的 Comp 列全部是"高斯数压缩比"，不是 MB。与剪枝基线的对比"we report the published results of each method"（Table 3 说明），即汇总各方发表数字，不是统一重跑。

## 主结果

- Table 2（累加行，全部场景平均）：基线 2.93M / 134 FPS / 23.2min → +SnugBox 1.82× FPS → +AccuTile 1.99× → +Soft Pruning 3.14×（1.79× 压缩）→ +Hard Pruning 0.28M（10.6×）/ 898 FPS（6.71×）/ 15.7min（1.47×）。
- Table 3（Mip-NeRF360）：本文全家桶 10.6× 压缩 / 6.51× FPS / 26.94 dB；对比 PUP 8.65× / 3.20× / 26.83 dB，Mini-Splat 6.84× / 27.34 dB，LightGaussian 2.94× / 2.69× / 27.28 dB，3D-GS 27.55 dB。（txt 中此表缺项以"-"补位且列被拉平，逐格对应按行序推断：FPS 列仅 8 值、Train 列仅 5 值，个别格位可能错位，引用建议核对 PDF。）
- Table 5/6：T&T 全家桶 23.45 dB（3D-GS 23.70）、压缩 10.1×；Deep Blending 29.32 dB 反超 3D-GS 的 29.09 dB、压缩 11.1×。
- Fig 1（truck 单场景）：2.6M → 0.26M 高斯，25.39 → 25.34 dB，184.87 → 1148.77 FPS。

## 消融

- 渐进叠加，三表联动：Table 1（逐函数耗时）、Table 2（高斯数/FPS/训练时长）、Table 3 下半（质量指标）共用同一套行结构（+SnugBox → +AccuTile → +Soft Pruning → +Hard Pruning），每个组件的时间、数量、画质贡献都能分离。
- 同预算分数替换（Table 4）：固定 PUP 的 post-hoc 管线（两轮"剪 66% + 微调 5000 步"，共剪 88.44%），最终都是 0.34M 高斯，只换分数——PUP 分数 26.2136 dB / SSIM 0.8044 / LPIPS 0.2731 / 378.57 FPS，本文分数 26.8658 dB / SSIM 0.8022 / LPIPS 0.2840 / 345.52 FPS：PSNR 高 0.65 dB，LPIPS 和 FPS 略差，全部照实报。
- 参数扫描（Fig 7，附录 A.2）：Soft (0, 50–95%) × Hard (0–40%) 每 5% 一档、每场景 3 次、六个指标的热图，据此选 (80%, 30%)——剪枝比例是扫出来的。
- 替换实验（Table 7/8）：对 StopThePop Tile-Based Culling 的逐函数、逐场景耗时对比，确认精确求交这条路里自己最快。
- 诚实披露（第 6 节 Limitations）：画质比 3D-GS 略降；自家分数与 PUP 分数单独对比仍有"a slight, yet noticeable, gap in performance"。

## 贡献列表（原文照抄）

1. "SnugBox: A precise algorithm for computing Gaussian-tile bounding box intersections."
2. "AccuTile: An extension of SnugBox for computing exact Gaussian-tile intersections."
3. "Soft Pruning: An augmentation for pruning Gaussians during densification."
4. "Hard Pruning: An augmentation for pruning Gaussians post-densification."
