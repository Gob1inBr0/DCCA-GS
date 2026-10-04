# 《InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2608.02437（v2，2026-08-04） |
| 发表 | SIGGRAPH Asia 2026 期刊轨道（据任务方信息；正文为 ACM 版式但未印接收标注） |
| 阅读材料 | 全文（.lit-cache/txt/2608.02437.txt；txt 中范数符号损坏为 □，已用 pymupdf 重抽 PDF 第 4–6 页复核公式） |
| 与 DCCA-GS 的关系 | 同属前馈高斯方向但任务错开：它做单图生成不做压缩，不构成竞品；其"解码端可重算量驱动资源分配"的信号设计可作参考 |

> 一句话定位：这篇做的是从一张照片一次前向生成可实时渲染的 3D 高斯场景（省掉多视角采集和逐场景优化）；它把高斯的生成位置从固定像素网格挪到深度几何引导的表面对齐支撑点上，再用一个共享 MLP 在任意坐标查询图像特征、预测有界属性残差，换来大视角偏移下更稳的场景结构。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

原话（第 1 节）："A central limitation is their pixel-aligned representation: Gaussian primitives are generated from fixed image-grid locations, so they can behave more like locally expanded splats than a surface-aligned scene layout."（中心局限是像素对齐表示：高斯基元从固定的图像网格位置生成，行为更像局部放大的泼溅块，而不是表面对齐的场景布局。）

解释：现有单图前馈 3DGS 方法（Splatter Image、Flash3D、SHARP）逐像素（或逐图像块）回归高斯属性，近视角看着锐利，但高斯之间的空间组织松散；目标相机一旦横向拉开（大基线），撕裂、表面断裂、结构松散、几何畸变就暴露出来。它的核心主张是单图 3DGS 应从"像素对齐表示"转向"表面对齐表示"，并给出两个必要条件：高斯的支撑位置要跟随深度导出的局部表面结构；高斯属性预测不能绑死在固定像素中心（第 1 节）。

### 支撑观察（直觉 / 数据 / 失败实验）

- 定性观察：图 1、图 3——大视角偏移下 SHARP 在大平面和物体边界出现撕裂、断连、局部漂移，Flash3D 出现全局模糊和几何拉伸；图 4 的法向图直接显示 SHARP 的高斯碎片化、本文的法向连续（第 4.3 节文字陈述）。
- 定量观察：表 1，SHARP 在 Tanks and Temples（户外大场景）PSNR 仅 15.750，InfiniSplat-RGB 为 17.118；DL3DV 上 SHARP 18.305 对 21.685。
- 消融证据：表 2，把几何引导采样换回像素对齐支撑，ScanNet++ PSNR 从 22.240 掉到 21.576；换掉隐式解码器（用 DPT 逐像素解码）掉到 20.811——两个组件各自都有贡献。
- 第 2.2 节关于"可学习 query token 在单图下缺几何锚点"的判断属于分析性论述，本文没有给出自己的失败实验数据支撑（作者论证，非实验观察）。

### 显然的路为什么没走

1. **可学习 query token 解码**（C3G、TokenGS 已在多视角设定走通）：第 2.2 节明确说，多视角设定里跨视角约束能帮可学习 token 找到紧凑一致的高斯布局；单图设定没有这些线索，token 缺显式几何锚点，难以沿连贯表面组织高斯。所以它保留查询式解码的灵活，但把查询点锚定在几何先验导出的支撑点上。
2. **DPT 式逐像素解码**：第 3.1 节说逐像素回归处理不了采样后的不规则支撑布局；表 2 的 w/o Implicit Decoder 行用 DPT 解码器验证这条路确实更差。
3. **多尺度 DINO 特征金字塔**（同组的 InfiniDepth 就是这么做的）：第 3.3 节说明本文故意只用单层 DINO 特征供语义骨架，另配 CNN 分支补局部纹理和边缘——输出对象从"任意分辨率深度图"变成了"一组高斯属性"，条件需求不同。
4. **扩散式 / 迭代生成路线**（CAT3D、Gen3C 等）：第 2.1 节说这类方法直接合成图像或依赖迭代生成，给不出紧凑、显式、实时可渲染的表示。

## 二学：机制（洞察怎么变成方法）

### 核心流程

该论文没有逐场景优化、没有编码期：训练一次，之后每张输入图一次前向出高斯。分推理期和训练期讲。

**推理期（第 3.1–3.3 节）**，四步：

1. **并行预处理**：冻结的单目深度模型 Φ_geo（DepthPro）从输入图 I 得到稠密深度图 D 和内参估计 K̂；可训练的双分支编码器 Φ_img 得到 DINO 语义特征图 F_dino 和 CNN 纹理特征图 F_cnn。骨干为 DINOv3 ViT-L/16 加 128 通道 CNN（7×7 步幅 2 卷积起步，八个 3×3 残差块，多尺度特征经 3×3 卷积融合、1×1 卷积投影到 128 通道，第 4.1 节）。
2. **几何引导采样**：把深度图反投影成逐像素 3D 点；相邻像素连成局部三角形，丢弃顶点相对深度变化过大的三角形（防止一个面横跨前景背景的深度断裂）；每个三角形的 3D 面积归一化成采样权重 w_t，面积大或倾斜强的表面分到更多支撑点；在对应 2D 三角形区域内采样出 N 个查询坐标（本文 N = 1.5M，第 4.1 节）。
3. **基础高斯初始化**：每个查询坐标 q_i 反投影成 3D 位置 μ̄_i，颜色取输入图双线性采样值 c̄_i = I(q_i)，尺度、旋转、不透明度用统一基础初始化（具体值论文未给）——得到基础高斯集 Ḡ。
4. **隐式高斯解码**（三小步，全部由所有支撑点共享）：特征查询——在 q_i 处双线性查询 F_dino、F_cnn；特征融合——CNN 特征先线性投影到 DINO 维度，再逐通道门控加权融合成描述子 h_i；参数更新——共享 MLP 把 h_i 映射成 14 维有界残差（图平面位置 2 维、深度 1 维、尺度 3 维、旋转四元数 4 维、颜色 3 维、不透明度 1 维），位置残差按 λ 缩放后反投影回 3D，尺度、颜色、不透明度在各自激活函数的反函数空间里加偏移再过激活以保持取值合法，旋转在基础四元数上加更新后归一化。输出最终高斯集 G，直接实时渲染。

**训练期（第 3.4 节）**：

- 数据：Hypersim 合成室内场景，每个样本一个上下文视角加三个目标视角；前向生成高斯后在目标相机下渲染，用 RGB L1 损失加 VGG 感知损失（含 Gram 矩阵项，仅作用于目标视角）监督。
- 高斯正则：尺度范围正则（log 尺度钳在 [−8, −3]，归一化尺度空间）加邻近支撑点尺度与不透明度的平滑正则，稳住显式高斯属性，防止退化成过薄、半透明、局部不稳定的泼溅。
- 配置：8 张 NVIDIA H20、约 10 万步、AdamW、学习率 5×10⁻⁵、单卡批大小 1；损失权重 λ_rgb = 1、λ_perc = 1、λ_reg = 0.1（第 4.1 节）。

量化、熵编码、剪枝：该论文全都不涉及——不压缩高斯，生成 1.5M 个就存 1.5M 个、渲染 1.5M 个。

两个组件为什么绑定：几何引导采样挣脱了像素网格，但产出的不规则布局逐像素回归处理不了；隐式解码能处理任意布局，但没有几何支架就退化成在无结构点集上操作（第 3.1 节原话，见四学翻译段）。

### 关键公式

原文公式无编号，按第 3.2–3.4 节出现顺序转写；txt 里范数符号损坏，以下已经 pymupdf 对照 PDF 第 4–6 页复核。原文公式以 PDF 为准。

流程（第 3.1 节）：

- (D, K̂) = Φ_geo(I)；(F_dino, F_cnn) = Φ_img(I)；(S, Ḡ) = Q(D, I, K̂)；G = D_θ(Ḡ, S, F_dino, F_cnn)。Φ_geo 是冻结深度模型，Φ_img 是双分支编码器，Q 是几何引导采样，D_θ 是隐式解码器，S 是支撑点集，Ḡ 是基础高斯集。

几何引导采样（第 3.2 节）：

- 反投影：X(p) = Π⁻¹(p, D(p); K̂)。p 是像素坐标，Π⁻¹ 是由 K̂ 定义的 Back-projection（反投影）函数。
- 三角形面积：A_t = ½‖(X(p_b) − X(p_a)) × (X(p_c) − X(p_a))‖₂。由相邻像素 p_a、p_b、p_c 组成的局部三角形反投影后的 3D 面积，即叉积模长的一半。
- 采样权重：w_t = A_t / Σ_{t′} A_{t′}。按面积归一化，决定每个局部表面分到多少支撑点。

基础高斯初始化（第 3.2 节）：

- ḡ_i = (μ̄_i, s̄_i, r̄_i, c̄_i, ᾱ_i)，其中 μ̄_i = Π⁻¹(q_i, D(q_i); K̂)，c̄_i = I(q_i)；s̄_i、r̄_i、ᾱ_i 用统一基础初始化。

隐式解码（第 3.3 节）：

- 特征查询：f_i^dino = B(F_dino, q_i)，f_i^cnn = B(F_cnn, q_i)。B 是双线性采样。
- 门控融合：f̃_i^cnn = P(f_i^cnn)；α_i = σ(W_g[f_i^dino, f̃_i^cnn])；h_i = α_i ⊙ f_i^dino + (1 − α_i) ⊙ f̃_i^cnn。P 是线性投影，σ 是 sigmoid，α_i 是逐通道门控，h_i 是送入高斯 MLP 的融合描述子。
- 参数更新：Δg_i = MLP_θ(h_i) = (Δu_i, Δz_i, Δs_i, Δr_i, Δc_i, Δα_i) ∈ R¹⁴。Δu_i ∈ R² 更新图平面位置，Δz_i 更新深度方向，Δs_i ∈ R³、Δr_i ∈ R⁴、Δc_i ∈ R³、Δα_i ∈ R 分别更新尺度、四元数旋转、颜色、不透明度。
- 位置合成：μ_i = Π⁻¹(ū_i + λ_xy Δu_i, d̄_i + λ_z Δz_i; K̂)。ū_i、d̄_i 是基础支撑位置和基础深度，λ_xy、λ_z 控制更新范围。
- 属性合成：g_i^(a) = ψ_a(ψ_a⁻¹(ḡ_i^(a)) + λ_a Δg_i^(a))，a ∈ {scale, color, opacity}。ψ_a 是该属性的激活函数，偏移加在激活的反函数（无约束）空间里，再过激活保证取值合法。

训练目标（第 3.4 节）：

- 渲染：Î_t = R(G; K_t, T_t)。R 是高斯渲染器，K_t、T_t 是目标相机内参和位姿。
- L_rgb = (1/|T|) Σ_{t∈T} ‖Î_t − I_t‖₁。
- L_perc = (1/|T|) Σ_{t∈T} Σ_l (‖φ_l(Î_t) − φ_l(I_t)‖₂² + γ‖G_l(Î_t) − G_l(I_t)‖_F²)。φ_l 是冻结 VGG 第 l 层特征，G_l 是对应 Gram 矩阵，γ = 10。
- L_reg = λ_scale·L_scale + λ_smooth·L_smooth。
- L_scale = (1/N) Σ_i [ReLU(ℓ_min − log s_i) + ReLU(log s_i − ℓ_max)]，[ℓ_min, ℓ_max] = [−8, −3]。
- L_smooth = (1/N) Σ_i (1/|N(i)|) Σ_{j∈N(i)} (‖log s_i − log s_j‖₁ + |α_i − α_j|)。N(i) 是 q_i 的邻近支撑点。
- 总目标：L = λ_rgb·L_rgb + λ_perc·L_perc + λ_reg·L_reg。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 两个：(1) 支撑点放哪——深度反投影后每个局部三角形的 3D 表面积，作为采样权重（面积大或倾斜强则多分）；(2) 属性怎么定——每个支撑点坐标处查询到的 DINO 语义特征与 CNN 纹理特征，门控融合后决定高斯属性残差 |
| 哪一端算得出（编码端/解码端/仅训练期） | 推理端一次前向全部算出；它没有编码端/解码端之分，单张输入图进来即生成高斯 |
| 有无侧信息 | 无码流概念故无侧信息；几何先验由冻结的 DepthPro（RGB 版）或 InfiniDepth-Metric 加每图 1500 个稀疏深度点（LiDAR 版）在推理端从输入算出（第 3.4 节） |
| 训练期还是编码期 | 面积分配与特征查询在推理期逐图完成；MLP 参数训练期学定、推理期冻结 |
| 和渲染质量的距离（直接测质量还是代理量） | 面积分配和深度先验都是代理量，不直接测质量；质量靠训练期目标视角渲染损失端到端约束 |

## 三学：证明与包装（创新怎么立住）

### 实验口径

- **体积/码率**：论文不涉及压缩，未报告表示的存储体积、码率、模型参数量（论文未报告）。
- **数据集与协议**：跨数据集零样本。训练只用 Hypersim 合成室内数据；评测用 ETH3D、ScanNet++、Tanks and Temples、DL3DV 四个真实数据集，均不参与训练。每个数据集 512 个源-目标视角对：固定随机种子选样，四个相机基线区间 [0, 0.5)、[0.5, 1)、[1, 2)、[2, ∞) 米各 128 对，要求源-目标视锥重叠大于 60%，过滤过大旋转和背向视角（第 4.1 节）。评测分辨率：ETH3D 2016×1344、ScanNet++ 1536×1024、Tanks and Temples 与 DL3DV 1920×1080。用源视角几何投影加形态学操作生成可见性掩码，所有方法同一掩码（第 4.1 节）。
- **和谁比**：RGB-only 设定比 SHARP、官方 Flash3D、Flash3D-DepthPro（把官方 Flash3D 的深度模型换成 DepthPro，控制深度后端变量）、LVSM、LagerNVS 共 5 个；RGB+深度传感器设定比 ADGaussian。InfiniSplat-RGB 与 SHARP 同用 DepthPro（第 4.1 节）。
- **指标**：PSNR、SSIM、LPIPS（第 4.1 节）。

### 主结果表抄录（Table 1，共 8 行，全表抄录）

列名：四个数据集各报 PSNR（dB，越高越好）/ SSIM（越高越好）/ LPIPS（越低越好），末三列为四数据集平均（LiDAR 组平均为 ETH3D、ScanNet++、Tanks-and-Temples 三数据集平均）。

| 方法 | ETH3D PSNR/SSIM/LPIPS | ScanNet++ PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DL3DV PSNR/SSIM/LPIPS | 平均 PSNR/SSIM/LPIPS |
| --- | --- | --- | --- | --- | --- |
| LVSM | 17.306/0.532/0.488 | 13.177/0.499/0.538 | 13.696/0.394/0.531 | 14.249/0.311/0.518 | 14.607/0.434/0.519 |
| LagerNVS | 18.130/0.598/0.462 | 16.749/0.634/0.395 | 16.844/0.661/0.371 | 18.118/0.510/0.571 | 17.460/0.601/0.450 |
| Flash3D | 18.546/0.874/0.239 | 17.375/0.782/0.350 | 15.952/0.674/0.326 | 17.502/0.622/0.395 | 17.344/0.738/0.328 |
| Flash3D-DepthPro | 17.917/0.853/0.257 | 17.561/0.775/0.367 | 12.168/0.652/0.410 | 17.602/0.642/0.405 | 16.312/0.731/0.360 |
| SHARP | 19.046/0.868/0.241 | 20.801/0.838/0.273 | 15.750/0.674/0.310 | 18.305/0.652/0.370 | 18.475/0.758/0.299 |
| InfiniSplat-RGB | 20.531/0.895/0.220 | 22.240/0.864/0.270 | 17.118/0.695/0.306 | 21.685/0.772/0.310 | 20.394/0.806/0.277 |
| ADGaussian | 12.752/0.749/0.337 | 10.474/0.565/0.506 | 13.523/0.604/0.412 | –（论文未报告） | 12.249/0.639/0.418 |
| InfiniSplat-LiDAR | 25.880/0.946/0.151 | 24.148/0.896/0.237 | 17.617/0.712/0.287 | –（论文未报告） | 22.548/0.851/0.225 |

正文引用的关键数字（第 4.2 节）：四数据集平均对 SHARP 提 PSNR +1.919、SSIM +0.048，LPIPS 降 0.022；对 Flash3D、Flash3D-DepthPro、LagerNVS 平均 PSNR 分别 +3.050、+4.082、+2.934。逐数据集对 SHARP 的 PSNR 增益：ETH3D +1.485、ScanNet++ +1.439、T&T +1.368、DL3DV +3.380。LiDAR 设定对 ADGaussian 提 PSNR +10.299、SSIM +0.212，LPIPS 降 0.193。

### 消融设计（注明表号）

- **逐组件开关**（Table 2，ETH3D + ScanNet++ 双数据集）：去学习更新（只渲染基础高斯）/ 去 DINO 分支 / 去 CNN 分支 / 去高斯正则 / 去几何引导采样（换回像素对齐支撑但保留隐式解码）/ 去隐式解码（换 DPT 逐像素解码器）。设计亮点：几何引导采样和隐式解码两个耦合组件被拆开单独消融，各自贡献可分。
- **支撑预算消融**（Table 3，ScanNet++）：0.5M/1.0M/1.5M/2.0M 支撑点，同时报表示推理时间和渲染时间，论证 1.5M 接近质量平台而 2.0M 只增加耗时（1.5M 为 22.240/0.864/0.270，2.0M 为 22.237/0.864/0.271）。
- **噪声鲁棒性**（Table 4，ETH3D）：对 1500 个稀疏深度提示加乘性噪声 d′ = d(1+ε)，ε 服从 N(0, σ²)，σ 取 0/1/3/5%，PSNR 从 25.880 平滑降到 23.595，无突然失效。
- **配套定性消融**：图 7（去学习更新变模糊）、图 8（去 DINO 出大洞、去 CNN 变糊）、图 9（去正则出现半透明退化）、图 10（像素对齐支撑出裂缝、DPT 解码器出破洞）。

### 贡献列表（原文照抄 + 逐条标注）

1. "A surface-aligned Gaussian representation for robust large-baseline NVS. We introduce a geometry-guided support sampling strategy that instantiates this representation by placing 2D supports according to depth-induced local surface structures rather than fixed image-grid locations. By aligning support locations with geometric priors, the predicted primitives can better assemble into coherent surfaces, leading to more stable rendering results under large-baseline viewpoint changes." —— [机制] 表面对齐高斯表示：按深度导出的局部表面结构（而非固定图像网格）放置 2D 支撑点，使高斯能拼成连贯表面。
2. "Implicit Gaussian decoding for single-image 3DGS. We formulate Gaussian attribute prediction as query-conditioned implicit decoding over sampled supports and queried image features. This decoder turns geometry-guided supports into Gaussian primitives and allows a shared prediction function to operate on support sets with different densities and spatial arrangements." —— [机制] 隐式高斯解码：把属性预测写成对采样支撑点和查询特征的条件式隐式解码，一个共享函数处理任意密度和空间排布的支撑集。

### 讲故事方式

主线一句话：单图前馈 3DGS 的瓶颈不是渲染锐度，而是"高斯长在像素网格上"；把生成位置挪到深度几何引导的表面对齐支撑点、用共享隐式 MLP 解码属性，大视角偏移下的结构稳定性就上来了。

最有力的是图 4：RGB 渲染和法向图并排对比，法向图直接暴露高斯是否在空间中拼成连续表面——SHARP 的碎片化表面和本文的连续法向一眼可辨，把"表面对齐"这个抽象主张变成看得见的证据；再配合表 1 四个数据集全指标领先，主张的论证链就完整了。

## 四学：原文关键段翻译

### 方法节核心段（第 3.1 节）

> **原文（第 3.1 节）**："The two core stages are coupled by design. Geometry-guided sampling breaks free from the pixel grid but produces an irregular layout that per-pixel regression cannot handle; implicit decoding provides the flexibility to operate on arbitrary layouts but would lack surface awareness without the geometric scaffold provided by sampled supports. Each stage necessitates the other: without the decoder, irregular supports cannot be turned into complete Gaussians; without geometry-guided supports, the decoder reduces to operating on an unstructured point set with no surface prior."
>
> **译文**：两个核心阶段在设计上是相互耦合的。几何引导采样摆脱了像素网格，但产出的是逐像素回归处理不了的不规则布局；隐式解码提供了在任意布局上操作的灵活性，但若没有采样支撑点提供的几何支架，它会缺少表面感知。每个阶段都以对方为前提：没有解码器，不规则支撑点变不成完整高斯；没有几何引导的支撑点，解码器就退化成在没有表面先验的无结构点集上操作。

### 方法节核心段（第 3.2 节）

> **原文（第 3.2 节）**："The role of this sampling strategy is not simply to increase the number of points, but to change the support domain of Gaussian generation. Because the base Gaussians are sampled from depth-induced local surface patches, the decoder can learn Gaussian placement, shape, and appearance on supports that better align with the input geometry. For large planes, slanted surfaces, object boundaries, and regions with depth variation, this flexible query sampling can reduce the discretization limitation introduced by fixed pixel grid points, making the generated Gaussians easier to organize into spatially coherent structures rather than a set of unrelated point-like splats."
>
> **译文**：这个采样策略的作用不只是增加点的数量，而是改变高斯生成的支撑域。因为基础高斯是从深度导出的局部表面块上采样出来的，解码器可以在与输入几何更对齐的支撑点上学习高斯的放置、形状和外观。对大平面、倾斜表面、物体边界和有深度变化的区域，这种灵活的查询采样能减弱固定像素格点带来的离散化限制，让生成的高斯更容易组织成空间连贯的结构，而不是一堆互不相关的点状泼溅。

### 主结果表述段（第 4.2 节）

> **原文（第 4.2 节）**："Table 1 reports the quantitative comparison between InfiniSplat and the baselines. The table reports PSNR, SSIM, and LPIPS as separate metric columns. InfiniSplat-RGB achieves the highest PSNR, highest SSIM, and lowest LPIPS on all four RGB-only datasets. On the four-dataset average, InfiniSplat-RGB reaches 20.394/0.806/0.277, improving over SHARP by +1.919 PSNR and +0.048 SSIM while reducing LPIPS by 0.022. Compared with Flash3D, Flash3D-DepthPro, and LagerNVS, InfiniSplat-RGB improves the average PSNR by +3.050, +4.082, and +2.934, respectively, while reducing average LPIPS by 0.051, 0.083, and 0.173. These results show that the improvement of InfiniSplat does not come from a single dataset or a single baseline, but remains consistent across indoor scans, large outdoor scenes, and DL3DV open-world scenes."
>
> **译文**：表 1 报告了 InfiniSplat 与各基线的定量比较，PSNR、SSIM 和 LPIPS 作为独立指标列报告。在全部四个 RGB-only 数据集上，InfiniSplat-RGB 取得最高 PSNR、最高 SSIM 和最低 LPIPS。四数据集平均，InfiniSplat-RGB 达到 20.394/0.806/0.277，比 SHARP 提升 PSNR +1.919、SSIM +0.048，同时 LPIPS 降低 0.022。与 Flash3D、Flash3D-DepthPro 和 LagerNVS 相比，InfiniSplat-RGB 的平均 PSNR 分别提升 +3.050、+4.082、+2.934，平均 LPIPS 分别降低 0.051、0.083、0.173。这些结果表明 InfiniSplat 的提升不是来自单一数据集或单一基线，而是在室内扫描、大尺度室外场景和 DL3DV 开放世界场景上都保持一致。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | 组合式创新：坐标查询隐式解码（LIIF、InfiniDepth 一脉）加查询式高斯解码（TokenGS、C3G 已把高斯预测从像素解耦）都不是新的；新的地方是论证两者在单图设定下互为前提并绑定成框架，加上按局部表面积分配支撑点。相对 2026-09 已有工作属于有效但非开范式的推进 |
| 证据强度 | 高 | 四个真实数据集各 512 对的跨数据集零样本协议写得清楚（固定种子、基线分档、视锥重叠过滤、统一掩码）；8 个方法行含控制深度后端的对照（Flash3D-DepthPro）；组件、预算、噪声三路消融齐全。扣分项：正文只给项目页（https://zju3dv.github.io/InfiniSplat）未给代码链接，参数量与存储体积未报告 |
| 对期刊版的威胁度 | 低 | 它做单图生成不做压缩，不碰渐进码流、零侧信息熵编码、内容自适应量化三条主张中的任何一条；"按解码端可重算的量分配资源"的思想与我们零侧信息契约同构，是启发不是撞车 |

## 对 DCCA-GS 的可借鉴点

1. **按可重算量分配资源的设计律**（第 3.2 节的面积权重 w_t）：它的支撑点分配量（深度反投影后的局部 3D 表面积）完全由推理端从输入重算，不需要传输任何分配表——和我们"解码端能重算的才可以用"是同一条设计原则。可做的实验：把我们的训练期渲染敏感度监督换成（或级联上）解码端可重算的几何代理量（如锚点邻域深度方差），验证零侧信息预算分配能否省掉敏感度监督那一支。预期收益是省一份监督开销；风险是几何代理量离渲染质量的距离比敏感度远，需对照率失真曲线确认。
2. **预算消融的论证方式**（Table 3）：预算、质量、推理时间、渲染时间四列一起报，选"接近质量平台的最小预算"作默认值。可直接照搬到我们码率档实验：报每个画质档"接近质量平台的最小比特"，论证渐进阶梯的档位划分。
3. **有界残差更新**（第 3.3 节）：先给几何初始化，MLP 只预测激活函数反函数空间内的有界偏移。这与我们增强层"在基础值附近补精度"的思路同构，可对照它在激活空间加偏移与直接量化残差两种做法；但注意我们有学习式概率熵模型零增益的明确负结果，此条只作低优先级参考。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 表示与解码方式 | 单图前馈生成 1.5M 个支撑点高斯，共享 MLP 在任意坐标查询双分支特征、预测 14 维有界残差（第 3.3 节，每图支撑数见第 4.1 节） | 锚点式 3DGS，锚点属性量化后熵编码进码流，解码端熵解码还原（HAC++ 底座） | 它解决"从一张图生成哪些高斯"，我们解决"训练好的高斯怎么压小"；它的解码是生成式的，不受码率约束，与我们的率失真优化目标不同 |
| 资源分配怎么定 | 按深度反投影后的局部 3D 表面积比例分配支撑点（第 3.2 节 w_t），推理端可重算 | 训练期渲染敏感度监督加解码端可重算的内容复杂度量化，共享一个 8→32→3 小 MLP，零侧信息 | 两者都遵守"分配量必须可在本地重算"；它的分配决定生成密度，我们的分配决定码流比特；可试探把它的几何代理量引入我们的分配 |
| 侧信息 | 几何先验由冻结的 DepthPro 或 InfiniDepth-Metric（加 1500 个稀疏深度点）在推理端从输入图算出，无需传输（第 3.4 节） | 零侧信息契约：量化步长与编码上下文全部由解码端从已解码数据重算 | 两边都坚持"能重算就不传"；差别在它重算的起点是输入图像，我们的起点是已解码的码流字节，我们的约束更紧 |
| 量化/熵编码/渐进可截断 | 该论文不涉及（无码流概念，不做压缩） | 嵌套质量阶梯、层级序文件布局、二值分解区间编码、层条件熵编码 | 该论文不涉及 |
