# 《Non-Uniform Quantisation for 3DGS Compression》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2608.28272 |
| 发表 | arXiv 预印本（v1，2026-08-28），定位为 MPEG 3DGS 标准化的正式技术贡献 |
| 阅读材料 | 全文（.lit-cache/txt/2608.28272.txt）；公式与表格版式已逐项对照 PDF（pymupdf）核验 |
| 与 DCCA-GS 的关系 | 同类竞品（量化方向）：面向 MPEG 标准的编码端重要性加权非均匀量化，与我们"解码端重算的嵌套质量阶梯量化"是同一个问题的两条相反路线 |

> 一句话定位：给 G-PCC/V-PCC 这两个 3DGS 标准码流换掉均匀量化——按"每个高斯对渲染像素的贡献平方和"加权做 Lloyd-Max 非均匀量化，把量化表本身再用第二级量化压小，量化后落在同一体素的高斯按重要性合并；省码率靠"比特花在影响画面的高斯上"加"减少高斯个数"，其中合并部分已被 V-PCC Amd1 for GS 正式采纳（第 1 节、第 7 节）。

作者：Bert Van Hauwermeiren、Adrian Munteanu（布鲁塞尔自由大学），Patrice Rondao Alface（Nokia 比利时）。开源：未找到（论文未给代码链接）。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

第 1 节原话："Both codecs apply uniform quantisation to the 3DGS geometry and attributes before entropy coding. This approach is suboptimal, as it fails to account for the non-uniform distribution of Gaussian data and the varying perceptual importance of individual Gaussians."（两个标准编码器都在熵编码前对几何和属性做均匀量化，这忽略了高斯数据的非均匀分布和单个高斯之间感知重要性的差异。）

我的解释：MPEG 把 3DGS 当点云编（G-PCC 走八叉树加 RAHT，V-PCC 走分块投影加视频编码，见第 1 节），均匀量化对每一个数一视同仁。但 3DGS 里有的高斯几乎不影响画面，有的高斯决定画面，把比特平均分给两者是浪费。第二层矛盾在第 2 节末尾点明：学术上更强的压缩方法（Scaffold-GS、HAC 这类）都要改表示、改渲染器，而 "these methods require modified representations and renderers, which are not allowed in the current G-PCC and V-PCC amendments"——标准化场景下不允许改表示。所以它把问题收窄成：不改表示、不改熵编码后端，只动量化这一个环节，能拿到多少收益。

### 支撑观察（直觉 / 数据 / 失败实验）

- 渲染误差可以分解到每个高斯：第 2 节式 1 的 alpha 混合公式说明像素颜色是各高斯颜色的加权和，第 3.1 节由此推出每个高斯的量化误差对总渲染误差的贡献系数（式 2）。这是从渲染方程推导出来的，不是经验观察。
- 高斯位置量化后同体素冗余真实存在且数量可观：Table 6，MPEG Scenes 平均高斯数从不合并的 495486.5 降到带判据合并的 376527.5，减少约 24%。
- 小 MLP 能替代全场景渲染来估重要性：Table 3，MLP 近似权重在 MPEG Scenes G-PCC 上等质量省码率 48.08%，反而好于精确渲染权重的 38.14%；Table 4，在 MPEG Objects G-PCC 上渲染法预处理要 2131.01 秒，MLP 法只要 13.72 秒。
- 位置位深存在"再低就崩"的阈值：第 5.1 节文字加 Table 2，位深从 18 降到 14/12 时等质量码率大幅下降，但 MPEG Scenes V-PCC 的 12-bit 行在 RGB-PSNR 列变成 +5.75（码率反升）、YUV-PSNR 列出现 nan（该配置下质量曲线够不到锚点的质量区间，Bjontegaard 插值失败），说明画质已经崩掉。

### 显然的路为什么没走

- 显然路一：改表示、改训练（锚点生成、哈希网格上下文那类，RD 性能更强）。不走，第 2 节明确说 MPEG 现行修正案不允许改表示和渲染器——这是标准化论文的硬约束，和我们的零侧信息契约是同一种"先把游戏规则写死再解题"的思路。
- 显然路二：像 Adaptive Voxelization [26] 那样自适应体素化后再微调（fine-tuning）。不走微调，第 3.3 节开头说 "Rather than relying on simple attribute averaging followed by costly fine-tuning iterations [26], we propose a series of importance-weighted merging strategies designed for high accuracy without retraining"——合并免训练，编码即用。
- 显然路三：动标准熵编码器内部（改 RAHT、改视频编码配置）。不走，第 7 节声明方法"entirely agnostic to the underlying entropy coding engine"。动熵编码后端在标准组织里阻力最大，只动量化前置环节政治上可行。
- 显然路四：直接剪掉不重要的高斯（pruning）。不作为主路线，第 6 节把合并定位成剪枝的"轻量替代"和互补项：剪枝丢弃低影响基元，合并保留它们、只消除同格冗余。

## 二学：机制（洞察怎么变成方法）

### 核心流程

整个管线在编码端跑，解码端不做任何训练、渲染或重算（Figure 1 给了全流程图）。

**编码端逐步骤：**

1. **重要性估计**（第 3.1 节，式 2、式 3）。输入：训练好的 3DGS 模型。精确做法：全场景渲染一遍，累计每个高斯在所有像素上 alpha 混合贡献的平方和 w_g。快速做法：一个 205 参数的小 MLP，输入是三个 scale 分量的乘积、opacity、位置（式 3），输出近似重要性。MLP 离线训练，跨场景通用（第 4.1.1 节用留一法验证：每个数据集留一个场景训练 MLP，其余场景只做评测）。不使用球谐系数和旋转（第 3.1 节"On the choice of inputs"段：视角近似均匀分布时旋转不影响重要性；靠场景中心的高斯更可能是前景）。
2. **位置量化（即体素化）**（第 3.2 节）。对高斯位置做重要性加权的 Lloyd-Max 非均匀量化，量化档数对应 14 bit（正面场景）/ 11 bit（物体）（第 4.1.4 节；Table 2 加粗的选定位深是 V-PCC 场景 14、G-PCC 场景 12、V-PCC 物体 11、G-PCC 物体 9，与第 4.1.4 节的单一说法不完全一致，如实记录）。输出：每个高斯的量化位置、量化器边界和重建值表。
3. **同体素合并**（第 3.3 节，式 4-8）。落在同一体素的高斯，先按不相似度（式 8）筛：低于阈值才允许合并。合并的三种几何方案：加权参数平均（四元数用 Markley 等人的优化式平均，式 5，避免 q 与 -q 双覆盖问题）、加权协方差空间平均（式 6，对协方差矩阵加权平均后做特征值分解拆回 scale 和旋转）、加权对数欧氏平均（式 7，对称正定矩阵的黎曼均值近似）；opacity 和球谐系数直接加权平均。消融（第 5.3 节，Table 5）选定加权参数平均。
4. **属性量化**（第 3.2 节）。对 scale、rotation、opacity、SH 各通道分别做加权 Lloyd-Max（权重还是那个重要性）。每套量化器的重建值表本身是浮点元数据，直接传要 32n bit（n 为档数），是码率瓶颈；于是把相邻重建值的差分 d_i = y_i − y_{i−1} 当作新信号，再做一次加权 Lloyd-Max（第二级，2 bit，第 4.1.4 节）。码流里只存：首个重建值 y_0、K 个差分重建值 {d_k}、每个重建值的差分索引 k_i。解码端按 y_i = y_{i−1} + d_{k_i} 递推出整张表。元数据从 32n bit 降到 32(K+1) + log2(K)·n bit（第 3.2 节给出 K=4、n 在 2^10 到 2^18 之间时的量级估计）。
5. **熵编码**。量化后的符号交给现成的标准后端：G-PCC（八叉树几何 + RAHT 属性，release_mpeg151）或 V-PCC（分块投影 + HM-16.20 HEVC Main 10，mpeg152-gs-anchor）（第 4.1.1 节）。本文不改后端。

**解码端：** 读码流，用码流里的量化器元数据重建量化表（递推公式），反量化，走标准解码器和标准渲染器。没有解码端重算。

### 关键公式（按原文抄，注明编号）

符号抽取可能损坏，以下均按 txt 抄录并已对照 PDF 页面核验；个别符号在 txt 中显示为乱码大算符的，已按 PDF 更正并注明。

式 1（第 2 节），像素颜色的 alpha 混合与逐高斯贡献：

> C(p) = Σ_{g_i} C_{g_i} · α_{g_i} · Π_{j=1}^{i−1} (1 − α_{g_j}) = Σ_g w_{p,g} · C_g　　(1)

其中 w_{p,g} 是高斯 g 在像素 p 的贡献（自身密度乘以排在它前面所有高斯的透射率）。原文公式以 PDF 为准。

第 3.1 节未编号推导，渲染图像量化误差 D 展开为逐高斯项加交叉项：

> D = (1/N) Σ_p (C(p) − Ĉ(p))² = (1/N) Σ_p (Σ_g w_{p,g} ε_g)² = (1/N) Σ_p Σ_g w²_{p,g} ε²_g + (1/N) Σ_p Σ_{g≠g'} w_{p,g} w_{p,g'} ε_g ε_{g'}

其中 ε_g = C_g − Ĉ_g 是高斯 g 颜色参数的量化误差。引用高码率量化理论 [8]（Gersho 与 Gray）假设量化误差独立、零均值，交叉项期望为零，得到：

> E[D] ≈ (1/N) Σ_g ( Σ_p w²_{p,g} ) E[ε²_g]

由此得每高斯重要性（式 2）：

> w_g = Σ_p w²_{p,g} = Σ_p α_i² Π_{j=1}^{i−1} (1 − α_j)²　　(2)

（原式下标即如此印刷：α_i 指该高斯自己的密度，α_j 指排在它之前的高斯密度；原文公式以 PDF 为准。）注意 w_g 是"贡献的平方和"，比贡献本身更强调同时出现在多个像素上的高斯。

式 3（第 3.1 节），MLP 近似重要性。txt 中大算符损坏，按 PDF 核正为连乘号：

> ŵ_g = MLP( Π_{i=1}^{3} s_{g,i} , o_g , μ_g )　　(3)

即输入是三个 scale 分量的乘积（不是三个分量各自输入）、opacity、位置，共 5 个标量。

第 3.2 节未编号公式，加权 MSE 目标与加权质心更新（y 为重建值，b 为边界，w(x) 为该标量值所属高斯的重要性）：

> D = E[ w(x)(x − Q(x))² ] = Σ_{i=0}^{n−1} Σ_{x∈[b_i, b_{i+1})} w(x)(x − y_i)²
>
> y_i = Σ_{x∈[b_i,b_{i+1})} w(x)·x ／ Σ_{x∈[b_i,b_{i+1})} w(x)

第二级量化：对差分 d_i = y_i − y_{i−1} 再做一次加权 Lloyd-Max，额外失真分解为（原文中间步骤以省略号带过）：

> D̂ = Σ_{k=0}^{K−1} Σ_{d_i∈[b'_k, b'_{k+1})} W(d_i)(d_i − d̂_k)²，其中 W(d_i) = Σ_{x∈[b_i,b_{i+1})} w(x)

即每个差分的权重是它所在原始 bin 内全部取值重要性之和。解码端重建量化表：

> y_i = y_0（i=0）；y_i = y_{i−1} + d̂_{k_i}（i>0）

元数据开销：从 32n bit 降到 32(K+1) + log2(K)·n bit（第 3.2 节）。

第 3.3 节，合并相关的四个式子。式 4 是被否定的朴素四元数线性平均（不满足 q 与 −q 等价的反对称性）：

> q_linear = (1/Σ_i w_i) Σ_i w_i q_i　　(4)

式 5 是采用的 Markley 等人优化式四元数平均：

> q = argmax_q qᵀ ( Σ_i w_i q_i q_iᵀ ) q　　(5)

式 6 加权协方差空间平均（Σ = R S Sᵀ Rᵀ，平均后做特征值分解拆出 scale 与旋转）：

> Σ = (1/Σ_i w_i) Σ_i Σ_i · w_i　　(6)

式 7 加权对数欧氏平均（对称正定矩阵黎曼均值的近似）：

> Σ = exp( (1/Σ_i w_i) Σ_i log(Σ_i) · w_i )　　(7)

式 8 合并判据（不相似度低于阈值才合并；λ1 到 λ4 的取值论文未报告）：

> dissimilarity_{i,j} = λ1‖SH_i − SH_j‖ + λ2‖Σ_i − Σ_j‖ + λ3|o_i − o_j| + λ4·min{w_i, w_j}　　(8)

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 每个高斯的重要性 w_g：它在所有像素上 alpha 混合贡献的平方和（式 2），数学上等于"该高斯量化误差对总渲染误差期望的放大系数" |
| 哪一端算得出（编码端/解码端/仅训练期） | 仅编码端需要。精确值要全场景渲染一遍；实用版用 205 参数小 MLP 从（scale 乘积、opacity、位置）预测，跨场景通用（第 3.1、4.1.4 节）。解码端完全不用这个信号 |
| 有无侧信息 | 重要性本身不进码流、解码端不使用，所以不构成侧信息；但量化器重建值表（y_0、K 个差分重建值、每符号差分索引）必须写码流——这是论文认账的开销，并用第二级量化专门去压它（第 3.2 节） |
| 训练期还是编码期 | 编码期。MLP 离线训练好（留一法，第 4.1.1 节），对每个新场景只需一次前向推理；管线本身不修改 3DGS 训练 |
| 和渲染质量的距离（直接测质量还是代理量） | 直接挂在渲染过程上：w_g 从渲染方程的逐像素贡献推导，最小化的就是渲染误差的期望（高码率假设下），比"参数误差""几何体积"这类代理近得多；MLP 版是对这个直接量的二次近似（Table 3 证明损失很小甚至为正收益） |

### 与码流组织相关的细节

- 渐进层怎么划分：不做渐进。四个评测码率点是四套独立配置各自编码出的独立码流（第 4.1.1 节："Testing is conducted across four standardised ratepoints"，各码率点只保证预处理一致），切换画质档要换文件。
- 文件布局：未描述，直接使用 G-PCC/V-PCC 标准码流容器；量化器元数据如何嵌入标准属性字段的细节论文未报告。
- 解码端重算哪些量：没有。解码端按码流里的量化器元数据递推重建量化表（第 3.2 节），反量化后走标准解码，不重算任何内容相关量。
- 侧信息：量化器元数据必须写码流（32(K+1) + log2(K)·n bit，第 3.2 节）；重要性 MLP 不传、解码端不用。
- 可截断性：无。该论文不涉及码流截断或单文件多画质档。

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 平台与协议：MPEG 3DGS-PCC 测试平台，按共通测试条件（CTC）执行（第 4.1.1 节）。两个标准熵编码后端：G-PCC（release_mpeg151，八叉树几何 + RAHT 属性编码）、V-PCC（mpeg152-gs-anchor，HM-16.20 HEVC Main 10 分块视频编码）。四个标准码率档，各档熵编码配置不同、预处理管线一致。
- 体积/码率怎么算：依赖标准参考软件输出的码流比特数，论文未报告逐字段体积分解。质量指标：PSNR、SSIM、IVSSIM、LPIPS（第 4.1.3 节）；RD 收益用 Bjontegaard 等质量码率变化（BD-rate）表示，锚点是标准自带的均匀量化配置。
- 表格列含义说明（重要）：消融表（Table 2/3/5/6/7/8）的"BD-Rate [%]"是跨五个指标列的总表头，每列数值是"用该质量指标算出的等质量码率变化"，负数代表省码率。这是笔者按 PDF 表头跨列版式确认的读法（Table 2 渲染图核对），正文没有逐列解释；这也解释了 Table 2 中 V-PCC 场景 12-bit 行 YUV-PSNR 列出现 nan（该配置够不到锚点质量区间，无法插值）。
- 数据集：MPEG 官方 3DGS 数据集两类——大尺度正面场景；高细节静态物体与人像（举了 bartender_stable、cinema_stable、lego_bugatti、plant 等，第 4.1.1 节）。因为管线带机器学习组件，采用留一法：每个数据集留一个场景训 MLP，其余只评测。
- 基线：均匀量化锚点（位置 18 bit；属性 G-PCC 12 bit、V-PCC 10 bit，第 4.1.2 节）；Adaptive Voxelization [26]（体素化阈值参数设 20）；FlexGaussian [24]（原本不兼容有损熵编码，按原样评测，只有一个码率点）。
- 训练/实现配置（第 4.1.4 节）：体素化 14 bit（正面场景）/ 11 bit（物体）；第二级量化 2 bit；重要性 MLP 两个隐藏层（16 和 6），共 205 可训练参数（按输入 5 维核验参数量吻合：5×16+16 + 16×6+6 + 6×1+1 = 205）；Adam，学习率 0.01，50 轮。硬件 i9-13900 + RTX 4090。

### 主结果表抄录（注明表号）

主要 RD 对比在 Figure 4（曲线图，无汇总数字）；量化数字证据集中在消融表。下抄 Table 2 全表（位深消融，即本文最核心的数字证据）。

Table 2（体素化位深消融；BD-Rate [%] 为总表头，五列分别为用该指标算出的等质量码率变化，负 = 省码率；加粗为作者标注）：

| 数据集 | 编码器 | Bits | RGB-PSNR | YUV-PSNR | YUV-SSIM | YUV-IVSSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPEG Scenes | V-PCC | 18 | -5.63 | -4.44 | -7.93 | -0.66 | -4.98 |
| MPEG Scenes | V-PCC | 16 | -14.44 | -13.55 | -18.51 | -10.65 | -16.26 |
| MPEG Scenes | V-PCC | 14 | **-28.27** | -28.04 | **-27.65** | -31.54 | **-30.01** |
| MPEG Scenes | V-PCC | 12 | 5.75 | **-30.60** | nan | **-54.96** | -17.18 |
| MPEG Scenes | G-PCC | 18 | -11.78 | 1.91 | -29.31 | -6.33 | -8.25 |
| MPEG Scenes | G-PCC | 16 | -23.18 | -13.82 | -43.56 | -20.79 | -24.39 |
| MPEG Scenes | G-PCC | 14 | -48.08 | -40.78 | -59.72 | -44.50 | -44.14 |
| MPEG Scenes | G-PCC | 12 | **-71.96** | **-69.30** | **-75.56** | **-71.51** | **-70.12** |
| MPEG Objects | V-PCC | 12 | -43.38 | -43.13 | -40.72 | -45.72 | -41.89 |
| MPEG Objects | V-PCC | 11 | **-44.17** | -43.92 | **-41.68** | -48.51 | **-43.44** |
| MPEG Objects | V-PCC | 10 | -43.73 | -43.23 | -40.42 | -51.14 | -43.22 |
| MPEG Objects | V-PCC | 9 | -43.82 | **-44.33** | -20.26 | **-54.57** | -42.83 |
| MPEG Objects | G-PCC | 12 | -28.76 | -27.98 | -26.33 | -29.53 | -28.69 |
| MPEG Objects | G-PCC | 11 | -32.95 | -32.13 | -30.55 | -34.29 | -32.97 |
| MPEG Objects | G-PCC | 10 | -37.36 | -36.72 | -34.23 | -39.07 | -37.90 |
| MPEG Objects | G-PCC | 9 | **-43.53** | **-43.08** | **-38.94** | **-47.08** | **-45.06** |

Table 3（权重来源消融；数值含义与 Table 2 相同）：

| 数据集 | 编码器 | 权重 | RGB-PSNR | YUV-PSNR | YUV-SSIM | YUV-IVSSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPEG Scenes | V-PCC | Uniform | -11.42 | -11.24 | -10.93 | -14.96 | -14.50 |
| MPEG Scenes | V-PCC | Rendered | -21.07 | -20.42 | -18.49 | -24.66 | -22.15 |
| MPEG Scenes | V-PCC | Approximate | -28.27 | -28.04 | -27.65 | -31.54 | -30.01 |
| MPEG Scenes | G-PCC | Uniform | -29.74 | -22.89 | -52.38 | -28.12 | -35.45 |
| MPEG Scenes | G-PCC | Rendered | -38.14 | -30.14 | -56.08 | -34.36 | -38.83 |
| MPEG Scenes | G-PCC | Approximate | -48.08 | -40.78 | -59.72 | -44.50 | -44.14 |
| MPEG Objects | V-PCC | Uniform | -43.78 | -43.54 | -41.32 | -47.99 | -43.24 |
| MPEG Objects | V-PCC | Rendered | -44.26 | -43.95 | -41.37 | -48.39 | -43.17 |
| MPEG Objects | V-PCC | Approximate | -44.17 | -43.92 | -41.68 | -48.51 | -43.44 |
| MPEG Objects | G-PCC | Uniform | -32.43 | -31.56 | -30.11 | -33.47 | -32.52 |
| MPEG Objects | G-PCC | Rendered | -32.87 | -32.08 | -30.50 | -33.22 | -33.02 |
| MPEG Objects | G-PCC | Approximate | -32.95 | -32.13 | -30.55 | -34.29 | -32.97 |

Table 5（合并方式消融）共 5 种方式 × 4 个数据集-编码器组合 = 20 行，此处节选 3 种关键方式；另两种（加权协方差、加权对数欧氏）在正文中概述：

| 数据集 | 编码器 | 合并方式 | RGB-PSNR | YUV-PSNR | YUV-SSIM | YUV-IVSSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPEG Scenes | V-PCC | No Merging | -10.70 | -10.25 | -11.30 | -13.34 | -14.82 |
| MPEG Scenes | V-PCC | Basic Averaging | -9.08 | -11.03 | -17.37 | -20.93 | -16.25 |
| MPEG Scenes | V-PCC | Weighted Parameter Averaging | -28.27 | -28.04 | -27.65 | -31.54 | -30.01 |
| MPEG Scenes | G-PCC | No Merging | -24.05 | -19.86 | -52.36 | -22.79 | -34.84 |
| MPEG Scenes | G-PCC | Basic Averaging | -40.24 | -34.50 | -59.50 | -38.04 | -42.37 |
| MPEG Scenes | G-PCC | Weighted Parameter Averaging | -48.08 | -40.78 | -59.72 | -44.50 | -44.14 |
| MPEG Objects | V-PCC | No Merging | -43.88 | -43.66 | -41.19 | -48.07 | -43.21 |
| MPEG Objects | V-PCC | Basic Averaging | -41.59 | -41.36 | -40.97 | -47.67 | -42.23 |
| MPEG Objects | V-PCC | Weighted Parameter Averaging | -44.17 | -43.92 | -41.68 | -48.51 | -43.44 |
| MPEG Objects | G-PCC | No Merging | -32.60 | -31.70 | -30.09 | -33.85 | -32.62 |
| MPEG Objects | G-PCC | Basic Averaging | -32.47 | -31.65 | -30.35 | -33.53 | -32.73 |
| MPEG Objects | G-PCC | Weighted Parameter Averaging | -32.95 | -32.13 | -30.55 | -34.29 | -32.97 |

节选外两种方式的结论（Table 5 其余行）：在 MPEG Scenes V-PCC 上加权协方差最差（RGB 列 -6.26），加权对数欧氏居中（-13.62）；G-PCC 两类场景下协方差与对数欧氏都明显差于加权参数平均。

Table 8（逐属性开关）共 20 行，此处节选 Positions 行（论文加粗标注的最大收益来源）：

| 数据集 | 编码器 | 属性 | RGB-PSNR | YUV-PSNR | YUV-SSIM | YUV-IVSSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPEG Scenes | V-PCC | Positions | -22.30 | -22.71 | -23.24 | -29.73 | -25.16 |
| MPEG Scenes | G-PCC | Positions | -37.68 | -33.87 | -55.73 | -35.50 | -28.25 |
| MPEG Objects | V-PCC | Positions | -44.79 | -44.60 | -44.37 | -50.73 | -46.05 |
| MPEG Objects | G-PCC | Positions | -21.79 | -21.81 | -22.20 | -22.73 | -22.52 |

Table 8 其余属性的走向（正文第 5.5 节加数字）：其余属性在 V-PCC 下收益小得多（如 MPEG Scenes V-PCC 球谐 -6.60、scale +0.78、rotation +1.36、opacity -0.11），在 G-PCC 下正负随机（如 MPEG Scenes G-PCC 球谐 +6.14、opacity +14.81；MPEG Objects G-PCC 球谐 +9.85、scale +10.06；正值代表等质量下码率反升，即对该属性单独开非均匀量化是亏的；例外是 G-PCC 的 rotation，MPEG Scenes 下 -32.74）。

正文引用的其他关键数字（标出处）：

- 复杂度（Table 1）：MPEG Scenes V-PCC 下，均匀量化的预处理/编码/解码/后处理为 11.77 / 759.82 / 4.95 / 10.78 秒，本文方法为 116.18 / 351.63 / 4.08 / 9.20 秒——预处理贵 10 倍，编码反而更快（合并减了基元数），解码持平。第 4.2 节与第 6 节据此主张"复杂度移到编码端，解码端实时性不受影响"。
- 合并判据的隔离实验（Table 6）：MPEG Scenes G-PCC 下不合并 -24.06、带判据合并 -48.08、全部合并 -44.21（RGB 列），三者的平均高斯数分别为 495486.5 / 376527.5 / 375847.5——带判据比全部合并只多保留 680 个高斯，但等质量码率好约 4 个百分点，说明"哪些高斯不许合并"的判据本身有独立价值。
- 第二级量化位深（Table 7 + 第 5.4 节）：2 bit 基本最优（MPEG Scenes：V-PCC -28.27、G-PCC -48.08），1 bit 也能拿到大部分收益（-24.78 / -45.68），3、4 bit 无进一步收益。

### 消融设计（注明表号）

值得学走的对照设计：

- 位深扫描（Table 2）：每个编码器扫 4 个位深取值，配合"选定与均匀量化画质相当的最低位深"的选值规则（第 5.1 节）。这把"非均匀量化的收益"和"降位深的收益"融在同一张表里，读者能看出非均匀化把标准强制的 18 bit 假设压到 12-14 bit。
- 三种权重来源同台对照（Table 3 + Table 4）：均匀权重（隔离出加权的净贡献）、精确渲染权重（上限）、MLP 近似（实用配置），并同时给四段耗时（Table 4）。这是"信号怎么算"的完整代价-收益对照，比只报最终配置有说服力。
- 合并方式五选一（Table 5）+ 判据开关与高斯个数并列（Table 6）：Table 6 把"合并了多少高斯"（# Splats 列）和"质量收益"放在一起，做到同基元数下比较判据的净贡献。
- 逐属性开关（Table 8）：每次只对一个属性开非均匀量化，定位收益来源（位置主导）。这个设计我们做分字段粗化档位时可以直接套用。

### 贡献列表（原文照抄 + 逐条标注）

1. "A novel non-uniform quantisation scheme designed to minimise the weighted reconstruction error. By integrating an importance metric, we prioritise high-impact Gaussians to optimise the rate-distortion tradeoff." —— [机制] 重要性加权的非均匀量化（几何+属性），目标从参数误差换成加权重建误差。
2. "A comparison of several novel importance-weighted merging approaches for Gaussians that become co-located after voxelisation." —— [工程] 体素化后同格高斯的多种重要性加权合并方案的设计与对比（引言注明该贡献已被 V-PCC Amd1 for GS 采纳）。
3. "An extensive performance analysis conducted in accordance with MPEG's 3DGS common test conditions (CTC). … The weighted merging contribution has already been adopted in the V-PCC Amd1 for GS." —— [工程] 按 MPEG CTC 的系统评测；合并贡献已实际写入 V-PCC Amd1（第 1 节）。

### 讲故事方式

主线一句话：标准里的均匀量化是白给的低效；把渲染贡献当权重、把量化表压小、把同格高斯合并，不碰标准和渲染器也能白捡两到七成码率。最有力的一张表是 Table 2：G-PCC 场景 12 bit 下等质量省 71.96%，直接击穿"位置必须 18 bit"的标准锚点假设；最有力的一组证据是 Table 3 配 Table 4——205 个参数的小 MLP 既替掉了 2131 秒的全场景渲染，还比精确权重更好，这让整个方案在标准化流程里真正可用。Figure 4 的 RD 曲线是给标准会议看的全景图，但没有给汇总 BD-rate 数字表，与基线的对比数字要靠读者从消融表间接推断，这是包装上的明显缺口。

## 四学：原文关键段翻译（深读层必填）

### 方法节核心段（选 2 段）

> **原文（第 3.1 节）**："This per-Gaussian importance metric 𝑤𝑔 can be easily derived from the per-pixel contributions 𝑤𝑝,𝑔, which are already computed during pixel-wise rasterisation. However, because full-scene rasterisation is computationally expensive, we propose a high-speed alternative: an approximate weighting function that predicts importance based on each Gaussian's scale, opacity and position. Given the complex, non-linear relationship between a Gaussian's geometric parameters and its final visibility, we employ a tiny, generalisable Multi-Layer Perceptron (MLP) to estimate this value."
>
> **译文**：这个逐高斯重要性指标 w_g 可以很方便地从逐像素贡献 w_{p,g} 推出来，而后者在逐像素光栅化过程中本来就要计算。然而，全场景光栅化的计算代价太高，我们提出一个高速替代方案：一个近似加权函数，基于每个高斯的 scale、opacity 和位置来预测其重要性。考虑到高斯几何参数与它最终可见度之间复杂、非线性的关系，我们用一个小而可泛化的多层感知机（MLP）来估计这个值。

> **原文（第 3.2 节）**："This is interesting because the goal of 3DGS compression is not to reconstruct the 3D Gaussians as closely as possible, but rather to maintain the quality of the rasterised images. Some Gaussians have a bigger impact on the image generation than others; These Gaussians should thus also have a larger weight in the quantisation process. This weighted non-uniform quantiser allows for significantly higher image reconstruction quality; however, the bitrate overhead of the quantiser metadata limits the rate-distortion performance."
>
> **译文**：这一点很有意思，因为 3DGS 压缩的目标不是尽可能精确地还原 3D 高斯本身，而是保持光栅化图像的质量。有些高斯对图像生成的影响比其他高斯大；这些高斯在量化过程中因此也应占更大的权重。这种加权非均匀量化器能带来显著更高的图像重建质量；但是，量化器元数据带来的码率开销限制了率失真性能。

### 主结果表述段（选 1 段）

> **原文（第 4.2 节）**："We evaluate the quantitative rate-distortion (RD) performance of the proposed method against state-of-the-art approaches in Figure 4. Our proposed method consistently outperforms uniform voxelisation across all evaluated datasets and codecs. Compared to Adaptive Voxelization [26], we achieve significantly better performance when using V-PCC and on the MPEG scenes. On the MPEG Objects dataset using G-PCC, Adaptive Voxelization exhibits better rate-distortion performance under certain configurations; however, our method yields a higher absolute quality ceiling."
>
> **译文**：我们在图 4 中对比了所提方法与当前先进方法的率失真性能。在所有评测的数据集和编码器上，所提方法都稳定优于均匀体素化。与自适应体素化 [26] 相比，我们在使用 V-PCC 时以及在 MPEG 场景类数据集上取得明显更好的性能。在使用 G-PCC 的 MPEG 物体类数据集上，自适应体素化在某些配置下率失真表现更好；但我们的方法绝对质量上限更高。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | 非均匀量化和 Lloyd-Max 本身是经典技术，HAC 等已有逐锚点自适应量化；新意在于把渲染误差分解成逐高斯平方贡献权重（式 2）、做出第一套完全兼容标准点云编码器的几何+属性非均匀量化框架（作者自称 first，第 1 节），以及用第二级量化压量化器元数据这个小而实的技巧；合并策略对比偏工程。以 2026-09 已有工作为参照 |
| 证据强度 | 高 | MPEG CTC 官方测试平台、两个标准熵编码后端、官方数据集加留一法、六个消融表覆盖全部组件、四段计时复杂度表、合并贡献已进 V-PCC Amd1 可在标准文档复核。缺口：没有开源代码链接；与 FlexGaussian/Adaptive Voxelization 的对比只给 RD 曲线图（Figure 4），没有汇总 BD-rate 数字；λ1-λ4 与合并阈值取值论文未报告 |
| 对期刊版的威胁度 | 中 | 撞的是"内容自适应量化"这条主张：它在标准轨上证明了按重要性分配量化比特收益巨大，且已被标准采纳，期刊版审稿人很可能拿它当"标准轨代表方法"要求对比或讨论。但它不渐进、量化器元数据必须写码流（与我们零侧信息契约相反）、面向标准 3DGS 表示而非锚点表示，与我们的渐进单文件码流主张不直接撞车。机制层面（编码端加权 Lloyd-Max + 传元数据 vs 解码端重算嵌套阶梯）是同一问题的两条相反路线，必须在相关工作中正面处理 |

## 对 DCCA-GS 的可借鉴点

1. 式 2 的贡献平方和分解 → 用作我们"训练期渲染敏感度监督"的理论解释和校准基准。它的推导说明：在高码率假设下，渲染 MSE 对每高斯量化误差的敏感度就是逐像素贡献的平方和。我们可以算我们训练期敏感度信号与 w_g 的相关系数（实验：同一批场景两个信号逐高斯算 Spearman 相关）；若高度相关，等于给我们的敏感度监督找到一个有出处的设计依据，写期刊时能回答"为什么这个监督信号合理"。
2. 第二级量化压大表（32n → 32(K+1)+log2(K)·n，第 3.2 节）→ 用在我们任何必须写进码流的浮点表上。我们当前零侧信息契约下没有量化器元数据，但（贡献组×粗值幅度桶）复合条件表若在期刊版某配置下需要随码流传输，可套这个"对表本身再做一次量化"的办法压表。需要实验：对条件表按 1/2/3 bit 二阶量化，测体积与画质变化。注意我们已有的负结果（条件表细化、MLP 拟合码表）不覆盖"对必传的表做二阶量化"这一手。
3. 同体素重要性加权合并（Table 5、Table 6，免训练，已进 V-PCC Amd1）→ 作为剪枝之外第二条减基元路线的参考。我们的底座是锚点表示，基元由小 MLP 生成，不能直接搬"合并同格高斯"；但"量化后对同格锚点按重要性加权合并"是可试探的对应物（实验：在 SPA 剪枝之外，对量化到同一格的锚点做合并，测 total_MB 与 PSNR 的变化）。它与剪枝正交：剪枝删低影响基元，合并删空间冗余。
4. 逐属性开关消融的设计（Table 8）→ 直接套用到我们的分字段粗化档位报告上：逐字段单独开/关粗化，量化每个字段的净贡献，让"哪些字段吃掉了体积"有逐项数字，而不是只报总收益。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 量化怎么定 | 编码端按重要性加权 Lloyd-Max 非均匀量化（基层无渐进概念，一档到位）；量化器重建值表写进码流，用第二级 2-bit 量化压表的体积（第 3.2 节） | 嵌套质量阶梯量化：基层传全锚点粗量化值，增强层逐层补精度；量化步长等参数解码端从已解码数据重算，不写码流 | 同一个"内容自适应量化"目标的相反解：它把量化表当必要开销并设计机制去压它；我们从契约上禁止传表，改用解码端重算。它的 Table 2 证明位深本身是最大杠杆（12-14 bit 足够），这与我们基层粗量化的合理性互为佐证 |
| 熵编码上下文 | 该论文不涉及内部设计：直接用 G-PCC 八叉树+RAHT 或 V-PCC+HEVC 现成后端，声明与熵编码引擎无关（第 7 节） | 二值分解区间编码（零标志→符号→幅度逐位是非题）+（贡献组×粗值幅度桶）复合条件表 + 层条件熵编码（细层用已解码粗层作上下文） | 它把全部创新压在量化前置环节，熵编码原封不动；我们的创新主要在熵编码与码流组织。两者几乎不重叠，期刊版可定位成互补而非同质竞争 |
| 渐进/可截断 | 该论文不涉及：四个码率档是四套独立配置各自编码的独立码流（第 4.1.1 节），不渐进、不可截断 | 单文件渐进分层码流，任意画质档都是文件的连续字节前缀，层级序文件布局，bit-exact 截断验证 | 完全错开。它的存在反而说明标准轨目前没有渐进方案——这正是我们主创新的空间 |
| 侧信息 | 量化器元数据必须写码流（作者视为固有开销并专门压缩）；重要性 MLP 不传、解码端不用（第 3.1、3.2 节） | 零侧信息契约：解码端能重算的才可以用，否则必须写码流；8→32→3 小 MLP 输出被解码端复算 | 关键分歧点：它的小 MLP 只在编码端用，解码端永远不需要重要性；我们的 MLP 输出必须解码端可复算才能用。同一"极小 MLP 打分"的组件，在我们框架里要过多一道"解码端可重算"的门槛 |
| 冗余消减（剪枝/合并） | 量化后同体素高斯按不相似度阈值加权合并，免训练，已被 V-PCC Amd1 for GS 采纳（第 1、3.3 节） | SPA（GaussianSpa 式训练期 ADMM 剪枝）+ Mini-Splatting depth-reinit（深度反投影增密）作为基础措施；不做量化后合并 | 它的合并发生在编码期、作用于量化后的标准 3DGS；我们的剪枝发生在训练期。可试探的方向见可借鉴点 3（同格锚点合并） |
