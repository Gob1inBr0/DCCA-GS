# 《Non-Uniform Quantisation for 3DGS Compression》精读笔记

- arXiv：https://arxiv.org/abs/2608.28272
- 发表：arXiv 预印本（v1，2026-08-28），定位为 MPEG 3DGS 标准化的正式技术贡献
- 作者：Bert Van Hauwermeiren、Adrian Munteanu（布鲁塞尔自由大学），Patrice Rondao Alface（Nokia 比利时）
- 开源：未找到（论文未给代码链接）
- 来源说明：公式与表格版式已逐项对照 PDF（pymupdf）核验

> 一句话定位：给 G-PCC/V-PCC 这两个 3DGS 标准码流换掉均匀量化——按"每个高斯对渲染像素的贡献平方和"加权做 Lloyd-Max 非均匀量化，把量化表本身再用第二级量化压小，量化后落在同一体素的高斯按重要性合并；省码率靠"比特花在影响画面的高斯上"加"减少高斯个数"，其中合并部分已被 V-PCC Amd1 for GS 正式采纳（第 1 节、第 7 节）。

## 问题

第 1 节原话："Both codecs apply uniform quantisation to the 3DGS geometry and attributes before entropy coding. This approach is suboptimal, as it fails to account for the non-uniform distribution of Gaussian data and the varying perceptual importance of individual Gaussians." MPEG 把 3DGS 当点云编（G-PCC 走八叉树加 RAHT，V-PCC 走分块投影加视频编码，见第 1 节），均匀量化对每一个数一视同仁。第 2 节末尾点明第二层约束：学术上更强的压缩方法（Scaffold-GS、HAC 这类）"require modified representations and renderers, which are not allowed in the current G-PCC and V-PCC amendments"——标准化场景下不允许改表示。所以它把问题收窄成：不改表示、不改熵编码后端，只动量化这一个环节。

支撑观察：

- 渲染误差可以分解到每个高斯：第 2 节式 1 的 alpha 混合公式 C(p) = Σ_{g_i} C_{g_i}·α_{g_i}·Π_{j=1}^{i−1} (1 − α_{g_j}) = Σ_g w_{p,g}·C_g 说明像素颜色是各高斯颜色的加权和（w_{p,g} 是高斯 g 在像素 p 的贡献：自身密度乘以排在它前面所有高斯的透射率），第 3.1 节由此推出每个高斯的量化误差对总渲染误差的贡献系数（式 2）。这是从渲染方程推导出来的，不是经验观察。
- 高斯位置量化后同体素冗余真实存在且数量可观：Table 6，MPEG Scenes 平均高斯数从不合并的 495486.5 降到带判据合并的 376527.5，减少约 24%。
- 小 MLP 能替代全场景渲染来估重要性：Table 3，MLP 近似权重在 MPEG Scenes G-PCC 上等质量省码率 48.08%，反而好于精确渲染权重的 38.14%；Table 4，在 MPEG Objects G-PCC 上渲染法预处理要 2131.01 秒，MLP 法只要 13.72 秒。
- 位置位深存在"再低就崩"的阈值：第 5.1 节文字加 Table 2，位深从 18 降到 14/12 时等质量码率大幅下降，但 MPEG Scenes V-PCC 的 12-bit 行在 RGB-PSNR 列变成 +5.75（码率反升）、YUV-PSNR 列出现 nan（该配置下质量曲线够不到锚点的质量区间，Bjontegaard 插值失败）。

没走的路：改表示、改训练（锚点生成、哈希网格上下文那类）——MPEG 现行修正案不允许（第 2 节）；像 Adaptive Voxelization [26] 那样自适应体素化后再微调——第 3.3 节开头说 "Rather than relying on simple attribute averaging followed by costly fine-tuning iterations [26], we propose a series of importance-weighted merging strategies designed for high accuracy without retraining"，合并免训练；动标准熵编码器内部——第 7 节声明与熵编码引擎无关；直接剪掉不重要的高斯——第 6 节把合并定位成剪枝的"轻量替代"和互补项：剪枝丢弃低影响基元，合并保留它们、只消除同格冗余。

## 方法

整个管线在编码端跑，解码端不做任何训练、渲染或重算（Figure 1 给了全流程图）。

**编码端逐步骤：**

1. **重要性估计**（第 3.1 节，式 2、式 3）。精确做法：全场景渲染一遍，累计每个高斯在所有像素上 alpha 混合贡献的平方和 w_g = Σ_p w²_{p,g} = Σ_p α_i² Π_{j=1}^{i−1} (1 − α_j)²（式 2；原式下标即如此印刷：α_i 指该高斯自己的密度，α_j 指排在它之前的高斯密度，原文公式以 PDF 为准）。推导：渲染图像量化误差 D = (1/N) Σ_p (Σ_g w_{p,g} ε_g)² 展开为逐高斯项加交叉项，引用高码率量化理论 [8]（Gersho 与 Gray）假设量化误差独立、零均值、交叉项期望为零，得 E[D] ≈ (1/N) Σ_g ( Σ_p w²_{p,g} ) E[ε²_g]，其中 ε_g = C_g − Ĉ_g 是颜色参数的量化误差。w_g 是"贡献的平方和"，比贡献本身更强调同时出现在多个像素上的高斯。快速做法：205 参数小 MLP，ŵ_g = MLP( Π_{i=1}^{3} s_{g,i} , o_g , μ_g )（式 3，输入是三个 scale 分量的乘积、opacity、位置，共 5 个标量），离线训练、跨场景通用（第 4.1.1 节用留一法：每个数据集留一个场景训练 MLP，其余场景只做评测）。不使用球谐系数和旋转（第 3.1 节"On the choice of inputs"段：视角近似均匀分布时旋转不影响重要性；靠场景中心的高斯更可能是前景）。
2. **位置量化（即体素化）**（第 3.2 节）：对高斯位置做重要性加权的 Lloyd-Max 非均匀量化，量化档数对应 14 bit（正面场景）/ 11 bit（物体）（第 4.1.4 节；Table 2 加粗的选定位深是 V-PCC 场景 14、G-PCC 场景 12、V-PCC 物体 11、G-PCC 物体 9，与第 4.1.4 节的单一说法不完全一致，如实记录）。加权 MSE 目标 D = E[ w(x)(x − Q(x))² ] = Σ_{i=0}^{n−1} Σ_{x∈[b_i, b_{i+1})} w(x)(x − y_i)²，加权质心更新 y_i = Σ_{x∈[b_i,b_{i+1})} w(x)·x ／ Σ_{x∈[b_i,b_{i+1})} w(x)（y 为重建值，b 为边界，w(x) 为该标量值所属高斯的重要性）。输出：每个高斯的量化位置、量化器边界和重建值表。
3. **同体素合并**（第 3.3 节，式 4-8）：落在同一体素的高斯，先按不相似度筛（式 8：dissimilarity_{i,j} = λ1‖SH_i − SH_j‖ + λ2‖Σ_i − Σ_j‖ + λ3|o_i − o_j| + λ4·min{w_i, w_j}，低于阈值才允许合并；λ1 到 λ4 的取值论文未报告）。合并的三种几何方案：加权参数平均（四元数用 Markley 等人的优化式平均，式 5：q = argmax_q qᵀ ( Σ_i w_i q_i q_iᵀ ) q，避免 q 与 -q 双覆盖问题；式 4 的朴素线性平均 q_linear = (1/Σ_i w_i) Σ_i w_i q_i 因不满足反对称性被否定）、加权协方差空间平均（式 6：Σ = (1/Σ_i w_i) Σ_i Σ_i·w_i，对协方差矩阵加权平均后做特征值分解拆回 scale 和旋转）、加权对数欧氏平均（式 7：Σ = exp( (1/Σ_i w_i) Σ_i log(Σ_i)·w_i )，对称正定矩阵黎曼均值的近似）；opacity 和球谐系数直接加权平均。消融（第 5.3 节，Table 5）选定加权参数平均。
4. **属性量化**（第 3.2 节）：对 scale、rotation、opacity、SH 各通道分别做加权 Lloyd-Max（权重还是那个重要性）。每套量化器的重建值表本身是浮点元数据，直接传要 32n bit（n 为档数），是码率瓶颈；于是把相邻重建值的差分 d_i = y_i − y_{i−1} 当作新信号，再做一次加权 Lloyd-Max（第二级，2 bit，第 4.1.4 节；额外失真 D̂ = Σ_k Σ_{d_i∈[b'_k, b'_{k+1})} W(d_i)(d_i − d̂_k)²，其中每个差分的权重 W(d_i) 是它所在原始 bin 内全部取值重要性之和）。码流里只存：首个重建值 y_0、K 个差分重建值 {d_k}、每个重建值的差分索引 k_i。解码端按 y_i = y_0（i=0）、y_i = y_{i−1} + d̂_{k_i}（i>0）递推出整张表。元数据从 32n bit 降到 32(K+1) + log2(K)·n bit（第 3.2 节给出 K=4、n 在 2^10 到 2^18 之间时的量级估计）。
5. **熵编码**：量化后的符号交给现成的标准后端：G-PCC（八叉树几何 + RAHT 属性，release_mpeg151）或 V-PCC（分块投影 + HM-16.20 HEVC Main 10，mpeg152-gs-anchor）（第 4.1.1 节）。本文不改后端，第 7 节声明方法"entirely agnostic to the underlying entropy coding engine"。

**解码端**：读码流，用码流里的量化器元数据重建量化表（递推公式），反量化，走标准解码器和标准渲染器。没有解码端重算。

### 码流组织相关事实

- 渐进层：不做渐进。四个评测码率点是四套独立配置各自编码出的独立码流（第 4.1.1 节："Testing is conducted across four standardised ratepoints"，各码率点只保证预处理一致），切换画质档要换文件。可截断性：无，不涉及码流截断或单文件多画质档。
- 文件布局：未描述，直接使用 G-PCC/V-PCC 标准码流容器；量化器元数据如何嵌入标准属性字段的细节论文未报告。
- 侧信息：量化器元数据必须写码流（32(K+1) + log2(K)·n bit，第 3.2 节，这是论文认账的开销并专门用第二级量化去压它）；重要性 MLP 不传、解码端不用。

## 实验口径

- 平台与协议：MPEG 3DGS-PCC 测试平台，按共通测试条件（CTC）执行（第 4.1.1 节）。两个标准熵编码后端：G-PCC（release_mpeg151，八叉树几何 + RAHT 属性编码）、V-PCC（mpeg152-gs-anchor，HM-16.20 HEVC Main 10 分块视频编码）。四个标准码率档，各档熵编码配置不同、预处理管线一致。
- 体积/码率怎么算：依赖标准参考软件输出的码流比特数，论文未报告逐字段体积分解。质量指标：PSNR、SSIM、IVSSIM、LPIPS（第 4.1.3 节）；RD 收益用 Bjontegaard 等质量码率变化（BD-rate）表示，锚点是标准自带的均匀量化配置。
- 表格列含义说明（重要）：消融表（Table 2/3/5/6/7/8）的"BD-Rate [%]"是跨五个指标列的总表头，每列数值是"用该质量指标算出的等质量码率变化"，负数代表省码率。这是按 PDF 表头跨列版式确认的读法（正文没有逐列解释）；这也解释了 Table 2 中 V-PCC 场景 12-bit 行 YUV-PSNR 列出现 nan（该配置够不到锚点质量区间，无法插值）。
- 数据集：MPEG 官方 3DGS 数据集两类——大尺度正面场景；高细节静态物体与人像（举了 bartender_stable、cinema_stable、lego_bugatti、plant 等，第 4.1.1 节）。因为管线带机器学习组件，采用留一法：每个数据集留一个场景训 MLP，其余只评测。
- 基线：均匀量化锚点（位置 18 bit；属性 G-PCC 12 bit、V-PCC 10 bit，第 4.1.2 节）；Adaptive Voxelization [26]（体素化阈值参数设 20）；FlexGaussian [24]（原本不兼容有损熵编码，按原样评测，只有一个码率点）。
- 训练/实现配置（第 4.1.4 节）：体素化 14 bit（正面场景）/ 11 bit（物体）；第二级量化 2 bit；重要性 MLP 两个隐藏层（16 和 6），共 205 可训练参数（按输入 5 维核验参数量吻合：5×16+16 + 16×6+6 + 6×1+1 = 205）；Adam，学习率 0.01，50 轮。硬件 i9-13900 + RTX 4090。

## 主结果

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
- 与基线对比（第 4.2 节原文）："We evaluate the quantitative rate-distortion (RD) performance of the proposed method against state-of-the-art approaches in Figure 4. Our proposed method consistently outperforms uniform voxelisation across all evaluated datasets and codecs. Compared to Adaptive Voxelization [26], we achieve significantly better performance when using V-PCC and on the MPEG scenes. On the MPEG Objects dataset using G-PCC, Adaptive Voxelization exhibits better rate-distortion performance under certain configurations; however, our method yields a higher absolute quality ceiling."

## 贡献列表（原文照抄）

1. "A novel non-uniform quantisation scheme designed to minimise the weighted reconstruction error. By integrating an importance metric, we prioritise high-impact Gaussians to optimise the rate-distortion tradeoff."
2. "A comparison of several novel importance-weighted merging approaches for Gaussians that become co-located after voxelisation."
3. "An extensive performance analysis conducted in accordance with MPEG's 3DGS common test conditions (CTC). … The weighted merging contribution has already been adopted in the V-PCC Amd1 for GS."
