# 《LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2311.17245 |
| 发表 | NeurIPS 2024 |
| 阅读材料 | 全文（.lit-cache/txt/2311.17245.txt） |
| 与 DCCA-GS 的关系 | 同类竞品：训后"剪枝+蒸馏+VQ"路线的开山作；它的 global significance（渲染贡献统计）与我们 I6 的训练期渲染敏感度监督回答同一个问题——哪个高斯重要——但两端和口径都不同 |

> 一句话定位：把训好的 3D-GS 压约 15 倍——按"全局显著性"剪掉贡献小的高斯再微调恢复、把 3 阶球谐函数（SH）蒸馏到 2 阶、只对最不显著的 SH 做向量量化，Mip-NeRF 360 上存储从 782MB 降到 45MB（Table 1），FPS 从 144 提到 237。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾
3D-GS 的启发式致密化（clone/split）把 SfM 稀疏点膨胀到百万级高斯，原文："heuristic densification of sparse SfM points into dense Gaussians often results in overparameterization, leading to excessive storage and slower rendering speeds"（第 1 节）。后果有两个：单场景存储上 GB（Bicycle 1.4GB），高斯太多光栅化也变慢。所以它同时压两头：高斯数量 N 和特征维度 F（大头是 SH）。

### 支撑观察（直觉 / 数据 / 失败实验）
- 失败实验（第 3.2 节，Fig. 3）：直接拿 opacity 当剪枝标准，PSNR 从 27.2 掉到 25.3。低 opacity 不等于不重要——很多小 opacity 高斯叠起来撑着细节。这个反例直接逼出了 global significance 的设计。
- 第 3.3 节：未压缩表示里 81.3% 的特征维度被 SH 占据，每个高斯要存 (45+3) 个浮点数。
- 第 3.4 节：作者经验观察，对位置、旋转、尺度全部属性做 VQ 会显著掉精度，后面 Table 4 证实（全属性 VQ 后 PSNR 崩到 23.11）。

### 显然的路为什么没走
- opacity 剪枝：Fig. 3 已证明掉点，弃。
- 直接截断 SH 阶数：Table 2 #4 显示硬截断后 PSNR 从 31.64 掉到 30.32，丢高光反射；所以改成"蒸馏"——让低阶学生模型在伪视图上跟全阶老师学，把视角效果补回来。
- 全属性 VQ：Table 4 证明崩盘，改成只对显著性最低 60% 的 SH 量化，其余属性和高级 SH 用 float16 直存。
- 训练期改造致密化（后来 EAGLES、Compact 3DGS 走的路）：LightGaussian 选训后处理，换取即插即用——Table 5 显示同一套剪枝直接套在 Scaffold-GS 上也有效。这是它与同期方法的分工：LightGaussian 管训后管线和通用性，Compact 3DGS（Lee）/EAGLES 管训练期一体化。

## 二学：机制（洞察怎么变成方法）

### 核心流程
三步流水线（Fig. 2、附录 Algorithm 1/2），全部发生在 3D-GS 训完之后，解码端渲染方式不变：
1. 剪枝+恢复（训后）：遍历全部训练视图的像素/光线，统计每个高斯的 global significance，按分数排序剪掉最低的一部分；再用光度损失在原训练视图上微调 5000 步（Gaussian Co-adaptation，恢复期不加致密化）。
2. SH 蒸馏（训后）：teacher 是全 3 阶模型，student 截到 2 阶；在训练相机位置上加高斯噪声合成伪视图，最小化师生两模型渲染像素的差。
3. VQ（编码期）：K-means 初始化 8192 大小的码本，只对显著性最低 60% 的 SH 做量化；然后固定"高斯到码字"的映射，用光度损失微调码本和其余属性 5000 步。位置、不透明度、旋转、尺度、以及显著性高的 SH 不量化，float16 直存。

### 关键公式（按原文抄，注明编号）
- 高斯与协方差（Eq. 1）：`G(X) = e^{-1/2 X^T Σ^{-1} X}, Σ = RSS^T R^T`。
- 像素混合（Eq. 2）：`C = Σ_{i∈N} c_i α_i Π_{j=1}^{i-1}(1-α_i)`。txt 中连乘项写作 (1-α_i)，按渲染方程应为 (1-α_j)，疑为原文或抽取笔误，以 PDF 为准。
- 全局显著性（Eq. 3）：`GS_j = Σ_{i=1}^{M·H·W} 1(G(X_j), r_i) · σ_j · T · γ(Σ_j)`。
  - j 是高斯序号，i 是像素/光线，M、H、W 是训练视图数、图高、图宽；
  - `1(G(X_j), r_i)` 是指示函数：高斯 j 与第 i 条光线是否相交（hit count）；
  - σ_j 是该高斯不透明度；T 是光线到达它之前的透过率，`T = Π_{i=1}^{j-1}(1-σ_i)`；
  - γ(Σ_j) 是体积项，`V(Σ_j) = 4/3 π a b c`（a、b、c 为 scale 的三个维度）。
- 体积归一化（txt 未显示编号，实现细节里称 Eq. 4，原文公式以 PDF 为准）：`γ(Σ) = (V_norm)^β`，`V_norm = min(V(Σ)/V_max90, 1)`。V_max90 是全部高斯按体积排序后 90% 分位的体积：除以它再截到 1，防止背景大漂浮高斯垄断分数；β 取 0.1（第 4.1 节）。
- SH 蒸馏损失（txt 中标注为 (6)）：`L_distill = (1/HW) Σ_i ||C_teacher(r_i;[R|t]) − C_student(r_i;[R|t])||_2^2`，伪视图位置 `t_pseudo = t_train + N(0, σ²)`，σ=0.1。
- VQ 码字更新（第 3.4 节，公式按 txt 重排，以 PDF 为准）：`c_k = λ_d·c_k + (1−λ_d)·(1/T_k)·Σ_{g_j∈R(c_k)} GS_j·g_j`，其中 `T_k = Σ_{g_j∈R(c_k)} GS_j`，λ_d=0.8。即码字朝分到该码的所有高斯的加权平均移动，权重就是显著性分——显著的高斯更少拉动码字。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | global significance：训练视图上按像素混合贡献统计的重要性分数 = 被光线命中次数 × 不透明度 × 前序透过率 × 归一化体积（Eq. 3） |
| 哪一端算得出 | 编码端训后处理阶段：需要拿原始训练视图重新渲染统计（附录 Algorithm 2），解码端算不出 |
| 有无侧信息 | 分数本身不进码流（只决定剪谁、量化谁），解码端不需要，所以不是码流意义上的侧信息；但它也不是解码端可重算量，是一次性的编码端统计——与 DCCA-GS"零侧信息=解码端可重算"的口径不同 |
| 训练期还是编码期 | 训后（编码期）；剪枝和 VQ 之后各有一次 5000 步的恢复微调 |
| 和渲染质量的距离 | 代理量：统计"贡献量"而非"删掉后质量掉多少"；但三项因子（命中、不透明度、透过率）都是渲染公式 Eq. 2 的组成部分，比单一 opacity 更贴近渲染（Table 3 逐项证实） |

## 三学：证明与包装（创新怎么立住）

### 实验口径
数据集：Mip-NeRF 360（9 个实景场景）+ Tanks & Temples（选景同 3D-GS），附录加 NeRF-Synthetic。指标 PSNR/SSIM/LPIPS，体积报最终模型 MB（论文未逐项披露哪些计入，未报告熵编码），FPS 在 A6000 上测。对比对象：3D-GS 与 Compressed 3D-GS（作者自己重训、标 *）、Compact 3D-GS、Plenoxels、Instant-NGP、Mip-NeRF 360、VQ-DVGO（Table 1）。

### 主结果（注明表号）
- Table 1：Mip-NeRF 360 上 782MB/27.40 PSNR/0.813 SSIM → 45MB/27.13/0.806，FPS 144→237；Tanks & Temples 上 433MB/23.66 → 25MB/23.44，FPS 106→357。
- Table 5：同一剪枝套在 Scaffold-GS 上，173.60→112.56MB，FPS 152→178，PSNR 27.96→27.78（正文第 4.2 节说明剪掉了 80% 的 neural Gaussians）。
- 附录 Table 7：Blender 合成数据平均 52.381MB→7.838MB。

### 消融设计（注明表号）
- Table 2：九步渐进消融（剪枝→co-adapt→SH 截断→光度→蒸馏→伪视图→码本量化→VQ 微调），Room 场景。零样本剪枝掉到 30.67（-0.67dB），co-adapt 恢复到 31.64（反超基线 31.34）——"剪枝必然掉点、恢复能补回"的证据链完整。
- Table 3：显著性分数构成消融，最值得学：只 hit count 28.16 → 乘 opacity 30.27 → 再乘体积 30.67 → 加 co-adapt 31.64。逐项证明渲染公式每个因子都有贡献。
- Table 4：VQ 范围消融：全属性 23.11（崩）→ 全属性×显著性 26.23 → 仅 SH 30.68 → 仅 SH×显著性 31.16 → 加微调 31.40（21.10MB）。证明"选择性压缩"两层都必要。
- Table 6：SH 阶数消融：3 阶 SSIM 0.927 → 2 阶+蒸馏 0.926 → 1 阶+蒸馏 0.923。
- Fig. 6：剪枝率 70% 以上、VQ 率 65% 以上质量开始陡降——给出安全工作点。

### 贡献列表（原文照抄 + 逐条标注；论文无编号列表，自第 1 节末段归纳并注明"自行归纳"）
1. "To reduce Gaussian count (N), we propose a Gaussian Pruning and Recovery step, identifying and removing Gaussians with minimal impact on visual quality, followed by recovery to ensure smooth adaptation." —— [机制] 显著性剪枝加恢复微调
2. "For compressing features (F), we introduce an SH Distillation process to compact higher-degree spherical harmonic (SH) coefficients, supported by pseudo-view augmentation." —— [机制] SH 高阶到低阶的知识蒸馏
3. "Additionally, we employ Vector Quantization (VQ) to adaptively select a codebook of Gaussian attributes (e.g, positions, scales, and rotations), reducing precision for less significant features and applying quantization-aware fine-tuning to maintain quality." —— [机制] 显著性引导的选择性 VQ
4. "Our Gaussian Pruning and Recovery method generalizes well, enhancing performance across different 3D Gaussian formats, such as Scaffold-GS." —— [泛化验证] 说明剪枝策略与具体表示解耦

### 讲故事方式
主线一句话：3D-GS 致密化过度参数化 → 用渲染公式本身反推每个高斯的重要性 → 剪掉冗余后用蒸馏和量化继续压。最有说服力的是 Fig. 3（反例：opacity 剪枝掉 1.9dB）配 Table 3（正例：三项因子逐个加上去逐步恢复）——一反一正把"重要性信号该怎么设计"讲透了；体感最强的数字是 Table 1 的 782MB→45MB 且 FPS 翻倍。

## 对 DCCA-GS 的可借鉴点

1. 重要性分数的"逐项消融"写法（Table 3）：把渲染公式拆成命中次数、不透明度、透过率、体积逐项加回去验证。我们 I2 的公式特征（局部密度、尺度各向异性、offset 能量、掩码激活比例）应该做同样的逐项消融，证明每个特征都有增量，而不是整体报一个消融。
2. 显著性引导的选择性压缩（Table 4）：敏感字段留高精度、只压最不敏感的部分，与我们"每锚点量化步长乘子"是同一思想。差异是它用编码端训练视图统计、我们用解码端可重算特征——正好构成我们强调零侧信息时的对照基线，论文里可以直接引用对比。
3. 体积归一化的教训：体积项必须除以 90% 分位再截断，否则背景大高斯垄断分数（Eq. 4 附近的讨论）。我们做渲染敏感度统计（I6 的梯度 EMA）时同样要防大尺度锚点主导统计量，值得加归一化消融。
4. 剪后恢复的训练预算参考：零样本剪枝掉 0.67dB、5000 步微调反超基线（Table 2 #2→#3），支持我们基础措施 SPA 剪枝后保留恢复训练的设置，也给出恢复步数的量级参考。
