# 《Towards Practical Compression of 3D Gaussian Splatting》精读笔记

- arXiv：https://arxiv.org/abs/2609.30245
- 发表：arXiv 预印本（v1，2026-09-24，cs.CV；未标注会议接收信息）
- 来源说明：txt 含少量控制字符已清理，公式与 PDF 逐条核对一致

方法名 COSA-GS：Context without Spatial Aggregation for Gaussian Splatting（不做空间聚合的上下文模型）。代码开源：https://github.com/pengpeng-yu/COSA-GS 。

> 一句话定位：这篇把锚点式 3DGS 压缩的上下文模型从"跨锚点的空间聚合"（hash 网格、锚点间依赖）缩成"单个锚点内部的因果链"，上下文网络只剩线性层加 GELU；再把整个上下文网络做成纯整数推理，让不同 GPU/CPU 平台解码出逐位相同的熵编码符号。省体积靠更便宜但够用的条件概率模型加上率失真训练，卖点的另一半是解码快（flowers 场景 0.1279 秒，对照 HAC++ 24.95 秒，Figure 2）和跨平台一致（Table 2）。

## 问题与支撑观察

摘要原话："Existing compression methods often rely on spatial context modeling over irregular 3D representations, increasing the complexity of training and coding. Meanwhile, floating-point context inference can introduce numerical inconsistencies across platforms, causing entropy-decoding failures."

它针对 HAC/HAC++/ContextGS 这一支的两大工程毛病。第一，空间上下文要在不规则的三维锚点集合上做邻域聚合（查 hash 网格、按层级排锚点），模型大、流程绕、解码慢；第二，熵解码要求编解码两端算出的概率完全一样，浮点运算在不同硬件上结果可能有微小差别，概率一错熵解码就崩（浮点跨平台不一致不是本文发现的，引用的是图像/视频压缩界的既有结论：Ballé et al. 2019、Koyuncu et al. 2022，第 1 节、第 2.3 节）。解码慢的量化证据：Figure 2，flowers 场景 HAC++（30K）解码 24.95 秒、14.44 MiB、21.20 dB；PCGS（40K）25.49 秒、15.64 MiB、21.22 dB；COSA-GS（15+30K）0.1279 秒、9.99 MiB、21.37 dB。

核心直觉（引言原话）："Our intuition is that 3DGS optimization itself can reduce inter-anchor dependence when guided by an appropriate context model."（让每个锚点学到有区分度的空间信息，把剩下的依赖关在锚点内部。）未单独给实验，间接支撑在第 4.3 节消融：去掉全部上下文来源才明显掉点，只留可学习潜变量（完全不用坐标）时性能仍接近完整模型（Figure 4）。

没走的路：继续用 hash 网格做空间上下文（HAC++ 路线）——它有效但模型复杂、解码慢（第 1、2.2 节），替代办法是把条件信息拆成两路便宜的来源——锚点自己的坐标（解码端已解出，天然可得）加每个场景优化出来的小潜变量；传校准信息修数值偏差（Tian et al. 2023 路线）——能解决跨平台问题但要传额外侧信息（第 2.3 节），它选择把网络整体整数化，不传校准数据；锚点分级粗到细（ContextGS 路线）和渐进码流（第 2.2 节提到 PCGS 用渐进掩码加渐进量化做增量码流）都不做——每个率失真权重 λ 单独训练一个模型，输出单码率码流。

## 方法

底座是 Scaffold-GS：场景由 N 个锚点表示，锚点 i 有坐标 xi、特征 fi、对数域位置缩放 ri、K 个高斯的偏移量 oi 和对数域高斯缩放 si，第 k 个高斯中心 pi,k = xi + exp(ri)·oi,k（第 3.1 节，式 1）。渲染时共享 MLP 从锚点特征和视角条件预测颜色、不透明度、旋转和缩放调制（式 1：{(ci,k, αi,k, qi,k, ui,k)}_{k=1}^K = Dθ(fi, vi)，sG_{i,k} = exp(si) ⊙ sigmoid(ui,k)）。COSA-GS 在此之上加每锚点可学习潜变量 zi（默认 4 通道，第 4.3 节）和偏移掩码 mi，得到增广表示 A = {(xi, zi, Vi, mi)}（第 3.2 节）。

**训练期**（默认 15+30K，即 15K Scaffold-GS 初始化加 30K 率失真优化，共 45K 次迭代，第 4.1 节）：

1. 先按 Scaffold-GS 优化 15K 次得到初始锚点表示（第 4.1 节）。
2. 30K 率失真优化：损失 L = (1−α)·L1 + α·(1−SSIM) + λ·Lrate/(8·10^6)，α = 0.2（式 8，第 3.6 节）。率项 Lrate = Rz + Rf + Rr + Ro + Rs，其中锚点级率项 Rg（g 属于 z、f、r、s）只对"还有活跃偏移"的锚点计，偏移率 Ro 只对掩码为 1 的偏移计（式 7：Rg = Σ_i ai·Σ_j ℓg_{i,j}，Ro = Σ_i Σ_k mi,k·ℓo_{i,k}，ai = 1[Σ_k mi,k > 0] 表示锚点是否活跃）。
3. 训练时量化用加性均匀噪声代替取整，得到松弛潜变量供后续通道预测和上下文构建（第 3.4 节）。
4. 剪枝（摘要称 adaptive Gaussian pruning）：偏移掩码 mi 由可学习标量 ηi 二值化而来，mi = 1[sigmoid(ηi) > ϵm]，ϵm 是掩码阈值（数值论文未报告），训练用直通梯度（第 3.5 节）；掩码在渲染时压制不活跃高斯，没有活跃偏移的锚点整个剪掉（第 3.6 节）。除掩码机制外论文没有单独的剪枝小节。
5. QAT 变体（量化感知训练）：训练中插入伪量化——激活按张量仿射量化、权重按输出通道对称量化，都是 8-bit，前向反量化计算、取整处走直通梯度（第 3.7 节，式 9：qx = clip(⌊x/sx⌉ + zx, qmin, qmax)）；量化观察器在默认 30K 配置的第 25,000 次迭代冻结（附录 A.4）。

**编码期**（第 3.3–3.5 节）：

1. 锚点坐标量化 qi = ⌊xi/∆x⌉（∆x 为坐标量化步长，数值论文未报告），按八叉树递归编码：每个父节点存 8 个子节点的占据模式，同一层的占据符号共享一个经验分布 pℓ(v) = (1/Nℓ)·Σ_{j=1}^{Nℓ} 1[ωℓ,j = v]（式 2，ωℓ,j ∈ {1,…,255}），坐标占据和偏移掩码用分类编码并把这个分布的 CDF 写进码流（第 3.3 节、A.4）。
2. 解码端同一步骤先重建整数坐标 x̂i = ∆x·qi，归一化后送小 MLP Ex 得到几何上下文 cg_i = Ex(x_norm_i)（第 3.4 节）。编码端用相同坐标。
3. 锚点潜变量逐通道条件编码：p(ẑi | cg_i) = Π_{c=1}^{Cz} p(ẑi,c | cg_i, ẑi,<c)（式 3，hat 表示量化重建值），每个通道一个单独 MLP，从几何上下文和已重建的前几个通道预测高斯均值和标准差；通道间串行、锚点间并行（第 3.4 节）。
4. 潜变量全部重建后拼出锚点上下文 ca_i = Fa([cg_i ; Ez(ẑi)])（式 5），Ez 和 Fa 都是带 GELU 激活的线性变换。整个上下文模型只有线性变换和激活（摘要）。
5. 属性编码按式 6 的因果顺序：p(V̂i | ca_i, mi) = p(f̂i | ca_i) · p(r̂i | ca_i) · p(ô_i | ca_i, r̂i, mi) · p(ŝ_i | ca_i, r̂i)（mi ∈ {0,1}^K 是偏移掩码）；特征各通道并行；r̂i 重建后先嵌入，过两个分支专用 MLP，把结果加到 ca_i 上，分别条件偏移分布和高斯缩放分布；只有掩码为 1 的偏移被建模和写码流（第 3.5 节）。
6. 每个标量按式 4 量化成整数残差并用离散零均值高斯编码：kv = ⌊(v − µ)/∆g⌉，v̂ = µ + ∆g·kv，ℓv = −log2 P(kv; σ)，P(kv; σ) 是标准差为 σ 的离散化零均值高斯赋给残差符号的概率，ℓv 是估计码长；每组 g 属于 {z, f, r, o, s} 有一个学出来的量化步长 ∆g，组内所有锚点、所有分量共享（第 3.4 节）。
7. 熵编码后端：CPU rANS，16 位概率精度，每个高斯属性流默认拆 4 个独立块、最多 4 个 CPU 线程并行（A.3）。量化步长导出成 (ρg, τg) 整数对写码流（A.4，式 17：∆g_int = ρg / (S·2^{τg}) ≈ ∆g）。QAT 变体的编码还要导出整数化网络参数。

**解码期**（第 3.3、3.7、A.4 节）：

1. 解八叉树逐层展开，精确恢复最细层整数坐标，Morton 序与属性流一致（第 3.3 节、A.4）。
2. 坐标定点归一化（式 15：ui,d = R(2^S·ci,d, Ld − 1) − S，共享定点尺度 S = 2^20），几何上下文直接从整数算，不先还原浮点世界坐标（A.4）。
3. 潜变量逐通道：整数网络预测均值和"标准差表索引"，索引定点取整裁剪后从 128 张离线生成的共享整数 CDF 表中选一张（式 16：ℓ = clip(R(v(q), S), 0, 127)）；rANS 解出残差 kv 后按式 18（qv̂ = qµ + R(kv·ρg, 2^{τg})，式 13 的无浮点实现）做定点重建，重建值反馈给下一通道预测。
4. 属性组同流程，r̂ 的定点重建值反馈去条件偏移和高斯缩放分布；位置缩放和高斯缩放在实现对数域编码（A.4）。
5. 特征、偏移、对数高斯缩放不再反馈给任何后续预测，最后才转回浮点（式 19：v̂ = qµ/S + kv·∆g_int），这一步不影响熵解码一致性（A.4）。
6. 并行的 rANS 块写固定输出槽，线程调度不影响解码序列（A.4）。解出完整 Scaffold-GS 表示（渲染网络参数以 FP16 存码流，A.4）后做新视角渲染。

整数推理的关键配置（第 3.7 节、A.4）：线性层 int8 权重和激活（[−127,127]）、int32 偏置和累加器、int64 重标度中间量；定点重标度 qy = clip((a ⊙ ρ)/2^τ + zy, qmin, qmax)（式 11，整数乘子 ρ 加 2 的幂除数 2^τ）；GELU 拆成整数可直接算的 max 项加有界非线性修正项 h(t) = t·Φ(−t)（式 12），h 用量化查找表近似，避免浮点计算标准高斯累积分布 Φ，查找表覆盖 [0,6]、间隔 1/512、24 位小数，编解码两端共享；128 张高斯 CDF 表，标准差从 0.1 到 256 对数均匀分布，离线生成、跨场景共享；整数取整 R(a, d) = sgn(a)·⌊|a|/d + 1/2⌋（四舍五入、半数远离零；参数导出和源残差量化则用半数到偶，式 14）。保证的是"熵解码符号及其定点因果状态"一致，明确不保证浮点渲染结果逐位一致（A.4）。

### 码流组织相关事实

1. 文件布局（Figure 1）：场景码流包含 metadata、锚点坐标（八叉树）、位置缩放、偏移掩码、高斯缩放、偏移、锚点潜变量、锚点特征、神经网络模型参数；解码后恢复 Scaffold-GS 表示做渲染。
2. 位分配（Figure 5，Mip-NeRF360）：锚点特征加偏移合计占 77.3%（λ=0.0006，均 8.55 MiB）到 53.8%（λ=0.008，均 1.32 MiB）；锚点潜变量 3.4%–8.5%；锚点坐标从 5.8% 升到 15.4%；模型参数加元数据从 0.6% 升到 3.9%。低码率点上坐标、模型、元数据的占比变大。
3. 渐进层划分：没有。每个 λ 独立训练一个模型，一次产出一个单码率码流；训练档位是 15+20K、15+30K、15+45K 三种迭代数和各自的 QAT 变体（第 4.1 节）。可截断性：不涉及，跨平台一致性验证是整条码流解码后的符号逐位相等（45/45，Table 2），没有前缀截断或多档设计。
4. 解码端重算的量：几何上下文（整数域直接从已解码坐标算，A.4）、各属性组的 µ 和 σ（上下文网络）、因果链上先前重建值；八叉树各层占据分布同层共享经验分布，但 CDF 本身随码流传输（A.4）。
5. 侧信息：场景码流里含整数网络参数、重标度参数、每组重建步长 (ρg, τg)、坐标元数据、掩码 CDF、压缩流、FP16 渲染网络参数；另外解码还需要项目源代码、模型配置文件、共享的高斯 CDF 表和 GELU 表——这些跨场景共享资源不计入场景大小（A.4"一致性范围"）。

## 实验口径

- 数据集：Mip-NeRF360（9 场景）、DeepBlending（2 场景）、Tanks&Temples（2 场景）、BungeeNeRF（6 场景）（A.1）。宽度超过 1600 像素的图降到 1600 宽、保持长宽比；Mip-NeRF360/Tanks&Temples/DeepBlending 每 8 帧取 1 帧测试，BungeeNeRF 前 31 帧测试（A.1）。数据划分和分辨率设置沿用 Scaffold-GS、HAC++、PCGS、CAT-3DGS 公开代码的设置（A.1）。
- 体积：compressed scene size，单位 MiB。计入网络参数、重标度参数、步长、坐标元数据、掩码 CDF、压缩流、FP16 渲染网络；跨场景共享的高斯 CDF 表和 GELU 表、源代码、配置文件不计入（A.4）。
- 对比方法：ContextGS、HAC++、CAT-3DGS、PCGS、SpeedyGS；这些方法的率失真数据取自各自论文报告的每场景指标，不是重跑（第 4.1 节）。
- 训练配置：默认 15+30K（共 45K 次迭代），另评 15+20K、15+45K 和 15+30K QAT（第 4.1 节）。硬件为 NVIDIA 5880 Ada GPU 加 AMD EPYC 9654 CPU（第 4.1 节）；计时协议：一次预热后测，CUDA 前后同步，编码从内存中训练好的模型到内存码流，组件计时声明为"诊断性的、不是互斥分解"（A.3）。

## 主结果

Table 1：COSA-GS（15+30K）在 Mip-NeRF360 上的质量、体积与运行时间（共 6 行，全表照抄）：

| λ | PSNR (dB) | SSIM | LPIPS | Size (MiB) | Total Enc. (s) | Total Dec. (s) | Entropy Dec. (s) | Anchor Coord. Dec. (s) | NN Dec. (s) | Scaffold 15K Train (s) | RD 30K Train (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .0006 | 27.94 | 0.812 | 0.234 | 8.55 | 0.090 | 0.112 | 0.090 | 0.017 | 0.008 | 479 | 3321 |
| .0010 | 27.86 | 0.809 | 0.242 | 6.56 | 0.080 | 0.101 | 0.082 | 0.015 | 0.007 | （同上） | 3279 |
| .0020 | 27.61 | 0.798 | 0.263 | 4.20 | 0.064 | 0.081 | 0.065 | 0.012 | 0.006 | （同上） | 3238 |
| .0040 | 27.18 | 0.776 | 0.298 | 2.42 | 0.048 | 0.062 | 0.048 | 0.010 | 0.004 | （同上） | 3206 |
| .0080 | 26.54 | 0.741 | 0.341 | 1.32 | 0.036 | 0.044 | 0.034 | 0.007 | 0.003 | （同上） | 3210 |
| Avg. | 27.42 | 0.787 | 0.276 | 4.61 | 0.063 | 0.080 | 0.064 | 0.012 | 0.005 | 479 | 3251 |

（原始表格中 Scaffold 15K Train 与 RD 30K Train 分列；15K 一列仅首行和 Avg 行标 479，其余行留空，此处按原文保留。）

Table 2：COSA-GS（15+30K，QAT）跨平台解码时间与一致性（Mip-NeRF360，全部码流在 5880 Ada + EPYC 9654 平台编码；共 3 行，全表照抄）。最后一列是 45 条测试码流（5 个码率点 × 9 个场景）中熵解码符号与锚点坐标完全一致的条数：

| GPU | CPU | λ=.0006 | .001 | .002 | .004 | .008 | 平均 | 一致性 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA 5880Ada | AMD EPYC 9654 | 0.121 | 0.110 | 0.096 | 0.076 | 0.061 | 0.093 | 45/45 |
| NVIDIA 4090 | AMD EPYC 7R32 | 0.145 | 0.134 | 0.114 | 0.090 | 0.071 | 0.111 | 45/45 |
| NVIDIA V100 | Intel Xeon Gold 6258R | 0.184 | 0.173 | 0.149 | 0.120 | 0.096 | 0.144 | 45/45 |

Table 4（15+30K 设置的数据集平均，共 20 行，此处按数据集节选范围）：Mip-NeRF360 五个 λ 点 PSNR 26.5370–27.9403 dB、体积 1.3158–8.5471 MiB；DeepBlending 五个点 PSNR 29.6241–30.5802 dB、体积 0.6383–3.3463 MiB；Tanks&Temples 五个点 PSNR 23.4090–24.4956 dB、体积 0.9933–5.0108 MiB；BungeeNeRF 五个点 PSNR 26.4055–27.2069 dB、体积 7.6483–16.5453 MiB。编码 0.0243–0.1597 秒、解码 0.0308–0.1747 秒（逐点数值见表 4 原文）。

与基线的正面对比没有数值表格，全部在 Figure 3 的率失真曲线里（PSNR/SSIM/LPIPS 对体积，四个数据集；基线数字取自各自论文，第 4.1 节）。正文给的图上读数：COSA-GS 在 Mip-NeRF360 上覆盖 1.32–8.55 MiB、26.54–27.94 dB，在 4.20 MiB 处达到 27.61 dB（第 4.2 节，对应 Table 1 λ=.002 行）。原文对曲线的表述（第 4.2 节）："the SSIM and LPIPS curves also show better overall trade-offs than ContextGS, HAC++, CAT-3DGS, and PCGS at comparable sizes. SpeedyGS achieves higher SSIM and lower LPIPS at several bitrate points, while COSA-GS provides a stronger PSNR–size trade-off." Figure 2（flowers 场景整场景数字）：HAC++ (30K) 21.20 dB、14.44 MiB、编码 12.90 秒、解码 24.95 秒；PCGS (40K) 21.22 dB、15.64 MiB、编码 13.63 秒、解码 25.49 秒；COSA-GS (15+30K) 21.37 dB、9.99 MiB、编码 0.1009 秒、解码 0.1279 秒。

三档训练步数和 QAT 的完整数字在 Table 3（15+45K）、Table 5（15+30K QAT）、Table 6（15+20K）；A.7 汇总：三个浮点设置下数据集平均编码 0.0240–0.1752 秒、解码 0.0264–0.1839 秒，所有码率点都低于 0.24 秒。

## 消融

消融在 Mip-NeRF360 上用 15+30K 设置做（第 4.3 节），结果以曲线呈现（Figure 4 主文、Figure 7 附录扩展到三个指标），没有消融数值表：

1. 逐来源开关（Figure 4 左、中）：分别去掉几何上下文、去掉锚点潜变量、两个都去。单去一个只小幅掉点，两个都去才明显掉点；"无几何上下文"变体接近完整模型。
2. 替换实验——位置缩放维度（Figure 4 左、中）：把每锚点标量位置缩放换成三维向量，整体无收益，高码率点还略退化（第 4.3 节）。
3. 容量扫描（Figure 4 右）：上下文通道 24→32 无提升；特征通道 32→50 最高码率点 PSNR 略升但低码率点退化；潜变量通道 1/4/8 曲线接近（第 4.3 节）。
4. 训练步数渐进对比（Figure 3，附录 Figure 6）：15+20K / 15+30K / 15+45K 三档曲线同图比较，步数增加收益递减，BungeeNeRF 上三档几乎重合（第 4.2 节、A.5）。
5. QAT 消融（Figure 3/6，Table 5）：整数化上下文推理的率失真曲线与浮点 15+30K 基本重合，仅 DeepBlending 和 BungeeNeRF 的部分低码率点有小损失（第 4.2 节）。

## 贡献列表（原文照抄）

1. "We propose COSA-GS, a simple 3DGS compression framework that integrates context construction and attribute coding through anchor-wise causal factorization without spatial context aggregation."
2. "We develop an integer inference pipeline with quantization-aware training for the context model, enabling consistent entropy decoding across heterogeneous platforms."
3. "We combine the proposed model with rate–distortion optimization and adaptive masking, achieving state-of-the-art compression performance and fast decoding on Mip-NeRF360, DeepBlending, Tanks&Temples, and BungeeNeRF."
