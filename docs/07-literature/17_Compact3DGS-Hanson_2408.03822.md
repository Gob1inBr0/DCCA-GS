# 《Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2408.03822 |
| 发表 | arXiv 预印本（IEEE 期刊版式，Manuscript received August 2024，未标明期刊名）；早期版本发表于 CVPR 2024（highlight），题为 "Compact 3D Gaussian Representation for Radiance Field"，即文中引文 [20] |
| 阅读材料 | 全文（.lit-cache/txt/2408.03822.txt） |
| 与 DCCA-GS 的关系 | 同类竞品：训练期 learnable mask 剪枝 + 神经场颜色 + 残差 VQ 的端到端路线；它的掩码激活量与我们 I2 公式特征里的"掩码激活比例"直接相关 |
| 作者说明 | txt 首页作者为 Joo Chan Lee、Daniel Rho、Xiangyu Sun、Jong Hwan Ko、Eunbyung Park（成均馆大学/UNC，第 1 页）。本编号是 CVPR 2024 同名工作的期刊扩展版，第一作者 Joo Chan Lee；文件名中的 "Hanson" 系误标（Alex Hanson 是 Speedy-Splat 的作者），与本论文无关，文件名按要求保留但以此为准 |

> 一句话定位：用训练期可学习掩码（同时作用在 scale 和 opacity 上）删掉冗余高斯，SH 换成 hash grid+小 MLP 的神经场，scale/rotation/时间属性用残差向量量化——静态场景对 3DGS 压缩超 25 倍、动态场景对 STG 压缩超 12 倍，渲染还更快（摘要；Table I、Table IV）。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾
3D-GS 需要海量高斯才能保质量，存储常超 1GB（第 I 节），而致密化"produces myriads of redundant and insignificant Gaussians"（第 III-A 节）。对训后剪枝路线（LightGaussian 等），它在引言里给出针对性批评：静态场景训后估计显著性可行，动态场景要衡量每个高斯在整个时间轴上的重要性，训后再算很困难（第 I 节、IV-A 节）；且训后处理无法降低训练期显存。所以它把剪枝搬进训练期，用梯度下降直接学出每个高斯对全部时间步渲染的影响。

### 支撑观察（直觉 / 数据 / 失败实验）——"哪些参数冗余"的证据
- 数量冗余（Fig. 3 + Table VI 前两行，Bonsai 场景）：2.34M 高斯 PSNR 29.87，删到 967K 后反而 29.91——"×2.42 Gaussians show similar performance"（第 III-A 节），约六成高斯是冗余的。
- 每高斯参数冗余（第 III-C 节）：SH 占每高斯 59 个参数中的 48。
- 逐属性存储分解（Table VIII，Mip-NeRF 360 平均）：3DGS 的 746MB 里颜色（SH）占 606.9MB，其余为 scale 37.9MB、位置 37.9MB、opacity 12.6MB、rotation 8.3MB——颜色是绝对大头。
- 几何相似性（第 III-B 节）："the geometrical shapes of most Gaussians are very similar, showing only minor differences in scale and rotation"，且场景由大量小高斯组成、单个基元不需要多大多样性——所以 scale/rotation 可以共用小码本（码本大小 C=64、级数 L=6，第 V-A 节）。Fig. 9-(b) 可视化 learned 码本索引：低阶码本幅度大、分布均匀，越往后级码字幅度越小、分布越不均，说明各级残差确实在递减。
- 时间冗余（第 I 节）：动态场景里"groups of moving parts typically follow similar motion patterns"，静止区域完全不动——时间属性同样可码本化。

### 显然的路为什么没走
- 训后显著性剪枝（LightGaussian、Compressed 3DGS 路线）：静态可用，但动态场景难以估计全时程重要性（第 IV-A 节），也压不了训练期显存；训练期掩码靠梯度下降同时吸收全时间轴的渲染影响。
- 每高斯独立存 SH：不利用空间冗余（相邻高斯颜色相近，第 I 节），换成共享的 hash grid 神经场按位置查颜色。
- 一次性 VQ（K-means 直接量化）：计算复杂度和 GPU 显存高（第 III-B 节，引 SoundStream），改用 L 级级联残差 VQ，每级码本可以很小。
- EAGLES 式做法：第 II-A-3 节点名批评"EAGLES controlled the number of Gaussians by just adjusting the densification schedule, resulting in a sub-optimal reduction"，并称自己是唯一在训练期真正 mask 掉无效高斯的工作。注意这句对 EAGLES 的描述不准确（EAGLES 第 4.3 节有显式 influence 剪枝，不只是调致密化日程），属同期竞争的表述，引用时要分辨。

## 二学：机制（洞察怎么变成方法）

### 核心流程
静态场景（第 III 节，底座 3DGS）：
1. 训练期剪枝：每个高斯带一个可学习的 mask 参数 m_n，前向生成二值掩码 M_n（STE，Eq. 5），同时作用到 scale（Eq. 6）和 opacity（Eq. 7）；每次致密化步骤按掩码删除被 mask 的高斯；与 3D-GS 中途停止致密化不同，mask 全程贯穿训练，训练期显存也随之下降（Fig. 3）；训练结束后被 mask 的高斯直接删除，掩码本身不入码流。
2. 颜色（第 III-C 节）：无界位置先 contract 到有界（Eq. 13），过 hash grid（2 通道特征、16 个分辨率 16~4096）+ 2 层 64 通道 MLP 输出 0 阶 SH 再转 RGB（Eq. 12）——不再逐高斯存 48 维 SH。
3. 几何（第 III-B 节）：scale 和 rotation（在 mask 之前）过 L 级残差 VQ（Eq. 9-11）；码本 K-means 初始化，只在最后 1K 迭代启用 R-VQ 损失（其余时间 L_r=L_s=0）以控制开销（第 III-D 节）。
4. 后处理（Ours+PP，第 V-A-3 节）：位置和标量属性存 16 位半精度；hash 参数与标量做 8-bit min-max 量化；剪掉绝对值小于 0.1 的 hash 参数；高斯按 Morton 序排列；8-bit 值和 R-VQ 索引做 Huffman 编码再 DEFLATE 压缩。

动态场景的 timescale 处理（第 IV 节，二学重点；底座 STG）：STG 给每个高斯一个时间中心 μ_n（该高斯最显著的时刻），位置/旋转轨迹用多项式系数建模（Eq. 15-16），时间可见性用径向基 `o_n(t) = s_on·exp(−ξ_n|t−μ_n|²)`（Eq. 19），其中 ξ_n 是时间尺度，表示该高斯的有效持续时长。压缩侧的处理：(1) 掩码同时作用到时变协方差（Eq. 23）和时变 opacity（Eq. 24）——掩码参与每个时间步的渲染，梯度自然累计出全时间轴的影响，不需要训后逐时刻统计，这正是它对训后路线的核心优势；(2) R-VQ 作用于时间不变的几何（s_n、sr_n）、旋转轨迹系数 v_n,k 和时间颜色特征 sc_n,7:9；(3) 位置多项式系数 u_n,k 精度要求高且多项式表示已紧凑，不做额外压缩（第 IV-B 节）；(4) 静态颜色特征 sc_n,1:6 由神经场在 canonical 位置处输出（Eq. 25）。

### 关键公式（按原文抄，注明编号）
- 二值掩码 STE（Eq. 5）：`M_n = sg(1[σ(m_n) > ϵ] − σ(m_n)) + σ(m_n)`。sg 是 stop-gradient，1[·] 指示函数，σ 是 sigmoid，ϵ 是掩码阈值：前向用二值 0/1，反向梯度走 σ(m_n)。
- 掩码作用（Eq. 6、7）：`Σ̂_n = R(r_n)S(M_n s_n)S(M_n s_n)^T R(r_n)^T`，`α̂_n(x) = M_n o_n exp(−1/2 (x−p′_n)^T Σ̂′_n (x−p′_n))`——体量和不透明度两头同时乘掩码。
- 掩码正则（Eq. 8）：`L_m = (1/N) Σ_{n=1}^N σ(m_n)`，压低全体掩码激活之和，控制留下来的高斯数量。
- 总损失（Eq. 14）：`L = L_ren + λ_m L_m + L_r + L_s`。λ_m 实景 5e-4、合成场景 4e-3；掩码学习率实景 1e-2、合成 1e-3（第 V-A 节）。
- 残差 VQ（Eq. 9-11）：`r̂_n^l = Σ_{j=1}^l Z^j[i_n^j]`，`i_n^l = argmin_k ||Z^l[k] − (r_n − r̂_n^{l−1})||_2^2`，`r̂_n^0 = 0`；码本损失 `L_r = (1/NC) Σ_{n=1}^N Σ_{k=1}^L ||sg[r_n − r̂_n^{k−1}] − Z^k[i_n^k]||_2^2`——每级只量化上一级剩下的残差，scale 同理（L_s）。
- 神经场颜色（Eq. 12、13）：`c_n(d) = f(contract(p_n), d; θ)`，`contract(p) = p (||p||≤1)；(2−1/||p||)·p/||p|| (||p||>1)`。
- 动态时间 opacity（Eq. 19，STG 原式）：`o_n(t) = s_on exp(−ξ_n|t−μ_n|²)`。
- 渐进正则调度（重点说明）：txt 只给出 λ_m 为固定超参（实景 5e-4、合成 4e-3）与掩码学习率取值，以及神经场学习率在 5K/15K/25K 迭代乘 0.33 的衰减；全文未描述 λ_m 随迭代递增的调度曲线公式。CVPR 版论文中的渐进调度设计在本 txt 中无法确认，按规范注明"原文公式以 PDF 为准"。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | 掩码激活 σ(m_n)：训练期学出的"该高斯对全场景（动态场景为全时间轴）渲染的贡献"，由渲染损失的梯度与 L_m 正则共同塑形 |
| 哪一端算得出 | 训练期编码端学出；训练完掩码即丢弃，码流里只剩留下的高斯，解码端不算 |
| 有无侧信息 | 无（掩码不入码流） |
| 训练期还是编码期 | 训练期（端到端，与致密化同步删除） |
| 和渲染质量的距离 | 间接代理：由渲染损失梯度间接塑形，没有显式"删掉后掉多少质量"的度量；论文的论点是梯度下降学的就是实际渲染影响（第 IV-A 节） |

## 三学：证明与包装（创新怎么立住）

### 实验口径
静态：Mip-NeRF 360 + Tanks & Temples + Deep Blending（3DGS 协议）+ NeRF-Synthetic，30K 迭代；动态：DyNeRF、Technicolor 多视角视频，25K 迭代，底座 STG。对比数字引各原论文 + 自跑 3DGS*/STG*。存储分两档报：Ours（16-bit 位置与标量 + hash + 码本与索引）与 Ours+PP（另加 8-bit 量化、hash 剪枝、Huffman、DEFLATE）。Table VIII 给出逐属性存储分解，口径清楚，这一点比 LightGaussian、EAGLES 都做得细。

### 主结果（注明表号）
- Table I：Mip-NeRF 360 上 3DGS* 746MB/27.46 PSNR → Ours 48.8MB/27.08 → Ours+PP 26.2MB/27.03（约 28 倍）；Tanks & Temples 432MB→39.4MB→18.9MB。
- Table II：Deep Blending 上 Ours 29.79 PSNR 超 3DGS* 的 29.46，43.2MB→21.6MB。
- Table III：NeRF-Synthetic 上 68.1MB→5.55MB（×0.08），+PP 2.47MB（×0.04），FPS 359→545。
- Table IV：DyNeRF 上 STG* 197MB/31.94 → Ours 21.8MB/31.73 → Ours+PP 15.4MB/31.69（对 STG 超 9 倍，+PP 超 12 倍）；Table V：Technicolor 上每帧 1.3MB→0.16MB。

### 消融设计（注明表号）
- Table VI（静态，Playroom+Bonsai 双场景并排）：逐组件累加（Mask→Col→Geo→Half→Post）。Mask 一步把 Playroom 从 2.34M 高斯/553MB 压到 967K/228MB 且 PSNR 微升（29.87→29.91）、FPS 154→254；Col 再压到 59MB；Geo 到 44MB；Half 到 38MB；Post 到 17MB。每步的质量/大小/速度三轴都报。
- Table VII（动态，Painter+Cut Roasted Beef）：Mask 把 Painter 从 553K 压到 145K 高斯（-75%）且 FPS +20%、PSNR 持平（36.21→36.29）；Time、Geo 码本逐步把 84.1MB 压到 14.0MB，+PP 6.56MB。
- Table VIII：逐属性存储账本——3DGS 颜色 606.9MB 对应 Ours 的 hash grid + MLP（+PP 后 hash 8-bit+剪枝+Huffman 为 7.4MB、MLP 0.016MB），是"冗余在哪、压掉多少"最直观的一张表。
- Fig. 9：R-VQ 前后椭球与渲染结果几乎无差别 + 各级码本索引分布（低级幅度大且均匀、高级幅度小且不均），支撑"几何高度相似、残差逐级递减"的假设。

### 贡献列表（论文无编号 contributions，从摘要与引言末段自行归纳并注明）
1. "we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance" —— [机制] 训练期掩码剪枝，静态与动态通用
2. "we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics" —— [机制] 用共享神经场替代逐高斯 SH
3. "Finally, we learn codebooks to compactly represent the geometric and temporal attributes by residual vector quantization" —— [机制] 残差 VQ 压几何与时间属性
4. 扩展版新增：动态场景的 space-time masking 与时间属性码本、对 STG 超 10 倍的参数效率（第 I 节 1)2)3) 点）—— [扩展] 期刊版相对 CVPR 版的增量

### 讲故事方式
主线一句话：3DGS 的冗余分两层——数量冗余（约 2.4 倍）和单高斯参数冗余（SH 48/59、几何高度相似、运动轨迹相似），分别用训练期掩码和"神经场+残差码本"对付。最有说服力的是 Fig. 3（高斯数随迭代曲线：同样 PSNR 下数量减半、训练还更快）配 Table VIII（逐属性存储账本：颜色 606.9MB 压到 7.4MB）；对动态场景则是 Table VII 里 Painter 场景删 75% 高斯不掉点。

## 对 DCCA-GS 的可借鉴点

1. 掩码激活作为贡献信号的引用支撑（Eq. 5、8 与 Table VI）：它证明训练期学出的门控激活与渲染贡献强相关（掩码删 58% 高斯 PSNR 反而微升）。我们 I2 用"掩码激活比例"做公式特征，可以直接引用此文佐证该特征的合理性；它对 scale 与 opacity 两头同时乘掩码的设计（Eq. 6、7）也提示我们 SPA 剪枝之外可考虑在 scale 侧加门控，形成双重证据。
2. R-VQ 只在最后 1K 迭代启用的成本控制技巧（第 III-D 节）：量化感知模块不必全程训练。我们若做量化感知微调实验，可以直接采用"主体训完+短程量化感知"的日程，避免拖慢主训练。
3. 训练期剪枝优于训后剪枝的论据（第 I 节、IV-A 节）：动态场景全时程重要性难在训后估计、训后处理压不了训练显存——这条论据同样支持我们 I6"训练期做敏感度监督"的动机叙述，可在 related work 中引用（注意它对 EAGLES 的描述不准确，引用时只取其对训后路线的批评部分）。
4. Table VIII 式逐属性存储账本的写法：按字段列压缩前后大小并给出总量，与我们与 HAC++ 一致的体积口径天然匹配，论文实验节可以直接采用这种呈现方式。
