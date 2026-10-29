# 《Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields》精读笔记

- arXiv：https://arxiv.org/abs/2408.03822
- 发表：arXiv 预印本（IEEE 期刊版式，Manuscript received August 2024，未标明期刊名）；早期版本发表于 CVPR 2024（highlight），题为 "Compact 3D Gaussian Representation for Radiance Field"，即文中引文 [20]
- 作者：Joo Chan Lee、Daniel Rho、Xiangyu Sun、Jong Hwan Ko、Eunbyung Park（成均馆大学/UNC，第 1 页）。本编号是 CVPR 2024 同名工作的期刊扩展版，第一作者 Joo Chan Lee；文件名中的 "Hanson" 系误标（Alex Hanson 是 Speedy-Splat 的作者），与本论文无关，文件名按要求保留但以此为准

> 一句话定位：用训练期可学习掩码（同时作用在 scale 和 opacity 上）删掉冗余高斯，SH 换成 hash grid+小 MLP 的神经场，scale/rotation/时间属性用残差向量量化——静态场景对 3DGS 压缩超 25 倍、动态场景对 STG 压缩超 12 倍，渲染还更快（摘要；Table I、Table IV）。

## 问题与支撑观察

3D-GS 需要海量高斯才能保质量，存储常超 1GB（第 I 节），而致密化"produces myriads of redundant and insignificant Gaussians"（第 III-A 节）。对训后剪枝路线（LightGaussian 等）的批评：静态场景训后估计显著性可行，动态场景要衡量每个高斯在整个时间轴上的重要性，训后再算很困难（第 I 节、IV-A 节）；且训后处理无法降低训练期显存。所以它把剪枝搬进训练期，用梯度下降直接学出每个高斯对全部时间步渲染的影响。

- 数量冗余（Fig. 3 + Table VI 前两行，Bonsai 场景）：2.34M 高斯 PSNR 29.87，删到 967K 后反而 29.91——"×2.42 Gaussians show similar performance"（第 III-A 节），约六成高斯是冗余的。
- 每高斯参数冗余（第 III-C 节）：SH 占每高斯 59 个参数中的 48。
- 逐属性存储分解（Table VIII，Mip-NeRF 360 平均）：3DGS 的 746MB 里颜色（SH）占 606.9MB，其余为 scale 37.9MB、位置 37.9MB、opacity 12.6MB、rotation 8.3MB——颜色是绝对大头。
- 几何相似性（第 III-B 节）："the geometrical shapes of most Gaussians are very similar, showing only minor differences in scale and rotation"，且场景由大量小高斯组成、单个基元不需要多大多样性——所以 scale/rotation 可以共用小码本（码本大小 C=64、级数 L=6，第 V-A 节）。Fig. 9-(b) 可视化 learned 码本索引：低阶码本幅度大、分布均匀，越往后级码字幅度越小、分布越不均，各级残差确实在递减。
- 时间冗余（第 I 节）：动态场景里"groups of moving parts typically follow similar motion patterns"，静止区域完全不动——时间属性同样可码本化。
- 一次性 VQ（K-means 直接量化）计算复杂度和 GPU 显存高（第 III-B 节，引 SoundStream），改用 L 级级联残差 VQ，每级码本可以很小。
- 第 II-A-3 节点名批评 EAGLES："EAGLES controlled the number of Gaussians by just adjusting the densification schedule, resulting in a sub-optimal reduction"，并称自己是唯一在训练期真正 mask 掉无效高斯的工作。（注意：这句对 EAGLES 的描述不准确，EAGLES 第 4.3 节有显式 influence 剪枝，不只是调致密化日程。）

## 方法

静态场景（第 III 节，底座 3DGS）：

1. 训练期剪枝：每个高斯带一个可学习的 mask 参数 m_n，前向生成二值掩码 M_n（STE，Eq. 5：`M_n = sg(1[σ(m_n) > ϵ] − σ(m_n)) + σ(m_n)`，sg 是 stop-gradient，前向用二值 0/1，反向梯度走 σ(m_n)），同时作用到 scale（Eq. 6）和 opacity（Eq. 7）：`Σ̂_n = R(r_n)S(M_n s_n)S(M_n s_n)^T R(r_n)^T`，`α̂_n(x) = M_n o_n exp(−1/2 (x−p′_n)^T Σ̂′_n (x−p′_n))`——体量和不透明度两头同时乘掩码。每次致密化步骤按掩码删除被 mask 的高斯；与 3D-GS 中途停止致密化不同，mask 全程贯穿训练，训练期显存也随之下降（Fig. 3）；训练结束后被 mask 的高斯直接删除，掩码本身不入码流。掩码正则（Eq. 8）：`L_m = (1/N) Σ_{n=1}^N σ(m_n)`；总损失（Eq. 14）：`L = L_ren + λ_m L_m + L_r + L_s`。λ_m 实景 5e-4、合成场景 4e-3；掩码学习率实景 1e-2、合成 1e-3（第 V-A 节）。txt 只给出这些固定超参，以及神经场学习率在 5K/15K/25K 迭代乘 0.33 的衰减；全文未描述 λ_m 随迭代递增的调度曲线公式（CVPR 版论文中的渐进调度设计在本 txt 中无法确认，原文公式以 PDF 为准）。
2. 颜色（第 III-C 节）：无界位置先 contract 到有界（Eq. 13：`contract(p) = p (||p||≤1)；(2−1/||p||)·p/||p|| (||p||>1)`），过 hash grid（2 通道特征、16 个分辨率 16~4096）+ 2 层 64 通道 MLP 输出 0 阶 SH 再转 RGB（Eq. 12：`c_n(d) = f(contract(p_n), d; θ)`）——不再逐高斯存 48 维 SH。
3. 几何（第 III-B 节）：scale 和 rotation（在 mask 之前）过 L 级残差 VQ（Eq. 9-11：`r̂_n^l = Σ_{j=1}^l Z^j[i_n^j]`，`i_n^l = argmin_k ||Z^l[k] − (r_n − r̂_n^{l−1})||_2^2`，`r̂_n^0 = 0`；码本损失 `L_r = (1/NC) Σ_{n=1}^N Σ_{k=1}^L ||sg[r_n − r̂_n^{k−1}] − Z^k[i_n^k]||_2^2`——每级只量化上一级剩下的残差，scale 同理 L_s）；码本 K-means 初始化，只在最后 1K 迭代启用 R-VQ 损失（其余时间 L_r=L_s=0）以控制开销（第 III-D 节）。
4. 后处理（Ours+PP，第 V-A-3 节）：位置和标量属性存 16 位半精度；hash 参数与标量做 8-bit min-max 量化；剪掉绝对值小于 0.1 的 hash 参数；高斯按 Morton 序排列；8-bit 值和 R-VQ 索引做 Huffman 编码再 DEFLATE 压缩。

动态场景的 timescale 处理（第 IV 节，底座 STG）：STG 给每个高斯一个时间中心 μ_n（该高斯最显著的时刻），位置/旋转轨迹用多项式系数建模（Eq. 15-16），时间可见性用径向基 `o_n(t) = s_on·exp(−ξ_n|t−μ_n|²)`（Eq. 19），其中 ξ_n 是时间尺度，表示该高斯的有效持续时长。压缩侧的处理：(1) 掩码同时作用到时变协方差（Eq. 23）和时变 opacity（Eq. 24）——掩码参与每个时间步的渲染，梯度自然累计出全时间轴的影响，不需要训后逐时刻统计；(2) R-VQ 作用于时间不变的几何（s_n、sr_n）、旋转轨迹系数 v_n,k 和时间颜色特征 sc_n,7:9；(3) 位置多项式系数 u_n,k 精度要求高且多项式表示已紧凑，不做额外压缩（第 IV-B 节）；(4) 静态颜色特征 sc_n,1:6 由神经场在 canonical 位置处输出（Eq. 25）。

## 实验口径

静态：Mip-NeRF 360 + Tanks & Temples + Deep Blending（3DGS 协议）+ NeRF-Synthetic，30K 迭代；动态：DyNeRF、Technicolor 多视角视频，25K 迭代，底座 STG。对比数字引各原论文 + 自跑 3DGS*/STG*。存储分两档报：Ours（16-bit 位置与标量 + hash + 码本与索引）与 Ours+PP（另加 8-bit 量化、hash 剪枝、Huffman、DEFLATE）。Table VIII 给出逐属性存储分解。

## 主结果

- Table I：Mip-NeRF 360 上 3DGS* 746MB/27.46 PSNR → Ours 48.8MB/27.08 → Ours+PP 26.2MB/27.03（约 28 倍）；Tanks & Temples 432MB→39.4MB→18.9MB。
- Table II：Deep Blending 上 Ours 29.79 PSNR 超 3DGS* 的 29.46，43.2MB→21.6MB。
- Table III：NeRF-Synthetic 上 68.1MB→5.55MB（×0.08），+PP 2.47MB（×0.04），FPS 359→545。
- Table IV：DyNeRF 上 STG* 197MB/31.94 → Ours 21.8MB/31.73 → Ours+PP 15.4MB/31.69（对 STG 超 9 倍，+PP 超 12 倍）；Table V：Technicolor 上每帧 1.3MB→0.16MB。

## 消融

- Table VI（静态，Playroom+Bonsai 双场景并排，逐组件累加 Mask→Col→Geo→Half→Post）：Mask 一步把 Playroom 从 2.34M 高斯/553MB 压到 967K/228MB 且 PSNR 微升（29.87→29.91）、FPS 154→254；Col 再压到 59MB；Geo 到 44MB；Half 到 38MB；Post 到 17MB。每步的质量/大小/速度三轴都报。
- Table VII（动态，Painter+Cut Roasted Beef）：Mask 把 Painter 从 553K 压到 145K 高斯（-75%）且 FPS +20%、PSNR 持平（36.21→36.29）；Time、Geo 码本逐步把 84.1MB 压到 14.0MB，+PP 6.56MB。
- Table VIII：逐属性存储账本——3DGS 颜色 606.9MB 对应 Ours 的 hash grid + MLP（+PP 后 hash 8-bit+剪枝+Huffman 为 7.4MB、MLP 0.016MB）。
- Fig. 9：R-VQ 前后椭球与渲染结果几乎无差别 + 各级码本索引分布（低级幅度大且均匀、高级幅度小且不均），支撑"几何高度相似、残差逐级递减"的假设。
