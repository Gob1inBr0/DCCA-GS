# 《Reducing the Memory Footprint of 3D Gaussian Splatting》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2406.17074 |
| 发表 | I3D 2024（Proc. ACM Comput. Graph. Interact. Tech. Vol.7 No.1，2024 年 5 月），Inria |
| 阅读材料 | 全文（.lit-cache/txt/2406.17074.txt） |
| 与 DCCA-GS 的关系 | 同类竞品：3DGS 原班团队（Kerbl、Drettakis）的"无锚点、无码流契约"工程压缩路线；剪枝与量化思路可作我们基础措施的对照参考 |

> 一句话定位：3DGS 作者组对自己方法的减负方案——训练期分辨率感知剪枝（砍约 60% 高斯）、训练中途逐高斯自适应降 SH 阶数、训练后 K-means 码本量化加半精度浮点，三者合计磁盘体积 27x、平均只掉 0.21 dB，渲染反而快 1.7x（摘要、Fig. 3）。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾
3DGS 质量、速度都好，但"memory requirements ... unreasonably high"（700 MB–1.2 GB/场景，第 1 节），存和传都难。它把体积拆成三个来源逐一处理（第 1 节）：高斯数量太多（致密化"wasteful"）、SH 系数太多（全场一律 3 阶，没有视相关效果的地方也用 3 阶）、属性精度太高（多数属性不需要高动态范围）。第 3 节给出了体积构成的分析：每高斯 14 + 3·Σ(2i+1) 个 float，3 阶 SH 时共 59 个 float，其中 45 个（76%）是 SH——SH 是大头。

### 支撑观察（直觉 / 数据 / 失败实验）
1. 高斯过密：第 3 节观察"3DGS creates unnecessarily dense sets of primitives"，理由是致密化只看位置梯度，无法预知最终需要多少高斯。
2. SH 冗余：多数场景大部分是漫反射材质，RGB 就够（第 3 节）；同时提醒视相关效果也可能由多个高斯组合形成（反射面后的"反射几何"），所以不能全局砍 SH，要逐高斯判。
3. 精度冗余：opacity、scale、rotation、SH 系数动态范围小、对小误差不敏感，可量化；位置不行——"reducing the accuracy of the positions ... leads to a significant degradation in quality"（第 3 节）。这条和 Niedermayr 第 6.5 节的失败实验互相印证。
4. Fig. 8 的反向证据：删 scaling 最小的高斯比删 opacity 最低的高斯伤质量得多，说明"低不透明度优先删"是对的。

### 显然的路为什么没走
1. 更激进的低 opacity 剪枝（直接提高原有 culling 力度）：能减数量，但第 4.1 节指出高斯过密是空间局部现象，按 opacity 一刀切对准不了"哪里过密"；它的做法是先算空间冗余分数再结合 opacity 挑人，Table 3 证明两者结合比任何单一策略多剪 9%。
2. 全场统一降 SH 阶数（比如全降 0 阶）：会伤视相关效果；它选逐高斯按多视角颜色变化判定保留几阶（第 4.2 节）。
3. 熵编码/更强的编码器：它根本没做（码本索引定长 1 字节，无 Huffman/算术编码）——这条路线留给了 Niedermayr（DEFLATE）和 HAC++ 系（学习型熵模型）。它靠纯"结构压缩"（数量 + 精度）拿 27x，这是和码流系方法对比时最关键的口径差异。

## 二学：机制（洞察怎么变成方法）

### 核心流程
三个组件动在三个不同环节（Fig. 3 的标注："1. Pruning (start-to-end)、2. SH Assignment (halfway)、3. Codebook (after)"）：
- 剪枝：训练全程，每 1000 次迭代执行一次（第 4.1 节）。
- SH 自适应：训练中途一次，15K 迭代（致密化刚停止时）执行，留后半程训练补偿降阶损失（第 4.2 节）。
- 量化：训练结束后的后处理（第 4.3 节），不动训练。
另外两处配合的训练改动：opacity 的 L1 稀疏损失（第 4.1 节，配合剪枝鼓励低贡献高斯出现）和 SH 系数稀疏损失（第 4.2 节，让高阶系数本来就更接近 0，降阶代价更小）。实现层面，15K 之后继续删 opacity < 1/255 的高斯（第 5 节）。

剪枝机制（第 4.1 节，分辨率感知冗余分数）：对高斯 g，找能看到它的视角中最近的一个（像素足迹 a_min 最小），以"每面面积 a_min 的立方体的半对角线"为半径做球（原文："a radius equal to √a_min·√3/2, i.e., half the diagonal of a cube where each face has area a_min"；txt 符号抽取损坏，此处按原文文字描述），数球内与 g 相交的其他高斯个数（k-NN 取 30 邻居做候选，椭球-点相交近似）。冗余分数在空间传播时取相交区域中的最小值（保守原则：只要碰到不冗余的区域就给低分）。阈值 τ_p = (μ + λ_r·σ)，λ_r = 1，再修正为 τ_p = max(μ + λ_r·σ, 3)——分数低于 3 的高斯可能提供细节，不删；分数是整数，即实际保留分数 ≥4 的候选。命中候选不一次全删，只删其中 50%，按低 opacity 挑。另有简单的低 opacity 剪枝：每次删 3% 最低 opacity（上限 0.05）。两路合计约剪 60%。

SH 自适应机制（第 4.2 节）：两个判据。(a) 全视角颜色方差判据：对每个高斯，按平均透射率加权算各视角评估颜色的逐通道均值 μc 和标准差 σc（式 (1)(2)），σc < ε_σ = 0.04 的高斯视颜色为视角无关，只留 0 阶（RGB），关掉高阶。(b) 降阶距离判据：用前 q 阶（q ∈ [0,1,2]）算颜色 c_q，与全 3 阶颜色 c_3 的欧氏距离按视角加权平均得 d_q，取最小的 q 使 d_q < ε_cdist = 0.04，删更高阶。结果分布：89% / 0.1% / 2.7% / 8.2% 的高斯保留 0/1/2/3 阶。存储时按 4 个阶数组分开存（第 5 节）。

量化机制（第 4.3 节）：
- 方案：K-means 码本，索引 1 字节（码本 256 条）。向量属性共享一个码本但三个分量各自索引。量化字段清单：opacity（1 个码本）、scaling 三分量（1 个共享码本）、四元数实部（1 个）、虚部（1 个）、基色 RGB（1 个）、15 组 SH 系数每组 3 通道各 1 个（15 个）——共 20 个码本。体积从 56N×4 字节降到 56N + 20×256×4 字节，码本常数项约 20 KB，可忽略。位置不量化。
- 半精度：剩下没量化的浮点（位置和全部码本条目）转 16-bit half-float。
- 没有 weight prediction：特别要说明，这份 arXiv v1 全文里不存在"weight prediction / 存残差还是存预测"的机制（全文检索无此内容；该机制应属作者后续扩展版本）。本篇的量化只有"码本索引 + 半精度"两种手段，码本存的是聚类中心本身，高斯侧只存索引，没有逐高斯残差。

### 关键公式（按原文抄，注明编号）
式 (1)（第 4.2 节）：μc = Σ_i c_i·T̄_i / Σ_i T̄_i，σc = Σ_i (c_i − μc)²·T̄_i / Σ_i T̄_i——第 i 个视角下该高斯评估颜色 c_i 的透射率加权均值与标准差。T̄_i 由式 (2) 定义：T̄_i = Σ_k T_ik / P，P 是该高斯在第 i 视角溅到的像素数，T_ik 是对像素 k 的透射率。
（剪枝部分原文无编号公式，τ_p、a_min 半径等以文字给出，见上。）

### 信号设计（重点）
| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 两种：剪枝用"空间冗余度"（可见最高分辨率下单位区域的高斯堆积数）+ opacity 作为贡献代理；SH 降阶用"跨视角颜色一致性"（σc、d_q）。都不是渲染梯度类敏感度——与 Niedermayr 和我们 I6 是不同家族的信号 |
| 哪一端算得出（编码端/解码端/仅训练期） | 冗余分数和颜色一致性都需要训练视图集合（透射率要在视角上统计），属编码端/训练期信号；但思路上有解码端化潜力（见可借鉴点） |
| 有无侧信息 | 无额外侧信息。SH 阶数选择通过把高斯分进 4 个阶数组隐式记录（第 5 节），码本随文件存 |
| 训练期还是编码期 | 剪枝训练期循环执行；SH 判定训练期一次（15K）；量化编码期后处理 |
| 和渲染质量的距离（直接测质量还是代理量） | 代理量：冗余度基于采样定理式直觉（分辨率内堆更多高斯无意义），颜色一致性基于数值评估，都不直接测渲染质量 |

## 三学：证明与包装（创新怎么立住）

### 实验口径
- 体积口径：磁盘文件大小（表内记 "Mem"，MB）。计入项：全部属性（码本索引 1 字节/分量 + 码本开销 20×256×4 字节 + half-float 位置与码本条目）。不计入项：没有任何解码器/熵模型网络——它没有学习型解码，也就没有 HAC++ total_MB 口径里的 mlp/hash/bounds/header 项；也没有熵编码收益。27x 全部来自"高斯更少 + SH 更少 + 精度更低"。
- VRAM 与磁盘不同：当前实现是解析时解压、渲染不走码本（第 5 节），第 6.1 节明说"integrating codebooks and half-float type handling into the renderer would enable the same reduction for a scene's VRAM consumption"——即当前 VRAM 未同步缩到 1/27。
- FPS 口径：只测 CUDA 光栅化例程本身，不含 blitting、swapchain 等图形 API 开销（第 6 节开头注明）。
- 数据集与协议：3DGS 原论文全套场景（MipNeRF360 全部、Deep Blending 2 景、Tanks&Temples 2 景），30K 迭代；基线用官方 GitHub 代码复现（3DGS*，数字比原论文略好，Table 2 脚注）。RTX A6000。

### 主结果（注明表号）
- Table 2：MipNeRF360 上 3DGS* 744 MB / 27.42 PSNR / 178 FPS → Ours 29 MB / 27.10 PSNR / 284 FPS（25.7x、掉 0.32 dB、快 1.6x）；Tanks&Temples 412→14 MB（27.6x，23.66→23.57）；Deep Blending 630→18 MB（34.8x，29.47→29.63 反升）。三档配置：Ours 29/14/18 MB，Low 46/21/35 MB，High 23/10/13 MB。
- Fig. 3 标注三组件的累计贡献：剪枝 +0.03 dB / 2.37x；+SH 自适应 −0.14 dB / 8.0x；+量化 −0.21 dB / 27.4x。
- Fig. 1 与 Table 7（bicycle 逐场景）：1.4 GB（表中 1362 MB）→48 MB，PSNR 25.10→25.06，高斯 6.05M→2.41M，115→260 FPS；手机 WebGL 下载 140s→4.8s（Fig. 1 标注；第 6.1 节另报本地 wifi 下 120s→5s、约 24x）。
- Table 4（half-float 消融）：Mip360 上仅 K-means 为 38 MB（27.17 PSNR），加 half-float 到 29 MB（27.10），质量只掉 0.07 dB；量化后位置占模型体积从 5% 升到 28%——位置是压不动的存量老大。

### 消融设计（注明表号）
- Table 1：三组件渐进累加表，每个数据集一列，逐级报 Mem 倍数和指标（Mip360：744→339（×2.2）→102（×7.5）→29 MB（×25.7））。
- Table 3（剪枝策略对照，值得学）：同一数据集上对比四种剪法——只按 opacity（Mip360 剩 1.64M，48%）、只按冗余分数+低 opacity 挑（1.88M，56%）、冗余分数+随机挑（1.91M，57%）、完整方法（1.46M，43%）。"同候选集、换挑选策略"的对照设计直接干净。
- Table 4：量化方案替换（仅 K-means 对 K-means+half-float），并报位置占比，说明下一步瓶颈在位置。
- 附录 Table 5/6：与并发预印本（EAGLES、Compact3D、Compressed3D、Compact3DGS、LightGaussian）同数据集对比，分"含/不含版权受限两景"两张表——处理数据集口径争议的方式值得学。
- 附录正文口径争论（与我们的对比写作直接相关）：Compact3D 只报三数据集平均 54 MB，本文全量法 26 MB、Low 档 41 MB（附录正文文字）；对 Niedermayr（Compressed3D）的批评是"压缩率 impressive，但其 DEFLATE 约占 2x，不能直接映射到渲染期 VRAM"。

### 贡献列表（原文照抄 + 逐条标注）
1. "An efficient, resolution-aware primitive pruning approach, that reduces unnecessary points during optimization, leading to a total reduction of the primitive count by 60%." —— [机制] 分辨率感知冗余剪枝，训练期环节。
2. "An adaptive adjustment method to choose the number of SH bands required for each 3DGS primitive, significantly reducing overall memory footprint." —— [机制] 逐高斯自适应 SH 阶数，训练中途环节。
3. "A codebook-based quantization method, together with a half-float representation for more efficient storage of the representation, resulting in further memory reduction." —— [工程] 训练后量化和半精度，纯后处理环节。

### 讲故事方式
主线一句话：把体积拆成数量、SH、精度三笔账，各还一笔。最有力的图是 Fig. 3——一张图同时交代三个组件、各自动在训练哪个环节、累计 dB 损失和累计压缩倍数（+0.03/2.37x → −0.14/8.0x → −0.21/27.4x），方法结构一目了然；配套 Table 1 的渐进数字表。这种"流水线示意图上直接标各段收益"的画法对我们呈现"基础措施 + 主创新"的分解非常适用。

## 对 DCCA-GS 的可借鉴点

1. 分辨率感知冗余分数可能改造成解码端可重算特征（对着我们的主要问题）。它的冗余度=近邻堆积数，且用"最近视角像素足迹"定标——本质是局部密度，但比我们 I2 现用的局部密度多了几何分辨率含义（同样堆积，分辨率高时不算冗余）。建议实验：在我们数据上算这个"像素足迹加权的局部密度"特征，测它与式 (3) 敏感度（Niedermayr 协议）的相关性；若显著高于现有特征的 |r|<0.005，就有资格进 AQM 的公式特征组，仍是零侧信息。
2. 剪枝消融的对照设计（Table 3）：同候选集下换挑选策略（低 opacity 对随机）。我们验证 I6 监督信号有效性时可以照做——同一 AQM 结构，监督信号分别用 I6 梯度 EMA、随机、或无监督，其余全同，直接归因到信号本身。
3. SH 占 76% 体积的账目（第 3 节）提醒我们：HAC++ 口径下 feat/offsets 的分级量化是对的路，但论文里报体积构成表（各字段占比、量化前后）能让"为什么动这些字段"更有说服力，照它的方式列一张。
4. 口径对照要写清（三学重点，供相关工作章节用）：它的 27x 无熵编码、无解码器网络，属于"结构压缩"；HAC++ 系 total_MB 含 mlp 等解码侧开销但享受熵编码收益；Niedermayr 含 DEFLATE 约 2x。三家的 MB 数字互相不可直接比，我们报告对比时需要一张口径说明表（是否含解码器/是否含熵编码/位置精度）。
