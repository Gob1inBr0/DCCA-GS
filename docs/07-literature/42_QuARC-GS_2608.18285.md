# 《QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2608.18285 |
| 发表 | arXiv 预印本（v1，2026-08-18，eess.IV；AAAI 排版格式的短文，PDF 共 9 页：正文与图表约 7 页加参考文献 2 页，论文未标注会议接收信息）。开源：https://github.com/high-performance-computational-optics/QuARC-GS |
| 阅读材料 | 全文（.lit-cache/txt/2608.18285.txt，摘要至参考文献完整）。公式 (2)(3)(4) 的求和号与取整符号在 txt 抽取中有损，已用 pymupdf 抽取 PDF 第 4、5 页文本逐式核对；表 1-3 数字用 PDF 第 6 页文本复核一致；图 4 的坐标轴数值是矢量图形，无法从文本层提取 |
| 与 DCCA-GS 的关系 | 重点竞品（动态场景流式压缩分支，量化锚点残差路线）：锚点、量化、熵编码三个要素都与我们重合；但它在动态场景做时间维的逐帧在线流式，每帧一个固定质量档，不做单文件渐进码流，量化步长固定、不按内容自适应 |

> 一句话定位：给动态场景的在线自由视角视频传输压每帧载荷——用一个 canonical（基准）帧加每帧高度压缩的残差表示场景，三层锚点层级让少量锚点共享运动，量化感知训练把"几乎不动的锚点"直接打进零 bin（静态/动态的区分由熵编码器白拿，不用传掩码），增密只允许发生在"形变解释不了的外观变化"处；结果是每帧 36 KB（N3DV）达到与 400 KB 的 ReCon-GS 完全相同的 32.16 dB（表 1），每帧存储最高省 11 倍（摘要、第 5 节）。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

它认定的矛盾是：在线自由视角视频（FVV）要"存储小、重建快、质量高"三者同时成立，而现有方法在长视频下存储需求会叠加（第 1 节："memory demands compound"）。它把现有紧凑 FVV 方法分成两条路线，并各指出一个缺口（第 1 节）：

- **率感知编码路线**（QUEEN、4DGC）：QUEEN 对非位置属性残差做量化潜码解码，省了很多，但"运动在单个高斯的粒度上仍然难以低精度量化"；4DGC 用紧凑运动网格加稀疏补偿高斯，运动网格经过率失真联合优化后"仍然是帧间载荷的主体部分"。
- **锚点参数化路线**（iFVC、ReCon-GS、HiCoM、ComGS）：让空间上相关的多个高斯共享形变，明显减少了独立运动参数的数量，但"共享层级场里的锚点属性残差仍然带来高存储成本"。

于是它点出自己的切入点（第 1 节原话）："This leaves direct low-precision coding of small anchor residuals as a distinct compression opportunity."——锚点残差里大量是小到无所谓的运动，直接对它们做低精度编码是一个没人占住的机会。

### 支撑观察（直觉 / 数据 / 失败实验）

- **"大多数锚点几乎不动"**：写在 3.4 节开头（"the deformation field is highly redundant: most anchors barely move between frames"），是作者观察，没有单独的统计图表支撑；间接证据有两条：图 2(b) 画了形变直方图示意（静态锚点堆积在零附近），表 3 的 w/o quantization 一行显示关掉量化后质量逐位不变、存储从 36/441 涨到 402/441 KB——说明冗余确实几乎全部集中在小残差上。
- **"形变解释不了的变化才需要新高斯"**：3.5 节的推理——能被形变解释的区域，相邻两帧的重建误差相似，损失差梯度小；真正在变的区域梯度大。梯度是在施加形变之后算的，所以天然排除了几何运动的影响。
- **无门控增密导致体积线性增长**：图 5 显示关掉变化门控后每帧净增高斯数大得多，场景尺寸随时间稳定增长；开门控后场景尺寸在整段序列里近似恒定（第 4.5 节）。

### 显然的路为什么没走

- **没给每个高斯传独立运动向量**：3.3 节开头直接说代价"prohibitively expensive"（贵得没法用）。
- **没用单层稀疏锚点**：同一段指出单层锚点"lacks the capacity to model complex scene dynamics"（容量不够，建不了复杂动态）。
- **没走"全精度训练、编码时再取整"**：3.4 节给出两个理由——小到察觉不到的运动也要花码率；事后取整造成训练值和传输值不一致，直接损害重建。所以把取整搬进训练循环（STE），让模型在量化值上优化。
- **没传静态/动态掩码**：3.4 节的替代做法是让量化本身完成分割——小于半个步长的残差取整后就是单位变换，全部堆进零 bin，熵编码器自动吃到概率质量集中，"sparsity is captured directly by the entropy coder, without requiring any separate mask"。
- **没做无差别增密**：3.5 节指出无差别增密会把静态内容重新编码一遍，"unnecessarily increases the representation size"；替代方案是用帧间损失差的梯度门控增密。
- **没走视频编解码器混合路线**：相关工作 2.3 节列了 CodecGS、GIFStream 等把高斯属性重组成视频码流的方法，但本文保持在纯 3DGS 优化范式内，不引入标准视频编码器。

## 二学：机制（洞察怎么变成方法）

### 核心流程

先说时间线设定：这是在线逐帧优化范式，**每帧的训练过程就是编码过程**（第 3.2 节；4.2 节"strictly causal online manner"，严格因果的在线方式）。全部流程分三段。

**第 0 段：仅首帧（t = 0），canonical 基础模型。**
输入：t=0 的多视角图像。做法：标准 3DGS 优化 15,000 次迭代，球谐阶数上限设 1，训练中对高斯位置注入噪声（系数 0.01，取自 HiCoM，Gao et al. 2024），得到紧凑基础高斯集 G_B（3.2 节、4.2 节）。输出：canonical 模型，之后所有帧都由它经形变得到（图 2(a)）。基础模型本身要传输/存储：表 1 存储列是"不含初始帧/含初始帧"两个数，36/56 KB 相差 20 KB/帧，按 300 帧均摊倒推基础模型约 5.9 MB——这是推算值，论文未直接报告基础模型体积。

**编码端：每个后续帧（t > 0），六步。**

1. **热启动**：从上一帧优化完的模型初始化当前帧（3.2 节）。
2. **采样锚点与绑定（零字节成本）**：在当前高斯位置的体素网格上均匀下采样，生成三层锚点：层数 L=3；最细层锚点数 A1=⌈N/m⌉（N 是高斯总数，m 是每个锚点最多绑几个高斯），之后每层 A_ℓ=⌊A_{ℓ−1}/r⌋，相邻层锚点数比 r=3；总锚点数 A=Σ A_ℓ（3.3 节；4.2 节给出 m=4，即每格最多 4 个高斯）。每个高斯在每一层用最近邻搜索（1-NN）绑到锚点，得绑定图 I∈{1,…,A}^{L×N}。关键设计：体素下采样和 1-NN 绑定都是"以（帧号 t，层号 ℓ）为种子的确定性伪随机函数"，canonical 几何加 t 就能唯一决定锚点集合和绑定图，所以这两样**一个字节都不用传**，只传形变场 D（3.3 节 "Zero-cost binding"）。
3. **量化感知形变优化（100 次迭代，4.2 节）**：输入是当前帧多视角真值图。只优化锚点形变场 D={δa}∈R^{A×7}——每个锚点存 7 维：平移 δt∈R³ 加相对单位四元数的旋转残差 δr∈R⁴（3.3 节）。前向传播中按式 (3) 把残差取整到固定步长（STE：前向取整、反向梯度原样通过全精度 D），渲染时按式 (2) 把三层锚点形变求和聚合到每个高斯上（3.4 节）。输出：量化后的形变场 D̂，训练中渲染用的就是它，和将来传输的值完全一致。
4. **熵编码形变场**：量化后平移、旋转都是整数值符号，两部分**分别独立**熵编码，编码代价用经验香农熵估计（3.4 节）。静态锚点全部落在零 bin，稀疏性由熵编码器直接吃到，不传掩码。
5. **变化门控增密（再 100 次迭代，4.2 节）**：先施加形变（式 2），然后在一个固定的 4 训练视角子集上计算式 (4)——当前帧与前一帧 L1 损失之差对高斯视空间位置求梯度、取模长；再用式 (5) 归一化成权重 wi，去调制自适应克隆-分裂（clone-and-split）之前的累积增密梯度（3.5 节）。效果：只有"形变解释不了的外观变化"大的区域才长新高斯，静态难区域不再重复分配外观容量（3.5 节）。新增高斯的属性进入帧载荷。
6. **组装与输出帧载荷**：量化形变符号 + 新增高斯属性，熵编码后输出（图 2(d)）；同一份量化残差同时用于热启动下一帧（图 2(d)）。

**解码端。** 逐帧：解码形变残差符号与新增高斯属性 → 用种子重算锚点集合与绑定图 → 按式 (2) 对 canonical 高斯施加形变 → 渲染自由视角视频（图 2(d) "Decode"）。帧间是因果链（每帧依赖前一帧结果），论文未讨论随机跳帧接入或丢包恢复。

量化动在第 3-4 步（训练循环内 + 熵编码前），熵编码在第 4 步和第 6 步，增密在第 5 步。没有剪枝模块——表 3 题注提到"New GS/f 排除剪枝"，说明流程里有剪枝但论文正文未描述其规则。

### 关键公式（按原文抄，注明编号）

txt 中式 (2)(3)(4) 的数学符号抽取有损，以下按 pymupdf 抽取的 PDF 第 4、5 页文本层抄录；原文公式以 PDF 为准。

式 (1)（第 3.1 节，3DGS 原始目标函数）：
L_3DGS = (1−λ)·L1 + λ·L_D-SSIM。
L1 是逐像素绝对误差，L_D-SSIM 是 D-SSIM 损失，λ 权衡两者。

式 (2)（第 3.3 节，形变聚合）：
μ′_i = μ_i + Σ_{ℓ=1}^{L} δt_{I[ℓ,i]}，  q′_i = normalize( Σ_{ℓ=1}^{L} δr_{I[ℓ,i]} )。
μ_i、q_i 是高斯 i 的原始位置与旋转；I[ℓ,i] 是高斯 i 在第 ℓ 层绑定的锚点编号；δt、δr 是该锚点的平移与旋转残差；三层求和后得到形变后的位置 μ′_i 和旋转 q′_i（旋转残差相对单位四元数定义，直接求和再归一化，不是四元数乘法组合，式 2 原文如此）。

式 (3)（第 3.4 节，STE 量化）：
δ̂t_a = Δt · ⌊δt_a / Δt⌉，  δ̂r_a = e + Δr · ⌊(δr_a − e) / Δr⌉。
⌊·⌉ 是取整到最近整数；e=(1,0,0,0) 是单位四元数；Δt、Δr 分别是平移和旋转的量化步长。前向传播用量化值 δ̂（通过式 2 作用于高斯），反向传播梯度流过全精度 δ（STE）。

式 (4)（第 3.5 节，帧间变化信号）：
g_i = ‖ ∇_{x_i} ( L1(Î, I_t) − L1(Î, I_{t−1}) ) ‖₂。
L1 是平均绝对误差；Î 是渲染图像；I_t、I_{t−1} 是当前帧和前一帧的真值图；x_i 是高斯 i 的视空间位置；∇_{x_i} 是对渲染损失求关于 x_i 的梯度。含义：在施加形变之后，当前帧与前一帧重建误差之差在某高斯位置处的梯度模长——形变能解释的区域两帧误差相似、梯度小；真正在变化的区域梯度大。

式 (5)（第 3.5 节，梯度归一化）：
w_i = g_i / (g_i + median(g))。
g_i 是累积梯度模长，median(g) 取全体中位数。归一化后的权重 wi 用于在自适应克隆-分裂之前调制累积增密梯度（3.5 节）。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 两个信号。其一，锚点运动残差 δa 本身就是"运动量"信号：量化后幅度小于半个步长的残差归零，等价于把"够不动的锚点"判定为静态（3.4 节）。其二，帧间外观变化信号 g_i（式 4）：形变施加之后两帧渲染损失之差的视空间梯度模长，度量"几何运动解释不了的外观变化"（3.5 节） |
| 哪一端算得出（编码端/解码端/仅训练期） | 都只在编码端（在线优化端）算。解码端不算任何信号：它只收熵编码后的量化残差与新增高斯属性，锚点与绑定靠种子重算（3.3 节） |
| 有无侧信息 | 无显式侧信息。要传的只有：量化形变符号 + 新增高斯属性（图 2(d)）。不传的：绑定图（种子重算）、静态/动态掩码（零 bin 吸收）、量化步长（固定超参数，本来就不用传） |
| 训练期还是编码期 | 训练期=编码期，两者合一（在线逐帧优化范式，3.2 节）。量化直接写进训练循环（STE），不是事后处理 |
| 和渲染质量的距离 | 量化无距离：训练中渲染用的就是传输值，所见即所得（3.4 节）。g_i 是质量的代理量（损失差的梯度，间接），但消融表 3 证明门控开关对 PSNR 的影响只有 0.07 dB（32.16→32.09） |

### 与码流组织相关的细节（压缩类论文必填）

- **每帧载荷的内容**：量化后的锚点形变符号（平移、旋转两部分分别独立熵编码）+ 变化门控后新增高斯的属性（3.4 节、3.5 节、图 2(d)）。新增高斯属性用什么精度、什么编码器，论文未报告——只说"entropy-coded and transmitted"（图 2(d)），也没有逐字段的字节分解。
- **零字节元数据靠重算**：锚点集合与绑定图由"canonical 几何 + 种子（t, ℓ）"确定性重算，一个字节不占（3.3 节 "Zero-cost binding"）。这是全文与"解码端重算"原则最直接呼应的一处。
- **静态锚点零开销**：量化把小于 Δ/2 的残差归入零 bin，稀疏性由熵编码的概率集中自动兑现，不需要掩码或门控项（3.4 节）。
- **无渐进层、无可截断性**：每帧一个固定质量档的压缩载荷（36 KB/帧，N3DV，4.3 节）。没有单文件多画质档，没有字节前缀可截断结构；文中的 "streaming" 指时间维的逐帧在线传输，不是质量维的渐进。相关工作里提到 PCGS（Chen et al. 2026）做静态场景的渐进锚点压缩和 4DGCPro（Zheng et al. 2025）做渐进体积视频，但本文自己不做渐进。
- **帧间因果依赖**：每帧从上一帧热启动，严格因果在线（4.2 节）；解码端必须从头逐帧解，论文未讨论随机接入。
- **初始帧单独成档**：表 1/表 2 存储列分"不含初始帧/含初始帧"两个数（36/56、18/26 KB），初始帧模型是唯一的"基础层"，但它不是一个可独立截断的分层码流，只是计账方式的差别（表 1、表 2 题注）。
- **熵编码的实现深度**：只到"整数值符号 + 经验香农熵估计代价"（3.4 节），未说明用算术编码还是其他具体编码器，也未给出实测码长与熵估计的差距。

## 三学：证明与包装（创新怎么立住）

### 实验设置与统计方式

- **数据集（4.1 节）**：N3DV（Neural 3D Video，6 个室内动态场景，每场景 18-21 路同步视频，各 300 帧、30 FPS、2704×2028）；MeetRoom（3 个动态场景，13 路同步视频，各 300 帧、30 FPS、1280×720）。每个多视角视频的**第一个视角留作测试**，其余用于训练。
- **存储的统计方式（表 1、表 2 题注）**：KB/帧，分"不含初始帧/含初始帧"两数（斜杠分隔）；训练时间含首帧训练。论文未报告逐字段字节分解。
- **训练配置（4.2 节）**：首帧 15,000 次迭代、SH 阶数 1、噪声注入 0.01；后续每帧 100 次迭代量化形变 + 100 次迭代变化门控增密；三层密度自适应锚点层级、每格最多 4 个高斯；锚点超参数沿用 ReCon-GS 默认值；Adam 优化；PyTorch，Ubuntu 24.04，NVIDIA RTX PRO 6000。量化步长：平移 Δt=0.0017、旋转 Δr=0.01（世界空间固定值，全场景统一）。门控用固定 4 个训练视角。
- **对比对象（表 1、表 2）**：离线方法 STG、SaRO-GS、Swift4D、SplineGS；在线方法 Dynamic 3DGS、StreamRF、3DGStream、4DGC、QUEEN、HiCoM、ComGS（s/l 两个规格）、ReCon-GS。其中 3DGStream 和 ReCon-GS 带 dagger 号，是作者用官方代码在同一环境复现的，表 1 的复现值取 3 次运行均值（表 1 题注）。
- **质量指标**：PSNR、SSIM、LPIPS（表 1）；表 2 无 LPIPS。另有渲染速度（FPS）与训练时间（秒）。

### 主结果表抄录（注明表号）

**表 1（N3DV 数据集，共 14 个方法行，全部抄录）。** 列：PSNR (dB) 越高越好、SSIM 越高越好、LPIPS 越低越好、存储 (KB) 越低越好（"不含初始帧/含初始帧"）、训练 (秒) 越低越好（含首帧）、渲染 (FPS) 越高越好。表 1 题注：带 † 的方法由作者用官方代码在同一环境复现、取 3 次运行均值；每列前三名在原文中高亮。

| 类别 | 方法 | PSNR (dB) | SSIM | LPIPS | 存储 (KB) | 训练 (秒) | 渲染 (FPS) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 离线 | STG | 32.05 | 0.948 | – | 670 | 20 | 140 |
| 离线 | SaRO-GS | 32.15 | – | – | 1000 | – | 40 |
| 离线 | Swift4D | 32.23 | – | – | 400 | 5.0 | 125 |
| 离线 | SplineGS | 32.60 | – | – | – | 11 | 76 |
| 在线 | Dynamic 3DGS | 30.67 | – | – | –/9200 | 560 | – |
| 在线 | StreamRF | 30.68 | 0.930 | – | 17700/31500 | 15 | 12 |
| 在线 | 3DGStream† | 31.35 | 0.948 | 0.130 | 7600/7800 | 8.1 | 245 |
| 在线 | 4DGC | 31.58 | 0.943 | – | –/500 | 50 | 168 |
| 在线 | QUEEN-1 | 32.19 | 0.946 | 0.136 | –/750 | 7.9 | 248 |
| 在线 | HiCoM | 31.17 | – | – | 900 | 11 | 260 |
| 在线 | ComGS-s | 31.87 | 0.943 | 0.132 | 49 | 37 | 91 |
| 在线 | ComGS-l | 32.12 | 0.945 | 0.129 | 106 | 43 | 147 |
| 在线 | ReCon-GS† | 32.16 | 0.951 | 0.129 | 400/440 | 4.7 | 272 |
| 在线 | **QuARC-GS** | 32.16 | 0.951 | 0.129 | 36/56 | 5.0 | 243 |

注：表 1 文本层印作 "QUEEN-1"，图 1 图例作 "QUEEN-l"（QUEEN 论文有 m/l 两个规格变体），应为同一方法，无衬线字体里 1 与 l 难分辨，此处照抄表 1。图 1（对数存储轴 10¹–10⁴ KB，PSNR 31.2–32.4，气泡大小表示训练时间 5/20/50 秒）里 QuARC-GS 位于 36 KB、32.16 dB，是图中所有方法里存储最低、质量与最高档持平的点。

**表 2（MeetRoom 数据集，共 4 行，全部抄录；此表无 LPIPS 列）。**

| 方法 | PSNR (dB) | SSIM | 存储 (KB) | 训练 (秒) | FPS |
| --- | --- | --- | --- | --- | --- |
| 3DGStream† | 29.30 | 0.948 | 4000/4100 | 4.77 | 260 |
| HiCoM | 26.73 | – | 600 | 9.8 | 258 |
| ReCon-GS† | 30.68 | 0.949 | 300/390 | 3.2 | 303 |
| **QuARC-GS** | 30.67 | 0.949 | 18/26 | 3.3 | 271 |

**表 3（N3DV 上的组件消融，共 3 行，全部抄录）。** "New GS/f" 是每帧新增高斯数（排除剪枝，表 3 题注）；存储分"不含初始帧/含初始帧"。

| 变体 | PSNR (dB) | SSIM | LPIPS | 存储 (KB/帧) | 新增高斯/帧 |
| --- | --- | --- | --- | --- | --- |
| 完整版（Full, Ours） | 32.16 | 0.951 | 0.129 | 36/56 | 153 |
| 去掉量化（w/o quantization） | 32.16 | 0.951 | 0.129 | 402/441 | 154 |
| 去掉变化门控（w/o change gate） | 32.09 | 0.951 | 0.129 | 40/60 | 468 |

正文引用的关键数字（第 4.3 节）：每帧 36 KB（N3DV）、18 KB（MeetRoom）；"up to 11×"（摘要、第 5 节）对应表 1 中 400/36≈11.1 倍（对比 ReCon-GS），或表 3 中 402/36（关量化后的自身对照）。

### 消融设计（注明表号）

值得学走的对照设计有三组：

- **逐组件开关（表 3）**：两个开关各管一个载荷分量，互不重叠——量化管运动载荷（关掉后存储 36→402 KB，11 倍，质量三位小数完全不变，新增高斯数几乎不动 153→154）；门控管外观载荷（关掉后每帧新增高斯 153→468，3 倍，存储 36→40 KB 微涨，PSNR 只掉 0.07 dB）。"关掉后质量不动、体积暴涨"是最干净的反事实证明。
- **量化步长扫描的率失真前沿（图 4）**：Δt 从细到粗扫，画 PSNR 对每帧存储的曲线。两个方向的失效都报告了：步长变细，质量几乎不涨、存储快涨；步长过粗，质量下降，而且"过度粗糙的运动拟合不上场景，反而触发额外增密"——存储不再下降甚至上升。最终配置取曲线拐点。这张图同时解释了为什么固定步长可行（拐点附近平坦）。
- **时间维稳定性曲线（图 5）**：有无门控下每帧净增高斯数随时间的曲线。无门控时场景尺寸随时间稳定增长，有门控时近似恒定——直接支撑"长序列可持续流式"的主张，这是流式论文特有、静态压缩论文没有的一类证据。

### 贡献列表（原文照抄 + 逐条标注）

原文有明确贡献列表（第 1 节末 "Our contributions are summarized as follows"）：

1. "QuARC-GS: We present a compact online FVV reconstruction framework that combines a persistent multi-level anchor motion representation with compact per-frame appearance updates." —— [机制] 框架本身：跨帧复用的多级锚点运动表示，加每帧紧凑的外观更新。
2. "Quantization-aware anchor residuals: We optimize per-anchor motion residuals using their forward-pass quantized values, so the deformation rendered during training matches the transmitted motion payload." —— [机制] 训练期直接用前向传播的量化值优化锚点运动残差，训练时渲染的形变与实际传输的运动载荷完全一致。
3. "Change-gated densification: We use temporal change after anchor deformation to gate incremental Gaussian creation, reducing redundant appearance capacity in static but difficult regions." —— [机制] 用锚点形变之后的时域变化来决定是否生成新高斯，避免给"静止但难重建"的区域重复分配外观容量。

三点都成立，但第 2 点的 STE 量化框架取自 EAGLES/QUEEN（3.4 节明说 "we adopt a quantization framework during training (Girish, Gupta, and Shrivastava 2024; Girish et al. 2024)"），第 3 点的视空间梯度思路也取自 QUEEN（3.5 节引 Girish et al. 2024）；本文自己的部分是：把量化对象从逐高斯残差换成锚点残差、观察到零 bin 白送静态/动态分割、以及在 ReCon-GS 底座上把两件事组合成完整流式系统。

### 讲故事方式

主线一句话：每帧载荷的冗余集中在两处——"几乎不动的锚点"和"静态区域的重复增密"；前者用量化打进零 bin（掩码免费），后者用帧间损失差梯度挡在门外，于是 36 KB/帧做到 400 KB 同质量。最有力的一张图是图 1：对数存储轴上的率失真散点，QuARC-GS 的点比所有在线方法低一到两个数量级、质量与最高档持平，一眼看出"省了一个数量级、质量没动"。最能拆开机制的是表 3：w/o quantization 一行（402/441 KB，质量逐位不变）直接给出 11 倍的出处，把"存储到底省在哪"回答得没有歧义。

## 四学：原文关键段翻译（深读层必填）

### 方法节核心段（选 2 段）

> **原文（第 3.4 节第一段及段末）**："Even in a compact hierarchical anchor representation, the deformation field is highly redundant: most anchors barely move between frames. Thus, optimizing D in full precision and rounding it afterward is wasteful: small, imperceptible motions consume bitrate, and post-hoc rounding introduces a train/transmit mismatch that degrades reconstruction. To address this, we adopt a quantization framework during training (Girish, Gupta, and Shrivastava 2024; Girish et al. 2024). Each attribute of the anchor residual is rounded to a fixed step during the forward pass with an STE, which rounds in the forward pass and passes the gradient through unchanged (Bengio, Léonard, and Courville 2013). ... While the rounded, transmitted value D̂ is used to transform the Gaussians via Eq. (2), the gradient still flows through the full-precision D in the backward pass, so that surviving anchors adapt to the quantization."
>
> **译文**：即使在紧凑的层级锚点表示里，形变场仍然高度冗余：大多数锚点在帧与帧之间几乎不动。因此，先用全精度优化 D、事后再取整是浪费的：微小到察觉不到的运动也要消耗码率，而事后取整引入训练值与传输值之间的不一致，损害重建质量。为解决这个问题，我们在训练期采用量化框架（Girish, Gupta, and Shrivastava 2024; Girish et al. 2024）。锚点残差的每个属性在前向传播中被取整到固定步长，用的是直通估计器（STE）——前向取整、梯度原样通过（Bengio, Léonard, and Courville 2013）。……取整后用于传输的值 D̂ 通过式 (2) 作用到高斯上，而反向传播时梯度仍然流过全精度的 D，于是存活下来的锚点能够适应量化。

> **原文（第 3.4 节 "Static and dynamic sparsity" 段）**："The proposed quantization naturally induces a dynamic/static segmentation of the deformation field. Anchor residuals whose magnitude is below Δ/2 are mapped to the identity transform after rounding, whereas larger residuals survive as non-zero motion parameters. Consequently, the anchors are partitioned into a dynamic set, containing at least one non-identity quantized residual, and a static set, whose residuals all quantize to the identity. The quantized deformation field D̂ is entropy-coded independently for translational and rotational components: after quantization, both are represented by integer-valued symbols, and the coding cost is estimated using its empirical Shannon entropy. Since static anchors are mapped to the identity transform, they accumulate in the zero bin, increasing its probability mass and lowering the overall coding cost. As a result, sparsity is captured directly by the entropy coder, without requiring any separate mask or gating term to transmit."
>
> **译文**：所提出的量化自然地在形变场上诱导出动态/静态分割。幅度低于 Δ/2 的锚点残差在取整后被映射为单位变换，更大的残差则作为非零运动参数存活下来。于是锚点被分成两个集合：动态集合，至少含一个非单位的量化残差；静态集合，其残差全部量化为单位变换。量化后的形变场 D̂ 对平移和旋转两个分量分别独立做熵编码：量化之后两者都用整数值符号表示，编码代价用其经验香农熵估计。由于静态锚点被映射为单位变换，它们堆积在零 bin 里，提高了零 bin 的概率质量，降低了整体编码代价。这样一来，稀疏性直接由熵编码器捕获，不需要传输任何单独的掩码或门控项。

### 主结果表述段（选 1 段）

> **原文（第 4.3 节第一段）**："We evaluate QuARC-GS on N3DV (Table 1) and MeetRoom (Table 2) against state-of-the-art (SOTA) offline and online dynamic scene reconstruction methods. As shown in both tables, QuARC-GS matches the rendering quality of the best online SOTA in terms of PSNR, SSIM, and LPIPS (i.e., standard metrics for NVS rendering quality) while transmitting only 36 KB per frame on N3DV and 18 KB per frame on MeetRoom. This represents only a small fraction of the payload required by even the most compact SOTA, yet incurs no measurable loss in fidelity. These storage savings arise because quantization-aware anchor deformation collapses static motion to the identity transformation, while change-gated densification suppresses redundant Gaussians, allowing the transmitted bitstream to encode only genuinely dynamic content."
>
> **译文**：我们在 N3DV（表 1）和 MeetRoom（表 2）上，把 QuARC-GS 与最先进的离线和在线动态场景重建方法进行比较。如两表所示，QuARC-GS 在 PSNR、SSIM 和 LPIPS（新视角合成渲染质量的标准指标）上与最好的在线方法持平，同时在 N3DV 上每帧只传输 36 KB、在 MeetRoom 上每帧只传输 18 KB。这只相当于最紧凑的同类最优方法所需载荷的一小部分，却没有可测量的保真度损失。这些存储节省来自：量化感知的锚点形变把静态运动塌缩为单位变换，变化门控增密抑制了冗余高斯，使传输的码流只编码真正动态的内容。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | 三个组件都有明确出处：层级锚点形变沿用 ReCon-GS（3.2 节自述 "Following (Fu et al. 2025)"），STE 量化训练框架取自 EAGLES/QUEEN（3.4 节自述），视空间梯度门控取自 QUEEN（3.5 节引用）。属于本文的部分是把量化对象换成锚点残差、"零 bin 白送静态/动态分割、免掩码"这个观察，以及完整的流式系统组合——每件都干净，但没有一件是全新机制 |
| 证据强度 | 中 | 两个标准数据集；表 1 共 14 个方法行，其中 2 个关键基线（3DGStream、ReCon-GS）用官方代码同环境复现、3 次运行均值；逐组件消融（表 3）+ 步长扫描率失真前沿（图 4）+ 长序列体积曲线（图 5）；代码开源可复核。缺：表 2 只有 4 行且无 LPIPS；无逐场景分解；图 4、图 5 无数值刻度；新增高斯属性的具体编码方式与逐字段字节分解论文未报告；同方向、同样以"每帧 0.1 MB"为目标的 iFVC 没有进对比表 |
| 对期刊版的威胁度 | 中 | 撞两点：其一，"量化感知锚点残差训练"与我们的 B3 阶梯对齐量化训练同类（训练期让模型适应量化值），审稿人可能拿它要求我们写清差异——差异是真实的（它固定步长、单质量档、动态场景；我们内容自适应、零侧信息、渐进码流、静态场景），但必须在期刊版里主动写明；其二，zero-cost binding 的"解码端重算"与零侧信息契约的原则重合（范围窄，只覆盖绑定图）。不撞：单文件渐进分层码流、内容复杂度自适应量化它都不涉及 |

## 对 DCCA-GS 的可借鉴点

1. **零成本绑定（种子重算锚点集合与绑定图，3.3 节）→ 用到我们码流的头部字段。** 我们码流里锚点分组、分组顺序这类元数据，凡能改成"解码端从已解码几何加种子重算"的，可以再省头部字节。预期收益：头部字段占我们的体积比例不大，但零侧信息契约的覆盖面会变宽。验证实验：重算结果与编码端做 bit-exact 对比（我们已有 bit-exact 验证流程，成本低）。
2. **量化零 bin 吸收静态性、免掩码（3.4 节）→ 对照我们锚点位置字段的"零标志→符号→幅度"二值分解。** 它的思路是让粗量化步长本身把小值送进零 bin，零不用显式编码；我们是显式传零标志。预期收益：锚点 xyz 字段零占比高的分组上，码长可能下降。验证实验：同率失真下对比两种方案在同一字段上的实测码长。
3. **固定步长扫描的率失真前沿图（图 4 的做法）→ 用作我们内容自适应量化的证明图。** 画一条全场景统一固定步长从细到粗扫描的率失真曲线，再叠上我们按内容自适应分配步长的点：如果自适应点落在任何单一固定步长曲线的左上方，内容自适应的主张就有了最直接的证据。这张图我们目前没有，成本只是一组扫描实验。
4. **消融的立论方式（表 3）→ 用到我们的消融表呈现。** 学它"关掉一个组件、质量几乎不动、体积暴涨 N 倍"的干净对照，以及为每个载荷分量（运动/外观）各配一个开关、证明两者互不重叠。我们的逐组件消融已有，可补充类似的"每档新增字节数/增量高斯数"这类结构性指标，让每个模块管什么一目了然。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 量化怎么定 | 固定世界空间步长：平移 Δt=0.0017、旋转 Δr=0.01，全场景统一，手工取率失真曲线拐点（4.2 节、图 4） | 解码端可重算的内容复杂度量化：按分组预测概率参数的 8→32→3 小 MLP 定步长，训练期渲染敏感度监督，零侧信息（规范第一节） | 它证明固定步长在动态流式里已够用（拐点附近平坦，图 4）；我们的差异点是内容自适应。启示：期刊版需要自己的固定步长扫描实验，证明自适应优于任何单一固定步长 |
| 熵编码上下文 | 平移、旋转两部分各自独立编码，整数值符号，代价按经验香农熵估计，无上下文模型、无条件表（3.4 节） | 二值分解区间编码（零标志→符号→幅度逐位是非题），（贡献组×粗值幅度桶）复合条件表做分组频率统计，细层用已解码粗层作上下文的层条件熵编码 | 它的 36 KB/帧主要靠量化零 bin 省出来，不是靠上下文建模——说明小载荷场景下"量化分好、熵编码从简"也能到一个数量级。启示：它反衬我们上下文建模的必要性要在消融里立住；反过来它免掩码的零 bin 思路可与我们的零标志对比码长 |
| 渐进/可截断 | 不涉及。每帧一个固定质量档的压缩载荷，无单文件多画质档、无字节前缀可截断；"streaming"指时间维逐帧在线（图 2(d)、4.3 节） | 单文件渐进分层码流：任意画质档都是文件连续字节前缀，层级序文件布局，真实可截断并经 bit-exact 验证 | 完全错开：它是时间维"每帧一个包"，我们是质量维"一个文件多档"。启示：动态场景加渐进码流的组合目前是公开空白，期刊版可作为延伸方向声明 |
| 侧信息 | 三处免传：绑定图由种子重算（3.3 节 zero-cost binding）、静态/动态掩码由零 bin 吸收（3.4 节）、量化步长是固定超参数所以不用传 | 零侧信息契约覆盖全部解码端可重算量：量化步长与编码上下文都从已解码数据重算，重算不了的不进码流不罢休 | 它的重算原则用得比我们窄（只覆盖绑定图），步长靠"固定"绕过而非"重算"；我们的契约要求连步长本身都内容自适应且零侧信息。启示：该原则在动态流式同样成立且有效，可引用它作旁证 |
| 场景与表示 | 动态场景：canonical 帧加每帧三层锚点形变残差（7 维：平移 3 维加旋转残差 4 维），外观增量靠变化门控新增高斯（3.2-3.5 节） | 静态场景单模型：HAC++ 底座锚点，压缩全部属性字段（xyz/特征/缩放/偏移/掩码等，规范第一节） | 场景不同导致载荷结构不同：它每帧只传 7 维锚点残差加少量新高斯，我们一次压全模型。启示：它的"canonical 加残差"结构意味着若做动态扩展，我们的渐进分层码流可以直接套在每帧残差上，与它的量化零 bin 方案互补 |
