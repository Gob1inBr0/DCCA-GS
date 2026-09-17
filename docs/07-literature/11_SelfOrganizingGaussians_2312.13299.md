# 《Compact 3D Scene Representation via Self-Organizing Gaussian Grids》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2312.13299 |
| 发表 | CVPR 2024 |
| 阅读材料 | 全文（.lit-cache/txt/2312.13299.txt） |
| 与 DCCA-GS 的关系 | 同类竞品：无锚点、无码本的"2D 网格 + 现成图像编解码器"路线，和我们的哈希锚点 + 量化的路线正交 |

> 一句话定位：把无序的高斯属性表重新排列成 2D 图像网格，让相邻位置属性连续，再用 JPEG XL 这类现成图像压缩器存盘，体积降到 3DGS 的 1/17 到 1/42（Table 1），渲染质量基本不掉。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾
3DGS 渲染快、质量高，但"stores millions of parameters in a large unorganized list"（第 1 节），一个场景几百 MB，小设备放不下。它的判断是：问题不在参数数量本身，而在存储结构——无序列表没有可利用的冗余。高斯的排列顺序本来就不影响渲染结果（"numerous permutations of Gaussian parameters can equivalently represent a scene"，摘要），这份自由度被 3DGS 浪费了，应该用它换可压缩性。

### 支撑观察（直觉 / 数据 / 失败实验）
1. 感知冗余的前提（等价排列）是作者论断，没有单独实验，但符合 3DGS 常识：高斯集合无序。
2. 图像编解码器对平滑内容高效：第 3.2 节开头说明，2D 压缩方法在"内容噪声小、能从邻居预测"时最省码。
3. 关键实验证据在第 4.2 节 Table 3（Truck 场景）：vanilla 3DGS 训练完直接套用它的压缩管线（不排序、不平滑），w/ SH 只剩 21.98 PSNR / 71.7 MB；而排序 + 平滑一起训练后是 25.37 PSNR / 34.3 MB。同样的压缩器，排布不同，体积差 2 倍、质量差 3.4 dB——这是"排布决定可压性"最硬的证据。附录 C 的 Fig. 3 给了同样的视觉对比。
4. Table 4（Truck，无 SH）：排序后即使用完全无损的 JPEG XL，66.05 MB 也比 PLY 的 104.89 MB 小，说明收益主要来自排序让数据本身可压，而不是量化损失。

### 显然的路为什么没走
1. 直接压缩 .ply 文件（zip、DRACO 等点云压缩）：Table 3 显示 .ply.zip 只从 624.7 MB 降到 548.2 MB，16 倍的差距；DRACO 实现不支持额外属性（第 4.2 节），且后处理压缩吃不到训练期塑造出的平滑性。
2. 向量量化/码本（当时 NeRF 压缩的主流）：它没走码本路线，而是把属性当图像压。理由是排布好之后，现成图像编解码器就够强，还能保留 3DGS 的数据结构、解码后直接进既有渲染器。
3. 只在压缩时平滑（后处理）：它明确选择在训练期就把场景"推向"可压缩的等价解（第 3.2 节），而不是压完再修——Table 3 的对照组证明后处理路线质量掉太多。

## 二学：机制（洞察怎么变成方法）

### 核心流程
- 训练期：训练开始时排一次序；每次 densification（3DGS 的增密步骤）之后重新排序，修复增密和剪枝破坏的邻域关系。整个训练期间在 3DGS 损失上加一个 2D 平滑正则，梯度穿过渲染器，让优化器在"渲染质量相当"的解里偏好 2D 平滑的那一个。所有属性通道共用同一个排列（每个网格位置对应同一个高斯，Fig. 3）。
- 网格构造：高斯数增长时取能完全填满的最大正方形边长，放不下的按 opacity 最低裁掉（第 3.1 节）。Fig. 8 显示删掉 opacity 最低 30% 的高斯 PSNR 几乎不变。
- 编码期：训练结束后一次性量化 + 图像压缩（第 3.4 节）。没有码本：各属性先按统计范围裁剪（RGB 裁到 [-2,4]，opacity [-6,12]，非 DC 的 SH [-1,1]，旋转 [-1,2]，取自 360 数据集 1%–99% 分位数），再均匀线性量化，量化级数 q_coords = 2^14、q_scale = q_opacity = q_rotation = 2^6、q_SH_rest = 2^5。RGB 网格用有损 JPEG XL（8-bit，质量 100），其余属性用无损 JPEG XL。
- 解码期：解图像、反量化、按 3DGS 的激活方式恢复（scale 过指数、opacity 过 sigmoid、坐标过它自己的对数收缩），得到和原始 3DGS 完全同构的数据，直接进既有渲染器。解码端不需要任何网络或码本查找。

### 关键公式（按原文抄，注明编号）
全文唯一的编号公式是坐标收缩（第 3.3 节，式 (1)）：

x = sign(x_log) × (exp(|x_log|) − 1)

x_log 是网络存的原始坐标，经此激活后场景中心获得更高的相对精度，且可逆，用于替代 Mip-NeRF 360 式的空间收缩。

注意两点（以 PDF 为准的地方）：
- 用户常关心的"排序损失"在原文没有编号公式。第 3.1 节用文字描述排序目标：先对当前网格做 2D 高斯低通滤波得到平滑目标网格，再把所有元素重新分配到与平滑目标最匹配的位置（块内 4 个元素一组，枚举全部 24 种排列取总距离最小），用排序网格与目标网格的平均 L2 距离衡量收敛。迭代时滤波半径按 φ = 0.95φ 递减、块大小 β = φ + 1（不小于 16），直到 φ < 1。这就是 PLAS（Parallel Linear Assignment Sorting），在 GPU 上几秒内排完数百万高斯（附录 C：512×512×3 网格 5.7 秒，VAD 4.02，对比 FLAS 的 131 秒 / 3.53）。
- 平滑正则也没有编号公式：第 3.2 节文字描述为"对每个属性通道做 2D 高斯模糊，当前值与模糊值的偏差用 Huber 损失（比 MSE 抗离群点、实测更利于压缩），乘权重 λ 后加进 3DGS 损失"。附录 Table 2 给出实现配置：kernel 5、sigma 3、λ = 1.0；各通道损失权重为 opacity 0.09、rotation 0.91、其余为 0；排序距离里 position、color（SH DC）、scaling 权重 1.0，其余为 0。附录 B.4 解释这个分工：position/color/scale 对 PSNR 影响大，作为排序键；opacity 和 rotation 影响小，靠邻居平滑跟着变可压即可。

### 信号设计（重点）
| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 不是逐高斯的信号。它优化的是全局的 2D 网格局部平滑度（排序 + 平滑正则），对所有高斯一视同仁，不做重要性分级 |
| 哪一端算得出（编码端/解码端/仅训练期） | 平滑性是训练期和编码端的属性；解码端不需要算任何信号，解图反量化即可 |
| 有无侧信息 | 无。排列本身不进码流——属性值按排列后的网格存，解码端不需要知道原始顺序 |
| 训练期还是编码期 | 排序和平滑正则在训练期（每次增密后重排）；量化是一次性的编码期后处理（结论中把"训练中量化"列为未来工作） |
| 和渲染质量的距离（直接测质量还是代理量） | 代理量：平滑正则的梯度穿过渲染器，与渲染损失联合优化，间接服务质量 |

## 三学：证明与包装（创新怎么立住）

### 实验口径
- 对比对象：vanilla 3DGS（.ply，全 SH）及其去 SH 版、Mip-NeRF 360、VQ-TensoRF、INGP 等，另有博客方案 Making Gaussian Splats smaller（Table 3）。
- 体积口径：量化 + JPEG XL 后各属性文件之和，单位 MB。基线是 3DGS 官方 .ply。没有熵编码网络、没有解码器。
- 数据集：Mip-NeRF360、Tanks&Temples、Deep Blending、Synthetic-NeRF；参数全数据集共用（附录 B.4）。
- 需要注意的口径：它改了 3DGS 的致密化超参（附录 Table 2：densification interval 100→1000、grad threshold 2e-4→7e-5、min opacity 0.005→0.1、opacity reset 3000→∞、percent dense 0.01→0.1），高斯数更少（Truck 1.55M 对 3DGS 的 2.58M，第 4 节），也不做周期性剪枝（附录 Table 3 说明）。所以体积下降有一部分来自高斯更少，不能全记在压缩管线上；但质量持平说明省掉的是冗余。

### 主结果（注明表号）
- Table 1：Mip-NeRF360 上 3DGS 785 MB / 27.55 PSNR → Ours 40.3 MB / 27.64 PSNR（质量还略升）；Tanks&Temples 454→21.4 MB；Deep Blending 699→16.8 MB；Synthetic-NeRF 71.6→4.1 MB。压缩 17x–42x，Deep Blending 上最高 41.6x（第 4 节）。
- 去 SH 训练时（Table 1）：Mip360 212→16.7 MB、T&T 122→8.2、DB 191→5.5、Synthetic 19.4→2.0，相对 vanilla 3DGS 最高 127x 且 PSNR 更高（DB：30.50 对 30.07）。
- Table 3（Truck）：排序+平滑 25.37 PSNR / 34.3 MB，对比 vanilla 3DGS 直接压缩的 21.98 / 71.7 MB、.ply.zip 的 548.2 MB、博客方案 25.05 / 41.5 MB。
- 附录 Table 1（SSIM）：Ours 与 3DGS 基本持平（如 Mip360 均 0.814）。
- 附录 C：最大场景 Garden 4.37M 高斯，网格边长 2091，排序耗时远小于 1 分钟。

### 消融设计（注明表号）
- Table 2：平滑正则强度 λ 扫描（0.01–1.5），给出"λ 小→质量高体积大"的单调权衡（如 λ=0.05 时 24.48 PSNR / 18.35 MB，λ=1.5 时 24.02 / 10.73 MB）。
- Table 3 的"3DGS Our Compression"行是同预算对照：同一压缩管线，去掉排序+平滑后质量体积双崩，直接证明两个组件缺一不可。
- Table 4：压缩格式替换实验（PLY/NPZ/无损 JXL/PNG16/EXR/有损 JXL），同一排序后的数据换编解码器，11.86 MB 时仍有 25.14 PSNR。
- Fig. 8：渐进消融——按 opacity / scaling 从低到高删高斯，opacity 轴删 30% 不掉点，scaling 轴掉得快，支撑"按 opacity 溢出裁剪"的选择。
- 值得学的写法：把"表示改造（训练期）"和"压缩器（编码期）"拆成可独立开关的两级来消融，对照组都用同一压缩器。

### 贡献列表（原文照抄 + 逐条标注）
1. "We propose a new compact scene representation and training concept for 3DGS, structuring the high-dimensional features in a smooth 2D grid, which can be efficiently encoded using state-of-the-art compression methods." —— [机制] 训练期塑造可压表示，这是全文核心。
2. "We introduce an efficient 2D sorting algorithm called Parallel Linear Assignment Sorting (PLAS) that sorts millions of 3DGS parameters on the GPU in seconds." —— [机制] 让排序能嵌进训练循环的工程前提。
3. "We provide a simple to use interface for compressing and decompressing the resulting 3D scenes. The decompressed reconstructions share the structure of 3DGS, allowing integration into established renderers." —— [工程] 兼容性卖点。
4. "We efficiently reduce the storage size by a factor of 17x to 42x while maintaining high visual quality" —— [结果] 第 4 条偏结果陈述，算半条凑数。

### 讲故事方式
主线一句话：高斯顺序是自由度，把它花在"排成平滑图像"上，剩下的交给现成图像压缩器。最有力的是正文 Table 3：一张表同时给出"不排序直接压会崩"（21.98 PSNR）、"排序+平滑后同管线很好"（25.37）、"zip/博客方案都不行"三个对照，主创新和竞品差距一目了然。Fig. 1/Fig. 7 的彩色网格图则把"属性排完像图"这个抽象概念变成一眼能懂的画面。

## 对 DCCA-GS 的可借鉴点

1. "训练期塑造可压表示"的立论方式：它用 Table 3 的同管线对照证明"后处理压不掉的冗余，训练期可以预塑"。我们做消融时可以照搬这个两级对照——同一量化/熵编码管线，开/关 AQM 监督各跑一次，直接展示 I6 塑造出的参数分布更可压。
2. 按属性敏感度分工的超参写法（附录 B.4）：对质量影响大的字段做精排序/精编码，影响小的字段靠平滑搭车。这和我们 AQM"按锚点分级"正交，是"按字段分级"的思路；我们的量化字段分组（feat/scaling/offsets）可以参考它的权重表形式报告各字段配置。
3. 体积口径警示：它的压缩倍数混入了"高斯数更少"的贡献（致密化超参大改）。我们报告对比数字时应说明各方法的高斯/锚点数量，避免压缩率被剪枝稀释或夸大。
4. 它不区分高斯重要性、全员平滑，因此对"少量高敏感高斯"没有保护机制；这正好是我们敏感度信号（I6）的差异化空间，对比实验可以直接引用它的 Table 3 管线作为无信号基线。
