# 《Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2312.00109 |
| 发表 | CVPR 2024 |
| 阅读材料 | 全文（.lit-cache/txt/2312.00109.txt） |
| 与 DCCA-GS 的关系 | 表示源头：锚点+神经高斯表示的出处，是 HAC/HAC++/我们框架的表示层地基 |

> 一句话定位：把 3DGS"每个高斯独立存全套属性"改成"稀疏锚点挂 k 个神经高斯、属性由视角条件的小 MLP 现场解码"，同画质下体积降 4-10 倍；它本身不做量化、熵编码和压缩型剪枝，但把待存储对象从百万级高斯缩成少量锚点属性，且证明这些属性空间上可预测——这是后来 HAC/HAC++ 能做上下文熵编码的前提。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

原话（第 1 节）：3D-GS "tends to excessively expand Gaussian balls to accommodate every training view, thereby neglecting scene structure. This results in significant redundancy and limits its scalability"，并且 "view-dependent effects are baked into individual Gaussian parameters with little interpolation capabilities, making it less robust to substantial view changes and lighting effects"。两个矛盾：一是冗余——为了让每个训练视角都拟合好，高斯数量无限膨胀；二是脆弱——视角相关效果被写死在每个高斯的固定参数里，换个视角或光照就出问题。

### 支撑观察（直觉 / 数据 / 失败实验）

- Figure 1：同一场景 3D-GS 与本文并排，例如 17.16dB/242MB/127FPS vs 20.41dB/66MB/110FPS（本文画质反而更高）——冗余的直观证据。
- 第 4.2 节 View Adaptability + Figure 7：同一个神经高斯从不同视角解码出的属性强度分布不同、且局部连续——"属性应当由视角推导而不是存死"的直接证据。
- Figure 6：锚点特征做 K-means 聚成 3 类后，栏杆、婴儿车、桌面、显示器、墙面、地面各自聚在一起——锚点特征编码了局部场景内容，空间上相关。这条观察是后来 HAC"锚点属性可被哈希网格预测"的伏笔。
- Figure 9：不同初始 k 值最终收敛到相近的激活高斯数——结构自己会挑出非冗余的高斯数量。
- 第 4.4 节局限：初始 SfM 点质量决定上限，大范围无纹理区域初始化差——生长操作只能部分补救。

### 显然的路为什么没走

1. 继续调 3D-GS 的剪枝/致密化启发式：只能事后补救。冗余的根源是"属性逐高斯存储 + 过拟合单个训练视角"，不改结构省不下来。
2. 直接砍每高斯的属性维度（降球谐阶数、共享码本）：这是后续工作（LightGaussian 等）走的路；本文更彻底——属性干脆不存，全部由 MLP 现场推导。
3. 回到 NeRF 式全局 MLP 或稠密特征网格：全局 MLP 每条光线要采样大量点，渲染慢；稠密网格方法（Plenoxel、Instant-NGP）"struggle to represent empty space effectively"（第 2 节）。本文要保住 3D-GS 的光栅化速度（千像素分辨率约 100 FPS），所以只对视锥内锚点激活、用 2 层小 MLP 一次性解码，并按不透明度筛掉无关紧要的神经高斯。

## 二学：机制（洞察怎么变成方法）

### 核心流程

本文没有独立的编码期/码流，量化、熵编码都不存在；只有训练期和渲染期。

- 初始化（训练期开始）：COLMAP SfM 点云 P（第 3.2.1 节）按体素尺寸 ϵ 体素化（Eq 4），去重后每个体素中心作锚点 v，挂 32 维特征 fv（R32）、3 维缩放因子 lv、k 个 3 维可学习偏移量 Ov（k=10，第 4.1 节）。
- 训练期每步：对视锥内可见锚点，用相机-锚点相对距离 δvc 和方向向量 dvc 预测三层特征权重（Eq 6），加权合成视角自适应特征（Eq 7）；再由 4 个 2 层 MLP（Fα/Fc/Fs/Fq，隐藏维 32）一次前向解出 k 个神经高斯的不透明度、颜色、尺度、四元数（Eq 8，补充材料 Eq 15/16）；高斯位置 = 锚点位置 + 偏移×锚点缩放（Eq 9）。只保留不透明度大于阈值 τα 的高斯参与光栅化。损失为 L1 + SSIM + 体积正则（Eq 11、12），训练 30k 迭代。
- 训练期锚点精炼（第 3.3 节）：生长——把神经高斯按体素聚合，梯度在 N=100 次迭代上的平均 ∇g 超过 τg 的体素长出新锚点（多分辨率，Eq 10，加随机淘汰）；剪枝——锚点下神经高斯累计不透明度低于 0.5 则删锚点。
- 渲染期：与训练期同一套解码 MLP，无任何后处理。存储只需锚点（位置+fv+lv+Ov）和解码 MLP。

它为压缩创造了什么条件（本文自己的存储是原始浮点数，不压缩）：

1. 属性从"存"变"算"：按第 3.2.1-3.2.2 节的维度算，每锚点只存 3（xyz）+32（特征）+3（缩放）+10×3（偏移）=68 个数，k=10 个神经高斯的全部渲染属性（不透明度/颜色/尺度/旋转）现场解码、不进存储。3DGS 每高斯则要存位置、球谐颜色系数、不透明度、尺度、旋转（第 3.1 节）。待压缩对象缩小约两个数量级。
2. 属性空间可预测：Figure 6 证明锚点特征局部相关，Figure 7 证明解码出的属性随视角连续变化——"知道位置就能猜属性分布"在这套表示上成立。HAC 的哈希网格上下文熵模型正是建立在这条性质上；我们框架"解码端可重算的复杂度特征"（局部密度、offset 能量）同样依赖这套表示。

### 关键公式（按原文抄，注明编号）

txt 抽取时第 3.2 节的公式编号 (5)-(9) 顺序错乱，以下按内容对应编号抄录，原文公式以 PDF 为准。

- Eq (4)：V = P/ϵ —— SfM 点云按体素尺寸 ϵ 量化，V 是体素中心，即锚点位置。
- Eq (5)：δvc = ∥xv − xc∥2，dvc = (xv − xc)/∥xv − xc∥2 —— 锚点到相机的相对距离与方向。
- Eq (6)：{w, w1, w2} = Softmax(Fw(δvc, dvc)) —— 小 MLP 按视角预测特征库权重。
- Eq (7)：f̂v = w·fv + w1·fv↓1 + w2·fv↓2 —— 全分辨率与两级降采样特征加权合成，得到视角自适应锚点特征。
- Eq (8)：{α0, ..., αk−1} = Fα(f̂v, δvc, dvc) —— 不透明度解码。补充材料说明输出用 Tanh 激活，0 是天然筛选阈值；颜色 {ci} = Sigmoid(Fc)（Eq 15）、尺度 {si} = Sigmoid(Fs)·sv（Eq 16，sv 为锚点基础缩放）、四元数做归一化。
- Eq (9)：{µ0, ..., µk−1} = xv + {O0, ..., Ok−1}·lv —— 神经高斯位置 = 锚点 + 可学习偏移×锚点缩放。
- Eq (10)：ϵg(m) = ϵg/4m−1，τg(m) = τg × 2m−1 —— 多分辨率生长的体素尺寸与梯度阈值。
- Eq (11)：L = L1 + λSSIM·LSSIM + λvol·Lvol；Eq (12)：Lvol = Σi Prod(si) —— 体积正则鼓励神经高斯小且少重叠。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 生长信号：体素内神经高斯梯度在 N=100 次迭代的平均 ∇g，超过 τg（默认 64ϵ，无纹理为主场景 16ϵ）即长新锚点；剪枝信号：锚点下 k 个神经高斯的累计不透明度低于 0.5 即删锚点（第 3.3、4.1 节） |
| 哪一端算得出（编码端/解码端/仅训练期） | 仅训练期：∇g 依赖训练视图的反传梯度；渲染端只有"不透明度大于 τα 才光栅化"的激活筛选（第 3.2.2 节） |
| 有无侧信息 | 无。训练期启发式，不涉及码流 |
| 训练期还是编码期 | 训练期（30k 迭代内每轮精炼时执行） |
| 和渲染质量的距离（直接测质量还是代理量） | 代理量：梯度近似训练视图重建误差，不直接度量测试视图质量；剪枝用的累计不透明度也是几何占位代理（Figure 8 显示它有隐式几何估计作用） |

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 数据：27 个场景——Mip-NeRF360（7 个）、Tanks&Temples（2）、DeepBlending（2）、Blender 合成、BungeeNeRF（6）、VR-NeRF（2）（第 4.1 节）。
- 指标：PSNR/SSIM/LPIPS + 存储体积（MB）+ 渲染速度（FPS）。体积就是模型存储大小，未做熵编码。
- 对比：主对比 3D-GS（两者都训 30k 迭代）；Mip-NeRF360/iNGP/Plenoxels 的数字转引自 3D-GS 论文（Table 1 说明）。逐场景结果在补充材料 Table 6-17。
- 训练配置要点：k=10，所有 MLP 2 层、隐藏维 32，λSSIM=0.2，λvol=0.001（第 4.1 节）。

### 主结果（注明表号）

- Table 1：Tanks&Temples PSNR 23.96 vs 3D-GS 23.14；DeepBlending 30.21 vs 29.41；Mip-NeRF360 28.84 vs 28.69（打平或更好）。
- Table 2：DeepBlending 66MB vs 676MB（10.2 倍降）且 FPS 139 vs 109；Mip-NeRF360 156MB（4.4 倍降）、FPS 102 vs 97；Tanks&Temples 87MB（4.7 倍降）。
- Table 3：BungeeNeRF PSNR 27.01 vs 24.89、203MB vs 1606MB（7.9 倍降）；Blender 合成 14MB vs 53MB（3.8 倍降）。
- Table 5：DB-Playroom 上无精炼/只剪枝/只生长/全开四档 PSNR 为 28.45/29.12/30.54/30.62——去掉生长（只留剪枝）从 30.62 掉到 29.12，两个操作全去掉再掉到 28.45——生长对弱初始化场景至关重要。

### 消融设计（注明表号）

- Table 4：逐个叠加两种推理期筛选（FILTER1 视锥筛选 / FILTER2 不透明度筛选），画质几乎不动（DB-Playroom 30.3-30.62），FPS 从 84 升到 150——干净的逐组件开关设计。
- Table 5：无精炼/只剪枝/只生长/全开四档对照，分离两个操作各自的贡献（生长管画质、剪枝管体积与残余锚点质量）。
- Figure 9：不同初始 k 收敛到相近激活数——一种"结构自我调节"的替代验证设计，可学。

### 贡献列表（原文照抄 + 逐条标注）

1. "Leveraging scene structure, we initiate anchor points from a sparse voxel grid to guide the distribution of local 3D Gaussians, forming a hierarchical and region-aware scene representation" —— [机制] 锚点化表示，后续所有锚点系压缩工作的地基。
2. "Within the view frustum, we predict neural Gaussians from each anchor on-the-fly to accommodate diverse viewing directions and distances, resulting in more robust novel view synthesis" —— [机制] 神经高斯+视角条件解码，属性不存带来的体积收益来源。
3. "We develop a more reliable anchor growing and pruning strategy utilizing the predicted neural Gaussians for better scene coverage" —— [工程] 基于解码属性的梯度生长与不透明度剪枝。

### 讲故事方式

主线一句话：3DGS 让高斯过拟合每个训练视角，本文让高斯"住"在锚点上、属性按视角现场推导——又稳又省。最有力的是 Figure 1 与 Table 2 的组合：Figure 1 顶行给出单场景 17.16dB/242MB vs 20.41dB/66MB 的直观对比，Table 2 给出多数据集"体积降 4-10 倍、FPS 不降反升"的系统数字，画质/体积/速度三个维度一次立住。支撑"视角自适应"这个独有主张的关键证据是 Figure 7 的属性分布可视化。

## 对 DCCA-GS 的可借鉴点

1. 表示层的两条性质要守住：锚点特征局部相关（Figure 6）和解码属性视角连续（Figure 7）。我们机制 A 用的"局部密度、offset 能量"等解码端可重算特征之所以可行，前提就是这套表示的属性可预测性；任何改动锚点分布的操作（如 Mini-Splatting depth-reinit 增密）之后都应复查这两条性质是否仍成立（可视化聚类+属性连续性检查即可）。
2. 生长/剪枝信号与我们机制 B 同源但粒度不同：它把每高斯梯度聚合到体素再对比阈值，我们用每属性渲染梯度 EMA 直接监督 MLP。写论文时可引用它做对照，并补一个"体素聚合梯度 vs 每属性梯度 EMA"的替换实验，突出我们的信号更细、可微、无体素离散化。
3. 不透明度筛选的隐式几何作用（Figure 8：随机点云初始化下，激活神经高斯能还原推土机粗结构）说明掩码激活比例作为复杂度特征有表示层依据，可并入机制 A 的公式特征组并做增量消融。
