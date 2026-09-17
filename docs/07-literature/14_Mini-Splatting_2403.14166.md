# 《Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2403.14166 |
| 发表 | ECCV 2024（本 txt 为 arXiv v3，2024-10-16；txt 正文未标会议出处，会议信息取自任务说明） |
| 阅读材料 | 全文（.lit-cache/txt/2403.14166.txt，含附录 A–H 与 Algorithm 1） |
| 与 DCCA-GS 的关系 | 基础措施来源：我们的 Mini-Splatting depth-reinit 出自此文；它的"同数量比质量"曲线（Fig 7）是我们 SPA+MiniSplat 实验该学的口径；完整 densification/simplification 我们尚未移植 |

> 一句话定位：它论证 3DGS 的首要问题不是高斯数量而是高斯"铺的位置"——中心在图像上扎堆（overlapping）、该有的地方没有（under-reconstruction）；先用 blur split 和 depth reinitialization 把高斯铺回表面，再用 intersection preserving 和重要性采样收数量，0.49M 高斯做到 3.35M 高斯 3DGS 的画质（Table 1）。

> 任务说明核对：任务提示中"densification 三招（blur-split、distance sampling 等）"与 txt 不符：本文 densification 是 blur split + depth reinitialization 两招，simplification 是 intersection preserving + importance-weighted sampling 两招，txt 中没有 "distance sampling" 这个词。以下全部以 txt 实际内容为准。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

第 1 节：压缩系方法（Niedermayr 等、Lee 等、LightGaussian）"primarily aim at storage compression, often neglect the inefficient spatial distribution of input Gaussians. As a result, direct pruning, as practiced in these approaches, tends to yield suboptimal simplification outcomes"——直接剪枝默认"高斯已经铺好了，删掉不重要的就行"，但高斯根本没铺好，删完还是错的。第 3.2 节把"没铺好"拆成两个现象：'overlapping'（大部分高斯挤在局部区域，邻近区域反而稀疏）和 'under-reconstruction'（某些区域建不出细节，渲染成高斯模糊状的伪影）。

### 支撑观察（直觉 / 数据 / 失败实验）

- Fig 2(a)（第 3.2 节）：把 vanilla 3DGS 的高斯中心投影到渲染图上画成蓝点。参照物是深度传感器或多视立体匹配产生的点云——那些点均匀贴在物体表面，而 3DGS 的中心明显扎堆。第一行展示 overlapping，第二行展示 under-reconstruction。
- Fig 2(b)–(e)：把同一模型往下删，四种方法全失败——剪枝到 0.6M 掉到 24.9 dB（原 25.2），随机采样 0.6M 掉到 24.8，网格采样 1.1M 掉到 23.5，保密度采样 0.6M 掉到 24.7（图注数字）。删法不同、结局相同，overlapping 和 under-reconstruction 纹丝不动。这是"数量不是瓶颈、位置才是"最直接的实验证据（这句口号是我们的概括，原文的说法是 inefficient spatial distribution）。
- Table 1 加 Fig 1：Mini-Splatting 用 0.49M 高斯拿到 27.34 dB（3DGS* 为 3.35M、27.47 dB），而 Mini-Splatting-D 用更多高斯（4.69M）拿到更高画质 27.51 dB——画质不随数量单调变化，随分布变化。
- Appendix B（Fig 12、Table 5）：把初始化换成 MVS 稠密点云（每场景采到 200 万点），3DGS 依然在自行车车架和辐条周围扎堆（Fig 12）；Table 5 中 3DGS dense 27.78 dB 对 Mini-Splatting-D sparse 27.54 dB——好初始化只部分缓解，分布问题主要在自适应密度控制过程中自己长出来。

### 显然的路为什么没走

- 直接剪枝和点云采样：Fig 2(b)–(e) 已经证伪，写进正文当反面对照。
- 沿用 3DGS 的梯度式 split/clone：第 4.1 节指出它在颜色平滑过渡区失效（Fig 3）， oversized 高斯被保留，原版只能按屏幕空间尺度阈值剪掉，但"this pruning method does not address the issue of 'under-reconstruction'"。
- 最直觉的 depth reinit——用 alpha blending 深度反投影重初始化：Table 4 第一行，画质崩到 17.67 dB / SSIM 0.513。附录 C（Fig 13）归因于三种伪影：depth collapse（黑背景色被混进深度，背景深度塌陷）、object misalignment（大高斯和 floater 污染混合深度）、blending boundary（边界处权重都低，深度糊）。所以它改用"单高斯椭球与光线求交的中点深度、每像素只取贡献最大者"。
- 传统点云简化算法（最远点采样、特征点保持等）：第 2.2 节，"non-trivial to directly apply these sampling techniques to 3DGS due to differences in representations"——3DGS 的参数可优化、重要性随训练漂移，静态采样规则跟不上。

## 二学：机制（洞察怎么变成方法）

### 核心流程

三个变体（第 5 节）。主变体 Mini-Splatting（资源效率导向）：共 30K 步优化。15K 步之前是增密阶段——每次增密迭代先做 blur split、再做 3DGS 原有 split/clone；depth reinitialization 每 5K 步执行一次（附录 F、Algorithm 1）。15K 和 20K 两个点做简化——15K 时 intersection preserving 加 importance-weighted sampling（采样后直接用高斯中心重开一版模型），20K 时 intersection preserving 加少量直接剪枝。SH 系数在增密阶段不开、简化后才升阶（作者观察视角相关颜色对增密几乎没帮助）。Mini-Splatting-D 去掉简化，画质优先；Mini-Splatting-C 在 Mini-Splatting 模型上做后处理压缩：中心存 float32，其余属性用 RAHT 变换编码（深度 16、量化步长 0.02）再加 zip 无损压缩（附录 F）。剪枝、量化、熵编码各自动在哪个环节：剪枝在训练期调度里（15K/20K），量化和熵编码只在 C 变体的训后处理中出现。

### 关键公式（按原文抄，注明编号）

渲染（式 1）：c(x) = Σ_{i=1..N} w_i · c_i，其中 w_i = T_i · α_i · G_i^{2D}(x)，T_i = Π_{j=1..i-1} (1 − α_j · G_j^{2D}(x))。txt 中该式以 LaTeX 源码保留、内容完整；c_i 为 SH 建模的视角相关颜色，G_i^{2D} 是经局部仿射变换投影的 2D 高斯。

blur split 判据（式 2）：G^blur = {G_i | S_i > T_blur ∧ i ∈ [1, N]}，T_blur = θ_blur · H · W。其中 S_i = Σ_{x=(1,1)}^{(H,W)} δ(i(x) = i_max(x))，i_max(x) = arg max_i w_i 是该像素 alpha-blending 权重最大者的索引，S_i 即"以最大贡献者身份覆盖的像素面积"，在前向光栅化里顺手可算；θ_blur = 2×10⁻⁴（第 4.1 节）。命中的高斯按 3DGS 原有 split 方式拆分。

中点深度（附录 D，式 4–7）：把尺度 s = (sx, sy, sz) 的高斯建成椭球 g(x, y, z) = x²/sx² + y²/sy² + z²/sz² = 1，光线 r(t) = o + td 代入得二次方程（式 4–5），解 t = (−b ± √(b²−4ac)) / 2a（式 6），取两交点中点 p^mid = r(t^mid) = r(−b/2a)（式 7），判别式 Δ = b²−4ac 顺带判定光线与椭球是否相交。多高斯情形每像素只取贡献最大者：d_mid = d_{i_max}^mid，i_max = arg max_i w_i（第 4.1 节）——不做 blending，就是为了绕开上述三种伪影。作者还证明该中点与"沿光线密度最大处"的 t^opt = −B/(2C) 数值相同（式 8–11），选前者是因为多一个判别式可用。

intersection preserving（式 3）：G^int = {G_i | i ∈ I_max ∧ i ∈ [1, N]}，I_max 是该图渲染索引图 i_max(x) 中出现过的全部索引——只保留"亲自提供过交点"的高斯。

重要性加权采样（第 4.2 节）：采样概率 P_i = I_i / Σ_{i=1..N} I_i（txt 抽取此式为乱码"Pi = PNIi I i=1 i"，按上下文为重要性归一化，原文公式以 PDF 为准）。动机：直接剪枝在删除比例大时崩，因为相邻高斯重要性相近、会被一起删掉破坏局部几何；随机采样按概率删能保住整体几何，证据是剪枝后的中心 chamfer distance 随保留比例下降明显、采样则好得多（第 4.2 节、Fig 6）。重要性取值（附录 E）：室内用累计 blending 权重 I¹_i = Σ_{j=1..K} w_ij（K 为与该高斯相交的光线总数）；室外用式 12：I²_i = Σ_{m=1..M} I_i^{(m)} · δ(i ∈ I_max^{(m)})，其中 I_i^{(m)} = Σ_{j=1..K} w_ij^{(m)} / S_i^{(m)}——只对提供过交点的图像累计，并除以投影面积，压住天空和远景大高斯。作者自认这套设计是 case-dependent 的手工技巧，放在附录当实验经验。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 三类前向光栅化统计量：(a) S_i 最大贡献覆盖面积（blur split 的"过大"判据）；(b) argmax 索引图 + 椭球中点深度（depth reinit 与 intersection preserving 用）；(c) 累计 blending 权重重要性 I_i（采样用） |
| 哪一端算得出（编码端/解码端/仅训练期） | 仅训练期：作者改了 3DGS 的光栅化模块来渲染索引图和深度点（第 6 节 Implementation Details）；这些量不进最终模型，解码端无需重算 |
| 有无侧信息 | 无（全部来自模型自身前向统计；I² 用到的 S_i 是投影面积，同样前向可算） |
| 训练期还是编码期 | 训练期（增密/简化调度嵌在 30K 步优化里） |
| 和渲染质量的距离（直接测质量还是代理量） | 代理量：重要性是前向贡献，不直接测质量；作者用 chamfer distance 间接验证采样保几何（第 4.2 节、Fig 6），最终以 PSNR/SSIM/LPIPS 验证 |

## 三学：证明与包装（创新怎么立住）

### 实验口径

数据集 Mip-NeRF360、Tanks&Temples、Deep Blending，完全沿用 3DGS 官方协议（场景选择、train/test 划分、分辨率，第 6 节）。指标：SSIM/PSNR/LPIPS/高斯数 Num（Table 1）。资源口径：训练时长、训练峰值显存（torch.cuda.max_memory_allocated）、FPS、渲染峰值显存（Table 2）。存储口径：Mini-Splatting-C 的码率-质量曲线（Fig 9），txt 未给具体 MB 数字。对手：重训的 3DGS*、NeRF 系（Zip-NeRF 等）、以及同数量剪枝系——VQ（把 [21] 的剪枝用于 3DGS）、LightGaussian（按官方仓库重训 prune+recovery）、Lee 等的数字取自其论文（第 6.1 节）。训练配置要点：30K 步、15K 关增密、15K/20K 简化、SH 15K 后才启用。

### 主结果（注明表号）

- Table 1：Mip-NeRF360 上 Mini-Splatting 0.49M 高斯 / 27.34 dB / SSIM 0.822，对 3DGS* 的 3.35M / 27.47 dB / 0.815——高斯少 7 倍画质基本持平；Deep Blending 上 0.35M / 29.98 dB 反超 3DGS* 的 2.82M / 29.54 dB；T&T 是弱点（23.18 对 23.66 dB），作者归因于 train 场景大片天空没有深度可用（第 6.1 节、附录 G）。Mini-Splatting-D 则全面小幅领先（Mip-NeRF360 27.51 dB、4.69M）。
- Table 2：Mip-NeRF360 室外，Mini-Splatting 0.57M 高斯、训练 17m56s、渲染 410 FPS、渲染峰值显存 0.40GB，对 3DGS 的 4.86M、30m8s、98 FPS、2.79GB；还把 Mini-Splatting* 在 GTX 1060 6G 上训成（室外 101m11s、64 FPS），支持端侧训练的主张。
- Fig 1（bicycle 单场景）：Mini-Splatting 0.6M / 430 FPS / 17min / 25.2 dB，3DGS 6.1M / 66 FPS / 35min / 25.2 dB。
- 存储：Fig 9 显示 Mini-Splatting-C 只靠基础后处理就压过专门做压缩的 Lee 等（Compact 3DGS）和 Niedermayr 等（正文注明的两条对比曲线 [20, 27]；txt 无具体数字，论文未报告）。

### 消融设计（注明表号）

- 逐组件开关（Table 3）：3DGS 基线 27.47 dB / LPIPS 0.216 / 3.35M → +blur split 27.47 dB / 0.195 / 3.74M → +depth reinit 27.54 dB / 0.175 / 4.32M。注意增密是"加高斯换质量"、简化是"减数量保质量"，两件事分开消融；作者还点出高斯数与 LPIPS 高度相关。
- 替换实验（Table 4）：深度来源 blending / center / mid 三选一——blending 崩（17.67 dB），center 与 mid 几乎同画质（27.57 对 27.54 dB）。作者照实承认 center 够用，选 mid 是看重点云重建质量（Fig 5）和法向估计的可扩展性。这种"差别不大也照实报"的写法值得学。
- 同数量对照曲线（Fig 7，本文最值得学的口径）：靠调重要性采样的 sampling ratio 扫出整条"高斯数-画质"曲线，与 VQ、LightGaussian（重训）、Lee 等在相同数量处直接比，结论是"our Mini-Splatting outperforms all its counterparts at similar numbers of Gaussians"（第 6.1 节）。固定预算、只比质量，正好堵住"你画质好是因为你高斯多"的质疑。
- 简化消融（Fig 10）：从 Mini-Splatting-D 出发，先加直接剪枝调成基线曲线 Add Pruning，再加 intersection preserving（Add Intersection）和采样（Add Sampling）；同等画质下高斯数约为直接剪枝的一半（第 6.2 节）。
- 初始化对照（Table 5）：sparse / dense 初始化 × 三个方法的对照表，证明增密算法的收益不靠初始化红利。
- 重要性度量对照（Fig 14，附录 E）：Imp1（纯 blending 权重）对 Imp2（交点过滤 + 面积归一），室内外互有胜负，作者把它降级为实验技巧。

### 贡献列表（原文照抄 + 逐条标注）

1. "Through a deep analysis of the spatial distribution of Gaussians, we observe that the vanilla 3DGS presents two significant phenomena: 'overlapping' and 'under-reconstruction'. We identify that this inefficient distribution limits the rendering quality and speed, and it is non-trivial to achieve a minimal Gaussian representation while maintaining rendering quality." —— [观察] 分布诊断，全文立论根基。
2. "We propose a Gaussian densification and simplification algorithm to reorganize the spatial positions of Gaussians rather than directly pruning them." —— [机制] 核心主张：重排位置，而非直接删。
3. "By integrating densification and simplification with corresponding further processing, our proposed Mini-Splatting achieves a balanced trade-off between rendering quality, resource consumption, and storage. Extensive experimental results on multiple benchmarks and datasets demonstrate the potential and scalability of our method." —— [系统] 三个变体分别覆盖资源、画质、存储需求。

### 讲故事方式

主线一句话："高斯没铺好，先铺好再谈删。"最有力的是 Fig 2：一张诊断图同时给出定性证据（中心投影的扎堆和空洞）与定量证据（四种删法 PSNR 全掉），让"直接剪枝不行"一眼可见；Fig 7 的同数量曲线负责收尾，证明铺好之后再删也能赢。 teaser（Fig 1）把 6.1M→0.6M、430 FPS、画质持平三个卖点压在一张图里。

## 对 DCCA-GS 的可借鉴点

1. blur split（未移植，值得补）：判据 S_i 只依赖前向光栅化的 argmax 索引图，成本极低，针对的正是 depth-reinit 管不到的 oversized 高斯——我们目前的基础措施只有深度反投影增密一路。可以加在训练期增密环节：对持续以最大贡献覆盖异常大屏幕面积的派生高斯做拆分或惩罚。要跑的实验：HAC++ 底座上 depth-reinit 与 depth-reinit + blur-split 对照，看同等 total_MB 下的 PSNR/LPIPS。
2. 简化两招（未移植，选择性补）：intersection preserving 是"只留真正参与过渲染的高斯"；importance-weighted sampling 用"按重要性概率采样"代替硬剪枝，保几何的证据是 Fig 6 的 chamfer distance。它与我们已有的 SPA（训练期 ADMM 剪枝）动的是同一个环节：SPA 是确定性软剪枝，采样是随机化保分布的删减。值得做"SPA vs 重要性采样 vs 叠加"的同预算对照；移植的主要工作量在锚点表示——采样对象要重新定义（对 anchor 还是 offset 派生高斯采样）。
3. 同预算实验口径（直接搬）：Fig 7 靠调采样比例扫出"质量-数量"整条曲线的做法，正是我们 SPA+MiniSplat 实验该学的——不要只报单点，固定底座、扫预算、画质量-高斯数（或质量-total_MB）曲线，对手用 VQ/LightGaussian 式直接剪枝。这套对照能把"我们的剪枝/量化信号在同预算下更好"立住。
4. 诊断证据的呈现（直接搬）：把基元中心投影到渲染图、配一组"删法全失败"对照（Fig 2），是我们论证"锚点/offset 分布也需要内容复杂度自适应"时可以直接复用的动机图形式，成本低、说服力强。
