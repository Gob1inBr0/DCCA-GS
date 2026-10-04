# 《Mind the Gap: Standard 3DGS Evaluation Primarily Measures Near-Trajectory Interpolation》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2607.01556 |
| 发表 | arXiv 预印本（v1，2026-07-02，AMD 公司工作，未标注会议接收信息） |
| 阅读材料 | 全文（.lit-cache/txt/2607.01556.txt，14 页，抽取完整；图 2、图 4a 的个别面板数字用 pymupdf 从 PDF 第 6、7、11 页复核过） |
| 与 DCCA-GS 的关系 | 评测参考：不和我们抢任何方法主张，但直接约束"我们论文的对比数字该怎么解释、该怎么报告" |

> 一句话定位：这篇没有提出任何新表示或新压缩方法，它用一组"训练图数量完全相同、只改留出视角怎么挑"的对照实验证明：社区用了三年的每 8 张留一张的评测协议，测出来的其实是轨迹附近的插值质量；换成空间上连续留出的外推设置，同样的模型掉 3–12 dB，比绝大多数论文里方法之间零点几 dB 的差距大一个量级。所有压缩论文的 RD 对比结论都建在这个协议上，所以这篇值得精读。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

标准协议把图像按文件名排序、每 8 张留 1 张做测试（第 3.1 节），每张测试图两侧都有训练过的邻居。它的原话："the metric primarily measures how well the model interpolates between nearby trained views rather than how well it generalizes to unseen spatial regions"——指标主要在测模型在相邻训练视角之间插值的能力，而不是它对没见过的空间区域的泛化能力。更要害的是第 1 节那句："A method comparison claiming '0.5 dB better' is making a claim smaller than the gap between evaluation modes"——论文里常见的"我们比基线好 0.5 dB"这种话，比评测方式本身造成的差异还小。作者认为这会错误引导方法选择：大家都在优化"复现训练视角"，而不是"泛化到新视角"。

### 支撑观察（直觉 / 数据 / 失败实验）

- 第 1 节：插值下 30 dB 以上的模型，空间留出下掉到 18–25 dB（表 3：counter 场景插值 29.00 → 外推 17.91）。
- 图 2（第 4 节）：同一张测试图，插值臂模型渲染 34.4 dB 和 27.2 dB，外推臂模型只有 23.9 dB 和 14.6 dB，而两个模型训练图数量完全一样。
- 第 4.2 节：16 个场景 × 3 个方法共 48 个组合，外推全部变差，无一例外。
- 第 5.1 节图 3：外推测试视角到最近训练视角的平均角距离 15.2 度，插值视角只有 3.7 度（4.1 倍）；逐图 PSNR 与角距离的相关系数 r = −0.58（n = 492，p < 10−44），且在 9 个场景内部逐个成立（场景内平均 r = −0.66）。
- 这些观察全部来自它自己跑的 502 次训练（附录 C 给了次数明细），不是作者直觉。

### 显然的路为什么没走

1. 显然的路一：直接换一套现成的外推基准（如 EUVS、VEGS 的城市外推数据）。它不换数据集，而是在标准数据集上做双臂对照、两臂训练图数量取齐。原因：换数据集会把场景本身难度的差异混进来；它要隔离的唯一变量是"留出视角的空间分布"，所以在同一场景上构造两种留出方式、训练图数量完全相同（第 3.2 节）。
2. 显然的路二：发明一个正则化方法把泛化补回来（DropoutGS、DropGaussian 那条线）。它真的测了六种训练期正则化（表 16），最好的只涨 0.09–0.17 dB，激进的反而让训练崩掉（CAT 掉 3.10 dB）。把这条路走了一遍并用数据证明它不通，这本身就是论文论证的一部分——差距是数据覆盖问题，不是优化问题。
3. 显然的路三：用学出来的不确定性或覆盖度估计（FisherRF、COVER、PRIMU）做诊断。这些都要先训练好一个模型（表 10）；它选了只需要相机位姿、训练前就能算的"最近训练视角角距离"，零成本，还能反过来指导拍摄规划。

## 二学：机制（洞察怎么变成方法）

这篇是评测方法论文，没有训练/编码/解码流程，下面按协议构造、诊断分析、应用三层讲。

### 核心流程

**公平匹配数协议（第 3.2 节，图 1）**，输入是场景的 N 个相机和位姿：

1. 插值臂（标准协议）：按文件名排序，每 8 张留 1 张，留出 K = ⌈N/8⌉ 张，用 N−K 张训练，在 K 张留出图上评测。每张测试图两侧都有训练邻居。
2. 外推臂：把相机绕场景质心按方位角排序，从一个固定随机起点（种子 42）开始，留出连续的 K 个相机，同样 N−K 张训练、K 张评测。测试视角到最近训练视角的角距离大幅变大。
3. 两臂训练图数量严格相同、留出图数量严格相同，唯一差别是留出视角在空间上摊开还是连成一片。协议里用到的每场景划分文件全部公开在附录（附录 D 的工件清单）。

**诊断分析（第 5 节）**：同一批训练好的模型，渲染时把球谐次数压到 0（只剩漫反射颜色）再算一次差距，把总差距拆成"几何占位分量"（SH0）和"视角相关分量"（SH3 − SH0）（第 5.2 节，表 5）；再用完全无高斯、无球谐的 Instant-NGP 在相同划分文件下复跑一遍，看差距是否还在（第 5.3 节，表 6）。

**两个对照实验排除混淆（第 4.2 节）**：同图对照——两臂评测集交集里的 31 张图，同一张真值、同一视角，只有训练集不同，差距仍 +5.81 dB（附录 G 表 11），说明不是"外推图本身更难"；最近图拷贝基线——直接拷贝最近训练视角的真值图，两臂只差 0.9 dB（插值 12.6 / 外推 11.7，附录 H 表 12），说明差距不能靠"外推区的真值图和训练图长得不像"来解释。

**拍摄规划应用（第 6 节，表 7）**：从外推臂训练集出发，每次从留出扇区里加 M 张图，用"贪心挑离现有训练视角最远的角度"和随机挑各做 5 次重复、共 240 次运行，验证覆盖度导向的选视角确实比随机好（31/40 组合胜出，M=1 时 9/10 场景胜出，平均 +0.60 dB）。

**工具箱（附录 E）**：create_sector_holdout.py 生成扇区留出划分、evaluate_lto.py 一条命令评测、coverage_diagnostic.py 只用位姿输出平均最近邻角距离、最大角空隙和建议补拍位置；附 16 场景 × 3 方法的基线分数。代码"正在准备公开"（脚注 1），截至本版本尚未放出。

### 关键公式（按原文抄，注明编号）

全文只有一条编号公式（第 3.2 节，式 1）：

Δ = PSNR_interp − PSNR_extrap（式 1）

- Δ：同一方法在同一场景上插值臂与外推臂的 PSNR 差，正值表示插值更好。
- PSNR_interp：插值臂（每 8 张留 1 张）训练出的模型在留出图上的 PSNR。
- PSNR_extrap：外推臂（连续扇区留出）训练出的模型在留出图上的 PSNR。
- 两臂训练图数都是 N−K，所以 Δ 不含训练数据量的影响。原文没有其他编号公式；表 5 里 Gap@SH0、Gap@SH3 是把式 1 用在 SH 次数 0 和 3 的渲染上，VD contribution = Gap@SH3 − Gap@SH0。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | 两个：(1) 协议变量——留出视角的空间分布（摊开 vs 连续扇区）；(2) 诊断信号——每个测试视角到最近训练视角的角距离（第 5.1 节、表 10） |
| 哪一端算得出 | 只需要相机位姿（COLMAP 就有），训练前就能算，不依赖任何训练好的模型 |
| 有无侧信息 | 无，纯评测侧和拍摄规划侧的量，不进码流 |
| 训练期还是编码期 | 都不是，属于数据采集阶段和评测阶段 |
| 和渲染质量的距离 | 是代理量：与逐图 PSNR 相关 r = −0.58（第 5.1 节），作者说明它比学出来的不确定性方法简单，且在他们的数据上没有被更复杂的特征或非线性模型超过 |

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 质量指标：PSNR / SSIM / LPIPS。不涉及体积和码率（不是压缩论文，论文未报告任何模型大小数字）。
- 数据集（第 4.1 节）：10 个实拍场景——MipNeRF360 的 9 个（室外 bicycle/flowers/garden/stump/treehill 用 images_4 四分之一分辨率，室内 bonsai/counter/kitchen/room 用 images_2 二分之一分辨率）加 Tanks and Temples 的 truck；6 个生成视角场景——HY-World 2.0 管线的 case000（199 张）、case001（178 张）和作者用 CC0 全景图生成的 courtyard/gazebo/yaris/alps（各 283 张），全部转成 COLMAP 格式。
- 方法（第 4.1 节、表 9）：Official 3DGS（gsplat 实现）、Mip-Splatting（kernel size 0.1）、3DGS-MCMC，都用默认超参、30K 迭代；Instant-NGP 用 20K 步；前馈模型 MVSplat、DepthSplat 用各自官方权重（附录 I）。
- 规模（附录 C）：502 次基于优化的训练（主表 96 + Official 多种子 40 + Mip 多种子 40 + 多扇区 32 + 正则化 54 + 拍摄规划 240），另有 130 次前馈推理评测。基线全部锁定上游 commit（附录 D：3DGS b9da16c、Mip-Splatting dda02ab、MCMC 7b4fc9f）。
- 多种子（附录 B）：Official 3DGS 在 10 个场景各跑 3 个种子；Mip-Splatting 在两个翻转场景加跑 2–5 个种子；MCMC 和其余 Mip 行是单次运行。

### 主结果表抄录

**表 3（实拍场景，公平匹配数协议；Official 3DGS 行为 3 个种子的均值±标准差，Mip 与 MCMC 除翻转验证外为单次运行；IntR/ExtR = 插值/外推排名，1 最好；⋆ = 多种子确认的 Mip 与 Official 排名翻转）**，共 30 行 + 3 行均值，全表抄录：

| 场景 | 方法 | 插值 | 外推 | 差距 | IntR | ExtR |
| --- | --- | --- | --- | --- | --- | --- |
| bicycle⋆ | Official 3DGS | 25.12±.12 | 18.89±.15 | 6.23±.03 | 3 | 1 |
| bicycle⋆ | Mip-Splatting | 25.27 | 18.63 | 6.64 | 1 | 2 |
| bicycle⋆ | 3DGS-MCMC | 25.22 | 18.53 | 6.69 | 2 | 3 |
| flowers | Official 3DGS | 21.49±.07 | 18.15±.06 | 3.35±.02 | 2 | 2 |
| flowers | Mip-Splatting | 21.36 | 18.10 | 3.27 | 3 | 3 |
| flowers | 3DGS-MCMC | 21.52 | 18.35 | 3.17 | 1 | 1 |
| garden⋆ | Official 3DGS | 27.29±.05 | 23.65±.09 | 3.63±.06 | 3 | 1 |
| garden⋆ | Mip-Splatting | 27.37 | 23.02 | 4.35 | 2 | 3 |
| garden⋆ | 3DGS-MCMC | 27.46 | 23.57 | 3.89 | 1 | 2 |
| stump | Official 3DGS | 26.41±.24 | 21.64±.50 | 4.77±.26 | 2 | 1 |
| stump | Mip-Splatting | 26.32 | 21.42 | 4.90 | 3 | 3 |
| stump | 3DGS-MCMC | 26.72 | 21.59 | 5.13 | 1 | 2 |
| treehill | Official 3DGS | 22.38±.12 | 16.36±.09 | 6.03±.04 | 2 | 1 |
| treehill | Mip-Splatting | 21.94 | 16.14 | 5.80 | 3 | 3 |
| treehill | 3DGS-MCMC | 22.40 | 16.26 | 6.14 | 1 | 2 |
| bonsai | Official 3DGS | 32.07±.06 | 24.34±.23 | 7.74±.19 | 2 | 2 |
| bonsai | Mip-Splatting | 31.97 | 24.24 | 7.73 | 3 | 3 |
| bonsai | 3DGS-MCMC | 32.44 | 25.20 | 7.24 | 1 | 1 |
| counter | Official 3DGS | 29.00±.07 | 17.91±.15 | 11.09±.22 | 3 | 3 |
| counter | Mip-Splatting | 29.15 | 18.19 | 10.95 | 2 | 2 |
| counter | 3DGS-MCMC | 29.18 | 18.31 | 10.87 | 1 | 1 |
| kitchen | Official 3DGS | 31.18±.14 | 23.91±.12 | 7.26±.23 | 3 | 1 |
| kitchen | Mip-Splatting | 31.45 | 23.78 | 7.67 | 1 | 3 |
| kitchen | 3DGS-MCMC | 31.45 | 23.79 | 7.66 | 2 | 2 |
| room | Official 3DGS | 31.39±.15 | 26.85±.24 | 4.54±.33 | 3 | 3 |
| room | Mip-Splatting | 31.59 | 27.13 | 4.46 | 2 | 2 |
| room | 3DGS-MCMC | 31.69 | 27.27 | 4.42 | 1 | 1 |
| truck | Official 3DGS | 25.29±.08 | 21.48±.16 | 3.81±.10 | 2 | 2 |
| truck | Mip-Splatting | 25.26 | 21.28 | 3.98 | 3 | 3 |
| truck | 3DGS-MCMC | 26.03 | 22.08 | 3.95 | 1 | 1 |
| 均值 | Official | 27.16 | 21.32 | 5.84 | | |
| 均值 | Mip-Splat | 27.17 | 21.19 | 5.98 | | |
| 均值 | MCMC | 27.41 | 21.50 | 5.92 | | |

**表 2（6 个生成视角场景，HY-World；括号内为该列方法排名；† = 单次运行下 Mip 与 Official 的排名翻转）**，共 18 行 + 3 行均值，全表抄录：

| 场景 | 方法 | 插值 | 外推 | 差距 |
| --- | --- | --- | --- | --- |
| case000 | Official 3DGS | 31.36 (3) | 19.40 (3) | 11.96 |
| case000 | Mip-Splatting | 31.68 (2) | 19.67 (2) | 12.01 |
| case000 | 3DGS-MCMC | 31.82 (1) | 19.85 (1) | 11.97 |
| case001† | Official 3DGS | 24.34 (3) | 14.63 (2) | 9.71 |
| case001† | Mip-Splatting | 24.52 (2) | 14.44 (3) | 10.08 |
| case001† | 3DGS-MCMC | 25.67 (1) | 14.67 (1) | 11.00 |
| courtyard | Official 3DGS | 21.83 (3) | 14.40 (3) | 7.43 |
| courtyard | Mip-Splatting | 22.00 (2) | 14.62 (2) | 7.38 |
| courtyard | 3DGS-MCMC | 22.45 (1) | 14.64 (1) | 7.80 |
| gazebo† | Official 3DGS | 20.67 (3) | 12.50 (1) | 8.17 |
| gazebo† | Mip-Splatting | 21.29 (2) | 11.89 (3) | 9.40 |
| gazebo† | 3DGS-MCMC | 22.60 (1) | 12.36 (2) | 10.24 |
| yaris | Official 3DGS | 17.74 (3) | 10.56 (3) | 7.18 |
| yaris | Mip-Splatting | 18.58 (2) | 10.82 (2) | 7.75 |
| yaris | 3DGS-MCMC | 18.62 (1) | 10.90 (1) | 7.72 |
| alps | Official 3DGS | 17.34 (3) | 14.08 (3) | 3.27 |
| alps | Mip-Splatting | 17.96 (2) | 14.31 (2) | 3.65 |
| alps | 3DGS-MCMC | 19.04 (1) | 14.58 (1) | 4.46 |
| 均值 | Official | 22.21 | 14.26 | 7.95 |
| 均值 | Mip-Splat | 22.67 | 14.29 | 8.38 |
| 均值 | MCMC | 23.37 | 14.50 | 8.87 |

生成场景的差距（三方法均值行 7.95–8.87 dB）约为实拍（5.84–5.98 dB）的 1.4 倍（第 4.2 节），作者归因于扩散模型生成的相机轨迹同样平滑。SSIM 和 LPIPS 方向一致：表 8（单种子 Official）均值 PSNR 27.02 → 21.25（+5.77）、SSIM 0.815 → 0.692、LPIPS 0.214 → 0.300；第 4.2 节正文引 SSIM 平均差 0.124、LPIPS 0.086（附录 A）。

### 消融设计（注明表号）

值得学走的对照设计，按论证作用列：

- **同图对照**（附录 G 表 11）：两臂评测集交集的 31 张图上差距 +5.81 dB 且 31 张全正，其中 counter +9.40、bonsai +8.35——排除"外推图本身更难"。
- **最近图拷贝基线**（附录 H 表 12）：拷贝最近训练视角真值，插值 12.6 / 外推 11.7 dB、只差 0.9 dB（SSIM 差 0.013）——排除"真值相似度差异"。这两个对照是它区别于以往"观察型"论文的关键。
- **多种子验证**（附录 B）：30 个场景-种子组合差距全正；种子间差距标准差 0.02（flowers）到 0.33 dB（room），平均 0.15 dB；平均差距 5.84 dB 是平均种子波动的 39 倍，最差的 room 也有 13.6 倍。
- **多扇区旋转**（表 4）：4 个场景各转 8 个扇区起点共 32 次运行，每个位置差距都为正（garden +2.3±1.0、bicycle +4.7±2.5、kitchen +6.1±1.6、bonsai +6.6±1.6），幅度随区域难度变、方向不变；室内两个场景同样成立，说明不需要空间上严格连通的留出区。
- **球谐次数分解**（表 5，图 4）：SH0（纯漫反射）分量平均占 62%，视角相关分量占 38%；bicycle/stump/treehill 三个场景上提高 SH 次数反而降低外推 PSNR（视角相关颜色过拟合到训练方向）。作者自己标注了局限：SH 可以补偿几何误差（"appearance compensation"），两个分量相互作用、不是严格可加。
- **跨表示验证**（表 6）：Instant-NGP 在相同划分下 9 个场景差距全正，均值 +3.73 dB（95% 置信区间 [2.14, 5.33]；符号检验 p = 0.002，配对 t 检验 p = 6×10−4），且逐场景差距与 3DGS 差距相关 r = 0.90（p = 0.001）——counter 最大、flowers/stump 最小在两种表示下一致，说明是场景几何层面的共同原因。前馈模型同样复现：MipNeRF360 零样本 256×256 下 MVSplat +0.99±1.00 dB、DepthSplat +1.56±1.16 dB（54 个组合中 46 个为正，符号检验 p < 10−5，表 13/14）；换到域内 RealEstate10K 38 个测试场景、只改目标帧远近，DepthSplat +7.3 dB（25.5 → 18.3）、MVSplat +9.1 dB（25.7 → 16.6）（第 7 节）。
- **正则化消融**（表 16）：六种训练期干预对 10 个实拍场景的外推 PSNR——SH 次数退火 +0.17 dB（9/10 场景改善）、视角相关正则 +0.12（9/10）、尺度正则 +0.09（8/10）、30% 高斯丢弃 −0.39（2/10）、2 度相机抖动 −2.85（2/10）、激进的 CAT −3.10（0/9 场景，训练不稳）。全部远小于 5.84 dB 的差距本身。
- **下游任务传播**（附录 J 表 15）：6 个场景上，外推渲染图估出的深度（Depth Anything V2）与真值照片深度的不一致在 5/6 场景变差（RMSE +42% 到 +248%，room 例外 −28%）；无参考感知质量 MUSIQ 在 6/6 场景下降（均值 −5.1%）。
- **协议审计**（表 17）：核查 11 篇近期论文的源码或方法节，全部在平滑轨迹数据上用 every-8 留出——包括 3DGS、Mip-Splatting、3DGS-MCMC、2DGS、Scaffold-GS、Compact3D、AbsGS、DropGaussian、DropoutGS、EDGS、DropAnSH。这条对我们尤其重要：Scaffold-GS 是 HAC++ 的底座，压缩线论文全部在这个口径下比较。

### 贡献列表（原文照抄 + 逐条标注）

1. "A fair matched-count evaluation protocol that measures interpolation and extrapolation on genuinely unseen images with identical training data counts (§3)." —— [机制] 训练图数取齐的双臂对照协议，把评测差异归因到留出视角的空间分布这一个变量。
2. "A systematic quantification of the gap across 16 scenes, 3 methods, and 502 training runs with targeted multi-seed validation—showing the gap is 39× larger than training noise and is, in two multi-seed-confirmed cases, consequential for method ranking (§4)." —— [证据] 16 场景 × 3 方法的大规模量化加定点多种子验证。
3. "Mechanistic decomposition: SH-degree analysis attributes 62% of the gap to a diffuse/geometry-proxy component and 38% to view-dependent color, with higher SH degrees hurting extrapolation on 3/9 scenes (§5). Per-image quality correlates with nearest-training-view distance (r = −0.58, n = 492), providing a zero-cost diagnostic." —— [机制] 球谐次数分解加零成本角距离诊断。
4. "Coverage-guided view selection: a zero-cost heuristic that outperforms random placement on 31/40 scene-budget combinations (10 scenes, 240 runs), strongest at M =1 (9/10 scenes)." —— [工程] 把诊断信号用于拍摄规划。
5. "Cross-representation generality: under our matched-count protocol the gap persists in a non-Gaussian NeRF (Instant-NGP; +3.7 dB, r = 0.90 with the 3DGS gap), and an analogous gap appears in two feed-forward methods under a distance-based target protocol (MVSplat, DepthSplat; +7–9 dB in-domain on RealEstate10K)—the effect tracks spatial coverage, not Gaussian primitives (§5, §7)." —— [证据] 跨三类表示家族的普遍性。
6. "A spatial-holdout benchmark toolkit: split generation, a one-command evaluator, and a pose-only coverage diagnostic, with standardized splits and baseline metrics for 16 scenes (Appendix E)." —— [工程] 工具箱（代码标注"准备中"，尚未放出）。

### 讲故事方式

主线一句话：你用了三年的评测协议测的不是它名字里承诺的东西，而且测错的那部分比论文之间比来比去的差距大得多。最有力的工件是图 2 加表 3 的组合：图 2 给出两张同图对照的视觉证据（同一测试图，插值模型 34.4/27.2 dB，外推模型 23.9/14.6 dB），表 3 用 48 个组合全正的完整性和多种子把结论钉死；再配上"方法间差距 <2.5 dB"这个参照系，让"3–12 dB"有了痛感。标题 "Mind the Gap" 本身就是全文的包装。

## 四学：原文关键段翻译（深读层必填）

### 方法节核心段（选 2 段）

> **原文（第 3.1 节）**："The standard 3DGS evaluation protocol [1] sorts images by filename, holds out every 8th image as the test set, and trains on the remaining ~87.5%. Each held-out image has trained neighbors on both sides—the metric measures how well the model interpolates between nearby views, not how well it synthesizes genuinely novel viewpoints from unseen spatial regions."
>
> **译文**：标准 3DGS 评测协议把图像按文件名排序，每 8 张留出 1 张作为测试集，在剩下约 87.5% 上训练。每张留出图两侧都有训练过的邻居视角——所以这个指标衡量的是模型在邻近视角之间插值的能力，而不是它从训练中未见过的空间区域合成真正新视角的能力。

> **原文（第 3.2 节）**："Both arms train on exactly N−K images and evaluate on exactly K genuinely unseen images. The designed contrast is whether the held-out images are spatially interleaved with training views (interpolation) or concentrated in an unseen region (extrapolation). By fixing the same held-out count K for both arms, the comparison isolates the effect of the holdout's spatial distribution rather than the amount of training data."
>
> **译文**：两臂都在恰好 N−K 张图像上训练，都在恰好 K 张真正未见过的图像上评测。设计出的对照是：留出图像在空间上与训练视角交错分布（插值），还是集中在一片未见过的区域（外推）。由于两臂留出数量 K 固定相同，这个比较隔离的是留出视角空间分布的影响，而不是训练数据数量的影响。

### 主结果表述段（选 1 段）

> **原文（第 4.2 节 Real captures 段）**："Table 3 shows the gap across 10 real-capture scenes. Every scene and every method shows a positive gap, ranging from 3.2 to 11.1 dB (mean 5.84 dB for Official 3DGS). The gap is consistent across all three metrics: SSIM shows a mean gap of 0.124 and LPIPS of 0.086 (Appendix A). It is consistently larger than the inter-method performance differences on the same scenes (<2.5 dB on standard benchmarks)."
>
> **译文**：表 3 展示了 10 个实拍场景上的差距。每个场景、每种方法都出现正差距，范围 3.2 到 11.1 dB（Official 3DGS 平均 5.84 dB）。差距在全部三项指标上一致：SSIM 平均差 0.124，LPIPS 平均差 0.086（附录 A）。它始终大于同一批场景上方法之间的性能差异（标准基准上小于 2.5 dB）。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | 按协议留出会导致外推失真这件事本身不新：Nerfbusters、EUVS、VEGS、FreeVS、NeRF Director、NerfBaselines 都观察过（该文自己的表 1 承认）。新的组合是：训练图数取齐的双臂对照（排除数据量混淆）+ 大规模量化 + 同图/拷贝双对照 + SH 分解与角距离诊断 + 跨表示验证，全部叠在一份协议里（表 1 显示它是第一个全占的）。单项均有人做过，组合和量化深度是新的 |
| 证据强度 | 高 | 502 次训练、3 种子多种子验证、两个专门对照实验、8 扇区旋转鲁棒性、跨 3 个表示家族、统计检验齐全、基线锁定 commit、附录 D 的工件清单把每个数字映射到结果文件。弱点：工具箱代码"准备中"未放出（脚注 1），Mip/MCMC 主表多为单次运行，前馈实验绝对 PSNR 只有 8–10 dB（域失配） |
| 对期刊版的威胁度 | 中 | 不撞 DCCA-GS 的三条方法主张（渐进分层码流、零侧信息、内容自适应量化），它不提出任何压缩方法。它动摇的是评测口径的解释权：我们的 150 视角自采数据如果沿航线顺序均匀留出，报告的 PSNR 同样只是轨迹附近插值质量，审稿人可以引这篇文章要求补空间留出实验或至少补覆盖度诊断。缓解：它自己的数据显示差距在不同方法上几乎相同（表 3 均值行 5.84/5.98/5.92 dB），同一划分下的相对比较（我们与 HAC++ 系的 RD 排名）内部仍然成立，且它明说标准留出对轨迹附近渲染仍然有效 |

## 对 DCCA-GS 的可借鉴点

1. **零成本覆盖诊断写进我们的数据集节**（学它的角距离信号，附录 E 的 coverage_diagnostic.py 思路）：对我们的 150 视角自采数据算每个测试视角到最近训练视角的角距离分布（均值、最大空隙），一次性说清我们的划分属于插值主导还是外推。只需位姿，不用重训任何模型。预期收益：提前堵住这一类审稿意见，也顺便确认我们的场景视角间距本来是否就比较稀（若本来角距离就大，标准协议的失真在我们的数据上会比 MipNeRF360 轻）。要跑的实验：对 150 视角逐场景统计一遍，写成数据集描述的一小段。
2. **同图对照 + 拷贝基线这两个对照设计**（表 11、表 12）：如果审稿人质疑我们自采数据上对比的公平性，这两个对照可以直接搬——同图对照查"留出区域难度不均"，拷贝基线给我们数据上 PSNR 的地板参照。成本各一次评测脚本。
3. **多种子噪声参照的报告写法**（附录 B）：期刊版报 RD 增益时，补 1–2 个场景 × 3 种子估出训练随机波动，把我们的增益表述成"是种子间波动的多少倍"。它用 39 倍/13.6 倍这种写法让 5.84 dB 的结论立住，我们同样可以用极低的成本让关键增益更硬。
4. **按它的建议清单报告协议**（第 7 节建议 4 条）：保留 every-N 结果保证可比性，写明留出构造规则，报告逐测试视角角距离。若期刊版有余力，加一个场景的空间留出 RD 曲线作为附加实验，证明我们的渐进码流在外推设置下排序不翻转。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 评测划分怎么定 | 双臂对照：每 8 张留 1 张（插值臂）对比方位角连续扇区（外推臂），两臂训练图数严格相同（第 3.2 节） | 自采无人机航拍 150 视角、1600px 分辨率设置；留出规则在本笔记依据的材料中未写明，但底座 HAC++ 系论文全部用顺序均匀留出（该文表 17 审计确认 Scaffold-GS 等 11 篇如此） | 它证明顺序均匀留出只测轨迹附近插值。对我们的适用性：无人机航线若是绕圈或网格，方位角扇区未必对应连续空间区域（该文 Limitations 自己承认室内场景有同样问题）；结论应限定为"轨迹附近渲染质量"，并按可借鉴点 1 补角距离统计 |
| RD 对比结论怎么解释 | 方法间插值差距 <2.5 dB，小于协议本身造成的 3–12 dB；两个多种子确认的排名翻转（第 4.2 节：garden 插值 Mip 高 0.07、外推 Official 高 0.52，93% 种子组合；bicycle +0.15 / +0.22，89%；表 3 中以 ⋆ 标注，表内单次运行数值略有出入） | 与 HAC++ 系（HAC++/HAC/ContextGS/CAT/PC-GS）在同一划分下比压缩 RD | 它的数据同时显示差距本身几乎不随方法变（表 3 均值行 5.84/5.98/5.92 dB），所以同一划分下的相对排名大体保得住；受冲击的是"某方法泛化更好"这类说法和零点几 dB 量级的主张。我们的压缩增益若达到 dB 以上量级，受此威胁很小，但仍应写明结论属于插值口径 |
| 诊断信号与侧信息 | 最近训练视角角距离：零成本、只需位姿、训练前可用，是唯一不用训练模型就能做拍摄规划的方法（表 10 与 FisherRF/COVER/PRIMU 对比） | 零侧信息契约：解码端能重算的才可以用，否则必须写进码流 | 不冲突：它的信号只在评测侧和采集规划侧，不进码流。可直接借用到我们数据集描述和补拍建议，不动我们的码流契约 |
| 渐进可截断 / 熵编码 | 该论文不涉及 | 单文件渐进分层码流、分字段粗化、层条件熵编码 | 该论文不涉及 |
