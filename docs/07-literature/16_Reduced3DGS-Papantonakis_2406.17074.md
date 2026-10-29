# 《Reducing the Memory Footprint of 3D Gaussian Splatting》精读笔记

- arXiv：https://arxiv.org/abs/2406.17074
- 发表：I3D 2024（Proc. ACM Comput. Graph. Interact. Tech. Vol.7 No.1，2024 年 5 月）；作者属 Inria，为 3DGS 原班团队（Kerbl、Drettakis）

> 一句话定位：3DGS 作者组对自己方法的减负方案——训练期分辨率感知剪枝（砍约 60% 高斯）、训练中途逐高斯自适应降 SH 阶数、训练后 K-means 码本量化加半精度浮点，三者合计磁盘体积 27x、平均只掉 0.21 dB，渲染反而快 1.7x（摘要、Fig. 3）。

## 问题与体积构成

3DGS 质量与速度都好，但"memory requirements ... unreasonably high"（700 MB–1.2 GB/场景，第 1 节）。体积拆成三个来源（第 1 节）：高斯数量太多（致密化"wasteful"）、SH 系数太多（全场一律 3 阶，没有视相关效果的地方也用 3 阶）、属性精度太高（多数属性不需要高动态范围）。第 3 节的体积构成：每高斯 14 + 3·Σ(2i+1) 个 float，3 阶 SH 时共 59 个 float，其中 45 个（76%）是 SH。

支撑观察（第 3 节）："3DGS creates unnecessarily dense sets of primitives"，致密化只看位置梯度，无法预知最终需要多少高斯；高斯过密是空间局部现象，按 opacity 一刀切对准不了"哪里过密"。多数场景大部分是漫反射材质，RGB 就够；但视相关效果也可能由多个高斯组合形成（反射面后的"反射几何"），所以不能全场统一降 SH 阶数，要逐高斯判。opacity、scale、rotation、SH 系数动态范围小、对小误差不敏感，可量化；位置不行——"reducing the accuracy of the positions ... leads to a significant degradation in quality"。Fig. 8 的反向证据：删 scaling 最小的高斯比删 opacity 最低的高斯伤质量得多。

## 方法

三个组件动在三个不同环节（Fig. 3 标注："1. Pruning (start-to-end)、2. SH Assignment (halfway)、3. Codebook (after)"）。

**1. 分辨率感知剪枝**（第 4.1 节，训练全程每 1000 次迭代一次）：对高斯 g，找能看到它的视角中最近的一个（像素足迹 a_min 最小），以"a radius equal to √a_min·√3/2, i.e., half the diagonal of a cube where each face has area a_min"为半径做球（txt 符号抽取损坏，此处按原文文字描述），数球内与 g 相交的其他高斯个数（k-NN 取 30 邻居做候选，椭球-点相交近似）。冗余分数在空间传播时取相交区域中的最小值（保守原则：只要碰到不冗余的区域就给低分）。阈值 τ_p = max(μ + λ_r·σ, 3)，λ_r = 1（分数是整数，即实际保留分数 ≥4 的候选；分数低于 3 的高斯可能提供细节，不删）。命中候选不一次全删，只删其中 50%，按低 opacity 挑。另有简单的低 opacity 剪枝：每次删 3% 最低 opacity（上限 0.05）。两路合计约剪 60%。配合训练改动：opacity 的 L1 稀疏损失（鼓励低贡献高斯出现）；15K 之后继续删 opacity < 1/255 的高斯（第 5 节）。

**2. 逐高斯自适应 SH 阶数**（第 4.2 节，15K 迭代、致密化刚停止时执行一次，留后半程训练补偿降阶损失）：判据 (a) 全视角颜色方差——按平均透射率加权算各视角评估颜色的逐通道均值 μc 和标准差 σc（式 (1)(2)；T̄_i = Σ_k T_ik / P，P 是该高斯在第 i 视角溅到的像素数），σc < ε_σ = 0.04 的高斯视为视角无关，只留 0 阶（RGB）。判据 (b) 降阶距离——用前 q 阶（q ∈ [0,1,2]）算颜色 c_q，与全 3 阶颜色 c_3 的欧氏距离按视角加权平均得 d_q，取最小的 q 使 d_q < ε_cdist = 0.04，删更高阶。结果分布：89% / 0.1% / 2.7% / 8.2% 的高斯保留 0/1/2/3 阶。存储时按 4 个阶数组分开存（第 5 节），阶数选择由此隐式记录，无额外侧信息。配合 SH 系数稀疏损失（让高阶系数本来就更接近 0，降阶代价更小）。

**3. 码本量化 + 半精度**（第 4.3 节，训练结束后的后处理，不动训练）：K-means 码本，索引 1 字节（码本 256 条）；向量属性共享一个码本但三个分量各自索引。量化字段：opacity（1 个码本）、scaling 三分量（1 个共享码本）、四元数实部（1 个）、虚部（1 个）、基色 RGB（1 个）、15 组 SH 系数每组 3 通道各 1 个（15 个）——共 20 个码本。体积从 56N×4 字节降到 56N + 20×256×4 字节，码本常数项约 20 KB，可忽略。位置不量化；位置和全部码本条目转 16-bit half-float。这份 arXiv v1 全文不存在"weight prediction / 存残差还是存预测"的机制（全文检索无此内容；该机制应属作者后续扩展版本）：码本存聚类中心本身，高斯侧只存索引，无逐高斯残差。也没有熵编码（码本索引定长 1 字节，无 Huffman/算术编码）。

## 实验口径

- 体积口径：磁盘文件大小（表内记 "Mem"，MB），计入全部属性（码本索引 1 字节/分量 + 码本开销 20×256×4 字节 + half-float 位置与码本条目）；没有学习型解码器，也没有熵编码收益，27x 全部来自"高斯更少 + SH 更少 + 精度更低"。当前实现是解析时解压、渲染不走码本（第 5 节），第 6.1 节明说"integrating codebooks and half-float type handling into the renderer would enable the same reduction for a scene's VRAM consumption"——当前 VRAM 未同步缩到 1/27。
- FPS 口径：只测 CUDA 光栅化例程本身，不含 blitting、swapchain 等图形 API 开销（第 6 节开头注明）。
- 数据集与协议：3DGS 原论文全套场景（MipNeRF360 全部、Deep Blending 2 景、Tanks&Temples 2 景），30K 迭代；基线用官方 GitHub 代码复现（3DGS*，数字比原论文略好，Table 2 脚注）。RTX A6000。

## 主结果

- Table 2：MipNeRF360 上 3DGS* 744 MB / 27.42 PSNR / 178 FPS → Ours 29 MB / 27.10 PSNR / 284 FPS（25.7x、掉 0.32 dB、快 1.6x）；Tanks&Temples 412→14 MB（27.6x，23.66→23.57）；Deep Blending 630→18 MB（34.8x，29.47→29.63 反升）。三档配置：Ours 29/14/18 MB，Low 46/21/35 MB，High 23/10/13 MB。
- Fig. 3 三组件累计贡献：剪枝 +0.03 dB / 2.37x；+SH 自适应 −0.14 dB / 8.0x；+量化 −0.21 dB / 27.4x。
- Fig. 1 与 Table 7（bicycle 逐场景）：1.4 GB（表中 1362 MB）→48 MB，PSNR 25.10→25.06，高斯 6.05M→2.41M，115→260 FPS；手机 WebGL 下载 140s→4.8s（Fig. 1 标注；第 6.1 节另报本地 wifi 下 120s→5s、约 24x）。
- Table 1（三组件渐进累加，每个数据集一列，逐级报 Mem 倍数和指标）：Mip360 744→339（×2.2）→102（×7.5）→29 MB（×25.7）。
- Table 3（剪枝策略对照，Mip360）：只按 opacity 剩 1.64M（48%）、只按冗余分数+低 opacity 挑 1.88M（56%）、冗余分数+随机挑 1.91M（57%）、完整方法 1.46M（43%）——冗余分数与 opacity 结合比任何单一策略多剪 9%。
- Table 4（half-float 消融）：Mip360 上仅 K-means 为 38 MB（27.17 PSNR），加 half-float 到 29 MB（27.10），质量只掉 0.07 dB；量化后位置占模型体积从 5% 升到 28%。
- 附录 Table 5/6：与并发预印本（EAGLES、Compact3D、Compressed3D、Compact3DGS、LightGaussian）同数据集对比，分"含/不含版权受限两景"两张表。附录正文：Compact3D 只报三数据集平均 54 MB，本文全量法 26 MB、Low 档 41 MB；对 Niedermayr（Compressed3D）的批评是"压缩率 impressive，但其 DEFLATE 约占 2x，不能直接映射到渲染期 VRAM"。

## 贡献列表（原文照抄）

1. "An efficient, resolution-aware primitive pruning approach, that reduces unnecessary points during optimization, leading to a total reduction of the primitive count by 60%."
2. "An adaptive adjustment method to choose the number of SH bands required for each 3DGS primitive, significantly reducing overall memory footprint."
3. "A codebook-based quantization method, together with a half-float representation for more efficient storage of the representation, resulting in further memory reduction."
