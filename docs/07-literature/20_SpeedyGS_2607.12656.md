# 《SpeedyGS: Content-Aware 3D Gaussian Splatting Compression via Two-Stage Optimization》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2607.12656 |
| 发表 | arXiv 预印本（v1，2026-07-14，eess.SP；ACM 模板排版，正文未标注会议接收信息） |
| 作者 | Junteng Zhang、Tong Chen、Zhan Ma（南京大学）；Yuxin Zhao、Yibo Shi、Jing Wang（华为） |
| 阅读材料 | 全文（.lit-cache/txt/2607.12656.txt，正文+补充材料 A–F 完整）。txt 有 30 处公式定界符被抽成控制字符，已按括号语义修复；式 (10)(11)(14)(18)(22)(23) 与表 1 已对照 PDF 抽页核验，一致 |
| 开源 | 论文未提及代码发布（全文无代码链接） |
| 与 DCCA-GS 的关系 | 压缩方向直接竞品：同样走"训练期内容自适应量化 + 剪枝 + 熵编码"路线，但底座是裸 3DGS/Mini-Splatting2（非锚点表示），且完全不涉及渐进码流与零侧信息——它把每场景学习式熵编码器参数随码流传输，与我们的零侧信息契约正面相对 |

> 一句话定位：把训练期压缩拆成"结构成形"和"统计编码"两段：前段用一个几乎零成本的码率代理（各通道动态范围除以量化步长取对数，乘以掩码保留率）同时驱动可学习的逐通道量化与 Gumbel 掩码剪枝；后段用 7.6K 参数的稀疏八叉树卷积网络编几何、单层 Mamba 在 Morton 序属性 token 上做组内自回归编属性，另给定长编码的 UltraFast 变体把解码压到约零延迟。结果：体积 6.51 MB 对 HAC++ 的 11.18 MB，训练 559 s 对 2119 s，解码 3.27 s 对 17.17 s，渲染 208 FPS（表 1）。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

两个。其一，一段式联合优化太慢。现有最好的逐场景压缩方法（HAC、ContextGS、CAT-3DGS、HAC++）都把"结构成形"（structural formation，指决定高斯数量与精度、得到紧凑表示的那段优化，脚注 1 说明它也叫 compaction）和"统计编码"（熵编码）放在同一个率失真目标里联合优化（第 1、2 节）。论文认为这让参数空间又大又耦合：N×59 个高斯参数、59 个量化变量、N×2 个掩码参数加编码器参数 θ 一起搜（式 4），"empirically leads to slower convergence"（第 1 节）。其二，同行只报渲染速度、不报解码负担："these methods tend to pursue high compression performance, while overlooking the decoding burden (which is unavoidable before rendering) and solely claiming 'rendering speed' advantages"（第 1 节），并点名领先方法 CAT-3DGS 解码一个约 50 万锚点的场景要超过 10 秒（第 1 节）；表 1 里 CAT-3DGS 的解码时间是 85.44 s。

### 支撑观察（直觉 / 数据 / 失败实验）

- 图 4（第 3.2 节）：bicycle 场景（λ=2.5e-5）从同一个 10k 步检查点分别用两段式和一段式继续训练：一段式约 1500 s、两段式约 580 s，省 61.3% 时间；代价是画质差 0.02 dB、体积多 1.5%。这是"解耦可行"的直接证据，也是全篇方法动机的实验支点。
- 图 5（第 3.3 节）：左图显示不同通道对量化位深（4–20 bit）的敏感度不同，最低位深处两条曲线的画质损失标注为 0.21 dB 和 0.26 dB（图注未写明对应哪两个通道）；右图显示剪枝比例的余量：原始 27.45 dB / 3.2M 参数，剪 53% 后 27.44 dB / 1.5M 参数（几乎无损），剪到约 94% 才掉到 26.46 dB / 0.2M——"该精的地方精、该删的删"需要逐通道量化加自适应剪枝，而不是统一精度、统一比例。
- 图 14（补充材料 D 节）：18-bit 坐标精度下，高斯点云的全局密度为 2e-6，与 LiDAR 点云的 1e-6 同量级——把"高斯几何近似稀疏点云"从直觉变成测量，支撑用点云压缩技术编几何。
- 搜索空间论证（式 4，第 3.2 节）：一段式要同时搜 N×59+59+N×2+|θ| 个变量，两段式把每段缩小成子问题。这是概念论证，没有单独消融，但图 4 和表 2 的"去掉两段式"一行共同支撑。
- 反向证据（诚实披露）：表 2 显示一段式联合优化的 BD-Rate 反而好 1.81%（−1.81%），代价是训练时间 2.56×。也就是说解耦不是免费的，论文自己的数据承认了这一点。

### 显然的路为什么没走

- 一段式联合优化（现状主流做法）：没选。表 2 给出代价（训练 1429 s 对 559 s），图 4 给出收敛曲线；他们接受 1.81% 的 BD-Rate 损失换 2.56× 训练加速。
- 结构成形阶段直接用真熵模型当码率项：没选，因为那会重新把结构优化和上下文熵编码耦合起来（第 3.3.1 节）。替代品是两层近似：先零均值拉普拉斯加高分辨率量化近似得到 log2(b/q)（式 10），再把尺度参数 b 换成更粗的通道动态范围（式 11）。补充材料表 3 对照了中间版本（按拉普拉斯尺度参数的代理）：BD-Rate 还能再好 0.43%，但训练 663 s 对 559 s，最终保留更粗的值域版——"代理粗一点、训练快一点"的取舍写成了数据。
- 几何用现成点云编码器：G-PCC 性能有限；通用学习式点云编码器太重（一篇 8M 参数、每场景超过 2 s；最轻的 RENO 也有 325K 参数，第 3.4.2 节）。他们按场景训一个 7.6K 参数的稀疏卷积网络。表 2 的"几何改用 G-PCC"一行：BD-Rate 只差 0.12%，但解码 10.06 s 对 3.27 s（几何解码 7.03 s 对 0.24 s，慢约 29×）——体积上 G-PCC 没输多少，输在解码速度，这正好服务他们的"解码延迟"主线。
- 锚点表示（Scaffold-GS 系）：没选，留在裸 3DGS（底座 Mini-Splatting2）。论文没有正面论证为什么不用锚点，能从结果反推理由：渲染速度（208 对 139/127 FPS，表 1）以及锚点表示转回 3DGS 再渲染的额外开销（约 5 ms，第 4.2.1 节）；代价也写明了："The proposed method is based on the improved work Mini-Splatting2 [12] of vanilla 3DGS, so the upper limit of fidelity will be constrained by the baseline"（图 18 说明）。

## 二学：机制（洞察怎么变成方法）

### 核心流程

输入是多视角图像，底座是 vanilla 3DGS（实际用 Mini-Splatting2 训练初始表示）。训练共 30,000 步，分三段（补充材料 C.1 节，图 10）：

1. **0–10k 步，初始表示**（第 4.1.1 节、补充材料 C.3 节）：按 Mini-Splatting2 原流程优化；其中 3k–8k 步插入 GaussianSpa 式稀疏化（不透明度收缩加可见性剔除），高斯数从 0.49M 降到 0.32M（图 13）。这一步是基础措施，论文不把它算作贡献。
2. **10k–25k 步，结构成形阶段**（第 3.3 节）：联合优化三类变量——高斯参数 G、每通道可学习量化步长 q_j（全场景 59 个通道各一个标量）、每高斯可学习保留分数 s_i。s_i 经 sigmoid 变成保留概率 p_i（式 13），Gumbel-Softmax 采软掩码再二值化成硬掩码（式 14），梯度经 masked rasterization 反传。总损失 = 渲染失真（L1 加 SSIM，式 39）+ λ×率代理（式 17）；率代理 = 掩码保留率 L_mask 乘以各通道量化代价之和 L_quant（式 16）。可学习掩码每 1,000 步执行一次真实剪枝（C.1 节）。λ 从 {2e-4, 1e-4, 5e-5, 2.5e-5} 取四个值，得到四个率失真点（第 4.1.1 节）。
3. **25k–30k 步，统计编码阶段**（第 3.4 节）：G、Q、M 冻结（式 6），只优化两个编码器。几何编码器：高斯点云体素化成多尺度结构，逐级下采样时把高分辨率占据图编成 0–255 的 8-bit 稀疏八叉树 token，用稀疏 CNN 预测 p(t_i|t_{i−1})（式 18；结构见图 11：Embedding(255,8)→SparseConv→SparseResNet×2→SparseConv(8,255,1)→Softmax）。属性编码器：在已解码几何位置上按 Morton 序把高斯排成一维 token 序列，分 G 组（每组 W 个高斯），属性线性嵌入拼傅里叶位置编码，过一个 Mamba 块加线性头做组内自回归（式 22；结构见图 12）。为省内存，统计编码阶段随机抽 5% 高斯训练（沿袭 HAC，第 4.1.1 节）；几何编码器优化器每两步更新一次（C.1 节）。总码率 R_coding = R_geo + R_attr（式 40）。

**编码期（写码流）**：几何走八叉树 token 的条件算术编码；体素化撞到同一位置的重复点（约占总点数 1%）只编一次，重复计数用 LZMA 压缩，开销不到 1KB（第 3.4.2 节）。属性按 Morton 序展平、分组后用训好的 Mamba 做条件算术编码。两个编码器的参数用 LZMA 无损压缩后写进码流（补充材料 E.3 节）。UltraFast 变体不走学习式编码：每通道按训练定下的精度定长存储，附加各通道精度元数据（总共只有 59 个通道：几何 3 个、属性 56 个，开销可忽略，第 3.4.1 节）。

**解码期**：先几何后属性。前一个八叉树 token 喂进稀疏 CNN 得到当前 token 的分布，逐 token 算术解码恢复占据；几何解完后按 Morton 序逐组解码属性 token（组内自回归；W=256 为主模型，Fast 版降到 W=32 提高并行度，第 4.1.1 节）。解码耗时的大头是算术编码不是网络：高码率点属性解码共 3.03 s，其中网络推理只占 0.24 s、算术编码占 2.79 s（表 5）。Fast 版把属性解码降到 1.90 s；UltraFast 定长解码总计 0.00025 s（表 5）。

### 关键公式（按原文抄，注明编号）

预备（第 3.1 节）：

- 式 (1)：G(x) = exp(−½(x−μ)ᵀΣ⁻¹(x−μ))，高斯基函数。
- 式 (2)：C = Σ_{i∈N} c_i α_i Π_{j=1}^{i−1}(1−α_j)，按深度排序的 alpha 混合。

问题分解（第 3.2 节）：

- 式 (3)：min_{G,Q,M,θ} [D(G,Q,M) + λR(G,Q,M;θ)]，一段式联合目标（作为对照）。
- 式 (4)：搜索空间 N×59 + 59 + N×2 + |θ|（N 为高斯数，59 为每高斯参数通道数，|θ| 为编码器参数量）。
- 式 (5)：min_{G,Q,M} [D_render(G,Q,M) + λR_formation(G,Q,M)]，结构成形阶段目标。
- 式 (6)：min_θ R_coding(G\*,Q\*,M\*;θ)，统计编码阶段目标（G、Q、M 冻结为第一阶段的解）。

率代理（第 3.3.1 节）：

- 式 (7)：L_coding = E[−log2 p_θ(y|c)]，真熵编码代价（条件拉普拉斯模型），本阶段不直接优化它。
- 式 (8)：p(ỹ;b) = 1/(2b)·exp[−|ỹ|/b]，去掉上下文的零均值拉普拉斯分布，ỹ = y − mean(y)。
- 式 (9)：E[|ỹ|/(b·ln2) + log2(2b)]，上式对应的期望负对数似然（码长）。
- 式 (10)：H(ŷ) ≈ h(ỹ) − log2 q = log2(2eb) − log2 q ≈ log2(b/q) + const，用高分辨率量化结论把"量化后熵"近似成信号尺度与步长之比（完整推导在补充材料 A.3 节，式 30–34）。
- 式 (11)：L_j = log2((max P_j − min P_j)/q_j)，最终代理：用第 j 通道动态范围替代尺度 b。
- 式 (12)：L_quant = Σ_j L_j，各通道求和。
- 式 (13)：p_i = σ(s_i)，第 i 个高斯的保留概率。
- 式 (14)：M_i = GumbelSoftmax(p_i)，软掩码采样，随后二值化为硬掩码 M̂_i ∈ {0,1}。
- 式 (15)：L_mask = (1/Ñ)·Σ_{i=1}^{N} M̂_i，保留率（Ñ 为成形前的高斯数）。
- 式 (16)：R_proxy = L_mask · L_quant，两因子相乘：数量乘精度。乘法耦合是他们的设计选择，表 2 显示它比加性掩码损失更好且无需手调 α。
- 式 (17)：L = D_render + λ·L_formation，成形阶段总损失（L_formation 即 R_proxy）。

统计编码（第 3.4.2 节）：

- 式 (18)：R_geo = −Σ_{i=1}^{k} log p(t_i|t_{i−1})，几何八叉树 token 的码率（k 为点云最大分辨率；txt 与 PDF 此处均写 log 未标底 2，原文如此）。
- 式 (19)：a_i = [c_i, α_i, s_i, r_i]，一个高斯的颜色、不透明度、缩放、旋转全部属性拼成一个 token。
- 式 (20)：φ(μ) = [μ; sin(2πFμ), cos(2πFμ)] ∈ R^{3+2×3L}，傅里叶位置编码，F = [f_0,…,f_{L−1}]，L=12。
- 式 (21)：PE(μ) = Linear(φ(μ)) ∈ R^d，d=32。
- 式 (22)：p(a_{1:W}) = Π_{i=1}^{W} p(a_i | Mamba(a_{<i}, PE(μ_{≤i})))，组内自回归：条件和输入都来自组内已解码属性与位置编码。
- 式 (23)：R_attr = −(1/G)·Σ_{i=1}^{G} log2 p(a_i)（求和上下标原文如此，G 为组数；按上下文应为对属性 token 的平均码长，原文公式以 PDF 为准）。

补充材料另给出成形总损失式 (39) 与编码码率式 (40) R_coding = R_geo + R_attr；式 (24)–(38) 是式 (7)–(12) 的完整推导（A.1–A.4 节）。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 两个相乘的代理量：精度维 = 各通道 log2(动态范围/量化步长)（式 11，量化后熵的对数近似）；数量维 = 掩码保留率（式 15）。合起来是熵编码开销的轻量近似（式 16） |
| 哪一端算得出（编码端/解码端/仅训练期） | 仅训练期：动态范围、可学习步长和保留分数都是训练态变量，解码端没有也不需要这个信号 |
| 有无侧信息 | 代理本身不进码流；但为它优化出的量化精度（UltraFast 的通道位宽元数据）和两个熵编码器参数要写进码流（E.3 节：bicycle 高码率点模型参数 0.21 MB，占 3%） |
| 训练期还是编码期 | 训练期（10k–25k 步随训练联合优化，25k 后冻结） |
| 和渲染质量的距离（直接测质量还是代理量） | 纯代理量：不测渲染质量，质量由同一损失里的渲染失真项承担，代理只管码率侧 |

与我们 DCCA-GS 的信号（解码端可重算的内容复杂度量化 × 训练期渲染敏感度监督，共享一个 8→32→3 小 MLP）相比：他们的信号更粗（59 个通道标量，场景内所有高斯共享）但更便宜（不用跑任何网络）；我们的信号细到分组、但要靠共享 MLP 且必须满足解码端可重算。他们的信号训练结束就失效，码流里要带走模型参数；我们的信号解码端能重算，码流里不带。

### 与码流组织相关的细节

- 渐进层：不涉及。每个 λ 训出一个完整文件；Fast/UltraFast 是编码工具变体（换编码器），不是同一个码流的前缀档。
- 文件布局（E.3 节，图 16）：几何 + 属性 + 模型参数三块。bicycle 高码率点 8.11 MB：几何 1.57（19%）、颜色 2.88（36%）、不透明度 0.39（5%）、缩放 1.25（15%）、旋转 1.81（22%）、模型参数 0.21 MB（3%）；低码率点 2.38 MB 中模型参数占 9%——场景越小，学习式编码器的固定开销占比越大。
- 解码端重算：几乎没有。属性编码以已解码几何位置为条件，这是全篇唯一"用解码端已解数据当下文"的地方（与我们的层间条件化同思路，但他们用在几何到属性之间）；量化步长和编码器参数都靠码流自带。学习式路径的逐通道步长 q_j 怎么写进码流，论文只对定长路径明说了精度元数据（第 3.4.1 节），学习式路径未明说。
- 可截断性：不支持。没有部分解码、画质分档的语义；UltraFast 解码接近零是靠放弃学习式统计编码换来的（BD-Rate 代价 +89.08%，表 2），不是靠截断。

## 三学：证明与包装（创新怎么立住）

### 实验设计（体积/码率怎么算、和谁比、怎么训）

- 数据：Mip-NeRF360 九景 + Tanks&Temples 两景（truck/train）+ DeepBlending 两景（drjohnson/playroom）共 13 景；场景选择、训练/测试划分、分辨率照官方 HAC 实现执行（补充材料 B 节）。
- 指标：SSIM/PSNR/LPIPS + 体积 MB + BD-Rate。注意 BD-Rate 的计算指标不统一：正文 4.2.1 节的 50.52%/71.14% 按 SSIM 算，表 2 消融按 PSNR 算（表 2 说明），补充材料 F.1 节的 29.86% 又按 PSNR 算。
- 体积怎么算：压缩文件总大小，含模型参数（E.3 节）。与我们 total_MB 字段求和的算法大体可比，但他们多一块每场景熵编码器参数。
- 和谁比：原版 3DGS、Scaffold-GS、九个 3DGS 系方法、七个 Scaffold-GS 系方法（含 HAC、HAC++、CAT-3DGS、ContextGS、PCGS、SizeGS、MoPGS 等），共 18 个对比方法；Scaffold-GS 锚点特征维取 50，与 HAC++/CAT-3DGS 一致（第 4.1.3 节）。
- 复杂度对比的计算方式要注意：表 1 里基线的训练/解码时间取各自原配置在 λ_o=0.002 单点的数值，SpeedyGS 自己取高码率点 λ=2.5e-5——所以表 1 的 HAC++ 体积 11.18 MB 与补充材料表 6 里 HAC++ 高码率 18.48 MB、低码率 8.34 MB 对不上，是计算方式不同，引用时不能混。
- 训练配置：30,000 步（含 5,000 步统计编码），λ∈{2e-4, 1e-4, 5e-5, 2.5e-5}，统计编码阶段随机抽 5% 高斯，单张消费级 GPU（型号未写明），L=12，W=256（Fast 版 32）。

### 主结果表抄录（注明表号）

表 1（Mip-NeRF360，主结果加复杂度，共 9 行，全抄）：

| 方法 | SSIM↑ | PSNR↑ | LPIPS↓ | 体积(MB)↓ | 两段式 | 训练(s)↓ | 解码(s)↓ | FPS↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3DGS (SIGGRAPH'23) | 0.813 | 27.49 | 0.222 | 744.70 | – | 1440 | – | 124 |
| Mini-Splatting2 (TPAMI'25) | 0.821 | 27.43 | 0.214 | 159.18 | – | 246 | – | 201 |
| Scaffold-GS (CVPR'24) | 0.812 | 27.81 | 0.228 | 227.95 | – | 1088 | – | 135 |
| HAC (ECCV'24) | 0.807 | 27.53 | 0.238 | 15.26 | ✗ | 1803 | 11.82 | 133 |
| CAT-3DGS (ICLR'25) | 0.809 | 27.77 | 0.241 | 12.35 | ✗ | 5349 | 85.44 | 127 |
| HAC++ (TPAMI'25) | 0.807 | 27.75 | 0.242 | 11.18 | ✗ | 2119 | 17.17 | 139 |
| Ours | 0.816 | 27.32 | 0.228 | 6.51 | ✓ | 559 | 3.27 | 208 |
| Ours-Fast | 0.816 | 27.32 | 0.228 | 7.03 | ✓ | 538 | 2.14 | 208 |
| Ours-UltraFast | 0.816 | 27.32 | 0.228 | 11.95 | ✓ | 313 | ≈0 | 208 |

（表 1 说明：基线复杂度取 λ_o=0.002 原配置单点，Ours 取 λ=2.5e-5 高码率点。另注意 Ours-Fast 体积在表 1 为 7.03 MB、在表 6/表 7 为 7.23 MB，原文未解释，两处照抄。）

表 6 节选（补充材料，Mip-NeRF360 列，高/低码率点；表 6 共 32 行含三个数据集，此处节选与我们最相关的行）：

| 方法 | SSIM↑ | PSNR↑ | 体积(MB)↓ |
| --- | --- | --- | --- |
| HAC 高/低码率 | 0.811 / 0.807 | 27.77 / 27.53 | 21.87 / 15.26 |
| ContextGS 高/低 | 0.811 / 0.808 | 27.75 / 27.62 | 18.41 / 12.68 |
| CAT-3DGS 高/低 | 0.809 / 0.730 | 27.77 / 25.82 | 12.35 / 1.72 |
| HAC++ 高/低 | 0.811 / 0.803 | 27.82 / 27.60 | 18.48 / 8.34 |
| PCGS (AAAI'26) 高/低 | 0.811 / 0.808 | 27.96 / 27.82 | 24.74 / 12.05 |
| Ours 高/低 | 0.815 / 0.777 | 27.32 / 26.47 | 6.51 / 2.01 |

（Ours-Fast 高/低 7.23/2.22 MB，Ours-UltraFast 高/低 11.95/3.75 MB，同表。正文与补充材料引用的关键数字：对原版 3DGS 的体积比，Mip-NeRF360 约 114×（4.2.1 节）、T&T 149×、DeepBlending 318×（F.1 节）；bicycle 单景 1461 MB 压到 8.11 MB（图 17）；对 CAT-3DGS 的 BD-Rate 按 SSIM 50.52%（Fast 版 44.81%），DeepBlending 上按 PSNR 29.86%（F.1 节）；对 HAC 按 SSIM 71.14%（4.2.1 节）；压缩比高于 RDO-Gaussian 3.60×、CompGS 2.53×（F.1 节）；对 CAT-3DGS 训练加速 5349 s 对 559 s，约 9.6×（表 1，论文引言称 9×）。）

值得注意的读法：同样在高码率点上，SpeedyGS 的 PSNR（27.32）低于 HAC++（27.82）和 CAT-3DGS（27.77），体积约为前者的三分之一；它的 BD-Rate 优势来自整条曲线的形状——它的低码率点是 26.47 dB / 2.01 MB，而 CAT-3DGS 的低码率点 25.82 dB / 1.72 MB 在低码率端塌得厉害（表 6）。

### 消融设计（注明表号）

表 2（Mip-NeRF360，BD-Rate 按 PSNR，共 12 行，全抄）：

| 消融项 | BD-Rate↓ | 训练(s)↓ | 解码(s)↓ | FPS↑ |
| --- | --- | --- | --- | --- |
| SpeedyGS（基准） | 0.00% | 559 | 3.27 | 208 |
| 优化策略：去掉两段式 | −1.81% | 1429 | 3.28 | 206 |
| 率代理：无掩码 | +79.03% | 618 | 2.92 | 181 |
| 率代理：加性掩码 α=0.0001 | +18.44% | 583 | 3.36 | 193 |
| 率代理：加性掩码 α=0.0005 | +4.94% | 563 | 3.34 | 206 |
| 率代理：加性掩码 α=0.001 | +12.64% | 541 | 3.48 | 214 |
| 率代理：加性掩码 α=0.002 | +5.03% | 531 | 3.53 | 218 |
| 统计编码：无统计编码（改定长编码） | +89.08% | 313 | ≈0 | 208 |
| 统计编码：无属性编码 | +67.60% | 432 | 0.24 | 208 |
| 统计编码：无几何编码 | +28.10% | 440 | 3.03 | 208 |
| 统计编码：几何改用 G-PCC | +0.12% | 440 | 10.06 | 208 |
| 训练加速：无优化器调度 | +1.10% | 1143 | 3.24 | 209 |

值得学走的对照设计：

- 乘法对加性的掩码消融，并扫 α 的四个取值（表 2 四行加"无掩码"行）：加性掩码的性能随 α 非单调（+4.94% 到 +18.44% 来回跳），乘法版稳且免手调——证明"乘法耦合省掉一个超参"不是空话。
- 代理替换对照（补充材料表 3，共 2 行）：值域代理 0.00%/559 s 对拉普拉斯尺度代理 −0.43%/663 s，保留更粗代理的理由写成训练时间。这种"把自家方案换成更强替身、按代价取舍"的对照写法少见。
- 解码时间分解（表 5）：几何/属性 × 网络推理/算术编码四个象限，三个变体 × 两个 λ 逐格列出。直接指出瓶颈在算术编码（2.79 s）不在网络（0.24 s），为"网络够轻"提供数字。
- 量化方案对照（图 8a）：统一 20-bit 几何加 10-bit 属性的定点量化把底座 159.18 MB 压到 49.46 MB（图注标 3.22×），自适应量化再压到 27.40 MB（对底座共 5.81×）。位置编码对照（图 8b）：无位置编码 6.68 MB、线性 6.64 MB、傅里叶 6.51 MB，解码复杂度 100%/100.43%/102.18%。
- 稀疏化前置消融（补充材料表 4）：加 GaussianSpa 后 Ours 参数 0.65M 对 Mini-Splatting2 的 0.67M，画质 27.44 对 27.43 dB（Mip-NeRF360）——证明基础措施不伤画质。

### 贡献列表（原文照抄 + 逐条标注）

1. "Motivated by the fundamental difference in the objectives of structural formation and statistical coding, we decouple the two stages, enabling stage-wise optimization with explicit content awareness. In particular, the formation stage proposes a lightweight rate proxy without actual entropy coding to optimize adaptive quantization and pruning for R-D performance, while the next statistical coding stage incorporates the complexity trade-off into the design of context-adaptive codes to reduce statistical redundancy." —— [机制] 两段式分解 + 轻量码率代理 + 复杂度可控编码，全篇核心主张。
2. "Upon the two-stage optimization, SpeedyGS also achieves leading performance. Compared to vanilla 3DGS at comparable perceptual quality, it achieves a compression ratio of approximately 160×, a 2× improvement in optimization efficiency, rendering speed, and low decoding latency. It also offers a better overall trade-off than leading representative methods such as CAT-3DGS and HAC++. Additionally, SpeedyGS's two-stage optimization substantially reduces the training search space, achieving a notable 9× training speedup over CAT-3DGS as in Table 1." —— [凑数] 结果复述当贡献条目（160× 是摘要的说法；正文实测 Mip-NeRF360 约 114×、T&T 149×、DeepBlending 318×，补充材料 F.1 节，摘要与正文数字不一致）。
3. "SpeedyGS offers the flexibility to modify or replace individual stages as required. For instance, by parallelizing statistical attribute coding, SpeedyGS-Fast achieves a decoding latency of only 2.14 s (see Table 1), while still outperforming CAT-3DGS and HAC++. SpeedyGS-UltraFast employs fixed-length codes, resulting in almost zero decoding latency while maintaining comparable compression performance to CAT-3DGS." —— [工程] 变体灵活性：Fast（组大小降到 32 提高并行）与 UltraFast（定长编码）两个部署形态。

### 讲故事方式

主线一句话："压缩分两段训，解码才算得快。"证据链是三张表各扛一个论点：图 4 证明两段式省 61.3% 训练时间只损 0.02 dB；表 1 把体积/训练/解码/渲染四列并排，让 HAC++ 和 CAT-3DGS 在解码列上难看（17.17 s、85.44 s 对 3.27 s）；表 5 的四象限分解坐实"重的是算术编码不是网络"，支撑 Fast/UltraFast 变体的可行性。图 1 的云端到设备工作流图用具体计时把动机翻译成产品语言（414 MB 原始模型在 10 MB/s 带宽下要下 41.4 s，压缩后 0.67–0.96 s）。诚实的一笔：图 18 说明里承认画质上限被 Mini-Splatting2 底座限制。

## 四学：原文关键段翻译

### 方法节核心段（选 2 段）

> **原文（第 3.3.1 节 Rate Proxy）**："Existing learned compression methods [6, 7, 43] typically define the rate term as the expected code length induced by an entropy model, which also corresponds to the statistical coding stage in our framework: L_coding = E[−log2 p_θ(y|c)], (7) where p_θ(y|c) is modeled by a conditional Laplace distribution. Directly optimizing this objective during formation would require coupling structure optimization with context-dependent entropy coding, which substantially increases the optimization complexity. We therefore seek a proxy of L_coding for efficient optimization during structural formation. As a first simplification, we remove the context dependency and model the centered variable ˜y = y − mean(y) with a zero-mean Laplace distribution: p(˜y; b) = 1/(2b) exp[−|˜y|/b], (8) whose corresponding expected negative log-likelihood is E[|˜y|/(b ln 2) + log2(2b)], (9) where b is the scale parameter in the Laplace distribution. This model avoids expensive context modeling while preserving the dependence of entropy coding cost on the signal scale. ... By the standard high-resolution quantization result [15], with the derivation in the supplementary material, H(ŷ) ≈ h(˜y) − log2 q = log2(2eb) − log2 q ≈ log2(b/q) + const, (10) which indicates that, under the Laplace assumption, the entropy coding cost after quantization is mainly governed by the ratio between the signal scale b and the quantization step size q."
>
> **译文**：现有的学习式压缩方法 [6,7,43] 通常把码率项定义为由熵模型导出的期望码长，这也对应我们框架里的统计编码阶段：L_coding = E[−log2 p_θ(y|c)]（式 7），其中 p_θ(y|c) 用条件拉普拉斯分布建模。若在结构成形阶段直接优化这个目标，就得把结构优化和依赖上下文的熵编码耦合在一起，会大幅增加优化复杂度。因此我们为 L_coding 找一个代理量，让结构成形阶段能高效优化。第一步简化是去掉上下文依赖，把中心化变量 ỹ = y − mean(y) 用零均值拉普拉斯分布建模：p(ỹ;b) = 1/(2b)·exp[−|ỹ|/b]（式 8），其对应的期望负对数似然是 E[|ỹ|/(b·ln2) + log2(2b)]（式 9），b 是拉普拉斯分布的尺度参数。这个模型避开了昂贵的上下文建模，同时保住熵编码开销对信号尺度的依赖。……按标准的高分辨率量化结论 [15]（推导见补充材料），H(ŷ) ≈ h(ỹ) − log2 q = log2(2eb) − log2 q ≈ log2(b/q) + 常数（式 10）。这表明，在拉普拉斯假设下，量化之后的熵编码开销主要由信号尺度 b 与量化步长 q 之比决定。

> **原文（第 3.4.2 节 Autoregressive Attribute Coding）**："Inspired by the success of image autoregressive models [34] and recent transformer-based tokenizers [10, 42] that reshape high-dimensional grids into 1D token sequences, we adopt a token-based paradigm for Gaussian attribute compression. Upon the decoded geometry positions, we use Morton ordering to sort the Gaussian functions, converting the 3D Gaussians into a 1D token sequence, as shown in Fig. 3. Specifically, each token corresponds to one Gaussian and is defined by its full attribute vector: a_i = [c_i, α_i, s_i, r_i], (19) where c_i, α_i, s_i, and r_i denote the color, opacity, scale, and rotation attributes of the i-th Gaussian, respectively. In this way, the attributes of each Gaussian are modeled jointly as a single token, rather than as attribute channels. This spatially coherent ordering preserves locality, ensuring that neighboring Gaussians remain adjacent in the sequence and that local correlations among their attributes are retained, which facilitates effective autoregressive modeling with a causal State Space Model such as Mamba. To further improve the parallel decoding capability, we divide the 1D sequence into G groups, each containing W Gaussians."
>
> **译文**：受图像自回归模型 [34] 和近年把高维网格重排成一维 token 序列的 transformer 分词器 [10,42] 的启发，我们对高斯属性压缩采用基于 token 的做法。在已解码的几何位置之上，用 Morton 排序给高斯排序，把三维高斯转成一维 token 序列（图 3）。具体地，每个 token 对应一个高斯，由它的完整属性向量定义：a_i = [c_i, α_i, s_i, r_i]（式 19），其中 c_i、α_i、s_i、r_i 分别是第 i 个高斯的颜色、不透明度、缩放和旋转属性。这样，每个高斯的全部属性作为一个 token 联合建模，而不是按属性通道分别建模。这种空间连贯的排序保住了局部性：相邻高斯在序列里仍然相邻，它们属性之间的局部相关得以保留，方便用 Mamba 这类因果状态空间模型做有效的自回归建模。为了进一步提高并行解码能力，把一维序列分成 G 组，每组含 W 个高斯。

### 主结果表述段（选 1 段）

> **原文（第 4.2.1 节）**："Figure 6 and Table 1 demonstrate the advanced performance of SpeedyGS on the Mip-NeRF360. Compared with vanilla 3DGS, our full model reduces the scene size from 744.70 MB to 6.51 MB, corresponding to a compression ratio of about 114×, while also increasing rendering speed from 124 to 208 FPS. Our lightweight variants still achieve high compression ratios, with Ours-Fast and Ours-UltraFast reaching about 106× and 62×, respectively. Similar trends are observed on Tanks&Temples and DeepBlending, as reported in the supplementary material. Compared with previous compression methods, SpeedyGS achieves the smallest storage size among all compared methods in Table 1, demonstrating superior compression efficiency. In particular, on Mip-NeRF360, SpeedyGS achieves a 50.52% BD-Rate gain over CAT-3DGS and a 71.14% BD-Rate gain over HAC on the SSIM metric."
>
> **译文**：图 6 和表 1 展示了 SpeedyGS 在 Mip-NeRF360 上的先进性能。与原版 3DGS 相比，我们的完整模型把场景体积从 744.70 MB 降到 6.51 MB，对应约 114× 的压缩比，同时把渲染速度从 124 FPS 提到 208 FPS。轻量变体仍保持高压缩比：Ours-Fast 与 Ours-UltraFast 分别约 106× 和 62×。Tanks&Temples 和 DeepBlending 上趋势类似（见补充材料）。与已有压缩方法相比，SpeedyGS 是表 1 中所有对比方法里存储体积最小的，展现了更高的压缩效率。特别地，在 Mip-NeRF360 上按 SSIM 计算，SpeedyGS 相对 CAT-3DGS 取得 50.52% 的 BD-Rate 增益，相对 HAC 取得 71.14% 的 BD-Rate 增益。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | 两段式分解作为框架主张有辨识度（此前主流是一段式联合优化），率代理的推导（拉普拉斯→高分辨率量化→值域/步长比，式 8–11）干净；但各组件都有先例：可学习量化步长、Gumbel 掩码剪枝（MaskGaussian 一类）、Morton 序自回归属性编码（CAT-3DGS 等通道自回归的变体）、点云式几何编码；组合方式与复杂度分档（Fast/UltraFast）是主要新东西 |
| 证据强度 | 中 | 三个标准数据集 13 景、18 个对比方法、逐景表（表 7–9）、解码四象限分解（表 5）、每个设计点都有消融（含替换自家代理的表 3 和 G-PCC 替换）；扣分：无开源（文中无代码链接）、基线复杂度只取单点原配置数值（表 1 说明）、高码率点 PSNR 低于 HAC++/CAT-3DGS（表 1，体积优势与画质优势不在同一点上）、摘要 160× 与补充材料 318×（DeepBlending，F.1 节）说法不一致 |
| 对期刊版的威胁度 | 中 | 撞"训练期内容自适应量化"这条主张的措辞（它是通道级 59 个标量、精度元数据写码流；我们是逐锚点内容复杂度、解码端重算、零侧信息）；它还把我们的底座 HAC++ 在体积/训练/解码/渲染四列上全面压过（表 1），投稿场景里审稿人可能要求对比。没撞的部分：渐进单文件码流、可截断性、零侧信息契约——该论文完全不涉及（每个 λ 一个独立文件，熵编码器参数还要随码流传输）；表示也不同（裸 3DGS 对锚点） |

## 对 DCCA-GS 的可借鉴点

1. 解码复杂度的报告格式（学表 5 → 用到我们论文的实验节）：四象限（几何/属性 × 网络推理/算术编码）分解，逐变体、逐 λ 列出。我们的零侧信息轻量解码主张正需要这种表来量化"开销在哪"（他们的结论是开销在算术编码不在网络；我们的二值分解区间编码是纯查表加算术编码，同样的分解能直接支撑解码快的主张）。我们的渐进码流还能多报一列他们给不出的——截断到第 k 档时的解码时间，把可截断性从功能描述变成数字。
2. 值域/步长对数当码率代理（学式 11 → 用于我们训练期需要轻量码率估计的环节）：它证明 log2(动态范围/步长) 这种一次遍历就能算的量，配合乘法掩码项能在训练期替代真熵模型，训练时间还省（补充材料表 3 给出"代理强度阶梯"：真熵模型 > 拉普拉斯尺度代理 > 值域代理，代价递减）。若我们的 B3 阶梯对齐量化训练（量化步长与阶梯取值保持一致的那套训练方式）或分字段粗化取值评估需要码率信号，可以按同一阶梯跑同预算对照。这属于训练期信号，不违反零侧信息约束。
3. 部署形态透明定价的写法（学 Fast/UltraFast 的呈现 → 用于我们论文写作）：同一框架多个变体，每个变体的 BD-Rate 代价明码标价（UltraFast +89.08%，表 2）。我们"任一前缀即一个画质档"的故事可以借这个格式：给每个档报画质加解码时间，让渐进的收益可测量，也顺带回应"解码近零"这个卖点——我们的前缀截断天然覆盖它。
4. 不需要跟进、但要备好回答的问题：它在裸 3DGS 属性上用学习式熵编码拿到大增益（表 2：改定长编码 BD-Rate 恶化 +89.08%），表面上与我们"学习式概率熵模型生产 A/B 零增益"的负结果冲突。差异在表示和参照系：他们的参照是定长编码、表示是没有上下文模型的裸属性；我们的锚点特征已有哈希网格上下文和条件表在压。期刊版若被审稿人拿这篇追问"为什么不用学习式熵模型"，用表示差异加我们的 A/B 证据回答，并指出他们为此付出的代价：每场景 0.21 MB 模型参数随码流传输（E.3 节），低码率点占总体积 9%（图 16）——这正是零侧信息契约要省掉的东西。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 量化怎么定 | 每通道一个可学习步长 q_j（59 个标量，全场景共享），与掩码剪枝在率代理目标下联合训练（式 11/16/17）；UltraFast 把结果固化成通道位宽元数据写入码流（3.4.1 节） | 解码端可重算的内容复杂度量化 × 训练期渲染敏感度监督，共享一个 8→32→3 小 MLP 按分组定步长，零侧信息 | 都属于"训练期内容自适应量化"，但自适应的分辨单位不同：它到通道（59 个标量），我们到分组/锚点；它把精度元数据写码流，我们解码端重算。它证明通道级标量已能拿到大头收益（图 8a：统一 20/10-bit 定点 49.46 MB 对自适应 27.40 MB），我们要能说明分组级带来的增量在哪 |
| 熵编码上下文 | 学习式：几何 = 稀疏八叉树 token 逐 token 自回归（7.6K 参数稀疏 CNN，式 18）；属性 = Morton 序 + 傅里叶位置编码 + 单层 Mamba 组内自回归（式 22）；编码器参数 LZMA 压缩后随码流传（0.21 MB，E.3 节） | 分组频率统计表（贡献组×粗值幅度桶复合条件表）+ 层间条件化（细层用已解码粗层当上下文）+ 二值分解区间编码，全部解码端可重算，零学习模型 | 它的上下文模型表达力强，但要背模型参数、解码要跑网络（高码率点属性解码 3.03 s，表 5）；我们的表轻、解码快、零侧信息。注意两边参照系不同：它对照的是定长编码，我们的负结果是对照生产条件表，不能直接比大小 |
| 渐进/可截断 | 不涉及：每个 λ 训出独立完整文件，Fast/UltraFast 是编码工具变体不是码流前缀，没有部分解码语义 | 单文件渐进分层码流：任意画质档都是文件连续字节前缀；嵌套质量阶梯（基层传全锚点粗量化值，增强层逐层补精度）、层级序文件布局、bit-exact 验证 | 该论文不涉及渐进码流，这是与我们错开最大的地方；反过来它的 UltraFast 证明"解码近零"这个卖点有市场，我们的前缀截断天然覆盖这一形态，论文里值得点破 |
| 侧信息 | 有：每场景两个熵编码器参数（LZMA，bicycle 高码率点 0.21 MB、占 3%，低码率点占 9%，图 16）+ 通道精度元数据 + 重复点计数（约 1% 的点，不到 1KB） | 零侧信息契约：解码端能重算的才可以用，否则写码流；量化步长/编码上下文全部解码端从已解码数据重算 | 正面相对的设计取舍：它用 3–9% 的体积换学习式编码的表达力，我们省掉这块也放弃学习式上下文。0.21 MB 是每场景固定成本，场景越小占比越大（图 16：低码率点占 9%），把它放到别的数据上评估时要先算这笔固定开销 |
| 剪枝 | 可学习分数 + Gumbel-Softmax 软掩码 + masked rasterization 反传，掩码保留率乘进率代理（式 13–16），每 1,000 步执行一次真剪枝 | GaussianSpa 式训练期 ADMM 剪枝作为基础措施（不作创新点），另有 Mini-Splatting depth-reinit 增密 | 它同样把 GaussianSpa 用作前置稀疏化（补充材料 C.3 节，0.49M→0.32M），与我们的基础措施同源；它的掩码项与量化项乘法耦合、免手调 α（表 2 消融），可作我们剪枝与码率目标耦合方式的参照 |
