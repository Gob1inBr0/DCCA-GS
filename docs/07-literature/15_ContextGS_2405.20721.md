# 《ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2405.20721 |
| 发表 | NeurIPS 2024（任务指定；txt 为 arXiv v1，2024-05-31，页眉标注 "Preprint. Under review."） |
| 阅读材料 | 全文（.lit-cache/txt/2405.20721.txt） |
| 与 DCCA-GS 的关系 | 同类竞品 + 上下文路线对照：它是"锚点级自回归上下文"的代表，与我们底座 HAC++ 的"哈希上下文 + 通道自回归"是两条路线，直接关系我们零侧信息框架的上下文选择 |

> 一句话定位：第一个把图像压缩的上下文模型搬到**锚点层级**的方法——把锚点按体素尺寸分成 3 层，粗层先编码、已解码的粗层锚点当细层锚点的熵编码上下文，再配一个量化的每锚点超先验特征兜底最粗层；对 Scaffold-GS 底座平均省 15×、对 3DGS 省 100×，质量持平或更高。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

第 1 节 + 第 2.2 节，三层递进：

1. 通用压缩方法"逐点独立编码"："Existing methods primarily compress neural Gaussians individually and independently, i.e., coding all the neural Gaussians at the same time, with little design for their interactions and spatial dependence."
2. Scaffold-GS 消掉了体素内冗余，但锚点之间的冗余没动："the similarity among anchors remains high in certain areas, indicating that spatial redundancy still exists"（Fig. 1(c)）。
3. **对哈希上下文路线（HAC）的批评就在这里**：第 2.2 节原话——"successive work [5] demonstrates its effectiveness by further introducing hash-feature as a prior for entropy coding. However, [5] codes all the anchors at the same time, and its spatial redundancy can be further reduced."——即 HAC 的哈希先验虽然各自锚点估得准，但所有锚点是**一次性并行**编码的，锚点与锚点之间的空间依赖完全没被熵模型利用。这就是"空间冗余没挖够"的原话出处。

### 支撑观察（直觉 / 数据 / 失败实验）

- Fig. 1(c)（第 1 节）：Scaffold-GS 训练出的 0 层锚点与其 1、2 层上下文锚点之间的余弦相似度——相似度在不少区域仍然很高，锚点间冗余存在的直接证据。
- Fig. 1(d)（第 1 节）：在"基于 Scaffold-GS 的熵编码强基线"上测锚点级上下文的 bit 节省，配 Fig. 1 的 10× 压缩展示（Scaffold-GS 248MB/24.50dB 对 ContextGS 21.81MB/25.08dB，Figure 1）。
- Fig. 6（第 5.3 节）：层级比例 τ 从 0.2 扫到 0.8（0.8 端点已对照 PDF 第 8 页图轴核实），PSNR 基本不动、体积随 τ 变大——支撑"分层方式对超参不敏感、τ=0.2 全局通用"。

### 显然的路为什么没走

- 显然路 A：把图像压缩的逐像素/逐特征自回归直接搬过来。它自己给出否定理由（第 5.3 节）：图像自回归"predicts pixels/latent features one by one, introducing a loop of at least thousands of operations"，太慢；改成锚点级分组自回归后循环只有 3 轮（每层一轮），层内并行。
- 显然路 B：LOD/Octree 式分层（Octree-GS [28]）。第 5.3 节指出已有 LOD 方法"the anchors from different levels are stored separately"，各层重复存储；它用 anchor forward（粗层锚点直接复用进最细层渲染，式 6 去重）避免重复，Table 3 证实：不复用 15.54MB、不分层 14.73MB、复用后 13.80MB（BungeeNeRF）。
- 显然路 C：顺手把锚点位置也熵编码。第 5.3 节专门讨论并放弃："the precision of the anchor position is essential to the performance of the model and an adaptive quantization strategy leads to serious performance degradation"；固定步长可行但"greatly decreases the coding speed"，且位置只占码流一小部分（Table 4），索性不编。

## 二学：机制（洞察怎么变成方法）

### 核心流程

底座是 Scaffold-GS（锚点 f∈R^32/50、位置 x、缩放 l、K 个 offset O；式 2-3 由锚点特征预测神经高斯属性）。三层结构（图 3，K=3）：

**训练期**：
1. 锚点分级（第 4.1 节，式 4-6）：初始化后一次性完成。自底向上：V̂₀=全部锚点；对 level k 用体素尺寸 ϵ_k=κ_k·ϵ 把 k−1 层锚点的位置量化（式 5 的 x̂=⌊x/ϵ_k⌋），同一量化格里按下标最小规则 `j = min{i : x̂_i = x̂}` 选一个代表升入 level k（式 5 的 M 映射）；再逐层去重 V₂=V̂₂, V₁=V̂₁\V̂₂, V₀=V̂₀\V̂₁（式 6）——每个锚点只在最细的所属层存一次，粗层锚点直接被更细层复用（anchor forward）。κ_k 不手工调，而是设相邻层锚点数比 |V_{i+1}|/|V_i|≈τ=0.2，二分搜索出 {κ_k}（第 4.1 节）。
2. 熵模型（第 4.2 节）：每个锚点的属性（特征 f、缩放 l、offset O）都用"量化高斯 × 均匀噪声"的因子化先验建模（式 7），分布参数 μ、σ、Δ 由一个按层独立的 MLP 从先验向量 ψ 算出；先验 ψ 的构成是全文关键（式 8）：
   - 最粗层 k=K−1（没有更粗的已解码锚点）：ψ = [x̂]（只有自己的量化位置）；
   - 其余层 k<K−1：ψ = [f_{j}^{k+1}; l_{j}^{k+1}; x̂_i^k]——**粗层相邻锚点已解码的特征和缩放**，拼上当前锚点的量化位置。"相邻"由分级的体素包含关系直接给出（第 4.1 节：adjacent = the anchors in the same voxel of level k−1 with v），所以映射可追溯、解码端可复现。
3. 每锚点再配一个可学习超先验向量 z∈R^{50//hc}（hc=4，即约 12 维），量化成整数 ẑ 后走全因子化密度模型（式 9），**ẑ 要进码流**；最终先验 ψ̂=[ẑ; ψ]。
4. 训练目标（式 10-11）：L = L_scaffold + λe·L_entropy + λm·L_m，L_entropy = 超先验的码率 + 各层、各锚点、各属性（f、l、O）在 ψ̂ 条件下的码率之和。λm=5e-4（沿 Compact3DGS 的 offset 掩码损失）。

**编码期**：严格按 level K−1 → 0 的顺序编码；每层内各锚点并行（上下文只来自粗层）；每锚点先编 ẑ，再按式 7 的分布算术编码 f、l、O。**锚点位置 x 不熵编码**，原样存（第 5.3 节）。

**解码期**：同序分层解码，每层解码时上下文 ψ 由已解码的粗层锚点现算（式 8 的量在解码端全部可得）；渲染管线与 Scaffold-GS 完全一致，"without introducing overhead"（图 3 说明）。

### 关键公式（按原文抄，注明编号）

式 (7)（锚点属性的分布）：

> p_{f^k}(f_i^k | ψ_i^k) = ( ½·(N(μ_i^k, σ_i^k) ∗ U(−Δ_i^k/2, Δ_i^k/2)) )(f_i^k)，  μ_i^k, σ_i^k, Δ_i^k = F_{f^k}(ψ_i^k)

F_{f^k} 是属于 level k 的 MLP；μ/σ 是逐通道高斯参数，Δ 是量化区间宽度（均匀噪声卷积近似量化）。式 (8)（先验构成，上面的核心流程已引）。式 (9)（超先验分布，txt 有损）：p_z̃|Θ(z̃_i|Θ) = Π_j p_{z_j}|Θ(·) ∗ U(−½,½)(z̃_ij)，全因子化，逐通道独立。式 (10)：

> L = L_scaffold + λe·L_entropy + λm·L_m

式 (11)（熵损失，txt 数学结构有损，按可读部分转述并以 PDF 为准）：第一项 E[−log₂ p_z̃|Θ(z̃_i|Θ)] 是编超先验的码率；第二项是分层求和的 −log₂ Π_{f∈{f^k, l^k, O^k}} p_f(f_i | ψ̂_i^k)——即**三个属性 f、l、O 共用同一套条件分布族**逐属性编码。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | 锚点间条件依赖：已解码粗层锚点的特征/缩放 + 当前锚点量化位置 + 量化超先验 → 每锚点各属性的分布参数 |
| 哪一端算得出 | 编码端与解码端都能算——解码端按同序解码，上下文全部来自已解码属性和自身位置，可重算 |
| 有无侧信息 | 超先验 ẑ 本身进码流（自己付码字，Table 4 显示总量不大）；分级结构不进码流，由位置 + 固定常数（K=3、τ=0.2）+ 体素尺寸推出 |
| 训练期还是编码期 | 熵模型训练期学，编码/解码期实际执行（训练时 ψ 用当前锚点值模拟解码顺序） |
| 和渲染质量的距离 | 代理量：纯码率侧信号，靠 λe 与渲染损失联合训练间接服务质量；质量变化作为副产品被报告（体积省的同时 PSNR 反升，Table 1） |

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 数据集：Mip-NeRF 360（9 个场景全用，跟随 HAC 口径）、Tanks&Temples、DeepBlending、BungeeNeRF；指标 PSNR/SSIM/LPIPS + 存储 MB。
- 配置：基于 Scaffold-GS；K=3 层、τ=0.2、hc=4；锚点特征维度 50（跟随 HAC）；λm=5e-4；30000 迭代（与 Scaffold-GS、HAC 一致）；锚点生长超参同 Scaffold-GS。两个码率点由 λe 调出（low-rate λe=0.004；high-rate 0.0005/0.001，Table 6-9），λe 归一化方式与 HAC 相同。
- 对比：3DGS、Scaffold-GS、EAGLES、LightGaussian、Compact3DGS、Compressed3D、Morgenstern et al.、Navaneet et al.、HAC。

### 主结果（注明表号）

- **对 HAC（Table 1，每格为 PSNR/SSIM/LPIPS/MB）**：Mip-NeRF 360 上 HAC 27.53/0.807/0.238/15.26，Ours low-rate 27.62/0.808/0.237/12.68、high-rate 27.75/0.811/0.231/18.41；Tanks&Temples HAC 24.04/0.846/0.187/8.10，Ours 24.20/…/7.05 与 24.29/…/11.80；DeepBlending HAC 29.98/0.902/0.269/4.35，Ours 30.11/…/3.45 与 30.39/…/6.60；**BungeeNeRF 差距最大：HAC 26.48/0.845/0.250/18.49，Ours low-rate 26.90/0.866/0.222/14.00**——体积更小（14.00 对 18.49MB）还高 0.42dB。RD 曲线整体压过 HAC（Figure 4）。
- 对 Scaffold-GS（Table 1）：BungeeNeRF 26.90dB/14.0MB 对 26.62dB/183.0MB；全文口径"15× 对 Scaffold-GS、100× 对 3DGS"（贡献列表、摘要）。
- 视觉对比（Figure 5）：T&T train 场景 HAC 22.15dB/7.0MB 对 Ours 22.42dB/6.4MB；Bungee 场景 HAC 27.60dB/17.3MB 对 Ours 28.08dB/13.2MB。
- 存储分解（Table 4，BungeeNeRF rome 场景）：Scaffold-GS 186.7MB/26.25dB，ContextGS 14.06MB/26.38dB（不编位置）；若开位置编码 13.62MB/26.38dB，但编码/解码时间 41.33s/51.58s 对 20.40s/17.85s——位置编码省 0.44MB、时间翻倍，故默认关闭。
- **无 HAC++ 对比（论文未报告）**：HAC++（arXiv 2501.12255）晚于本文 v1。

### 消融设计（注明表号）

- **逐组件 2×2 开关（Table 2，BungeeNeRF）**：基线"Scaffold-GS + 熵编码 + 掩码损失"（Ours w/o HP w/o CM）18.67MB/26.93dB；只加锚点级上下文 CM 15.03MB（−19.5%）；只加超先验 HP 15.41MB（−17.5%）；全开 14.00MB/26.90dB（对基线 −25.0%、对 Scaffold-GS 183.0MB 为 −92.3%）。注意正文第 5.3 节自述"reducing the file size by 21% and 10.17%, respectively… 26.1% and 92.5%"，与按 Table 2 直接计算的数字略有出入，以表为准。这个消融的价值：把"上下文模型"和"超先验"两个贡献拆干净，且基线本身已是强熵编码基线（和我们"基础措施不算创新"的口径同思路）。
- **anchor forward 消融（Table 3）**：不分层 14.73MB / 只分层不复用 15.54MB / 完整 13.80MB——不复用反而比不分层更差，说明"分层 + 复用"必须绑在一起，这个对照设计值得学（专门验证机制内部耦合）。
- **超参稳健性（Figure 6）**：τ∈{0.2,…,0.8}，PSNR 稳定、体积单调变化，支撑固定 τ=0.2 不逐场景调参。
- **位置编码开关（Table 5）**：w/ vs w/o 锚点位置编码全四个数据集对照——低码率档省体积（Mip360 11.32 对 12.68MB），高码率档反而更大（21.07 对 18.41MB）。

### 贡献列表（原文照抄 + 逐条标注）

1. "We propose the first context model for 3DGS at the anchor level. By predicting the properties of anchors that are not coded yet given already coded ones, we greatly eliminate the spatial redundancy among channels." —— [机制] 首个锚点级上下文模型（末尾 "among channels" 疑为 "among anchors" 的笔误，照抄 txt）。
2. "We propose a unified compressing framework with the factorized prior, enabling end-to-end entropy coding of anchor features. Besides, a strategy for anchor layering is proposed, which allows already decoded anchors to quickly locate adjacent anchors that are to be decoded. Meanwhile, the proposed method avoids redundant coding of anchors by the proposed anchor forward." —— [机制+工程] 因子化先验统一编码 f/l/O + 可追溯的分层策略 + 去重的 anchor forward。
3. "The experimental results on real-world datasets demonstrate the effectiveness of the proposed method compared with SOTA and concurrent works. On average across all datasets, our model achieves a compression ratio of 15× compared to the Scaffold-GS model we used as the backbone and 100× compared to the standard 3DGS model, while maintaining comparable or even enhanced fidelity." —— [实证] 四数据集、含 HAC 对比，数字见主结果。

三条均实，无凑数。

### 讲故事方式

主线一句话："Scaffold-GS 消了高斯对锚点的冗余，我消锚点对锚点的冗余——粗层锚点编码细层锚点。"最有力的是 Figure 1 的四联画：冗余存在（余弦相似度热图）→ 我的机制（分级示意）→ bit 节省 → 10× 压缩的渲染对比，一页讲完"为什么-怎么做-值多少"。Table 2 的强基线消融是给审稿人的定心丸：增益不是熵编码本身带来的。

## 对 DCCA-GS 的可借鉴点

1. **锚点级上下文能否嫁接进我们框架（任务重点，按硬约束逐项分析）**：
   - **解码端可重算性：成立**。式 8 的上下文全部由"已解码锚点的属性 + 自身量化位置"构成，解码端按 K−1→0 同序解码即可现算，无需任何侧信息——这一点不违反我们的零侧信息约束。
   - **码流契约：有一处硬伤、一处软伤**。硬伤：它的完整版每锚点带一个量化超先验向量 ẑ，这是**新增码流字段**，直接违反"不改码流契约"。零新增字段的嫁接版只取式 8 的 k<K−1 分支（ψ=[f̂_j^{k+1}; l̂_j^{k+1}; x̂_i]，粗层已解码特征+缩放+位置），字段集合不变。软伤：解码顺序从"全部锚点一次解码"变成"分 3 层串行"，且分级映射（式 4-6 的体素量化 + 最小下标代表点规则）必须在解码端按同一规则从存储位置重算——需要把 K=3、τ=0.2 定成常数（ContextGS 本来就全数据集共用）并让体素尺寸从已有 header/bounds 推出，否则就要动 header。这属于"解释规则"的扩充而非字段增删，是否可接受要在我们契约文档里明确裁决。
   - **与我们 SPA 的相互作用是最大风险**：ContextGS 在锚点初始化后一次性分级；我们的锚点集合被 SPA 和 densify 动态增删，分级必须在 SPA 收尾后重算，而熵模型是训练学的——训练期上下文结构与最终编码期不一致。可行的合并方案：SPA 删净锚点后冻结集合，重算分级，再对熵模型（含我们共享的 AQM MLP 之外的条件分支）做一段 finetune——这正好能和 GaussianSpa 借鉴点 1 的 light tuning 阶段合成同一个收尾阶段。
   - **收益预期要先打折**：ContextGS 对 HAC 的最大增益在 BungeeNeRF（同体积 +0.42dB，Table 1）；我们的锚点数已被 SPA 砍到约 1/2 甚至更低，锚点间冗余同步缩小，且我们底座是 HAC++（上下文已比 HAC 强）。验证实验：DB playroom，SPA 冻结后，哈希上下文对照"哈希 + 粗层已解码锚点上下文"，报 BD-rate 与解码时间。
2. **解码开销的现成答辩证据（第 5.3 节）**：锚点级自回归只有 K=3 轮循环、层内并行，解码 17.85s（Table 4）与 HAC 量级相当——将来若我们做任何"按组条件化"的上下文，可用它反驳"自回归=解码慢"的天然质疑。
3. **位置字段的教训（第 5.3 节 + Table 4/5）**：给锚点位置做熵编码，低码率省一点、高码率反而变大、编解码时间翻倍。我们本就把 xyz 原样进码流，这份证据支持我们继续不动位置字段，把压缩预算留给 feat/scaling/offsets。
4. **消融设计可搬**：Table 2 的"强基线 + 2×2 组件开关"与 Table 3 的"机制内部耦合对照（分层 vs 复用）"正是我们主实验章节该有的两组对照——尤其 Table 3 提醒我们：如果将来引入任何分层/分组结构，要单独验证"结构"与"结构带来的复用"是否必须绑定。
