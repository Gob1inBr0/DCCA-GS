# 《Provable Pruning for Efficient 3D Gaussian Splatting via Coresets》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2607.02721 |
| 发表 | arXiv 预印本（v1，2026-07-02，cs.CV），论文未标注会议接收信息 |
| 作者/机构 | Waseem Mousa, Alaa Maalouf（海法大学计算机系） |
| 阅读材料 | 全文（.lit-cache/txt/2607.02721.txt；该 txt 含控制字符，公式按 pymupdf 重新抽取的全文核对） |
| 开源 | github.com/waseem-m/3dgs_provable_coresets（论文首页声明代码开源） |
| 与 DCCA-GS 的关系 | 同属剪枝方向的基础措施来源参考：我们用 GaussianSpa 式训练期 ADMM 剪枝，本文是首个给剪枝加乘法渲染保证的 coreset 定理，两者的保证形式可直接对比 |

> 一句话定位：这篇论文研究"能不能证明地用一个加权小高斯子集（coreset）替换整个 3DGS 场景而不掉画质"。结论分两层：不加限制时乘法保证不可能存在（定理 1）；限定在给定渲染分辨率的有限代表查询族上，给出首个带乘法保证的 3DGS 剪枝定理（定理 2/3/4）。省体积的方式是剪枝（删高斯+重加权），不含量化或熵编码。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

原文（第 1 节"The gap"段）："the compressed Gaussian set is often followed by a costly recovery stage, requiring substantial hardware resources and many fine-tuning iterations. Second, existing approaches are largely heuristic, providing no provable guarantees on approximation error or subset size." 翻译过来：现有剪枝方法有两个缺口——剪完通常要昂贵的恢复训练来补救画质；方法本身是经验规则，对近似误差和子集大小没有任何可证明的保证。

它的做法是把问题换成 coreset 语言：coreset 是一个小的加权子集，对一族查询近似保持某个目标函数。对应到 3DGS：查询 = （相机位姿，像素，RGB 通道），目标 = 该通道的标量渲染值。中央问题（第 1 节原话）："Can a full 3DGS scene be provably replaced by a much smaller weighted subset of the gaussian while preserving the target rendering objective, and under what assumptions is such a guarantee possible?"

### 支撑观察（直觉 / 数据 / 失败实验）

理论观察来自前向遮挡耦合这个结构性事实（第 1 节"The challenge"）：从前到后的合成通过透射率把高斯两两耦合，删掉或改动靠前的一个高斯会影响后面所有高斯的贡献。这解释了为什么神经网络的 coreset 剪枝定理（引用 27-29、51）不能直接搬过来——那些工作假设目标函数在输入固定后有固定的逐项分解，而真渲染目标没有。

实验观察：图 2（第 4 节）显示剪得越狠（比率 0.90→0.99），本文方法对对手的 PSNR/SSIM 差距越大，即"激进压缩+不恢复"是重要性估计质量最要紧的区域；图 3 显示恢复预算只有 10-200 次迭代时优势保持，GHAP 要到 200 次迭代才接近它。

### 显然的路为什么没走

- 直接套用神经网络剪枝的 coreset 框架（Baykal 等、Liebenwein 等，引用 27-29）：不行，因为真渲染目标随子集变化而改变逐项分解（透射率耦合）。本文的办法是先把目标松弛成"固定全场景透射率"的逐项形式拿到定理 2，再用对数透射率稳定性假设把保证转移回真渲染（定理 3）——这是这篇论文最核心的设计取舍。
- 把保证铺到所有连续视角上：定理 1 证明做不到，所以主动退到"给定渲染分辨率诱导的有限查询族"（第 3.2 节），再用 Lipschitz + ρ-网把有限保证延拓到紧致区域（第 3.4 节），而不是硬要一个无限保证。
- 用不透明度、梯度等代理量打分：它不用代理，直接用渲染目标本身的逐高斯分解 a(G,gi,q) 定义敏感度（式 1），分数和要保的目标是同一个东西。

## 二学：机制（洞察怎么变成方法）

### 核心流程

本文是剪枝方法，只有"预训练完成后的一次性剪枝"一个环节，不含量化、熵编码、解码端流程。

**输入**：预训练好的 3DGS 场景 G = {g1,…,gN}，每个高斯 gi = (μi, Σi, αi, ci)（均值、协方差、不透明度、视角相关颜色，第 3 节）；目标保留数 m（剪枝预算，算法 1）。

1. **构造代表查询族 Q′**（算法 1 第 1 步，第 3.2 节"Setting"）：从代表相机、像素/tile、RGB 通道取一个有限查询集。每个查询 q = (C, x, λ)：相机外参 C、图像平面位置 x、输出通道 λ。Q′ 的密度决定保证覆盖位姿平移/旋转的精细程度。
2. **算每个高斯对每个查询的贡献**（算法 1 第 2 步）：投影核值 k(gi,q)，衰减因子 ρ(gi,q) = αi·k(gi,q)，前缀透射率 T(G,gi,q) = ∏_{j≺G,q i}(1 − ρ(gj,q))（≺G,q 是按视深排的从前到后序），单项贡献 a(G,gi,q) = ci(q)·ρ(gi,q)·T(G,gi,q)，全场景标量渲染值 A(G,q) = Σ_{gi∈G} a(G,gi,q)。丢弃 A(G,q)=0 的查询。
3. **算敏感度并归一化成采样概率**（算法 1 第 3 步，式 1）：imp(gi,q) = a(G,gi,q)/A(G,q)，敏感度 s(gi) = max_{q∈Q′} imp(gi,q)，总敏感度 S = Σ_j s(gj)，采样概率 p(gi) = s(gi)/S。
4. **按概率采样 m 个高斯**（算法 1 第 4 步）：i.i.d. 采 m 次，ni 记 gi 被采中的次数。
5. **赋逆概率权重**（算法 1 第 5 步）：wi = ni/(m·p(gi))。wi=0 表示删除，wi>0 表示保留并重加权；输出是加权剪枝场景，‖w‖0 ≤ m。对固定目标松弛，这个估计量在每个查询上无偏。

**实验用的聚合档位**（第 3.5 节）：把"单个查询"从单通道逐步聚合成像素、tile（tile 划分与 3DGS 光栅化内核的 tile 一致）、整个场景级目标，敏感度采样论证不变，只是随机变量换成聚合后的目标。主实验用 per-scene（场景级聚合）变体（第 4 节）。另有两个工程变体：L2 分数 aL2 = ci(q)²ρ(gi,q)²T(G,gi,q)² 和去颜色分数 aNC = ρ(gi,q)T(G,gi,q)（第 3.5 节；上标按 txt 抄写，原文公式以 PDF 为准）。

### 关键公式（按原文抄，注明编号）

（符号按 pymupdf 重新抽取文本转写；抽取可能弄乱上下标，原文公式以 PDF 为准。）

- 渲染查询与逐项贡献（第 3 节）：ρ(gi,q) := αi·k(gi,q)；T(G,gi,q) := ∏_{j≺G,q i}(1 − ρ(gj,q))；a(G,gi,q) := ci(q)·ρ(gi,q)·T(G,gi,q)；A(G,q) := Σ_{gi∈G} a(G,gi,q)。k 是 gi 在位姿 C 下投到像素 x 的投影高斯核值。
- 加权剪枝场景的真渲染（第 3 节）：Aw(G,q) := Σ_{gi∈G} wi·ci(q)·ρ(gi,q)·Tw(G,gi,q)，其中 Tw(G,gi,q) := ∏_{j≺G,q i}(1 − wj·ρ(gj,q))。注意权重直接乘进衰减因子，透射率结构随之改变。
- 定义 1（乘法 ε-coreset）：对每 q∈U，|A(G,q) − Aw(G,q)| ≤ ε·A(G,q)。
- 定义 2（固定目标松弛，第 3.2 节）：Ath_w(G,q) := Σ_{gi∈G} wi·a(G,gi,q)——每个逐项项冻结在全场景透射率下。这是纯理论量，定理 2 的保证对象。
- 式 (1)（敏感度）：s(gi) := max_{q∈Q′} a(G,gi,q)/A(G,q)。
- 定理 2（查询数依赖的 coreset，第 3.2 节）：若 m ≥ (3S/εc²)·log(2|Q′|/δ)，则以概率 ≥ 1−δ，w 是 Ath 在 Q′ 上的 εc-coreset，且 ‖w‖0 ≤ m。子集大小只随查询数 |Q′|（即渲染分辨率）对数增长。
- 定义 2（加权渲染有效性）与假设 1（对数透射率稳定性，第 3.3 节）：有效性要求 0 ≤ wi·ρ(gi,q) < 1；稳定性要求存在 γ ≥ 0 使 |log T(G,gi,q) − log Tw(G,gi,q)| ≤ γ 对所有 q∈Q′ 和 wi>0 的 i 成立。
- 定理 3（有限查询真渲染 coreset，第 3.3 节）：在有效性和假设 1 下，同一样本量条件，以概率 ≥ 1−δ，w 是真渲染目标 A 在 Q′ 上的 εr-coreset，εr := max{1 − e^{−γ(1−εc)}, e^{γ(1+εc)} − 1}。这是"目标保证 → 渲染保证"的转移定理，误差多出 e^{±γ} 一项。
- 假设 2 / 定义 3 / 定理 4（紧致区域延拓，第 3.4 节）：区域上要求全场景与剪枝场景渲染目标 Lipschitz（常数 Lfull、Lw）且 A(G,q) ≥ Amin > 0；Q′r 是半径 ρ 的 ρ-网。若代表集上误差为 ε0，则整个区域上 εreg := ε0 + ((1+ε0)·Lfull + Lw)·ρ/Amin。推论 1 反过来给定目标误差 ε̄ 求 ρ 的上界：ρ ≤ (ε̄−ε0)·Amin/((1+ε0)·Lfull + Lw)。
- 引理 1（附录 C.3，乘法 Chernoff）：0 ≤ Zt ≤ 1、均值 μ 的 i.i.d. 变量满足 Pr[|（1/m)ΣZt − μ| > εμ] ≤ 2exp(−ε²μm/3)。定理 2 的证明就是用它加对 Q′ 的联合界。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 敏感度 s(gi)：该高斯在全部代表渲染查询上的最大归一化贡献 max_q a(G,gi,q)/A(G,q)；即"删掉它最坏会伤哪个查询多少" |
| 哪一端算得出（编码端/解码端/仅训练期） | 只在部署前的剪枝端算：需要对全部代表视角做前向并累积每个高斯的逐查询贡献；解码端不参与 |
| 有无侧信息 | 不产生码流侧信息：分数只在剪枝期内部使用，输出是标准高斯子集+标量权重，不随文件传输任何分数 |
| 训练期还是编码期 | 预训练完成之后、部署之前（post hoc），与训练过程解耦 |
| 和渲染质量的距离（直接测质量还是代理量） | 不是代理量：分数直接取自要保的渲染目标 A 的逐高斯分解，是目标本身；但保证对象经过两层放宽（固定目标松弛 + 透射率稳定性） |

（本文不含量化、熵编码、渐进码流组织，无"与码流组织相关的细节"一节。）

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 体积/码率怎么算：论文不报告任何 MB、存储或 FPS 数字（全文检索无此类数字），只用剪枝比率（删掉的高斯比例，0.80-0.99）和恢复迭代数作为压缩轴；剪枝输出是标准 3DGS 场景，体积随高斯数缩减，但论文未给换算。
- 数据集与协议（第 4 节、附录 A.1）：3DGS 原文的 13 个场景——Mip-NeRF 360 九景、Tanks & Temples 两景（truck/train）、Deep Blending 两景（drjohnson/playroom）。prune-only（剪完零微调）与短恢复（微调 10/50/100/200 次迭代）两种设定；指标 PSNR/SSIM/LPIPS。硬件：一张 NVIDIA A100-SXM4-40GB + AMD EPYC 7742。
- 和谁比（第 4 节）：GHAP（最优传输式全局精简）、PUP 3D-GS（不确定性剪枝）、Trimming the Fat、均匀采样。未与任何量化/码流类压缩方法（HAC、LightGaussian 等）比存储或体积。
- 训练配置要点：默认用 per-scene 敏感度变体；预训练场景与剪枝预算 m 的具体来源论文未报告细节。

### 主结果表抄录（注明表号）

表 1（Per-dataset average prune-only results，剪枝比率 0.90 与 0.99，数据集平均，共 10 行）全抄如下；列组为三个数据集各自 PSNR↑/SSIM↑/LPIPS↓：

| 剪枝比率 | 方法 | Mip-NeRF 360 PSNR | SSIM | LPIPS | Deep Blending PSNR | SSIM | LPIPS | Tanks&Temples PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | GHAP | 16.76 | 0.458 | 0.494 | 20.53 | 0.744 | 0.437 | 14.83 | 0.488 | 0.483 |
| 0.90 | PUP | 15.79 | 0.535 | 0.429 | 19.52 | 0.765 | 0.393 | 13.38 | 0.577 | 0.406 |
| 0.90 | Trim | 14.45 | 0.413 | 0.480 | 16.52 | 0.642 | 0.429 | 12.46 | 0.471 | 0.458 |
| 0.90 | Unif | 13.16 | 0.391 | 0.516 | 15.69 | 0.654 | 0.501 | 10.22 | 0.437 | 0.482 |
| 0.90 | Ours | 18.16 | 0.580 | 0.417 | 22.16 | 0.790 | 0.390 | 16.35 | 0.599 | 0.407 |
| 0.99 | GHAP | 10.19 | 0.177 | 0.664 | 10.14 | 0.401 | 0.653 | 7.94 | 0.224 | 0.675 |
| 0.99 | PUP | 10.67 | 0.227 | 0.624 | 10.13 | 0.428 | 0.621 | 8.94 | 0.320 | 0.627 |
| 0.99 | Trim | 9.58 | 0.136 | 0.661 | 7.66 | 0.132 | 0.683 | 6.23 | 0.104 | 0.668 |
| 0.99 | Unif | 10.27 | 0.174 | 0.642 | 7.84 | 0.189 | 0.696 | 6.92 | 0.163 | 0.658 |
| 0.99 | Ours | 13.97 | 0.396 | 0.600 | 15.62 | 0.660 | 0.526 | 11.98 | 0.435 | 0.606 |

补充（表 4 共 18 行=6 比率×5 方法，此处节选 0.95 一档）：剪枝 0.95 时 Ours 在 Mip-NeRF 360 平均 PSNR 16.52 / SSIM 0.506 / LPIPS 0.484，对照 GHAP 14.15 / 0.361 / 0.556、PUP 13.49 / 0.416 / 0.508（表 4）。

### 消融设计（注明表号/图号）

- **剪枝比率×数据集全矩阵对照**（表 2：0.80/0.85/0.90 逐场景；表 3：0.95/0.97/0.99 逐场景）：13 景 × 3 指标 × 6 比率 × 5 方法的完整记录，值得学的是把"平均趋势"和"逐场景反转"分开报告（附录 A.4 明确说数据集平均可能掩盖单景反转）。
- **敏感度分辨率消融**（图 5 prune-only、图 4 恢复后；附录图 23）：per-channel / per-pixel / per-tile / per-image / per-scene 五档聚合 + scene-L2-nc 工程变体。结论：聚合越粗越稳，per-scene 最好，其次 per-tile、per-pixel、per-channel（第 4 节正文）。
- **短恢复对照**（图 3/10 Mip-NeRF 360、图 11 Deep Blending、图 12 T&T；恢复预算 10/50/100/200 迭代）：分离"剪完立刻的质量"和"小预算能恢复多少"两种效应；发现 GHAP 在 200 迭代后逼近（第 4 节），并给出解释——GHAP 是合并式全局精简，恢复自由度更大。
- **转置恢复曲线**（图 7/8/9）：固定变体，x 轴换剪枝比率、每条曲线一个恢复迭代数，直接读出"哪一档前缀在零恢复时已可用"。

### 贡献列表（原文照抄 + 逐条标注）

1. "Impossibility in full generality. We prove that no non-trivial multiplicative coreset guarantee can hold for unrestricted 3DGS rendering: when the query family is rich enough to isolate individual Gaussians, every Gaussian may become necessary; see theorem 1." —— [机制] 负结果：查询族够富（能隔离单个高斯）时每个高斯都不可少，任何非平凡乘法保证都不成立。
2. "Resolution-dependent provable 3DGS pruning (coreset). First, for a desired rendering resolution (a given set or a grid of views/rays), and a fixed full-scene reference decomposition, we provide the first provable pruning (coreset construction) algorithm for 3DGS; the remaining Gaussian subset size scales logarithmically by the size of the desired views set (resolution); see section 3.2 and theorem 2. Then, we prove that transfer from a fixed-objective guarantee for true 3DGS re-rendering is possible under explicit validity and log-transmittance stability assumptions; see section 3.3 and theorem 3." —— [机制] 正结果主体：有限代表查询族上的首个 3DGS 剪枝定理（子集大小随分辨率对数增长），加真渲染转移定理。
3. "Extension beyond finite resolutions (query sets). We extend the representative-query theorem from finite query families to compact query regions using representative covers and Lipschitz control; see section 3.4 and theorem 4." —— [机制] 把有限族保证延拓到紧致查询区域，代价是再加一项与覆盖半径成正比的误差。
4. "Open-source implementation. We release the sensitivity computation, coreset construction, pruning, and evaluation pipeline at github.com/waseem-m/3dgs_provable_coresets." —— [工程] 开源敏感度计算、coreset 构造、剪枝与评测全流程。

### 讲故事方式

主线一句话："不加限制不可能 → 加一个实际中就有的限制（渲染分辨率）就可能 → 保证可转移到真渲染 → 可延拓到连续区域"。附录 D.3 把它总结成四层陈述（impossibility / fixed-objective existence / transfer / compact-region extension）。最有力的一张图是图 2：上排是本文减对手的逐指标差值曲线，随剪枝比率 0.90→0.99 差距扩大，把"激进压缩+不恢复"这个卖点画成了一条越拉越开的曲线；最有力的一张表是表 1（0.90/0.99 平均上 Ours 全面第一）。

## 四学：原文关键段翻译

### 方法节核心段（第 3.2 节，2 段）

> **原文（第 3.2 节）**："We seek to quantify the importance of each Gaussian. Intuitively, a Gaussian may contribute very differently across queries: it may be highly influential for some views, rays, or pixels, and nearly irrelevant for others. We therefore first define the contribution of each Gaussian to each query. We then define its global importance, or sensitivity, as the maximum relative contribution over all queries. This identifies Gaussians whose removal could cause large approximation errors, and distinguishes them from Gaussians that are consistently negligible."
>
> **译文**：我们想量化每个高斯的重要性。直观上说，一个高斯在不同查询上的贡献可以差别很大：它可能对某些视角、光线或像素影响很大，而对另一些几乎无关。因此我们先定义每个高斯对每个查询的贡献，然后把它的全局重要性（即敏感度）定义为在全部查询上的最大相对贡献。这样就能识别出"删掉会造成大近似误差"的高斯，并把它们与一贯可忽略的高斯区分开。

> **原文（第 3.2 节）**："Pruning pipeline: 3DGS Coreset via Sensitivity sampling. For every Gaussian gi, define p(gi) := s(gi)/S. For a target sample size m, sample (i.i.d) m Gaussian G′ = {g′1, . . . , g′m} from the scene G, where each g′i is sampled with probability p(g′i). Define ni to be the number of times gi was sampled. The output weight vector w ∈[0, ∞)N is then wi := ni/(mp(gi)). Equivalently, each sampled occurrence of Gaussian gi contributes an inverse-probability weight (mp(gi))−1. The resulting weighted reduced scene is supported on at most m Gaussians and is unbiased for the fixed-objective relaxation on every fixed query."
>
> **译文**：剪枝流程：通过敏感度采样构造 3DGS coreset。对每个高斯 gi，定义 p(gi) := s(gi)/S。给定目标样本数 m，从场景 G 中独立同分布地采样 m 个高斯 G′ = {g′1,…,g′m}，每个 g′i 以概率 p(g′i) 被采到。记 ni 为 gi 被采到的次数，输出权重向量 w ∈[0,∞)^N 取 wi := ni/(m·p(gi))。等价地说，gi 的每次被采中都贡献一个逆概率权重 (m·p(gi))^{-1}。所得加权缩减场景至多含 m 个高斯，且对每个固定查询在固定目标松弛下是无偏的。

### 主结果表述段（第 4 节）

> **原文（第 4 节）**："Prune-only results at aggressive compression. We evaluate prune-only fidelity at the strongest compression levels where at most 20% of the Gaussians are retained. As shown in table 1, for prune ratios 0.9 and 0.99, our method variants consistently outperform the competing methods across all test datasets, achieving the best PSNR, SSIM, and LPIPS scores in the prune-only setting. The same trend persists across the remaining compression ratios: {0.80, 0.85, 0.90} and {0.95, 0.97, 0.99} which are reported, respectively, in table 2 and table 3 in the appendix due to space."
>
> **译文**：激进压缩下的纯剪枝结果。我们在最强压缩档（至多保留 20% 高斯）上评估纯剪枝保真度。如表 1 所示，在剪枝比率 0.9 和 0.99 下，我们的方法变体在全部测试数据集上一致优于对手方法，在纯剪枝设定下取得最好的 PSNR、SSIM 和 LPIPS 分数。其余压缩比率 {0.80, 0.85, 0.90} 与 {0.95, 0.97, 0.99} 上的同样趋势，因篇幅原因分别报告于附录表 2 与表 3。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 高 | 截至 2026-09 的 3DGS 剪枝文献（Trimming the Fat、PUP 3D-GS、GaussianSpa、GHAP）全部是启发式；本文给出首个乘法保证剪枝定理，且"不可能定理 + 分辨率相关正定理"的问法本身有独立理论价值 |
| 证据强度 | 中 | 13 场景、4 基线、6 个剪枝比率、多恢复预算、代码开源，覆盖面合格；但全部结果只报画质不报体积/存储/耗时（论文未报告任何 MB 数字），理论参数（S、γ、ε）在实验中没有实例化或测量，恢复实验只到 200 迭代 |
| 对期刊版的威胁度 | 低 | 不撞渐进码流、零侧信息、内容自适应量化三条主张：它是纯剪枝，不改参数表示、不产码流；与我们的唯一交集是"渲染敏感度"这个词——我们的敏感度监督服务于量化步长，它的敏感度服务于剪枝采样，机制和用途都不同 |

## 对 DCCA-GS 的可借鉴点

1. **敏感度分数定义**（最大归一化贡献 max_q a/A）→ 可用于给我们剪枝环节的打分做参照实现：GaussianSpa 式 ADMM 给的是训练期稀疏化的 mask，没有显式"最坏情况贡献"解释。可以在自采无人机场景上跑一个对照：同预算下比 ADMM mask 与敏感度 top-m 剪枝的 PSNR（剪完零微调口径），验证两者选出的高斯重合度与质量差。预期收益是得到一个可写进论文的"我们的剪枝分数等价于某种重要性估计"的论证，成本是要在前向里累积逐高斯贡献。
2. **per-tile 聚合的粒度与 3DGS 光栅化的 tile 划分天然一致**（第 3.5 节，tile 粒度与渲染核一致）→ 我们若要把敏感度算进训练期监督，按 tile 聚合比对每像素便宜一个量级，且消融（图 5）显示 per-tile 已接近 per-scene。这支持我们现有"共享一个 MLP、场景级监督"的聚合选择，可作为引用依据。
3. **"不恢复/短恢复"评估口径** → 我们的渐进码流低档位本质上是"粗剪枝+粗量化"的前缀，可以借用它的口径报告零微调/极短微调下的低档画质，证明低档前缀本来就可用，而不只是中间过渡态。实验量小（同模型多测几次推理）。
4. **不可能定理的写法教训**（第 3.1 节：查询族够富时保证必失效）→ 写期刊版时对"零侧信息可重算"这类主张要先划清适用边界（哪个口径、哪些量），把"哪里不保证"写明白，防止审稿人用极端反例攻击。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 剪枝分数怎么定 | 剪枝期对全部代表视角前向累积逐高斯贡献，取最大归一化贡献作敏感度，按敏感度采样（式 1、算法 1） | 训练期 GaussianSpa 式 ADMM：稀疏化项与重建项交替优化，把不重要高斯的贡献压到零，mask 是训练的副产品 | 它的分数显式对着渲染目标、有保证但只管有限查询族；我们的分数是训练内生的、无保证但和训练目标联合优化。两者选中的高斯未必相同，是可跑的对照实验 |
| 保证形式 | 乘法保证：有限查询族上固定目标 (1±εc)（定理 2），真渲染 ±εr（定理 3），紧区域 εreg（定理 4）；无限制情形证明不可能（定理 1） | 无形式化保证；质量靠 RD 曲线和 HAC++ 基线对照支撑 | 这篇论文证明了一个有用的反面事实：任何"带保证"的主张必须先说清查询族和目标。我们若想给渐进码流的低档位加可证明陈述，应该套它的"固定查询族+逐项分解+稳定性"框架，而不是追求全域保证 |
| 剪完之后怎么办 | 主打剪完不动或只微调 10-200 次迭代（第 4 节），明确不追求大预算下的最优 | 剪枝融在训练期（SPA），没有独立的"剪完再恢复"阶段；后续是量化与熵编码管线 | 定位不同：它卖"低算力部署下剪枝即用"，我们卖"训练完产出一个多档码流"。它的实验恰好量化了"恢复预算 vs 剪枝质量"的收益曲线，可用来论证我们不需要纯剪枝基线的长恢复 |
| 输出表示与解码端 | 标准 3DGS 高斯的加权子集，任何标准渲染管线可渲染（第 2 节"representation-preserving"） | 锚点式 HAC++ 码流，需要专门解码器与上下文模型 | 它放弃了压缩率上限换解码兼容性；我们放弃了渲染器兼容性换码率。两者在"谁能直接吃输出"上完全错开，不构成正面竞争 |
| 量化 / 熵编码 / 渐进码流 / 侧信息 | 该论文不涉及（不改参数表示、不产码流，分数不随文件传输） | 单文件渐进分层码流、零侧信息重算、二值分解区间编码、层条件熵编码 | 完全错开的议题；本文可作为"剪枝这一层怎么做最稳"的引用支撑，不冲突 |
