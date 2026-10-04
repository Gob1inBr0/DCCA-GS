# 《Compressing 3D Gaussian Splatting via Cross-Representation Priors》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2609.23005 |
| 发表 | IEEE Transactions on Image Processing（已接收，版权行 © 2026 IEEE；arXiv v1 2026-09-19） |
| 阅读材料 | 全文（.lit-cache/txt/2609.23005.txt，14 页 PDF 核对过 Table I/II/III、式 (8)、Fig. 8 关键页） |
| 与 DCCA-GS 的关系 | 同类竞品：锚级条件熵建模（特征相似度定根叶链接、已解码根特征做叶锚熵编码条件）+ 哈希网格共享特征进锚表示；无渐进码流、有侧信息 |

> 一句话定位：这篇在 HAC 式锚点压缩框架上做了两件事来压锚间冗余——按锚特征余弦相似度（而不是空间邻近）建一对一的"根-叶"链接，叶锚用已解码根特征做条件熵编码（COHS）；再把锚特征拆成"哈希网格查询的共享分量 + 单独编码的个体分量"（SFA）。对比 HAC 平均省 41.87% 码率（Table II），但换码率点要重新训练、码流里带 3.74% 的结构侧信息，没有渐进分层。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

论文认为锚级冗余的利用方式被两个框架限死了（第 I 节原话）：

- HAC "lacks the use of cross-anchor priors"——每个锚独立编码，哈希网格只负责预测概率参数，锚与锚之间的相关性没有进熵模型；
- ContextGS "constructs a hierarchical anchor structure to model contextual dependencies. However, its partition mechanism is determined solely by spatial proximity"——层级依赖只看空间距离，"ignoring the feature-level relationships among anchors"。

作者的判断是：锚点聚成高阶结构之后，"a distinct form of redundancy emerges across anchors themselves"（第 I 节引 [13]），而现有方法要么完全不管（HAC），要么只用空间邻近管（ContextGS），所以剩下的收益在"跨锚、且不限于局部空间"的先验里。

### 支撑观察（直觉 / 数据 / 失败实验）

1. **长程特征对应的分布证据（Fig. 3，第 IV-B 节）**：在 Mip-NeRF360 Bicycle 场景训练出的锚集里均匀抽 10000 个查询锚，按余弦相似度阈值 0.7 / 0.9 / 0.95 找对应锚，算查询锚与对应锚的欧氏距离（按场景锚集包围盒对角线归一化）的累计分布。曲线直到约 0.5（场景空间跨度的一半）还在增长，说明相当一部分特征高度相似的锚在空间上离得很远。这是全文的立论实验：空间邻近假设（ContextGS 的分层依据）漏掉了一类真实的冗余。
2. **共享特征承载低频信息的可视化（Fig. 4，第 IV-C 节）**：同一个训练好的模型，把共享特征 fs 置零渲染，画面出现结构缺失和局部不一致；把个体特征 fi 置零渲染，只剩粗糙但全局一致的低频布局。作者据此说 fs 主要装场景级的低频外观，高频细节仍在个体特征里。
3. **团队背景线索（第 IV-B 节 + 参考文献 [56]）**：COHS 的思想来源写得明白——"learned image compression has explored content-adaptive dependencies beyond fixed spatial neighborhoods"（引他们自己组的 Content-Aware Mamba，ICLR 2026 [56]）。这是把学习图像压缩里"内容自适应上下文"的传统搬进 3DGS 锚编码，通讯作者 Guo Lu 组本身是做学习视频/图像压缩的。观察 1 是为这个移植补的场景内证据。

### 显然的路为什么没走

- **为什么不建多层层级（ContextGS 的加深版）**：第 IV-B1 节专门解释——多跳依赖会让量化噪声、锚掩码、或上一层解码不准传导到下游所有锚的概率预测，解码器要跟更长的依赖顺序，随机访问更不灵活。所以只用一层根-叶直达依赖，并把这说成复杂度和鲁棒性的取舍，深层层级留给未来工作。
- **为什么不用空间邻近做配对（ContextGS 已做的事）**：Fig. 3 直接给出了反面证据——特征对应的锚常常不在附近。
- **为什么不像 HAC 那样把哈希网格只当熵模型参数预测器**：第 IV-C 节对比——HAC 的哈希特征只进熵模型，SFA 把同一个上下文哈希网格的特征拼进锚表示、直接参与高斯生成，把"每锚特征里重复携带的场景公共分量"搬到共享表示里。作者明确说区别在于先验的"来源和用途"（第 IV-A 节末段）。
- **为什么不做在线检索部署**：训练期在线检索候选、更新链接，但最终码流只存掩码和链接向量，部署端不需要任何特征检索（第 IV-B1 节 bitstream format 段）——把结构搜索的成本留在训练期，推理只消费结果。

## 二学：机制（洞察怎么变成方法）

### 核心流程

底座是 HAC：锚点表示 + 二值多分辨率哈希网格做熵模型上下文 + 高斯分布似然 + 率失真联合训练（第 IV-A 节）。CRP-GS 加两个模块，训练、编码、解码分开讲：

**训练期**

1. 前 30000 迭代正常训练（HAC 式稳定期），不加 COHS——因为早期锚特征不稳定，相似度检索没有意义（第 IV-B1 节、V-A 节）。
2. 第 30000 迭代起启用 COHS，再训 10000 迭代。启用时做第一次依赖更新：对每个锚在全体锚里按余弦相似度取前 M=10 个候选（候选集 C），然后跑 Algorithm 1（锚依赖更新）把锚分成三类：叶锚（有链接指向自己的根）、根锚（被至少一个叶指向）、自由锚（无链接、也不被指向）。阈值从 τ=1 按步长 Δτ=0.05 递减到 τmin=0.8，每轮里按锚序号遍历：未链接、未当过根的活跃锚，在自己的候选里找相似度≥τ、同样未链接、不是根的锚，取相似度最高的那个作为自己的根；一个锚只能当一次根（根集合 R 去重），所以这是一对一的链接。训练期间依赖更新共做两次（第 V-A 节），每次用当前最新的锚特征重算候选池再重建链接——因为率失真训练会让锚特征持续演化。
3. 率失真联合优化（Fig. 2 双路）：率路算三类锚的条件似然、累积成熵损失；失真路走 Scaffold 式渲染管线，锚特征换成聚合特征 fa（SFA），渲染出图算 L1/SSIM。损失见式 (8)。

**编码期**（第 IV-B2 节）

1. 从码流外（训练产物）拿到掩码 m 和链接向量 L，确定根、叶、自由三类锚的划分。
2. 固定顺序编码：先根锚、再自由锚（两类共用基础熵模型，式 (5)），最后叶锚（条件叶熵模型，式 (4)）。scale 和 offset 属性也走基础熵模型。
3. 每个根锚编码完后，其量化解码特征 ˆf(r) 被缓存，作为它对应叶锚编码时的条件——编码端和解码端用的是同一个量化后的根特征，保证两侧条件严格一致。
4. 锚位置（几何）用标准点云编解码器压缩（第 V-C 节 Discussion 段）。

**解码期**（第 IV-B2 节 bitstream format 段 + 解码顺序段）

1. 先从码流读活跃锚掩码 m 和链接向量 L，确定性重建根/叶/自由划分——不需要任何在线检索或全对特征搜索。
2. 按同样顺序解码：根（基础模型）→ 缓存解码特征 → 自由（基础模型）→ 叶（条件模型，条件是缓存的根特征）。
3. 组内按存储的锚序号顺序处理。

### 关键公式（按原文抄，注明编号）

式 (1)（第 III 节，3DGS 高斯基元，预备知识）：

```
G(x) = e^{ -1/2 (x-µ)^T P^{-1} (x-µ) }
```

x ∈ R³ 是三维坐标，P 是协方差矩阵，分解为 P = R S S^T R^T（旋转 R、缩放 S）。txt 抽取的指数排版有错乱，写法以 PDF 为准。

式 (2)（第 III 节，α 混合渲染）：

```
C(x′) = Σ_{i∈N} c_i σ_i Π_{j=1}^{i-1} (1-σ_j),   σ_i = α_i G′_i(x′)
```

x′ 是像素位置，N 是对该像素有贡献的高斯数，G′ 是投影到二维的泼溅高斯。同上，排版以 PDF 为准。

式 (3)（第 IV-B2 节，量化区间的概率）：

```
p(ˆz_i) = ∫_{ˆz_i - q_i/2}^{ˆz_i + q_i/2} φ_{µ_i,σ_i}(x) dx = Φ_{µ_i,σ_i}(ˆz_i + q_i/2) − Φ_{µ_i,σ_i}(ˆz_i − q_i/2)
```

ˆz_i 是第 i 个锚的某个待熵编码属性（特征 f ∈ R^{D}、scale l ∈ R^6、offset o ∈ R^{3K} 三者之一）的量化值，φ 和 Φ 分别是高斯的概率密度函数和累积分布函数，µ_i、σ_i 是熵模型预测的高斯参数，q_i 是量化步长。即：量化值取到某符号的概率 = 高斯密度在 ± 半个步长区间上的积分。

式 (4)（第 IV-B2 节，叶锚条件熵模型——本文核心公式）：

```
µ^(l)_j = MLP([ ˆf^(r)_i ; x_i ; x_j ]),   σ^(l)_j = MLP(f^h)
```

根锚 i 与叶锚 j 配对。均值 µ^(l)_j 由三部分拼接后过 MLP：已解码（量化后）的根特征 ˆf^(r)_i、根位置 x_i、叶位置 x_j。标准差 σ^(l)_j 只从哈希网格上下文特征 f^h 预测，不看根特征。注意这个不对称设计：根的对应关系只影响均值，方差仍由空间上下文决定。

式 (5)（第 IV-B2 节，基础熵模型）：

```
µ^(b)_k, σ^(b)_k = MLP(f^h)
```

根锚、自由锚的特征，以及全部非特征属性（scale、offset），共用这一个模型：均值方差都从哈希网格上下文 f^h 预测（继承 HAC）。

式 (6)（第 IV-C 节，SFA 特征聚合）：

```
f_s = Interp(x_a, H_s),   f_a = f_s ⊕ f_i,   f_a ∈ R^{D0}
```

x_a 是锚位置（量化后），H_s 是上下文二值哈希网格，Interp 是插值查询；f_s 是共享特征（维度 Ds = ⌊ρD0⌉，ρ 是共享比例），f_i 是单独熵编码的个体特征（维度 Di = D0 − Ds），⊕ 是拼接。聚合维度与原锚特征一致（替换 Scaffold-GS 的 f），但每锚要熵编码的维度从 D0 降到 Di。共享特征本身不做熵编码（第 IV-C 节："the shared feature is not directly entropy-coded"），它的成本藏在哈希网格 H_s 里。

式 (7)（第 IV-C 节，高斯属性生成，继承 Scaffold-GS）：

```
{c_i, r_i, s_i, α_i}^K_{i=1} = F_s(f_a, δ_c, d⃗_c)
```

F_s 是轻量 MLP，δ_c = ||x − x_c||₂ 是点到锚中心的距离，d⃗_c 是归一化方向向量；输出 K 个高片的颜色、旋转、缩放、不透明度。高斯位置 = 锚位置 + offset·l（l 是锚的 scale 向量）。

式 (8)（第 IV-D 节，总损失）：

```
L = L_Scaffold + λ_m·L_m + λ_e·(L_entropy + L_hash) / (N(D_i + 6 + 3K))
```

txt 抽取时分式断行，容易误读成 λ_e 只乘 L_entropy；已对照 PDF 第 8 页：λ_e 乘整个分式，正文也写明"the normalized bit-consumption proxy Lentropy + Lhash, scaled by λe and normalized by the number of per-anchor coded elements"。失真项 L_Scaffold 是 Scaffold-GS 的 SSIM/L1 加缩放正则，L_m 是锚掩码正则（控制锚数量）；率项是熵损失加哈希网格损失，按每锚编码元素数 N(D_i + 6 + 3K) 归一化（特征 D_i 维 + scale 6 维 + offset 3K 维）。

式 (9)（第 IV-D 节，熵损失）：

```
L_entropy = Σ_{i=1}^{N} Σ_{z_i∈{f_i, l_i, o_i}} Σ_{j=1}^{D_z} (−log2 p(ˆz_{i,j}))
```

对全部锚的全部编码属性（特征、scale、offset）逐元素求负对数似然，D_z 是对应属性的维度（D_i、6 或 3K）。

### 信号设计（重点）

本文有两个"信号"，分开填：

**信号一：COHS 的根特征条件（熵编码上下文）**

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | 锚特征余弦相似度选出的"根锚"的量化解码特征，作为叶锚熵编码的条件；链接本身（谁是谁的根）由训练期检索决定 |
| 哪一端算得出 | 链接在编码期由码流里的链接向量直接给出（训练期搜好）；条件特征（量化根特征）编码端解码端都可得——根先解码 |
| 有无侧信息 | 有。活跃锚掩码 + 链接向量共 0.19 MB，占最终 4.99 MB 码流的 3.74%（Table V） |
| 训练期还是编码期 | 链接在训练期建立（30000 迭代后两次更新）；编码期只按表执行 |
| 和渲染质量的距离 | 代理量：条件熵模型降低预测误差 → 降低码率，间接通过率失真训练影响质量；不直接测渲染质量 |

**信号二：SFA 的共享特征（表示级全局先验）**

| 问题 | 答案 |
| --- | --- |
| 信号是什么 | 锚位置查询上下文哈希网格得到的共享特征 fs，代表场景级低频公共分量；与个体特征拼成聚合特征参与高斯生成 |
| 哪一端算得出 | 编码端解码端都能算：两端持有同一个哈希网格，按锚位置插值即可 |
| 有无侧信息 | 共享特征本身不入码流，但哈希网格 H_s 本身要传输（HAC 框架本来就传；Fig. 8 中 Hash 占 CRP-GS 码流 31.1%）。论文未说明是否与熵模型的哈希网格共用同一张（Fig. 2 只画了一个 contextual hash grid，第 V-A 节说网格参数"configured following [14]"即沿用 HAC 配置） |
| 训练期还是编码期 | 哈希网格在训练期学出 |
| 和渲染质量的距离 | 这条不经过熵模型，直接进渲染管线（式 (7)），是对渲染质量的直接干预；同时通过 H(f_i \| f_s) 的降低间接省码率（第 IV-C 节末段） |

### 与码流组织相关的细节（压缩类论文必填）

- **渐进层划分：没有。** 整个码流是单档的，换码率靠改 λ_e 重新训练一个模型（λ_e 从 1×10⁻⁴ 到 4×10⁻³，第 V-A 节），Fig. 5 的 RD 曲线上每个点对应一次独立训练。论文不涉及可截断码流。
- **文件布局（第 IV-B1 节 bitstream format 段）**：码流里比 HAC 多两样结构数据——活跃锚掩码 m ∈ {0,1}^{N0}（N0 是掩码前锚数，按锚序号存 0/1）和链接向量 L ∈ ({1,…,Na} ∪ ∅)^{Na}（Na 是保留的活跃锚数；非空项存所选根的活跃序号，空项用哨兵值 −1）。解码器由 m 和 L 确定性恢复根/叶/自由划分：L[i]≠∅ 即叶锚且 L[i] 是它的根；被任何叶指向的是根；L[i]=∅ 且不被指向是自由锚。候选集 C 和训练期的中间依赖指派不进码流。
- **解码端重算哪些量**：几乎不重算。划分从码流元数据读出；熵模型参数（µ、σ）由网络从哈希网格特征/根特征前向算出。这与我们"量化步长、编码上下文全部从已解码数据重算"的契约完全不同。
- **侧信息**：掩码 0.03 MB（0.60%）+ 链接向量 0.16 MB（3.14%）= 0.19 MB（3.74%），相对 4.99 MB 的示例码流（Table V）；这笔开销已计入全部 RD 数字。此外锚几何用标准点云编解码器压缩（第 V-C 节）。
- **可截断性**：不存在。作者只保证依赖顺序浅（一层），"make partial or random access less flexible"是他们拒绝深层级的理由之一，但单层内仍是全量解码。
- **解码顺序**：根 → 自由 → 叶，组内按锚序号（第 IV-B2 节）。

## 三学：证明与包装（创新怎么立住）

### 实验口径

- **指标**：PSNR 为主（RGB 空间），辅 SSIM、LPIPS；体积 = 熵编码后码流总文件大小，单位 MB（第 V-A 节）。多码率点用 λ_e 扫出来。
- **BD 指标协议（第 V-A 节）**：先删被支配的 RD 点，剩余曲线用保形 PCHIP 插值拟合（率轴取对数文件大小）；BD-rate 在公共 PSNR 区间上积分对数文件大小，BD-PSNR 在公共对数大小区间上积分 PSNR；没有有效公共区间就记"–"且不计入平均。这个协议比多数 3DGS 压缩论文只报两个工作点严谨。
- **数据集**：Mip-NeRF360、Tanks&Temples、DeepBlending、BungeeNeRF 四个公开数据集（第 V-A 节）。全部是公开学术场景，不含大规模航拍数据。
- **基线**：13 个方法，覆盖原始 3DGS、Scaffold-GS、剪枝类（EAGLES、LightGaussian）、向量量化类（Compressed3D、Navaneet et al.）、熵编码类（HAC、HAC++、HEMGS、Morgenstern et al.）、自回归类（ContextGS）、渐进类（PCGS，AAAI 2026）（第 V-B 节）。没比 CAT-3DGS（引用了但表里没有）。
- **训练配置（第 V-A 节）**：PyTorch，单张 RTX 3090；30000 迭代稳定期 + 10000 迭代 COHS 期；候选池 M=10、τ_min=0.8、Δτ=0.05、依赖更新两次；SFA 共享比例 40%（HAC 的 50 维锚特征拆成 20 维共享 + 30 维单独编码）；λ_m=5×10⁻⁴，λ_e∈[1×10⁻⁴, 4×10⁻³]；初始量化步长 f_i 为 1.0、l 为 0.001、o 为 0.2。基线训练预算"follow the corresponding source papers"。

### 主结果表抄录（注明表号）

**Table I（共 14 行，此处节选 9 行：原始参照 + 全部锚级/渐进竞品 + 本文两点；原表还有 EAGLES、LightGaussian、Compressed3D、Morgenstern et al.、Navaneet et al. 五行剪枝/量化类方法）。** 每数据集四列：PSNR(dB)↑ / SSIM↑ / LPIPS↓ / SIZE(MB)↓。数字照抄 txt 与 PDF 核对结果。

| 方法 | Mip360 PSNR | Mip360 SSIM | Mip360 LPIPS | Mip360 MB | T&T PSNR | T&T SSIM | T&T LPIPS | T&T MB | DB PSNR | DB SSIM | DB LPIPS | DB MB | Bungee PSNR | Bungee SSIM | Bungee LPIPS | Bungee MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3DGS | 27.49 | 0.813 | 0.222 | 744.7 | 23.69 | 0.844 | 0.178 | 431.0 | 29.42 | 0.899 | 0.247 | 663.9 | 24.87 | 0.841 | 0.205 | 1616 |
| Scaffold-GS | 27.50 | 0.806 | 0.252 | 253.9 | 23.96 | 0.853 | 0.177 | 86.50 | 30.21 | 0.906 | 0.254 | 66.00 | 26.62 | 0.865 | 0.241 | 183.0 |
| HAC | 27.53 | 0.807 | 0.238 | 15.26 | 24.04 | 0.846 | 0.187 | 8.10 | 29.98 | 0.902 | 0.269 | 4.35 | 26.48 | 0.845 | 0.250 | 18.49 |
| ContextGS | 27.62 | 0.808 | 0.237 | 12.68 | 24.20 | 0.852 | 0.184 | 7.05 | 30.11 | 0.907 | 0.265 | 3.45 | 26.90 | 0.866 | 0.222 | 14.00 |
| HAC++ | 27.60 | 0.803 | 0.253 | 8.34 | 24.22 | 0.849 | 0.190 | 5.18 | 30.16 | 0.907 | 0.266 | 2.91 | 26.78 | 0.858 | 0.235 | 11.75 |
| HEMGS | 27.68 | 0.809 | 0.239 | 12.52 | 24.41 | 0.854 | 0.183 | 6.13 | 30.24 | 0.909 | 0.258 | 3.67 | – | – | – | – |
| PCGS | 27.69 | 0.808 | 0.237 | 12.64 | 24.27 | 0.850 | 0.184 | 6.31 | 30.14 | 0.906 | 0.264 | 3.71 | 27.01 | 0.873 | 0.206 | 14.07 |
| Ours-lowrate | 27.73 | 0.836 | 0.210 | 11.98 | 24.07 | 0.836 | 0.210 | 3.96 | 29.87 | 0.898 | 0.287 | 1.95 | 26.44 | 0.846 | 0.252 | 13.75 |
| Ours-highrate | 27.94 | 0.812 | 0.232 | 18.36 | 24.55 | 0.850 | 0.188 | 7.00 | 30.41 | 0.906 | 0.260 | 5.53 | 27.05 | 0.873 | 0.210 | 21.84 |

注：照 PDF 核对，Ours-lowrate 在 Mip360 与 T&T 的 SSIM/LPIPS 数值（0.836/0.210）原文即相同。两点定义：Ours-lowrate 用 λ_e=4×10⁻³（重码率惩罚），Ours-highrate 用 λ_e=1×10⁻⁴（重保真）（第 V-A 节）。注意 Ours 的两个工作点总体积都大于 HAC++ 同数据集的体积——它的优势区间在中间码率，不在极端低码率。

**Table II 全抄（BD-rate %↓ / BD-PSNR dB↑，相对五个基线）**：

| 数据集 | vs HAC 率 | vs HAC 质 | vs ContextGS 率 | vs ContextGS 质 | vs HAC++ 率 | vs HAC++ 质 | vs HEMGS 率 | vs HEMGS 质 | vs PCGS 率 | vs PCGS 质 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mip-NeRF360 | -39.28 | +0.334 | – | +0.204 | -1.97 | +0.047 | -8.99 | +0.050 | -14.32 | +0.083 |
| Tanks&Temples | -56.90 | +0.586 | -52.41 | +0.420 | -24.31 | +0.208 | -6.82 | +0.035 | -36.02 | +0.277 |
| DeepBlending | -44.25 | +0.591 | -30.78 | +0.208 | -2.30 | +0.045 | -8.02 | +0.054 | -23.76 | +0.171 |
| BungeeNeRF | -27.06 | +0.595 | +15.24 | -0.147 | +32.56 | -0.282 | – | – | +41.96 | -0.287 |
| 平均 | -41.87 | +0.526 | -22.65 | +0.171 | +0.99 | +0.005 | -7.94 | +0.046 | -8.04 | +0.061 |

关键读法：对 HAC、ContextGS 是全面胜；对 HAC++ 平均只打平（+0.99% 码率、+0.005 dB），且 BungeeNeRF 上明显输（+32.56% 码率、-0.282 dB）；正文第 V-C 节自己承认 "HAC++ attains a better extreme low-bitrate point on Mip-NeRF360 and BungeeNeRF"。Table I 里 HAC++ 的单点体积仍全面最小，CRP-GS 的赢面靠 BD 区间积分（中高码率段）。定性对比（Fig. 6，第 V-C 节）：counter 场景相对 HAC 从 8.98 MB 降到 7.51 MB（约 -16.4%）且 PSNR 反升（30.81 → 31.16 dB）；truck 场景从 8.80 MB 降到 6.22 MB（约 -29.3%）。

**Table III 全抄（逐组件消融，三场景，模型体积单位 MB）**：

| 方法 | Train PSNR | Train MB | Room PSNR | Room MB | DrJohnson PSNR | DrJohnson MB |
| --- | --- | --- | --- | --- | --- | --- |
| 完整模型 | 22.87 | 4.67 | 32.01 | 3.43 | 29.72 | 3.71 |
| 去 SFA | 22.84 (-0.03 dB) | 5.73 (+22.7%) | 31.91 (-0.10 dB) | 3.80 (+11.0%) | 29.70 (-0.02 dB) | 3.91 (+5.4%) |
| 去 SFA 和 COHS | 22.55 (-0.32 dB) | 5.74 (+22.9%) | 31.64 (-0.37 dB) | 3.94 (+15.0%) | 29.60 (-0.11 dB) | 3.93 (+6.0%) |
| 再去锚位置编码 AC | 22.55 (-0.32 dB) | 6.59 (+41.1%) | 31.64 (-0.37 dB) | 4.70 (+37.2%) | 29.60 (-0.11 dB) | 4.51 (+21.7%) |

**Table IV 全抄（COHS 链接策略替换消融）**：

| 方法 | PSNR (dB) | 体积 (MB) |
| --- | --- | --- |
| 完整方法（余弦相似度链接） | 22.41 | 5.89 |
| 随机链接 | 22.35 | 6.41 |
| 最近邻（距离）链接 | 22.28 | 6.22 |
| 无链接 | 22.30 | 6.13 |

**Table V 全抄（COHS 结构侧信息开销，相对 4.99 MB 示例码流）**：

| 项目 | 大小 (MB) | 占比 (%) |
| --- | --- | --- |
| 活跃锚掩码 | 0.03 | 0.60 |
| 链接向量 L | 0.16 | 3.14 |
| 合计 | 0.19 | 3.74 |

### 消融设计（注明表号）

值得学走的对照设计有四个：

1. **渐进开关（Table III）**：完整 → 去 SFA → 去 SFA+COHS → 再去 AC，三级逐层剥离，同时报 PSNR 和体积。设计干净处：AC 不参与渲染，所以后两行 PSNR 相同、只有体积变，符合机制预期。
2. **替换实验（Table IV）**：把"按特征相似度链接"这一个决策换成随机链接、最近邻链接、不链接三种对照，直接验证"特征对应优于空间邻近和随机"。这是对核心主张最对症的一张表——距离链接（22.28 dB）甚至比无链接（22.30 dB）还差一点，说明硬套空间邻近条件不但无益、可能有害。
3. **比例扫描（Fig. 7a）**：共享特征比例 80% / 40% / 0% 三档，40% 最好；论文解释高比例"overshare features and reduce distinctiveness"，低比例吃不到共享表示的好处。
4. **开销透明化（Table V + Fig. 8 + 运行时段）**：把新机制的码流代价（3.74%）、码流构成（Fig. 8：CRP-GS 特征占 31.2% vs HAC 41.4%、锚占 9.0% vs 19.5%，31.25 dB/4.99 MB vs 31.16 dB/7.25 MB；txt 抽取的饼图部分标签不完整，完整构成以 PDF 图为准）、解码时间（30.257 s vs HAC 29.186 s）和渲染延迟（15.4 ms/帧 vs 11.0 ms/帧，+4.4 ms，约 +40%，第 V-C 节 Runtime 段）全部亮出来。渲染变慢的原因写明是 SFA 渲染期查哈希网格加拼接。

### 贡献列表（原文照抄 + 逐条标注）

1. "We propose a correspondence-oriented hierarchical structure that leverages cross-anchor correspondence to construct accurate contextual references. This enables a subset of decompressed anchors to serve as priors for decoding others, significantly improving rate-distortion performance." —— [机制] COHS：按特征对应建根叶链接，已解码根特征做叶锚条件熵编码的先验。
2. "We propose a shared feature aggregation approach that introduces global shared priors to suppress redundant information in individual anchor features, thereby enhancing structural compactness and coding efficiency." —— [机制] SFA：锚特征拆共享分量（哈希网格查询）+ 个体分量（熵编码），共享分量直接进渲染。
3. "Comprehensive evaluations across multiple 3DGS compression benchmarks demonstrate that the proposed CRP-GS framework provides a favorable overall rate-distortion trade-off, offering an average of over 30% storage reduction compared to the baseline HAC while achieving competitive visual quality across datasets." —— [实验] 多基准评测；30% 是对比 HAC 的存储节省（Table II 的 BD 口径给到 41.87%）。

### 讲故事方式

主线一句话："锚间冗余还有一层没被利用——特征相似但空间远离的锚对，把它们的对应关系变成条件熵编码的上下文，再把场景公共分量搬到共享哈希特征里。"最有力的一张图是 Fig. 3：它先用量化证据（CDF 长到 0.5 场景跨度）把"空间邻近假设不成立"钉住，后面的 COHS 才有立足点；配合 Table IV 的替换消融（距离链接反而更差），观察、方法、验证三步对得上。叙事上它把自己定位成"第三条设计轴"：HAC 是空间上下文、ContextGS 是空间层级、CRP-GS 是表示级对应（第 IV-A 节末段专门用一段话区分各种先验的来源和用途）。弱点也摆在明面上：Table II 里 BungeeNeRF 输 HAC++ 照登，运行时段照写渲染 +40%。

## 四学：原文关键段翻译（深读层必填）

### 方法节核心段（选 2 段）

> **原文（第 IV-B2 节 COHS Entropy Coding）**："Once anchor context dependencies are established, each root anchor provides a contextual prior for encoding its corresponding leaf anchor, as illustrated in Fig. 2. Suppose the i-th and j-th anchors form a root-leaf pair, with their entropy-coded anchor features denoted as f(r)_i and f(l)_j, respectively. Since the root anchor is decoded before its dependent leaf anchor, the conditional entropy model uses the quantized root feature ˆf(r)_i, which is the root representation available at both the encoder and decoder. The context modeling of the leaf anchor feature f(l)_j can then be formulated as: µ(l)_j = MLP([ˆf(r)_i; x_i; x_j]), σ(l)_j = MLP(f^h). … During entropy decoding, anchors are processed in a fixed order. The decoder first reconstructs the COHS partition from the active-anchor mask m and the link vector L defined in the bitstream-format paragraph. Within each group, anchors follow their stored anchor-index order. The decoder first decodes root anchors with the base entropy model and caches their quantized decoded features, then decodes free anchors with the same base model, and finally decodes leaf anchor features with the conditional leaf entropy model in Eq. (4). Since both encoder and decoder condition on the same quantized decoded root feature ˆf(r)_i, the conditioning information is strictly identical on both sides and is available before decoding the dependent leaf anchor."
>
> **译文**：锚上下文依赖建立之后，每个根锚为编码其对应叶锚提供上下文先验（如图 2）。设第 i 个锚与第 j 个锚构成根叶对，其熵编码特征分别记为 f(r)_i 和 f(l)_j。由于根锚先于其依赖的叶锚解码，条件熵模型使用量化后的根特征 ˆf(r)_i——这是编码端与解码端都能拿到的根表示。叶锚特征 f(l)_j 的上下文建模即：均值由 [量化根特征；根位置；叶位置] 拼接过 MLP 得到，标准差由哈希网格特征过 MLP 得到。……熵解码时锚按固定顺序处理。解码器先由码流里的活跃锚掩码 m 和链接向量 L 重建划分，组内按存储的锚序号顺序处理；先用基础熵模型解码根锚并缓存其量化解码特征，再用同一基础模型解码自由锚，最后用条件叶熵模型（式 (4)）解码叶锚特征。由于编码端与解码端的条件都是同一个量化后的根特征，两侧的条件信息严格一致，且在解码依赖它的叶锚之前就已就绪。

> **原文（第 IV-B1 节 Bitstream format）**："After the last renewal phase, COHS introduces lightweight structural side information into the final bitstream. The bitstream stores an active-anchor mask m ∈ {0,1}^{N0} over the pre-masking anchor-index range and a final link vector L ∈ ({1,…,Na} ∪ {∅})^{Na} over the active-anchor index range, where N0 is the number of anchors before masking and Na is the number of retained active anchors. … For an active anchor i, L[i] ≠ ∅ denotes a leaf anchor and L[i] identifies its root. An active anchor r is a root anchor iff there exists an active anchor i such that L[i] = r. An active anchor is free iff L[i] = ∅ and it is not referenced by any active anchor. Given m and L, the decoder can deterministically recover the root, leaf, and free-anchor partition. The candidate set C and the intermediate dependency assignments generated during renewal are used only for training-time structure construction and are not stored in the final bitstream. Therefore, deployment does not require online candidate retrieval, all-pair feature search, or dependency renewal."
>
> **译文**：最后一次依赖更新之后，COHS 把轻量的结构性侧信息写进最终码流。码流存储两部分：掩码前锚序号范围上的活跃锚掩码 m ∈ {0,1}^{N0}（N0 是掩码前的锚总数），以及活跃锚序号范围上的最终链接向量 L ∈ ({1,…,Na} ∪ {∅})^{Na}（Na 是保留的活跃锚数）。对活跃锚 i，L[i] 非空表示它是叶锚，L[i] 给出它的根；活跃锚 r 是根锚，当且仅当存在活跃锚 i 使 L[i]=r；活跃锚是自由锚，当且仅当 L[i] 为空且不被任何活跃锚引用。给定 m 和 L，解码器可以确定性地恢复根、叶、自由的划分。候选集 C 和更新过程产生的中间依赖指派只用于训练期的结构构建，不写进码流。因此部署不需要在线候选检索、全对特征搜索或依赖更新。

### 主结果表述段（选 1 段）

> **原文（第 V-C 节第一段）**："As demonstrated in Table I, CRP-GS achieves a favorable storage-quality trade-off compared with recent 3DGS compression approaches, including HAC++ [16], ContextGS [13], HEMGS [15], and PCGS [64]. Specifically, CRP-GS reduces storage requirements by 98.39% over the original 3DGS framework and by 95.31% relative to the anchor structure Scaffold-GS implementation. Ours-highrate improves PSNR over HAC++ across the evaluated datasets and yields competitive LPIPS values. At the low-bitrate regime, the trade-off becomes dataset-dependent: on Tanks&Temples and DeepBlending, Ours-lowrate produces substantially smaller bitstreams with only minor PSNR reductions, whereas HAC++ attains a better extreme low-bitrate point on Mip-NeRF360 and BungeeNeRF. CRP-GS is expected to be most beneficial when reliable cross-anchor correspondences and reusable scene-level priors exist; the improvement may be less pronounced otherwise."
>
> **译文**：如表 I 所示，CRP-GS 与近期 3DGS 压缩方法（含 HAC++、ContextGS、HEMGS、PCGS）相比取得了有利的存储-质量折中。具体地，CRP-GS 相对原始 3DGS 省 98.39% 存储，相对锚结构的 Scaffold-GS 实现 省 95.31%。Ours-highrate 在所有评测数据集上 PSNR 高于 HAC++，LPIPS 有竞争力。低码率区间则因数据集而异：在 Tanks&Temples 和 DeepBlending 上，Ours-lowrate 以很小的 PSNR 代价换来了显著更小的码流；而在 Mip-NeRF360 和 BungeeNeRF 上，HAC++ 的极低码率点更好。当场景中存在可靠的跨锚对应和可复用的场景级先验时，CRP-GS 预期收益最大；否则提升可能不明显。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | "上下文来源从空间邻近换成特征对应"在 3DGS 锚编码里是新轴（ContextGS/CAT/HEMGS 都没做），Fig. 3 观察加 Table IV 替换消融的论证也完整；但本质是把学习图像压缩里现成的"内容自适应非局部上下文"思路（其引文 [56] 即自家工作）移植到锚编码，机制上是条件熵模型换条件信号源，单层根叶设计也偏保守。SFA 与 Scaffold/HAC 的哈希网格一网两用，属于低成本组合创新 |
| 证据强度 | 高 | 4 个公开数据集、13 个基线、14 行大表，BD 指标用了去支配点 + PCHIP 保形插值的完整协议；消融覆盖逐组件开关（Table III）、替换对照（Table IV）、比例扫描（Fig. 7a）、码流构成（Fig. 8）、侧信息开销（Table V）、运行时；诚实报告 BungeeNeRF 输 HAC++ 和渲染延迟 +40%。未见开源代码声明，暂无法复核 |
| 对期刊版的威胁度 | 中 | 撞的是"锚级条件熵建模"这一大类：审稿人会把 CRP-GS 当最新 TIP 竞品要求对比。但不撞我们的三条核心主张——它没有渐进分层码流（单档、换码率重训）、有 3.74% 结构侧信息（违反我们零侧信息契约）、没有内容自适应量化。它的条件方向（跨锚特征相似度）与我们的（同锚粗层到细层）正交。真正要防的是它把 PCGS（AAAI 2026 渐进压缩）拉进基线并压住——渐进方向已有强论文，我们的渐进主张必须与 PCGS 正面区分 |

## 对 DCCA-GS 的可借鉴点

1. **侧信息与码流构成的透明化报告格式（Table V + Fig. 8 + 运行时段）**：学它的呈现方式而不是机制——我们写期刊版时，可以给"零侧信息"主张配一张同类对照：把 CRP-GS 的掩码+链接向量 0.19 MB（3.74%，Table V）作为"结构先验入码流"的代价实例，我们的分组/层归属全部解码端重算、该项为 0。不用跑新实验，整理现有 total_MB 各字段占比即可。
2. **"空间邻近条件可能有害"的反证（Table IV：距离链接 22.28 dB 差于无链接 22.30 dB）**：这个结论可以反过来支撑我们层条件熵编码的设计合理性检查——如果未来有人建议给复合条件表加空间邻居分组，这条是现成的反面证据（不同框架，需小实验确认后再引用）。
3. **特征对应信号本身（COHS）——先小实验再决定**：它的跨锚条件与我们层条件熵编码方向正交，理论上可叠加（同一锚的粗层上下文 + 相似锚的上下文）。但要清醒：COHS 依赖学习式条件熵模型（MLP 预测均值方差），而我们已有明确负结果——学习式概率熵模型在生产 A/B 中零增益、MLP 拟合码表 +35-38%。若要试，最低成本路径是把"特征相似度分组"作为复合条件表的分组键（统计频率表，不是学习式熵模型），在 HAC++ 底座上跑一组 A/B 看分组信息能否解码端重算；重算不出来就违反零侧信息契约，直接放弃。
4. **诚实报告弱点的写法**：它在正文照登 BungeeNeRF 对 HAC++ 的 +32.56% 码率和渲染 +40% 延迟，反而增强了 Table I/II 主结论的可信度。期刊版我们报告阶梯对齐量化的适用边界时可以采用同样的写法。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 量化怎么定 | 固定初始量化步长（特征 1.0 / scale 0.001 / offset 0.2，第 V-A 节），码率档位靠 λ_e∈[1×10⁻⁴, 4×10⁻³] 扫描，每个码率点单独训练一个模型（Fig. 5 每点一次训练） | B3 阶梯对齐量化训练 + 嵌套质量阶梯：基层传全锚点粗量化值，增强层逐层补精度，一个码流携带多个画质档 | 它换一档码率要重训一次；我们单文件多档。这是主创新的差异化空间，也说明"多档单文件"在 TIP 级竞品里仍是空白 |
| 熵编码上下文 | COHS：特征余弦相似度定一对一根叶链接，叶锚均值用 [量化根特征；根位置；叶位置] 过 MLP 预测（式 (4)），标准差只看哈希网格；根/自由锚走 HAC 式基础模型（式 (5)）；SFA 再把哈希共享特征拼进锚表示 | （贡献组×粗值幅度桶）复合条件表（分组频率统计）+ 层条件熵编码（细层用已解码粗层符号当上下文）+ 二值分解区间编码 | 两家的条件轴正交：它跨"特征相似、空间无关"的锚，我们跨"同一锚的粗层到细层"。它是学习式条件熵模型（需训练 MLP），我们是统计频率表（解码端可重算）；我们已有学习式熵模型零增益的负结果，两个框架不能直接互换条件机制 |
| 渐进/可截断 | 该论文不涉及渐进码流：单档码流、全量解码；单层根叶依赖是它对"访问灵活性"的让步设计（第 IV-B1 节明说深层级会伤害随机访问） | 单文件渐进分层码流是主创新：任意档是文件连续字节前缀，真实可截断，bit-exact 验证 | 它给了一个可引用的论据：TIP 级工作也承认深层依赖伤害访问灵活性——支持我们把渐进做成"层内浅依赖"的设计选择。同时它表里的 PCGS（AAAI 2026）说明渐进压缩方向已有论文发表在先，期刊版必须与它正面比较 |
| 侧信息 | 有：活跃锚掩码 + 链接向量 0.19 MB，占码流 3.74%（Table V）；哈希网格整体入码流（Fig. 8 中 Hash 占 31.1%）；解码端不重算结构 | 零侧信息硬约束：量化步长、编码上下文全部从已解码数据重算，共享一个 8→32→3 小 MLP，结构信息（分组、层归属）解码端可得 | 对照证据：CRP-GS 为拿条件熵编码收益支付了 3.74% 结构元数据，我们主张同一类收益不付这笔钱。写期刊版时这是现成的对比行 |
| 锚位置编码 | 锚几何用标准点云编解码器压缩（第 V-C 节）；消融中去掉锚位置编码体积涨 21.7%–41.1%（Table III） | xyz 字段计入 total_MB 口径（各字段比特数求和换算 MB），编码方式按 HAC++ 底座 | 部分重叠：它把锚位置编码列为消融项并给出量化收益，说明位置编码是锚级框架里不小的一块；我们的字段级体积占比报告可以按同样的粒度写 |
