# 《Single-Event Upsets in 3D Gaussian Splatting Rendering: Bit-Level Criticality, Spatial Extent, and a Parallel Support Guard》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2606.21791 |
| 发表 | arXiv 预印本（v1，2026-06-19，cs.GR），Bahçeşehir University |
| 阅读材料 | 全文（.lit-cache/txt/2606.21791.txt；txt 混有控制字符已清洗，关键表格数字已用 pymupdf 对 PDF 逐项核对） |
| 与 DCCA-GS 的关系 | 可靠性方向的评测参考，不是压缩竞品；它测出的字段/比特敏感度排序是我们分字段粗化档位的实证依据 |

> 一句话定位：这篇论文不做压缩，它用 380 万次受控单比特翻转测出 3DGS 六个参数字段在 fp32/fp16/bf16 下每个比特位的敏感度排序——对数缩放的符号位翻一位就能让一个高斯撑满最多 75.7% 的画面——据此给出"逐图元截断到训练取值范围"的防护（每帧 76 微秒），并证明该防护下任何单比特翻转都无法造成覆盖全帧的损坏。对我们而言，它是一份现成的"哪个字段、哪段比特要留精度"的实测依据。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾
它认为 3DGS 正在进入硬件不可靠的部署环境（航天航空处理器、移动与机器人边缘设备、有"静默数据损坏"记录的渲染集群），而它的容错性质没人测过。原话（第 1 节）："Whether this redundancy makes the representation robust, or instead concentrates the risk into a few rare failures, has not been established."——几十万图元的冗余到底是让表示更抗造，还是把风险集中到少数罕见故障上，只能靠直接测量回答。神经网络的比特翻转攻击已有成熟研究（几比特就能毁掉分类器），但 3DGS 结构不同：权重被整个输出共享，高斯只影响画面一小块区域。

### 支撑观察（直觉 / 数据 / 数据来源）
观察全部来自自己的故障注入实验，不是作者直觉：第 5.1 节图 1 的"字段 × 比特位"平均损坏脚印热图几乎全黑（绝大多数单比特翻转一个像素都改变不了），只有少数格子亮——对数缩放符号位最亮，其次它的高指数位，再次是位置均值和直流颜色的高指数位；表 2 给出逐字段汇总数字。"比特翻转真实存在"的前提引自第 2 节的两项工业研究（生产集群静默数据损坏）和 GPU 中子束照射实验（文献 [18][3]）。

### 显然的路为什么没走
- 没有沿用 DNN 容错的现成工具（指令级注入器 SASSIFI/NVBitFI、张量级 PyTorchFI）：3DGS 不是神经网络，每个比特属于一个有名有姓的几何量，值得按"字段 × 比特位"逐格测，而不是当一层黑盒。
- 防护没有选纠错码或三模冗余：表 3 显示它们分别要约 1.3 倍和 3 倍内存，而"灾难性翻转恰好就是把值推出训练取值范围的那些翻转"（第 4 节），所以零内存开销的截断就够。
- 指标没有只用全局 PSNR：第 3 节明说全帧 PSNR 区分度差，几十万图元里坏一个几乎不动全局平均，所以主打局部指标"损坏脚印"。

## 二学：机制（洞察怎么变成方法）

### 核心流程
**训练期**：用开源 gsplat 训练 4 个 NeRF-synthetic 场景（chair、lego、ficus、hotdog），带致密化。训练后做一次性统计（算法 1 第 1–3 行）：对每字段每分量取全部图元的最小/最大值，得到"支撑盒"，此后不再变。

**测量期（相当于编码期的角色）**：故障点是四元组（字段，图元，分量，比特）。六个字段是部署检查点存储的全部内容：位置均值（3 维）、对数缩放（3 维，渲染时作用 exp）、四元数（4 维，渲染时归一化）、不透明度 logit（1 维，渲染时作用 logistic 函数）、球谐直流颜色项（3 维）、球谐高阶项（其余）。注入的是存储值本身（优化空间的对数缩放、不透明度 logit），因为那才是显存里的东西（第 3 节）。每个注入五步：把存储值重解释成同宽无符号整数、翻指定位、重解释回浮点；写入 GPU 驻留的工作副本；一次批式光栅化调用渲染 K 个留出视角（K 值论文未报告）；GPU 上算全部指标；恢复原值。覆盖 4 场景 × 6 字段 × 全部比特位 × 3 精度（fp32：符号位 31、指数位 23–30、尾数位 0–22；fp16：5 位指数 10 位尾数；bf16：8 位指数 7 位尾数），每格采样数千次，共 380 万次、5.3 GPU 小时、59% 利用率（第 3 节、第 5 节开头）。

**部署期（防护运行）**：每帧渲染前逐图元逐字段逐分量检查：值不在支撑盒内或为非有限值就截断到盒内，本节点出界计数加一（算法 1 第 4–12 行）；计数超过健康阈值就向主机报静默数据损坏警报——防护同时是检测器，不许掩盖正在劣化的设备（第 6 节）。防护是节点本地的：sort-first 渲染的每个节点只处理自己拥有的图元，无同步点、无跨节点通信（第 4 节、图 7）。

### 关键公式（按原文抄，注明编号；上标在 txt 抽取中被拍平，恢复依据上下文，原文公式以 PDF 为准）
1. 引理 1（单比特翻转的值扰动，第 4 节）：正常数 θ = (−1)^ε·2^e·(1+m)，尾数宽 p=23。翻尾数位 b：|Δθ| = 2^(e+b−p)——恰好等于该位的权重；翻符号位：Δθ = −2θ；翻指数位 j：数值乘 2^(±2^j)，最高几位指数位直接溢出成非有限值。fp16（p=10）、bf16（p=7）同理。
2. 命题 1（一阶图像扰动界，第 4 节）：I = R(ϕ(θ), ·)，则 ∥ΔI∥∞ ≤ ∥∂R/∂ϕ∥∞ · |ϕ′(θ)| · |Δθ| + O(Δθ²)。对缩放字段 ϕ=exp，|ϕ′(θ)|=e^θ=s，所以相对尺度变化 |Δs|/s = |Δθ|：尾数翻转造成固定相对变化 2^(b−p)，与指数无关；符号翻转把典型训练值 θ≈−3 变成 +3，图元尺寸乘 e^6 ≈ 4×10²。不透明度经 logistic 函数传导但被压在 [0,1]，温和得多。
3. 引理 2（冗余缩放，第 4 节）：中位单次翻转误差 σ²(N) = Θ(N^(−α))，α>0 随场景而定；均值误差不分享这个缩放，被罕见的缩放符号位爆炸钉住。
4. 定理 3（剂量预算，第 4 节）：k 次独立均匀单比特翻转下 E∥ΔI∥² = k·σ²(N)（各次翻转的受影响像素集合期望不相交），首次超阈值 τ 的剂量 kτ = Θ(N^α·τ)。防护移除重尾后这个可加律才成立。
5. 推论 2（分布式清洗周期，第 4 节）：节点每 M 帧重施防护、比特翻转率 λ/位/帧、存储 b=Θ(N) 位时，M ≤ kτ/(λb) = Θ(N^(α−1)·τ/λ)。
6. 定理 1 / 定理 2（第 4 节）：防护在干净模型上是恒等映射，出盒翻转被完全纠正（定理 1）；加防护后每图元参数都在盒内，投影尺度至多为 exp(h)（训练出的最大对数缩放），故任何单比特翻转都无法造成覆盖全帧的损坏（定理 2）。
7. 可靠性模型（第 5.9 节）：一帧在 k 次独立翻转后发生灾难的概率写成 1−(1−p_c)^k，p_c = 0.634%，与实测剂量曲线吻合。

### 信号设计（重点）
| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 比特级临界度：一次单比特翻转造成的"损坏脚印"（相对干净渲染颜色变化超过 1/255 的像素占比，定义 1）与灾难判定（非有限输出或脚印超过帧的 1%，定义 2） |
| 哪一端算得出（编码端/解码端/仅训练期） | 实测版需要渲染（编码端/部署端都行）；闭式界（引理 1+命题 1）和第 5.11 节的分类器只需要字段与比特位置，任何一端无需渲染就能算 |
| 有无侧信息 | 无。支撑盒（每字段每分量 min/max）是训练后模型自身的统计，不是额外学习的网络 |
| 训练期还是编码期 | 训练后一次性统计（支撑盒）+ 训练后一次性大规模测量（临界度地图）；运行期每帧只做截断 |
| 和渲染质量的距离（直接测质量还是代理量） | 直接测渲染输出：局部脚印为主，全局 PSNR/SSIM/LPIPS 为辅；不经过任何代理 |

本文不是压缩论文，无码流组织内容，按规范删去"与码流组织相关的细节"一节。

## 三学：证明与包装（创新怎么立住）

### 实验口径
- 训练与测量都用 gsplat；场景为 4 个 NeRF-synthetic 场景，规模与干净保真见表 1（chair 134,826 图元 / 39.22 dB；lego 121,691 / 30.70 dB；ficus 125,606 / 31.99 dB；hotdog 63,452 / 38.50 dB）。另用真实场景 Tanks-and-Temples truck（2,056,645 图元）复验（第 5.6 节）。
- 体积/码率：不涉及（不压缩）。注入规模：380 万次单比特（第 5 节开头），缩放律子实验另有 1900 万次（第 5.7 节），带防护复跑 768,000 次（第 5.5 节）。
- 对比对象：DNN 容错文献的结论（对比性论述）+ 同一故障网格上的四种备选防护（表 3）。
- 训练配置：gsplat 致密化，每场景数分钟（附录 B）。

### 主结果表抄录
**表 2（fp32 逐字段单比特翻转严重度，按场景和比特汇总；脚印为改变像素百分比）——全表 6 行照抄：**

| 字段 | 中位脚印 | p95 | p99 | 最大 | 均值 | 灾难率（%，95% Wilson 置信区间） |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 0.000 | 0.015 | 0.05 | 1.8 | 0.004 | 0.004 [0.002, 0.008] |
| log-scale | 0.000 | 0.237 | 6.16 | 99.4 | 0.351 | 2.782 [2.719, 2.847] |
| quat | 0.000 | 0.002 | 0.01 | 0.6 | 0.000 | 0.000 [0.000, 0.002] |
| opacity | 0.000 | 0.003 | 0.02 | 2.4 | 0.002 | 0.004 [0.002, 0.007] |
| color (DC) | 0.000 | 0.005 | 0.02 | 1.0 | 0.001 | 1.016 [0.978, 1.056] |
| color (SH) | 0.000 | 0.000 | 0.01 | 0.7 | 0.000 | 0.000 [0.000, 0.002] |

正文补充（第 5.1 节）：单看对数缩放符号位，平均脚印 10.3%、第 99 百分位 75.7%，比任何尾数类高两到三个数量级。

**表 3（同一 fp32 故障网格上各防护措施对照，按场景汇总）——全表 5 行照抄：**

| 防护 | 灾难率（%） | 平均脚印（%） | 代价 |
| --- | --- | --- | --- |
| none | 0.552 | 0.0880 | 0 |
| support guard（全字段截断） | 0.052 | 0.0035 | 1× 内存，约 0.1 ms/帧 |
| selective guard（只截缩放+不透明度） | 0.156 | 0.0035 | 1× 内存，<0.1 ms/帧 |
| ECC sign+exp（只纠符号+指数位） | 0.000 | 0.0002 | 约 1.3× 内存，奇偶校验 |
| full duplication（全量复制） | 0.000 | 0.0000 | 3× 内存，投票 |

**表 4（255×10⁶ 存储位模型在代表性翻转率下两次灾难帧间的平均时间）——全表 3 行照抄：**

| 环境 | 无防护 | 有防护 |
| --- | --- | --- |
| 地面（海平面） | 71 yr | 732 yr |
| 航空（约 10 km） | 86 d | 2.4 yr |
| 低地球轨道 | 3 d | 27 d |

正文关键数字（标节号）：防护消解 90.4% 的灾难翻转，缩放符号位翻转的平均全局 PSNR 从 49.2 dB 提到 65.8 dB，768,000 次带防护翻转最坏脚印 11.68%、残留灾难 470 次，代价 76 微秒/帧 = 单视角渲染的 0.07 倍（第 5.5 节）；无防护吸收 1,000 个同时随机翻转后平均 PSNR 跌破 30 dB，最重剂量（20,000 个）跌到 10.6 dB，有防护同剂量 21.8 dB（第 5.4 节，图 4a）；缩放律：中位单次误差对图元数的对数回归指数 α=2.78（R²=0.982），均值误差指数仅 0.09，防护在最大规模把均值单次误差降 24 倍（第 5.7 节）；临界分类器只用字段+比特位置，ROC 曲线下面积 0.997，加上取值信息 0.999，留一场景交叉验证 0.999（最小 0.998）（第 5.11 节）；真实场景 truck：缩放符号位 p99 脚印 64.0%，随机缩放符号位均值脚印 3.00%，均匀随机单比特灾难率 0.50% 被防护压到 0.000%（第 5.6 节）；防护成本随规模近线性，6,600 万图元（29.9 GB 显存、1240 亿存储位）时 9.66 ms/帧 = 该规模渲染的 18.3%，跳过高阶球谐（每图元 59 个分量中的 45 个）只碰四分之一参数内存，有效读取带宽 763 GB/s；每帧 1,000 次翻转持续 300 帧的故障风暴下，带防护帧时延 22.9 ms 对无防护 20.5 ms（第 5.12 节）。

### 消融设计（注明表号/图号）
- 同预算对照（防护设计空间）：表 3 把全字段截断、只截缩放+不透明度、只纠符号+指数位的 ECC、全量复制放在同一故障网格上比灾难率和代价。
- 跨精度对照：图 3b 按"字段 × 精度"给灾难率，得出"降精度不减少暴露、只挪位置"（第 5.3 节）。
- 规模对照：图 5 把训练模型降采样到不同图元数，分别量"冗余预算"（平均 PSNR 跌破 30 dB 的同时翻转数）和单次缩放符号位脚印，证明预算随数量长、单次严重度不随数量变（第 5.6 节）；第 5.7 节用 1900 万次注入拟合缩放指数。
- 跨场景泛化：真实场景 truck 复验（第 5.6 节，图 6）；分类器留一场景交叉验证（第 5.11 节）；Blackwell 与 Ada（L40S）两代硬件上字段-比特排序一致（第 5.12 节）。
- 分布式实测：2 张物理 GPU 双排名 + NCCL all-gather（2.6 GB/s）实测，无防护缩放符号位翻转污染 2/2 节点、防护限制到 1 个；4 卡 L40S 上 4/4 对 1（第 5.10 节）。
- 已知瑕疵：正文三处数字自相矛盾——第 5.6 节"预算从 6,741 图元的 5,000 增长到 134,826 图元的 5,000"（两端相同，疑为笔误）；第 5.10 节最慢排名时间"从干净场景的 1.42 ms 升到无防护的 1.34 ms"且防护"恢复到"同一组数字；第 5.12 节"平均缩放符号位脚印从 0.0% 降到 99.58%"（顺序疑颠倒）。均按 PDF 核对为原文如此，引用时以图为准。

### 贡献列表（引言贡献段原文照抄 + 逐条标注）
1. "Bit-level criticality. Single-bit SEU risk in 3DGS is highly concentrated. The median upset is perceptually invisible, and the dominant case is the sign bit of the logarithmic scale: flipping it enlarges a single primitive to cover a mean of 10.3% and up to 75.7% of the image (Section 5.1)." —— [机制] 实测结论：3DGS 的单比特风险高度集中，中位翻转感知不可见，主导项是对数缩放符号位。
2. "A predictive perturbation bound. A closed-form bound from the IEEE-754 layout [8] and the splatting activations predicts the per-bit severity ordering, including why the scale and opacity activations make their sign and high exponent bits the dominant ones; it is validated against the data (Section 4, Section 5.2)." —— [机制] 从浮点格式和泼溅激活函数推出闭式扰动界，预测比特严重度排序并经数据验证。
3. "A support guard. The catastrophic upsets are exactly those that push a value outside the trained parameter range. A per-primitive clamp to the per-field support box neutralizes 90.4% of them at a cost of 76 µs per frame, leaves a clean model unchanged, and is proved to make frame-covering corruption impossible under any single-bit upset (Theorem 1, Theorem 2, Section 5.5)." —— [机制] 训练支撑盒逐图元截断防护，带安全性和完备性证明。
4. "Accumulated dose and distributed rendering." —— [工程] 累积剂量实验（至 20,000 个同时翻转）与分布式渲染下防护的节点本地抑制。
5. "Scaling, design space, reliability, and prediction." —— [工程/实验] 缩放律实测、备选防护对照、轨道翻转率下的可靠性估计、免注入的临界比特分类器。

### 讲故事方式
主线一句话："冗余保护典型翻转、不保护最坏翻转；最坏翻转可以预测、也可以用一次截断消掉。"最有力的是图 1（字段 × 比特位脚印热图）：近乎全黑的背景上只有对数缩放符号位一带发亮，一图同时立住"高度冗余"与"风险集中"两个前提，后面的防护和理论都从这张图引出。表 2 是它的数字版，表 3 是防护主张的落点。

## 四学：原文关键段翻译（深读层必填）

### 方法节核心段（第 3 节，选 2 段）
> **原文（第 3 节 Parameter layout）**："A trained model has N primitives. Primitive i carries a mean µi ∈ R3, a logarithmic scale si ∈ R3 (the renderer applies exp), a quaternion qi ∈ R4 (normalized at render time), an opacity logit oi ∈ R (the renderer applies the logistic σ), and spherical-harmonic color coefficients split into a direct-current term c0i ∈ R3 and higher-order terms cNi ∈ R((ℓ+1)2−1×3). These are exactly the six fields a deployed checkpoint stores in memory. We inject into the stored representation, which is the optimization-space value (logarithmic scale, opacity logit), because that is what physically resides in VRAM."
>
> **译文**：训练后的模型有 N 个图元。图元 i 携带一个位置均值 µi（三维实向量）、一个对数缩放 si（三维，渲染器对它作用 exp）、一个四元数 qi（四维，渲染时归一化）、一个不透明度 logit oi（标量，渲染器对它作用 logistic 函数 σ），以及球谐颜色系数——系数分成直流项 c0i（三维）和高阶项 cNi（维数为 ((ℓ+1)²−1)×3）。这正是一个部署检查点存进内存的全部六个字段。我们注入的对象是存储表示，也就是优化空间的取值（对数缩放、不透明度 logit），因为这才是物理上驻留在显存里的东西。

> **原文（第 3 节 Metrics and severity，含定义 1、定义 2）**："For each injection we render K held-out views, composite over white, and compare to the uncorrupted render at the same precision. We record the peak pixel error ∥∆I∥∞, the perceptual distance LPIPS, the structural similarity SSIM, the PSNR over the full frame, and a non-finite flag. Because the representation is redundant, full-frame PSNR is a poor discriminator: a single corrupted primitive among hundreds of thousands barely moves a global average. We therefore lead with two local quantities. Definition 1 (Corruption footprint). The corruption footprint of an injection is the fraction of pixels whose color changes by more than one 8-bit level (1/255) relative to the clean render, averaged over the K views. Definition 2 (Catastrophic upset). An upset is catastrophic if it produces a non-finite render or a corruption footprint above 1% of the frame."
>
> **译文**：每次注入渲染 K 个留出视角，在白底上合成，与同一精度下的无损坏渲染比较。我们记录峰值像素误差 ∥∆I∥∞、感知距离 LPIPS、结构相似度 SSIM、全帧 PSNR，以及一个非有限标志。因为这个表示是冗余的，全帧 PSNR 区分度很差：几十万个图元里一个坏图元几乎动不了全局平均。所以我们主打两个局部量。定义 1（损坏脚印）：一次注入的损坏脚印，是颜色相对干净渲染变化超过一个 8 比特档位（1/255）的像素占比，在 K 个视角上取平均。定义 2（灾难性翻转）：一次翻转若产生非有限的渲染结果，或损坏脚印超过帧的 1%，即为灾难性。

### 主结果表述段（第 5.5 节）
> **原文（第 5.5 节）**："We re-run the campaign on the same fault grid with the support guard of Theorems 1 and 2 enabled. The guard neutralizes 90.4% of catastrophic upsets, and for the dominant scale sign-bit upsets it raises the mean global PSNR from 49.2 dB to 65.8 dB, while leaving clean fidelity unchanged by construction. The completeness of Theorem 2 is borne out empirically: across all 768,000 guarded single-bit upsets the worst corruption footprint observed was 11.68% of the frame, with 470 residual catastrophic events, so the heavy tail of Figure 4b is gone. The cost is 76 µs per frame, or 0.07× a single-view render, and because it is a pure per-primitive clamp it parallelizes trivially and can be run as a periodic scrub rather than every frame. What remains is the in-support mantissa-level residual that Lemma 1 bounds and that is perceptually invisible. These per-mitigation outcomes are collected against the alternatives in Table 3."
>
> **译文**：我们在同一个故障网格上重跑实验，这次启用定理 1 和定理 2 的支撑防护。防护消解了 90.4% 的灾难性翻转；对占主导的缩放符号位翻转，它把平均全局 PSNR 从 49.2 dB 提到 65.8 dB，同时按构造不改变干净保真度。定理 2 的完备性得到实验证实：在全部 768,000 次带防护的单比特翻转中，观察到的最坏损坏脚印是帧的 11.68%，残留灾难事件 470 次，图 4b 的重尾消失了。代价是每帧 76 微秒，即单视角渲染的 0.07 倍；因为它是纯逐图元截断，并行化轻而易举，也可以按周期清洗的方式运行而不必每帧执行。剩下的是支撑盒内的尾数量级残差，引理 1 给出它的界，感知上不可见。这些逐防护措施的结果与备选方案一起收在表 3 里。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | 比特翻转注入方法学在 DNN 领域已成熟（SASSIFI、PyTorchFI、bit-flip attack），新意在对象转换：对显式几何表示逐字段逐比特建临界度地图，并从测量推出带证明的防护；截至 2026-09 的 3DGS 压缩文献里没有重叠工作 |
| 证据强度 | 高 | 380 万次注入覆盖全字段全比特全精度，闭式界与实测互验，真实场景复验，2/4 卡真实互连实测，代码、模型、逐注入记录全部开源可复现（HuggingFace: Lightcap/seu-3dgs）；扣分项：正文三处数字自相矛盾、部分关键量（K 值）未报告 |
| 对期刊版的威胁度 | 低 | 不碰渐进码流、零侧信息、内容自适应量化任何一条主张，方向是部署可靠性；它"字段敏感度差异巨大"的数据恰好支持我们分字段区别对待的设计，是助力不是威胁 |

## 对 DCCA-GS 的可借鉴点

1. 用它的比特敏感度排序校准我们的分字段粗化档位。它的结论（第 5.1 节表 2、第 5.2 节图 3a）：缩放字段最敏感（尾数高位每降一位峰值误差约翻倍，实测对数斜率 0.94），球谐高阶系数最不敏感（表 2 灾难率 0.000%，第 6 节解释为高阶球谐只能改颜色、不能改空间范围）。我们的档位是 feat/offset 粗 8 倍、scaling 对数域只粗 2 倍，方向与它的排序一致，现在有了外部依据。可跑的验证：把 scaling 档位放宽到 4 倍、8 倍测 RD 损失，看是否显著劣于 feat/offset 同倍率粗化；反向把 feat/offset 收到 4 倍看收益是否封顶。
2. 解码端支撑盒校验，零侧信息兼容。它每帧把参数截断到训练 min/max 盒并把出界次数上报为损坏信号（算法 1）。我们的解码端本来就能从已解码数据（基层全锚点粗量化值）重算各字段取值范围，把"解码值出界"用作码流损坏或截断错误的检测信号，成本近乎为零，也不违反零侧信息契约。要验证的是：渐进截断点上的量化值是否会自然出界（若会，需按档位放宽盒界）。
3. "降精度不减少暴露、只挪位置"（第 5.3 节）对基层粗化是一条设计红线：bf16 保留 8 位指数，指数位翻转照样爆炸。对应到我们：基层粗量化不能把 scaling 的符号和数量级压掉，粗化只该动尾数——我们 scaling 在对数域 2 倍步进保留符号与量级，正好满足；这条可作为基层档位不能放宽到符号翻转级别的论证写进期刊版。
4. 免测量的临界性预测（第 5.11 节）：只用字段+比特位置就能以 0.997 的 ROC 面积预测灾难位，且跨场景迁移。启示："哪个字段哪段比特重要"主要由参数化决定、不由场景内容决定，我们为不同数据集固定同一套分字段档位有正当性。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 量化/比特分配怎么定 | 不做有损量化；测存储格式（fp32/fp16/bf16）下每字段每比特的故障敏感度：缩放字段符号位与高指数位主导、尾数只有高几位有效、降精度只挪暴露位置（表 2、图 1、图 3b） | 训练期渲染敏感度监督 + 分字段粗化档位（feat/offset 8 倍、scaling 对数域 2 倍），嵌套质量阶梯逐层补精度 | 两边独立得出"缩放字段要区别对待"：它靠故障测量，我们靠 RD 收益。它的逐比特排序可当每字段"尾数保几位"的外部依据；我们缺的正是这种按字段按比特的系统性敏感度扫描 |
| 熵编码上下文 | 该论文不涉及 | 层条件熵编码（细层用已解码粗层作上下文）+ 复合条件表 | 完全错开的议题 |
| 渐进/可截断 | 该论文不涉及码流组织；但其剂量实验（第 5.4 节）说明冗余只保护典型翻转、最坏翻转靠范围约束消掉 | 单文件渐进分层码流，任意档为连续字节前缀，bit-exact 可截断 | 间接相关：截到低档时每字段精度同时下降，它提示最不能丢的是缩放字段的符号与量级（同借鉴点 3） |
| 侧信息 | 无侧信息主张；支撑盒是训练后一次 min/max 统计，防护按构造不伤干净模型（定理 1） | 硬约束零侧信息：解码端能重算的才可以用 | 它的支撑盒与我们的解码端重算同构（都从模型自身统计推导约束、不带额外码流）；可把出界检测加进解码端作完整性校验 |
| 部署可靠性/容错 | 主议题：380 万次注入 + 支撑盒截断防护 + 正确性证明 + 分布式节点本地抑制 | 该论文不涉及 | 码流若用于航天/边缘场景，它的数据说明单比特故障的主要风险面就是 scaling 字段，可写进应用动机 |
