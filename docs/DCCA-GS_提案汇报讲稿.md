# DCCA-GS 提案汇报稿（中文口播 + 英文提案对照）

> 对应 Word 提案：`3366-14-M0015-DCCA-GS Decoder-Reproducible Content-Adaptive
> Compression for Anchor-Based 3D Gaussian Splatting.docx`
> 使用说明：正文的“口播”段落可以直接照念；每一段下面的“提案对照”是从 Word
> 提案里摘出的对应英文原文（标注章节/表格）。念到哪一段，就把专家的目光引到
> 对应的英文行。总时长约 4–5 分钟。

## 0. 时长与页面分配

| 顺序 | 口播内容 | 提案对照位置 | 时长 |
| --- | --- | --- | --- |
| 1 | 开场与一句话贡献 | Abstract | 30 s |
| 2 | 背景与动机 | §1 Introduction | 30 s |
| 3 | 基线 HAC++ | §2.1 | 30 s |
| 4 | 创新点一：FQ+RSS | §2.2 | 70 s |
| 5 | 创新点二：MLP 量化 | §2.3 | 40 s |
| 6 | 创新点三：SPA | §2.4 | 40 s |
| 7 | 实验协议与主结果 | §3.1–§3.2、Table 1/2 | 60 s |
| 8 | RD 行为与消融 | §3.3–§3.4、Table 3/4、Fig. 1 | 60 s |
| 9 | 体积账与可信度 | §3.5、Table 5 | 30 s |
| 10 | 结论 | §4 | 20 s |

---

## 1. 开场（30 s）

**口播**

各位老师好，我今天汇报的提案是 DCCA-GS：面向锚点式三维高斯泼溅的解码端可复现
内容自适应压缩。我们做的是在 HAC++ 的锚点-哈希熵编码框架上做三个改动：第一，
解码端可复现的内容自适应量化，也就是 FQ 加 RSS，这是一个联合设计；第二，混合
精度的 MLP 权重量化；第三，训练侧的锚点稀疏 SPA。

**提案对照（Abstract）**

> This contribution describes DCCA-GS, built on the HAC++ anchor-hash
> entropy-coding framework. DCCA-GS keeps the HAC++ main pipeline (structured
> anchors, hash-grid context, conditional entropy model, arithmetic coding,
> G-PCC geometry) and adds three evaluated components: (1) decoder-reproducible
> content-adaptive quantization (FQ+RSS) … (2) mixed-precision post-training
> quantization of the decoder MLPs … and (3) a GaussianSpa-style training-side
> anchor sparsity budget with ADMM projection (SPA).

## 2. 背景与动机（30 s）

**口播**

先说为什么做这件事。3DGS 渲染效果好，但它把几何、协方差、不透明度、外观按基元
显式存储，场景规模一大，模型体积就线性上涨。无人机航拍和室内多视角采集正好踩中
这个痛点：场景大、细节多，存储和带宽是部署的主要瓶颈。而且一个可部署的方案，
光压缩率高不够，解码端必须自包含、可复现、推理高效。HAC++ 已经用哈希网格加熵
编码把压缩率做得很高，于是我们在它的基础上进行提升。

**提案对照（§1 Introduction）**

> UAV aerial surveys, indoor multi-view captures, and museum-scale
> reconstructions all exhibit large spatial extent combined with fine local
> detail, which is exactly the regime in which storage and bandwidth dominate
> deployment cost. Beyond compression ratio, a practical solution must be
> decoder-self-contained, bit-exactly reproducible, and efficient at inference
> time.
>
> HAC++ is the reference framework of this contribution. DCCA-GS keeps the
> Anchor-Hash main pipeline and its mask-aware probability factorization, and
> concentrates changes on three surfaces with separable effects: the
> quantization step, the training objective, and the transmitted model itself.

## 3. 基线 HAC++（30 s）

**口播**

基线一句话讲清楚：HAC++ 对每个锚点维护特征、缩放和 K 个候选偏移；用一个
mask-aware 的概率分解，让被掩码掉的偏移维度不编码；多分辨率哈希网格加一个
mlp_grid 预测每个字段的均值、方差、混合权重和基础量化步长 logits；符号用整数
CDF 算术编码，锚点坐标用 G-PCC。关键是训练、编码、解码共用同一条条件模型路径，
这是各端可复现的基础。

**提案对照（§2.1 Baseline: HAC++ Anchor-Hash Codec）**

> For the i-th anchor, x(i) denotes its position, f(i) its feature, s(i) its
> scaling, and o(i,k), m(i,k) denote the k-th candidate offset and its learned
> mask, with K = 10 offsets per anchor and feature dimension 50. The mask-aware
> entropy model factorizes the attribute joint distribution so that masked
> offset dimensions are not coded. … The conditioning vector is produced by a
> multi-resolution hash grid followed by mlp_grid, which predicts per-field
> mean, scale, mixture weight, and base quantization logits qa/qs/qo.

## 4. 创新点一：解码端可复现的内容自适应量化（FQ+RSS）（70 s）

### 4.1 FQ：步长由解码端重算，零侧信息

**口播**

第一个创新点解决量化步长怎么来的问题。统一量化对平坦区域和细节区域用同一个
步长，这不合理：平坦区域可以量化粗一点省码流，细节区域需要更细来保精度。我们的
关键设计是解码端可复现：步长根本不传输，而是由一个四维结构描述符经过一个小网络
直接算出来。这四个量分别是锚点密度、预测缩放的各向异性、预测偏移的能量和有效
掩码比例；它们全部可以从锚点坐标、mlp_grid 输出和已经解码的掩码重建。编码端和
解码端用同样的输入、同样的规则，所以算出的步长完全一致，码流里没有逐锚点的步长
数组，零侧信息。

**提案对照（§2.2）**

> Uniform quantization cannot serve flat and detailed regions equally.
> DCCA-GS computes the step for field t as Q = Q0 × (1 + tanh(z) × α) for
> feature, scaling, and offset, and alpha = 0.35. … The logits z_t come from a
> small complexity network mlp_complexity whose input is a 4-dimensional
> descriptor: local anchor density, predicted scaling anisotropy, predicted
> offset energy and active mask ratio. All four structural quantities are
> recomputed by the decoder from anchor coordinates, mlp_grid outputs, and
> already-decoded masks; therefore the steps are bit-exactly reproducible and
> the bitstream contains no per-anchor step array.

### 4.2 RSS：同一个网络的训练目标

**口播**

第二个部分回答步长“应该是什么”。我们累积每个锚点渲染损失梯度范数的指数滑动
平均，做有界归一化后作为目标乘子，监督同一个复杂度网络去逼近渲染敏感度最优的
步长。它只影响训练，编码和解码仍然走 FQ 的公式路径，所以不增加任何码流字段。

**提案对照（§2.2）**

> Render-sensitivity supervision (RSS) defines the training objective of the
> same complexity network: the multiplier target is derived from per-anchor
> render-loss gradient norms accumulated with an exponential moving average …
> RSS changes only the training objective; encode and decode still follow the
> FQ formula path, so no bitstream field is added.

### 4.3 为什么是联合设计

**口播**

所以 FQ 和 RSS 是一个创新点的两半：FQ 决定步长怎么由解码端重算，RSS 决定步长
应该跟着渲染敏感度怎么走，二者共用同一个复杂度网络。高码率下 RSS 大约带来
0.1 dB 的增益；但在低码率 SPA 场景下它是关键——去掉 RSS，PSNR 直接掉 0.42 dB，
而 FQ 没有 RSS 时反而有害，具体数字在提案 Table 4。所以我们按联合设计报告，
不在低码率工作点上单独讨论 FQ。

**提案对照（§2.2 + §3.4）**

> FQ and RSS are two halves of one joint design and share the same complexity
> network: FQ defines how each quantization step is recomputed from
> decoder-reconstructable quantities, while RSS defines what the step should
> be by supervising the network toward render-sensitive multipliers. They are
> therefore reported jointly, and isolated FQ conclusions are avoided in the
> low-rate SPA regime (see Section 3.4).

## 5. 创新点二：混合精度 MLP 权重量化（40 s）

**口播**

第二个创新点在模型本身。解码侧的 MLP 既参与熵模型，也参与渲染，一共六组。我们
按输出通道做对称量化，再用静态算术编码压缩整数索引。位宽不是统一的：复杂度和
形变网络压到 8 位，其余网络保持 16 位。这个选择来自逐 MLP 的敏感性消融：8 位的
不透明度会直接伤渲染，8 位的网格会让熵模型变差、属性码率反而增加约 0.25 MiB，
而复杂度、形变网络 8 位几乎无损。量化后的真实载荷计入报告体积，提案里 5.65 到
5.49 MiB 就是这个口径。

**提案对照（§2.3 + §3.5）**

> Decoder MLPs participate in the entropy model (mlp_grid, mlp_deform,
> mlp_complexity) and in rendering (mlp_opacity, mlp_cov, mlp_color). DCCA-GS
> quantizes each parameter tensor per output channel with a symmetric scale …
> The recommended map is complexity and deform at 8 bit and the remaining
> groups at 16 bit. … the attribute bitstream is regenerated after quantization
> and the compressed MLP payload replaces the float32 accounting in the
> reported size.
>
> (§3.5) … MLP weights account for only 5.9% under the 32-bit accounting
> convention; … the decode-required size is further reduced from 5.65 MB to
> 5.49 MB without introducing any side information.

## 6. 创新点三：训练侧锚点稀疏 SPA（40 s）

**口播**

第三个创新点是训练侧稀疏。它和编码端 top-k 剪枝的区别是：SPA 在训练中用 ADMM
硬预算直接控制锚点数，剪掉之后幸存锚点还能在预算下继续训练。具体地，每个锚点有
一个软得分 a，是掩码的平均；每 100 步做一次 ADMM 更新：硬掩码 z 取 a 加乘子 u
的 TopK，u 再按约束裁剪；预算锚定到历史最大锚点数乘一个线性斜坡的比例，避免
“每轮减半”的几何塌缩；增广损失进训练，SPA 状态不进码流。~~结果是在相同锚点数下，
它比编码端 top-k 高 5.5 dB；代价是体积省约 74%、PSNR 掉 0.84 dB，所以我们把它
定义为低码率工作点，而不是免费提升。~~

**提案对照（§2.4 + §3.4）**

> DCCA-GS's anchor mask provides a soft per-anchor score a = mean(mask),
> averaged over the K offset candidates. SPA adds two training-only tensors, a
> binary hard mask z and a multiplier u, and every 100 iterations performs the
> ADMM update z = TopK(a+u, κ), u = clamp(u+a−z, −1, 1). The budget is κ =
> round(max(N_ref) × ratio(t)) … anchoring the budget to the maximum observed
> anchor count prevents the geometric collapse of a ratio × N(t) schedule.
> SPA state is never written to the bitstream.
>
> (§3.4) … it reduces the decoded payload from approximately 4.38 MB to
> 1.13 MB (≈74%) at a cost of 0.84 dB PSNR …

## 7. 实验协议与主结果（60 s）

**口播**

实验我们选了两个数据集，覆盖两种互补的采集模态：4-28 是一个室外无人机航拍的，有1200 张训练图像、150 个验证视角的大场景；Deep Blending 是室内手持、以物体为中心的两个小场景。评估统一在解码后的真实码流上行：1600 宽、test_every取 8，体积按解码必需载荷计。

先看 4-28 的主结果。110k 迭代加 MLP 量化，我们做到 28.82 dB、5.49 MiB；HAC++论文同场景的参考点是 28.31 dB、6.95 MiB——质量高 0.51 dB，体积小约 21%，质量和紧凑性同时拿到。开 SPA 之后进一步到 28.30 dB、3.77 MiB，体积再省约 31%。

Deep Blending 上同样清楚：高码率点 playroom 30.73 dB、4.22 MiB，drjohnson30.04 dB、6.85 MiB；SPA 低码率点分别到 29.90 dB、0.96 MiB 和 28.76 dB、1.55 MiB，直接进入现有综述方法没有覆盖的小于 1 MiB 的区间。

**提案对照（§3.1 + §3.2、Table 1/2）**

> (§3.1) Evaluation is performed on decoded bitstreams at 1600-pixel width with
> data_factor = 1 and test_every = 8. Metrics are PSNR, SSIM, and VGG-LPIPS.
> Sizes are decode-required payload in MiB.
>
> (§3.2) DCCA-GS achieves the highest PSNR of 28.82 dB among all methods …
> Compared with the same-framework baseline HAC++ (28.31 dB, 6.95 MB), DCCA-GS
> improves PSNR by 0.51 dB while reducing the required size by about 21% …
>
> (Table 2) … the high-quality point (FQ + RSS + MLP quantization, without SPA)
> reaches 30.73 dB PSNR, 0.913 SSIM, and 4.22 MB; the low-bitrate SPA point
> reaches 29.90 dB PSNR and 0.96 MB.

## 8. RD 行为与消融（60 s）

**口播**

再看码率行为。λ 从 0.001 扫到 0.004，playroom 体积从 5.80 降到 3.01 MB，
drjohnson 从 9.63 降到 4.76 MB，PSNR 几乎不变，说明码率可以连续切换；开 SPA 后
曲线延伸到 1 MiB 以下，形成从高质量到极低码率的连续操作区间。消融方面，SPA 是
省体积的主导因素，4.38 到 1.13 MiB，约 74%，代价 0.84 dB；SPA 下 RSS 贡献
0.42 dB；FQ 没有 RSS 时反而差 0.29 dB；FQ 加 RSS 联合是 0.13 dB。这些数字都在
提案 Table 4，也解释了为什么 FQ 和 RSS 必须联合报告。

**提案对照（§3.3 + §3.4、Table 3/4、Fig. 1）**

> (§3.3) As λ increases from 0.001 to 0.004, the decode-required size of
> playroom and drjohnson smoothly decreases from 5.80 MB and 9.63 MB to
> 3.01 MB and 4.76 MB, respectively, while PSNR remains highly stable within
> 30.62–30.73 dB and 30.04–30.09 dB.
>
> (§3.4) Enabling SPA is the dominant rate-reduction factor … the full model
> outperforms SPA without RSS by 0.42 dB PSNR … the SPA+FQ (w/o RSS) variant is
> 0.29 dB worse than SPA-only … the full model is 0.13 dB better than the
> SPA-only baseline.

## 9. 体积账与可信度（30 s）

**口播**

最后看一下钱花在哪、结果可不可信。字段级分解显示，特征、缩放、偏移这三类属性
合计占约 82.8%，哈希参数加边界和头部合计不到 0.5%，说明码率都花在内容属性上。
编码效率方面，实际码流比特除以模型交叉熵接近 1.0，说明算术编码器已经贴紧熵模型；
KL 审计给出缩放字段至少 0.49 bit/符号的模型侧冗余下界。所有码流都通过了独立
解码验证，符号、量化步长、整数 CDF 不匹配数都是零。

**提案对照（§3.5 + Abstract 末尾）**

> (§3.5) Attribute fields—feature, scaling, and offsets—together account for
> about 82.8% of the total size … while hash parameters and bounds/header
> overhead account for less than 0.5% in total …
>
> (Abstract) We also report quantitative analyses of codec efficiency
> (actual/estimated bits close to 1.0), a KL-redundancy lower bound for the
> scaling field … All bitstreams pass bit-exact standalone decoding
> verification.

## 10. 结论（20 s）

**口播**

总结一下：DCCA-GS 保留 HAC++ 主链路，把码率控制的三个自由度做成了解码端可复现
或训练侧的改动——FQ+RSS 决定步长，MLP 量化压缩模型载荷，SPA 控制锚点数量。
4-28 上我们做到 28.82 dB、5.49 MiB，质量和体积同时优于 HAC++ 参考点；SPA 把
操作区间延伸到亚 1 MiB。解码完全自包含、逐比特可复现，压缩比最高约 111 倍。
以上就是我的汇报，谢谢各位老师。

**提案对照（§4 Conclusion）**

> By introducing decoder-reproducible content-adaptive quantization with
> render-sensitivity supervision (FQ+RSS), mixed-precision MLP weight
> quantization, and training-side ADMM sparsity, DCCA-GS achieves significant
> bitrate savings while maintaining high rendering fidelity. … pushing the
> compression ratio up to 111× with minimal quality loss.

---

## 附录 A：评审追问备查（对应提案细节）

### A.1 FQ 与原版 HAC++ AQM 的关系（§2.2）

**口播**

如果被问到和原版自适应量化 AQM 的区别：原版 AQM 是 MLP 从哈希特征预测每锚点步长
修正量，编码解码用同一套权重重算，也是零侧信息。我们保留这个框架，只升级输入和
目标：输入换成解码端可重建的四维结构描述符，目标换成 RSS 的渲染敏感度监督。

**提案对照**

> (原版) q = Q0 × (1 + tanh(r))，r = MLP_q(f^h)
> (FQ)   q = Q0 × (1 + tanh(z) × α)，z = mlp_complexity(4 维结构描述符)

Q0 对特征、缩放、偏移分别取 1、0.001、0.2；α=0.35。

### A.2 消融交互（Table 4，供追问）

| 对比 | 发现 |
| --- | --- |
| FQ 单独（无 SPA） | +0.002 dB，几乎无贡献 |
| RSS 在 SPA 下 | +0.42 dB，低码率关键 |
| FQ 在 SPA 下（无 RSS） | −0.29 dB，反而有害 |
| FQ+RSS 联合（SPA 下） | +0.13 dB |

> (§3.4) This suggests that FQ should be reported jointly with RSS, and that
> isolated FQ conclusions should be avoided in the low-rate SPA operating
> point.

### A.3 体积口径

报告体积 = 解码必需载荷：G-PCC 锚点坐标 + 特征/缩放/偏移算术流 + 掩码 +
哈希参数 + 压缩后的 MLP 载荷 + 边界/头部；不包含检查点与调试数据。

> (§3.1) Sizes are decode-required payload in MiB.

### A.4 已记录的负结果（Abstract 末尾）

**口播**

如果被问到为什么不做跨锚点上下文、残差编码或敏感度侧信息：提案里记录了三个负
结果——跨锚点条件熵增益低于 2.3%，残差编码无收益，敏感度侧信息与解码端可重算
输入的相关性接近零。这些说明熵模型输入侧和编码器本身都已经接近饱和，进一步压缩
的空间主要在表示侧，也就是锚点数、步长和模型载荷，这正是三个创新点的选点依据。

**提案对照**

> … documented negative results (context conditioning below 2.3%, residual
> coding, and sensitivity side information). All bitstreams pass bit-exact
> standalone decoding verification.
