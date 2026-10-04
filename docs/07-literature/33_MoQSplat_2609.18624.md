# 《MoQSplat: Adaptive Progressive Streaming of 3D Gaussian Splatting via MoQ》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2609.18624 |
| 发表 | IEEE 出版物（具体会议论文未写明，版权页标注 © 2026 IEEE）；arXiv v1 2026-09-16，cs.MM |
| 阅读材料 | 全文（.lit-cache/txt/2609.18624.txt，与 PDF 逐页核对一致，共 7 页） |
| 与 DCCA-GS 的关系 | 同属 3DGS 渐进传输方向，但它在传输层（MoQ/QUIC 多条流按优先级调度）做渐进，我们在比特流层（单文件可截断熵编码码流）做渐进——是传输侧参照对象与互补方案，不是压缩码流竞品 |

> 一句话定位：把 3DGS 场景按"空间块→语义物体簇→质量档→单个 splat 属性"四级切分，映射到 IETF Media over QUIC（MoQ）的四级内容层级上，每个质量档走一条独立 QUIC 流，订阅端按 6 自由度视锥、距离和注视对齐动态调各条流的优先级，实现没有连接级队头阻塞的自适应渐进流式播放；它不做任何压缩（属性按原始 float32 传输），省的是"先送哪些字节"的调度，不是字节总量。

## 一学：问题与洞察（创新从哪来）

### 它认定的主要矛盾

论文引言认定两级矛盾：

1. 3DGS 场景体积太大：要表现复杂的高光反射需要最多 3 阶球谐系数（每个基元 45 个参数，引言引 [3]），真实场景由数百万个 splat 组成、超过数 GB，标准网络上的实时传输是瓶颈。
2. 现有传输方案都建在 HTTP 自适应流播（HAS，如 MPEG-DASH）上：LapisGS [5] 用不透明度优化做渐进分层，LTS [6] 把空间分块和 DASH 分段结合，L3GS [7] 用视口预测排物体顺序。但 TCP 传输有队头阻塞（Head-of-Line blocking）：丢一个包，后面所有数据都得等重传；而 HAS 的分段粒度粗，不适合 splat 这种细粒度数据。论文原话："This combination of TCP's rigid reliability and HAS's coarse segment granularity is ill-suited to fine-grained 3DGS data."（第 I 节）

HTTP/3 over QUIC 用多条复用的 UDP 流缓解了队头阻塞，但标准的客户端拉取式分段仍会放大延迟。MoQ 是 IETF 正在制定的低延迟交互媒体传输标准（Internet-Draft draft-ietf-moq-transport-18，引文 [10]），采用发布/订阅架构，内容按 Track / Group / Subgroup / Object 四级组织，可经中间 Relay 转发。MoQ 已被用于点云流播 [11][12]，但还没人把 3DGS 的各向异性 scale、opacity、球谐系数这套数据映射上去——这就是它填的空。

### 支撑观察（直觉 / 数据 / 失败实验）

- 观察一（动机层，第 I 节）：TCP+HAS 的既有 3DGS 流式工作存在队头阻塞和分段过粗的问题。这是对已有系统文献的归纳，论文没做自己的对照测量。
- 观察二（内容层，第 IV 节 Fig. 3、Fig. 4）：同样数量的 splat，按不透明度剪枝排序送出去比按 scale 排序画质高；等量 20% 步长切五档时，不透明度排序在前两档（Q1，约 40% splat）就接近饱和，之后增益很小。这是论文唯一由自己实验支撑的内容侧观察。
- 观察三（传输层，第 IV 节 Table I）：带宽从 50 Mbps 翻倍到 100 Mbps，最终渲染的 splat 只从约 1.49M 增到约 1.51M——说明优先级调度确实在起作用：高优先级内容先到，带宽富余后只有低优先级增强层在慢慢补。
- 这三条观察里，只有观察二和观察三有本文自己的数据；观察一引用的是 LapisGS、LTS、L3GS 等论文的既有结论，本文没有做 TCP 与 QUIC 的传输对照实验来量化队头阻塞到底损失多少。这一点在判断证据强度时要注意。

### 显然的路为什么没走

- 为什么不沿用 HTTP/DASH（LapisGS/LTS/L3GS 的路线）？因为 TCP 队头阻塞和分段粒度是这套方案的固有缺陷，换传输协议是它认定的根本解法。
- 为什么不在服务端做每客户端的智能调度？它反着走：Relay 完全不看内容（content-agnostic，"No scene-aware scheduling"，Fig. 1），Publisher 一次性把所有空间和质量变体发布上去，适配决策全部推给订阅端（stateless、subscriber-driven）。理由是按需静态场景分发天然适合这样做，且多订阅者场景下服务端无需维护每个客户端的状态（第 II.A 节）。这是 MoQ 架构的原生设计哲学，论文顺势而为而不是自建调度服务。
- 为什么不做压缩？第 III 节给出明确答案："Gaussian attributes are encoded as raw float32 arrays, since a standardized 3DGS compression codec has not yet been finalized."（高斯属性编码为原始 float32 数组，因为标准化的 3DGS 压缩编解码器还没定稿。）它把压缩整个跳过，只解决"字节以什么顺序、什么粒度、走哪条流"的问题。这恰好划清了它和我们的边界。
- 为什么押注一个还没定稿的 IETF 标准（MoQ 截至 2026 年 5 月仍是 Internet-Draft，引文 [10]）而不自建传输？论文没有直接回答，但可以从它的立场推断：作者团队（奥地利克拉根福大学 ATHENA 实验室，Timmerer 组）长期参与 MPEG-DASH 和 MoQ 标准化（引文 [4] 的作者就有该组成员），押注标准意味着方案能跟着标准演进、Relay 可以用社区现成实现（他们确实直接用了 moq-relay 参考实现）。代价是论文承认"MoQ transport 继续向 RFC 标准化演进时，未来可能需要修订以对齐最终规范"（第 III 节 Relay Implementation 段）。

## 二学：机制（洞察怎么变成方法）

### 核心流程

本文是系统论文，没有训练期（不训练、不微调 3DGS 模型，直接拿现成场景切分）。按角色分三个环节讲。

**发布端（Publisher），离线一次性内容准备（第 III 节，五步）：**

1. 语义物体检测：从 N 个视角渲染内容，对渲染图跑 2D 物体检测提取 Group。检测器原文写 YOLOv26x，引文 [19] 为 YOLO-World（模型名与引文不一致，原文如此）。每个检测框可选按预定义的 padding 因子扩张，保证对 splat 区域的覆盖偏保守。
2. 合并投票：同一物体被多个视角重复检测，用两种方式之一合并——对相应高斯 splat 集合算 3D Jaccard 指数（IoU），或比较关联 splat 的质心接近度。多次观测合并成一个统一的框。
3. 聚类：把多视角检测框投回 splat 空间，对物体质心跑 DBSCAN（引文 [16]）聚成簇，空间相近的 splat 归入同一 Group；空间上邻近的多个语义物体可能被并进一个簇。
4. 渐进质量档构建：实现设计中"独立几何启发式"——按 scale 和 opacity 两个代理量做几何剪枝排序，步长 20%，切成 5 个渐进质量档（基层 BL + 增强层 EL1–EL4），每档含 20% 的 splat。
5. 复杂度核算：多视角渲染加 2D 检测为 O(N·T_YOLO)；投影回 3D 加 DBSCAN 为 O(M log M)（M 为高斯质心数）；不透明度排序也是 O(M log M)。整条流水线完全离线，属于一次性内容摄取，流播期间没有预处理延迟。

层级实现细节：moq-lite（Rust 写的 Pub/Sub 框架，跑在 QUIC 和 WebTransport 上）原本只支持 Track/Group/Object 三级，作者为它扩展出显式的 Subgroup 层。每个 Subgroup 对应一条独立 QUIC 流。Publisher 给 Subgroup 设静态基线优先级：基层设低数值（MoQ 优先级是 [0, 255] 的整数，0 最优先），每个后续增强层依次升高（优先级递减）（第 II.B 节）。manifest（论文称 Catalog）以 JSON 形式作为一条独立的 MoQ Track 发布，内容包括 Track 标识、空间包围盒、Group 与 Subgroup 的组织结构、Group 的 3D 质心（第 II.B 节）。

**转发端（Relay）：**直接采用 moq-relay（moq-lite 参考实现），内容无关、按优先级转发的 QUIC 转发服务；转发 SUBSCRIBE_UPDATE 等上游控制消息；拥塞由 Relay 隐式处理，高优先级 Subgroup 先发（第 III、IV 节）。

**订阅端（Subscriber），播放期驱动适配（第 III 节，三个组件）：**

1. 订阅层级与状态机：建立连接后取回 Catalog，重建 Track/Group/Subgroup 层级；给每个 Subgroup 维护一个应用层状态机（MISSING / DOWNLOADING / CACHED）。DOWNLOADING 状态下发生丢包或超时，等超时阈值后状态回退到 MISSING，重新发 SUBSCRIBE 订阅；此时渲染器回退到显示已缓存的基层 S1 和低编号增强层，避免画面卡死。
2. 视锥引导的优先级循环：100 毫秒间隔的异步控制循环。每轮取当前 6 自由度位姿，对 Catalog 里所有 Group 按式 (1) 算优先级分数 ΔP；某 Subgroup 状态为 MISSING 就带算好的 ΔP 发新 SUBSCRIBE，已在 DOWNLOADING 就发 SUBSCRIBE_UPDATE 在不断连的情况下改这条流在线上的优先级。收到的 Object 拆包进本地内存池，一个 Subgroup 的 Object 收齐就转 CACHED。
3. 水位式显存换页：设固定 GPU 显存预算 Mmax 和高水位阈值（如 0.9×Mmax）。活动 splat 内存超过阈值时，保护所有基层 Subgroup 不被逐出，把增强层按当前 ΔP 排序，逐出优先级最低的增强层，其张量降到系统内存（CACHED 态）；用户转动视角、某缓存对象的 ΔP 回升时，再把数据提升回显存渲染，避免重复走网络。

兜底行为（第 II.B 节）：如果订阅端不用任何优先级算法，所有 Subgroup 的订阅端优先级相同，调度完全退回 Publisher 的静态渐进优先级——所有可见簇的基层同时发，然后所有第一增强层，依次类推，构成基线传送模式。

### 关键公式（按原文抄，注明编号）

全文只有一个公式，式 (1)（第 III 节，订阅端优先级循环）：

```
∆P = Wd·max(0, 1 − min(D, Dmax)/Dmax) + Wa·max(0, cos θ)    (1)
```

（按 txt 与 PDF 抄录，结构完整，两处 max、一个分式，无损坏。）

逐项含义（原文第 III 节）：
- ΔP：一个 Group 的订阅优先级分数，值越大越该先送；
- D：订阅端相机位置到该 Group 3D 质心的欧氏距离；
- θ：相机前向视线向量与"相机指向 Group 质心"向量之间的夹角；
- Dmax：预期最大观看距离；
- Wd、Wa：两个缩放权重，分别加强"近处优先"和"视线中央（注视对齐）优先"。

两项都截断在 0 以下（max(0, ·)），即太远的物体和视野边缘的物体不贡献分数。注意 ΔP 是按 Group 算的，实际订阅和调优先级的单位是 Group 下的各 Subgroup。Wd、Wa 的取值论文未报告。

### 信号设计（重点）

| 问题 | 答案 |
| --- | --- |
| 信号是什么（重要性/复杂度/敏感度/显著度…） | 两个信号。内容侧重要性代理：scale 与 opacity（第 III 节第 4 步，用于把 splat 排序切成质量档；设计中另提空间显著度=体积与孤立程度，第 II.A 节）；传输侧视口效用：ΔP，距离项加注视对齐项（式 1，用于排流的先后） |
| 哪一端算得出（编码端/解码端/仅训练期） | scale/opacity 排序在发布端离线算好写进层级；ΔP 由订阅端（客户端）播放期实时算 |
| 有无侧信息 | 有，而且是必需侧信息：manifest（Catalog）单独成一条 MoQ Track 先行传输，写明 Track 包围盒、Group/Subgroup 结构、Group 3D 质心；订阅端的视口适配完全依赖这份元数据（第 II.B、III 节） |
| 训练期还是编码期 | 无训练期。内容切分在发布前的离线预处理完成（一次性内容摄取，第 III 节第 5 步明确说"operates entirely offline"） |
| 和渲染质量的距离（直接测质量还是代理量） | 两层都是代理量：opacity/scale 是几何代理，不看渲染误差；ΔP 是视口效用代理，不测质量。渲染质量只在发布端评估阶段用 VMAF 事后测量（第 IV 节），不参与任何运行时决策 |

### 与码流组织相关的细节（压缩类论文必填）

它是传输组织论文，渐进是"多条流"式的，与我们的"单文件"式逐条对照：

1. 渐进层怎么划分：按 splat 排序后等量截断。原型用 scale 与 opacity 作排序代理，步长 20%，切成 5 档（BL + EL1–EL4），每档 20% splat（第 III 节第 4 步）。设计稿里还提出第二种未实现的划分：按属性拆层——基层 S1 传几何加 0 阶球谐（漫反射色），高阶球谐推迟到增强层，增强层渲染依赖 S1 几何（解码依赖，第 II.A 节）。
2. 文件布局：没有单文件码流的概念。四级层级各自独立成为传输对象：Track=不重叠空间块，Group=语义物体簇，Subgroup=一条独立 QUIC 流，Object=网络传输单元（封装单个 splat 的几何、opacity、球谐等属性，第 II.A 节）。manifest 单独一条 Track。
3. 可截断性：不体现在字节层面。一条流（Subgroup）内部是完整的 float32 数组，不可在流内截断；"截断"靠少订阅几条流实现，粒度最细到一条 Subgroup 流。队头阻塞靠 QUIC 多流互相独立来消除，不靠部分可靠性。
4. 解码端重算：没有。所有切分决策在发布端定死并写进 manifest；客户端只算订阅优先级，不重算任何量化参数或编码上下文。
5. 侧信息：manifest 是必需的先决侧信息（包围盒、质心、档位结构），没有它订阅端无法做视口适配。
6. 优先级体系：MoQ 优先级为 [0, 255] 整数、0 最优先；调度引擎先看订阅端指派的优先级，发布端静态优先级只作平手裁决（第 II.B 节）。

## 三学：证明与包装（创新怎么立住）

### 实验口径

- 两阶段评估（第 IV 节）：先评发布端离线内容准备的质量，再评端到端流式传输在不同带宽下的表现。
- 内容：Mip-NeRF360 数据集（引文 [20]）的 Bicycle 场景，两个阶段共用；发布端评估聚焦从该场景聚出的 Bench–Bicycle 物体簇。单场景，没有第二个场景。
- 质量指标：VMAF（Netflix 提出的视频画质客观指标，模拟人眼感受打 0 到 100 分，分越高画质越好）。相机沿 ±90° 方位角平滑环绕目标簇，固定距离、仰角和相机内参；每段 10 秒、30 帧/秒，共 300 帧；以未剪枝模型为参考评估剪枝后的模型。
- 硬件：Ubuntu 22.04，Intel Xeon Gold 5218 @ 2.30 GHz，双 NVIDIA Quadro GV100（第 IV 节）。
- 端到端协议：Publisher 预处理后发布到 Relay，四种带宽（10/30/50/100 Mbps）下记录送达 Subgroup 数、接收数据量、最终渲染 splat 数（Table I）。
- 没有 PSNR/SSIM/LPIPS，没有码率-失真曲线，没有与 LapisGS/LTS/L3GS 的定量对比。

### 主结果表抄录（注明表号）

Table I（"Subscriber-driven Delivery Performance — Bicycle Scene"，第 IV 节），共 4 行（4 档带宽），全抄：

| 链路带宽 (Mbps) | 送达 Subgroup 数 | 接收数据 (MB) | 最终渲染 splat 数 |
| --- | --- | --- | --- |
| 10 | 397 | 37.4 | 224,710 |
| 30 | 2,432 | 337.8 | 1,400,909 |
| 50 | 2,628 | 363.2 | 1,491,950 |
| 100 | 2,853 | 390.6 | 1,512,214 |

（列名 Link BW 直译"链路带宽"，下文简称带宽。）

正文引用的关键数字（第 IV 节）：10 Mbps 下约 22.5 万 splat，"粗糙但可持续渲染"；30 Mbps 下 Subgroup 从 397 增到 2,432、splat 从约 22.5 万增到 140 万；100 Mbps 相比 50 Mbps 带宽翻倍只带来小幅增加。Bicycle 场景的 splat 总数论文未报告，100 Mbps 收到约 151 万是否等于全集也未说明。

Fig. 4（"Progressive quality enhancement measured by VMAF"，第 IV 节）：横轴 Q0–Q4 五档（Q0=BL，Q4=BL+EL1+…+EL4），纵轴 VMAF。曲线目测近似值（图上读数，正文未给具体数值，原文公式与图表以 PDF 为准）：不透明度排序 Q0 约 41 → Q1 约 76 → Q2 起约 100 饱和；scale 排序 Q0 约 28 → Q1 约 40 → Q2 约 56 → Q3 约 81 → Q4 才到 100。正文结论（第 IV 节）："opacity-based pruning consistently outperforms scale-based pruning by achieving higher perceptual quality with the same amount of rendered splats"；且"gains diminish under opacity-based pruning as quality saturates beyond Q2 (Fig. 4)"（不透明度剪枝在 Q2 之后饱和，再补 splat 增益很小）。论文还指出各层 splat 数相同但感知贡献差别很大（第 IV 节"Enhancement Layer Analysis"段）。

### 消融设计（注明表号）

没有正式消融表。能算对照实验的有两处：

1. 排序代理量替换对照（Fig. 3 + Fig. 4）：同层数（5 档）、同步长（20%）、同渲染相机协议，只把排序量从 scale 换成 opacity——一变量对照，设计干净，值得学。
2. 带宽扫描（Table I）：四档带宽下同一发布内容的表现，可视为传输侧参数扫描。

缺的对照：无语义 Group 聚类有无的对比、无订阅端优先级算法有无的对比（只在第 II.B 节文字描述了无优先级算法时的兜底行为）、无与既有渐进 3DGS 方案的数值对比。

### 贡献列表（原文照抄 + 逐条标注）

1. "MoQ hierarchy mapping for 3DGS: We map 3DGS onto MoQ's four-level content hierarchy, detecting semantic objects and defining quality refinements into distinct Subgroups." —— [机制] 把 3DGS 映射到 MoQ 四级层级：空间 Track、语义 Group、质量档 Subgroup。
2. "Subscriber-driven adaptation: We introduce a fully stateless client-side subscription model. The client independently calculates utility based on frustum alignment and Euclidean distance, dynamically adjusting layer granularity via SUBSCRIBE commands." —— [机制] 全无状态的客户端订阅模型：客户端按视锥对齐与欧氏距离自算效用，用 SUBSCRIBE / SUBSCRIBE_UPDATE 动态调整订阅的档位粒度。
3. "Framework implementation: We outline the end-to-end Python-based MoQSplat implementation built over the customized moq-lite Pub/Sub framework and PyTorch-based gsplat renderer." —— [工程] 端到端原型：定制 moq-lite（加了 Subgroup 层）加 PyTorch gsplat 渲染器。

结论段另有一个首创性主张："the first framework to map 3D Gaussian Splatting (3DGS) content onto the Media over QUIC (MoQ) hierarchical content structure"（第 V 节）。对照点云领域已有 MoQ 工作 [11][12]，这个"第一"限于 3DGS 范围，成立。

### 讲故事方式

主线一句话：3DGS 太大传不动 → 现有 HAS 方案受 TCP 队头阻塞和粗分段拖累 → MoQ 的四级层级和多流机制与 3DGS 的"空间/语义/质量/属性"结构天然一一对应 → 给出映射 + 原型 + 初步验证。
最有力的一张图是 Fig. 4：同一 splat 预算下不透明度与 scale 两条 VMAF 曲线的明显分离，是论文中唯一支撑"内容侧切分策略有效"的定量证据（Fig. 3 只是视觉对照）。整体是系统论文的讲法：架构图（Fig. 1、Fig. 2）承担主要叙事，实验定位为"preliminary"（初步）可行性验证，不是压缩论文的 RD 曲线讲法。作者自己在结论里承认原型和评估都是初步的（第 V 节）。

## 四学：原文关键段翻译（深读层必填）

### 方法节核心段（两段）

> **原文（第 II.A 节，Subgroup 划分）**："Subgroup. Each Group is partitioned into Subgroups representing quality levels (S1, . . . , Sm) using two efficient heuristics: 1) Independent Geometric Heuristics - partition the splat population by ranking spatial saliency (volume and isolation) to ensure base layer S1 minimizes visual holes. Subgroup boundaries are configured via percentage thresholds or fixed cardinality budgets, and can integrate pruning [17] or compression [18]. 2) Dependent Attribute-Based Partitioning - splits splat attributes across layers. Base layer S1 transmits geometry and 0-th order Spherical Harmonics (SH) diffuse color, while higher-order SH coefficients are deferred to enhancement layers. This introduces a decoding dependency requiring S1 geometry before enhancement layers can be rendered."
>
> **译文**：Subgroup（子组）。每个 Group 再划分为代表质量档（S1, …, Sm）的 Subgroup，用两种高效启发式：1）独立几何启发式——按空间显著度（体积与孤立程度）对 splat 总体排序后划分，使基层 S1 尽量减少视觉空洞。Subgroup 的边界通过百分比阈值或固定数量预算来配置，并可集成剪枝 [17] 或压缩 [18]。2）依赖式按属性划分——把 splat 的属性拆到不同层。基层 S1 传输几何和 0 阶球谐（漫反射颜色），高阶球谐系数推迟到增强层。这引入一种解码依赖：必须先拿到 S1 的几何，增强层才能渲染。

> **原文（第 II.B 节，传输优先级）**："To enable progressive delivery, the Publisher assigns static baseline priorities to Subgroups. Specifically, these priorities are set to low values (representing high priority, as MoQ priorities are integers in the interval [0, 255] where 0 is the most prioritized) for the base layer, and progressively higher values (lower priority) for each subsequent enhancement layer. Consistent with MoQ's prioritization model, the scheduling engine evaluates subscriber-assigned subscription priorities first, using publisher-assigned priorities only as a tie-breaker."
>
> **译文**：为了实现渐进传输，Publisher 给每个 Subgroup 指派静态基线优先级。具体做法：基层设为低数值（代表高优先级，因为 MoQ 的优先级是 [0, 255] 区间内的整数，0 最优先），后续每个增强层依次设更高的数值（更低优先级）。与 MoQ 的优先级模型一致，调度引擎先评估订阅端指派的订阅优先级，Publisher 指派的优先级只用作平手裁决。

### 主结果表述段（一段）

> **原文（第 IV 节，End-to-End Streaming Delivery）**："Table I shows subscriber-driven delivery metrics across four bandwidth conditions. At 10 Mbps, the Subscriber receives 397 Subgroups (37.4 MB), corresponding to approximately 225K Gaussian splats and yielding a coarse but continuously renderable scene representation. Increasing the available bandwidth to 30 Mbps substantially improves delivery performance, with the number of received Subgroups increasing from 397 to 2,432 and the reconstructed splats growing from approximately 225K to 1.4M. Further increasing the bandwidth to 50 Mbps results in a more modest improvement, reaching 2,628 delivered Subgroups and approximately 1.49M reconstructed splats. Finally, at 100 Mbps, the Subscriber receives 2,853 Subgroups and approximately 1.51M splats, representing only a small increase over the 50 Mbps case despite a twofold increase in available bandwidth."
>
> **译文**：表 I 展示了四种带宽条件下订阅端驱动的传输指标。10 Mbps 下，订阅端收到 397 个 Subgroup（37.4 MB），约对应 22.5 万个高斯 splat，得到一个粗糙但可持续渲染的场景表示。把可用带宽提到 30 Mbps 明显改善传输表现：收到的 Subgroup 从 397 增到 2,432，重建 splat 从约 22.5 万增到 140 万。继续提到 50 Mbps 带来更温和的提升，达到 2,628 个送达 Subgroup 和约 149 万重建 splat。最后，100 Mbps 下订阅端收到 2,853 个 Subgroup 和约 151 万 splat，相比 50 Mbps 只有很小的增加——尽管可用带宽翻了一倍。

## 评级

| 维度 | 等级 | 理由 |
| --- | --- | --- |
| 新颖性 | 中 | "第一个把 3DGS 映射到 MoQ 层级"这个空位是真的（点云已有 [11][12]），四级映射加订阅端驱动的组合有系统价值；但单个组件——语义聚类（YOLO 检测加 DBSCAN）、不透明度排序渐进切档、视口优先级——都是已有技术的组装，内容侧没有任何压缩或表示创新 |
| 证据强度 | 低 | 单场景（Mip-NeRF360 Bicycle）、无 PSNR/SSIM/LPIPS、无与 LapisGS/LTS/L3GS 的定量对比、无正式消融表；定量证据只有一张 4 行传输表（Table I）和一条 VMAF 曲线（Fig. 4，正文未给数值）。加分项：代码开源（github.com/emanuele-artioli/MoQSplat，摘要声明），可复核 |
| 对期刊版的威胁度 | 低 | 不撞我们的三条主张：它没有单文件渐进码流（多条流各装完整 float32 数组）、没有零侧信息（manifest 是必需先决侧信息）、没有内容自适应量化（完全不量化）。可借的是它梳理的 MoQ 标准化背景和相关工作（LapisGS/LTS/L3GS），以及"传输侧渐进 vs 码流侧渐进"这条可写进期刊版相关工作的对照线 |

## 对 DCCA-GS 的可借鉴点

1. 等量切档的贡献不均匀证据（Fig. 4）：不透明度排序下，前两档（约 40% splat）就接近质量饱和，后两档增益很小。这直接支持我们嵌套质量阶梯"基层承担大部分质量"的设计，并给出一个可操作的定层数方法：在我们自采场景上画"画质对已解码前缀比例"的曲线，找饱和点来选增强层层数和粗化步长，而不是按惯例定层数。验证实验成本很低（我们已有 bit-exact 前缀解码，只需按前缀比例渲染测 PSNR/VMAF）。
2. 排序代理量一变量对照的设计（Fig. 3 + Fig. 4）：同层数、同步长、只换排序量（scale 换 opacity）就能干净地证明哪种重要性排序更该用。我们层级序文件布局中"哪个 splat 排在字节流前面"同样是可换的代理量（我们当前按内容复杂度加渲染敏感度监督排序），可以把这个对照直接搬进我们的消融表。
3. 传输层与码流层的正交性叙事（期刊版相关工作可用）：MoQSplat 的 Subgroup=独立 QUIC 流与我们的"任意画质档都是单文件连续字节前缀"是同一需求（渐进、可按重要性先送）在网络层和码流层的两种解。网络侧方案不减少要传的字节总量（float32 裸传、场景几 GB 就发几 GB），码流侧方案把字节总量本身压下来且保持前缀可解码；两者正交，我们的码流可以直接作为 MoQ Object 的载荷。这条对照能把我们"为什么单文件渐进码流是必要的"讲得更清楚。
4. 视口相关的动态优先级思想：它按 6 自由度位姿实时改流的优先级。我们的分层顺序在训练期定死、与视口无关（这保证任何客户端拿到同一文件都正确解码，也满足零侧信息契约）；若引入视口相关排序会破坏可重算契约。只适合在期刊版讨论部分提一句（单文件前缀与视口自适应的张力），不建议采纳进方法。

## 与 DCCA-GS 正面对比

| 问题 | 它的做法 | DCCA-GS 的做法 | 差异与启示 |
| --- | --- | --- | --- |
| 量化怎么定 | 不量化：高斯属性按原始 float32 数组传输，理由是标准化 3DGS 压缩编解码器未定稿（第 III 节 MoQ Hierarchy Extension 段） | 嵌套质量阶梯：基层传全锚点粗量化值，增强层逐层补精度；分字段粗化档位；量化步长解码端从已解码数据重算 | 它完全不管字节总量，只管先送谁；我们把总量本身压下来。两层正交，可叠加 |
| 熵编码上下文 | 该论文不涉及（无熵编码） | 二值分解区间编码（零标志→符号→幅度逐位是非题）；层条件熵编码（细层用已解码粗层作上下文）；（贡献组×粗值幅度桶）复合条件表分组频率统计 | 完全错开的议题 |
| 渐进/可截断 | 流级渐进：一个场景拆成多条独立 QUIC 流（每质量档一条），可按流订阅、改优先级、取消；流内不可截断字节；队头阻塞靠 QUIC 多流消除 | 码流级渐进：单文件，任意画质档都是文件的连续字节前缀，真实可截断并做了 bit-exact 验证 | 同一需求在两层解决：它的最细粒度是"一条流"，我们的最细粒度是"字节"；它要求发布端预生成并缓存全部变体，我们一个文件携带全部档 |
| 侧信息 | 有，且必需：manifest（Catalog）单独成一条 Track 先行传输，含 Track 包围盒、Group 3D 质心、Subgroup 档位结构，没有它订阅端无法做视口适配 | 零侧信息契约：量化步长、编码上下文全部解码端从已解码数据重算；内容复杂度与渲染敏感度共享一个 8→32→3 小 MLP，零侧信息 | 方向相反：它把元数据整套先发，换服务端无状态和多订阅者效率；我们把元数据做成解码端可重算，换单文件自包含。流播场景下它的 manifest 相对场景体积可忽略，两种取向各有适用面 |
| 质量档划分依据 | scale 与 opacity 几何代理排序，20% 步长等量切 5 档（原型实现）；设计稿另提按属性拆层（基层几何加 0 阶球谐，高阶球谐进增强层） | 训练期渲染敏感度监督与内容复杂度（共享 MLP）决定分层；B3 阶梯对齐量化训练 | 它的排序只看几何量不看渲染误差（Fig. 4 的饱和现象正说明这种切法前重后轻）；我们的分层依据有渲染敏感度监督，与最终画质的距离更近 |
