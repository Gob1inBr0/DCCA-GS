# 《MoQSplat: Adaptive Progressive Streaming of 3D Gaussian Splatting via MoQ》精读笔记

| 项 | 内容 |
| --- | --- |
| arXiv | https://arxiv.org/abs/2609.18624 |
| 发表 | IEEE 出版物（具体会议论文未写明，版权页标注 © 2026 IEEE）；arXiv v1 2026-09-16，cs.MM |
| 作者/单位 | 奥地利克拉根福大学 ATHENA 实验室（Timmerer 组），长期参与 MPEG-DASH 和 MoQ 标准化（引文 [4] 的作者含该组成员） |
| 开源 | github.com/emanuele-artioli/MoQSplat（摘要声明） |

> 一句话定位：把 3DGS 场景按"空间块→语义物体簇→质量档→单个 splat 属性"四级切分，映射到 IETF Media over QUIC（MoQ）的四级内容层级上，每个质量档走一条独立 QUIC 流，订阅端按 6 自由度视锥、距离和注视对齐动态调各条流的优先级，实现没有连接级队头阻塞的自适应渐进流式播放；它不做任何压缩（属性按原始 float32 传输），省的是"先送哪些字节"的调度，不是字节总量。

## 方法要点

- 背景：MoQ 是 IETF 正在制定的低延迟交互媒体传输标准（Internet-Draft draft-ietf-moq-transport-18，引文 [10]），发布/订阅架构，内容按 Track / Group / Subgroup / Object 四级组织，可经中间 Relay 转发；已被用于点云流播 [11][12]，此前没有 3DGS 映射（论文自称 "the first framework to map 3D Gaussian Splatting (3DGS) content onto the Media over QUIC (MoQ) hierarchical content structure"，第 V 节；限于 3DGS 范围）。动机（第 I 节）：3DGS 场景由数百万 splat 组成、超过数 GB（最多 3 阶球谐每基元 45 个参数，引 [3]）；既有 3DGS 流式工作都建在 HTTP 自适应流播（HAS）上——LapisGS [5] 用不透明度优化做渐进分层、LTS [6] 把空间分块和 DASH 分段结合、L3GS [7] 用视口预测排物体顺序——TCP 有队头阻塞，HAS 分段粒度粗（"This combination of TCP's rigid reliability and HAS's coarse segment granularity is ill-suited to fine-grained 3DGS data."）。
- 系统论文，无训练期（不训练、不微调 3DGS 模型，直接拿现成场景切分）。属性按原始 float32 数组传输，论文给的理由："Gaussian attributes are encoded as raw float32 arrays, since a standardized 3DGS compression codec has not yet been finalized."（第 III 节）。
- 发布端 Publisher（第 III 节，离线一次性内容准备，五步）：① 从 N 个视角渲染内容，对渲染图跑 2D 物体检测提取 Group（检测器原文写 YOLOv26x，引文 [19] 为 YOLO-World，模型名与引文不一致，原文如此；每个检测框可选按预定义 padding 因子扩张，覆盖偏保守）；② 同一物体被多视角重复检测，用 3D Jaccard 指数（IoU）或关联 splat 质心接近度两种方式之一合并投票；③ 检测框投回 splat 空间，对物体质心跑 DBSCAN（引文 [16]）聚成簇，空间相近的 splat 归入同一 Group；④ 渐进质量档构建：实现设计中"独立几何启发式"——按 scale 和 opacity 两个代理量做几何剪枝排序，步长 20%，切成 5 个渐进质量档（基层 BL + 增强层 EL1–EL4），每档含 20% 的 splat；⑤ 复杂度：多视角渲染加 2D 检测为 O(N·T_YOLO)，投影回 3D 加 DBSCAN 为 O(M log M)，不透明度排序也是 O(M log M)；整条流水线完全离线，流播期间没有预处理延迟。设计稿另提出第二种未实现的划分：按属性拆层——基层 S1 传几何加 0 阶球谐（漫反射色），高阶球谐推迟到增强层，增强层渲染依赖 S1 几何（解码依赖，第 II.A 节）。
- 层级实现：moq-lite（Rust 写的 Pub/Sub 框架，跑在 QUIC 和 WebTransport 上）原本只支持 Track/Group/Object 三级，作者为它扩展出显式 Subgroup 层，每个 Subgroup 对应一条独立 QUIC 流；Track=不重叠空间块，Group=语义物体簇，Object=网络传输单元（封装单个 splat 的几何、opacity、球谐等属性）。Publisher 给 Subgroup 设静态基线优先级：MoQ 优先级是 [0, 255] 整数、0 最优先，基层设低数值，每个后续增强层依次升高；调度引擎先看订阅端指派的优先级，发布端静态优先级只作平手裁决（第 II.B 节）。manifest（论文称 Catalog）以 JSON 形式作为一条独立 MoQ Track 发布，内容包括 Track 标识、空间包围盒、Group 与 Subgroup 的组织结构、Group 的 3D 质心——它是必需的先决侧信息，订阅端的视口适配完全依赖这份元数据。
- 转发端 Relay：直接采用 moq-relay（moq-lite 参考实现），内容无关（content-agnostic，"No scene-aware scheduling"）、按优先级转发的 QUIC 转发服务；转发 SUBSCRIBE_UPDATE 等上游控制消息；拥塞由 Relay 隐式处理，高优先级 Subgroup 先发。Publisher 一次性把所有空间和质量变体发布上去，适配决策全部推给订阅端（stateless、subscriber-driven，第 II.A 节）。
- 订阅端 Subscriber（第 III 节，三个组件）：① 订阅层级与状态机——取回 Catalog 重建层级后，给每个 Subgroup 维护应用层状态机（MISSING / DOWNLOADING / CACHED）；DOWNLOADING 状态下丢包或超时，等超时阈值后回退 MISSING 重新发 SUBSCRIBE，此时渲染器回退到显示已缓存的基层 S1 和低编号增强层。② 视锥引导的优先级循环——100 毫秒间隔的异步控制循环，每轮取当前 6 自由度位姿，对 Catalog 里所有 Group 按式 (1) 算优先级分数 ΔP；MISSING 的带算好的 ΔP 发新 SUBSCRIBE，DOWNLOADING 的发 SUBSCRIBE_UPDATE 在不断连的情况下改这条流在线上的优先级；收到的 Object 拆包进本地内存池，一个 Subgroup 的 Object 收齐转 CACHED。③ 水位式显存换页——固定 GPU 显存预算 Mmax 和高水位阈值（如 0.9×Mmax），超阈值时保护所有基层 Subgroup 不被逐出，把增强层按当前 ΔP 排序、逐出优先级最低的增强层到系统内存；用户转动视角、某缓存对象的 ΔP 回升时再把数据提升回显存渲染。
- 兜底行为（第 II.B 节）：订阅端不用优先级算法时，所有 Subgroup 的订阅端优先级相同，调度退回 Publisher 的静态渐进优先级——所有可见簇的基层同时发，然后所有第一增强层，依次类推。
- 码流组织：没有单文件码流概念。所有切分决策在发布端定死并写进 manifest，客户端只算订阅优先级，不重算任何量化参数或编码上下文。一条 Subgroup 流内部是完整的 float32 数组，不可在流内截断；"截断"靠少订阅几条流实现，粒度最细到一条 Subgroup 流；队头阻塞靠 QUIC 多流互相独立消除。

全文只有一个公式，式 (1)（第 III 节，订阅端优先级循环）：

```
∆P = Wd·max(0, 1 − min(D, Dmax)/Dmax) + Wa·max(0, cos θ)    (1)
```

ΔP：一个 Group 的订阅优先级分数，值越大越该先送；D：订阅端相机位置到该 Group 3D 质心的欧氏距离；θ：相机前向视线向量与"相机指向 Group 质心"向量之间的夹角；Dmax：预期最大观看距离；Wd、Wa：两个缩放权重（取值论文未报告）。两项都截断在 0 以下（太远的物体和视野边缘的物体不贡献分数）。实际订阅和调优先级的单位是 Group 下的各 Subgroup。

## 实验口径

- 两阶段评估（第 IV 节）：先评发布端离线内容准备的质量，再评端到端流式传输在不同带宽下的表现。内容：Mip-NeRF360 数据集（引文 [20]）的 Bicycle 场景，两阶段共用；发布端评估聚焦从该场景聚出的 Bench–Bicycle 物体簇。单场景，没有第二个场景。
- 质量指标：VMAF（0 到 100 分，分越高画质越好）。相机沿 ±90° 方位角平滑环绕目标簇，固定距离、仰角和相机内参；每段 10 秒、30 帧/秒，共 300 帧；以未剪枝模型为参考评估剪枝后的模型。
- 硬件：Ubuntu 22.04，Intel Xeon Gold 5218 @ 2.30 GHz，双 NVIDIA Quadro GV100。
- 端到端协议：Publisher 预处理后发布到 Relay，四种带宽（10/30/50/100 Mbps）下记录送达 Subgroup 数、接收数据量、最终渲染 splat 数。
- 没有 PSNR/SSIM/LPIPS，没有码率-失真曲线，没有与 LapisGS/LTS/L3GS 的定量对比。Bicycle 场景的 splat 总数论文未报告，100 Mbps 收到约 151 万是否等于全集也未说明。
- 消融：没有正式消融表。能算对照实验的有两处：排序代理量替换对照（Fig. 3 + Fig. 4：同层数 5 档、同步长 20%、同渲染相机协议，只把排序量从 scale 换成 opacity）和带宽扫描（Table I）。缺的对照：无语义 Group 聚类有无的对比、无订阅端优先级算法有无的对比（只在第 II.B 节文字描述了兜底行为）、无与既有渐进 3DGS 方案的数值对比。

## 主结果表抄录（Table I，"Subscriber-driven Delivery Performance — Bicycle Scene"，第 IV 节）

共 4 行（4 档带宽），全抄：

| 链路带宽 (Mbps) | 送达 Subgroup 数 | 接收数据 (MB) | 最终渲染 splat 数 |
| --- | --- | --- | --- |
| 10 | 397 | 37.4 | 224,710 |
| 30 | 2,432 | 337.8 | 1,400,909 |
| 50 | 2,628 | 363.2 | 1,491,950 |
| 100 | 2,853 | 390.6 | 1,512,214 |

正文引用的关键数字（第 IV 节）：10 Mbps 下约 22.5 万 splat，"粗糙但可持续渲染"；30 Mbps 下 Subgroup 从 397 增到 2,432、splat 从约 22.5 万增到 140 万；带宽从 50 Mbps 翻倍到 100 Mbps，最终渲染的 splat 只从约 1.49M 增到约 1.51M。

Fig. 4（"Progressive quality enhancement measured by VMAF"，第 IV 节）：横轴 Q0–Q4 五档（Q0=BL，Q4=BL+EL1+…+EL4），纵轴 VMAF。曲线目测近似值（图上读数，正文未给具体数值，原文图表以 PDF 为准）：不透明度排序 Q0 约 41 → Q1 约 76 → Q2 起约 100 饱和；scale 排序 Q0 约 28 → Q1 约 40 → Q2 约 56 → Q3 约 81 → Q4 才到 100。不透明度排序在前两档（Q1，约 40% splat）就接近饱和，之后增益很小（第 IV 节 Fig. 3、Fig. 4）。正文结论（第 IV 节）："opacity-based pruning consistently outperforms scale-based pruning by achieving higher perceptual quality with the same amount of rendered splats"；且 "gains diminish under opacity-based pruning as quality saturates beyond Q2 (Fig. 4)"。论文还指出各层 splat 数相同但感知贡献差别很大（第 IV 节 "Enhancement Layer Analysis" 段）。
