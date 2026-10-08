# DCCA-GS 中文审查总结与无人机高斯编码优化思路

日期：2026-10-04

本文总结本轮对 DCCA-GS 代码、现有 md 文档、2026 年高斯压缩新论文的检查结果，并重点回答两个问题：

1. 现在性能差的主要原因在哪里；
2. 新论文里有哪些思路可以借鉴到无人机高斯点云编码，尤其是怎么改熵模型。

## 1. 本轮阅读范围

### 代码与项目文档

本轮重点读取了：

- `README.md`
- `train.py`
- `scaffold_gs/config.py`
- `scaffold_gs/trainer.py`
- `scaffold_gs/hacpp.py`
- `scaffold_gs/hac_core.py`
- `scaffold_gs/codec.py`
- `scripts/runner_phg_cell.sh`
- `scripts/c25_real_bitstream.py`
- `docs/03-reports/创新点论文汇报_20261003.md`
- `docs/03-reports/全流程审查与创新方向_报告_20261001.md`
- `docs/data/experiments.csv`
- `docs/07-literature/` 下的 2026 年压缩相关精读笔记

### 外部论文与网页

检索和核对的主要新论文/页面包括：

- PCGS: Progressive Compression of 3D Gaussian Splatting, AAAI 2026  
  https://ojs.aaai.org/index.php/AAAI/article/view/37304
- ProGS: Towards Progressive Coding for 3D Gaussian Splatting, arXiv:2603.09703  
  https://arxiv.org/abs/2603.09703
- SCAR-GS: Spatial Context Attention for Residuals in Progressive Gaussian Splatting, arXiv:2601.04348  
  https://arxiv.org/abs/2601.04348
- SpeedyGS: Content-Aware 3D Gaussian Splatting Compression via Two-Stage Optimization, arXiv:2607.12656  
  https://arxiv.org/abs/2607.12656
- GoDe: Gaussians on Demand for Progressive Level of Detail and Scalable Compression, arXiv:2501.13558  
  https://arxiv.org/abs/2501.13558
- QuantizationGS, Computers & Graphics 2026  
  https://www.sciencedirect.com/science/article/pii/S0097849326000671

另外结合仓库已有精读笔记，重点吸收了这些 2026 工作：

- Practical Compression / COSA-GS：`docs/07-literature/22_PracticalCompression_2609.30245.md`
- SpeedyGS：`docs/07-literature/20_SpeedyGS_2607.12656.md`
- Non-Uniform Quantisation：`docs/07-literature/21_NonUniformQuant_2608.28272.md`
- Cross-Representation Priors / CRP-GS：`docs/07-literature/23_CrossReprPriors_2609.23005.md`
- CoSAG：`docs/07-literature/24_CoSAG_2607.10237.md`
- GS-PQM：`docs/07-literature/31_GS-PQM_2610.00195.md`
- MoQSplat：`docs/07-literature/33_MoQSplat_2609.18624.md`
- RD Primitive Selection / OIC-GS：`docs/07-literature/36_RDPrimSelect_2609.34367.md`
- QuARC-GS：`docs/07-literature/42_QuARC-GS_2608.18285.md`

说明：Google Scholar 页面本身不适合自动抓取，因此本轮采用普通 web 检索、arXiv/AAAI/出版社页面和本仓库精读笔记交叉核对。

## 2. 当前 DCCA-GS 代码状态

当前代码是一个基于 HAC++ / Scaffold-GS / gsplat 的锚点式高斯压缩系统。

主流程如下：

1. `train.py train` 构建 `hac_pp` 模型。
2. `scaffold_gs/trainer.py` 训练图像重建损失、码率损失、SPA 稀疏化、可选敏感度监督、语义监督和 B3 阶梯对齐损失。
3. `scaffold_gs/hacpp.py` 中的 `HACPlusModel` 包装 HAC++ core，负责量化噪声、码率估计、内容自适应量化步长、B3 粗层格点惩罚、压缩/解码。
4. `scripts/runner_phg_cell.sh` 负责 train -> compress -> decoded eval，并写 `provenance.txt`。
5. `scripts/c25_real_bitstream.py` 把已量化属性重新组织成真实区间编码的渐进分层码流，得到各前缀的字节数和画质。

当前重要默认值：

- `content_aware_quant=True`
- `spa_enabled=True`
- `spa_ratio=0.85`
- `mini_splat_enabled=True`
- `coarse_ladder_align=False`
- `feat_dim=32`
- `n_offsets=10`

已有但还需要继续打磨的实验开关：

- B3 粗层格点对齐：`coarse_ladder_align`
- 码率感知稀疏化：`spa_rate_aware`, `spa_rate_tau`, `spa_bit_budget`
- holdout 质量门控：`spa_holdout_gate`
- Fisher / sensitivity 变体：`sensitivity_second_order`, `sensitivity_use_fisher`
- 背景 feature codebook：`bg_codebook_enabled`
- 重要性加权重建损失：`importance_weighted_loss`

## 3. 当前性能差的主要原因

### 3.1 低码率瓶颈不是几何，而是基层属性熵

现有文档和审计结果已经把一个关键误区排除了：P0 起点下限不是由几何坐标主导。

现有 1-78 场景分层字节归因大致为：

- 基层属性：约 14.25 MB，占 86%
- 几何坐标：约 1.29 MB，占 8%
- 固定负载：约 1.06 MB，占 6%

因此，单独替换 G-PCC 或几何编码器不会解决当前低码率问题。无人机大场景当然需要更好的空间组织，但在当前 DCCA-GS 的低码率前缀里，真正要优先压的是属性，尤其是 feature / offset / scaling 的基层符号。

### 3.2 锚点数量是当前最有效的低码率杠杆

已有预算扫描显示，把 `spa_ratio` 从 0.85 往 0.70 / 0.60 / 0.50 降，可以把 P0 从 15.88 MB 压到 6.33 MB，且起点画质仍可用。

这说明锚点预算是当前最可靠的低码率旋钮。无人机高斯数据通常空间范围大、冗余区域多，预算轴比单纯细调量化步长更值得先做。

但现在 SPA 主要按“锚点数量”控预算，没有充分考虑每个锚点实际要花多少属性字节。因此下一步应从“数量预算”升级为“码率预算”。

### 3.3 B3 有帮助，但它是符号代理，不是真实 P0 画质优化

B3 的逻辑是把连续符号推向粗层格点，减少基层误差和增强层残差。但它没有直接渲染 P0 模型，也没有直接优化“用户最先看到的画面”。

已有结果表明：

- 紧预算下 B3 可以显著抬高 P0；
- 强权重可能损害满精度 P3；
- 去掉 scaling 字段没有消除满精度代价，说明代价来自粗层惩罚与重建损失的整体冲突。

因此，B3 下一步不应只继续改字段权重，而应改成“基层真实渲染损失”。

### 3.4 当前熵模型保守可靠，但还不够强

`scripts/c25_real_bitstream.py` 里当前主要是统计表式条件编码：贡献组、粗值幅度桶、层间残差等。这个路线稳定、可解释、解码端容易复现，但它对高维 feature 的跨通道、跨锚相关性利用不足。

脚本里已有 S4/S5 学习式概率头原型，但这类方案有一个论文叙事风险：如果每场景训练一个概率模型并随码流传输，就会削弱“零侧信息”主张。更合适的路线是固定通用模型或完全由已解码符号重算的上下文。

## 4. 新论文可借鉴点

### 4.1 PCGS：数量渐进 + 质量渐进

PCGS 已经明确做了 progressive masking 和 progressive quantization，因此 DCCA-GS 不能再说“首个渐进 3DGS 压缩”。

可借鉴点：

- 渐进码流不只可以细化量化精度，也可以逐步加入锚点/高斯数量；
- 数量层和质量层可以一起设计，而不是只做属性 bit-plane。

对 DCCA-GS 的启示：

- 当前拓扑一次到位、属性逐层细化。未来可做“极低码率拓扑基层”：只传最重要锚点 + 粗属性，增强层补剩余锚点和精度。
- 但这会带来结构字节，不能一开始就替换主线。建议作为 UAV 低带宽传输扩展，而不是当前论文主贡献。

### 4.2 ProGS：八叉树/拓扑渐进

ProGS 用 octree 组织 3DGS，做 streaming-friendly progressive coding，是 DCCA-GS 在“渐进锚点压缩”叙事上的直接竞品。

可借鉴点：

- parent-closed / octree prefix 的结构渐进对流式传输很自然；
- 对无人机场景，空间层级和视距层级很重要。

对 DCCA-GS 的启示：

- DCCA-GS 应把定位收窄到“零侧信息、可截断前缀、属性精度渐进”；
- UAV 应用上可以把 ProGS 式空间层级和 DCCA-GS 属性前缀结合：空间 tile / octree 节点内存 DCCA 分层码流。

### 4.3 SCAR-GS：残差向量量化与空间注意力

SCAR-GS 认为 scalar quantization 难以捕获高维 feature residual 的相关性，改用 residual vector quantization 和空间上下文注意力。

可借鉴点：

- 当前 DCCA-GS 的 feature 基层属性字节占比高，逐标量编码很可能浪费了跨通道相关性；
- vector / product quantization 对 feature 字段尤其有价值。

对 DCCA-GS 的启示：

- 不能直接照搬每场景 RVQ 码本，否则会引入码本侧信息；
- 更适合做“通用固定变换/固定码本”：离线在多场景训练一个通用 feature transform 或 product codebook，作为解码器固定资产，不随场景传输；
- 然后仍用现有分层标量熵编码去编变换后的系数。

### 4.4 SpeedyGS：结构成形和统计编码解耦

SpeedyGS 的核心思想是把“结构成形”和“统计编码”拆开。结构阶段不用真实熵模型，而用轻量码率代理驱动可学习量化和剪枝。

可借鉴点：

- 当前 DCCA-GS 已有 rate loss，但锚点选择仍偏数量预算；
- 对无人机大场景，大量 sweep 不可能每次都跑昂贵熵编码探针。

对 DCCA-GS 的启示：

- 给 SPA 增加轻量码率代理：每锚点的 feature/offset/scaling 动态范围、估计 Q、mask 数量、贡献面积一起估算 bits；
- 先用这个代理筛 anchor，再用真实码流评估最终候选；
- 对应到代码，就是完善 `spa_rate_aware` / `spa_bit_budget`，把它从试验开关变成主路径。

### 4.5 COSA-GS / Practical Compression：锚内因果上下文 + 整数推理

这篇对 DCCA-GS 改熵模型最关键。它不靠复杂空间聚合，而是把上下文压成锚点内部的因果链，并把概率推理整数化，保证跨平台熵解码一致。

可借鉴点：

- 昂贵空间上下文不一定必要；
- 坐标上下文 + 小潜变量 + 属性因果顺序，可以有很强码率收益；
- 概率模型必须定义清楚哪些符号跨平台逐位一致；
- 8-bit QAT、定点重建、共享 CDF 表可以把浮点概率路径变成可部署路径。

对 DCCA-GS 的启示：

- 做一个“DCCA-Causal”熵模型：字段顺序为 coordinate/morton context -> scaling -> offset mask/offset -> feature，后续字段用已解码前序字段预测；
- 均值/尺度预测模型可以先固定为通用小 MLP，不随场景传输；
- 如果用 MLP，必须整数化或离线量化，不然会损害 bit-exact 解码。

### 4.6 Non-Uniform Quantisation：重要性加权非均匀量化

这篇从渲染方程推导每个高斯的重要性，再做加权 Lloyd-Max 非均匀量化。

可借鉴点：

- 重要性可以从 alpha compositing 中推导，不只是经验 opacity；
- 重要性 MLP 很小，能用 scale、opacity、position 近似；
- 非均匀量化不一定要改熵编码后端。

对 DCCA-GS 的启示：

- 当前内容自适应 Q 已有复杂度 MLP，但可加入更直接的 UAV 渲染贡献权重；
- 对无人机航拍，地面大平面/天空/远处区域的重要性分布很不均匀，统一粗化很浪费；
- 可以先只在编码端做重要性分桶：高贡献锚点用细基层，低贡献锚点用粗基层，分桶规则由解码端从已解码属性近似重算，避免传 importance map。

### 4.7 CRP-GS：跨锚根-叶条件熵模型

CRP-GS 用锚特征相似度建立根-叶链接：先编码根锚，再用已解码根特征条件编码叶锚。

可借鉴点：

- 相关锚不一定空间相邻；无人机场景中重复纹理、道路、建筑立面、农田块会产生长程相似；
- 已解码根特征可以作为叶特征的均值预测上下文；
- 一层依赖比多层层级更稳，不容易传播量化误差。

问题：

- CRP-GS 需要传链接向量和掩码，属于侧信息；
- 链接若每场景搜索并传输，会削弱 DCCA-GS 的零侧信息主张。

对 DCCA-GS 的折中方案：

- 不传任意链接，改成“Morton 邻域 + hash cell 代表锚 + 粗层同桶代表锚”的确定性根选择；
- 根选择规则由解码端可重算：例如同一空间 cell 内先出现/贡献最大的锚为 root，其余为 leaf；
- leaf 的 feature residual 用 root 的已解码粗 feature 预测。

### 4.8 CoSAG：Morton 排序暴露低熵结构

CoSAG 压的是语义场，不是 3DGS 本体，但它有一个很适合 DCCA-GS 的思想：先把符号沿 Morton 空间曲线重排，让分片常数/局部相似性变成长游程，再用简单熵编码吃掉。

可借鉴点：

- 不一定要先上神经熵模型；
- 排序和上下文定义本身就是熵模型的一部分；
- 父指针/层间众数这种确定性查表结构可以很便宜。

对 DCCA-GS 的启示：

- 目前已有 Morton sort，但可以更激进：按 UAV 地块/tile/cell 分组后，在组内按 field、贡献等级、粗符号 bucket 重排；
- 对 mask/offset zero flag，先试运行长编码或二值上下文模型，可能比复杂 MLP 更稳。

### 4.9 OIC-GS / RD Primitive Selection：已解码渲染状态上下文

这篇虽然是全景图像高斯编码，不是 3D 场景压缩，但它的熵模型非常值得借鉴。

关键思想：

- 熵模型不查随码流传输的 hash grid；
- 它用“已解码粗层渲染状态”作为上下文；
- 上下文由解码端逐层重算，天然零侧信息；
- 特征流占主要码率，因此直接优化每个 primitive 的码长，并让不透明度在率失真目标中自动归零。

对 DCCA-GS 的启示：

- DCCA-GS 的 progressive ladder 本身就有已解码粗层，完全可以用 L0/L1 的粗层属性作为 L1/L2/L3 的概率上下文；
- 现在的粗值幅度桶是手工表；可以升级为“粗层重建状态 -> residual 分布参数”的固定小网络或分桶表；
- 对无人机大场景，还可以用低分辨率 tile 渲染/覆盖状态作为上下文，编码增强层时预测哪里需要细节。

这是最适合 DCCA-GS 当前主张的熵模型方向。

### 4.10 QuARC-GS：零 bin 与确定性绑定

QuARC-GS 是动态场景，但两个思想可以借鉴：

- 小残差直接量化到零 bin，让熵编码自动吃掉静态/低变化区域，不传显式 mask；
- 锚点绑定由 canonical geometry + seed 确定性重算，不传绑定图。

对 DCCA-GS 的启示：

- 对 offset residual 或 enhancement residual，可以设计更强的 zero-bin dead-zone；
- 对 UAV tile 内 anchors，可用确定性 cell binding 替代某些显式分组信息。

### 4.11 GS-PQM：快速质量代理

GS-PQM 不做编码，但可作为筛选工具。

对 DCCA-GS 的启示：

- 150-view 评测太慢，尤其无人机大场景实验矩阵很大；
- 可以先用参数域质量 proxy 筛掉明显坏的预算/量化候选，再跑完整视角评测；
- 不能把它作为最终论文指标，但可做工程加速。

### 4.12 MoQSplat：传输层与 tile 化

MoQSplat 不做压缩，属性还是 float32；它主要讲如何把 3DGS 映射到 MoQ/QUIC 多流传输。

对 DCCA-GS 的启示：

- 无人机场景最终不是只需要一个小文件，还需要空间可访问；
- 可以把 DCCA-GS 单文件前缀结构放进 tile/subgroup；
- 质量轴由 DCCA 前缀解决，空间/视口轴由 tile manifest 解决。

## 5. 最建议改的熵模型：DCCA-ZCausal

这里给出一个最适合 DCCA-GS 当前论文主张的熵模型方向，暂命名为 DCCA-ZCausal：Zero-side-information Causal Entropy Model。

目标：

- 提升属性熵编码效率，尤其是 feature 和 offset；
- 保持单文件前缀可截断；
- 不传每场景熵模型、不传 hash grid、不传链接表；
- 编码端与解码端上下文完全一致；
- 后续可以整数化，保证跨平台 bit-exact。

### 5.1 编码顺序

建议顺序：

1. 几何坐标 / Morton 顺序；
2. scaling 粗层；
3. offset mask / offset 粗层；
4. feature 粗层；
5. scaling enhancement；
6. offset enhancement；
7. feature enhancement。

理由：

- scaling / offset 直接影响空间形状，可作为 feature 残差上下文；
- feature 字段维度大，最需要前序上下文；
- progressive enhancement 可以用已解码 coarse symbol 做条件。

### 5.2 上下文信号

全部上下文必须解码端可重算：

- field id：feat / scaling / offset；
- ladder level：L0/L1/L2/L3；
- Morton 邻域位置；
- 当前锚点坐标 cell；
- 已解码 coarse symbol；
- coarse symbol 的绝对值 bucket；
- 已解码 scaling bucket；
- offset mask 活跃数；
- 同 cell 或相邻 cell 的前序锚统计；
- 可选：低分辨率 tile 的 anchor density / opacity coverage。

这些上下文都来自已解码信息，不需要随码流传 per-scene 模型。

### 5.3 三个实现版本

#### 版本 A：纯表模型，最稳

把当前 `(贡献组 × 粗值幅度桶)` 扩展为：

```text
field × level × contribution_group × abs(coarse_bucket)
× scaling_bucket × offset_activity_bucket × local_density_bucket
```

优点：

- 不破坏零侧信息；
- 解码简单；
- 最容易做 bit-exact；
- 与现有 `c25_real_bitstream.py` 兼容。

缺点：

- 表会稀疏，需要 fallback；
- 对高维 feature 的连续相关性仍有限。

优先级：最高。建议先做。

#### 版本 B：固定通用小网络，性能更强

离线跨场景训练一个通用小 MLP，输入上述上下文，输出二值分解各阶段概率：

- nonzero flag；
- sign；
- magnitude continuation。

模型权重固定在代码/解码器中，不随场景传输。这样它不是 per-scene side information。

优点：

- 比表模型泛化更好；
- 可吸收 OIC-GS / COSA 的思路；
- 后续能做 int8 QAT。

缺点：

- 需要定义版本号；
- 论文中必须说明固定模型不计入 per-scene bitstream；
- 要做跨平台一致性测试。

优先级：第二阶段。

#### 版本 C：整数化固定网络，投稿级工程亮点

借鉴 COSA-GS：

- MLP 权重 int8；
- 激活 int8；
- 累加 int32；
- sigma / probability 查固定 CDF 表；
- GELU/SiLU 换查表或直接用 ReLU/分段线性；
- 所有概率最终离散到固定精度 CDF。

优点：

- 可主张跨平台 entropy-symbol 一致；
- 工程上比“浮点概率模型”更可信。

缺点：

- 开发成本高；
- 需要至少两套硬件/软件环境做 45/45 或类似一致性验收。

优先级：论文二阶段增强，不建议第一步直接做。

### 5.4 推荐先改哪个字段

按当前瓶颈，优先级为：

1. feature L0 和 feature enhancement residual；
2. offset enhancement residual；
3. scaling；
4. mask / flag。

feature 是最大头，先做 feature 才可能明显改善 P0。

### 5.5 建议的最小实验

先做一个离线 probe，不改训练：

1. 对已有 run 导出量化符号；
2. 用当前 `c25_real_bitstream.py` 统计每个字段、每层、每阶段字节；
3. 加入新上下文 bucket；
4. 只重新估计/重编码 feature 字段；
5. 对比真实 range-coded bytes；
6. 如果 feature 字段总字节下降 >= 8%，再接入完整分层解码评测。

预注册门：

- feature 字节下降 >= 8%；
- total progressive stream 下降 >= 5%；
- P0/P1/P2/P3 解码画质不变；
- final prefix 仍与生产解码 bit-exact。

### 5.6 为什么它适合无人机数据

无人机高斯数据有几个特点：

- 空间范围大；
- 地面、道路、植被、建筑重复结构多；
- 同一纹理类型在空间上可能长程重复；
- 相邻 tile 的 anchor 密度和属性分布强相关；
- 低码率应用更看重先传可用预览。

DCCA-ZCausal 可以利用：

- Morton 邻域：吃局部连续性；
- cell/tile density：吃航拍场景空间分布；
- coarse-to-fine residual：吃渐进层已有信息；
- scaling/offset -> feature 条件：吃几何到外观的因果关系；
- 固定通用模型：避免每个无人机场景都传模型。

这比“每场景训练一个熵模型并传权重”更适合 UAV 部署。

## 6. 其他推荐创新方向

### 6.1 Rate-aware SPA：把锚点预算改成比特预算

当前 `spa_ratio` 控制锚点数，不直接控制码率。建议完善：

- `spa_rate_aware=True`
- `spa_bit_budget=True`
- `bits_ema` 作为每锚点预算估计；
- selection score 改成 importance / bits^tau；
- 加 coverage constraint 防止 UAV 大平面出现洞。

目标：

- 同 P0 MB 下 PSNR +0.2 dB；
- 或同 PSNR 下 P0 bytes -8%。

### 6.2 Base-layer rendering loss：直接优化 P0

把 B3 从符号代理升级为真实 P0 渲染损失：

1. 训练末期构造 L0 反量化属性；
2. 用 L0 属性渲染少量训练视角；
3. 加低权重 L1/SSIM loss；
4. 同时保留 full precision loss。

目标：

- P0 +0.4 dB；
- P3 损失不超过 0.2 dB。

### 6.3 通用 feature transform / fixed PQ

借鉴 SCAR-GS / vector quantization，但避免 per-scene codebook：

- 跨场景学一个固定 feature transform；
- 或固定 product quantization codebook；
- 码流只传索引/残差，不传码本；
- 与分层 residual 结合。

目标：

- feature L0 字节 -12%；
- P0 PSNR 损失 < 0.1 dB。

### 6.4 非均匀量化基层

借鉴 Non-Uniform Quantisation：

- 用 opacity、scale volume、position、coverage 估计锚点重要性；
- 不传重要性图；
- 解码端可从已解码属性近似重算 bucket；
- 高重要性锚点基层更细，低重要性锚点基层更粗。

注意：

- 必须保证嵌套阶梯仍可解；
- 量化表不要每场景传大表，否则不适合 DCCA 主张。

### 6.5 UAV tile + progressive prefix

对 PKU-GS 这种无人机数据，最终应做：

- 空间 tile；
- 每 tile 一个 DCCA progressive stream；
- manifest 记录 tile bbox、prefix offsets、邻接关系；
- 传输端按视锥/距离/任务优先级调 tile。

这不是替代熵模型，而是系统层扩展。

## 7. 建议的实验顺序

推荐按风险从低到高执行：

1. **离线熵模型 probe**：扩展 `c25_real_bitstream.py` 的上下文 bucket，先看 feature bytes 能不能降。
2. **Rate-aware SPA**：启用/修正 `spa_rate_aware` 和 `spa_bit_budget`，跑 `spa_ratio=0.85/0.70/0.60/0.50` 对照。
3. **Base-layer rendering loss**：替代或增强 B3。
4. **固定通用小网络熵模型**：跨场景训练，固定进代码，不进码流。
5. **整数化概率路径**：作为论文工程亮点。
6. **UAV tile 封装**：做系统演示和应用扩展。

## 8. 本轮已写入的文件

以 2026-10-04 打包的最终状态为准。脚本最后一轮整理进了 `scripts-new/` 编号目录；本文件早先草稿里提到的 `scripts/run_dcca_train.sh`、`scripts/run_dcca_layered_eval.sh`、`scripts/run_dcca_ablation_matrix.sh` 三个名字在仓库和 git 历史里都不存在，实际对应 `scripts-new/` 的 00、01、03。

最终文件清单：

- `README.md`（整体重写为英文）
- `summary.md`（本文件；仓库根另有一份简短索引）
- `scaffold_gs/config.py`、`scaffold_gs/hacpp.py`、`scaffold_gs/trainer.py`（P0 渲染损失，见 §15.3）
- `scripts/c25_real_bitstream.py`（新增 `--zc-context` / `--zc-cell-size`，见 §10、§15.1）
- `scripts/zc_context.py`（新增，纯 NumPy 上下文函数）
- `tests/test_zc_context.py`（新增，CPU 单测）
- `scripts-new/`（新增编号入口目录：00 训练、01 分层码流、02 legacy/ZC 对比、03 消融矩阵、04 报告对比、05 码流入口、06 训练 runner、07 解码评估）

实现唯一来源原则：码流编解码只在 `scripts/c25_real_bitstream.py` + `scripts/zc_context.py`，训练 runner 只在 `scripts/runner_phg_cell.sh`，解码评估只在 `scripts/eval_decoded.py`；`scripts-new/` 里的 05/06/07 是委托入口，`lib_zc_context.py` 是兼容垫片，不再维护第二份拷贝。

## 9. 一句话结论

DCCA-GS 当前最值得投入的不是继续换几何编码，而是改属性熵模型和锚点预算：先做 decoder-reproducible 的 coarse-to-fine 因果熵模型，再做 rate-aware SPA 和基层真实渲染损失。这样既能针对无人机场景的低码率瓶颈，又不会牺牲“零侧信息、可截断前缀、bit-exact 解码”这条论文主线。

## 10. 组合新方案：UAV-DCCA-ZC

下面把上面几篇论文的可借鉴点组合成一个新的方案，暂命名为 **UAV-DCCA-ZC**：

> **UAV-DCCA-ZC: Zero-side-information Causal Progressive Compression for UAV Anchor Gaussian Splatting**

中文名可写作：

> **面向无人机锚点高斯的零侧信息因果渐进压缩**

它不是单独改一个模块，而是把“锚点预算、基层量化、因果熵模型、P0 真实渲染损失、空间 tile 传输”组合成一套适合无人机大场景的编码框架。

### 10.1 核心主张

UAV-DCCA-ZC 的主张是：

1. **码流仍然是单文件可截断前缀**：P0/P1/P2/P3 都是连续字节前缀。
2. **不传每场景熵模型**：不传 hash-grid 概率模型、不传 per-scene MLP、不传 root-leaf 任意链接表。
3. **熵模型上下文全部由已解码信息重算**：coarse symbol、Morton 邻域、local density、scaling/offset 状态、tile id 等都可在解码端得到。
4. **低码率第一目标是无人机预览质量**：直接优化 P0，而不是只优化最终 P3。
5. **锚点预算按 bit 而不是按数量控制**：稀疏化阶段知道哪些锚点贵、哪些锚点便宜。

一句话概括：

> 用 rate-aware SPA 决定“保留哪些锚点”，用 importance-aware base quantization 决定“基层给谁更多精度”，用 zero-side causal entropy model 决定“符号怎么更短”，用 base-layer rendering loss 决定“最先解出来的画面好不好”。

### 10.2 借鉴来源与组合方式

| 来源论文 | 借鉴点 | 在 UAV-DCCA-ZC 中怎么用 |
| --- | --- | --- |
| SpeedyGS | 结构成形和统计编码解耦、轻量码率代理 | 训练期用每锚点 bit proxy 做 rate-aware SPA |
| Non-Uniform Quantisation | 渲染贡献重要性、非均匀量化 | 基层量化按重要性分桶，不重要锚点更粗 |
| COSA-GS | 锚内因果熵模型、整数推理 | 用固定通用小模型或表模型预测 residual 概率 |
| OIC-GS / RD Primitive Selection | 已解码粗层状态作为熵上下文、零侧信息 | 用 L0/L1 已解码符号预测 enhancement residual |
| CRP-GS | 根-叶条件熵模型 | 不传链接表，改成确定性 cell root，leaf 用 root 条件编码 |
| CoSAG | Morton 排序暴露低熵结构 | tile/cell 内 Morton 重排和游程/二值 flag 上下文 |
| PCGS / ProGS | 渐进数量与质量组织 | 未来扩展为空间 tile + 属性前缀双渐进 |
| GS-PQM | 快速质量代理 | 用参数域 proxy 筛选大规模 ablation |

### 10.3 总体编码流程

#### 阶段 A：训练期 rate-aware 锚点成形

目标：不要只保留“看起来重要”的锚点，而是保留“单位 bit 贡献高”的锚点。

训练时为每个锚点维护：

```text
importance_i      渲染贡献 / 敏感度 / coverage
bits_i            feature + scaling + offset + mask 的 EMA 码率估计
density_i         局部空间密度
tile_i            UAV 空间 tile id
score_i = importance_i / (bits_i + eps)^tau
```

SPA 投影时按 `score_i` 排序，而不是只按 ADMM score 或 anchor count。

建议先实现两档：

- `tau=0.5`：温和惩罚高码率锚点；
- `tau=1.0`：完整 importance-per-bit。

保留 coverage constraint：

```text
每个粗空间 cell 至少保留 1 个 anchor
每个 UAV tile 至少保留若干 anchor
```

防止无人机大场景出现地面洞、建筑边缘断裂。

#### 阶段 B：重要性感知基层量化

目标：P0 的 bit 不平均分，而是优先给对画面贡献大的锚点。

先为每个锚点计算一个解码端可近似重算的重要性 bucket：

```text
imp_bucket_i = bucket(
  opacity/coverage,
  scale_volume,
  local_density,
  distance_to_tile_center,
  sensitivity_ema 可选
)
```

严格零侧信息版本不传 `imp_bucket_i`，而是用已解码或训练/解码都能得到的属性近似重算：

- anchor coordinate；
- scaling；
- mask 活跃数；
- coarse opacity 或 coverage proxy；
- local density。

基层量化倍率改成：

```text
高重要性锚点：feat/offset 4x 或 8x
中重要性锚点：feat/offset 8x
低重要性锚点：feat/offset 16x
scaling 单独保守处理：2x / 4x
```

注意：倍率必须写成 deterministic rule，不能每场景传表。

#### 阶段 C：DCCA-ZCausal 因果熵模型

这是新方案的核心。

##### C1. 当前问题

当前分层码流主要用：

```text
贡献组 × 粗值幅度桶
```

它稳，但不够利用无人机场景的结构：

- 同一条路、屋顶、农田、立面会长程重复；
- feature 通道之间相关；
- offset/scaling 与 feature 残差有关；
- coarse layer 已经提供了很强上下文，但现在只粗略使用。

##### C2. 新上下文

对每个待编码 residual symbol，构造：

```text
ctx = {
  field_id,                 # feat / scaling / offset
  level_id,                 # L0/L1/L2/L3
  channel_group,
  morton_parity_or_order,
  tile_id_or_tile_bucket,
  local_density_bucket,
  coarse_abs_bucket,
  coarse_sign,
  scaling_bucket,
  offset_activity_bucket,
  root_residual_bucket,
  previous_symbol_bucket
}
```

这些都必须来自：

- 已解码 coarse symbols；
- anchor 坐标；
- deterministic Morton / tile 规则；
- 已解码前序字段。

##### C3. 确定性 root-leaf 条件

借鉴 CRP-GS，但不传链接表。

对每个 tile/cell：

```text
root anchor = 该 cell 内 Morton 顺序最早的 active anchor
或 root anchor = coarse feature 能量最大的 anchor
leaf anchor = cell 内其他 anchors
```

编码 leaf feature residual 时，用 root 的已解码 coarse feature 做上下文：

```text
leaf_residual = q_leaf - predict(q_root, coarse_leaf, scaling_leaf, offset_state)
```

预测函数第一版不用神经网络：

```text
predict = root coarse value 的分桶中位数 / 简单差分
```

第二版再改成固定通用小 MLP。

##### C4. 三级实现

第一版：**增强表模型**

```text
P(nonzero/sign/magnitude_continue)
  = table[field, level, channel_group, coarse_bucket,
          scaling_bucket, offset_activity, density_bucket]
```

优点：容易接进 `c25_real_bitstream.py`，不改变主代码。

第二版：**固定通用小 MLP**

输入同上，输出二值分解概率。模型跨场景离线训练，固定进解码器，不随码流传输。

第三版：**整数化小 MLP**

用 int8 权重、int32 累加、固定 CDF 表，主张跨平台 entropy-symbol bit-exact。

### 10.4 阶段 D：基层真实渲染损失

目标：训练时直接优化 P0 画面，而不是只把符号推向粗格点。

在最后 4k-6k 步加入：

```text
L = L_full_render
  + lambda_rate * R
  + lambda_p0 * L_render(P0_dequantized_attributes)
  + lambda_ladder * L_symbol_alignment
```

建议先弱化或替代 B3：

```text
lambda_p0 = 0.05 ~ 0.15
lambda_ladder = 0.0 ~ 0.01
```

每 N 步只对 1 个训练视角计算 P0 渲染，控制成本。

判断标准：

- P0 PSNR +0.4 dB；
- P3 PSNR 损失 <0.2 dB；
- P0 bytes 不增加超过 3%。

### 10.5 阶段 E：UAV tile + 前缀传输

编码层仍然是 DCCA-ZCausal，但无人机场景最终要支持空间访问。

文件组织：

```text
scene.dccazc
  header
  global decoder/version info
  tile manifest
  tile_000:
    geometry
    P0 attributes
    P1 residual
    P2 residual
    P3 residual
  tile_001:
    ...
```

两个渐进轴：

- 质量轴：P0 -> P1 -> P2 -> P3；
- 空间轴：近处/视锥内 tile -> 远处 tile。

这部分适合无人机应用展示，但不是第一阶段核心论文贡献。

## 11. UAV-DCCA-ZC 的具体算法描述

### 11.1 训练目标

训练目标可写成：

```text
L = D_full
  + lambda_rate * R_est
  + lambda_p0 * D_p0
  + lambda_spa * L_spa_rate
  + lambda_reg * L_scale
```

其中：

- `D_full`：最终满精度渲染损失；
- `R_est`：HAC++ 码率估计；
- `D_p0`：基层 P0 反量化渲染损失；
- `L_spa_rate`：rate-aware SPA 约束；
- `L_scale`：scale regularization。

### 11.2 码流结构

单 tile 或全场景码流：

```text
Header
Geometry
Static tables/version id
Layer P0:
  scaling_L0
  offset_mask_L0
  offset_L0
  feature_L0
Layer P1:
  scaling_residual_L1
  offset_residual_L1
  feature_residual_L1
Layer P2:
  ...
Layer P3:
  ...
Checksum / bit-exact diagnostics
```

注意：`Static tables/version id` 只能指向固定解码器资产，不能携带每场景模型。

### 11.3 熵编码二值分解

沿用当前稳健的二值分解：

```text
A: residual != 0
B: sign
C: magnitude >= k+1
```

但概率由新上下文决定：

```text
p_A, p_C = ZCausal(ctx)
p_B = 0.5 或 context sign table
```

第一版 `ZCausal(ctx)` 是表；第二版是固定小 MLP。

### 11.4 解码流程

解码端按顺序：

1. 解 geometry，得到 anchor 坐标和 Morton 顺序；
2. 解 P0 scaling；
3. 用 P0 scaling 构造 scaling bucket；
4. 解 P0 offset/mask；
5. 构造 offset activity bucket；
6. 解 P0 feature；
7. 渲染 P0；
8. 对每个增强层，使用已解码 coarse state 构造上下文，解 residual；
9. 重构 P3；
10. 验证 final prefix 与生产解码 bit-exact。

## 12. 为什么这是新的

相对 PCGS/ProGS：

- 它们做渐进，但需要模型/网络/结构侧信息；
- UAV-DCCA-ZC 保持零 per-scene 熵模型侧信息。

相对 COSA-GS：

- COSA 是单码率；
- UAV-DCCA-ZC 是可截断前缀渐进。

相对 OIC-GS：

- OIC-GS 是全景 2D 图像 overfitting；
- UAV-DCCA-ZC 是 3D UAV anchor Gaussian scene。

相对 CRP-GS：

- CRP-GS 传根叶链接；
- UAV-DCCA-ZC 用确定性 cell root，不传任意链接。

相对 SCAR-GS：

- SCAR-GS 可用 RVQ/codebook；
- UAV-DCCA-ZC 优先固定通用 transform/codebook，不传每场景码本。

因此新意可以写成：

> 首个面向无人机锚点高斯的零侧信息因果渐进属性熵模型：用已解码粗层、确定性空间 root、局部密度和字段因果关系预测增强层残差概率，在不传每场景概率模型的前提下降低属性基层和增强层码率。

## 13. 最小落地计划

### Step 1：离线熵模型 probe

只改 `scripts/c25_real_bitstream.py`，不改训练。

增加上下文：

- `density_bucket`
- `scaling_bucket`
- `offset_activity_bucket`
- `cell_root_bucket`
- `previous_symbol_bucket`

实验：

```text
baseline table
vs
baseline + density
vs
baseline + scaling/offset
vs
baseline + deterministic root
vs
all combined
```

通过门：

- feature bytes -8%；
- total bytes -5%；
- 解码画质完全不变。

本轮代码已经实现 Step 1 的第一版：

- 文件：`scripts/c25_real_bitstream.py`
- 文件：`scripts/zc_context.py`
- 测试：`tests/test_zc_context.py`
- 新参数：`--zc-context {off,density,root,density-root}`
- 新参数：`--zc-cell-size <float>`
- 默认 `off`，旧行为不变。

实现内容：

1. 从 decoded anchor 坐标构建固定 3D cell；
2. 计算每个 anchor 所在 cell 的局部密度 bucket；
3. 每个 cell 选 Morton/decoded 顺序最早的 anchor 作为 deterministic root；
4. residual layer 的上下文从原来的 `|coarse| bucket` 扩展为：

```text
|coarse| bucket
|coarse| bucket × density bucket
|coarse| bucket × root coarse bucket
|coarse| bucket × density bucket × root coarse bucket
```

5. 所有新增上下文都由解码端可从 decoded geometry 和已解码 coarse symbols 重算，不引入每场景模型或链接表；
6. `--zc-cell-size <= 0` 时自动采用 `bbox_diag/512`，避免按场景手调；
7. 输出 json 新增 `bytes_by_field`、`bytes_by_field_level`、`params_by_field_level`，便于直接看 feature/offset/scaling 哪个字段收益最大。

建议运行：

```bash
bash scripts-new/01_eval_layered_bitstream.sh \
  <run_dir> <data_dir> \
  analysis/s2_prefix_sweep/<tag>_zc.json \
  analysis/s2_prefix_sweep/<tag>_zc.bin
```

01 默认就是 `--zc-context density-root --zc-cell-size 0`（自动推导 `bbox_diag/512`）；要固定 cell size 就在末尾追加 `--zc-context density-root --zc-cell-size 0.02`。

对照旧模型：

```bash
DCCA_ZC_CONTEXT=off bash scripts-new/01_eval_layered_bitstream.sh \
  <run_dir> <data_dir> \
  analysis/s2_prefix_sweep/<tag>_base.json \
  analysis/s2_prefix_sweep/<tag>_base.bin
```

或者用 02 一条命令生成两套码流并出对比报告：

```bash
bash scripts-new/02_compare_zc_bitstream.sh \
  <run_dir> <data_dir> analysis/s2_prefix_sweep <tag>
```

比较两个 json 里的：

- `conditional_saved_bytes`
- `results[*].real_bytes`
- `results[*].psnr_mean`
- `bytes_by_field`
- `bytes_by_field_level`

预期：PSNR 完全不变，若 `real_bytes` 下降则说明新上下文有效。

本地可运行验证：

- `py_compile` 通过；
- `zc_context` 纯函数断言测试通过；
- 当前机器缺 `pytest`，因此用直接调用方式执行了同一组测试。

### Step 2：rate-aware SPA

启用并修正：

- `spa_rate_aware`
- `spa_bit_budget`
- `bits_ema`

与 `spa_ratio=0.85/0.70/0.60/0.50` 交叉。

通过门：

- 同 P0 MB，PSNR +0.2 dB；
- 或同 PSNR，P0 MB -8%。

### Step 3：P0 rendering loss

在 `scaffold_gs/hacpp.py` 或 trainer 中加入 P0 dequantized render path。

通过门：

- P0 +0.4 dB；
- P3 损失 <0.2 dB。

### Step 4：固定通用小 MLP

用多个场景离线训练 ZCausal MLP，固定权重。

通过门：

- 比表模型再省 3%-5%；
- 两环境 entropy-symbol 一致。

### Step 5：UAV tile 封装

作为系统展示：

- 近处 tile 优先；
- 每 tile P0 优先；
- 后续补 P1-P3。

## 14. 论文故事草案

题目候选：

> UAV-DCCA: Zero-Side-Information Causal Progressive Compression for Drone-Scale Anchor Gaussian Splatting

贡献草案：

1. **Zero-side causal progressive entropy model**：用已解码粗层和确定性空间上下文预测属性残差概率，不传 per-scene 熵模型。
2. **Rate-aware anchor formation for UAV Gaussian scenes**：以 importance-per-bit 选择锚点，使低码率前缀不被高成本锚点拖累。
3. **Base-layer rendering optimization**：训练时直接优化可截断基层画面，改善传输起点质量。
4. **Drone-scale tiled progressive bitstream**：空间 tile 和质量前缀联合组织，适合无人机场景渐进传输。

第一篇论文建议只主张前 3 条，第 4 条作为系统扩展或补充材料。

## 15. 2026-10-04 本轮代码优化落地状态

本轮继续把前面结论里的三条关键建议从“方案”推进到“可运行代码路径”。

### 15.1 ZCausal 从 probe 升级为正式分层码流默认路径

- `scripts-new/01_eval_layered_bitstream.sh` 现在默认加入 `--zc-context density-root --zc-cell-size 0`。
- `--zc-cell-size 0` 表示从已解码 anchor 几何自动推导 `bbox_diag/512`，不需要传每场景调参信息。
- `scripts/c25_real_bitstream.py` 的 header 增加正式格式名：
  - `c25_layered_v4_zc`
  - `c25_layered_v4_zc_s5`
- 旧路径仍可复现：
  - `DCCA_ZC_CONTEXT=off bash scripts-new/01_eval_layered_bitstream.sh ...`
  - 或手动传 `--zc-context off`
- 新增对比工具（最终位于 `scripts-new/04_compare_zc_reports.py`，由 `scripts-new/02_compare_zc_bitstream.sh` 调用），用于比较 legacy JSON 和 ZC JSON 的总字节、字段字节、层级字节和 PSNR 是否完全一致。

注意：这一步已经把 ZCausal 作为 wrapper 默认码流路径，但概率模型仍是表模型，不是固定通用小 MLP。后续若要冲 CVPR，需要真实证明 `density-root` 在生产 composite context 上稳定省字节。

### 15.2 rate-aware SPA 接入默认实验矩阵

- `scripts-new/00_train_dcca.sh` 新增环境变量：
  - `DCCA_RATE_AWARE=1`
  - `DCCA_RATE_TAU=1.0`
- 打开后自动加入：
  - `--cfg.model.spa-rate-aware`
  - `--cfg.model.spa-bit-budget`
  - `--cfg.model.sensitivity-target-mode sens_per_bit`
- `scripts-new/03_run_ablation_matrix.sh` 默认新增：
  - `rate_r060`：单独 rate-aware SPA 分支；
  - `zccombo_r060`：`rate-aware + B3 + P0 render loss` 组合分支。
- 可用 `DCCA_SKIP_RATE_AWARE=1` 或 `DCCA_SKIP_COMBO=1` 跳过对应分支。

### 15.3 P0 dequantized rendering loss 已进入训练路径

- `scaffold_gs/config.py` 新增：
  - `p0_render_loss`
  - `p0_render_weight`
  - `p0_render_start_iter`
  - `p0_render_interval`
  - `p0_feat_step / p0_scaling_step / p0_offset_step`
- `scaffold_gs/hacpp.py` 新增 `quant_mode="p0"`：
  - 用训练期同一套 HAC++ 上下文 Q；
  - 先得到最终量化符号 `round(x / Q)`；
  - 再按基层 ladder 截断到 `feat/offset=8x, scaling=2x`；
  - 前向匹配 P0 反量化属性，反向用 STE。
- `scaffold_gs/trainer.py` 新增 P0 二次渲染 loss：
  - 默认后期触发；
  - 每 `p0_render_interval` 步额外渲染一次；
  - loss 使用同一训练视角的 P0 图像与 GT 的 L1/DSSIM 组合。
- P0 分支已排除 sensitivity retain-grad 路径，减少二次渲染的额外显存。

### 15.4 当前仍需实验确认的门槛

这次代码已把三条建议接入可运行路径，但还不能直接宣称 CVPR 级有效。

运行状态（截至 2026-10-04 22:12 打包）：除 `zc_context` 单测实际执行过之外，以上路径均未运行——没有训练日志、码流产物或对比报告；P0 渲染损失的三个文件改完后连 import 都未发生过。下面 1-4 就是当前的全部待办：

1. legacy vs `c25_layered_v4_zc`：确认 PSNR 完全一致、总码率下降，重点看 `bytes_by_field`。
2. baseline r0.60 vs `rate_r060`：确认同码率 P0/P3 是否改善，或同质量是否省字节。
3. baseline r0.60 vs `zccombo_r060`：确认 P0 rendering loss 是否真正提升 P0，且 P3 损失不超过 0.2 dB。
4. 至少 3 个无人机场景 + Mip-NeRF360 / Tanks&Temples 子集重复以上结论。
