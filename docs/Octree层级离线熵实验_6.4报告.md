# 方向四 6.4 最小实验报告（Octree-GS 式层级组织离线熵验证）

日期：2026-08-20；5090（A100 32GB）

## 输入与方法

- bitstream：`runs/mlp_quant_sens_cd8_rest16_110k/b8/bitstreams`
  （4-28，110k，FQ+RSS+MLP 量化，254,921 anchors）
- 脚本：`scripts/octree_level_entropy.py`
  （`--legacy-complexity-8dim` 兼容旧 8 维公式输入）
- 分组：按锚点尺度 `exp(scaling[:, :3]).mean` 的分位数等分为 2 / 3 层；
- 熵口径：复用 P0 离线方法——mlp_grid 预测的逐符号高斯熵参数 +
  Gaussian CDF 量化格概率，训练 80% / 验证 20%（Morton 序尾部）；
- “每层独立熵模型”以逐层逐维的残差均值偏移 + 尺度乘子近似，
  在训练段拟合、验证段评估。

## 结果（验证集 scaling + offsets 交叉熵）

| levels | H_base total MB | H_level total MB | scaling 增益 | offsets 增益 | total 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.3841 | 0.4248 | +1.21% | −30.52% | **−10.61%** |
| 3 | 0.3841 | 0.4117 | +2.25% | −23.11% | **−7.19%** |

（H_base 两行相同：同一 bitstream、同一验证集。）

## 结论

停止条件“分层熵增益 ≥3%”**未满足**（2 层 −10.6%、3 层 −7.2%），
按设计文档 §6.4 关闭方向四。

解读：

1. scaling 有微弱的层间可解释性（+1.2%~+2.3%），但不足以跨过 3% 门槛；
2. offsets 在逐层拟合后反而显著变差（−23%~−30%）：全局 mlp_grid 的
   条件高斯已按锚点上下文校准，逐层残差分布呈重尾，训练段拟合的
   层内方差低估尾部分布，验证段比特上升；
3. 结论与 P0（跨锚点/邻居条件熵增益 <1%）一致：PHG 的熵模型输入侧
   已接近饱和，层级身份（anchor scale 分组）作为新的条件信息没有
   可兑现的压缩收益。

备注：本次是“现有全局模型 + 层间校正”的离线验证，未训练真正的每层
独立哈希网格 + 熵模型；但按 P0 同口径门槛，方向四不值得进入真实编码。

## 产物

- 5090：`runs/octree_level_entropy/results.json`（3 层）、
  `runs/octree_level_entropy/results_l2.json`（2 层）
- 脚本：`scripts/octree_level_entropy.py`
