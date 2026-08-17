# 创新点 R（CompGS 式残差编码）阶段 A 报告：离线残差熵验证

日期：2026-08-17；状态：**已执行，判定 close，不进入阶段 B/C**

## 1. 实验对象

- Checkpoint：`runs/4-28_i6_90k_h32_l0p002/ckpts/ckpt_90000.pth`（4-28，h32，dim50，λ=0.002，90k）
- 锚点数：338,130（mask + Morton codec 顺序）
- 验证集：Morton 序最后 20%（约 67,626 个锚点）；前 80% 用于拟合预测器
- 符号、Q（含 I2 formula 乘子）、feat 通道自回归口径与真实 codec 完全一致
- 参考 total_MB：8.3069（该 run 的 bitstream 口径）

## 2. H_base（当前 codec 熵，验证集）

| 字段 | MB |
| --- | --- |
| feat | 0.9589 |
| scaling | 0.3333 |
| offsets | 0.2406 |
| 合计 | 1.5328 |

## 3. 变体结果（hidden=64，best per variant）

| 变体 | scaling MB | offsets MB | total MB | scaling+offsets 增益 | total 增益 | 净省 MB（扣 16-bit 预测器） |
| --- | --- | --- | --- | --- | --- | --- |
| R0 拉普拉斯（mlp_grid 均值做预测，零参数） | 0.3727 | 0.2005 | 1.5321 | +0.11% | +0.04% | +0.0006 |
| R0 高斯（同上） | 0.5648 | 0.2138 | 1.7375 | −35.7% | −13.4% | −0.205 |
| R1 ctx→MLP + 拉普拉斯 | 0.3963 | 0.2330 | 1.5882 | −9.7% | −3.6% | −0.084 |
| R2 跨字段（feat_q→scaling，feat_q+scaling_q→offsets）+ 拉普拉斯 | 0.3952 | 0.2330 | 1.5870 | −9.5% | −3.5% | −0.100 |
| R2 高斯 | 0.5154 | 0.2433 | 1.7176 | −32.2% | −12.1% | −0.231 |
| R3 feat 通道 delta | 0.3333 | 0.2406 | 2.2322 | 0.0% | −45.6% | −0.699 |
| R4 控制组（P0-1 式条件均值+log-scale，不编码残差） | 0.3225 | 0.1829 | 1.4643 | **+11.9%** | +4.5% | +0.009 |

（R1/R2 均为 s400/s1500 两档，取较优；R4 为 s1500 档。）

## 4. 判定

满足设计文档停止条件：

1. R2 的 scaling+offsets 增益 −9.5%（< 3%），净收益为负；
2. R2 高斯比 R4 差 16.5 个百分点（残差结构相对“条件均值+尺度”无优势）；
3. R0 无增益（+0.11%）且 R1/R2 无增益；
4. 预测器权重吃掉全部收益（净省为负）。

**结论：阶段 A 关闭 R 方向，不做最小编解码（阶段 B）与联合训练（阶段 C）。**

## 5. 解读

- `mlp_grid` 的高斯均值已经是强预测：直接对它做“零均值拉普拉斯残差”没有任何增益；
- 离线小 MLP（ctx / feat_q 输入）学到的预测器无法替代训练好的 `mlp_grid` 均值，残差编码反而更差；
- R4（P0-1 式：已解码 feat 调整 mean/scale）是唯一有效的方向，这正是 P0-1 已验证过的条件熵路线，与“残差结构”无关；
- R3 通道 delta 显著变差，说明通道间线性冗余已被 `Channel_CTX_fea` 通道自回归覆盖；
- 与 P0 系列结论一致：PHG 的属性熵模型已接近条件熵上限，CompGS 式“预测+残差”在本架构上没有可挖的净收益。

## 6. 产物与复现

- 脚本：`scripts/residual_feasibility.py`（R0–R4 全部变体 + 停止条件自动判定）
- 5090 结果：`runs/residual_feasibility_4_28_h32_l0p002.json`
- 复现：

```bash
cd /home/fansonglin/xieliang/chentong/PHG
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD PYTHONNOUSERSITE=1 \
PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH \
python scripts/residual_feasibility.py \
  --ckpt runs/4-28_i6_90k_h32_l0p002/ckpts/ckpt_90000.pth \
  --out runs/residual_feasibility_4_28_h32_l0p002.json
```
