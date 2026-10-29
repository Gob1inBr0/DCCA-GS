# 空间上下文相关门 阶段 A 报告（四航道 · 敏感度可预测性）

> 日期：2026-09-03
> 运行平台：LYH 机器（root@36.138.63.143:65022，8×A100-40GB）
> 设计出处：`../02-design/语义感知剪枝与空间上下文_实验设计.md` 方向一
> 结论：**加"四航道空间特征"对预测渲染敏感度没有增量价值（Δr≈0）；且敏感度目标本身
> 只有 0.28–0.29 的可预测性（低于 0.3 门限）。** 方向一不成立。

---

## 0. 结果

两个场景，随机 80/20 切分：

| 场景 | 集合 B 均值 r | 集合 D（B+空间特征）均值 r | Δ(D−B) feat r |
| --- | ---: | ---: | ---: |
| drjohnson | 0.281 | 0.280 | **−0.0010** |
| tandt_train | 0.287 | 0.287 | **+0.0015** |

**Δr≈0，几乎没变。** 这落在设计判定表里的"Δr≈0 → 不加"分支。同时：
**渲染敏感度目标本身，只能被很富的解码端上下文预测到 0.28–0.29**（低于 0.3 门限，
两个场景 gate 均为 FAIL）。

---

## 1. 输入集与目标

| 集合 | 内容 | 为什么测 |
| --- | --- | --- |
| A | 4 维公式特征（mlp_complexity 实际输入） | 基线，预期≈0 |
| B | A + 哈希上下文 + mlp_grid 输出（mean/scale/prob + qa/qs/qo） | 现状"富上下文" |
| D | B + `{r_norm, h_norm}` | 四航道空间特征是否有增量 |

目标：逐锚点渲染敏感度乘子 `sens = 1 + tanh(−z)`，`z = (EMA − mean)/mean`（I6 口径）。

---

## 2. 实验设置

| 项 | 值 |
| --- | --- |
| 场景 | drjohnson（T&T）、tandt_train（T&T） |
| checkpoint | `lyh_full_drjohnson/ckpts/ckpt_30000.pth`、`lyh_depth_tandt_train/ckpts/ckpt_30000.pth`（hac_pp、feat_dim=50、n_offsets=10、content_aware_quant=True、无 I1、mlp_complexity hidden=32/1层） |
| 数据 | `/mnt/003/dataset/tandt_db/db/drjohnson`、`/mnt/003/dataset/tandt_db/tandt/train`（1600 宽，test_every=8） |
| 敏感度采集 | `--steps 120`（训练视角循环渲染 120 次，L2 loss 反传，accumulate_sensitivity） |
| 切分 | `--shuffle-split`：随机 80/20 锚点切分（避免按 Morton 连续切分的空间自相关高估 r） |
| 拟合 | 每个输入集合一个 2 层 MLP（hidden=64），`--fit-steps 400`，在验证集算 Pearson r |
| 脚本 | `scripts/sens_spatial_gate.py`（基于 `sens_replace_gate.py` + 集合 D + `--shuffle-split`） |
| 结果 | `docs/data/gate_results/gate_{drjohnson,tandt_train}_shuffle.json` |

说明：原设计想在 4-28（无人机）上跑，但 LYH 上 4-28 只有原生 HAC++ 状态
（`point_cloud/.../checkpoint.pth`，无 `model_config`），不是 DCCA 训练器格式，`load_checkpoint`
无法加载。因此本报告在不冲突的 DCCA 格式、且有对应数据的场景（drjohnson / tandt_train）
上完成方向一的判定；四航道特例的无人机验证见 §6 局限。

---

## 3. 结果

### 3.1 drjohnson（85,846 锚点）

| 集合 | feat r | scaling r | offsets r | 均值 r | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.0277 | 0.0163 | 0.0052 | 0.016 | 0.1910 |
| B | 0.3073 | 0.3643 | 0.1715 | 0.281 | 0.1742 |
| C_k32 | 0.3104 | 0.3717 | 0.1762 | 0.286 | 0.1737 |
| **D** | **0.3063** | **0.3631** | **0.1706** | **0.280** | 0.1743 |

`best_mean_corr_non_A = 0.2861`，`Δ(D−B) feat r = −0.0010` → **FAIL（<0.3）**。

### 3.2 tandt_train（132,832 锚点）

| 集合 | feat r | scaling r | offsets r | 均值 r | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.0898 | 0.1280 | 0.0550 | 0.091 | 0.2891 |
| B | 0.2573 | 0.4463 | 0.1566 | 0.287 | 0.2602 |
| C_k32 | 0.2645 | 0.4506 | 0.1676 | 0.294 | 0.2589 |
| **D** | **0.2588** | **0.4460** | **0.1575** | **0.287** | 0.2601 |

`best_mean_corr_non_A = 0.2942`，`Δ(D−B) feat r = +0.0015` → **FAIL（<0.3）**。

**说明：** C_k（前 k 个 Morton 邻域已解码属性均值）只比 B 高 0.001–0.008，同样几乎无帮助；这里只列 k=32。

---

## 4. 解读

1. **集合 A → r≈0：4 维公式特征（mlp_complexity 真正的输入）对敏感度的预测力接近 0**
   （drjohnson 均值 0.016，tandt_train 0.091）——被监督网络的输入空间里，目标基本不可达。
2. **集合 B → 中等但低于 0.3**：哪怕吃到"哈希上下文 + mlp_grid 输出"这个很富且零侧信息的
   输入，对敏感度的可预测性也只到 0.28–0.29（均值），**没过 0.3 门槛**。对照：DINO 用同样的
   富上下文能到 r=0.60——**DINO 语义可预测（0.60）但目标不对；渲染敏感度目标对但不可预测
   （0.29）**。量化 Q 路径上，不存在一个"既=重要性、又=可被解码端重算"的目标。
3. **集合 D → Δr≈0：四航道空间特征无增量价值**。在集合 B 之上额外加 `r_norm`（远近）+
   `h_norm`（高低），Δr 只有 −0.001（drjohnson）/+0.0015（tandt_train），在数值噪声量级。
   "远近"已被局部密度 + 哈希上下文吸收，"高低"也没多解释任何方差。

---

## 5. 结论

1. **方向一（量化路径：空间上下文 × 敏感度目标）不成立。**
   两个场景一致：空间特征 Δr≈0，敏感度目标均值 0.28–0.29 < 0.3。既没有"输入够得着"，
   也没有"目标对应得上"。
2. 量化 Q 这条路，DINO（可预测但目标错）和敏感度（目标对但不可预测）都不成立；
   扩上下文（加空间特征/邻域）也补不上敏感度的不可预测性。

---

## 6. 局限

- **场景非无人机**：drjohnson / tandt_train 是室内/边界场景，不存在"四条不同距离航道"的结构。
  因此"四航道特征"本身在无人机上的价值**未被直接证伪**；被证伪的是"坐标派生空间特征
  （r_norm/h_norm）作为额外输入"这一**通用做法**在一般场景上无增量。
- **4-28（无人机）无法在本机测**：LYH 上 4-28 只有原生 HAC++ 状态（`point_cloud/.../checkpoint.pth`，
  无 `model_config`，`load_checkpoint` 抛 `KeyError`）；DCCA 格式 checkpoint 缺失。
- **敏感度 EMA 采 120 步**：`alpha=0.99` 的 EMA 需要更多步才稳定；120 步的 r 略偏保守。
  不影响"Δr≈0"的主结论（空间特征有无与采几步 EMA 无关）。
- **门只证明可预测性**：即使某集合 r 高，也不自动等于 RD 收益；但 r 低 = 一定白做。

---

## 7. 复现

```bash
# LYH 机器，DCCA 环境
cd /home/project2/DCCA-GS-minifull
export PYTHONPATH=/home/project2/DCCA-GS-minifull:/home/project2/gsplat-main

/home/project2/miniconda3/envs/DCCA/bin/python scripts/sens_spatial_gate.py \
  --ckpt /home/project2/runs/lyh_full_drjohnson/ckpts/ckpt_30000.pth \
  --data-dir /mnt/003/dataset/tandt_db/db/drjohnson \
  --steps 120 --k-list 8,16,32 --shuffle-split \
  --out docs/data/gate_results/gate_drjohnson_shuffle.json
```

（tandt_train 同理，`--ckpt lyh_depth_tandt_train/ckpts/ckpt_30000.pth`、
`--data-dir /mnt/003/dataset/tandt_db/tandt/train`。）

产物已在仓库：`docs/data/gate_results/gate_drjohnson_shuffle.json`、
`docs/data/gate_results/gate_tandt_train_shuffle.json`。
脚本：`scripts/sens_spatial_gate.py`（新增集合 D + `--shuffle-split`）。

---

## 8. 参考

- 设计：`../02-design/语义感知剪枝与空间上下文_实验设计.md`
- 之前失败复盘：`../03-reports/语义先验_阶段A报告.md`、`../02-design/语义先验实验设计.md`
- 门脚本：`scripts/sens_replace_gate.py`、`scripts/sens_spatial_gate.py`
- DCCA 代码/hacpp：`scaffold_gs/hacpp.py`、`hacplus/scene/gaussian_model.py`
