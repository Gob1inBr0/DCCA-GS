# DCCA-GS

**Progressive Layered Bitstream for Anchor-Based 3D Gaussian Splatting Compression**

一个基于 [gsplat](https://github.com/nerfstudio-project/gsplat) +
[Scaffold-GS](https://arxiv.org/abs/2312.00109) + [HAC++](https://arxiv.org/abs/2501.12255)
的锚点式三维高斯泼溅压缩框架。主贡献：**单文件渐进分层码流**——一个码流文件
携带多个画质档，任意质量档都是文件的连续字节前缀（收流方传到哪、解到哪），
且量化条件全部解码端可重算（零侧信息契约）。

---

## 🎯 主贡献：渐进分层码流 + 阶梯对齐训练

### 1. 渐进分层码流（表示与编码）

- **嵌套质量阶梯**：量化符号按分字段粗化档位组织（feat/offset 8 倍起步、
  scaling 对数域 2 倍起步），基层传完整空间结构，增强层逐层补精度；
- **二值分解区间编码**（constriction 真实编码器）：零标志 → 符号 → 幅度
  逐位是非题，每级概率有界、无零概率失败模式；增强层用（贡献组 × 粗值
  幅度桶）复合条件表——条件信息全部解码端可重算；
- **层级序文件布局**：所有字段的 L0 层在前、L1 层次之……因此**每个质量
  前缀 P_k 都是文件头部连续段**，解码端顺序读到任意截断点即得一档画质；
- **栅格一致性契约**：解码器导出本次解码实际使用的逐元素量化栅格
  （`last_decode_Q`），分层分析管线消费同一栅格并自检（逐位一致已验证：
  渐进满精度档 = 生产解码，bit-exact）。

### 2. B3：阶梯对齐量化训练（训练侧组件）

渐进基层（8 倍粗化）失真大。B3 在训练末期对"量化符号到粗阶梯倍数的偏移"
加惩罚，把符号推向基层友好的位置（Q 与 mask 梯度切断，权重随剩余步数
线性升满；配置三字段 `coarse_ladder_align/start_iter/weight`，默认关）。

同代码同种子配对判定（150 视角）：

| 档 | P0 增益 | 满精度档 | 判定 |
| --- | --- | --- | --- |
| λ0.0005（w=0.01） | +0.26 dB | +0.08 dB | 采纳 |
| λ0.004（w=0.05） | +0.92 dB（12 dB 悬崖上） | −0.03 dB（噪声级） | 采纳 |

λ0.0005 档四档全面更优（满精度省 4.2 MB 还高 0.08 dB）；机制收益随预算
变紧、悬崖变陡而放大。

### 3. 系统底座（单码率压缩）

学习式上下文熵编码 + HAC++ 谱系码流基座（G-PCC 锚点、哈希二值化、掩码
算术编码）。1-78 场景对 HAC++：同画质省约 30% 体积（可辩护口径：平价点
−24%、最近点对 −30.6% @ −0.146 dB）；HAC++ 曲线延伸到紧 λ（12.1 MB）后，
延伸段我方 base 仍占优 0.5 dB 以上；<12.1 MB 区间 HAC++ 无产品，我方
λ0.004 渐进曲线从 7 MB 起步独占。

> 数字口径：以上均为 150 视角 / 1600px GT / 生产量化栅格（2026-09-28 审计
> 统一后的口径）。单一 BD-rate 数字不使用（双方画质区间不重叠时外插主导，
> 不可辩护），论文用逐点对比。

---

## 🧱 组件分工（采用既有方法或已判定边界的部分）

- **SPA 剪枝**（GaussianSpa 锚点版）：基础措施，方法节引用，不算贡献；
- **次模覆盖选择**：紧预算专用组件（紧档双种子正、松档收益随删减量归零
  ——杠杆被 0.3% 删减量锁死），不包装成全曲线方法；
- **融合剪枝（感知调制）**：防坍缩鲁棒性组件，停止投入（记分板均值
  −0.06 dB），保留机制分析；
- **内容自适应量化（I2/I6）**：保留为组件；消融实测增益 +0.059 dB
  （噪声级），不作创新点主张。

**负结果台账**：条件表细化（粒度到顶）、学习式概率模型（生产 A/B 零增益）、
B3 强权重（w=0.05 松档满精度塌 0.48）等方向全部按预登记门证负关闭，记录在
`docs/data/experiments.csv`（唯一数字事实源）与 `docs/CHANGELOG_DETAILED.md`。

---

## 🧰 工程实现

- 8×A100 排队器与全闭环 runner（train → compress → decode → eval，
  bit-exact 校验 + run 溯源 provenance）；
- 分层码流分析管线（`scripts/c25_real_bitstream.py`、`full_view_eval.py`）：
  真实字节编码 + 栅格自检 + 逐前缀评测；
- 测试套件：`python -m pytest tests/`（合并线全绿）。

---

## 🚀 复现

场景：自采无人机航拍（4 航道协议：远/近 × 高/低），内部编号 1-78 / 2-06 /
4-10（外部数据集名待确认）。30k 全闭环（1-78 为例）：

```bash
cd <repo>
RUNS_ROOT=<runs_dir> RUNROOT=<repo> CONDA_ENV_BIN=<env>/bin \
GSPLAT_ROOT=<gsplat> bash scripts/runner_phg_cell.sh \
  <gpu> 1-78 <data_dir> 0.0005 <tag> 30000 15000 \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
  --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
  --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
  --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
  --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85 \
  [--cfg.model.coarse-ladder-align --cfg.model.coarse-ladder-weight 0.01]
```

分层码流编码与评测：

```bash
PYTHONPATH=<constriction_dir>:<repo> python scripts/c25_real_bitstream.py \
  --run <run_dir> --data-dir <data_dir> --field-aware \
  --data-factor 1 --max-width 1600 \
  --out <out>.json --bin <out>.bin
```

---

## 📁 证据地图

| 内容 | 位置 |
| --- | --- |
| 数字唯一事实源 | `docs/data/experiments.csv` |
| 全时序变更记录 | `docs/CHANGELOG_DETAILED.md` |
| 方法与设计文档 | `docs/02-design/`、`docs/06-planning/` |
| 判定门与负结果 | 各设计文档预登记节 + CSV notes |
