# DCCA-GS

**Decoder-Reproducible Content-Adaptive Compression for Anchor-Based 3D Gaussian Splatting**

一个基于 [gsplat](https://github.com/nerfstudio-project/gsplat) +
[Scaffold-GS](https://arxiv.org/abs/2312.00109) + [HAC++](https://arxiv.org/abs/2501.12255)
的锚点式三维高斯泼溅压缩框架。目标：**在零侧信息、不改变码流契约的前提下，提升率失真（RD）曲线**。

---

## 🎯 核心创新点（唯一主创新）

### 渲染敏感度损失的复杂度乘子

把"该给每个锚点多细的量化步长 Q"这一决策，交给一个**解码端可重算**的复杂度乘子：

```text
Q_field = Q0 × (1 + tanh(q_AQM)) × m_field
m_field = 1 + tanh(mlp_complexity(formula)) × α
```

- **I2（内容复杂度量化）**：`mlp_complexity` 由解码端可重算的公式特征（局部密度、尺度各向异性、
  偏移能量、掩码激活比例）预测量化步长，零侧信息；
- **I6（渲染敏感度监督）**：训练期用渲染损失对不同属性的梯度 EMA 监督同一个 `mlp_complexity`，
  使乘子学会"把更细的 Q 给对画面影响大的锚点"；只改训练目标，不进码流。

`mlp_complexity` 是 8→32→3 的小网络，与 AQM 共享；**编/解码端用同一份权重与公式路径，bit-exact**。

> 一句话：**用"解码端可重算的内容复杂度"×"训练期渲染敏感度监督"，共享一个小 MLP，零侧信息地做内容自适应量化。**

---

## 🧰 工程实现（非主创新，归在一起）

- feat_dim=32 泛化、哈希/条件熵模型/G-PCC/算术编码的高效实现；
- 面向 8×A100 的排队器与全闭环 runner（train → compress → decode → eval，bit-exact 校验）。

> 注：**MLP 权重量化已移除**。它只省约 0.5% 体积，且原实现是"decode 后再量化、只换体积那一行"，
> 口径不自洽，故不作为贡献点。

---

## 🧱 基础措施（采用既有方法组件，非本工作创新）

- **SPA**（GaussianSpa 式训练侧 ADMM 剪枝）：训练期显式锚点预算稀疏化；
- **Mini-Splatting depth-reinit**：生长停止时深度反投影增密，把锚点"铺"到场景表面。

二者作为基础措施使用；若后续对 SPA 做能显著提升其效果的改动，再单独列为创新点。

---

## 📊 规模口径

```text
total_MB = (bits_xyz + bits_feat + bits_scaling + bits_offsets
            + bits_masks + bits_hash + bits_mlp + bits_bounds + bits_header) / 8 / 1024 / 1024
```

`bits_mlp` 按 float32（32 bit/参数）计；与 HAC++ 原生口径一致。

---

## 📁 目录

- `scaffold_gs/`、`hacplus/`：训练/编解码核心；
- `scripts/runner_phg_cell.sh`：单场景全闭环 runner（train→compress→decode→eval）；
- `docs/01-architecture/DCCA-GS_创新点说明.md`：创新点三层定位与原理；
- `docs/03-reports/`：实验报告。
