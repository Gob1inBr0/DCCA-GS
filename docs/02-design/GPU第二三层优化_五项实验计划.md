# GPU 第二、三层优化：五项实验计划（预登记）

> 建立：2026-10-03。分支 perf/gpu-step-tax。本文件在动手前写死各实验的判定标准；
> 结果无论采纳或证负都按 fam/arscan 同款流程记录进 docs/CHANGELOG_DETAILED.md 与
> docs/data/experiments.csv（如产生 RD 数字）。

## 0. 现状基线（全部已实测）

- 每步墙钟 404.2 ms（505.5 基线 −20.0%，三次复测极差 0.9%），数值与历史逐位一致。
- 设备时间 326.7 ms/步：逐元素算子 95.6（8386 次发射）＞ tile 求交+排序 90.8 ＞
  哈希网格反传 44.6（单内核）＞ GEMM 33.9 ＞ 归约 23.6 ＞ 光栅化主内核 10.9。
- 墙钟与设备时间差 77.5 ms：CPU 发射/同步未重叠部分；同步驻留 98.8 ms/121 次。
- 已证负关闭：gsplat 1.6.0 main 快照（+35.3% 墙钟，AccuTile 代价，§28）。
- 测试负载：fam_base_l0004_s42 @30000（686,572 锚点）、room 778×518、A100-40G 借卡。

## 1. 共同协议（每个实验同一套）

- **A/B 口径**：同卡（借卡协议：RELEASE-n → 等显存清零 → 跑完自动回占）、
  同 checkpoint、同场景、同 seed 42。计时 = profile_step_tax.py，60 计时步 × 3 次
  取均值；画质 = 2200 步从零训练冒烟，读 [Eval @2200] PSNR 与终点锚点数。
- **采纳门（动手前写死）**：
  1. 三次均值墙钟改善 ≥ 3%（噪声带 0.9% 的 3 倍余）；
  2. 2200 步 PSNR 与同代基线差在 ±0.10 dB 内（fam 种子带 ±0.095 的圆整）；
  3. 无 NaN、终点锚点数与基线差 ≤ 2%；
  4. 涉及哈希网格数值的实验（B）加测：encode_attributes 总码率不高于基线 +1%。
- **叠加规则**：某实验采纳后，后续实验的基线 = main + 已采纳项。
- **证负即关**：记录数字与原因，登记重评条件，不反复重试。

## 2. 实验 A：torch.compile 包解码＋码率估计路径

- **打击面**：逐元素算子 95.6 ms（29.3%）＋ 部分 GEMM。
- **做法**：把 generate_gaussians 的分块解码循环体与 _estimate_rate_terms 的
  CDF 链抽成可编译函数，`torch.compile(mode="max-autotune-no-cudagraphs")`。
  不包 gsplat 光栅化与损失项。
- **动态 shape 对策（首要风险）**：可见锚点数逐帧变化 → 反复重编译。首选把解码
  块补齐到固定 16384 行（pad 行的 opacity 输出显式置 −1 使 sel=False 天然剔除，
  仅末段有 pad，额外计算 ≤ 一段）；备选 dynamic=True。
- **已知障碍**：STE_binary/STE_multistep 是 autograd.Function → dynamo graph
  break，收益打折。备选方案 B（仅在 break 损失过大时启用）：把 STE_binary 重写为
  纯原语 `out = x_q.detach() + m*x − (m*x).detach()`（m = (x.abs()<=1)）——前向
  逐位等于 x_q、反向梯度逐位等于 m，与现实现等价，且可被 inductor 融合。重写前
  先过等价性测试（并入 test_step_tax_bitexact.py）。
- **度量补充**：记录 torch._dynamo.explain 的 graph break 数；warmup 步数加倍
  吸收首编时间。
- **预登记判定**：共同采纳门。预期 −8-15%（95.6 ms 打五折 ≈ −48 ms）。
- **成本**：1 天。

## 3. 实验 B：哈希网格前向 half 精度

- **打击面**：kernel_grid_backward 44.6 ms（访存受限）＋ kernel_grid 前向 5.9 ms。
- **做法**：最小侵入版——只改 hacplus/utils/encodings.py 的 `_grid_encode`：
  查表与 dy_dx 中间量用 half，输出 cast 回 fp32 交下游（MLP 保持 fp32，不牵连）；
  网格 master 参数保持 fp32，前向读时 .half()（约 2 MB/次拷贝，可忽略）。
  编码器已有 autocast 分支（encodings.py:118）可参考实现。
- **数值风险**：STE 二值化阈值 |x|≤1 在 half 边界的行为；梯度幅度进入 half 范围
  的截断。先补单测（half 输入下 STE 前向/反向 vs fp32 参照的容差断言，非逐位），
  再上 A/B。
- **预登记判定**：共同采纳门＋码率门（encode_attributes 总码率 ≤ 基线 +1%）。
  预期 −5-6%（44.6 打对折）。
- **成本**：1-2 天。

## 4. 实验 C：absgrad=True 替换手工梯度聚合

- **打击面**：归约桶 23.6 ms 的一部分＋ retain_grad 的 autograd 路径；收益上限
  比 3-4% 可能更低（packed 行布局仍需按 gaussian_ids 聚合），如实按实测记。
- **做法**：renderer.py 传 absgrad=True；growth.py 的 packed 分支改读
  meta["means2d"].absgrad，删除手工 clone/rescale 链（0.5·W/0.5·H 缩放按
  gsplat 文档确认 absgrad 是否已含；不含则保留缩放）。
- **数值影响**：原子累加顺序不同 → 稠密化统计微变 → 配对判定类。
- **预登记判定**：共同采纳门。**成本**：半天-1 天。

## 5. 实验 D：tcnn.Encoding 替换自带 _gridencoder（第三层）

- **打击面**：44.6＋5.9 ms 网格前后向＋相邻 GEMM 一部分；最大单项候选。
- **关键约束（预登记即声明）**：tcnn 的哈希布局/函数与自带实现不同，已有
  checkpoint 的网格参数**不可迁移**——这是"换组件重训"而非"加速现模型"，
  与全部历史 run 的可比性断开，需另立新基线（同种子新训 30k）。
  输入维度限制：fully fused MLP 输入 ≤16，故只替换编码模块（3 维入、56 维出），
  MLP 保持 PyTorch。
- **依赖**：tcnn 编译（cutlass）；A/B/C 结果影响其优先级——若已累计 ≥20%，
  本项降级为"论文效率章节素材"暂缓。
- **预登记判定**：共同采纳门＋新训 30k 完整 RD 对比（不许只用 2200 步冒烟）。
  **成本**：3-5 天。**默认排后，视 A/B/C 结果启动。**

## 6. 实验 E：CUDA graph 稳态段捕获（第三层）

- **打击面**：77.5 ms CPU 间隙＋发射开销。
- **适用面窄的如实声明**：图捕获要求全程静态 shape——解码块可 pad 固定，但
  光栅化输入的高斯数随可见性逐帧变化，pad 到 worst-case 会浪费排序与渲染带宽。
  因此仅对"解码＋码率估计段"捕获，或直接用实验 A 的 reduce-overhead 模式
  （inductor 自带 cudagraph），二选一，避免重复建设。
- **依赖**：实验 A 先行。**预登记判定**：共同采纳门；预期不定（0-10%）。
  **成本**：2-3 天。**排最后。**

## 7. 执行顺序

```
A（compile，1 天）→ B（网格 half，1-2 天）→ C（absgrad，1 天）
   ↓ A/B/C 收齐
累计 ≥20% ? ── 是 → D 暂缓，出总结报告
   └─ 否 → D（tcnn，3-5 天）→ E（cuda graph，2-3 天）
```

每个实验完成后：数字入 analysis/step_tax_profile/、changelog 追加、
采纳项合入 perf/gpu-step-tax 并 ff main；证负项登记重评条件。

## 8. 风险登记

| 风险 | 缓解 |
| --- | --- |
| dynamo 在 hacpp 大函数上分析失败/超时 | 只编译抽取出的循环体函数，不编整个类方法 |
| graph break 吃掉 A 的收益 | _dynamo.explain 定位；备选 STE 原语化重写 |
| half 下 STE 梯度截断导致训练发散 | 先单测再 A/B；采纳门含码率不升 |
| 借卡被主会话任务竞争 | 借卡协议＋错峰；fam/reinit 批次优先 |
| absgrad 收益低于预期 | 预期已写保守（1-4%），证负直接关 |
| D 的可比性断开 | 预登记即声明"新训线"，不与历史 run 直接混比 |

## 9. 不做的事（边界）

- 不动方法语义（可见子集解码、全程半精度训练属实验设计变更，不在本计划）。
- 不再重试 gsplat 1.6（重评条件见 §28）。
- 不做 SSD/多卡分布式（当前单卡训练无此需求）。
