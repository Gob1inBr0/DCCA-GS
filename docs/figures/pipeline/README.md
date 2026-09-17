# DCCA-GS 全流程配图（13 张）

> 目标：**只看这些图，就能复述出 DCCA-GS 的工作内容，且与代码流程完全一致。**
> 每张图底部标注了对应的源码文件与行号；所有公式、默认值、迭代区间均逐条对照代码核实
> （2026-09-08，分支 `codex/importance-experiment`）。

## 阅读顺序

| 图 | 文件 | 内容 | 对应代码 |
|---|---|---|---|
| fig00 | `fig00-overview.svg` | **总览**：数据→训练→编码→码流→解码→评估 六阶段一张图；两个创新的落点；关键结果数字 | train.py · trainer.py · hacpp.py |
| fig01 | `fig01-training-loop.svg` | **训练主循环**：单步十步流水 + 0→30k 迭代时间轴（各机制生效窗口：噪声量化仿真 3k–10k、生长+SPA 1500–15k、深度再初始化 @15k、敏感度监督与乘子 ramp 20k–30k）；总损失五项分解 | trainer.py:157–366 |
| fig02 | `fig02-anchor-to-gaussians.svg` | **锚点→神经高斯→渲染**：属性面板、[feat‖视角]→三个共享 MLP、masks 软/硬两种作用、xyz=锚点+偏移·尺度、gsplat | hacpp.py:522–822 |
| fig03 | `fig03-entropy-coding.svg` | **HAC++ 熵编码**：哈希上下文→mlp_grid→熵参数+AQM；逐字段编码方式（G-PCC/算术/混合熵/通道组自回归）；total_MB 构成 | hacpp.py:1344–1711 |
| fig04 | `fig04-complexity-multiplier.svg` | **创新① 复杂度乘子**：4 个结构统计量（含公式）→ mlp_complexity(4→16→3) → m=1+tanh·strength（ramp 0→0.35）→ Q=Q0(1+tanh q)·m；三端一致契约；消融 −0.41dB | gaussian_model.py:674–808 · codec_consistency.py |
| fig05 | `fig05-sensitivity.svg` | **创新① 敏感度监督**：retain_grad→‖∂L/∂a‖→EMA(0.99)→相对归一化 z→target=clamp(1+tanh(−z),0.1,2)→MSE(1e-3)；两个下游用途；全程不进码流 | hacpp.py:713–721,1011–1183 |
| fig06 | `fig06-spa-admm.svg` | **基础机制 SPA-ADMM**：a=mean(mask)、预算 κ ramp（锚定历史最大 N）、z=TopK(a+u,κ)、对偶更新、prune 状态切片清单 | gaussian_model.py:1579–1806 |
| fig07 | `fig07-fusion-score.svg` | **创新② 融合分数（软）**：覆盖 c（8 视角贡献面积）+敏感度 s（硬帽 eff≤0.3）+0.25·(a+u)；w=0.5 无帽失败模式实测（27.51→25.95dB，锚点数不变）；接线条件与 engaged 日志 | gaussian_model.py:1670–1721 · mini_splat.py:282–305 |
| fig08 | `fig08-coverage-constraint.svg` | **创新② 覆盖约束（硬）**：floor(xyz/0.01) 粗胞、每被占用胞保 top-1（mandated+fill 算法）、2D 网格示意、与软帽互补 | gaussian_model.py:1725–1757 |
| fig09 | `fig09-depth-reinit.svg` | **深度再初始化**：@15000 锁预算→8 视角深度反投影→体素化 cap4000→append→竞争淘汰；full 模式（默认关） | hacpp.py:878–995 · mini_splat.py |
| fig10 | `fig10-encode-decode.svg` | **编码↔解码**：共享 Q 路径两端各跑一遍；码流文件清单与"不进码流"清单；bit-exact 诊断（SHA-256，mismatch=0） | hacpp.py:1344–1993 |
| fig11 | `fig11-evaluation.svg` | **评估**：统一协议（3 场景/30k/seed42/fp32/bit-exact）；RD 扫描流程；BD-rate（旧 −11.1% 作废注记）；已有结果 | scripts/rd_sweep.py |
| fig12 | `fig12-audit-negative.svg` | **过程资产**：融合分数静默门控审计（根因链/修复/教训，commit d3b4f49）；负结果与改道清单（7 项） | docs/03-reports · docs/02-design |

## 全局配色（所有图一致）

- **蓝** = HAC++ 原有机制
- **绿** = 创新① 复杂度乘子（量化路径，解码端重算）
- **橙** = 创新② 覆盖感知融合剪枝（训练侧投影）
- **紫** = 码流内容
- **灰虚线** = 训练期专用信号（永不进码流）
- **红** = 失败模式 / 警示

## 已知与论文正文的两处差异（以代码为准，图中已标注）

1. mlp_complexity 实为 **4→16→3**（论文写 8→32→3 为旧版；PHG v2 删除 4 个恒零图像统计）；
2. 覆盖约束粗胞尺寸实为 **0.01**（论文写 δ=0.05 为旧值），min_per_cell=1。

## 再生成与质检

```bash
cd docs/figures/pipeline
python3 make_all.py      # 生成 13 张 SVG
python3 qa_check.py      # 几何质检：文字溢出/互压/画布越界/chip 互压/箭头悬空（应输出"无几何问题"）

# SVG → PNG（Edge 无头浏览器，按画布精确尺寸渲染；qlmanage 会裁切成正方形，勿用）
EDGE="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
for f in fig*.svg; do
  w=$(python3 -c "import re;print(re.search(r\"width='(\d+)'\",open('$f').read(200)).group(1))")
  h=$(python3 -c "import re;print(re.search(r\"height='(\d+)'\",open('$f').read(200)).group(1))")
  "$EDGE" --headless --disable-gpu --screenshot="$PWD/png/${f%.svg}.png" \
    --window-size="$w,$h" --default-background-color=FFFFFFFF "file://$PWD/$f"
done
```

- 生成脚本：`svgkit.py`（SVG 工具库）、`gen_a/b/c/d.py`（各图布局）、`make_all.py`、`qa_check.py`
- 成品：`*.svg`（矢量，可改字重排）与 `png/*.png`（每图 1:1 渲染）
