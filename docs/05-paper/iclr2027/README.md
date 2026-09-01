# ICLR 2027 Paper Draft: DCCA-GS

> 状态：初稿 0.2
> 编译：`pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
> 文件：`main.tex` + `references.bib`，官方 ICLR 2027 模板已在同目录。

## 内容

- 标题：DCCA-GS: Decoder-Reproducible Content-Adaptive Compression for Anchor-Based 3D Gaussian Splatting
- 结构：Abstract、Introduction、Related Work、Method、Experiments、Discussion and Open Questions、Conclusion、AI use statement、Ethics statement
- 实验：4-28、Deep Blending、Tanks & Temples、Mip-NeRF 360
- 方法：I2 解码端可重算公式量化、I6 渲染敏感度监督、MLP 混合精度量化、SPA ADMM 锚点稀疏、Mini-Splatting anchor reallocation

## 已知占位

1. Figure 1 目前是占位框，正式版本需要系统架构图；
2. 4-28 的完整 RD 曲线可引用 `../figures/e4_428_rd.pdf`；
3. Mini-Splatting 完整版矩阵仍在 LYH 运行，尚未回写最终数字；
4. 引用采用 `plainnat`；若期刊/会议要求其他格式，替换 `\bibliographystyle`。

> 数据策略更新（2026-08-28）：4-28、3-07、1-100 暂不用于本轮正式实验；大场景 110k 将改用 `/mnt` 中待选的新数据集。论文里现有 4-28 数字仅作内部初步参考，正式版会替换。

## 与提案的关系

- `../3366_摘要引言改写.md`：摘要/引言原始改写版本；
- `../3366_实验改写.md`：实验章节原始改写版本；
- `../后续实验计划.md`：下一阶段实验计划。

## 编译注意事项

本目录已补充 `iclr2027_conference.sty`, `iclr2027_conference.bst`, `math_commands.tex` 等官方模板文件。若 TeX Live 找不到本地样式，在编译目录执行即可：

```bash
cd docs/05-paper/iclr2027
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
