# 30k 结果 vs HAC++ high/low 对比

> 数据源：`/home/project2/HAC-plus/results/*.csv`（官方 HAC++），
> `/home/project2/runs/lyh_full_*`、`/home/project2/runs/lyh_depth_*`（DCCA-GS 30k，
> SPA 0.85，I2+I6，MLP 混合量化 cd8/rest16）。

| 场景 | HAC++ high PSNR/MB | HAC++ low PSNR/MB | full PSNR/MB | depth PSNR/MB | full vs low ΔPSNR | full vs low 体积比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| playroom | 30.931 / 4.352 | 30.687 / 2.418 | 30.524 / 1.190 | 30.271 / 1.255 | −0.163 | 49.2% |
| drjohnson | 29.756 / 6.215 | 29.632 / 3.401 | 29.414 / 1.912 | 29.502 / 2.058 | −0.218 | 56.2% |
| T&T train | 22.603 / 6.209 | 22.542 / 4.686 | 22.166 / 3.073 | 22.277 / 3.463 | −0.376 | 65.6% |
| T&T truck | 26.062 / 7.640 | 25.888 / 5.665 | 25.388 / 3.080 | 25.380 / 3.268 | −0.500 | 54.4% |
| Mip garden | 27.486 / 27.221 | 27.195 / 12.834 | 26.181 / 5.365 | 26.291 / 5.891 | −1.014 | 41.8% |
| Mip bicycle | 25.048 / 29.250 | 25.076 / 12.843 | 24.171 / 3.662 | 24.256 / 4.107 | −0.905 | 28.5% |
| Mip stump | 26.519 / 21.024 | 26.563 / 9.398 | 25.665 / 2.481 | 25.779 / 2.812 | −0.898 | 26.4% |

## 结论

1. 30k full 点在所有场景都落在 HAC++ lowrate 的左侧：体积约为 HAC++ lowrate 的
   26%–66%，随之损失约 0.16–1.01 dB；
2. 与 HAC++ highrate 相比，体积优势更明显（约 12%–50%），PSNR 损失约 0.34–1.31 dB；
3. 这是一个明确的**低码率、高质量折衷**结果，可以作为压缩论文的 RD 点，但不能宣称
   在同等 PSNR 下更优；
4. 完整版相对 depth 版体积更小，但多数场景 PSNR 更低；这与之前的结论一致。

## BD-rate 状态

- 七场景 full/depth 每组都只有 1 个 30k 点，无法按场景算 BD-rate；
- playroom 虽有 r=0.52/0.85/0.92/0.97 四个 full 点，但 DCCA full 体积范围约为
  0.47–1.86 MB，HAC++ low/high 为 2.42–4.35 MB，**没有共享码率区间**，BD-rate
  计算无定义；
- 1-78 上 no-SPA λ=0.002（22.98 MB）落在 HAC++ 区间内，但还缺少至少一个相邻率点；
  等 no-SPA/depth 多 λ 完成后才能给正式 BD-rate；
- 已有可报告的 BD-rate 仅是 **MiniSplat vs SPA baseline**：BD-PSNR +0.124 dB、
  BD-rate −8.6%，不是对 HAC++ 的结果。
