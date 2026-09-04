# 1-78 30k · 32-dim · DCCA-GS vs HAC++ 实验报告

> 日期：2026-09-04（训练已完成）
> 平台：LYH（root@36.138.63.143:65022，8×A100-40GB）
> 目的：验证改为 **feat_dim=32** 后，DCCA-GS 在 **30k**、无人机场景 **1-78** 上与 **HAC++**
> 做**公平比较**（同场景、同迭代、同 λ 集、同数据口径）。目标：**尽量与其他方法公平可比的 RD 结果**。

---

## 1. 实验设置（公平性关键项）

| 项 | 值 | 公平性说明 |
| --- | --- | --- |
| 场景 | 1-78（无人机，1200 图像 / 1050 train / 150 val） | 与 HAC++ 同数据（`/dev/shm/dcca_data/1-78/data`） |
| 迭代 | 30000 | 与 HAC++ 1-78 30k 参考点一致 |
| λ 集 | 0.004 / 0.002 / 0.0005 | 与 HAC++ 参考点相同的三个 λ |
| seed | 2026 之外用 42；DCCA 与 HAC++ 同 seed 口径 | 随机性对齐 |
| feat_dim | **32** | 本次改用 32（此前 50，见 `feat_dim扫描报告.md` 判定 32 更优） |
| n_offsets | 10 | 一致 |
| 协议 | data_factor=1、max_width=1600、test_every=8 | 统一评估协议 |
| DCCA 配置 | I2 + I6 + SPA(0.85) + MiniSplat(depth-reinit, 15000) + MLP quant(cd8/rest16) | 方法完整闭环 |
| 训练/编码/评估 | train → compress → decode → eval → MLP quant | 指标来自**解码后真实 bitstream** |
| 体积口径 | `decoded_eval.metrics_summary.codec_total_mb`（fp32）或 `mlp_quant_cd8_rest16.results.json.total_MB`（MLP量化） | **两方法须用同一口径**，见 §5 |

---

## 2. HAC++ 1-78 30k 参考（HANDOVER_20260902 §6.1）

| λ | 别名 | PSNR | SSIM | LPIPS | total_MB |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.004 | hacpp_1-78_high_l0004 | 27.8406 | 0.8829 | 0.1597 | 20.0265 |
| 0.002 | hacpp_1-78_mid_l0002 | 28.2105 | 0.8903 | 0.1514 | 26.2734 |
| 0.0005 | hacpp_1-78_low_l0005 | 28.6780 | 0.9001 | 0.1396 | 40.5589 |

> 注意：tag 的 high/low 指**体积语义**（λ=0.004 是 lowrate、λ=0.0005 是 highrate），不是 λ 大小。

---

## 3. DCCA-GS 1-78 30k（32-dim，seed 42）结果

来源：`/dev/shm/dcca_runs/1-78/dcca_1-78_30k_32dim_l000{4,2,5}/`。指标来自解码后真实 bitstream
（`decoded_eval/metrics.jsonl`），体积来自 `compress.log` 的 `total_MB`（fp32 口径，`bit_mlp` 为
float32≈0.16MB）与 `mlp_quant_cd8_rest16/results.json` 的 `total_MB`（MLP 量化后）。三单元均
`bit_exact_roundtrip=True`。

| λ | run | PSNR | SSIM | LPIPS | total_MB(fp32) | total_MB(MLP量化) | anchors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.004 | `dcca_1-78_30k_32dim_l0004` | 27.422 | 0.8638 | 0.1814 | 11.7835 | 11.7167 | 506,186 |
| 0.002 | `dcca_1-78_30k_32dim_l0002` | 27.642 | 0.8703 | 0.1718 | 15.3069 | 15.2408 | 557,363 |
| 0.0005 | `dcca_1-78_30k_32dim_l0005` | 27.980 | 0.8788 | 0.1605 | 23.7178 | 23.6491 | 796,719 |

**对比 HAC++ 参考（§2）**：同 λ 下 DCCA 体积明显更小（≈59%），但 PSNR 也低 0.42–0.70 dB。

**BD-rate**（共享 PSNR 区间 [27.841, 27.980]，很窄）：

| 口径 | BD-rate vs HAC++ |
| --- | ---: |
| DCCA fp32 total_MB | **+2.79%** |
| DCCA MLP 量化 total_MB | **+2.46%** |

> BD-rate 为正=在相同 PSNR 下 DCCA 用了更多码率（略差）。负值才表示更好。

---

## 4. 结果判读（待填）

### 4.1 判读

1. **DCCA 30k 显著更小、但画质略低**：三档 `total_MB` 约为 HAC++ 的 58–59%（
   11.78/20.03、15.31/26.27、23.72/40.56），但 PSNR 低 0.42–0.70 dB。DCCA 的曲线整体位于
   HAC++ 的左下方（更低码率、更低 PSNR）。
2. **BD-rate 为正（略差）**：在两条曲线唯一共有的窄 PSNR 区间 [27.84, 27.98] 内，
   DCCA 在相等质量下用了约 +2.46~2.79% 的码率。**结论：在 1-78 30k 协议下，DCCA-GS(32-dim)
   未能在"同质量更省"的意义上胜过 HAC++**；它换来的是"更小的体积 + 更低的画质"。
3. **窄区间导致 BD-rate 不稳健**：两条曲线 PSNR 几乎不重叠（DCCA 上限 27.98，HAC++ 下限 27.84），
   共享区仅 0.14 dB，BD-rate 在此区间主要受 DCCA 的 λ=0.0005 这一点与 HAC++ 外推影响。
   因此 +2.8% 只能作为"同质量大致相当、略偏差"的参考，不应过度解读。
4. **与 `feat_dim扫描` 的预期并不矛盾**：扫描显示 32-dim 优于 50-dim 是**相对自身维度**；
   这里与 HAC++ 对比体现的是**方法 vs 基线**的整体差距（SPA/MiniSplat/AQM 在 30k 小迭代下的
   表现），与维度无关。
5. **30k 是低迭代设置**：HANDOVER §6.2 中 DCCA 1-78 在 **110k** 无-SPA 达 28.46–29.24 dB，
   明显高于 30k 的 27.42–27.98。说明该协议下 DCCA 尚未充分发挥，30k 不足以体现方法优势。

---

## 5. 公平性/口径说明（重要）

- **同场景、同数据、同 30k、同 λ 集、同 seed**，这是与 HAC++ 公平比较的关键。
- **体积口径**必须统一：
  - DCCA 的 `total_MB(fp32)` 来自 `compress.log` 的 `total_MB`（`bit_mlp`=float32≈0.16MB），
    与 HAC++（原生 float32 MLP）口径一致 → **主比较用 fp32**；
  - DCCA 的 `total_MB(MLP量化)`（含创新点②压 MLP）是"方法最终尺寸"，仅作参考，单独标出，避免口径错配。
- **feat_dim 差异**：DCCA 用 32-dim（用户指定），HAC++ 参考点未注明 feat_dim（默认多为 50）。
  维度差异会小幅偏向 DCCA 的 feat 体积，但本结果 DCCA 仍偏差，因此不改变方向性结论。
- **PSNR 区间几乎不重叠**：BD-rate 的共享区间仅 0.14 dB，稳健性有限，结论应按"大致相当、略偏差"解读。

---

## 6. 结论（待填）

**结论：** 在 **1-78 · 30k · 32-dim · 同 λ、同数据、seed 42** 的公平对比下，DCCA-GS 相比 HAC++
**体积大幅更小（约 59%）但 PSNR 略低（0.42–0.70 dB）**；在仅有的窄 PSNR 重叠区，BD-rate ≈ **+2.5~2.8%**
（DCCA 略差）。因此 **30k 协议下 DCCA-GS 未实现"同质量更省码率"的胜出**，而是交出了
"更小体积/更低调质"的权衡点。要体现方法 RD 优势，需在更高迭代（如 110k）或更充分的架构/量化下复测。

---

## 7. 复现

```bash
# 训练+全闭环（LYH）：
cd /home/project2/DCCA-GS-minifull
RUNROOT=/home/project2/DCCA-GS-minifull RUNS_ROOT=/dev/shm/dcca_runs/1-78 \
CONDA_ENV_BIN=/home/project2/miniconda3/envs/DCCA/bin GSPLAT_ROOT=/home/project2/gsplat-main \
WAIT_VRAM_MB=35000 bash scripts/runner_phg_cell.sh <gpu> 1-78 /dev/shm/dcca_data/1-78/data <lambda> <tag> 30000 15000 \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
  --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
  --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 \
  --cfg.seed 42

# 采集 + BD-rate：
python scripts/collect_lyh_rd.py --scene 1-78 --run-root /dev/shm/dcca_runs/1-78 --out /home/project2/runs/rd_summary_1-78.json
python scripts/bdrate_compare_178.py --run-root /dev/shm/dcca_runs/1-78 --source fp32
```

---

## 8. 参考

- `../06-planning/HANDOVER_20260902.md`（§6.1 HAC++ 参考点、§2 30k 协议）
- `../03-reports/feat_dim扫描报告.md`（32-dim 判定）
- `scripts/bdrate_compare_178.py`、`scripts/collect_lyh_rd.py`、`scripts/runner_phg_cell.sh`
