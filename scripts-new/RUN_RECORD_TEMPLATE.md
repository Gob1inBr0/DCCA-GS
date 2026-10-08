# UAV-DCCA-ZC 运行记录模板

复制本文件为 `RUN_RECORD_<scene>_<date>.md`，每跑完一批实验就填一次。正式论文数字仍以 `docs/data/experiments.csv` 为准。

## 1. 环境

- 日期：
- 机器 / GPU：
- `RUNROOT`：
- `RUNS_ROOT`：
- `CONDA_ENV_BIN`：
- `GSPLAT_ROOT`：
- `CONSTRICTION_ROOT`：
- 代码版本 / git commit：
- 备注：

## 2. 数据

- 场景：
- 数据路径：
- 协议：`data_factor=1`, `max_width=1600`, `test_every=8`
- 训练步数：
- `lambda_rate`：

## 3. 训练与压缩

| tag | 命令 | 是否完成 | 关键文件 |
| --- | --- | --- | --- |
| r085 | `bash scripts-new/00_train_dcca.sh ... 0.85` |  | `provenance.txt`, `compress.log`, `eval.log`, `decoded_eval/metrics.jsonl` |
| r070 | `bash scripts-new/00_train_dcca.sh ... 0.70` |  |  |
| r060 | `bash scripts-new/00_train_dcca.sh ... 0.60` |  |  |
| r050 | `bash scripts-new/00_train_dcca.sh ... 0.50` |  |  |
| rate_r060 | `DCCA_RATE_AWARE=1 bash scripts-new/00_train_dcca.sh ... 0.60` |  |  |
| b3_r060 | `DCCA_B3=1 bash scripts-new/00_train_dcca.sh ... 0.60` |  |  |
| p0_r060 | `DCCA_P0_RENDER=1 bash scripts-new/00_train_dcca.sh ... 0.60` |  |  |
| zccombo_r060 | `DCCA_RATE_AWARE=1 DCCA_B3=1 DCCA_P0_RENDER=1 bash scripts-new/00_train_dcca.sh ... 0.60` |  |  |

## 4. ZCausal 码流对比

对重要 run 执行：

```bash
bash scripts-new/02_compare_zc_bitstream.sh \
  <run_dir> <data_dir> analysis/s2_prefix_sweep <tag>
```

| tag | legacy JSON | ZC JSON | compare txt | PSNR 是否一致 | 总字节变化 | feature 字节变化 |
| --- | --- | --- | --- | --- | --- | --- |
| r060 |  |  |  |  |  |  |
| rate_r060 |  |  |  |  |  |  |
| zccombo_r060 |  |  |  |  |  |  |

## 5. 需要写入 experiments.csv 的字段

- scene
- tag
- method / variant
- steps
- lambda
- spa_ratio
- anchors
- full decoded MB
- full PSNR / SSIM / LPIPS
- progressive prefix bytes
- progressive prefix PSNR
- ZC legacy-vs-test byte deltas
- 结论备注

## 6. 当前结论

- rate-aware SPA：
- B3：
- P0 rendering loss：
- ZCausal entropy：
- combo：
- 是否达到 CVPR 级证据门槛：
