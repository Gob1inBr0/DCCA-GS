# scripts-new 编号说明

这里是当前 UAV-DCCA-ZC 代码主线需要的训练、测试、码流对比和消融脚本。旧 `scripts/` 目录保留历史实验、绘图、审计和底层工具；新实验优先使用本目录。

实现唯一来源（2026-10-08 整理）：码流编解码只在 `scripts/c25_real_bitstream.py` + `scripts/zc_context.py`，训练 runner 只在 `scripts/runner_phg_cell.sh`，解码评估只在 `scripts/eval_decoded.py`。本目录的 05/06/07 是委托入口，`lib_zc_context.py` 是兼容垫片，不维护第二份拷贝，改实现只改 `scripts/` 一处。

## 编号入口

| 编号 | 脚本 | 作用 |
| --- | --- | --- |
| 00 | `00_train_dcca.sh` | 单次训练 -> 压缩 -> decoded-eval。支持 SPA、rate-aware SPA、B3、P0 rendering loss 开关。 |
| 01 | `01_eval_layered_bitstream.sh` | 对已有 run 生成正式分层渐进码流，默认启用 UAV-DCCA-ZC。 |
| 02 | `02_compare_zc_bitstream.sh` | 对同一 run 自动生成 legacy 与 ZCausal 两套码流，并输出对比报告。 |
| 03 | `03_run_ablation_matrix.sh` | 启动低码率消融矩阵：anchor budget、rate-aware、B3、P0、组合方案。 |
| 04 | `04_compare_zc_reports.py` | 比较 legacy/ZC 两个 JSON，输出 prefix、字段、层级字节差异。 |
| 05 | `05_encode_eval_layered_bitstream.py` | 分层码流入口，委托 `scripts/c25_real_bitstream.py`（唯一实现），被 01 调用。 |
| 06 | `06_internal_runner_phg_cell.sh` | 00 调用的内部 train -> compress -> decoded-eval runner，委托 `scripts/runner_phg_cell.sh`。 |
| 07 | `07_eval_decoded.py` | 06 调用的 decoded bitstream 渲染评估入口，委托 `scripts/eval_decoded.py`。 |

## 库文件

| 文件 | 作用 |
| --- | --- |
| `lib_zc_context.py` | 兼容垫片：实际实现在 `scripts/zc_context.py`（单测 `tests/test_zc_context.py`）。 |

## 推荐运行顺序

1. 单次训练：

```bash
bash scripts-new/00_train_dcca.sh \
  0 1-78 /path/to/PKUGS/1-78 dcca_178_r060_lam0005 0.0005 0.60
```

2. 测试正式 ZCausal 分层码流：

```bash
bash scripts-new/01_eval_layered_bitstream.sh \
  /path/to/runs/dcca_178_r060_lam0005 \
  /path/to/PKUGS/1-78 \
  analysis/s2_prefix_sweep/dcca_178_r060_zc.json \
  analysis/s2_prefix_sweep/dcca_178_r060_zc.bin
```

3. legacy/ZC 对比：

```bash
bash scripts-new/02_compare_zc_bitstream.sh \
  /path/to/runs/dcca_178_r060_lam0005 \
  /path/to/PKUGS/1-78 \
  analysis/s2_prefix_sweep \
  dcca_178_r060_lam0005
```

4. 消融矩阵：

```bash
bash scripts-new/03_run_ablation_matrix.sh \
  0,1,2,3 1-78 /path/to/PKUGS/1-78 dcca_178_matrix 0.0005
```

矩阵包含：`r085/r070/r060/r050` anchor budget 基线、`rate_r060`、`b3_r060`、`p0_r060`、`zccombo_r060`。

## 关键环境变量

- `DCCA_RATE_AWARE=1`：启用 rate-aware SPA 与 bit budget。
- `DCCA_B3=1`：启用 coarse-ladder progressive-aware training。
- `DCCA_P0_RENDER=1`：启用 P0 dequantized rendering loss。
- `DCCA_ZC_CONTEXT=off`：在 01/02 中关闭 ZCausal，跑 legacy 对照。
- `DCCA_ZC_CONTEXT=density-root`：正式 ZCausal 默认模式。
- `DCCA_ZC_CELL_SIZE=0`：由已解码几何自动推导 cell size。
- `RUNROOT` / `RUNS_ROOT` / `CONDA_ENV_BIN` / `GSPLAT_ROOT`：未设置时 06 使用仓库相对默认值（`RUNROOT`=仓库根、`RUNS_ROOT`=`$RUNROOT/runs`，不写死任何机器路径）；服务器上请显式导出 `CONDA_ENV_BIN`（HAC++ CUDA 扩展所在环境的 bin 目录）和 `GSPLAT_ROOT`。
