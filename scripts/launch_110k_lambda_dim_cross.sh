#!/bin/bash
# Launch the 4-28 110k lambda x feat_dim cross grid.
# Grid: lambda in {0.002, 0.004}, feat_dim in {16, 32, 50}.
# dim50/lambda0.004 already exists (ckpt_110000 from the 120k training run),
# so that cell only runs compress/eval/MLP-quant on the copied checkpoint.
# Everything else uses runner_4_28_110k_lambda_dim.sh (recommended config).
set -u

RUNROOT=/home/fansonglin/xieliang/chentong/PHG
LOGDIR=/home/fansonglin/data_space/web_scan/runs/logs_lxdim_110k
mkdir -p "$LOGDIR"
cd "$RUNROOT"

SRC=/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_120k/ckpts/ckpt_110000.pth

# Cell dim50/l0.004: no training, just copy + post-process (runs first on GPU0).
nohup bash scripts/runner_4_28_110k_lambda_dim.sh 0 lxdim_110k_dim50_l0p004 50 0.004 "$SRC" \
  > "$LOGDIR/cell_dim50_l0p004.log" 2>&1 &
echo "launched dim50/l0.004 (no-train) pid=$!"

# GPU0 queue: dim50/l0.002 (train), dim16/l0.004 (train)
nohup bash -c '
  bash scripts/runner_4_28_110k_lambda_dim.sh 0 lxdim_110k_dim50_l0p002 50 0.002 && echo CELL_OK || echo CELL_FAIL
  bash scripts/runner_4_28_110k_lambda_dim.sh 0 lxdim_110k_dim16_l0p004 16 0.004 && echo CELL_OK || echo CELL_FAIL
' > "$LOGDIR/gpu0_queue.log" 2>&1 &
echo "launched gpu0 queue pid=$!"

# GPU1 queue: dim32/l0.004 (train), dim32/l0.002 (train), dim16/l0.002 (train)
nohup bash -c '
  bash scripts/runner_4_28_110k_lambda_dim.sh 1 lxdim_110k_dim32_l0p004 32 0.004 && echo CELL_OK || echo CELL_FAIL
  bash scripts/runner_4_28_110k_lambda_dim.sh 1 lxdim_110k_dim32_l0p002 32 0.002 && echo CELL_OK || echo CELL_FAIL
  bash scripts/runner_4_28_110k_lambda_dim.sh 1 lxdim_110k_dim16_l0p002 16 0.002 && echo CELL_OK || echo CELL_FAIL
' > "$LOGDIR/gpu1_queue.log" 2>&1 &
echo "launched gpu1 queue pid=$!"

echo "ALL_QUEUES_LAUNCHED $(date)"
