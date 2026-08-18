#!/bin/bash
# 4-28 SPA cells, run after the DB SPA-RD queue finishes on both GPUs.
#   GPU0: 4-28 90k  lambda=0.004 SPA 0.5
#   GPU1: 4-28 110k lambda=0.004 SPA 0.5
set -e

GPU=$1
RUNS=/home/fansonglin/data_space/web_scan/runs
RUNROOT=/home/fansonglin/xieliang/chentong/PHG
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUNROOT"
cd "$RUNROOT"

echo "WAIT_DB_SPA_RD gpu=$GPU $(date)"
until grep -q QUEUE_SPA_RD_GPU0_ALL_DONE "$RUNS/spa_rd_gpu0.log" 2>/dev/null \
  && grep -q QUEUE_SPA_RD_GPU1_ALL_DONE "$RUNS/spa_rd_gpu1.log" 2>/dev/null; do
  sleep 600
done
echo "START_4_28_SPA gpu=$GPU $(date)"

D428=/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28
if [ "$GPU" = 0 ]; then
  bash scripts/runner_phg_cell.sh 0 "4-28" "$D428" 0.004 \
    4-28_i6_90k_h32_l0p004_spa0p5 90000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
else
  bash scripts/runner_phg_cell.sh 1 "4-28" "$D428" 0.004 \
    4-28_i6_110k_h32_l0p004_spa0p5 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
fi

echo "QUEUE_4_28_SPA_GPU${GPU}_ALL_DONE $(date)"
if [ "$GPU" = 0 ]; then
  python scripts/collect_queue_results.py
  echo "COLLECT_4_28_SPA_DONE $(date)"
fi
