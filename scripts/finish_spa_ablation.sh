#!/bin/bash
# Wait for the corrected SPA ablation cells (I2+SPA / SPA-only, 110k), collect.
set -e
RUNS=/home/fansonglin/data_space/web_scan/runs
L0=$RUNS/ablation_spa_i6off.runner.log
L1=$RUNS/ablation_spa_i2off.runner.log
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/fansonglin/xieliang/chentong/PHG
cd /home/fansonglin/xieliang/chentong/PHG
until grep -q ALL_DONE "$L0" && grep -q ALL_DONE "$L1"; do
  sleep 600
done
echo "SPA_ABLATION_ALL_DONE $(date)"
python scripts/collect_queue_results.py
echo "SPA_ABLATION_COLLECT_DONE $(date)"
