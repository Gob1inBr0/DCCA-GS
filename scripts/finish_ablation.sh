#!/bin/bash
# Wait for both DB ablation cells, then collect + append to CSV.
set -e
RUNS=/home/fansonglin/data_space/web_scan/runs
L0=$RUNS/db_playroom_i6_110k_h32_l0p002_ablation_i6off.log
L1=$RUNS/db_playroom_i6_110k_h32_l0p002_ablation_i2off.log
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/fansonglin/xieliang/chentong/PHG
cd /home/fansonglin/xieliang/chentong/PHG
until grep -q ALL_DONE "$L0" && grep -q ALL_DONE "$L1"; do
  sleep 600
done
echo "ABLATION_ALL_DONE $(date)"
python scripts/collect_queue_results.py
echo "ABLATION_COLLECT_DONE $(date)"
