#!/bin/bash
# Wait for the 110k SPA cell, then collect + append to CSV.
set -e
RUN=/home/fansonglin/data_space/web_scan/runs/ablation_spa110k.runner.log
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/fansonglin/xieliang/chentong/PHG
cd /home/fansonglin/xieliang/chentong/PHG
until grep -q ALL_DONE "$RUN"; do
  sleep 600
done
echo "SPA_110K_ALL_DONE $(date)"
python scripts/collect_queue_results.py
echo "SPA_110K_COLLECT_DONE $(date)"
