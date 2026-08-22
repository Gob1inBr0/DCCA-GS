#!/usr/bin/env bash
# P0 (E1+E4) queue for GPU1.
# 4-28 SPA0.85 + MiniSplat -> E1 playroom budget points (r=0.52/0.92/0.97).
set -euo pipefail
cd /home/fansonglin/data_space/DCCA-GS/PHG

bash scripts/run_p0_pipeline.sh 1 p0_e4_428_spa_mini 4-28 1 1 0.85
bash scripts/run_p0_pipeline.sh 1 p0_e1_playroom_r052_mini playroom 1 1 0.52
bash scripts/run_p0_pipeline.sh 1 p0_e1_playroom_r092_mini playroom 1 1 0.92
bash scripts/run_p0_pipeline.sh 1 p0_e1_playroom_r097_mini playroom 1 1 0.97

echo "QUEUE_GPU1_DONE"
