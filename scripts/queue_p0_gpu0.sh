#!/usr/bin/env bash
# P0 (E1+E4) queue for GPU0.
# E1 playroom r0.52/r0.92 -> E4 4-28 110k (SPA0.85 baseline, then MiniSplat non-SPA).
set -euo pipefail
cd /home/fansonglin/data_space/DCCA-GS/PHG

bash scripts/run_p0_pipeline.sh 0 p0_e1_playroom_r052_mini playroom 1 1 0.52
bash scripts/run_p0_pipeline.sh 0 p0_e1_playroom_r092_mini playroom 1 1 0.92
bash scripts/run_p0_pipeline.sh 0 p0_e4_428_110k_spa_base 4-28 1 0 0.85 110000 45000
bash scripts/run_p0_pipeline.sh 0 p0_e4_428_110k_nospa_mini 4-28 0 1 0.85 110000 45000

echo "QUEUE_GPU0_DONE"
