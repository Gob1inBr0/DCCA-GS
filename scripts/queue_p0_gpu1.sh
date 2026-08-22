#!/usr/bin/env bash
# P0 (E1+E4) queue for GPU1.
# E1 playroom r0.97 -> E4 4-28 110k (SPA0.85 + MiniSplat).
set -euo pipefail
cd /home/fansonglin/data_space/DCCA-GS/PHG

bash scripts/run_p0_pipeline.sh 1 p0_e1_playroom_r097_mini playroom 1 1 0.97
bash scripts/run_p0_pipeline.sh 1 p0_e4_428_110k_spa_mini 4-28 1 1 0.85 110000 45000

echo "QUEUE_GPU1_DONE"
