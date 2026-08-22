#!/usr/bin/env bash
# P0 (E1+E4) queue for GPU0.
# 4-28 SPA0.85 baseline -> 4-28 MiniSplat (non-SPA).
set -euo pipefail
cd /home/fansonglin/data_space/DCCA-GS/PHG

bash scripts/run_p0_pipeline.sh 0 p0_e4_428_spa_base 4-28 1 0 0.85
bash scripts/run_p0_pipeline.sh 0 p0_e4_428_nospa_mini 4-28 0 1 0.85

echo "QUEUE_GPU0_DONE"
