#!/usr/bin/env bash
# P0 (E1+E4) recovery queue for GPU1 (GPU0 is occupied by another user).
# wait r0.97 training -> finish r0.97 -> finish r0.52 -> r0.92 -> E4 110k x 3 (spa_mini, spa_base, nospa_mini).
set -euo pipefail
cd /home/fansonglin/data_space/DCCA-GS/PHG
BASE=/home/fansonglin/data_space/DCCA-GS

while ! grep -q "Training finished" "$BASE/runs/p0_e1_playroom_r097_mini/train.log" 2>/dev/null; do sleep 60; done
bash scripts/finish_p0_run.sh p0_e1_playroom_r097_mini playroom
bash scripts/finish_p0_run.sh p0_e1_playroom_r052_mini playroom
bash scripts/run_p0_pipeline_v2.sh 1 p0_e1_playroom_r092_mini playroom 1 1 0.92
bash scripts/run_p0_pipeline_v2.sh 1 p0_e4_428_110k_spa_mini 4-28 1 1 0.85 110000 45000
bash scripts/run_p0_pipeline_v2.sh 1 p0_e4_428_110k_spa_base 4-28 1 0 0.85 110000 45000
bash scripts/run_p0_pipeline_v2.sh 1 p0_e4_428_110k_nospa_mini 4-28 0 1 0.85 110000 45000

echo "QUEUE_GPU1_DONE"
