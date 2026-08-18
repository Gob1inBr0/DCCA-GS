#!/bin/bash
# Queue after the current DB lambda-RD cells:
#   Phase B : DB ablation (playroom 110k lambda 0.002: I6-off / I2-off)
#   Phase C : Mip360 30k x lambda {0.002,0.004} (garden, flowers, stump)
#   Phase D : Tanks&Temples 30k x lambda {0.002,0.004} (train, truck)
# Each GPU runs one lambda branch; GPU0 collects + appends to CSV at the end.
# usage: bash queue_after_db_rd.sh <gpu>
set -e

GPU=$1
RUNROOT=/home/fansonglin/xieliang/chentong/PHG
RUNS=/home/fansonglin/data_space/web_scan/runs
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUNROOT"
cd "$RUNROOT"

echo "QUEUE_WAIT_PHASE_A gpu=$GPU $(date)"
until grep -q COLLECT_DONE "$RUNS/collect_db_rd.log" 2>/dev/null; do
  sleep 600
done
echo "PHASE_A_DONE gpu=$GPU $(date)"

run_cell() {
  bash scripts/runner_phg_cell.sh "$@" || echo "CELL_FAIL $* $(date)"
}

DB=/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db
MIP=/home/fansonglin/xieliang/Chenzhenxin/dataset/360_v2
TNT=/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/tandt

if [ "$GPU" = 0 ]; then
  # Phase B: I6 off (I2 on)
  run_cell 0 playroom "$DB/playroom" 0.002 \
    db_playroom_i6_110k_h32_l0p002_ablation_i6off 110000 45000 \
    --cfg.model.no-sensitivity-enabled
  # Phase C + D: lambda 0.002
  for S in garden flowers stump; do
    run_cell 0 "mip360_${S}" "$MIP/$S" 0.002 \
      "mip360_${S}_i6_30k_h32_l0p002" 30000 15000
  done
  for S in train truck; do
    run_cell 0 "tandt_${S}" "$TNT/$S" 0.002 \
      "tandt_${S}_i6_30k_h32_l0p002" 30000 15000
  done
else
  # Phase B: I2 off (I6 off as well)
  run_cell 1 playroom "$DB/playroom" 0.002 \
    db_playroom_i6_110k_h32_l0p002_ablation_i2off 110000 45000 \
    --cfg.model.no-content-aware-quant --cfg.model.no-sensitivity-enabled
  # Phase C + D: lambda 0.004
  for S in garden flowers stump; do
    run_cell 1 "mip360_${S}" "$MIP/$S" 0.004 \
      "mip360_${S}_i6_30k_h32_l0p004" 30000 15000
  done
  for S in train truck; do
    run_cell 1 "tandt_${S}" "$TNT/$S" 0.004 \
      "tandt_${S}_i6_30k_h32_l0p004" 30000 15000
  done
fi

echo "QUEUE_GPU${GPU}_ALL_DONE $(date)"
if [ "$GPU" = 0 ]; then
  echo "COLLECT_QUEUE_START $(date)"
  python scripts/collect_queue_results.py
  echo "COLLECT_QUEUE_DONE $(date)"
fi
