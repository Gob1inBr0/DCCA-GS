#!/bin/bash
# Like step_sweep_4_28.sh, but waits for each checkpoint to appear so it can
# be used while a training run is still saving checkpoints.
#
# usage: bash scripts/step_sweep_wait_4_28.sh \
#   <gpu> <run_dir> <steps like "70000 80000 100000">
set -e
GPU=$1
RUN=$2
STEPS=$3

for S in $STEPS; do
  CKPT="$RUN/ckpts/ckpt_${S}.pth"
  echo "WAIT $S $(date)"
  while [ ! -f "$CKPT" ]; do
    sleep 60
  done
  bash /home/fansonglin/xieliang/chentong/PHG/scripts/step_sweep_4_28.sh \
    "$GPU" "$RUN" "$S"
done
echo "WAIT_STEP_SWEEP_ALL_DONE $(date)"
