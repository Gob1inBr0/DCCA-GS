#!/bin/bash
# Event-driven marker: exit once all full MiniSplat matrix cells reach ALL_DONE.
set -euo pipefail

RUNS=/home/T0ng/runs
tail -n0 -F \
  "$RUNS/lyh_full_drjohnson.launcher.log" \
  "$RUNS/lyh_full_tandt_train.launcher.log" \
  "$RUNS/lyh_full_tandt_truck.launcher.log" \
  "$RUNS/lyh_full_mip_garden.launcher.log" \
  "$RUNS/lyh_full_playroom.launcher.log" \
  "$RUNS/lyh_full_mip_bicycle.launcher.log" \
  "$RUNS/lyh_full_mip_stump.launcher.log" \
  | grep -m7 "ALL_DONE tag=" >/dev/null
echo "FULL_MATRIX_DONE" > "$RUNS/lyh_full_matrix_done"
