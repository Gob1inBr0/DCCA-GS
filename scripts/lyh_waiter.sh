#!/bin/bash
set -euo pipefail

RUNS=/home/T0ng/runs
cd /home/T0ng/DCCA-GS

FILES=(
  "$RUNS/lyh_p0_tandt_train_cell1.launcher.log"
  "$RUNS/lyh_p0_tandt_train_cell2.launcher.log"
  "$RUNS/lyh_p0_tandt_truck_cell1.launcher.log"
  "$RUNS/lyh_p0_tandt_truck_cell2.launcher.log"
  "$RUNS/lyh_p0_mip_garden_cell1.launcher.log"
)
HAVE=0
for f in "${FILES[@]}"; do
  if grep -q "ALL_DONE tag=" "$f" 2>/dev/null; then HAVE=$((HAVE + 1)); fi
done
WANT=${#FILES[@]}
if [ "$HAVE" -lt "$WANT" ]; then
  tail -n0 -F "${FILES[@]}" | grep -m "$((WANT - HAVE))" "ALL_DONE tag=" >/dev/null
fi

echo "WAVE1_P0_DONE" > "$RUNS/lyh_wave1_p0_done"
bash /home/T0ng/DCCA-GS/scripts/lyh_wave2.sh
echo "WAVE1_AND_WAVE2_DONE" > "$RUNS/lyh_waiter_done"
