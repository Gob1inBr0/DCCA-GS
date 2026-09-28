#!/usr/bin/env bash
# Finalize 1-78: wait for all DCCA / HAC++ jobs, archive to /mnt, then remove
# the temporary /dev/shm copies. Replaces the old watcher that only waited for
# the first three jobs and would have deleted the extra RD runs prematurely.

set -u

ROOT=/dev/shm/dcca_runs/1-78
DATA=/dev/shm/dcca_data/1-78
OUT=/mnt/newproject2/results/1-78_results.tar.zst

wait_for_all() {
  while true; do
    local initial_done=1
    for tag in lyh_178_baseline_110k lyh_178_depth_110k lyh_178_full_110k; do
      if ! grep -q "ALL_DONE tag=" "$ROOT/${tag}.launcher.log" 2>/dev/null; then
        initial_done=0
      fi
    done
    if [ "$initial_done" -eq 1 ] \
       && [ -f "$ROOT/rd_continue_done.txt" ] \
       && [ -f "$ROOT/depth_rd_done.txt" ]; then
      return 0
    fi
    sleep 300
  done
}

wait_no_training() {
  while pgrep -f \
    "train.py train|hacpp_1-78|dcca_1-78_nospa|lyh_178_(baseline|depth|full)_110k" \
    >/dev/null 2>&1; do
    sleep 120
  done
}

mkdir -p /mnt/newproject2/results

wait_for_all
wait_no_training

echo "ARCHIVE_START $(date)"
tar --zstd -cf "$OUT" -C "$ROOT" .
echo "ARCHIVE_DONE $(date)"

# Verify the archive before deleting the temporary copies.
zstd -t "$OUT"
echo "ARCHIVE_VERIFIED $(date)"

touch /mnt/newproject2/results/1-78_results.done
rm -rf "$ROOT" "$DATA"
echo "CLEANUP_DONE $(date)"
