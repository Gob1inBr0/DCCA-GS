#!/bin/bash
# one-shot: launch I6-off (SOG control) when a GPU has <=4500MB used.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-git
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
export PATH="/home/project2/tmc13:$PATH"
QLOG="$ROOT/jr2_wave3.log"
TAG=r2_I6off_new_r085_lam0002_s42
while true; do
  g=""
  for c in 0 1 2 3 4 5 6 7; do
    [ -f "$ROOT/.qlock_$c" ] && continue
    used=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 99999)
    [ "${used:-99999}" -le 4500 ] && g=$c && break
  done
  [ -n "$g" ] && break
  sleep 300
done
echo "I6OFF gpu=$g $(date)" >> "$QLOG"
echo "$TAG" > "$ROOT/.qlock_$g"
cd "$DCCA_ROOT"
RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
  "$g" 1-78 "$DATA" 0.002 "$TAG" 30000 15000 \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled \
  --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 \
  --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
  --cfg.model.no-sensitivity-enabled \
  --cfg.model.spa-post-ratio 0.85 \
  >> "$ROOT/${TAG}.launch.log" 2>&1 < /dev/null &
echo "I6OFF_LAUNCHED gpu=$g $(date)" >> "$QLOG"
