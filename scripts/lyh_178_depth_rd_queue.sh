#!/usr/bin/env bash
# 1-78 depth-reinit multi-lambda queue (SPA 0.85 + MiniSplat depth, no full).
# Uses GPU1/2/4/5 after the old depth/full retries are stopped; the baseline
# and no-SPA jobs on GPU0/3/6/7 are untouched. Writes depth_rd_done.txt after
# all points have completed.

set -u

ulimit -n 65536

ROOT=/dev/shm/dcca_runs/1-78
DATA=/dev/shm/dcca_data/1-78/data
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin

run_depth() {
  local gpu=$1 lambda=$2 tag=$3
  local log="$ROOT/${tag}.log"
  echo "START_DEPTH tag=$tag gpu=$gpu lambda=$lambda $(date)" > "$log"
  RUNS_ROOT="$ROOT" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=30000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lambda" "$tag" 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
    --cfg.model.mini-splat-enabled \
    --cfg.model.mini-splat-reinit-iter 15000 \
    --cfg.model.mini-splat-max-new 4000 \
    --cfg.model.mini-splat-views 8 \
    --cfg.model.mini-splat-voxel 0.0 \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  echo "DEPTH_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda"
}

wait_pid_file() {
  local name=$1
  local file="$ROOT/${name}.pid"
  [ -f "$file" ] || { echo "NO_PID_FILE $name"; return 0; }
  local pid
  pid=$(cat "$file")
  tail --pid="$pid" -f /dev/null || true
  echo "DEPTH_EXIT $name pid=$pid $(date)"
}

mkdir -p "$ROOT"

# Four depth-reinit lambda points, all with the patched stat-sync core.
run_depth 1 0.004 lyh_178_depth_110k_l0004
run_depth 4 0.002 lyh_178_depth_110k_l0002
run_depth 5 0.001 lyh_178_depth_110k_l0001
run_depth 2 0.0005 lyh_178_depth_110k_l0005

for tag in lyh_178_depth_110k_l0004 \
            lyh_178_depth_110k_l0002 \
            lyh_178_depth_110k_l0001 \
            lyh_178_depth_110k_l0005; do
  wait_pid_file "$tag"
done

echo "DEPTH_RD_DONE $(date)" | tee "$ROOT/depth_rd_done.txt"
