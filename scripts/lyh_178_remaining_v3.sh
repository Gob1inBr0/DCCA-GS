#!/usr/bin/env bash
# Restart remaining 1-78 jobs with the v3 stat-sync fix:
#   4 depth-reinit lambda points + 1 no-SPA lambda point (0.001).
# Writes both depth_rd_done.txt and rd_continue_done.txt only when all finish.

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

run_nospa() {
  local gpu=$1 lambda=$2 tag=$3
  local log="$ROOT/${tag}.log"
  echo "START_NOSPA tag=$tag gpu=$gpu lambda=$lambda $(date)" > "$log"
  RUNS_ROOT="$ROOT" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=30000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lambda" "$tag" 110000 45000 \
    --cfg.model.no-spa-enabled \
    --cfg.model.no-mini-splat-enabled \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  echo "NOSPA_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda"
}

wait_pid_file() {
  local name=$1
  local file="$ROOT/${name}.pid"
  [ -f "$file" ] || { echo "NO_PID_FILE $name"; return 0; }
  local pid
  pid=$(cat "$file")
  tail --pid="$pid" -f /dev/null || true
  echo "EXIT $name pid=$pid $(date)"
}

mkdir -p "$ROOT"

run_depth 0 0.004 lyh_178_depth_110k_l0004
run_depth 1 0.002 lyh_178_depth_110k_l0002
run_depth 2 0.001 lyh_178_depth_110k_l0001
run_depth 3 0.0005 lyh_178_depth_110k_l0005
run_nospa 4 0.001 dcca_1-78_nospa_l0001

for tag in lyh_178_depth_110k_l0004 \
            lyh_178_depth_110k_l0002 \
            lyh_178_depth_110k_l0001 \
            lyh_178_depth_110k_l0005 \
            dcca_1-78_nospa_l0001; do
  wait_pid_file "$tag"
done

echo "DEPTH_RD_DONE $(date)" | tee "$ROOT/depth_rd_done.txt"
echo "RD_CONTINUE_DONE $(date)" | tee "$ROOT/rd_continue_done.txt"
