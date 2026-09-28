#!/usr/bin/env bash
# Restart 1-78 30k DCCA jobs with the v4 stat-sync fix. HAC++ mid point is
# already completed and is not relaunched here.

set -u
ulimit -n 65536

ROOT=/dev/shm/dcca_runs/1-78
DATA=/dev/shm/dcca_data/1-78/data
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin

run_dcca() {
  local gpu=$1 lambda=$2 seed=$3 tag=$4
  local log="$ROOT/${tag}.log"
  echo "START_DCCA tag=$tag gpu=$gpu lambda=$lambda seed=$seed $(date)" > "$log"
  RUNS_ROOT="$ROOT" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=35000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lambda" "$tag" 30000 15000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
    --cfg.model.mini-splat-enabled \
    --cfg.model.mini-splat-reinit-iter 15000 \
    --cfg.model.mini-splat-max-new 4000 \
    --cfg.model.mini-splat-views 8 \
    --cfg.model.mini-splat-voxel 0.0 \
    --cfg.seed "$seed" \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  echo "DCCA_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda seed=$seed"
}

run_dcca_nospa() {
  local gpu=$1 lambda=$2 seed=$3 tag=$4
  local log="$ROOT/${tag}.log"
  echo "START_NOSPA tag=$tag gpu=$gpu lambda=$lambda seed=$seed $(date)" > "$log"
  RUNS_ROOT="$ROOT" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=35000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lambda" "$tag" 30000 15000 \
    --cfg.model.no-spa-enabled --cfg.model.no-mini-splat-enabled \
    --cfg.seed "$seed" \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  echo "NOSPA_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda seed=$seed"
}

mkdir -p "$ROOT"

run_dcca 0 0.004 42 lyh_178_30k_s42_l0004
run_dcca 1 0.002 42 lyh_178_30k_s42_l0002
run_dcca 2 0.0005 42 lyh_178_30k_s42_l0005
run_dcca 3 0.004 2026 lyh_178_30k_s2026_l0004
run_dcca 4 0.002 2026 lyh_178_30k_s2026_l0002
run_dcca 5 0.0005 2026 lyh_178_30k_s2026_l0005
run_dcca_nospa 7 0.004 42 lyh_178_30k_nospa_l0004
