#!/usr/bin/env bash
# Launch the 1-78 30k protocol on the current 8-GPU host.
# The host is shared with SHARC tasks; we stack jobs with WAIT_VRAM_MB=35000
# and never kill the existing SHARC processes.

set -u
ulimit -n 65536

ROOT=/dev/shm/dcca_runs/1-78
DATA=/dev/shm/dcca_data/1-78/data
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
HAC_ROOT=/home/project2/HAC-plus
HAC_PY=/mnt/003/conda_envs/HAC_plus/bin

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

run_hacpp() {
  local gpu=$1 lambda=$2 tag=$3
  local log="$ROOT/${tag}.log"
  cd "$HAC_ROOT" || exit 1
  echo "START_HACPP tag=$tag gpu=$gpu lambda=$lambda $(date)" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PATH="/home/project2/tmc13/build/tmc3:$PATH" \
  nohup "$HAC_PY/python" train.py \
    -s "$DATA" --eval --lod 0 --resolution -1 \
    --voxel_size 0.001 --update_init_factor 16 --iterations 30000 \
    -m "$ROOT/$tag" \
    --lmbda "$lambda" --mask_lr_final 0.0002 \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  echo "HACPP_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda"
}

mkdir -p "$ROOT"

# DCCA 3 lambda x 2 seeds, depth-reinit + SPA 0.85.
run_dcca 0 0.004 42 lyh_178_30k_s42_l0004
run_dcca 1 0.002 42 lyh_178_30k_s42_l0002
run_dcca 2 0.0005 42 lyh_178_30k_s42_l0005
run_dcca 3 0.004 2026 lyh_178_30k_s2026_l0004
run_dcca 4 0.002 2026 lyh_178_30k_s2026_l0002
run_dcca 5 0.0005 2026 lyh_178_30k_s2026_l0005

# HAC++ missing middle lambda point.
run_hacpp 6 0.002 hacpp_1-78_mid_l0002

# A no-SPA high-rate point for the DCCA-vs-HAC comparison.
run_dcca_nospa 7 0.004 42 lyh_178_30k_nospa_l0004
