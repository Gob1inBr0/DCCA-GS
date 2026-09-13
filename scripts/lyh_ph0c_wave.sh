#!/bin/bash
# Phase 0c (混杂解除) wave: fusA vs base at r_post=0.85, seed 42 exploration.
# 4 GPUs now + 5th job waits for the first free GPU. All 30k, protocol per design §2.1.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
F="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"
B="--cfg.model.spa-post-ratio 0.85"

launch() {
  local gpu=$1 lam=$2 tag=$3 extra=$4
  echo "START tag=$tag gpu=$gpu lambda=$lam $(date)" > "$ROOT/${tag}.launch.log"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lam" "$tag" 30000 15000 $Z $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
  echo "LAUNCHED gpu=$gpu tag=$tag"
}

launch 0 0.002  ph0c_fusA_r085_lam0002_s42 "$F"
launch 1 0.004  ph0c_fusA_r085_lam0004_s42 "$F"
launch 6 0.0005 ph0c_fusA_r085_lam0005_s42 "$F"
launch 7 0.004  ph0c_base_r085_lam0004_s42 "$B"

# 5th job: base r085 @0.0005 — waits for the first GPU under 5 GB.
nohup bash -c '
  ROOT=/dev/shm/dcca_runs
  DCCA_ROOT=/home/project2/DCCA-GS-minifull
  DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
  DATA=/dev/shm/dcca_data/1-78/data
  S0="--cfg.model.sensitivity-start-iter 0"
  Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
  B="--cfg.model.spa-post-ratio 0.85"
  tag=ph0c_base_r085_lam0005_s42
  while :; do
    free=""
    for g in 0 1 6 7; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
      [ "${u:-99999}" -le 5000 ] && [ ! -f "$ROOT/.pglock_$g" ] && free=$g && break
    done
    if [ -n "$free" ]; then
      touch "$ROOT/.pglock_$free"
      [ -f "$ROOT/$tag/decoded_eval/metrics.jsonl" ] && break
      echo "START tag=$tag gpu=$free $(date)" > "$ROOT/${tag}.launch.log"
      RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
      GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
      setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
        "$free" 1-78 "$DATA" 0.0005 "$tag" 30000 15000 $Z $B \
        >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
      break
    fi
    sleep 300
  done
' > "$ROOT/ph0c_base0005.waiter.log" 2>&1 < /dev/null &

echo "PH0C_LAUNCHED $(date)"
