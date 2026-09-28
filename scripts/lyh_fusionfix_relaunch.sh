#!/bin/bash
# Post-audit relaunch (2026-09-07):
#  A) kill the invalid 60k "full" trio (fusion was silently gated off) and
#     relaunch the same tags with the ENGAGED fusion code, GPUs 1/3/5;
#  B) launch the 30k real-fusion validation on 1-78 (3 lambdas) on GPUs 6/7.
set -u
ulimit -n 65536

DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data

S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 30000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
# fusion: weight defaults to 0.3 (hard-capped); coverage constraint on (cell 0.05)
FUSION_FIX="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05"

pkill -f "dcca_1-78_60k_32dim_l0004_full_s0" 2>/dev/null
pkill -f "dcca_1-78_60k_32dim_l0002_full_s0" 2>/dev/null
pkill -f "dcca_1-78_60k_32dim_l0005_full_s0" 2>/dev/null
sleep 5
rm -rf "$ROOT/dcca_1-78_60k_32dim_l0004_full_s0" "$ROOT/dcca_1-78_60k_32dim_l0002_full_s0" "$ROOT/dcca_1-78_60k_32dim_l0005_full_s0"

launch() {
  local gpu=$1 steps=$2 reinit=$3 lam=$4 tag=$5 extra=$6
  echo "START tag=$tag gpu=$gpu lambda=$lam steps=$steps $(date)" > "$ROOT/${tag}.launch.log"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lam" "$tag" "$steps" "$reinit" $Z $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
  echo "LAUNCHED gpu=$gpu tag=$tag pid=$!"
}

# A) 60k full trio, same tags so the running gen60k finalizer still covers them.
launch 1 60000 30000 0.004  dcca_1-78_60k_32dim_l0004_full_s0 "$FUSION_FIX"
launch 3 60000 30000 0.002  dcca_1-78_60k_32dim_l0002_full_s0 "$FUSION_FIX"
launch 5 60000 30000 0.0005 dcca_1-78_60k_32dim_l0005_full_s0 "$FUSION_FIX"

# B) 30k real-fusion validation on GPUs 6/7; third job waits for the first free GPU.
launch 6 30000 15000 0.004 dcca_1-78_30k_32dim_l0004_fusionfix_s0 "$FUSION_FIX"
launch 7 30000 15000 0.002 dcca_1-78_30k_32dim_l0002_fusionfix_s0 "$FUSION_FIX"

nohup bash -c '
  ROOT=/dev/shm/dcca_runs
  DCCA_ROOT=/home/project2/DCCA-GS-minifull
  DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
  DATA=/dev/shm/dcca_data/1-78/data
  S0="--cfg.model.sensitivity-start-iter 0"
  Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
  F="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05"
  tag=dcca_1-78_30k_32dim_l0005_fusionfix_s0
  while :; do
    free=""
    for g in 6 7; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
      [ "${u:-99999}" -le 3000 ] && [ ! -f "$ROOT/.glock_$g" ] && free=$g && break
    done
    if [ -n "$free" ]; then
      touch "$ROOT/.glock_$free"
      if [ ! -f "$ROOT/$tag/decoded_eval/metrics.jsonl" ]; then
        echo "START tag=$tag gpu=$free $(date)" > "$ROOT/${tag}.launch.log"
        RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
        GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
        setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
          "$free" 1-78 "$DATA" 0.0005 "$tag" 30000 15000 $Z $F \
          >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
      fi
      break
    fi
    sleep 300
  done
' > "$ROOT/fusionfix_l0005.waiter.log" 2>&1 < /dev/null &

echo "RELAUNCH_DONE $(date)"
