#!/bin/bash
# ph1 wave2: submodular follow-up arms at the tightest budget (lam 0.004).
# GPU 0 -> A2 (cover v1), GPU 4 -> A3 (cover_sens v2). Same code state as
# the running A2/A3@0.002 arms (ph1 fixes only) — single-variable lambda.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/ph1v4_queue.log"
S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"

launch() {
  local g=$1 tag=$2 mode=$3 lam=$4
  echo "ARM $tag gpu=$g mode=$mode lam=$lam $(date)" >> "$QLOG"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$g" 1-78 "$DATA" "$lam" "$tag" 30000 15000 $Z $FUSION \
    --cfg.model.submodular-mode "$mode" \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
}
launch 0 ph1v3_A2_cover_r085_lam0004_s42 cover 0.004
launch 4 ph1v3_A3_coversens_r085_lam0004_s42 cover_sens 0.004
echo "PH1_WAVE2_LAUNCHED $(date)" >> "$QLOG"
