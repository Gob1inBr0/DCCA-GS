#!/bin/bash
# ph1 wave3: complete the G1 matrix.
#   GPU 0: A2 @ lam 0.0005 s42   GPU 4: A3 @ lam 0.0005 s42
#   GPU 5: A2 @ lam 0.002  s104  GPU 7: A2 @ lam 0.004  s104
# (A3 s104 skipped unless s42 3-lambda row contradicts this plan — see
#  runbook §2: with s42 mean -0.156 the seed swing cannot reach +0.15.)
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/ph1v4_queue.log"
S0="--cfg.model.sensitivity-start-iter 0"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"

launch() {
  local g=$1 tag=$2 mode=$3 lam=$4 seed=$5
  echo "ARM $tag gpu=$g mode=$mode lam=$lam seed=$seed $(date)" >> "$QLOG"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$g" 1-78 "$DATA" "$lam" "$tag" 30000 15000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled \
    --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 \
    --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed "$seed" $S0 \
    $FUSION --cfg.model.submodular-mode "$mode" \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
}
launch 0 ph1v3_A2_cover_r085_lam0005_s42 cover 0.0005 42
launch 4 ph1v3_A3_coversens_r085_lam0005_s42 cover_sens 0.0005 42
launch 5 ph1v3_A2_cover_r085_lam0002_s104 cover 0.002 104
launch 7 ph1v3_A2_cover_r085_lam0004_s104 cover 0.004 104
echo "PH1_WAVE3_LAUNCHED $(date)" >> "$QLOG"
