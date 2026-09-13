#!/bin/bash
# Phase 0 final wave (2026-09-10 evening): 7 jobs on GPUs 1-7.
#  - seed-104 G0b replication: fusA vs base at r_post=0.85, 3 lambdas (5 jobs)
#    (base lam0002 reuses nothing: seed differs, must rerun)
#  - 0d safety regression at biting budget, lam0004 (2 jobs):
#      * nocap: fusion with UNcapped sensitivity share w=0.5 (expected: collapse
#        deeper than the 9/5 event, since the budget bites)
#      * fusA: capped+constrained (expected: no collapse)
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 $S0"
R="--cfg.model.spa-post-ratio 0.85"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 $R"

launch() {
  local gpu=$1 seed=$2 lam=$3 tag=$4 extra=$5
  echo "START tag=$tag gpu=$gpu lambda=$lam seed=$seed $(date)" > "$ROOT/${tag}.launch.log"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lam" "$tag" 30000 15000 $Z --cfg.seed "$seed" $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
  echo "LAUNCHED gpu=$gpu tag=$tag"
}

# --- seed-104 replication (G0b formal confirmation) ---
launch 1 104 0.002  ph0c_fusA_r085_lam0002_s104 "$FUSION"
launch 2 104 0.004  ph0c_fusA_r085_lam0004_s104 "$FUSION"
launch 3 104 0.0005 ph0c_fusA_r085_lam0005_s104 "$FUSION"
launch 4 104 0.004  ph0c_base_r085_lam0004_s104 "$R"
launch 5 104 0.0005 ph0c_base_r085_lam0005_s104 "$R"

# --- 0d safety regression at biting budget (seed 42) ---
# nocap: sensitivity share forced to 0.5 with NO coverage cap and NO constraint.
NOCAP="--cfg.model.fusion-prune --cfg.model.fusion-uncap --cfg.model.fusion-sensitivity-weight 0.5 --cfg.model.fusion-gamma 0.75 --cfg.model.no-spa-coverage-constraint $R"
launch 6 42 0.004  ph0d_nocap_r085_lam0004_s42 "$NOCAP"
launch 7 42 0.004  ph0d_fusA_r085_lam0004_s42 "$FUSION"

echo "PHASE0_FINAL_LAUNCHED $(date)"
