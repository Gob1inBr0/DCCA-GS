#!/bin/bash
# 60k iteration study on LYH (HAC++ Tab-XII analog): 1-78, 30k protocol scaled to 60k.
#   update_until 30000 (densify first half), mini-splat-reinit 30000, seed 42, feat_dim 32.
#   base (I2+I6) and full-method (fusion, covfix@0.004) x 3 lambdas = 6 jobs, GPUs 0-5.
set -u
ulimit -n 65536

DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs

S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 30000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
FUSION="--cfg.model.fusion-prune --cfg.model.fusion-sensitivity-weight 0.5"
COVFIX="$FUSION --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-coverage-min-per-cell 1"
DATA=/dev/shm/dcca_data/1-78/data

launch() {
  local gpu=$1 lam=$2 tag=$3 extra=$4
  echo "START tag=$tag gpu=$gpu lambda=$lam $(date)" > "$ROOT/${tag}.launch.log"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lam" "$tag" 60000 30000 $Z $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
  echo "LAUNCHED gpu=$gpu tag=$tag pid=$!"
}

launch 0 0.004  dcca_1-78_60k_32dim_l0004_base_s0   ""
launch 1 0.004  dcca_1-78_60k_32dim_l0004_full_s0   "$COVFIX"
launch 2 0.002  dcca_1-78_60k_32dim_l0002_base_s0   ""
launch 3 0.002  dcca_1-78_60k_32dim_l0002_full_s0   "$FUSION"
launch 4 0.0005 dcca_1-78_60k_32dim_l0005_base_s0   ""
launch 5 0.0005 dcca_1-78_60k_32dim_l0005_full_s0   "$FUSION"

TAGS="dcca_1-78_60k_32dim_l0004_base_s0 dcca_1-78_60k_32dim_l0004_full_s0 dcca_1-78_60k_32dim_l0002_base_s0 dcca_1-78_60k_32dim_l0002_full_s0 dcca_1-78_60k_32dim_l0005_base_s0 dcca_1-78_60k_32dim_l0005_full_s0"
nohup bash -c '
  ROOT=/dev/shm/dcca_runs
  TAGS="'"$TAGS"'"
  while :; do
    alldone=1
    for t in $TAGS; do
      if [ ! -f "$ROOT/$t/decoded_eval/metrics.jsonl" ] && ! grep -q "GAVE_UP" "$ROOT/$t.launch.log" 2>/dev/null; then
        alldone=0
      fi
    done
    [ "$alldone" = 1 ] && break
    sleep 600
  done
  {
    echo "GEN60K_SUMMARY $(date)"
    for t in $TAGS; do
      m=$(tail -1 "$ROOT/$t/decoded_eval/metrics.jsonl" 2>/dev/null || echo "GAVE_UP_OR_FAILED")
      mb=$(grep -oE "total_MB.?: *[0-9.]+" "$ROOT/$t/compress.log" 2>/dev/null | tail -1)
      be=$(grep -ho "\"bit_exact_roundtrip\": *[a-z]*" "$ROOT/$t/bitstreams"/*.json 2>/dev/null | head -1)
      echo "$t :: $m :: $mb :: $be"
    done
  } > "$ROOT/gen60k_summary.txt"
  touch "$ROOT/gen60k_done.txt"
' > "$ROOT/gen60k.finalizer.log" 2>&1 < /dev/null &

echo "FINALIZER_PID=$!"
echo "GEN60K_LAUNCHED $(date)"
