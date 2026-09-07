#!/bin/bash
# Generalization matrix on LYH: complete base + fusion curves for 2-06 / 4-10.
# Mirrors the validated 1-78 protocol exactly (see docs/03-reports/RD图_vs_HAC++_流程记录.md):
#   30k steps, feat_dim 32, seed 42, sensitivity-start-iter 0, SPA 0.85, MiniSplat reinit@15k;
#   fusion@0.004  -> covfix config (fusion + ADMM coverage constraint), the low-rate collapse fix;
#   fusion@0.0005 -> plain fusion (mid/high rate did not collapse on 1-78);
#   fusion@0.002  -> already DONE (bcd_queue2), not relaunched here.
# 6 jobs on GPUs 0-5; a finalizer writes gen_matrix_done.txt + summary when all finish.
set -u
ulimit -n 65536

DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs

S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
FUSION="--cfg.model.fusion-prune --cfg.model.fusion-sensitivity-weight 0.5"
COVFIX="$FUSION --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-coverage-min-per-cell 1"

launch() {
  local gpu=$1 scene=$2 lam=$3 tag=$4 extra=$5
  local data="/dev/shm/dcca_data/$scene/data"
  echo "START tag=$tag gpu=$gpu scene=$scene lambda=$lam $(date)" > "$ROOT/${tag}.launch.log"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" "$scene" "$data" "$lam" "$tag" 30000 15000 $Z $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
  echo "LAUNCHED gpu=$gpu tag=$tag pid=$!"
}

launch 0 2-06 0.004  dcca_2-06_30k_32dim_l0004_base_s0   ""
launch 1 2-06 0.004  dcca_2-06_30k_32dim_l0004_covfix_s0 "$COVFIX"
launch 2 2-06 0.0005 dcca_2-06_30k_32dim_l0005_fusion_s0 "$FUSION"
launch 3 4-10 0.004  dcca_4-10_30k_32dim_l0004_base_s0   ""
launch 4 4-10 0.004  dcca_4-10_30k_32dim_l0004_covfix_s0 "$COVFIX"
launch 5 4-10 0.0005 dcca_4-10_30k_32dim_l0005_fusion_s0 "$FUSION"

# Finalizer: event-driven done marker; also records GAVE_UP failures.
TAGS="dcca_2-06_30k_32dim_l0004_base_s0 dcca_2-06_30k_32dim_l0004_covfix_s0 dcca_2-06_30k_32dim_l0005_fusion_s0 dcca_4-10_30k_32dim_l0004_base_s0 dcca_4-10_30k_32dim_l0004_covfix_s0 dcca_4-10_30k_32dim_l0005_fusion_s0"
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
    echo "GEN_MATRIX_SUMMARY $(date)"
    for t in $TAGS; do
      m=$(tail -1 "$ROOT/$t/decoded_eval/metrics.jsonl" 2>/dev/null || echo "GAVE_UP_OR_FAILED")
      mb=$(grep -o "total_MB[^,} ]*" "$ROOT/$t/compress.log" 2>/dev/null | tail -1)
      be=$(grep -ho "\"bit_exact_roundtrip\": *[a-z]*" "$ROOT/$t/bitstreams"/*.json 2>/dev/null | head -1)
      echo "$t :: $m :: $mb :: $be"
    done
  } > "$ROOT/gen_matrix_summary.txt"
  touch "$ROOT/gen_matrix_done.txt"
' > "$ROOT/gen_matrix.finalizer.log" 2>&1 < /dev/null &

echo "FINALIZER_PID=$!"
echo "GEN_MATRIX_LAUNCHED $(date)"
