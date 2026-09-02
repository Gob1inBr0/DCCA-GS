#!/usr/bin/env bash
# DCCA-GS full 30k task queue:
#   1-78 current 7 jobs are already running; this queue watches them.
#   When GPUs free, it launches 2-06 / 4-10 main+no-SPA and 1-78 ablations.

set -u
ulimit -n 65536

DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA_ROOT=/dev/shm/dcca_data
LOCKDIR=/tmp/dcca_all_queue_locks
mkdir -p "$LOCKDIR"
rm -f "$LOCKDIR"/*.lock

prepare_data_206() {
  mkdir -p "$DATA_ROOT/2-06/data"
  ln -sfn "$DATA_ROOT/2-06_extract/2-06/colmap/images" "$DATA_ROOT/2-06/data/images"
  ln -sfn "$DATA_ROOT/2-06_extract/2-06/colmap/colmap/sparse" "$DATA_ROOT/2-06/data/sparse"
}

prepare_data_410() {
  mkdir -p "$DATA_ROOT/4-10/data"
  ln -sfn "$DATA_ROOT/4-10_extract/4-10/colmap/images" "$DATA_ROOT/4-10/data/images"
  ln -sfn "$DATA_ROOT/4-10_extract/4-10/colmap/colmap/sparse" "$DATA_ROOT/4-10/data/sparse"
}

free_gpu() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    if [ -f "$LOCKDIR/gpu_$gpu.lock" ]; then continue; fi
    if pgrep -f "^bash /home/project2/DCCA-GS-minifull/scripts/runner_phg_cell.sh $gpu " >/dev/null 2>&1; then continue; fi
    touch "$LOCKDIR/gpu_$gpu.lock"
    echo "$gpu"
    return 0
  done
  return 1
}

active_count() {
  pgrep -f "^bash /home/project2/DCCA-GS-minifull/scripts/runner_phg_cell.sh " 2>/dev/null | wc -l
}

run_job() {
  local scene=$1 data=$2 runroot=$3 tag=$4 lambda=$5 seed=$6 mode=$7 ratio=$8
  local gpu
  gpu=$(free_gpu) || return 1
  mkdir -p "$runroot"
  local log="$runroot/${tag}.log"
  local extra=()
  case "$mode" in
    main)
      extra=(--cfg.model.spa-enabled --cfg.model.spa-ratio "$ratio"
            --cfg.model.mini-splat-enabled
            --cfg.model.mini-splat-reinit-iter 15000
            --cfg.model.mini-splat-max-new 4000
            --cfg.model.mini-splat-views 8
            --cfg.model.mini-splat-voxel 0.0)
      ;;
    nospa)
      extra=(--cfg.model.no-spa-enabled --cfg.model.no-mini-splat-enabled)
      ;;
    full)
      extra=(--cfg.model.spa-enabled --cfg.model.spa-ratio "$ratio"
            --cfg.model.mini-splat-enabled
            --cfg.model.mini-splat-reinit-iter 15000
            --cfg.model.mini-splat-max-new 4000
            --cfg.model.mini-splat-views 8
            --cfg.model.mini-splat-voxel 0.0
            --cfg.model.mini-splat-full)
      ;;
    i2off)
      extra=(--cfg.model.no-content-aware-quant --cfg.model.no-sensitivity-enabled)
      ;;
    i6off)
      extra=(--cfg.model.no-sensitivity-enabled)
      ;;
    spa_ratio)
      extra=(--cfg.model.spa-enabled --cfg.model.spa-ratio "$ratio"
            --cfg.model.mini-splat-enabled
            --cfg.model.mini-splat-reinit-iter 15000
            --cfg.model.mini-splat-max-new 4000
            --cfg.model.mini-splat-views 8
            --cfg.model.mini-splat-voxel 0.0)
      ;;
  esac
  echo "START tag=$tag scene=$scene gpu=$gpu mode=$mode lambda=$lambda seed=$seed $(date)" > "$log"
  RUNS_ROOT="$runroot" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=35000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" "$scene" "$data" "$lambda" "$tag" 30000 15000 \
    --cfg.seed "$seed" "${extra[@]}" \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$runroot/${tag}.pid"
  echo "JOB_BG pid=$pid tag=$tag gpu=$gpu mode=$mode"
  (
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    rm -f "$LOCKDIR/gpu_$gpu.lock"
  ) &
}

prepare_data_206

JOBS=(
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_s42_l0004|0.004|42|main|0.85"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_s42_l0002|0.002|42|main|0.85"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_s42_l0005|0.0005|42|main|0.85"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_s2026_l0004|0.004|2026|main|0.85"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_s2026_l0002|0.002|2026|main|0.85"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_s2026_l0005|0.0005|2026|main|0.85"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_nospa_l0004|0.004|42|nospa|0"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_nospa_l0002|0.002|42|nospa|0"
  "2-06|$DATA_ROOT/2-06/data|$ROOT/2-06|lyh_206_30k_nospa_l0005|0.0005|42|nospa|0"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_s42_l0004|0.004|42|main|0.85"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_s42_l0002|0.002|42|main|0.85"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_s42_l0005|0.0005|42|main|0.85"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_s2026_l0004|0.004|2026|main|0.85"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_s2026_l0002|0.002|2026|main|0.85"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_s2026_l0005|0.0005|2026|main|0.85"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_nospa_l0004|0.004|42|nospa|0"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_nospa_l0002|0.002|42|nospa|0"
  "4-10|$DATA_ROOT/4-10/data|$ROOT/4-10|lyh_410_30k_nospa_l0005|0.0005|42|nospa|0"
  "1-78|$DATA_ROOT/1-78/data|$ROOT/1-78|lyh_178_30k_i2off|0.004|42|i2off|0"
  "1-78|$DATA_ROOT/1-78/data|$ROOT/1-78|lyh_178_30k_i6off|0.004|42|i6off|0"
  "1-78|$DATA_ROOT/1-78/data|$ROOT/1-78|lyh_178_30k_full|0.004|42|full|0.85"
  "1-78|$DATA_ROOT/1-78/data|$ROOT/1-78|lyh_178_30k_spa_r052|0.004|42|spa_ratio|0.52"
  "1-78|$DATA_ROOT/1-78/data|$ROOT/1-78|lyh_178_30k_spa_r092|0.004|42|spa_ratio|0.92"
  "1-78|$DATA_ROOT/1-78/data|$ROOT/1-78|lyh_178_30k_spa_r097|0.004|42|spa_ratio|0.97"
)

for job in "${JOBS[@]}"; do
  while [ "$(active_count)" -ge 8 ]; do sleep 30; done
  IFS='|' read -r scene data runroot tag lambda seed mode ratio <<< "$job"
  if [ "$scene" = "4-10" ]; then
    while [ ! -f "$DATA_ROOT/4-10_extract/DONE" ]; do sleep 60; done
    prepare_data_410
  fi
  run_job "$scene" "$data" "$runroot" "$tag" "$lambda" "$seed" "$mode" "$ratio" || {
    echo "QUEUE_JOB_FAILED tag=$tag $(date)"
  }
done

# Wait for all launched jobs before marking done.
while [ "$(active_count)" -gt 0 ]; do sleep 30; done
echo "DCCA_ALL_QUEUE_DONE $(date)" | tee /dev/shm/dcca_runs/dcca_all_queue_done.txt
