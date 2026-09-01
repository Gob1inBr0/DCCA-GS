#!/bin/bash
set -euo pipefail

cd /home/T0ng/DCCA-GS
RUNS=/home/T0ng/runs
CONDA_ENV_BIN=/home/T0ng/miniconda3/envs/DCCA/bin

launch() {
  local gpu=$1
  local scene=$2
  local data=$3
  local tag=$4
  local steps=$5
  local update=$6
  shift 6
  CONDA_ENV_BIN="$CONDA_ENV_BIN" \
  RUNS_ROOT="$RUNS" \
  RUNROOT=/home/T0ng/DCCA-GS \
  nohup bash scripts/runner_phg_cell.sh \
    "$gpu" "$scene" "$data" 0.004 "$tag" "$steps" "$update" "$@" \
    > "$RUNS/$tag.launcher.log" 2>&1 &
  echo "$tag pid=$!"
}

wait_batch() {
  local files=("$@")
  tail -n0 -F "${files[@]}" | grep -m "${#files[@]}" "ALL_DONE tag=" >/dev/null
}

SPA_ARGS=(--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85)
MINI_ARGS=(
  --cfg.model.mini-splat-enabled
  --cfg.model.mini-splat-reinit-iter 15000
  --cfg.model.mini-splat-max-new 4000
  --cfg.model.mini-splat-views 16
  --cfg.model.mini-splat-voxel 0.0
)

launch 2 mip_garden /home/data/mipnerf360/garden lyh_p0_mip_garden_cell2 30000 15000 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 3 mip_bicycle /home/data/mipnerf360/bicycle lyh_p0_mip_bicycle_cell1 30000 15000 "${SPA_ARGS[@]}"
launch 4 mip_bicycle /home/data/mipnerf360/bicycle lyh_p0_mip_bicycle_cell2 30000 15000 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 5 mip_stump /home/data/mipnerf360/stump lyh_p0_mip_stump_cell1 30000 15000 "${SPA_ARGS[@]}"
launch 6 mip_stump /home/data/mipnerf360/stump lyh_p0_mip_stump_cell2 30000 15000 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"

wait_batch \
  "$RUNS/lyh_p0_mip_garden_cell2.launcher.log" \
  "$RUNS/lyh_p0_mip_bicycle_cell1.launcher.log" \
  "$RUNS/lyh_p0_mip_bicycle_cell2.launcher.log" \
  "$RUNS/lyh_p0_mip_stump_cell1.launcher.log" \
  "$RUNS/lyh_p0_mip_stump_cell2.launcher.log"

echo "WAVE2_DONE" > "$RUNS/lyh_wave2_done"
