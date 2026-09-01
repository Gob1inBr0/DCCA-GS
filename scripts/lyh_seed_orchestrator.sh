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
  shift 4
  CONDA_ENV_BIN="$CONDA_ENV_BIN" \
  RUNS_ROOT="$RUNS" \
  RUNROOT=/home/T0ng/DCCA-GS \
  nohup bash scripts/runner_phg_cell.sh \
    "$gpu" "$scene" "$data" 0.004 "$tag" 30000 15000 "$@" \
    > "$RUNS/$tag.launcher.log" 2>&1 &
  echo "$tag pid=$!"
}

SPA_ARGS=(--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.seed 2026)
MINI_ARGS=(
  --cfg.model.mini-splat-enabled
  --cfg.model.mini-splat-reinit-iter 15000
  --cfg.model.mini-splat-max-new 4000
  --cfg.model.mini-splat-views 16
  --cfg.model.mini-splat-voxel 0.0
  --cfg.seed 2026
)

wait_batch() {
  local files=("$@")
  tail -n0 -F "${files[@]}" | grep -m "${#files[@]}" "ALL_DONE tag=" >/dev/null
}

# Wait until wave-1 P0 cells and wave-2 Mip360 cells are finished, so GPUs 2-6 are free.
while [ ! -f "$RUNS/lyh_wave2_done" ]; do
  sleep 60
done

echo "SEED_BATCH1"
launch 2 playroom /home/data/deep_blending/db/playroom lyh_seed2026_playroom_cell1 "${SPA_ARGS[@]}"
launch 3 playroom /home/data/deep_blending/db/playroom lyh_seed2026_playroom_cell2 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 4 drjohnson /home/data/deep_blending/db/drjohnson lyh_seed2026_drjohnson_cell1 "${SPA_ARGS[@]}"
launch 5 tandt_train /home/data/deep_blending/tandt/train lyh_seed2026_tandt_train_cell1 "${SPA_ARGS[@]}"
launch 6 tandt_train /home/data/deep_blending/tandt/train lyh_seed2026_tandt_train_cell2 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 7 tandt_truck /home/data/deep_blending/tandt/truck lyh_seed2026_tandt_truck_cell1 "${SPA_ARGS[@]}"

wait_batch \
  "$RUNS/lyh_seed2026_playroom_cell1.launcher.log" \
  "$RUNS/lyh_seed2026_playroom_cell2.launcher.log" \
  "$RUNS/lyh_seed2026_drjohnson_cell1.launcher.log" \
  "$RUNS/lyh_seed2026_tandt_train_cell1.launcher.log" \
  "$RUNS/lyh_seed2026_tandt_train_cell2.launcher.log" \
  "$RUNS/lyh_seed2026_tandt_truck_cell1.launcher.log"

echo "SEED_BATCH2"
launch 2 tandt_truck /home/data/deep_blending/tandt/truck lyh_seed2026_tandt_truck_cell2 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 3 mip_garden /home/data/mipnerf360/garden lyh_seed2026_mip_garden_cell1 "${SPA_ARGS[@]}"
launch 4 mip_garden /home/data/mipnerf360/garden lyh_seed2026_mip_garden_cell2 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 5 mip_bicycle /home/data/mipnerf360/bicycle lyh_seed2026_mip_bicycle_cell1 "${SPA_ARGS[@]}"
launch 6 mip_bicycle /home/data/mipnerf360/bicycle lyh_seed2026_mip_bicycle_cell2 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"
launch 7 mip_stump /home/data/mipnerf360/stump lyh_seed2026_mip_stump_cell1 "${SPA_ARGS[@]}"

wait_batch \
  "$RUNS/lyh_seed2026_tandt_truck_cell2.launcher.log" \
  "$RUNS/lyh_seed2026_mip_garden_cell1.launcher.log" \
  "$RUNS/lyh_seed2026_mip_garden_cell2.launcher.log" \
  "$RUNS/lyh_seed2026_mip_bicycle_cell1.launcher.log" \
  "$RUNS/lyh_seed2026_mip_bicycle_cell2.launcher.log" \
  "$RUNS/lyh_seed2026_mip_stump_cell1.launcher.log"

echo "SEED_BATCH3"
launch 2 mip_stump /home/data/mipnerf360/stump lyh_seed2026_mip_stump_cell2 "${SPA_ARGS[@]}" "${MINI_ARGS[@]}"

echo "SEED_ALL_DONE" > "$RUNS/lyh_seed_all_done"
