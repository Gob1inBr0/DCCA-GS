#!/bin/bash
# Launch MiniSplat full-mode generalization cells on LYH.
# Usage: bash scripts/run_minisplat_full_matrix.sh
set -euo pipefail

cd /home/T0ng/DCCA-GS-minifull
RUNS=/home/T0ng/runs
CONDA_ENV_BIN=/home/T0ng/miniconda3/envs/DCCA/bin
export WAIT_VRAM_MB=${WAIT_VRAM_MB:-30000}
export RUNS_ROOT="$RUNS"
export RUNROOT=/home/T0ng/DCCA-GS-minifull
export CONDA_ENV_BIN

SPA_ARGS=(--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85)
MINI_ARGS=(
  --cfg.model.mini-splat-enabled
  --cfg.model.mini-splat-reinit-iter 15000
  --cfg.model.mini-splat-max-new 4000
  --cfg.model.mini-splat-views 8
  --cfg.model.mini-splat-voxel 0.0
  --cfg.model.mini-splat-full
)

launch() {
  local gpu=$1 scene=$2 data=$3 tag=$4
  nohup bash scripts/runner_phg_cell.sh \
    "$gpu" "$scene" "$data" 0.004 "$tag" 30000 15000 \
    "${SPA_ARGS[@]}" "${MINI_ARGS[@]}" \
    > "$RUNS/$tag.launcher.log" 2>&1 &
  echo "$tag pid=$! gpu=$gpu"
}

launch 0 drjohnson /mnt/003/dataset/tandt_db/db/drjohnson lyh_full_drjohnson
launch 1 tandt_train /mnt/003/dataset/tandt_db/tandt/train lyh_full_tandt_train
launch 2 tandt_truck /mnt/003/dataset/tandt_db/tandt/truck lyh_full_tandt_truck
launch 3 mip_garden /mnt/003/dataset/360_v2/garden lyh_full_mip_garden
launch 4 playroom /mnt/003/dataset/tandt_db/db/playroom lyh_full_playroom
launch 5 mip_bicycle /mnt/003/dataset/360_v2/bicycle lyh_full_mip_bicycle
launch 6 mip_stump /mnt/003/dataset/360_v2/stump lyh_full_mip_stump
