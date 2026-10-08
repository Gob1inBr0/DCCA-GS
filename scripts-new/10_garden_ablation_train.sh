#!/usr/bin/env bash
# One garden ablation training arm on the LYH A100 host.
# Usage: bash scripts-new/10_garden_ablation_train.sh <gpu> <rate|p0>
#
# Replicates the std_garden_l0002_s42 baseline protocol exactly (spa r0.85,
# mini-splat, sensitivity-start-iter 0, data-factor 2, max-width 3200, 30k,
# seed 42, lambda 0.002) and adds one ablation switch per arm:
#   rate -> spa_rate_aware + spa_bit_budget + sens_per_bit
#   p0   -> P0 dequantized rendering loss (w=0.05, start 24000, every 8)
#   full -> v1 full config: rate-aware + B3 ladder align + P0 (the pre-
#           registered zccombo; ZC is bitstream-side and evaluated separately)
set -euo pipefail

GPU="${1:?gpu id required}"
ARM="${2:?arm required: rate|p0|full}"

RUNROOT="${RUNROOT:-/home/project2/DCCA-GS}"
RUNS_ROOT="${RUNS_ROOT:-/home/project2/dcca_runs}"
DATA="${DATA:-/home/project2/data/garden}"
PYBIN="${PYBIN:-/home/project2/miniconda3/envs/DCCA/bin}"

BASE_FLAGS=(
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85
  --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000
  --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8
  --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42
  --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85
  --cfg.data.data-factor 2 --cfg.data.max-width 3200
)

case "$ARM" in
  rate)
    TAG=std_garden_rate_l0002_s42
    ARM_FLAGS=(--cfg.model.spa-rate-aware --cfg.model.spa-rate-tau 1.0
      --cfg.model.spa-bit-budget
      --cfg.model.sensitivity-target-mode sens_per_bit)
    ;;
  p0)
    TAG=std_garden_p0_l0002_s42
    ARM_FLAGS=(--cfg.model.p0-render-loss --cfg.model.p0-render-weight 0.05
      --cfg.model.p0-render-start-iter 24000 --cfg.model.p0-render-interval 8)
    ;;
  full)
    TAG=std_garden_full_l0002_s42
    ARM_FLAGS=(--cfg.model.spa-rate-aware --cfg.model.spa-rate-tau 1.0
      --cfg.model.spa-bit-budget
      --cfg.model.sensitivity-target-mode sens_per_bit
      --cfg.model.coarse-ladder-align --cfg.model.coarse-ladder-start-iter 24000
      --cfg.model.coarse-ladder-weight 0.01
      --cfg.model.p0-render-loss --cfg.model.p0-render-weight 0.05
      --cfg.model.p0-render-start-iter 24000 --cfg.model.p0-render-interval 8)
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac

exec env RUNROOT="$RUNROOT" RUNS_ROOT="$RUNS_ROOT" \
  CONDA_ENV_BIN="$PYBIN" WAIT_VRAM_MB="${WAIT_VRAM_MB:-2000}" \
  bash "$RUNROOT/scripts-new/06_internal_runner_phg_cell.sh" \
  "$GPU" garden "$DATA" 0.002 "$TAG" 30000 15000 \
  "${BASE_FLAGS[@]}" "${ARM_FLAGS[@]}"
