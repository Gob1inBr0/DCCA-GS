#!/usr/bin/env bash
# One-command DCCA-GS train -> compress -> decoded-eval run.
#
# Usage:
#   bash scripts-new/00_train_dcca.sh <gpu> <scene> <data_dir> <tag> \
#     [lambda] [spa_ratio] [seed] [steps] [update_until]
#
# Environment:
#   RUNROOT        repository root, default: current working directory
#   RUNS_ROOT      output root, default: <RUNROOT>/runs
#   CONDA_ENV_BIN  conda env bin directory containing HAC++ extensions
#   GSPLAT_ROOT    local gsplat checkout
#   WAIT_VRAM_MB   max used VRAM before launch, default inherited by runner
#   DCCA_RATE_AWARE=1 enables spa_rate_aware + spa_bit_budget
#   DCCA_P0_RENDER=1 enables direct P0 dequantized rendering loss

set -euo pipefail

GPU="${1:?gpu id required}"
SCENE="${2:?scene name required}"
DATA_DIR="${3:?data dir required}"
TAG="${4:?run tag required}"
LAMBDA="${5:-0.0005}"
SPA_RATIO="${6:-0.85}"
SEED="${7:-42}"
MAX_STEPS="${8:-30000}"
UPDATE_UNTIL="${9:-15000}"

RUNROOT="${RUNROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$RUNROOT/runs}"
export RUNROOT RUNS_ROOT

EXTRA_FLAGS=(
  --cfg.model.spa-enabled
  --cfg.model.spa-ratio "$SPA_RATIO"
  --cfg.model.spa-post-ratio "$SPA_RATIO"
  --cfg.model.mini-splat-enabled
  --cfg.model.mini-splat-reinit-iter 15000
  --cfg.model.mini-splat-max-new 4000
  --cfg.model.mini-splat-views 8
  --cfg.model.mini-splat-voxel 0.0
  --cfg.seed "$SEED"
)

if [[ "${DCCA_B3:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(
    --cfg.model.coarse-ladder-align
    --cfg.model.coarse-ladder-start-iter "${DCCA_B3_START:-24000}"
    --cfg.model.coarse-ladder-weight "${DCCA_B3_WEIGHT:-0.01}"
  )
fi

if [[ "${DCCA_RATE_AWARE:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(
    --cfg.model.spa-rate-aware
    --cfg.model.spa-rate-tau "${DCCA_RATE_TAU:-1.0}"
    --cfg.model.spa-bit-budget
    --cfg.model.sensitivity-target-mode sens_per_bit
  )
fi

if [[ "${DCCA_P0_RENDER:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(
    --cfg.model.p0-render-loss
    --cfg.model.p0-render-weight "${DCCA_P0_WEIGHT:-0.05}"
    --cfg.model.p0-render-start-iter "${DCCA_P0_START:-24000}"
    --cfg.model.p0-render-interval "${DCCA_P0_INTERVAL:-8}"
  )
fi

echo "[DCCA] runroot=$RUNROOT"
echo "[DCCA] runs_root=$RUNS_ROOT"
echo "[DCCA] scene=$SCENE tag=$TAG lambda=$LAMBDA spa_ratio=$SPA_RATIO seed=$SEED"

bash "$RUNROOT/scripts-new/06_internal_runner_phg_cell.sh" \
  "$GPU" "$SCENE" "$DATA_DIR" "$LAMBDA" "$TAG" "$MAX_STEPS" "$UPDATE_UNTIL" \
  "${EXTRA_FLAGS[@]}"
