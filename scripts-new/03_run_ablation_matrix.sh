#!/usr/bin/env bash
# Launch a compact DCCA-GS experiment matrix for low-rate UAV scenes.
#
# Usage:
#   bash scripts-new/03_run_ablation_matrix.sh <gpu_csv> <scene> <data_dir> <tag_prefix> [lambda]
#
# Example:
#   bash scripts-new/03_run_ablation_matrix.sh 0,1,2,3 1-78 /data/PKUGS/1-78 dcca_178 0.0005

set -euo pipefail

GPU_CSV="${1:?comma-separated gpu list required}"
SCENE="${2:?scene required}"
DATA_DIR="${3:?data dir required}"
TAG_PREFIX="${4:?tag prefix required}"
LAMBDA="${5:-0.0005}"

RUNROOT="${RUNROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RUNROOT

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "No GPU ids supplied" >&2
  exit 1
fi

launch_budget() {
  local idx="$1"
  local ratio="$2"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local tag="${TAG_PREFIX}_r${ratio/./}_lam${LAMBDA/./}"
  echo "[DCCA] launch budget ratio=$ratio gpu=$gpu tag=$tag"
  bash "$RUNROOT/scripts-new/00_train_dcca.sh" \
    "$gpu" "$SCENE" "$DATA_DIR" "$tag" "$LAMBDA" "$ratio" 42 30000 15000 &
}

launch_budget 0 0.85
launch_budget 1 0.70
launch_budget 2 0.60
launch_budget 3 0.50

if [[ "${DCCA_SKIP_RATE_AWARE:-0}" != "1" ]]; then
  gpu="${GPUS[$((4 % ${#GPUS[@]}))]}"
  tag="${TAG_PREFIX}_rate_r060_lam${LAMBDA/./}"
  echo "[DCCA] launch rate-aware SPA candidate gpu=$gpu tag=$tag"
  DCCA_RATE_AWARE=1 DCCA_RATE_TAU="${DCCA_RATE_TAU:-1.0}" \
    bash "$RUNROOT/scripts-new/00_train_dcca.sh" \
    "$gpu" "$SCENE" "$DATA_DIR" "$tag" "$LAMBDA" 0.60 42 30000 15000 &
fi

if [[ "${DCCA_SKIP_B3:-0}" != "1" ]]; then
  gpu="${GPUS[$((5 % ${#GPUS[@]}))]}"
  tag="${TAG_PREFIX}_b3_r060_lam${LAMBDA/./}"
  echo "[DCCA] launch B3 candidate gpu=$gpu tag=$tag"
  DCCA_B3=1 DCCA_B3_WEIGHT="${DCCA_B3_WEIGHT:-0.01}" \
    bash "$RUNROOT/scripts-new/00_train_dcca.sh" \
    "$gpu" "$SCENE" "$DATA_DIR" "$tag" "$LAMBDA" 0.60 42 30000 15000 &
fi

if [[ "${DCCA_SKIP_P0:-0}" != "1" ]]; then
  gpu="${GPUS[$((6 % ${#GPUS[@]}))]}"
  tag="${TAG_PREFIX}_p0_r060_lam${LAMBDA/./}"
  echo "[DCCA] launch P0-render candidate gpu=$gpu tag=$tag"
  DCCA_P0_RENDER=1 \
  DCCA_P0_WEIGHT="${DCCA_P0_WEIGHT:-0.05}" \
  DCCA_P0_INTERVAL="${DCCA_P0_INTERVAL:-8}" \
    bash "$RUNROOT/scripts-new/00_train_dcca.sh" \
    "$gpu" "$SCENE" "$DATA_DIR" "$tag" "$LAMBDA" 0.60 42 30000 15000 &
fi

if [[ "${DCCA_SKIP_COMBO:-0}" != "1" ]]; then
  gpu="${GPUS[$((7 % ${#GPUS[@]}))]}"
  tag="${TAG_PREFIX}_zccombo_r060_lam${LAMBDA/./}"
  echo "[DCCA] launch rate-aware+B3+P0 candidate gpu=$gpu tag=$tag"
  DCCA_RATE_AWARE=1 DCCA_B3=1 DCCA_P0_RENDER=1 \
  DCCA_B3_WEIGHT="${DCCA_B3_WEIGHT:-0.01}" \
  DCCA_P0_WEIGHT="${DCCA_P0_WEIGHT:-0.05}" \
  DCCA_P0_INTERVAL="${DCCA_P0_INTERVAL:-8}" \
    bash "$RUNROOT/scripts-new/00_train_dcca.sh" \
    "$gpu" "$SCENE" "$DATA_DIR" "$tag" "$LAMBDA" 0.60 42 30000 15000 &
fi

wait
echo "[DCCA] matrix finished"
