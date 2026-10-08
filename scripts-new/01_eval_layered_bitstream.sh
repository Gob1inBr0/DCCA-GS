#!/usr/bin/env bash
# Evaluate the real progressive layered bitstream for an existing DCCA-GS run.
#
# Usage:
#   bash scripts-new/01_eval_layered_bitstream.sh <run_dir> <data_dir> [out_json] [out_bin] [extra args...]
#
# Environment:
#   RUNROOT            repository root, default: current working directory
#   CONSTRICTION_ROOT  optional path prepended to PYTHONPATH for constriction
#   DCCA_ZC_CONTEXT    default entropy path: density-root (set off for legacy)
#   DCCA_ZC_CELL_SIZE  default 0 = bbox_diag/512, decoder-recomputable
#   CUDA_VISIBLE_DEVICES may be set by caller

set -euo pipefail

RUN_DIR="${1:?run dir required}"
DATA_DIR="${2:?data dir required}"
OUT_JSON="${3:-$RUN_DIR/layered_bitstream.json}"
OUT_BIN="${4:-$RUN_DIR/layered_bitstream.bin}"
shift $(( $# >= 4 ? 4 : $# ))
EXTRA_ARGS=("$@")

has_zc_arg=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--zc-context" ]]; then
    has_zc_arg=1
    break
  fi
done
if [[ "$has_zc_arg" == "0" ]]; then
  EXTRA_ARGS+=(
    --zc-context "${DCCA_ZC_CONTEXT:-density-root}"
    --zc-cell-size "${DCCA_ZC_CELL_SIZE:-0}"
  )
fi

RUNROOT="${RUNROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${CONSTRICTION_ROOT:+$CONSTRICTION_ROOT:}$RUNROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_BIN")"

echo "[DCCA] layered eval run=$RUN_DIR"
echo "[DCCA] out_json=$OUT_JSON"
echo "[DCCA] out_bin=$OUT_BIN"

cd "$RUNROOT"
python scripts-new/05_encode_eval_layered_bitstream.py \
  --run "$RUN_DIR" \
  --data-dir "$DATA_DIR" \
  --field-aware \
  --data-factor 1 \
  --max-width 1600 \
  --out "$OUT_JSON" \
  --bin "$OUT_BIN" \
  "${EXTRA_ARGS[@]}"
