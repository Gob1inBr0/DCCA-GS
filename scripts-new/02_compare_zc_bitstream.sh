#!/usr/bin/env bash
# Build both legacy and UAV-DCCA-ZC layered streams for one trained run, then
# write a compact byte/PSNR comparison report.
#
# Usage:
#   bash scripts-new/02_compare_zc_bitstream.sh <run_dir> <data_dir> [out_dir] [tag]
#
# Example:
#   bash scripts-new/02_compare_zc_bitstream.sh \
#     /runs/dcca_178_r060_lam0005 /data/PKUGS/1-78 \
#     analysis/s2_prefix_sweep dcca_178_r060

set -euo pipefail

RUN_DIR="${1:?run dir required}"
DATA_DIR="${2:?data dir required}"
OUT_DIR="${3:-$RUN_DIR/analysis/s2_prefix_sweep}"
TAG="${4:-$(basename "$RUN_DIR")}"

RUNROOT="${RUNROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$OUT_DIR"

LEGACY_JSON="$OUT_DIR/${TAG}_legacy.json"
LEGACY_BIN="$OUT_DIR/${TAG}_legacy.bin"
ZC_JSON="$OUT_DIR/${TAG}_zc.json"
ZC_BIN="$OUT_DIR/${TAG}_zc.bin"
REPORT="$OUT_DIR/${TAG}_zc_compare.txt"

echo "[DCCA] legacy layered stream -> $LEGACY_JSON"
DCCA_ZC_CONTEXT=off bash "$RUNROOT/scripts-new/01_eval_layered_bitstream.sh" \
  "$RUN_DIR" "$DATA_DIR" "$LEGACY_JSON" "$LEGACY_BIN"

echo "[DCCA] ZCausal layered stream -> $ZC_JSON"
DCCA_ZC_CONTEXT="${DCCA_ZC_CONTEXT:-density-root}" \
DCCA_ZC_CELL_SIZE="${DCCA_ZC_CELL_SIZE:-0}" \
  bash "$RUNROOT/scripts-new/01_eval_layered_bitstream.sh" \
  "$RUN_DIR" "$DATA_DIR" "$ZC_JSON" "$ZC_BIN"

echo "[DCCA] compare legacy vs ZCausal -> $REPORT"
python "$RUNROOT/scripts-new/04_compare_zc_reports.py" \
  --base "$LEGACY_JSON" \
  --test "$ZC_JSON" \
  --out "$REPORT"

echo "[DCCA] done: $REPORT"
