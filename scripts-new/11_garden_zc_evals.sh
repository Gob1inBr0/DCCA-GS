#!/usr/bin/env bash
# Bitstream-side experiments for the garden RD figure and ZCausal ablation.
# All arms reuse EXISTING trained runs; no training happens here.
#   E1  regenerate the missing 150-view layered JSON for std_garden_l0001_s42
#   E2  ZCausal ablation on std_garden_l0002_s42: legacy (off) vs density-root,
#       same code, same quantizers -> decoded symbols identical by
#       construction, so any byte difference is pure entropy-model effect.
# Run in the foreground under nohup; needs one free GPU.
set -euo pipefail

GPU="${1:?gpu id required}"
RUNROOT="${RUNROOT:-/home/project2/DCCA-GS}"
RUNS_ROOT="${RUNS_ROOT:-/home/project2/dcca_runs}"
DATA="${DATA:-/home/project2/data/garden}"
PYBIN="${PYBIN:-/home/project2/miniconda3/envs/DCCA/bin}"
C25LIB="${C25LIB:-/home/project2/c25_pylib2}"
LOGDIR="${LOGDIR:-$RUNS_ROOT/garden_ablation_launch}"
mkdir -p "$LOGDIR"
export PATH="$PYBIN:$PATH"   # tmc3 (G-PCC) lives in the DCCA env bin
export CUDA_VISIBLE_DEVICES="$GPU"
EVAL_FLAGS=(--data-dir "$DATA" --data-factor 2 --max-width 3200)

echo "[E1] $(date '+%T') layered full-view JSON for std_garden_l0001_s42"
"$PYBIN/python" "$RUNROOT/scripts/full_view_eval.py" \
  --run "$RUNS_ROOT/std_garden_l0001_s42" "${EVAL_FLAGS[@]}" \
  --out "$RUNS_ROOT/std_garden_l0001_s42/layered_curve_150v.json" \
  > "$LOGDIR/e1_l0001_fullview.log" 2>&1

echo "[E2] $(date '+%T') ZCausal ablation on std_garden_l0002_s42"
L2="$RUNS_ROOT/std_garden_l0002_s42"
for CTX in off density-root; do
  echo "[E2] zc-context=$CTX $(date '+%T')"
  PYTHONPATH="$C25LIB" "$PYBIN/python" "$RUNROOT/scripts/c25_real_bitstream.py" \
    --run "$L2" "${EVAL_FLAGS[@]}" --field-aware \
    --mapped "$L2/std_garden_l0002_s42.mapped.npy" \
    --zc-context "$CTX" --zc-cell-size 0 \
    --out "$L2/zc_ablation_${CTX}.json" \
    --bin "$L2/zc_ablation_${CTX}.bin" \
    > "$LOGDIR/e2_zc_${CTX}.log" 2>&1
done

"$PYBIN/python" "$RUNROOT/scripts-new/04_compare_zc_reports.py" \
  --base "$L2/zc_ablation_off.json" \
  --test "$L2/zc_ablation_density-root.json" \
  --out "$L2/zc_ablation_compare.txt"

echo "[done] $(date '+%T') results in $L2/zc_ablation_*.json and zc_ablation_compare.txt"
