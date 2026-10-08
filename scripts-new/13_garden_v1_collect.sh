#!/usr/bin/env bash
# Wait for the two v1 ablation training arms to finish (the runner does
# train -> compress -> decoded-eval on its own), then produce the same
# full-view layered curves as the baseline runs, so the final RD figure
# compares v1 arms against the three reference curves under one protocol.
set -euo pipefail

GPU="${1:?gpu id required}"
RUNROOT="${RUNROOT:-/home/project2/DCCA-GS}"
RUNS_ROOT="${RUNS_ROOT:-/home/project2/dcca_runs}"
DATA="${DATA:-/home/project2/data/garden}"
PYBIN="${PYBIN:-/home/project2/miniconda3/envs/DCCA/bin}"
LOGDIR="${LOGDIR:-$RUNS_ROOT/garden_ablation_launch}"
mkdir -p "$LOGDIR"
export PATH="$PYBIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"

ARMS=(std_garden_rate_l0002_s42 std_garden_p0_l0002_s42)

for TAG in "${ARMS[@]}"; do
  R="$RUNS_ROOT/$TAG"
  echo "[wait] $TAG $(date '+%T')"
  for i in $(seq 1 96); do   # up to 8h, check every 5 min
    # metrics.jsonl is written only after train -> compress -> decoded-eval
    if [[ -s "$R/decoded_eval/metrics.jsonl" && -f "$R/bitstreams/codec_header.json" ]]; then
      break
    fi
    sleep 300
  done
  if [[ ! -s "$R/decoded_eval/metrics.jsonl" ]]; then
    echo "[skip] $TAG not finished in time" >&2
    continue
  fi
  echo "[E4] $TAG full-view layered curve $(date '+%T')"
  "$PYBIN/python" "$RUNROOT/scripts/full_view_eval.py" \
    --run "$R" --data-dir "$DATA" --data-factor 2 --max-width 3200 \
    --out "$R/layered_curve_150v.json" \
    > "$LOGDIR/e4_${TAG}.log" 2>&1
  tail -1 "$LOGDIR/e4_${TAG}.log" || true
done
echo "[done] $(date '+%T') v1 curves ready"
