#!/usr/bin/env bash
# Wait for p0r s104 arms to finish (runner writes decoded_eval/metrics.jsonl
# after train -> compress -> decoded-eval), then run the 150-view layered
# curve eval for each arm in parallel on its own GPU.
# tmc3 must resolve to the env bin copy (guide §4: the system /usr/local/bin
# one is a dangling symlink into the sealed disk) — hence PATH export below.
# Usage: setsid nohup bash 14_p0r_s104_collect.sh <tag...> >> collect.log 2>&1 < /dev/null &
set -euo pipefail

PYBIN=/home/project2/miniconda3/envs/DCCA/bin
export PATH="$PYBIN:$PATH"
RUNS_ROOT=/home/project2/dcca_runs
CODE=/home/project2/DCCA-GS
DATA=/dev/shm/octree_178

TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
  TAGS=(p0r_base_l0004_s104 p0r_pen_l0004_w005_s104 p0r_p0r_l0004_w005_s104)
fi
GPUS=(3 5 6)

echo "[collect] waiting for: ${TAGS[*]} $(date)"
for i in $(seq 1 110); do   # up to ~9h, poll every 5 min
  all=1
  for t in "${TAGS[@]}"; do
    [[ -s "$RUNS_ROOT/$t/decoded_eval/metrics.jsonl" ]] || all=0
  done
  [[ "$all" = 1 ]] && break
  sleep 300
done

for idx in "${!TAGS[@]}"; do
  t="${TAGS[$idx]}"
  if [[ ! -s "$RUNS_ROOT/$t/decoded_eval/metrics.jsonl" ]]; then
    echo "[skip] $t not finished in time $(date)"
    continue
  fi
  gpu="${GPUS[$idx]}"
  echo "[curve] $t on gpu$gpu $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=/home/T0ng/gsplat-main:$CODE \
    "$PYBIN/python" "$CODE/scripts/full_view_eval.py" \
    --run "$RUNS_ROOT/$t" --data-dir "$DATA" \
    --data-factor 1 --max-width 1600 \
    --out "$RUNS_ROOT/$t/layered_curve_150v.json" \
    >> "$RUNS_ROOT/$t.layered.log" 2>&1 &
done
wait
echo "ALL_CURVES_DONE $(date)"
