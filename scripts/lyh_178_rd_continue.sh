#!/usr/bin/env bash
# Continue 1-78 RD after the first wave. The first-wave jobs are already
# running (HAC++ high/low + DCCA no-SPA high). This script waits for them,
# then launches the remaining no-SPA rate points on the same GPUs.

set -u

ulimit -n 65536
export PATH="/home/project2/tmc13/build/tmc3:${PATH}"

ROOT=/dev/shm/dcca_runs/1-78
DATA=/dev/shm/dcca_data/1-78/data
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin

wait_pid_file() {
  local name=$1
  local file="$ROOT/${name}.pid"
  [ -f "$file" ] || { echo "NO_PID_FILE $name"; return 0; }
  local pid
  pid=$(cat "$file")
  echo "WAIT_PID $name pid=$pid $(date)"
  tail --pid="$pid" -f /dev/null || true
  echo "PID_EXIT $name pid=$pid $(date)"
}

run_dcca() {
  local gpu=$1 lambda=$2 tag=$3
  local log="$ROOT/${tag}.log"
  echo "START_DCCA tag=$tag gpu=$gpu lambda=$lambda $(date)" > "$log"
  RUNS_ROOT="$ROOT" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=30000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lambda" "$tag" 110000 45000 \
    --cfg.model.no-spa-enabled \
    --cfg.model.no-mini-splat-enabled \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  echo "DCCA_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda"
}

mkdir -p "$ROOT"

# First-wave jobs should already be present. This script only waits for the
# two HAC++ runs, then immediately starts two no-SPA points on GPUs 3/6. The
# third no-SPA point starts on GPU7 after the first no-SPA run finishes.
(
  wait_pid_file hacpp_1-78_high_l0004
  wait_pid_file hacpp_1-78_low_l0005
  echo "SECOND_WAVE_START $(date)"
  run_dcca 3 0.002 dcca_1-78_nospa_l0002
  run_dcca 6 0.0005 dcca_1-78_nospa_l0005
  wait_pid_file dcca_1-78_nospa_l0002
  wait_pid_file dcca_1-78_nospa_l0005
) &
wave2=$!

(
  wait_pid_file dcca_1-78_nospa_l0004
  echo "THIRD_WAVE_START $(date)"
  run_dcca 7 0.001 dcca_1-78_nospa_l0001
  wait_pid_file dcca_1-78_nospa_l0001
) &
wave3=$!

wait "$wave2" "$wave3"

echo "RD_CONTINUE_DONE $(date)" | tee "$ROOT/rd_continue_done.txt"
