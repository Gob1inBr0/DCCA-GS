#!/bin/bash
# jr2_total_queue.sh — unified scheduler (2026-09-17 20:15), supersedes
# jr2_wave2_queue.sh (whose leftover launches were s104r-duplicate / w020)
# and jr2_wave4_queue.sh. Card discipline: GPU 2 permanently reserved via
# the hold_gpu2_external lock, and at most MAX_OUR_ARMS=4 of OUR unfinished
# arms hold GPUs at any time (global gate, wave2 leftovers included).
# Resume-safe: a job whose runner log already shows ALL_DONE, or whose tag
# currently holds a lock (in flight elsewhere), is marked done at scan time.
# Priority: wave4 discriminators -> weights stage-1 (w020 then killed w010)
# -> wave3 sweep (I6off control, post-window 5000/10000).
# Pre-registered gates (docs/03-reports/结果异常与位置性负增益_报告及排查任务书.md):
#   r2_l0005_relax  >=27.80 dB @ <=21.4 MB (H1/B fix)
#   r2_A3_delayed   >=27.60 dB             (H2)
# Numbers source of truth: docs/data/experiments.csv.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-git
export PATH="/home/project2/tmc13:$PATH"
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/jr2_total_queue.log"
WHITELIST="0 1 4 5 7"
FREE_MB=4500
MAX_OUR_ARMS=4
S0="--cfg.model.sensitivity-start-iter 0"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"
B="--cfg.model.spa-post-ratio 0.85"
R100="--cfg.model.spa-post-ratio 1.0"

# ALL_DONE lands in the RUNNER's stdout, i.e. ${tag}.launch.log (the
# scheduler redirects it there); the trainer log ${tag}.log never contains
# it. Check launch.log first, fall back to the trainer log for wave1-style
# launches that redirected differently.
arm_done() {
  grep -qa "ALL_DONE" "$ROOT/$1.launch.log" 2>/dev/null \
    || grep -qa "ALL_DONE" "$ROOT/$1.log" 2>/dev/null
}

gpu_busy() {
  local g=$1
  local lock="$ROOT/.qlock_$g"
  [ -f "$lock" ] || return 1
  local tag
  tag=$(cat "$lock")
  if arm_done "$tag"; then
    rm -f "$lock"; return 1
  fi
  return 0
}

pick_gpu() {
  for g in $WHITELIST; do
    gpu_busy "$g" && continue
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
    if [ "${used:-99999}" -le "$FREE_MB" ]; then
      echo "$g"; return 0
    fi
  done
  return 1
}

# Our arms still holding a GPU (lock present, runner not ALL_DONE yet).
# The GPU-2 hold tag is a placeholder, not a job: excluded from the count.
our_busy_arms() {
  local n=0 f tag
  for f in "$ROOT"/.qlock_*; do
    [ -f "$f" ] || continue
    case "$f" in *".qlock_2") continue ;; esac
    tag=$(cat "$f" 2>/dev/null) || continue
    arm_done "$tag" || n=$((n + 1))
  done
  echo "$n"
}

tag_in_any_lock() {
  local f
  for f in "$ROOT"/.qlock_*; do
    [ -f "$f" ] || continue
    [ "$(cat "$f" 2>/dev/null)" = "$1" ] && return 0
  done
  return 1
}

launch() {
  local g=$1 tag=$2 lam=$3 seed=$4 extra=$5
  echo "$tag" > "$ROOT/.qlock_$g"
  echo "ARM $tag gpu=$g lam=$lam seed=$seed $(date)" >> "$QLOG"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$g" 1-78 "$DATA" "$lam" "$tag" 30000 15000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled \
    --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 \
    --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed "$seed" $S0 \
    $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
}

jobs=(
  "r2_l0005_relax_r085_lam0005_s42|0.0005|42|$R100"
  "r2_A3_delayed_r085_lam0002_s42|0.002|42|$FUSION --cfg.model.submodular-mode cover_sens --cfg.model.sensitivity-start-iter 20000"
  "r2_w020_fusA_r085_lam0004_s42|0.004|42|$FUSION --cfg.model.fusion-sensitivity-weight 0.20"
  "r2_w010_fusA_r085_lam0004_s42|0.004|42|$FUSION --cfg.model.fusion-sensitivity-weight 0.10"
  "r2_I6off_new_r085_lam0002_s42|0.002|42|$B --cfg.model.sensitivity-weight 0.0"
  "r2_B1_postw5000_r085_lam0002_s42|0.002|42|$B --cfg.model.spa-post-window 5000"
  "r2_B1_postw10000_r085_lam0002_s42|0.002|42|$B --cfg.model.spa-post-window 10000"
)

declare -A DONE
while true; do
  all_done=1
  for job in "${jobs[@]}"; do
    IFS='|' read -r tag lam seed extra <<< "$job"
    [ "${DONE[$tag]:-0}" = "1" ] && continue
    all_done=0
    if arm_done "$tag"; then
      DONE[$tag]=1; echo "SKIP_ALREADY_DONE $tag $(date)" >> "$QLOG"; continue
    fi
    if tag_in_any_lock "$tag"; then
      DONE[$tag]=1; echo "SKIP_IN_FLIGHT $tag $(date)" >> "$QLOG"; continue
    fi
    if [ "$(our_busy_arms)" -ge "$MAX_OUR_ARMS" ]; then
      break
    fi
    g=$(pick_gpu) || continue
    launch "$g" "$tag" "$lam" "$seed" "$extra"
    DONE[$tag]=1
    sleep 90   # let memory ramp before re-scanning
  done
  [ "$all_done" = "1" ] && break
  sleep 300
done
echo "JR2_TOTAL_QUEUE_ALL_LAUNCHED $(date)" >> "$QLOG"
