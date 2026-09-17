#!/bin/bash
# jr2 wave4: positional-negative-gain discriminative/fix arms (task书 2026-09-17).
# Same lock protocol as jr2_wave2_queue.sh; whitelist is 0/4/5/7 only (GPU 2
# reserved for an external user via the permanent hold_gpu2_external lock),
# and a global gate keeps OUR concurrent unfinished arms at <= 4.
#   a) r2_l0005_relax_r085_lam0005_s42 — ph0c_base_r085_lam0005_s42 config with
#      ONLY --cfg.model.spa-post-ratio 1.0 (no fusion-prune block, S0 kept).
#      PRE-REGISTERED GATE (H1/B fix): >=27.80 dB @ <=21.4 MB.
#   b) r2_A3_delayed_r085_lam0002_s42 — ph1v3_A3_coversens_r085_lam0002_s42
#      config (FUSION + cover_sens) with ONLY sensitivity-start-iter 20000.
#      PRE-REGISTERED GATE (H2): >=27.60 dB.
# Numbers source of truth: docs/data/experiments.csv.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-git
export PATH="/home/project2/tmc13:$PATH"
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/jr2_wave4.log"
WHITELIST="0 4 5 7"
FREE_MB=4500
MAX_OUR_ARMS=4
S0="--cfg.model.sensitivity-start-iter 0"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"
R100="--cfg.model.spa-post-ratio 1.0"

# Count our arms that still hold a GPU (lock present, runner log has no
# ALL_DONE yet). The GPU-2 hold tag is excluded: it is a placeholder, not a
# job, so it must not consume the concurrency budget.
our_busy_arms() {
  local n=0 f tag
  for f in "$ROOT"/.qlock_*; do
    [ -f "$f" ] || continue
    case "$f" in *".qlock_2") continue ;; esac
    tag=$(cat "$f" 2>/dev/null) || continue
    grep -qa "ALL_DONE" "$ROOT/${tag}.log" 2>/dev/null || n=$((n + 1))
  done
  echo "$n"
}

gpu_busy() {
  local g=$1
  local lock="$ROOT/.qlock_$g"
  [ -f "$lock" ] || return 1
  local tag
  tag=$(cat "$lock")
  if grep -qa "ALL_DONE" "$ROOT/${tag}.log" 2>/dev/null; then
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
)

declare -A DONE
launched_any=1
while true; do
  all_done=1
  launched_any=0
  for job in "${jobs[@]}"; do
    IFS='|' read -r tag lam seed extra <<< "$job"
    [ "${DONE[$tag]:-0}" = "1" ] && continue
    all_done=0
    # Global concurrency cap for our arms (incl. wave2 leftovers): only
    # launch when fewer than MAX_OUR_ARMS of ours are still holding GPUs.
    if [ "$(our_busy_arms)" -ge "$MAX_OUR_ARMS" ]; then
      break
    fi
    g=$(pick_gpu) || continue
    launch "$g" "$tag" "$lam" "$seed" "$extra"
    DONE[$tag]=1
    launched_any=1
    sleep 90   # let memory ramp before re-scanning
  done
  [ "$all_done" = "1" ] && break
  sleep 300
done
echo "JR2_WAVE4_ALL_LAUNCHED $(date)" >> "$QLOG"
