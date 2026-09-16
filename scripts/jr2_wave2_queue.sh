#!/bin/bash
# jr2 wave2 + G1 completion: serial scheduler for 8 remaining 30k runs.
# Launches each job onto any whitelist GPU with <=4500MB used; a launched
# job holds its GPU until the runner log shows ALL_DONE (then the lock is
# ignored and the GPU is reusable).
# Priority order: G1 completion (2) -> fusA new-code baseline -> fisher
# -> gate B0/B1 -> weights w010/w020.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-git
export PATH="/home/project2/tmc13:$PATH"
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/jr2_wave2.log"
WHITELIST="0 2 4 5 7"
FREE_MB=4500
S0="--cfg.model.sensitivity-start-iter 0"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"
R055="--cfg.model.spa-post-ratio 0.55"

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

F2="FUSION_PLACEHOLDER"
jobs=(
  "ph1v3_A2_cover_r085_lam0002_s104r|0.002|104|$FUSION --cfg.model.submodular-mode cover"
  "ph1v3_A2_cover_r085_lam0005_s104|0.0005|104|$FUSION --cfg.model.submodular-mode cover"
  "r2_fusA_new_r085_lam0004_s42|0.004|42|$FUSION"
  "r2_fisher_fusA_r085_lam0004_s42|0.004|42|$FUSION --cfg.model.sensitivity-second-order --cfg.model.sensitivity-use-fisher"
  "r2_gate_B0_new_r055_lam0002_s42|0.002|42|$R055"
  "r2_gate_B1_holdout_r055_lam0002_s42|0.002|42|$R055 --cfg.model.spa-holdout-gate"
  "r2_w010_fusA_r085_lam0004_s42|0.004|42|$FUSION --cfg.model.fusion-sensitivity-weight 0.10"
  "r2_w020_fusA_r085_lam0004_s42|0.004|42|$FUSION --cfg.model.fusion-sensitivity-weight 0.20"
)

declare -A DONE
# Pre-seed locks for wave1 arms (launched outside this scheduler): the lock
# expires automatically once their runner logs print ALL_DONE.
for pair in "0:r2_base_newcode_r085_lam0002_s42" "4:r2_R1_rate_r085_lam0002_s42" "5:r2_R2_bitbud_r085_lam0002_s42" "7:r2_SPB_base_r085_lam0002_s42"; do
  g=${pair%%:*}; tag=${pair##*:}
  [ -f "$ROOT/.qlock_$g" ] || echo "$tag" > "$ROOT/.qlock_$g"
done

launched_any=1
while true; do
  all_done=1
  launched_any=0
  for job in "${jobs[@]}"; do
    IFS='|' read -r tag lam seed extra <<< "$job"
    [ "${DONE[$tag]:-0}" = "1" ] && continue
    all_done=0
    g=$(pick_gpu) || continue
    extra=${extra//FUSION_PLACEHOLDER/$FUSION}
    launch "$g" "$tag" "$lam" "$seed" "$extra"
    DONE[$tag]=1
    launched_any=1
    sleep 90   # let memory ramp before re-scanning
  done
  [ "$all_done" = "1" ] && break
  sleep 300
done
echo "JR2_WAVE2_ALL_LAUNCHED $(date)" >> "$QLOG"
