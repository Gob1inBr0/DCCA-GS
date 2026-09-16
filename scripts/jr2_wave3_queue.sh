#!/bin/bash
# jr2 wave3 (GPU jobs, waits for wave2's eight runs to finish):
#   1. r2_I6off_new_r085_lam0002_s42  — SOG-style control, I6 supervision OFF
#     (vs r2_base_newcode which is I6 ON; isolates "training shapes
#      compressibility" from "different final Q")
#   2. r2_B1_postw5000_r085_lam0002_s42 — post_window 5000
#   3. r2_B1_postw10000_r085_lam0002_s42 — post_window 10000
#     (reference post_window 2000 = r2_base_newcode; pattern-6 B1 sweep)
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-git
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/jr2_wave3.log"
export PATH="/home/project2/tmc13:$PATH"

# Gate: all eight wave2 runs must show ALL_DONE before we take GPUs.
WAVE2_TAGS="ph1v3_A2_cover_r085_lam0002_s104r ph1v3_A2_cover_r085_lam0005_s104 r2_fusA_new_r085_lam0004_s42 r2_fisher_fusA_r085_lam0004_s42 r2_gate_B0_new_r055_lam0002_s42 r2_gate_B1_holdout_r055_lam0002_s42 r2_w010_fusA_r085_lam0004_s42 r2_w020_fusA_r085_lam0004_s42"
while true; do
  n=0
  for t in $WAVE2_TAGS; do
    grep -qa "ALL_DONE" "$ROOT/$t.log" 2>/dev/null && n=$((n + 1))
  done
  echo "WAITING wave2 done=$n/8 $(date)" >> "$QLOG"
  [ "$n" -ge 8 ] && break
  sleep 600
done
echo "GATE_OPEN wave2 complete $(date)" >> "$QLOG"

S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42"
B="--cfg.model.spa-post-ratio 0.85"

launch() {
  local g=$1 tag=$2 extra=$3
  echo "$tag" > "$ROOT/.qlock_$g"
  echo "ARM $tag gpu=$g $(date)" >> "$QLOG"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$g" 1-78 "$DATA" 0.002 "$tag" 30000 15000 $Z $B $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
}

JOBS=(
  "r2_I6off_new_r085_lam0002_s42|--cfg.model.sensitivity-weight 0.0"
  "r2_B1_postw5000_r085_lam0002_s42|--cfg.model.spa-post-window 5000"
  "r2_B1_postw10000_r085_lam0002_s42|--cfg.model.spa-post-window 10000"
)
declare -A DONE
WHITELIST="0 2 4 5 7"
while true; do
  all=1
  for job in "${JOBS[@]}"; do
    IFS='|' read -r tag extra <<< "$job"
    [ "${DONE[$tag]:-0}" = "1" ] && continue
    all=0
    g=""
    for cand in $WHITELIST; do
      lock="$ROOT/.qlock_$cand"
      if [ -f "$lock" ] && grep -qa "ALL_DONE" "$ROOT/$(cat "$lock").log" 2>/dev/null; then
        rm -f "$lock"
      fi
      [ -f "$lock" ] && continue
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$cand" 2>/dev/null || echo 99999)
      if [ "${used:-99999}" -le 4500 ]; then g=$cand; break; fi
    done
    [ -z "$g" ] && continue
    launch "$g" "$tag" "$extra"
    DONE[$tag]=1
    sleep 90
  done
  [ "$all" = "1" ] && break
  sleep 300
done
echo "JR2_WAVE3_ALL_LAUNCHED $(date)" >> "$QLOG"
