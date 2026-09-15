#!/bin/bash
# journal-round2 wave 1 (post git-sync), four solo GPUs.
#   GPU 0: base rerun on the new code (double-slice fix baseline)
#   GPU 4: R1 rate discount      GPU 5: R2 bit budget      GPU 7: SPB sens-per-bit
# All @ r085 / lam 0.002 / s42 — the main-table point; gates vs the new base.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-git
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/jr2_wave1.log"
S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
B="--cfg.model.spa-post-ratio 0.85"

launch() {
  local g=$1 tag=$2 extra=$3
  echo "ARM $tag gpu=$g $(date)" >> "$QLOG"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$g" 1-78 "$DATA" 0.002 "$tag" 30000 15000 $Z $B $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
}
launch 0 r2_base_newcode_r085_lam0002_s42 ""
launch 4 r2_R1_rate_r085_lam0002_s42  "--cfg.model.spa-rate-aware --cfg.model.spa-rate-tau 1.0"
launch 5 r2_R2_bitbud_r085_lam0002_s42 "--cfg.model.spa-bit-budget"
launch 7 r2_SPB_base_r085_lam0002_s42  "--cfg.model.sensitivity-target-mode sens_per_bit"
echo "JR2_WAVE1_LAUNCHED $(date)" >> "$QLOG"

# engagement watchers (audit lesson: a mechanism that silently fails must be caught)
watch() {
  local tag=$1 pattern=$2 deadline=$3
  local log="$ROOT/${tag}.log"
  local tries=0
  while [ "$tries" -lt 120 ]; do
    if grep -qa "$pattern" "$log" 2>/dev/null; then
      echo "ENGAGED tag=$tag pattern='$pattern' $(date)" >> "$QLOG"; return 0
    fi
    if grep -qaE "Training finished|GAVE_UP_TRAIN|Traceback" "$log" 2>/dev/null; then
      echo "ENGAGEMENT_FAIL tag=$tag (ended before match)" >> "$QLOG"; return 1
    fi
    local step
    step=$(grep -aoE 'step=[0-9]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "$step" ] && [ "$step" -gt "$deadline" ]; then
      echo "ENGAGEMENT_FAIL tag=$tag pattern='$pattern' at step=$step" >> "$QLOG"; return 1
    fi
    tries=$((tries + 1)); sleep 300
  done
  echo "ENGAGEMENT_FAIL tag=$tag: watcher cap" >> "$QLOG"
}
watch r2_R1_rate_r085_lam0002_s42  "bits_med=" 11000
watch r2_R2_bitbud_r085_lam0002_s42 "bit_budget=" 11000
watch r2_SPB_base_r085_lam0002_s42  "SensPerBit] engaged" 13000
echo "JR2_WAVE1_WATCHED $(date)" >> "$QLOG"
