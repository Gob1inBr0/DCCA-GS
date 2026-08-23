#!/usr/bin/env bash
# Disconnect-proof P0 orchestrator for 5090.
# Waits for a free GPU (<1500 MiB used), then runs the remaining P0 jobs in order.
# Run as:  nohup setsid bash scripts/p0_orchestrator.sh >/dev/null 2>&1 &
# Log:     /home/fansonglin/data_space/DCCA-GS/runs/p0_orchestrator.log
set -uo pipefail

BASE=/home/fansonglin/data_space/DCCA-GS
PHG="$BASE/PHG"
R="$BASE/runs"
LOG="$R/p0_orchestrator.log"
exec >> "$LOG" 2>&1

source /home/fansonglin/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate HAC_5090_a100 >/dev/null 2>&1
source "$PHG/scripts/env_5090.sh" >/dev/null 2>&1
cd "$PHG"

log() { echo "[$(date '+%F %T')] $*"; }

wait_gpu() {
  while :; do
    for i in 0 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null | head -1 | tr -d '[:space:]')
      if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 1500 ]; then
        echo "$i"; return 0
      fi
    done
    log "WAIT_GPU: both GPU0/GPU1 busy; sleep 120"
    sleep 120
  done
}

run_finish() {
  local tag=$1 scene=$2 gpu
  gpu=$(wait_gpu)
  log "FINISH_START tag=$tag gpu=$gpu"
  if ! CUDA_VISIBLE_DEVICES="$gpu" bash scripts/finish_p0_run.sh "$tag" "$scene"; then
    log "FINISH_FAILED tag=$tag rc=$?"
  else
    log "FINISH_OK tag=$tag"
  fi
}

run_pipeline() {
  local gpu=$1 tag=$2 scene=$3 spa=$4 mini=$5 ratio=$6 max_steps=$7 update_until=$8
  gpu=$(wait_gpu)
  log "PIPE_START tag=$tag gpu=$gpu"
  if ! bash scripts/run_p0_pipeline_v2.sh "$gpu" "$tag" "$scene" "$spa" "$mini" "$ratio" "$max_steps" "$update_until"; then
    log "PIPE_FAILED tag=$tag rc=$?"
  else
    log "PIPE_OK tag=$tag"
  fi
}

log "ORCHESTRATOR_START $(date)"

# E1 recovery
run_finish p0_e1_playroom_r052_mini playroom

# E4 4-28 110k (r=0.85), protocol matched to the reused baseline (mask-lr-final 0.002)
run_pipeline 0 p0_e4_428_110k_spa_mini 4-28 1 1 0.85 110000 45000
run_pipeline 0 p0_e4_428_110k_spa_base 4-28 1 0 0.85 110000 45000
run_pipeline 0 p0_e4_428_110k_nospa_mini 4-28 0 1 0.85 110000 45000

log "ORCHESTRATOR_DONE $(date)"
