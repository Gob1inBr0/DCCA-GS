#!/bin/bash
# Journal round 2 waves (docs/02-design/期刊版第二轮_四方向设计与查新.md).
# DISCIPLINE: GPU work starts only AFTER the ICRA submission (2026-09-15).
# Waves (selector = $1):
#   smoke   - 400-step launch sanity for every arm (no crash + startup banner)
#   rate    - R1 rate-discount base / R2 bit-budget base / R3 rate-discount fusA
#   spb     - D3 sens_per_bit target (complexity-multiplier supervision)
#   fisher  - F1 fusA + Fisher second-order score
#   gate    - G1 holdout recovery gate at the deep-pruning budget r_post=0.55
#   weights - W1..W3 fusion weight coordinate search, stage 1
# Every arm ships with an engagement hard-gate (audit lesson: a mechanism
# that silently fails must be caught by log evidence, not assumed).
# Baselines (docs/data/experiments.csv, 1-78 30k seed42):
#   ph0c_base_r085_lam0004_s42  26.9829 @ 11.08 MB
#   ph0c_fusA_r085_lam0004_s42  27.4258 @ 11.25 MB
#   ph0a_base_r055_lam0002_s42  27.2853 @ 11.10 MB
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
B="--cfg.model.spa-post-ratio 0.85"
F="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 --cfg.model.spa-post-ratio 0.85"

launch() {
  local gpu=$1 lam=$2 tag=$3 extra=$4
  echo "START tag=$tag gpu=$gpu lambda=$lam $(date)" > "$ROOT/${tag}.launch.log"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lam" "$tag" 30000 15000 $Z $extra \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
  echo "LAUNCHED gpu=$gpu tag=$tag"
}

# watch_engagement tag pattern deadline_step
# Greps the train log ($ROOT/${tag}.log — the path runner_phg_cell.sh writes)
# until `pattern` appears (engagement evidence), the training ends/crashes,
# a step counter passes `deadline`, or the wall-clock cap hits -> FAIL + void.
watch_engagement() {
  local tag=$1 pattern=$2 deadline=$3
  local log="$ROOT/${tag}.log"
  local tries=0 max_tries=180   # 180 x 5 min = 15 h wall-clock cap
  while [ "$tries" -lt "$max_tries" ]; do
    if grep -q "$pattern" "$log" 2>/dev/null; then
      echo "ENGAGED tag=$tag pattern='$pattern' $(date)"
      return 0
    fi
    if grep -qE "Training finished|GAVE_UP_TRAIN|Traceback" "$log" 2>/dev/null; then
      echo "ENGAGEMENT_FAIL tag=$tag: training ended before pattern matched"
      return 1
    fi
    local step
    step=$(grep -oE 'step=[0-9]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "$step" ] && [ "$step" -gt "$deadline" ]; then
      echo "ENGAGEMENT_FAIL tag=$tag pattern='$pattern' at step=$step"
      echo "  -> kill the run, register as void group in experiments.csv"
      return 1
    fi
    tries=$((tries + 1))
    sleep 300
  done
  echo "ENGAGEMENT_FAIL tag=$tag: watcher wall-clock cap reached"
  return 1
}

case "${1:-help}" in
  smoke)
    # 400-step sanity with a compressed projection window (update 100..350)
    # so PruneLog actually fires. Full bits engagement cannot fire in 400
    # steps (the entropy subsample starts at 10k); that is checked by the
    # per-wave watchers instead.
    for extra in "$B --cfg.model.spa-rate-aware" \
                 "$B --cfg.model.spa-bit-budget" \
                 "$F --cfg.model.spa-rate-aware" \
                 "$B --cfg.model.sensitivity-target-mode sens_per_bit" \
                 "$F --cfg.model.sensitivity-second-order --cfg.model.sensitivity-use-fisher" \
                 "--cfg.model.spa-post-ratio 0.55 --cfg.model.spa-holdout-gate"; do
      tag="smoke_r2_$(echo "$extra" | md5sum | cut -c1-6)"
      timeout -k 30 2400 "$DCCA_PY/python" "$DCCA_ROOT/train.py" train \
        --cfg.model.model-name hac_pp --cfg.data.data-dir "$DATA" \
        --cfg.data.result-dir "$ROOT/$tag" \
        --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
        --cfg.data.no-preload-images --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 32 \
        --cfg.model.n-offsets 10 --cfg.model.appearance-dim 0 --cfg.model.ratio 1 \
        --cfg.model.tile-size 32 --cfg.optim.max-steps 400 \
        --cfg.optim.update-from 100 --cfg.optim.update-until 350 \
        $Z $extra > "$ROOT/$tag.log" 2>&1
      rc=$?
      ok_banner=$(grep -c "TrainerVer" "$ROOT/$tag.log" || true)
      ok_prune=$(grep -c "PruneLog" "$ROOT/$tag.log" || true)
      ok_gate=$(grep -c "HoldoutGate" "$ROOT/$tag.log" || true)
      echo "SMOKE tag=$tag rc=$rc banner=$ok_banner prunelog=$ok_prune holdout=$ok_gate"
      grep -E "Traceback|Error" "$ROOT/$tag.log" | head -3
    done
    ;;
  rate)
    launch 0 0.004 r2_rate_base_r085_lam0004_s42  "$B --cfg.model.spa-rate-aware --cfg.model.spa-rate-tau 1.0"
    launch 1 0.004 r2_bitbud_base_r085_lam0004_s42 "$B --cfg.model.spa-bit-budget"
    launch 6 0.004 r2_rate_fusA_r085_lam0004_s42  "$F --cfg.model.spa-rate-aware"
    # Engagement: bits estimate needs ~10k steps of entropy subsampling;
    # discount/budget lines carry bits_med= in PruneLog. Deadline 11k.
    watch_engagement r2_rate_base_r085_lam0004_s42   "bits_med=" 11000
    watch_engagement r2_bitbud_base_r085_lam0004_s42 "bit_budget=" 11000
    watch_engagement r2_rate_fusA_r085_lam0004_s42   "bits_med=" 11000
    ;;
  spb)
    launch 0 0.004 r2_spb_base_r085_lam0004_s42 "$B --cfg.model.sensitivity-target-mode sens_per_bit"
    watch_engagement r2_spb_base_r085_lam0004_s42 "SensPerBit] engaged" 13000
    ;;
  fisher)
    launch 1 0.004 r2_fisher_fusA_r085_lam0004_s42 \
      "$F --cfg.model.sensitivity-second-order --cfg.model.sensitivity-use-fisher"
    watch_engagement r2_fisher_fusA_r085_lam0004_s42 "fisher_engaged=[1-9]" 11000
    ;;
  gate)
    # Deep-pruning regime where wrong pruning hurts: r_post=0.55 @ lam 0.002.
    # B0 baseline exists (ph0a_base_r055_lam0002_s42 27.2853 @ 11.10).
    launch 6 0.002 r2_gate_base_r055_lam0002_s42 \
      "--cfg.model.spa-post-ratio 0.55 --cfg.model.spa-holdout-gate"
    # The gate is armed from step 0 (HoldoutGate armed) and probes every
    # post-phase projection; engagement = HoldoutGate lines in the log.
    watch_engagement r2_gate_base_r055_lam0002_s42 "HoldoutGate" 16000
    ;;
  weights)
    # Stage 1: coordinate search over the sensitivity share w at gamma=0.25.
    # Stage 2 (manual, after CSV): gamma in {0.1, 0.5} at the best w.
    launch 7 0.004 r2_w010_fusA_r085_lam0004_s42 "$F --cfg.model.fusion-sensitivity-weight 0.10"
    launch 0 0.004 r2_w020_fusA_r085_lam0004_s42 "$F --cfg.model.fusion-sensitivity-weight 0.20"
    # w=0.30/gamma=0.25 baseline already exists: ph0c_fusA_r085_lam0004_s42.
    ;;
  *)
    echo "usage: $0 {smoke|rate|spb|fisher|gate|weights}"
    ;;
esac
