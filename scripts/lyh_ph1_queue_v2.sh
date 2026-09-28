#!/bin/bash
# Phase 1 queue v2 (post API fix): strict gate before full runs.
# 1) wait for a truly free GPU (whitelist, <5GB);
# 2) compressed smoke: REQUIRE exit=0 AND submod_keep>0 in log, else FAIL-stop;
# 3) A2 (cover) 30k, then A3 (cover_sens) 30k.
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/ph1v2_queue.log"

free_gpu() {
  for g in 1 2 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
    [ "${u:-99999}" -le 5000 ] && echo $g && return 0
  done
  return 1
}
wait_free() { while :; do g=$(free_gpu) && echo "$g" && return 0; sleep 300; done; }

S0="--cfg.model.sensitivity-start-iter 0"
Z="--cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 $S0"
R="--cfg.model.spa-post-ratio 0.85"
FUSION="--cfg.model.fusion-prune --cfg.model.spa-coverage-constraint --cfg.model.spa-coverage-cell-size 0.05 $R"

g=$(wait_free)
echo "SMOKE gpu=$g $(date)" >> "$QLOG"
cd "$DCCA_ROOT"
export PYTHONNOUSERSITE=1 PYTHONPATH="$DCCA_ROOT:/home/project2/gsplat-main" PATH="$DCCA_PY:$PATH"
rm -rf "$ROOT/smoke_submod" "$ROOT/smoke_submod.log"
CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout -k 30 2400 python train.py train \
  --cfg.model.model-name hac_pp --cfg.data.data-dir "$DATA" \
  --cfg.data.result-dir "$ROOT/smoke_submod" \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 32 \
  --cfg.model.n-offsets 10 --cfg.model.appearance-dim 0 --cfg.model.ratio 1 \
  --cfg.model.tile-size 32 --cfg.model.mlp-complexity-hidden 32 \
  --cfg.model.mlp-complexity-layers 1 --cfg.model.sensitivity-enabled \
  --cfg.model.sensitivity-start-iter 0 --cfg.model.sensitivity-weight 0.001 \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled \
  --cfg.model.mini-splat-reinit-iter 1000 --cfg.model.mini-splat-max-new 4000 \
  --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
  --cfg.model.fusion-prune --cfg.model.spa-coverage-constraint \
  --cfg.model.spa-coverage-cell-size 0.05 \
  --cfg.model.submodular-mode cover \
  --cfg.model.spa-post-ratio 0.7 --cfg.model.spa-post-window 200 \
  --cfg.optim.max-steps 1300 --cfg.optim.eval-steps 1300 --cfg.optim.save-steps 1300 \
  --cfg.optim.lambda-rate 0.002 --cfg.optim.mask-lr-final 0.002 \
  --cfg.optim.start-stat 200 --cfg.optim.update-from 100 --cfg.optim.update-until 1000 \
  --cfg.optim.update-interval 50 > "$ROOT/smoke_submod.log" 2>&1
rc=$?
keep=$(LC_ALL=C grep -aoE "submod_keep=[0-9]+" "$ROOT/smoke_submod.log" | tail -1 | grep -oE "[0-9]+")
echo "SMOKE_EXIT=$rc submod_keep=${keep:-0} $(date)" >> "$QLOG"
if [ "$rc" != "0" ] || [ -z "$keep" ] || [ "$keep" -le 0 ]; then
  echo "GATE_FAIL: submodular smoke did not engage (exit=$rc keep=${keep:-0}). A2/A3 NOT launched." >> "$QLOG"
  exit 1
fi

launch_arm() {
  local tag=$1 mode=$2
  local g
  g=$(wait_free)
  echo "ARM $tag gpu=$g mode=$mode $(date)" >> "$QLOG"
  RUNS_ROOT="$ROOT" RUNROOT="$DCCA_ROOT" CONDA_ENV_BIN="$DCCA_PY" \
  GSPLAT_ROOT=/home/project2/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$g" 1-78 "$DATA" 0.002 "$tag" 30000 15000 $Z $FUSION \
    --cfg.model.submodular-mode "$mode" \
    >> "$ROOT/${tag}.launch.log" 2>&1 < /dev/null &
}

launch_arm ph1_A2_subv1_r085_lam0002_s42 cover
launch_arm ph1_A3_subv2_r085_lam0002_s42 cover_sens
echo "PH1V2_QUEUE_DONE $(date)" >> "$QLOG"
