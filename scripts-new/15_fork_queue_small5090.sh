#!/usr/bin/env bash
# Fork-framework serial queue for small5090 (single free GPU, one line at a
# time). Spine first, then branches in dependency order; each branch uses the
# production runner (train -> compress -> decoded eval) plus the layered
# curve eval right after, so every arm lands fully registered by itself.
#
# Files (per 2026-10-12 fork design):
#   /home/fansonglin/data_space/Chentong/fork_exp/{code,runs,data}
# GPU: card 1 (card 0 runs another project — do not touch).
set -uo pipefail

BASE=/home/fansonglin/data_space/Chentong/fork_exp
CODE=$BASE/code/DCCA-GS
RUNS=$BASE/runs
DATA=$BASE/data/octree_178
PYBIN=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin
GPU=1
SCHED=$RUNS/camera_schedule_42.npy
export PATH="$BASE/bin:/usr/local/cuda-12.8/bin:$PATH"   # tmc3 + nvcc for gsplat JIT

mkdir -p "$RUNS" "$BASE/bin"
LOG=$RUNS/queue.log
echo "[queue] start $(date)" >> "$LOG"

# 0. Camera schedule (identical for spine and all branches)
if [ ! -f "$SCHED" ]; then
  "$PYBIN/python" "$CODE/scripts/gen_camera_schedule.py" \
    --n-cams 1050 --steps 30000 --seed 42 --out "$SCHED" >> "$LOG" 2>&1
fi

# 1. Spine: full 30k, no mechanism, archives at 15000/24000/30000.
#    Run directly (not via the runner) because the runner hard-codes
#    save-steps to max-steps; the fork needs mid-run archives. No bitstream
#    for the spine — it only produces the fork points and the terminal
#    reference render metrics.
if [ ! -f "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_30000.pth" ]; then
  echo "[queue] SPINE start $(date)" >> "$LOG"
  "$PYBIN/python" "$CODE/train.py" train \
    --cfg.model.model-name hac_pp \
    --cfg.data.data-dir "$DATA" \
    --cfg.data.result-dir "$RUNS/fork_spine_l0004_s42" \
    --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
    --cfg.data.no-preload-images \
    --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 32 --cfg.model.n-offsets 10 \
    --cfg.model.appearance-dim 0 --cfg.model.ratio 1 --cfg.model.tile-size 32 \
    --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
    --cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1 \
    --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
    --cfg.model.sensitivity-weight 0.001 \
    --cfg.optim.max-steps 30000 \
    --cfg.optim.eval-steps 30000 \
    --cfg.optim.save-steps 15000 24000 30000 \
    --cfg.optim.lambda-rate 0.004 --cfg.optim.mask-lr-final 0.002 \
    --cfg.optim.start-stat 500 --cfg.optim.update-from 1500 \
    --cfg.optim.update-until 15000 --cfg.optim.update-interval 100 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
    --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
    --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
    --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
    --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85 \
    --cfg.camera-schedule "$SCHED" \
    > "$RUNS/fork_spine_l0004_s42.log" 2>&1
  echo "[queue] SPINE exit=$? $(date)" >> "$LOG"
fi

# Branch launcher: production runner (train+compress+decoded) with resume,
# then the 150-view layered curve on the same card.
branch() {
  local tag=$1; shift
  if [ ! -f "$RUNS/$tag/decoded_eval/metrics.jsonl" ]; then
    echo "[queue] BRANCH $tag start $(date)" >> "$LOG"
    RUNS_ROOT="$RUNS" RUNROOT="$CODE" CONDA_ENV_BIN="$PYBIN" WAIT_VRAM_MB=30000 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    bash "$CODE/scripts/runner_phg_cell.sh" "$GPU" 1-78 "$DATA" 0.004 "$tag" 30000 15000 \
      --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
      --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
      --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
      --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
      --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85 \
      --cfg.camera-schedule "$SCHED" "$@" \
      >> "$RUNS/$tag.launch.log" 2>&1
    echo "[queue] BRANCH $tag runner exit=$? $(date)" >> "$LOG"
  fi
  if [ -f "$RUNS/$tag/decoded_eval/metrics.jsonl" ] && [ ! -f "$RUNS/$tag/layered_curve_150v.json" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$CODE" \
      "$PYBIN/python" "$CODE/scripts/full_view_eval.py" \
      --run "$RUNS/$tag" --data-dir "$DATA" --data-factor 1 --max-width 1600 \
      --out "$RUNS/$tag/layered_curve_150v.json" \
      >> "$RUNS/$tag.layered.log" 2>&1
    echo "[queue] BRANCH $tag layered exit=$? $(date)" >> "$LOG"
  fi
}

# 2. 15000-fork branches (empty control first: if its terminal PSNR deviates
# from the spine beyond the preregistered 0.08 dB the framework is invalid —
# the queue still completes remaining arms for the record, judgement stops).
branch fork_c0_15000 \
  --cfg.resume "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_15000.pth"
branch fork_e1_pen_15000 \
  --cfg.resume "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_15000.pth" \
  --cfg.model.coarse-ladder-align --cfg.model.coarse-ladder-weight 0.05 \
  --cfg.model.coarse-ladder-start-iter 15000
branch fork_e2_p0r_15000 \
  --cfg.resume "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_15000.pth" \
  --cfg.model.p0-render-loss --cfg.model.p0-render-weight 0.05 \
  --cfg.model.p0-render-start-iter 15000 --cfg.model.p0-render-ramp-iters 15000

# 3. 24000-fork branches (empty control, then the two current-shape arms).
branch fork_c0_24000 \
  --cfg.resume "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_24000.pth"
branch fork_c1_pen_24000 \
  --cfg.resume "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_24000.pth" \
  --cfg.model.coarse-ladder-align --cfg.model.coarse-ladder-weight 0.05
branch fork_c2_p0r_24000 \
  --cfg.resume "$RUNS/fork_spine_l0004_s42/ckpts/ckpt_24000.pth" \
  --cfg.model.p0-render-loss --cfg.model.p0-render-weight 0.05

echo "[queue] ALL_DONE $(date)" >> "$LOG"
