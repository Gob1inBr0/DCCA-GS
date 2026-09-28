#!/bin/bash
# ph1 smoke v3: provider-friendly smoke on a SHARED GPU.
# Changes vs v2 smoke: mini-splat-views 2 (provider spike ~1.5GB -> ~0.4GB),
# update-interval 400 (fewer, spaced projections), timeout 5400.
# Gate: submod_keep>0 in the train log. Arms are NOT auto-launched — the
# real A2/A3 runs need a solo GPU (provider spike + 33GB training does not
# fit next to a 5GB neighbor on a 40GB card).
set -u
ulimit -n 65536
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
ROOT=/dev/shm/dcca_runs
DATA=/dev/shm/dcca_data/1-78/data
QLOG="$ROOT/ph1v3_queue.log"
g=2
rm -rf "$ROOT/smoke_submod3" "$ROOT/smoke_submod3.log"
cd "$DCCA_ROOT"
export PYTHONNOUSERSITE=1 PYTHONPATH="$DCCA_ROOT:/home/project2/gsplat-main" PATH="$DCCA_PY:$PATH"
echo "SMOKE3 gpu=$g $(date)" >> "$QLOG"
CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout -k 30 5400 python train.py train \
  --cfg.model.model-name hac_pp --cfg.data.data-dir "$DATA" \
  --cfg.data.result-dir "$ROOT/smoke_submod3" \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 32 \
  --cfg.model.n-offsets 10 --cfg.model.appearance-dim 0 --cfg.model.ratio 1 \
  --cfg.model.tile-size 32 --cfg.model.mlp-complexity-hidden 32 \
  --cfg.model.mlp-complexity-layers 1 --cfg.model.sensitivity-enabled \
  --cfg.model.sensitivity-start-iter 0 --cfg.model.sensitivity-weight 0.001 \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 --cfg.model.mini-splat-enabled \
  --cfg.model.mini-splat-reinit-iter 1000 --cfg.model.mini-splat-max-new 4000 \
  --cfg.model.mini-splat-views 2 --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
  --cfg.model.fusion-prune --cfg.model.spa-coverage-constraint \
  --cfg.model.spa-coverage-cell-size 0.05 \
  --cfg.model.submodular-mode cover \
  --cfg.model.spa-post-ratio 0.7 --cfg.model.spa-post-window 200 \
  --cfg.optim.max-steps 1300 --cfg.optim.eval-steps 1300 --cfg.optim.save-steps 1300 \
  --cfg.optim.lambda-rate 0.002 --cfg.optim.mask-lr-final 0.002 \
  --cfg.optim.start-stat 200 --cfg.optim.update-from 100 --cfg.optim.update-until 1000 \
  --cfg.optim.update-interval 400 > "$ROOT/smoke_submod3.log" 2>&1
rc=$?
keep=$(LC_ALL=C grep -aoE "submod_keep=[0-9]+" "$ROOT/smoke_submod3.log" | tail -1 | grep -oE "[0-9]+")
ms=$(LC_ALL=C grep -aoE "greedy_ms=[0-9]+" "$ROOT/smoke_submod3.log" | tail -1)
echo "SMOKE3_EXIT=$rc submod_keep=${keep:-0} $ms $(date)" >> "$QLOG"
if [ "$rc" != "0" ] || [ -z "$keep" ] || [ "$keep" -le 0 ]; then
  echo "GATE_FAIL: submodular smoke did not engage (exit=$rc keep=${keep:-0})" >> "$QLOG"
  exit 1
fi
echo "GATE_PASS: greedy engaged, $ms — arms NOT auto-launched (shared GPU; launch on solo GPU or user call)" >> "$QLOG"
