#!/usr/bin/env bash
# Fork-framework smoke test on small5090 (run BEFORE the queue):
#   A. 200-step spine fragment with the fixed camera schedule and archives
#      at 150/200;
#   B. resume from the 150 archive for the remaining 50 steps — verifies the
#      [Resume] line (core-state restored), schedule indexing, and a clean
#      finish. Real fidelity vs the uninterrupted spine is the C0 arms' job
#      (preregistered 0.08 dB check), not this smoke's.
set -uo pipefail
BASE=/home/fansonglin/data_space/Chentong/fork_exp
CODE=$BASE/code/DCCA-GS
RUNS=$BASE/runs
DATA=$BASE/data/octree_178
PYBIN=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin
SCHED=$RUNS/camera_schedule_42.npy
export PATH="$BASE/bin:/usr/local/cuda-12.8/bin:$PATH"

set -e
"$PYBIN/python" "$CODE/scripts/gen_camera_schedule.py" \
  --n-cams 1050 --steps 30000 --seed 42 --out "$SCHED"

"$PYBIN/python" -m pytest "$CODE/tests/test_fork_resume.py" -q

"$PYBIN/python" "$CODE/train.py" train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir "$DATA" \
  --cfg.data.result-dir "$RUNS/smoke_fork_A" \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.tile-size 32 \
  --cfg.optim.max-steps 200 \
  --cfg.optim.eval-steps 200 --cfg.optim.save-steps 150 200 \
  --cfg.optim.update-from 1500 \
  --cfg.model.spa-enabled --cfg.model.mini-splat-enabled \
  --cfg.model.mini-splat-reinit-iter 15000 \
  --cfg.seed 42 --cfg.model.sensitivity-start-iter 0 \
  --cfg.camera-schedule "$SCHED" \
  > "$RUNS/smoke_fork_A.log" 2>&1
echo "[smoke] A done"

"$PYBIN/python" "$CODE/train.py" train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir "$DATA" \
  --cfg.data.result-dir "$RUNS/smoke_fork_B" \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.tile-size 32 \
  --cfg.optim.max-steps 200 \
  --cfg.optim.eval-steps 200 \
  --cfg.optim.update-from 1500 \
  --cfg.model.spa-enabled --cfg.model.mini-splat-enabled \
  --cfg.model.mini-splat-reinit-iter 15000 \
  --cfg.seed 42 --cfg.model.sensitivity-start-iter 0 \
  --cfg.camera-schedule "$SCHED" \
  --cfg.resume "$RUNS/smoke_fork_A/ckpts/ckpt_150.pth" \
  > "$RUNS/smoke_fork_B.log" 2>&1
echo "[smoke] B done"
grep -E "\[Resume\]|CameraSchedule" "$RUNS/smoke_fork_B.log"
echo "[smoke] ALL OK"
