#!/bin/bash
# Deep Blending (DB) 110k runner with the PHG recommended config:
# I2+I6, feat_dim=50, complexity hidden=32/1, voxel 0.001, n_offsets 10,
# appearance 0, tile 32, max-width 1600, data-factor 1, no-preload-images,
# lambda_rate=0.002, update_until=45000, mask_lr_final=0.002.
#
# mode=train : wait for enough free VRAM, train, then post-process.
# mode=post  : wait for an already-running training to finish, then post-process
#              (used for playroom which was launched with a bare train command).
#
# usage:
#   bash scripts/runner_db_110k.sh <gpu> <scene> <data_dir> <mode>
set -e

GPU=$1
SCENE=$2
DATA=$3
MODE=${4:-train}

TAG=db_${SCENE}_i6_110k_h32_l0p002
MAX_STEPS=110000
UPDATE_UNTIL=45000
R=/home/fansonglin/data_space/web_scan/runs/${TAG}
LOG=/home/fansonglin/data_space/web_scan/runs/${TAG}.log
RUNROOT=/home/fansonglin/xieliang/chentong/PHG
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUNROOT"
cd "$RUNROOT"

wait_vram() {
  # wait until memory.used <= $1 on the assigned GPU (stacking-friendly)
  local LIMIT=$1
  for attempt in $(seq 1 80); do
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    if [ "${USED_MB:-99999}" -le "$LIMIT" ]; then
      echo "VRAM_OK gpu=$GPU used=${USED_MB}MB limit=${LIMIT}MB $(date)"
      return 0
    fi
    echo "VRAM_WAIT gpu=$GPU used=${USED_MB}MB limit=${LIMIT}MB $(date)"
    sleep 180
  done
  echo "GAVE_UP_WAIT_VRAM"
  exit 1
}

if [ "$MODE" = "train" ]; then
  wait_vram 20000   # keep >= 12GB free so the heavy 4-28 runs are not OOM'd
  for attempt in $(seq 1 40); do
    echo "ATTEMPT $attempt $(date)"
    if CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python train.py train \
      --cfg.model.model-name hac_pp \
      --cfg.data.data-dir "$DATA" \
      --cfg.data.result-dir "$R" \
      --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
      --cfg.data.no-preload-images \
      --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 50 --cfg.model.n-offsets 10 \
      --cfg.model.appearance-dim 0 --cfg.model.ratio 1 \
      --cfg.model.tile-size 32 \
      --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
      --cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1 \
      --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
      --cfg.model.sensitivity-weight 0.001 \
      --cfg.optim.max-steps "$MAX_STEPS" --cfg.optim.eval-steps "$MAX_STEPS" \
      --cfg.optim.save-steps "$MAX_STEPS" \
      --cfg.optim.lambda-rate 0.002 --cfg.optim.mask-lr-final 0.002 \
      --cfg.optim.start-stat 500 --cfg.optim.update-from 1500 \
      --cfg.optim.update-until "$UPDATE_UNTIL" --cfg.optim.update-interval 100 \
      > "$LOG" 2>&1
    then
      echo "TRAIN_OK attempt=$attempt $(date)"
      break
    else
      echo "TRAIN_FAILED attempt=$attempt $(date)"
      sleep 180
    fi
  done
  grep -q "Training finished" "$LOG" || { echo "GAVE_UP_TRAIN tag=$TAG"; exit 1; }
else
  # post mode: wait for the externally-launched training to finish
  for i in $(seq 1 400); do
    if grep -q "Training finished" "$LOG" 2>/dev/null; then
      echo "TRAIN_FINISHED $(date)"
      break
    fi
    sleep 60
  done
  grep -q "Training finished" "$LOG" || { echo "GAVE_UP_WAIT_TRAIN tag=$TAG"; exit 1; }
  wait_vram 22000   # post-processing is light; keep >= 10GB free
fi

echo "COMPRESS_START $(date)"
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py compress \
  --cfg.ckpt "$R/ckpts/ckpt_${MAX_STEPS}.pth" \
  --cfg.out-dir "$R/bitstreams" --cfg.codec hac_pp > "$R/compress.log" 2>&1
echo "COMPRESS_DONE $(date)"

echo "EVAL_BASELINE_START $(date)"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval_decoded.py \
  --artifact-dir "$R/bitstreams" \
  --data-dir "$DATA" \
  --result-dir "$R/decoded_eval" \
  --data-factor 1 --max-width 1600 --no-preload-images > "$R/eval.log" 2>&1
echo "EVAL_BASELINE_DONE $(date)"

echo "MLP_QUANT_START $(date)"
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/mlp_quant_sweep.py \
  --ckpt "$R/ckpts/ckpt_${MAX_STEPS}.pth" \
  --data-dir "$DATA" \
  --result-dir "$R/mlp_quant_cd8_rest16" \
  --data-factor 1 --max-width 1600 --no-preload-images \
  --skip-baseline \
  --group-bits mlp_complexity:8 mlp_deform:8 mlp_opacity:16 mlp_cov:16 \
               mlp_color:16 mlp_grid:16 \
  > "$R/mlp_quant.log" 2>&1
echo "MLP_QUANT_DONE $(date)"

N_ANCHORS=$(python3 -c "import json; print(json.load(open('$R/bitstreams/hac_meta.json'))['num_anchors'])" 2>/dev/null || echo '?')
echo "ALL_DONE tag=$TAG scene=$SCENE mode=$MODE anchors=$N_ANCHORS $(date)"
