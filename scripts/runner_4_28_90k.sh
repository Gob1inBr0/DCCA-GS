#!/bin/bash
# 4-28 90k PHG runner.
# Speed protocol by default: eval only at the final step; 30k/60k only save
# checkpoints. Pass an explicit eval list as $7 to override.
#
# usage: bash scripts/runner_4_28_90k.sh \
#   <gpu> <tag> <feat_dim> <max_steps> <update_until> <save_list> [eval_list] [hidden]
set -e
GPU=$1
TAG=$2
DIM=$3
MAX_STEPS=$4
UPDATE_UNTIL=$5
SAVE_LIST=$6
EVAL_LIST=${7:-"$MAX_STEPS"}
HIDDEN=${8:-}
HIDDEN_ARG=""
if [ -n "$HIDDEN" ]; then
  HIDDEN_ARG="--cfg.model.mlp-complexity-hidden $HIDDEN"
fi

R=/home/fansonglin/data_space/web_scan/runs/4-28_${TAG}
LOG=/home/fansonglin/data_space/web_scan/runs/4-28_${TAG}.log
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/fansonglin/xieliang/chentong/PHG
cd /home/fansonglin/xieliang/chentong/PHG

for attempt in $(seq 1 40); do
  echo "ATTEMPT $attempt $(date)"
  FREE_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  if [ "${FREE_MB:-99999}" -gt 1500 ]; then
    echo "GPU_BUSY ${FREE_MB}MB, waiting $(date)"
    sleep 180
    continue
  fi
  echo "GPU_FREE ${FREE_MB}MB $(date)"
  rm -rf "$R"
  mkdir -p "$R"
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train.py train \
    --cfg.model.model-name hac_pp \
    --cfg.data.data-dir /home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 \
    --cfg.data.result-dir "$R" \
    --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
    --cfg.data.no-preload-images \
    --cfg.model.voxel-size 0.001 --cfg.model.feat-dim "$DIM" --cfg.model.n-offsets 10 \
    --cfg.model.appearance-dim 0 --cfg.model.ratio 1 \
    --cfg.model.tile-size 32 \
    --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
    --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
    --cfg.model.sensitivity-weight 0.001 \
    $HIDDEN_ARG \
    --cfg.optim.max-steps "$MAX_STEPS" --cfg.optim.eval-steps $EVAL_LIST --cfg.optim.save-steps $SAVE_LIST \
    --cfg.optim.lambda-rate 0.004 --cfg.optim.mask-lr-final 0.002 \
    --cfg.optim.start-stat 500 --cfg.optim.update-from 1500 --cfg.optim.update-until "$UPDATE_UNTIL" \
    --cfg.optim.update-interval 100 > "$LOG" 2>&1
  if grep -q "Training finished" "$LOG"; then
    echo "TRAIN_OK attempt=$attempt $(date)"
    break
  fi
  echo "TRAIN_FAILED attempt=$attempt $(date)"
  sleep 180
done

grep -q "Training finished" "$LOG" || { echo GAVE_UP; exit 1; }
echo "TRAIN_DONE $(date)"
python train.py compress \
  --cfg.ckpt "$R/ckpts/ckpt_${MAX_STEPS}.pth" \
  --cfg.out-dir "$R/bitstreams" --cfg.codec hac_pp > "$R/compress.log" 2>&1
echo "COMPRESS_DONE $(date)"
python scripts/eval_decoded.py \
  --artifact-dir "$R/bitstreams" \
  --data-dir /home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 \
  --result-dir "$R/decoded_eval" \
  --data-factor 1 --max-width 1600 --no-preload-images > "$R/eval.log" 2>&1
echo "ALL_DONE $(date)"
