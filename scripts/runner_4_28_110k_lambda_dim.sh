#!/bin/bash
# One cell of the 4-28 lambda x feat_dim cross at 110k.
# Recommended config otherwise: I2+I6, complexity hidden=32/1 layer,
# voxel 0.001, ratio 1, n_offsets 10, appearance 0, tile 32, max-width 1600,
# data-factor 1, no-preload-images, update_until 45000, mask_lr_final 0.002.
# After training: compress (hac_pp) -> baseline decode eval -> MLP quant
# (complexity/deform 8-bit, rest 16-bit) -> quant decode eval.
#
# usage:
#   bash scripts/runner_4_28_110k_lambda_dim.sh <gpu> <tag> <feat_dim> <lambda> [<src_ckpt>]
#   src_ckpt given -> skip training, copy that checkpoint (e.g. reuse dim50/l0.004 110k)
set -e

GPU=$1
TAG=$2
DIM=$3
LAMBDA=$4
SRC_CKPT=${5:-}

MAX_STEPS=110000
UPDATE_UNTIL=45000
DATA=/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28
R=/home/fansonglin/data_space/web_scan/runs/4-28_${TAG}
LOG=/home/fansonglin/data_space/web_scan/runs/4-28_${TAG}.log
RUNROOT=/home/fansonglin/xieliang/chentong/PHG
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUNROOT"
cd "$RUNROOT"

# Wait for the assigned GPU to be free (never kill other users' processes).
for attempt in $(seq 1 40); do
  USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  if [ "${USED_MB:-99999}" -le 1500 ]; then
    echo "GPU_FREE gpu=$GPU used=${USED_MB}MB $(date)"
    break
  fi
  echo "GPU_BUSY gpu=$GPU used=${USED_MB}MB waiting $(date)"
  sleep 180
done

rm -rf "$R"
mkdir -p "$R/ckpts"

if [ -n "$SRC_CKPT" ]; then
  cp "$SRC_CKPT" "$R/ckpts/ckpt_${MAX_STEPS}.pth"
  echo "CKPT_COPIED $SRC_CKPT $(date)"
else
  for attempt in $(seq 1 40); do
    echo "ATTEMPT $attempt $(date)"
    if CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python train.py train \
      --cfg.model.model-name hac_pp \
      --cfg.data.data-dir "$DATA" \
      --cfg.data.result-dir "$R" \
      --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
      --cfg.data.no-preload-images \
      --cfg.model.voxel-size 0.001 --cfg.model.feat-dim "$DIM" --cfg.model.n-offsets 10 \
      --cfg.model.appearance-dim 0 --cfg.model.ratio 1 \
      --cfg.model.tile-size 32 \
      --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
      --cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1 \
      --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
      --cfg.model.sensitivity-weight 0.001 \
      --cfg.optim.max-steps "$MAX_STEPS" --cfg.optim.eval-steps "$MAX_STEPS" \
      --cfg.optim.save-steps "$MAX_STEPS" \
      --cfg.optim.lambda-rate "$LAMBDA" --cfg.optim.mask-lr-final 0.002 \
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
echo "ALL_DONE tag=$TAG dim=$DIM lambda=$LAMBDA anchors=$N_ANCHORS $(date)"
