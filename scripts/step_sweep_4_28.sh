#!/bin/bash
# Compress + decode + evaluate intermediate checkpoints of an existing 4-28
# run, to find the best iteration around 90k.
#
# usage: bash scripts/step_sweep_4_28.sh \
#   <gpu> <run_dir> <steps like "30000 60000 90000">
set -e
GPU=$1
RUN=$2
STEPS=$3
DATA=/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28

export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/fansonglin/xieliang/chentong/PHG
cd /home/fansonglin/xieliang/chentong/PHG

for S in $STEPS; do
  CKPT="$RUN/ckpts/ckpt_${S}.pth"
  [ -f "$CKPT" ] || { echo "MISSING $CKPT"; continue; }
  echo "STEP $S $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train.py compress \
      --cfg.ckpt "$CKPT" \
      --cfg.out-dir "$RUN/bitstreams_step_${S}" \
      --cfg.codec hac_pp > "$RUN/compress_step_${S}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/eval_decoded.py \
      --artifact-dir "$RUN/bitstreams_step_${S}" \
      --data-dir "$DATA" \
      --result-dir "$RUN/eval_step_${S}" \
      --data-factor 1 --max-width 1600 --no-preload-images \
      > "$RUN/eval_step_${S}.log" 2>&1
  echo "STEP_DONE $S $(date)"
done
echo "STEP_SWEEP_ALL_DONE $(date)"
