#!/usr/bin/env bash
# ASG G1 cell: playroom 30k, cell2 protocol (I2+I6+SPA0.85+MiniSplat).
#
# usage: scripts/run_asg_g1_cell.sh <gpu> <tag> <rgb|asg> [max_steps=30000] [update_until=15000]
set -euo pipefail

GPU=$1
TAG=$2
COLOR=$3
MAX_STEPS=${4:-30000}
UPDATE_UNTIL=${5:-15000}
BASE=/home/fansonglin/data_space/DCCA-GS
PHG="$BASE/PHG-asg"
OUT="$BASE/runs/$TAG"
DATA="$BASE/data/playroom"

source /home/fansonglin/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate HAC_5090_a100 >/dev/null 2>&1
source "$PHG/scripts/env_5090.sh" >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$PHG"

echo "RUNNING" > "$OUT/STATUS"
echo "[$(date '+%F %T')] START tag=$TAG gpu=$GPU color=$COLOR"

python train.py train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir "$DATA" --cfg.data.result-dir "$OUT" \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.feat-dim 50 --cfg.model.tile-size 32 --cfg.model.appearance-dim 0 \
  --cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1 \
  --cfg.model.content-aware-quant --cfg.model.sensitivity-enabled \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
  --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter "$UPDATE_UNTIL" \
  --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
  --cfg.model.mini-splat-voxel 0.0 \
  --cfg.model.color-mode "$COLOR" \
  --cfg.optim.max-steps "$MAX_STEPS" --cfg.optim.update-until "$UPDATE_UNTIL" \
  --cfg.optim.save-steps "$MAX_STEPS" --cfg.optim.eval-steps "$MAX_STEPS" \
  --cfg.optim.lambda-rate 0.004 --cfg.optim.mask-lr-final 0.002 \
  > "$OUT/train.log" 2>&1
grep -q "Training finished" "$OUT/train.log" || {
  echo "TRAIN_FAILED tag=$TAG"; echo "FAILED" > "$OUT/STATUS"
  tail -40 "$OUT/train.log"; exit 1
}
echo "[$(date '+%F %T')] TRAIN_DONE tag=$TAG"

python train.py compress \
  --cfg.ckpt "$OUT/ckpts/ckpt_${MAX_STEPS}.pth" \
  --cfg.out-dir "$OUT/bitstreams" --cfg.codec hac_pp > "$OUT/compress.log" 2>&1
echo "[$(date '+%F %T')] COMPRESS_DONE tag=$TAG"

python scripts/eval_decoded.py \
  --artifact-dir "$OUT/bitstreams" --data-dir "$DATA" \
  --result-dir "$OUT/decoded_eval" \
  --data-factor 1 --max-width 1600 --no-preload-images > "$OUT/eval_decoded.log" 2>&1

echo "DONE" > "$OUT/STATUS"
echo "[$(date '+%F %T')] PIPELINE_DONE tag=$TAG"
tail -1 "$OUT/decoded_eval/metrics.jsonl" 2>/dev/null || true
