#!/usr/bin/env bash
# P0 pipeline: train -> HAC++ compress -> decode -> eval.
# Matches the protocol of runs/spa_minisplat_launch.sh (playroom) and
# runs/run_428_launch.sh (4-28), extended with SPA + MiniSplat toggles.
#
# usage: run_p0_pipeline.sh <gpu> <tag> <scene:playroom|4-28> <spa:0|1> <mini:0|1> <ratio>
#                            [max_steps=30000] [update_until=15000] [lambda=0.004] [extra args...]
set -euo pipefail

GPU=$1; TAG=$2; SCENE=$3; SPA=$4; MINI=$5; RATIO=$6
MAX_STEPS=${7:-30000}; UPDATE_UNTIL=${8:-15000}; LAMBDA=${9:-0.004}
if [ $# -ge 9 ]; then shift 9; else shift $#; fi
EXTRA="$*"

BASE=/home/fansonglin/data_space/DCCA-GS
PHG="$BASE/PHG"
OUT="$BASE/runs/$TAG"

source /home/fansonglin/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate HAC_5090_a100 >/dev/null 2>&1
source "$PHG/scripts/env_5090.sh" >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

case "$SCENE" in
  playroom) DATA="$BASE/data/playroom" ;;
  4-28)     DATA=/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 ;;
  *) echo "BAD_SCENE $SCENE"; exit 2 ;;
esac

SPA_ARGS=""
[ "$SPA" = "1" ] && SPA_ARGS="--cfg.model.spa-enabled --cfg.model.spa-ratio $RATIO"
MINI_ARGS=""
[ "$MINI" = "1" ] && MINI_ARGS="--cfg.model.mini-splat-enabled \
  --cfg.model.mini-splat-reinit-iter $UPDATE_UNTIL --cfg.model.mini-splat-max-new 4000 \
  --cfg.model.mini-splat-views 8 --cfg.model.mini-splat-voxel 0.0"

echo "[$(date '+%F %T')] START tag=$TAG gpu=$GPU scene=$SCENE spa=$SPA mini=$MINI ratio=$RATIO steps=$MAX_STEPS update_until=$UPDATE_UNTIL lambda=$LAMBDA"

python train.py train \
  --cfg.model.model-name hac_pp \
  --cfg.data.data-dir "$DATA" --cfg.data.result-dir "$OUT" \
  --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
  --cfg.data.no-preload-images \
  --cfg.model.feat-dim 50 --cfg.model.tile-size 32 --cfg.model.appearance-dim 0 \
  --cfg.model.mlp-complexity-hidden 32 \
  --cfg.model.content-aware-quant --cfg.model.sensitivity-enabled \
  $SPA_ARGS $MINI_ARGS $EXTRA \
  --cfg.optim.max-steps "$MAX_STEPS" --cfg.optim.update-until "$UPDATE_UNTIL" \
  --cfg.optim.save-steps "$MAX_STEPS" --cfg.optim.eval-steps "$MAX_STEPS" \
  --cfg.optim.lambda-rate "$LAMBDA" \
  > "$OUT/train.log" 2>&1
grep -q "Training finished" "$OUT/train.log" || { echo "TRAIN_FAILED tag=$TAG"; tail -40 "$OUT/train.log"; exit 1; }
echo "[$(date '+%F %T')] TRAIN_DONE tag=$TAG"

python train.py compress \
  --cfg.ckpt "$OUT/ckpts/ckpt_${MAX_STEPS}.pth" \
  --cfg.out-dir "$OUT/bitstreams" --cfg.codec hac_pp > "$OUT/compress.log" 2>&1
echo "[$(date '+%F %T')] COMPRESS_DONE tag=$TAG"

python scripts/eval_decoded.py \
  --artifact-dir "$OUT/bitstreams" --data-dir "$DATA" \
  --result-dir "$OUT/decoded_eval" \
  --data-factor 1 --max-width 1600 --no-preload-images > "$OUT/eval_decoded.log" 2>&1

echo "[$(date '+%F %T')] ALL_DONE tag=$TAG"
echo "=== METRICS $TAG ==="
tail -1 "$OUT/decoded_eval/metrics.jsonl" 2>/dev/null || true
python3 - "$OUT/bitstreams/hac_meta.json" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print(f"total_MB={d.get('total_MB')} coded_anchors={d.get('num_anchors')}")
PY
echo "[$(date '+%F %T')] PIPELINE_DONE tag=$TAG"
