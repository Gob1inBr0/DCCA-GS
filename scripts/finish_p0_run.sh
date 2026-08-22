#!/usr/bin/env bash
# Compress + decode + eval an already-trained P0 run (training skipped).
# usage: finish_p0_run.sh <tag> <scene:playroom|4-28>
set -euo pipefail

TAG=$1; SCENE=$2
BASE=/home/fansonglin/data_space/DCCA-GS
PHG="$BASE/PHG"
OUT="$BASE/runs/$TAG"

source /home/fansonglin/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate HAC_5090_a100 >/dev/null 2>&1
source "$PHG/scripts/env_5090.sh" >/dev/null 2>&1
cd "$PHG"

case "$SCENE" in
  playroom) DATA="$BASE/data/playroom" ;;
  4-28)     DATA=/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 ;;
  *) echo "BAD_SCENE $SCENE"; exit 2 ;;
esac

echo "[$(date '+%F %T')] FINISH_START tag=$TAG"
python train.py compress \
  --cfg.ckpt "$OUT/ckpts/ckpt_30000.pth" \
  --cfg.out-dir "$OUT/bitstreams" --cfg.codec hac_pp > "$OUT/compress.log" 2>&1
python scripts/eval_decoded.py \
  --artifact-dir "$OUT/bitstreams" --data-dir "$DATA" \
  --result-dir "$OUT/decoded_eval" \
  --data-factor 1 --max-width 1600 --no-preload-images > "$OUT/eval_decoded.log" 2>&1
echo "DONE" > "$OUT/STATUS"
echo "[$(date '+%F %T')] FINISH_DONE tag=$TAG"
echo "=== METRICS $TAG ==="
tail -1 "$OUT/decoded_eval/metrics.jsonl" 2>/dev/null || true
python3 - "$OUT/bitstreams/hac_meta.json" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print(f"total_MB={d.get('total_MB')} coded_anchors={d.get('num_anchors')}")
PY
