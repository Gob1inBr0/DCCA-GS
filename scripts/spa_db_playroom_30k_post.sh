#!/bin/bash
set -e
PHG=/home/fansonglin/xieliang/chentong/PHG
RUNS=/home/fansonglin/data_space/web_scan/runs
CKPT_BASE=$RUNS/spa_db_playroom_30k_base/ckpts/ckpt_30000.pth
CKPT_SPA=$RUNS/spa_db_playroom_30k_spa/ckpts/ckpt_30000.pth
DATA=/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db/playroom
export PYTHONPATH=$PHG
export PYTHONNOUSERSITE=1
export PATH=$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PHG"
echo "[spa_post] waiting for ckpts ..."
while [ ! -f "$CKPT_BASE" ] || [ ! -f "$CKPT_SPA" ]; do
    sleep 60
done
echo "[spa_post] training done $(date)"

echo "[spa_post] compress base (keep all)"
CUDA_VISIBLE_DEVICES=1 python train.py compress \
  --cfg.ckpt "$CKPT_BASE" \
  --cfg.out-dir "$RUNS/spa_db_playroom_30k_base_bit/bitstreams" \
  --cfg.codec hac_pp

echo "[spa_post] compress base topk0.5"
CUDA_VISIBLE_DEVICES=1 python train.py compress \
  --cfg.ckpt "$CKPT_BASE" \
  --cfg.out-dir "$RUNS/spa_db_playroom_30k_topk05_bit/bitstreams" \
  --cfg.codec hac_pp --cfg.mask-keep-ratio 0.5

echo "[spa_post] compress spa0.5 (keep all)"
CUDA_VISIBLE_DEVICES=1 python train.py compress \
  --cfg.ckpt "$CKPT_SPA" \
  --cfg.out-dir "$RUNS/spa_db_playroom_30k_spa_bit/bitstreams" \
  --cfg.codec hac_pp

for D in base topk05 spa; do
    echo "[spa_post] eval $D $(date)"
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_decoded.py \
      --artifact-dir "$RUNS/spa_db_playroom_30k_${D}_bit/bitstreams" \
      --data-dir "$DATA" \
      --result-dir "$RUNS/spa_db_playroom_30k_${D}_bit/decoded_eval" \
      --data-factor 1 --max-width 1600 --no-preload-images
done
echo "[spa_post] ALL_DONE $(date)"
