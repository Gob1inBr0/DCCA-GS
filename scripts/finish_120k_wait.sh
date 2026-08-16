#!/bin/bash
# Wait for a free GPU (respect other users), then finish the 4-28 120k run:
# compress + eval ckpt_120000, then step-sweep 60k..110k checkpoints.
set -e
RUN=/home/fansonglin/data_space/web_scan/runs/4-28_i6_90k_h32_120k
DATA=/home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28

export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/fansonglin/xieliang/chentong/PHG
cd /home/fansonglin/xieliang/chentong/PHG

GPU=""
for i in $(seq 1 240); do
  for g in 0 1; do
    MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "${MB:-99999}" -le 1500 ]; then
      GPU=$g
      break 2
    fi
  done
  echo "waiting for free GPU $(date)"
  sleep 180
done
[ -n "$GPU" ] || { echo GAVE_UP_NO_FREE_GPU; exit 1; }
echo "USING GPU $GPU $(date)"

CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py compress --cfg.ckpt "$RUN/ckpts/ckpt_120000.pth" \
  --cfg.out-dir "$RUN/bitstreams" --cfg.codec hac_pp > "$RUN/compress.log" 2>&1
echo "COMPRESS_DONE $(date)"

CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/eval_decoded.py --artifact-dir "$RUN/bitstreams" \
  --data-dir "$DATA" --result-dir "$RUN/decoded_eval" \
  --data-factor 1 --max-width 1600 --no-preload-images > "$RUN/eval.log" 2>&1
echo "EVAL_DONE $(date)"

bash scripts/step_sweep_4_28.sh "$GPU" "$RUN" "60000 70000 80000 90000 100000 110000"
echo "ALL_FINISH_DONE $(date)"
