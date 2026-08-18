#!/bin/bash
# SPA-on DB lambda-RD queue (110k, I2+I6+SPA ratio=0.5), runs after the
# corrected SPA ablation cells finish.
#   GPU0: playroom lambda=0.001 -> drjohnson 0.001 -> drjohnson 0.002
#   GPU1: playroom lambda=0.004 -> drjohnson 0.004
# usage: bash queue_spa_rd.sh <gpu>
set -e

GPU=$1
RUNS=/home/fansonglin/data_space/web_scan/runs
RUNROOT=/home/fansonglin/xieliang/chentong/PHG
export PATH=/home/fansonglin/miniconda3/envs/HAC_5090_a100/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUNROOT"
cd "$RUNROOT"

echo "SPA_RD_WAIT gpu=$GPU $(date)"
until grep -q ALL_DONE "$RUNS/ablation_spa_i6off.runner.log" 2>/dev/null \
  && grep -q ALL_DONE "$RUNS/ablation_spa_i2off.runner.log" 2>/dev/null; do
  sleep 600
done
echo "SPA_RD_START gpu=$GPU $(date)"

run_cell() {
  bash scripts/runner_phg_cell.sh "$@" || echo "CELL_FAIL $* $(date)"
}

DB=/home/fansonglin/xieliang/Chenzhenxin/dataset/tandt_db/db

if [ "$GPU" = 0 ]; then
  run_cell 0 playroom "$DB/playroom" 0.001 \
    db_playroom_i6_110k_h32_l0p001_spa0p5 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
  run_cell 0 drjohnson "$DB/drjohnson" 0.001 \
    db_drjohnson_i6_110k_h32_l0p001_spa0p5 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
  run_cell 0 drjohnson "$DB/drjohnson" 0.002 \
    db_drjohnson_i6_110k_h32_l0p002_spa0p5 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
else
  run_cell 1 playroom "$DB/playroom" 0.004 \
    db_playroom_i6_110k_h32_l0p004_spa0p5 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
  run_cell 1 drjohnson "$DB/drjohnson" 0.004 \
    db_drjohnson_i6_110k_h32_l0p004_spa0p5 110000 45000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.5 --cfg.model.spa-rho 0.001
fi

echo "QUEUE_SPA_RD_GPU${GPU}_ALL_DONE $(date)"
if [ "$GPU" = 0 ]; then
  python scripts/collect_queue_results.py
  echo "SPA_RD_COLLECT_DONE $(date)"
fi
