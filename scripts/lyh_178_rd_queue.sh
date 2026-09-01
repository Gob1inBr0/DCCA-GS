#!/usr/bin/env bash
# LYH 1-78 RD queue:
#   1) HAC++ high-rate (lmbda=0.004) and low-rate (lmbda=0.0005)
#   2) DCCA no-SPA / no-MiniSplat multi-rate points
#
# Run on the remote GPU host; this script only starts jobs on GPUs 3/6/7 and
# leaves the existing 1-78 SPA training (GPU0/1/2) untouched.

set -u

# HAC++ keeps one PIL Image handle per camera while loading the dataset;
# 1-78 has 1200 images, so raise the per-process fd limit before training.
ulimit -n 65536

export PATH="/home/project2/tmc13/build/tmc3:${PATH}"
export PYTHONPATH="/home/project2/HAC-plus:${PYTHONPATH:-}"

DATA=/dev/shm/dcca_data/1-78/data
ROOT=/dev/shm/dcca_runs/1-78
HAC_ROOT=/home/project2/HAC-plus
DCCA_ROOT=/home/project2/DCCA-GS-minifull
DCCA_PY=/home/project2/miniconda3/envs/DCCA/bin
HAC_PY=/mnt/003/conda_envs/HAC_plus/bin

wait_free_gpu() {
  local gpu=$1
  for _ in $(seq 1 120); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 99999)
    if [ "${used:-99999}" -le 30000 ]; then
      echo "GPU_FREE gpu=$gpu used=${used}MB"
      return 0
    fi
    sleep 30
  done
  echo "GPU_STILL_BUSY gpu=$gpu" >&2
  return 1
}

run_hacpp() {
  local gpu=$1 lambda=$2 tag=$3
  local log="$ROOT/${tag}.log"
  local mask
  mask=$(python3 -c "print(0.0001 * float('$lambda') / 0.001)"
  )
  cd "$HAC_ROOT" || exit 1
  echo "START_HACPP tag=$tag gpu=$gpu lambda=$lambda $(date)" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$HAC_PY/python" train.py \
    -s "$DATA" \
    --eval --lod 0 --resolution -1 \
    --voxel_size 0.001 --update_init_factor 16 --iterations 30000 \
    -m "$ROOT/$tag" \
    --lmbda "$lambda" --mask_lr_final "$mask" \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" >> "$ROOT/${tag}.pid"
  echo "HACPP_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda"
}

run_dcca() {
  local gpu=$1 lambda=$2 tag=$3
  local log="$ROOT/${tag}.log"
  wait_free_gpu "$gpu" || return 1
  echo "START_DCCA tag=$tag gpu=$gpu lambda=$lambda $(date)" > "$log"
  RUNS_ROOT="$ROOT" \
  RUNROOT="$DCCA_ROOT" \
  CONDA_ENV_BIN="$DCCA_PY" \
  WAIT_VRAM_MB=30000 \
  nohup bash "$DCCA_ROOT/scripts/runner_phg_cell.sh" \
    "$gpu" 1-78 "$DATA" "$lambda" "$tag" 110000 45000 \
    --cfg.model.no-spa-enabled \
    --cfg.model.no-mini-splat-enabled \
    >> "$log" 2>&1 &
  local pid=$!
  echo "$pid" >> "$ROOT/${tag}.pid"
  echo "DCCA_BG pid=$pid tag=$tag gpu=$gpu lambda=$lambda"
}

mkdir -p "$ROOT"

# First wave: HAC++ high/low + DCCA no-SPA high-rate.
# GPU 3 / 6 / 7 are the currently free cards on the host.
run_hacpp 3 0.004 hacpp_1-78_high_l0004
run_hacpp 6 0.0005 hacpp_1-78_low_l0005
run_dcca 7 0.004 dcca_1-78_nospa_l0004

echo "FIRST_WAVE_LAUNCHED $(date)"

# Wait for the first wave, then start the remaining no-SPA rate points.
# The three runner processes for the current SPA baseline/depth/full are on
# GPU0/1/2 and are intentionally not managed here.
for pid_file in "$ROOT"/hacpp_1-78_high_l0004.pid \
                "$ROOT"/hacpp_1-78_low_l0005.pid \
                "$ROOT"/dcca_1-78_nospa_l0004.pid; do
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    wait "$pid" || true
  fi
done

run_dcca 3 0.002 dcca_1-78_nospa_l0002
run_dcca 6 0.0005 dcca_1-78_nospa_l0005

for pid_file in "$ROOT"/dcca_1-78_nospa_l0002.pid \
                "$ROOT"/dcca_1-78_nospa_l0005.pid; do
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    wait "$pid" || true
  fi
done

echo "RD_QUEUE_DONE $(date)" | tee "$ROOT/rd_queue_done.txt"
