#!/usr/bin/env bash
# Standard-benchmark matrix queue for LYH (CVPR remaining-task #1).
# 24 trainings = 4 new scenes x 3 lambdas (bicycle, stump, playroom, drjohnson)
#              + 6 done scenes x 2 more lambdas (train, truck, bonsai, counter, kitchen, room).
# garden is complete (3 lambdas) and is not re-run.
# flowers/treehill: authors do not permit public sharing (data/flowers.txt) — excluded.
#
# Constraints (2026-10-13 user instructions):
#   - at most THREE GPUs at any time (never occupy all cards);
#   - NEVER delete any ply model (checkpoints are moved to tmpfs instead).
# Per-dataset protocol cloned from the existing registered arms:
#   Mip360 outdoor (bicycle, stump)   factor 2, width 3200  (garden protocol)
#   Mip360 indoor  (bonsai, counter,
#                   kitchen, room)    factor 4, width 1600
#   T&T (train, truck)                factor 2, width 1600
#   DB (playroom, drjohnson)          factor 1, width 1600
set -uo pipefail

CODE=/home/project2/DCCA-GS
RUNS=/home/project2/dcca_runs
DATA_ROOT=/home/project2/data
PYBIN=/home/project2/miniconda3/envs/DCCA/bin
GPUS_ALLOWED="0 1 2"          # never more than three cards
MAX_INFLIGHT=3
ARCHIVE=/dev/shm/dcca_ckpt_archive
mkdir -p "$RUNS/std_matrix" "$ARCHIVE"
LOG=$RUNS/std_matrix/queue.log
LOCK=$RUNS/std_matrix/.queue_running

[ -f "$LOCK" ] && { echo "std matrix queue already running" >&2; exit 1; }
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# tag | data_dir | factor | width | lambda
JOBS=(
  "std_bicycle_l0002_s42|$DATA_ROOT/bicycle|2|3200|0.0002"
  "std_stump_l0002_s42|$DATA_ROOT/stump|2|3200|0.0002"
  "std_playroom_l0002_s42|$DATA_ROOT/db/playroom|1|1600|0.0002"
  "std_drjohnson_l0002_s42|$DATA_ROOT/db/drjohnson|1|1600|0.0002"
  "std_train_l0004_s42|$DATA_ROOT/tandt/train|2|1600|0.0004"
  "std_truck_l0004_s42|$DATA_ROOT/tandt/truck|2|1600|0.0004"
  "std_bonsai_l0004_s42|$DATA_ROOT/bonsai|4|1600|0.0004"
  "std_counter_l0004_s42|$DATA_ROOT/counter|4|1600|0.0004"
  "std_kitchen_l0004_s42|$DATA_ROOT/kitchen|4|1600|0.0004"
  "std_room_l0004_s42|$DATA_ROOT/room|4|1600|0.0004"
  "std_train_l0001_s42|$DATA_ROOT/tandt/train|2|1600|0.0001"
  "std_truck_l0001_s42|$DATA_ROOT/tandt/truck|2|1600|0.0001"
  "std_bonsai_l0001_s42|$DATA_ROOT/bonsai|4|1600|0.0001"
  "std_counter_l0001_s42|$DATA_ROOT/counter|4|1600|0.0001"
  "std_kitchen_l0001_s42|$DATA_ROOT/kitchen|4|1600|0.0001"
  "std_room_l0001_s42|$DATA_ROOT/room|4|1600|0.0001"
  "std_bicycle_l0004_s42|$DATA_ROOT/bicycle|2|3200|0.0004"
  "std_stump_l0004_s42|$DATA_ROOT/stump|2|3200|0.0004"
  "std_playroom_l0004_s42|$DATA_ROOT/db/playroom|1|1600|0.0004"
  "std_drjohnson_l0004_s42|$DATA_ROOT/db/drjohnson|1|1600|0.0004"
  "std_bicycle_l0001_s42|$DATA_ROOT/bicycle|2|3200|0.0001"
  "std_stump_l0001_s42|$DATA_ROOT/stump|2|3200|0.0001"
  "std_playroom_l0001_s42|$DATA_ROOT/db/playroom|1|1600|0.0001"
  "std_drjohnson_l0001_s42|$DATA_ROOT/db/drjohnson|1|1600|0.0001"
)

free_gpu() {
  for g in $GPUS_ALLOWED; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "${used:-99999}" -lt 2000 ] && { echo "$g"; return 0; }
  done
  return 1
}

inflight() {
  local n=0 tag
  for tag in $(ls "$RUNS/std_matrix" 2>/dev/null | grep "^\.running_"); do
    t=${tag#.running_}
    [ -f "$RUNS/$t/decoded_eval/metrics.jsonl" ] || n=$((n+1))
  done
  echo "$n"
}

echo "[stdq] start $(date)" >> "$LOG"
while true; do
  # 1. harvest finished trainings: layered eval + archive checkpoints (ply untouched)
  for marker in $(ls "$RUNS/std_matrix" 2>/dev/null | grep "^\.running_"); do
    tag=${marker#.running_}
    if [ -f "$RUNS/$tag/decoded_eval/metrics.jsonl" ] && [ ! -f "$RUNS/std_matrix/.curved_$tag" ]; then
      # read the job's factor/width
      jobline=""
      for j in "${JOBS[@]}"; do [ "${j%%|*}" = "$tag" ] && jobline="$j"; done
      IFS='|' read -r _t data factor width _lam <<< "$jobline"
      g=$(free_gpu) || g=0
      echo "[stdq] layered $tag on gpu$g $(date '+%H:%M')" >> "$LOG"
      export PATH="$PYBIN:$PATH"
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/home/T0ng/gsplat-main:$CODE \
        "$PYBIN/python" "$CODE/scripts/full_view_eval.py" \
        --run "$RUNS/$tag" --data-dir "$data" --data-factor "$factor" --max-width "$width" \
        --out "$RUNS/$tag/layered_curve_150v.json" \
        >> "$RUNS/std_matrix/$tag.layered.log" 2>&1
      echo "[stdq] layered $tag exit=$? $(date '+%H:%M')" >> "$LOG"
      touch "$RUNS/std_matrix/.curved_$tag"
      # archive checkpoints to tmpfs (ply stays on disk)
      if [ -d "$RUNS/$tag/ckpts" ] && [ ! -L "$RUNS/$tag/ckpts" ]; then
        mv "$RUNS/$tag/ckpts" "$ARCHIVE/$tag-ckpts" && ln -s "$ARCHIVE/$tag-ckpts" "$RUNS/$tag/ckpts"
      fi
      rm -f "$RUNS/std_matrix/.running_$tag"
    fi
  done

  # 2. dispatch while < MAX_INFLIGHT and free GPU available
  for j in "${JOBS[@]}"; do
    IFS='|' read -r tag data factor width lam <<< "$j"
    [ -f "$RUNS/std_matrix/.curved_$tag" ] && continue
    [ -f "$RUNS/std_matrix/.running_$tag" ] && continue
    [ "$(inflight)" -ge "$MAX_INFLIGHT" ] && break
    g=$(free_gpu) || continue
    touch /home/project2/RELEASE-$g
    touch "$RUNS/std_matrix/.running_$tag"
    echo "[stdq] launch $tag on gpu$g lam=$lam factor=$factor width=$width $(date '+%H:%M')" >> "$LOG"
    RUNS_ROOT="$RUNS" RUNROOT="$CODE" CONDA_ENV_BIN="$PYBIN" \
    GSPLAT_ROOT=/home/T0ng/gsplat-main WAIT_VRAM_MB=30000 \
    setsid bash "$CODE/scripts/runner_phg_cell.sh" "$g" "$tag" "$data" "$lam" "$tag" 30000 15000 \
      --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
      --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
      --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
      --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
      --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85 \
      --cfg.data.data-factor "$factor" --cfg.data.max-width "$width" \
      >> "$RUNS/std_matrix/$tag.launch.log" 2>&1 < /dev/null &
    sleep 120   # let it allocate VRAM before counting it in
  done

  # 3. completion: all jobs curved
  n=0
  for j in "${JOBS[@]}"; do
    [ -f "$RUNS/std_matrix/.curved_${j%%|*}" ] && n=$((n+1))
  done
  [ "$n" -ge ${#JOBS[@]} ] && { echo "[stdq] ALL_DONE $(date)" >> "$LOG"; exit 0; }

  sleep 300
done
