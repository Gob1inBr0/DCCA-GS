#!/usr/bin/env bash
# Fork-framework multi-GPU queue for LYH. Waits for the current training
# batch to finish (~4h from 2026-10-13 02:40), then dispatches the seven
# fork lines onto whatever GPUs free up, respecting dependencies:
#   spine (30k, archives at 15000/24000/30000)
#     -> after ckpt_15000: C0_15000 (empty ctrl), E1 (penalty early), E2 (p0r early)
#     -> after ckpt_24000: C0_24000 (empty ctrl), C1 (penalty current), C2 (p0r current)
#   each arm's 150-view layered curve runs on a free GPU right after its
#   decoded_eval lands.
# Dispatch rule: poll every 10 min; one pending line per free GPU; a GPU is
# free when its used VRAM < 2 GB. Before dispatch we touch RELEASE-<gpu> so
# the gpu_keeper yields the card; keeper back-fills after our task ends.
set -uo pipefail

CODE=/home/project2/DCCA-GS
RUNS=/home/project2/dcca_runs/fork
DATA=/dev/shm/octree_178
PYBIN=/home/project2/miniconda3/envs/DCCA/bin
SCHED=$RUNS/camera_schedule_42.npy
GPUS_ALL="0 1 2 3 4 5 6 7"
mkdir -p "$RUNS" "$RUNS/logs"
LOG=$RUNS/queue.log

# Already-running guard: never double-start (2026-10-01 lesson).
[ -f "$RUNS/.queue_running" ] && { echo "queue already running" >&2; exit 1; }
touch "$RUNS/.queue_running"
trap 'rm -f "$RUNS/.queue_running"' EXIT

free_gpus() {
  local out=""
  for g in $GPUS_ALL; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "${used:-99999}" -lt 2000 ] && out="$out $g"
  done
  echo $out
}

launch() {  # launch <tag> <gpu> [extra flags...]
  local tag=$1 gpu=$2; shift 2
  RUNS_ROOT=/home/project2/dcca_runs RUNROOT="$CODE" CONDA_ENV_BIN="$PYBIN" \
  GSPLAT_ROOT=/home/T0ng/gsplat-main WAIT_VRAM_MB=30000 \
  setsid bash "$CODE/scripts/runner_phg_cell.sh" "$gpu" 1-78 "$DATA" 0.004 "$tag" 30000 15000 \
    --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
    --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
    --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
    --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
    --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85 \
    --cfg.camera-schedule "$SCHED" "$@" \
    >> "$RUNS/logs/$tag.launch.log" 2>&1 < /dev/null &
  echo "[queue] launched $tag on gpu$gpu $(date '+%H:%M')" >> "$LOG"
}

layered() {  # layered <tag> <gpu>
  local tag=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$CODE" \
    "$PYBIN/python" "$CODE/scripts/full_view_eval.py" \
    --run "/home/project2/dcca_runs/$tag" --data-dir "$DATA" \
    --data-factor 1 --max-width 1600 \
    --out "/home/project2/dcca_runs/$tag/layered_curve_150v.json" \
    >> "$RUNS/logs/$tag.layered.log" 2>&1 < /dev/null &
  echo "[queue] layered $tag on gpu$gpu $(date '+%H:%M')" >> "$LOG"
}

# state files: one marker per dispatched line
mark() { touch "$RUNS/.done_$1"; }
donep() { [ -f "$RUNS/.done_$1" ]; }

# camera schedule needs the dataset present
"$PYBIN/python" "$CODE/scripts/gen_camera_schedule.py" \
  --n-cams 1050 --steps 30000 --seed 42 --out "$SCHED" >> "$LOG" 2>&1

echo "[queue] fork queue start $(date)" >> "$LOG"
while true; do
  FREE=$(free_gpus)

  take_gpu() {  # pop first free gpu, touch RELEASE, echo it (empty if none)
    local g
    for g in $FREE; do
      echo "$g"; return 0
    done
    return 1
  }

  # 1. spine
  if ! donep spine; then
    g=$(take_gpu) || g=""
    if [ -n "$g" ]; then
      touch /home/project2/RELEASE-$g
      mark spine
      # launch_spine binds the GPU by rewriting CUDA_VISIBLE_DEVICES
      RUNS_ROOT=/home/project2/dcca_runs RUNROOT="$CODE" CONDA_ENV_BIN="$PYBIN" \
      CUDA_VISIBLE_DEVICES=$g setsid bash -c "
        export PATH=\"$PYBIN:\$PATH\"
        export PYTHONPATH=/home/T0ng/gsplat-main:$CODE
        cd $CODE
        python train.py train \
          --cfg.model.model-name hac_pp \
          --cfg.data.data-dir $DATA \
          --cfg.data.result-dir $RUNS/fork_spine_l0004_s42 \
          --cfg.data.data-factor 1 --cfg.data.max-width 1600 --cfg.data.test-every 8 \
          --cfg.data.no-preload-images \
          --cfg.model.voxel-size 0.001 --cfg.model.feat-dim 32 --cfg.model.n-offsets 10 \
          --cfg.model.appearance-dim 0 --cfg.model.ratio 1 --cfg.model.tile-size 32 \
          --cfg.model.content-aware-start-iter 20000 --cfg.model.content-aware-ramp-iters 10000 \
          --cfg.model.mlp-complexity-hidden 32 --cfg.model.mlp-complexity-layers 1 \
          --cfg.model.sensitivity-enabled --cfg.model.sensitivity-start-iter 20000 \
          --cfg.model.sensitivity-weight 0.001 \
          --cfg.optim.max-steps 30000 \
          --cfg.optim.eval-steps 30000 \
          --cfg.optim.save-steps 15000 24000 30000 \
          --cfg.optim.lambda-rate 0.004 --cfg.optim.mask-lr-final 0.002 \
          --cfg.optim.start-stat 500 --cfg.optim.update-from 1500 \
          --cfg.optim.update-until 15000 --cfg.optim.update-interval 100 \
          --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
          --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
          --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
          --cfg.model.mini-splat-voxel 0.0 --cfg.seed 42 \
          --cfg.model.sensitivity-start-iter 0 --cfg.model.spa-post-ratio 0.85 \
          --cfg.camera-schedule $SCHED \
          > $RUNS/logs/fork_spine_l0004_s42.log 2>&1
      " < /dev/null &
      echo "[queue] launched spine on gpu$g $(date '+%H:%M')" >> "$LOG"
      sleep 60   # let it allocate before assigning more
      FREE=$(free_gpus)
    fi
  fi

  SPINE_CKPT_15=$RUNS/fork_spine_l0004_s42/ckpts/ckpt_15000.pth
  SPINE_CKPT_24=$RUNS/fork_spine_l0004_s42/ckpts/ckpt_24000.pth

  # 2. 15000-fork arms
  if [ -f "$SPINE_CKPT_15" ]; then
    for spec in "fork_c0_15000:" "fork_e1_pen_15000:--cfg.model.coarse-ladder-align --cfg.model.coarse-ladder-weight 0.05 --cfg.model.coarse-ladder-start-iter 15000" \
                "fork_e2_p0r_15000:--cfg.model.p0-render-loss --cfg.model.p0-render-weight 0.05 --cfg.model.p0-render-start-iter 15000 --cfg.model.p0-render-ramp-iters 15000"; do
      tag=${spec%%:*}
      flags=${spec#*:}
      if ! donep "$tag"; then
        g=$(take_gpu) || g=""
        if [ -n "$g" ]; then
          touch /home/project2/RELEASE-$g
          mark "$tag"
          launch "$tag" "$g" --cfg.resume "$SPINE_CKPT_15" $flags
          sleep 60
          FREE=$(free_gpus)
        fi
      fi
    done
  fi

  # 3. 24000-fork arms
  if [ -f "$SPINE_CKPT_24" ]; then
    for spec in "fork_c0_24000:" "fork_c1_pen_24000:--cfg.model.coarse-ladder-align --cfg.model.coarse-ladder-weight 0.05" \
                "fork_c2_p0r_24000:--cfg.model.p0-render-loss --cfg.model.p0-render-weight 0.05"; do
      tag=${spec%%:*}
      flags=${spec#*:}
      if ! donep "$tag"; then
        g=$(take_gpu) || g=""
        if [ -n "$g" ]; then
          touch /home/project2/RELEASE-$g
          mark "$tag"
          launch "$tag" "$g" --cfg.resume "$SPINE_CKPT_24" $flags
          sleep 60
          FREE=$(free_gpus)
        fi
      fi
    done
  fi

  # 4. layered curves for finished arms
  for tag in fork_spine_l0004_s42 fork_c0_15000 fork_e1_pen_15000 fork_e2_p0r_15000 \
             fork_c0_24000 fork_c1_pen_24000 fork_c2_p0r_24000; do
    [ "$tag" = "fork_spine_l0004_s42" ] && continue   # spine needs no bitstream
    if [ -f "/home/project2/dcca_runs/$tag/decoded_eval/metrics.jsonl" ] && \
       [ ! -f "/home/project2/dcca_runs/$tag/layered_curve_150v.json" ] && \
       [ ! -f "$RUNS/.layered_running_$tag" ]; then
      g=$(take_gpu) || g=""
      if [ -n "$g" ]; then
        touch "$RUNS/.layered_running_$tag"
        layered "$tag" "$g"
      fi
    fi
  done

  # 5. completion: six arm curves done -> exit
  n=0
  for tag in fork_c0_15000 fork_e1_pen_15000 fork_e2_p0r_15000 \
             fork_c0_24000 fork_c1_pen_24000 fork_c2_p0r_24000; do
    [ -f "/home/project2/dcca_runs/$tag/layered_curve_150v.json" ] && n=$((n+1))
  done
  if [ "$n" -ge 6 ]; then
    echo "[queue] ALL_DONE $(date)" >> "$LOG"
    exit 0
  fi

  sleep 600
done
