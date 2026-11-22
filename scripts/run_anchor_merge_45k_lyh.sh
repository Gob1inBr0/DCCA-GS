#!/usr/bin/env bash
# Continue the corrected 64-neighbor model and its matched control to 45k.
set -euo pipefail
root=$1
runs=$(dirname "$(dirname "$root")")
base=$(dirname "$(dirname "$runs")")
environment="$runs/anchor_merge_new_position_room_20261120/attempt_02"
python="$base/GrowToCode/envs/pcgs/bin/python"
schedule="$runs/anchor_merge_inside_30k_room_20261120/attempt_01/camera_schedule.npy"
merged="$runs/anchor_merge_neighbors_room_20261122/attempt_01/no_minimum/ckpts/ckpt_30000.pth"
control="$runs/anchor_merge_saving_filter_room_20261120/attempt_02/control/ckpts/ckpt_30000.pth"
locks="$runs/.experiment-gpu-locks"
trap 'printf "%s\n" "$?" > "$root/experiment.exit"' EXIT
test -x "$python"
test -x "$environment/bin/tmc3"
test -s "$merged"
test -s "$control"
test -s "$schedule"
mkdir -p "$locks" "$root/tmp"
cd "$root/source"
test -z "$(git status --porcelain)"
{
    date -Is
    git rev-parse HEAD
    printf '%s\n' "$python" "$environment"
    sha256sum "$merged" "$control" "$schedule"
    sha256sum scripts/{run_anchor_merge_45k_lyh.sh,measure_anchor_merge_continuation.py} \
        scaffold_gs/{trainer,hacpp,config}.py
} > "$root/provenance.txt"
export PYTHONPATH="$PWD:$environment/deps"
export PATH="$environment/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="$environment/torch_extensions" TMPDIR="$root/tmp"

run_stage() (
    stage=$1
    source_checkpoint=$2
    claim=
    release() {
        status=$?
        if [[ -n "$claim" ]]; then
            rm -f "$claim/owner"
            rmdir "$claim"
        fi
        printf '%s\n' "$status" > "$root/$stage.exit"
    }
    trap release EXIT
    while [[ -z "$claim" ]]; do
        while IFS=, read -r gpu uuid free; do
            read -r gpu <<< "$gpu"
            read -r uuid <<< "$uuid"
            read -r free <<< "$free"
            if (( gpu < 4 || gpu > 7 || free < 14000 )); then continue; fi
            if mkdir "$locks/gpu-$gpu" 2>/dev/null; then
                claim="$locks/gpu-$gpu"
                printf '%s %s\n' "$root/$stage" "$BASHPID" > "$claim/owner"
                printf 'gpu=%s free_mib=%s\n' "$gpu" "$free" > "$root/$stage.gpu"
                export CUDA_VISIBLE_DEVICES=$uuid
                printf 'VRAM_OK stage=%s gpu=%s free_mib=%s %s\n' "$stage" "$gpu" "$free" "$(date -Is)"
                break
            fi
        done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits |
                 sort -t, -k3,3nr)
        [[ -n "$claim" ]] || sleep 30
    done
    out="$root/$stage"
    if [[ "$stage" == smoke ]]; then
        mkdir "$out"
        "$python" - "$source_checkpoint" "$out" <<'PY'
import sys
from pathlib import Path
import torch
from scripts.run_in_training_anchor_merge import smoke_reference
smoke_reference(torch.load(sys.argv[1], map_location="cpu", weights_only=False), Path(sys.argv[2]))
PY
        source_checkpoint="$out/smoke_reference.pth"
        end=30400
        interval=400
    else
        end=45000
        interval=2500
    fi
    "$python" -u scripts/measure_anchor_merge_continuation.py "$out" train \
        "$source_checkpoint" "$schedule" --end-step "$end" --eval-interval "$interval" \
        > "$root/$stage.train.log" 2>&1
    "$python" - "$source_checkpoint" "$out/train/ckpts/ckpt_$end.pth" "$out" <<'PY'
import json
import sys
from pathlib import Path
from scripts.compare_anchor_merge_saving_filter import verify_updates
result = verify_updates(Path(sys.argv[1]), Path(sys.argv[2]))
Path(sys.argv[3], "updates.json").write_text(json.dumps(result, indent=2) + "\n")
PY
    for (( step=30000+interval; step<=end; step+=interval )); do
        "$python" -u scripts/measure_anchor_merge_continuation.py "$out" decode "$step" \
            > "$root/$stage.decode_$step.log" 2>&1
    done
    date -Is > "$out/finished.txt"
)

run_stage smoke "$merged"
run_stage merged64 "$merged" > "$root/merged64.launch.log" 2>&1 &
merged_pid=$!
run_stage control "$control" > "$root/control.launch.log" 2>&1 &
control_pid=$!
merged_status=0
control_status=0
wait "$merged_pid" || merged_status=$?
wait "$control_pid" || control_status=$?
(( merged_status == 0 && control_status == 0 ))
