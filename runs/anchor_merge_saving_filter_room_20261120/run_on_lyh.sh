#!/usr/bin/env bash
set -euo pipefail
root=$1
runs=$(dirname "$(dirname "$root")")
base=$(dirname "$(dirname "$runs")")
original="$runs/anchor_merge_inside_30k_room_20261120/attempt_01"
sweep="$runs/anchor_merge_response_sweep_room_20261120/attempt_01"
environment="$runs/anchor_merge_new_position_room_20261120/attempt_02"
python="$base/GrowToCode/envs/pcgs/bin/python"
shared="$original/common/ckpts/ckpt_18000.pth"
schedule="$original/camera_schedule.npy"
locks="$runs/.experiment-gpu-locks"
trap 'printf "%s\n" "$?" > "$root/experiment.exit"' EXIT

test "$(cat "$original/common.exit")" = 0
test -x "$python"
test -x "$environment/bin/tmc3"
mkdir -p "$locks" "$root/tmp"
cd "$root/source"
{
    date -Is
    cat base-revision.txt
    printf '%s\n' "$python" "$sweep/source" "$environment"
    sha256sum "$shared" "$schedule" "$root/run_on_lyh.sh"
    sha256sum scaffold_gs/{anchor_merge,config,hacpp,trainer}.py \
        scripts/{run_in_training_anchor_merge,compare_anchor_merge_response,compare_anchor_merge_saving_filter}.py
} > "$root/provenance.txt"
export PYTHONPATH="$PWD:$environment/deps"
export PATH="$environment/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="$environment/torch_extensions" TMPDIR="$root/tmp"

run_stage() (
    stage=$1
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
            if (( gpu < 4 || gpu > 7 || free < 14000 )); then
                continue
            fi
            if mkdir "$locks/gpu-$gpu" 2>/dev/null; then
                claim="$locks/gpu-$gpu"
                printf '%s %s\n' "$root/$stage" "$BASHPID" > "$claim/owner"
                printf 'gpu=%s free_mib=%s\n' "$gpu" "$free" > "$root/$stage.gpu"
                export CUDA_VISIBLE_DEVICES=$uuid
                break
            fi
        done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits |
                 sort -t, -k3,3nr)
        [[ -n "$claim" ]] || sleep 30
    done
    "$python" -u scripts/compare_anchor_merge_saving_filter.py "$stage" "$root" "$shared" "$schedule" \
        > "$root/$stage.log" 2>&1
)

run_stage smoke
pids=()
for stage in control minimum_one_byte no_minimum; do
    run_stage "$stage" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
