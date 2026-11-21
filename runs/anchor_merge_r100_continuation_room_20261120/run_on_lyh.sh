#!/usr/bin/env bash
set -euo pipefail
root=$1
runs=$(dirname "$(dirname "$root")")
sweep="$runs/anchor_merge_response_sweep_room_20261120/attempt_01"
original="$runs/anchor_merge_inside_30k_room_20261120/attempt_01"
environment="$runs/anchor_merge_new_position_room_20261120/attempt_02"
python="$(dirname "$(dirname "$runs")")/GrowToCode/envs/pcgs/bin/python"
ckpt="$sweep/r100/ckpts/ckpt_30000.pth"
schedule="$original/camera_schedule.npy"
runner="$root/measure_anchor_merge_continuation.py"
locks="$runs/.experiment-gpu-locks"
claim=

on_exit() {
    status=$?
    if [[ -n "$claim" ]]; then
        rm -f "$claim/owner"
        rmdir "$claim"
    fi
    printf '%s\n' "$status" > "$root/continuation.exit"
}
trap on_exit EXIT
test "$(cat "$sweep/sweep.exit")" = 0
test -x "$python"
test -x "$environment/bin/tmc3"
cd "$sweep/source"
awk 'NF == 2 && length($1) == 64' "$sweep/provenance.txt" | sha256sum --check
{
    date -Is
    cat base-revision.txt
    printf '%s\n' "$PWD" "$python" "$environment"
    sha256sum "$ckpt" "$schedule" "$runner" "$root/run_on_lyh.sh"
} > "$root/provenance.txt"

mkdir -p "$locks" "$root/tmp"
export PYTHONPATH="$PWD:$environment/deps"
export PATH="$environment/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="$environment/torch_extensions" TMPDIR="$root/tmp"

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
            printf '%s %s\n' "$root" "$BASHPID" > "$claim/owner"
            printf 'gpu=%s free_mib=%s\n' "$gpu" "$free" > "$root/continuation.gpu"
            export CUDA_VISIBLE_DEVICES=$uuid
            break
        fi
    done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits |
             sort -t, -k3,3nr)
    [[ -n "$claim" ]] || sleep 30
done

"$python" -u "$runner" "$root" train "$ckpt" "$schedule" > "$root/train.log" 2>&1
for step in 31000 32000 33000 34000 35000; do
    "$python" -u "$runner" "$root" decode "$step" > "$root/decode_$step.log" 2>&1
done
