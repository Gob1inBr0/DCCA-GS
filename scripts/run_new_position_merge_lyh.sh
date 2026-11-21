#!/usr/bin/env bash
set -euo pipefail

# The caller provides site-specific locations and a fresh experiment root.
source_dir=$1
run_dir=$2
python_bin=$3
scene_dir=$4
reference=$5
metadata=$6
shared_locks=$7

cd "$source_dir"
export PYTHONPATH="$source_dir:$run_dir/deps"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MAX_JOBS=3
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="$run_dir/torch_extensions"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$run_dir/bin:$(dirname "$python_bin"):/usr/local/bin:$PATH"
export TMPDIR="$run_dir/tmp"
mkdir -p "$TMPDIR" "$shared_locks" "$run_dir/bin"

phase=setup
lock=
finish() {
    status=$?
    trap - EXIT
    if [[ -n "$lock" ]]; then
        rm -f "$lock/owner"
        rmdir "$lock"
    fi
    printf '{"phase":"%s","exit":%d}\n' "$phase" "$status" > "$run_dir/exit.json"
    exit "$status"
}
trap finish EXIT

printf '%s\n' "python=$python_bin" "scene=$scene_dir" "checkpoint=$reference"     "metadata=$metadata" "started=$(date -Is)" > "$run_dir/provenance.txt"
cat base-revision.txt >> "$run_dir/provenance.txt"
sha256sum "$reference" "$metadata" scripts/merge_anchors_new_position.py \
    scripts/run_new_position_merge_lyh.sh >> "$run_dir/provenance.txt"

# Install only Python dependencies in the run directory. Reuse the existing
# PyTorch and HAC CUDA extensions from the selected Python environment.
if [[ ! -d "$run_dir/deps/gsplat" ]]; then
    "$python_bin" -m pip install --disable-pip-version-check --target "$run_dir/deps" \
        --no-deps pycolmap==3.13.0 gsplat==1.5.3 torchmetrics==1.4.3 \
        lightning-utilities jaxtyping wadler-lindig imageio rich markdown-it-py \
        pygments mdurl tqdm \
        >> "$run_dir/dependencies.log" 2>&1
fi
"$python_bin" - <<'PY'
import torch
import _gridencoder
import arithmetic
import torch_scatter
import pycolmap
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from gsplat.rendering import rasterization
print("[Environment]", torch.__version__, torch.version.cuda, pycolmap.__version__, flush=True)
PY
tmc_build=
if ! command -v tmc3 >/dev/null; then
    (
        release=release-v23.0-rc2
        archive="$run_dir/tmc13.tar.gz"
        src="$run_dir/tmc13"
        curl -fSL --retry 2 --connect-timeout 10 --max-time 120 \
            "https://codeload.github.com/MPEGGroup/mpeg-pcc-tmc13/tar.gz/refs/tags/$release" \
            -o "$archive"
        mkdir "$src"
        tar -xf "$archive" --strip-components=1 -C "$src"
        cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$src/build" -j 4 --target tmc3
        install -m 755 "$src/build/tmc3/tmc3" "$run_dir/bin/tmc3"
        sha256sum "$archive" "$run_dir/bin/tmc3"
    ) > "$run_dir/tmc3.log" 2>&1 &
    tmc_build=$!
fi

phase=prepare_checkpoints
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/merge_anchors_new_position.py \
    "$reference" "$metadata" "$scene_dir" "$run_dir/checkpoints" \
    > "$run_dir/merge.log" 2>&1
cat "$run_dir/checkpoints/merge.json"

phase=waiting_for_gpu
echo '[Queue] looking for sufficient free memory on GPUs 4–7'
while [[ -z "$lock" ]]; do
    while IFS=, read -r gpu uuid free; do
        gpu=${gpu// /}
        uuid=${uuid// /}
        free=${free// /}
        if (( gpu < 4 || gpu > 7 || free < 14000 )); then
            continue
        fi
        candidate="$shared_locks/gpu-$gpu"
        if ! mkdir "$candidate" 2>/dev/null; then
            continue
        fi
        current_free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        if (( current_free < 14000 )); then
            rmdir "$candidate"
            continue
        fi
        lock=$candidate
        printf '%s %s %s\n' "$run_dir" "$$" "$uuid" > "$lock/owner"
        export CUDA_VISIBLE_DEVICES=$uuid
        printf 'gpu=%s (%s)\n' "$gpu" "$uuid" >> "$run_dir/provenance.txt"
        echo "[Queue] claimed GPU $gpu with $current_free MiB free"
        break
    done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits |
             sort -t, -k3,3nr)
    if [[ -z "$lock" ]]; then
        sleep 30
    fi
done

train() {
    local arm=$1
    local steps=$2
    local checkpoint=$3
    local out="$run_dir/$arm"
    mkdir -p "$out"
    "$python_bin" -u - "$checkpoint" "$out" "$steps" > "$out/train.log" 2>&1 <<'PY'
from pathlib import Path
import sys

import torch

from scaffold_gs.config import DataConfig, ModelConfig, OptimConfig, TrainConfig
from scaffold_gs.trainer import run_training

path, out_dir, steps = sys.argv[1:]
archive = torch.load(path, map_location="cpu", weights_only=False)
final_step = int(archive["iteration"]) + int(steps)
model = ModelConfig(**archive["model_config"])
data = DataConfig(**archive["data_config"])
data.result_dir = out_dir
data.preload_images = False
optim = OptimConfig(**archive["optim_config"])
optim.max_steps = final_step
optim.eval_steps = [final_step]
optim.save_steps = [final_step]
config = TrainConfig(model=model, data=data, optim=optim, seed=42, resume=path)
print(f"[Experiment] start={archive['iteration']} end={final_step} "
      f"start_anchors={len(archive['model_state']['_anchor'])}", flush=True)
run_training(config)
assert (Path(out_dir) / "ckpts" / f"ckpt_{final_step}.pth").is_file()
PY
}

encode_and_evaluate() {
    local arm=$1
    local checkpoint=$2
    local out="$run_dir/$arm"
    mkdir -p "$out"
    "$python_bin" -u - "$checkpoint" "$out" > "$out/compress.log" 2>&1 <<'PY'
from pathlib import Path
import sys

from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.trainer import load_checkpoint

checkpoint, out = sys.argv[1:]
model, _, _, _ = load_checkpoint(checkpoint, "cuda")
print(HACPlusCodec().encode(model, Path(out) / "bitstreams"), flush=True)
PY
    "$python_bin" -u scripts/eval_decoded.py --artifact-dir "$out/bitstreams" \
        --data-dir "$scene_dir" --result-dir "$out/decoded_eval" \
        --data-factor 2 --test-every 8 --max-width 1600 --no-preload-images \
        > "$out/decoded_eval.log" 2>&1
    test -s "$out/decoded_eval/metrics.jsonl"
}

phase=smoke
train smoke 400 "$run_dir/checkpoints/merged.pth"
if [[ -n "$tmc_build" ]]; then
    wait "$tmc_build"
fi
encode_and_evaluate merged_initial "$run_dir/checkpoints/merged.pth"
encode_and_evaluate control_initial "$run_dir/checkpoints/control.pth"
encode_and_evaluate smoke "$run_dir/smoke/ckpts/ckpt_30400.pth"

phase=paired_training
for arm in control merged; do
    train "$arm" 1500 "$run_dir/checkpoints/$arm.pth"
    encode_and_evaluate "$arm" "$run_dir/$arm/ckpts/ckpt_31500.pth"
done
phase=complete
