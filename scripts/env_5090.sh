#!/bin/bash
# 5090 PHG runtime environment (source, not execute):
#
#   conda activate HAC_5090_a100
#   source scripts/env_5090.sh
#
# Sets PYTHONPATH (repo root), PATH (conda env bin, contains tmc3/GPCC),
# PYTHONNOUSERSITE (ignore ~/.local site-packages) and
# PYTORCH_CUDA_ALLOC_CONF (expandable segments for long trainings).

export PHG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PHG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="$HOME/miniconda3/envs/HAC_5090_a100/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[env_5090] PHG_ROOT=$PHG_ROOT"
echo "[env_5090] python=$(command -v python) ($(python -V 2>&1))"
