#!/usr/bin/env bash
# Train -> compress -> decoded-eval runner; delegates to scripts/runner_phg_cell.sh.
#
# scripts/runner_phg_cell.sh is the single implementation. This wrapper keeps
# the numbered scripts-new entry and the 00_train_dcca.sh call path. Usage and
# environment variables (RUNROOT, RUNS_ROOT, CONDA_ENV_BIN, GSPLAT_ROOT,
# WAIT_VRAM_MB) are identical to the original runner.
set -euo pipefail

RUNROOT="${RUNROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RUNROOT
exec bash "$RUNROOT/scripts/runner_phg_cell.sh" "$@"
