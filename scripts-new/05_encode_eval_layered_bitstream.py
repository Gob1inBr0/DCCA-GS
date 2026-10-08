#!/usr/bin/env python3
"""Layered-bitstream encode+eval entry; delegates to scripts/c25_real_bitstream.py.

scripts/c25_real_bitstream.py is the single implementation of the ZC-aware
layered coder (with scripts/zc_context.py). This wrapper only keeps the
numbered scripts-new entry alive so documented commands
(01_eval_layered_bitstream.sh, RUN_RECORD examples) keep working without a
second full copy drifting out of sync.
"""
import os
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts" / "c25_real_bitstream.py"
os.execv(sys.executable, [sys.executable, str(TARGET), *sys.argv[1:]])
