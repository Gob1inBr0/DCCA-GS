#!/usr/bin/env python3
"""Decoded-bitstream render eval entry; delegates to scripts/eval_decoded.py.

Single implementation lives in scripts/eval_decoded.py; this wrapper keeps
the numbered scripts-new entry used by 06_internal_runner_phg_cell.sh.
"""
import os
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts" / "eval_decoded.py"
os.execv(sys.executable, [sys.executable, str(TARGET), *sys.argv[1:]])
