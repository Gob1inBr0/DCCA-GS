"""Compatibility shim: the real implementation is scripts/zc_context.py.

Kept so `from lib_zc_context import ...` keeps working for anything written
against this name. Unit tests live in tests/test_zc_context.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zc_context import (  # noqa: E402,F401
    build_zc_anchor_context,
    cell_ids,
    combine_zc_context,
    quantile_buckets,
    resolve_cell_size,
)
