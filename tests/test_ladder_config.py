"""Tests for the --ladder spec parser (c25_real_bitstream.parse_ladder).

Run: python -m pytest tests/test_ladder_config.py -q  (CPU only, <1s)

The parser is the guardrail for the 16x base-layer sweep: any ladder that
breaks the telescoping reconstruction (non-power-of-two steps, non-strict
decrease, missing full-precision end) must be rejected before an encode
starts, not discovered as a wrong curve afterwards.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from c25_real_bitstream import parse_ladder


def test_valid_specs():
    assert parse_ladder("feat=16,8,4,2,1") == {"feat": [16, 8, 4, 2, 1]}
    assert parse_ladder("feat=16,8,4,2,1 offset=16,8,4,2,1 scaling=2,1") == {
        "feat": [16, 8, 4, 2, 1],
        "offset": [16, 8, 4, 2, 1],
        "scaling": [2, 1],
    }
    # ratio > 2 between consecutive steps is legal (4x refinement jump)
    assert parse_ladder("feat=16,4,2,1") == {"feat": [16, 4, 2, 1]}
    # single-step ladder (= uniform full precision only) is legal
    assert parse_ladder("scaling=1") == {"scaling": [1]}


def test_rejects_unknown_field():
    with pytest.raises(SystemExit, match="unknown field"):
        parse_ladder("opacity=8,4,1")


def test_rejects_missing_full_precision_end():
    with pytest.raises(SystemExit, match="end at step 1"):
        parse_ladder("feat=16,8,4,2")


def test_rejects_non_power_of_two():
    with pytest.raises(SystemExit, match="powers of two"):
        parse_ladder("feat=12,4,2,1")


def test_rejects_non_decreasing():
    with pytest.raises(SystemExit, match="strictly decrease"):
        parse_ladder("feat=8,8,4,2,1")
    with pytest.raises(SystemExit, match="strictly decrease"):
        parse_ladder("feat=8,16,4,2,1")


def test_rejects_malformed_part():
    with pytest.raises(SystemExit, match="expected field=steps"):
        parse_ladder("feat16,8,1")


def test_all_fields_present_roundtrip():
    spec = ("feat=16,8,4,2,1 offset=16,8,4,2,1 scaling=4,2,1")
    out = parse_ladder(spec)
    assert set(out) == {"feat", "offset", "scaling"}
    for f, steps in out.items():
        assert steps[-1] == 1
        assert all(x & (x - 1) == 0 for x in steps)
