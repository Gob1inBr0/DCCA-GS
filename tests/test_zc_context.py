import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zc_context import (build_zc_anchor_context, combine_zc_context,
                        quantile_buckets, resolve_cell_size)


def test_quantile_buckets_constant_and_ordered():
    assert quantile_buckets(np.ones(5)).tolist() == [0, 0, 0, 0, 0]
    b = quantile_buckets(np.arange(9), 3)
    assert set(b.tolist()) == {0, 1, 2}
    assert np.all(b[:-1] <= b[1:])


def test_resolve_cell_size_auto_is_geometry_deterministic():
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    size, rule = resolve_cell_size(xyz, requested=0.0, divisor=10)
    assert np.isclose(size, 0.5)
    assert rule == "bbox_diag/10"
    fixed, fixed_rule = resolve_cell_size(xyz, requested=0.25)
    assert fixed == 0.25 and fixed_rule == "fixed"


def test_build_zc_anchor_context_roots_and_density():
    xyz = np.array([
        [0.00, 0.00, 0.00],
        [0.01, 0.00, 0.00],
        [1.00, 0.00, 0.00],
        [2.00, 0.00, 0.00],
        [2.01, 0.00, 0.00],
        [2.02, 0.00, 0.00],
    ])
    ctx = build_zc_anchor_context(xyz, cell_size=0.5)
    assert ctx["root_anchor"].tolist() == [0, 0, 2, 3, 3, 3]
    assert ctx["density_bucket"][3] >= ctx["density_bucket"][2]
    assert ctx["num_cells"] == 3


def test_combine_zc_context_cardinality():
    coarse = np.array([0, 1, 2], dtype=np.int16)
    density = np.array([0, 1, 2], dtype=np.int16)
    root = np.array([2, 1, 0], dtype=np.int16)
    off, card = combine_zc_context(coarse, "off", density, root)
    assert card == 1 and off.tolist() == coarse.tolist()
    both, card = combine_zc_context(coarse, "density-root", density, root)
    assert card == 9
    assert both.tolist() == [
        0 * 9 + (0 * 3 + 2),
        1 * 9 + (1 * 3 + 1),
        2 * 9 + (2 * 3 + 0),
    ]
