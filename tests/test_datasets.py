"""CPU-only tests for dataset sizing helpers."""

from scaffold_gs.datasets import max_width_size


def test_max_width_size_official_rule():
    # 3795x2134 -> 1600x899, matching HAC++ resolution=-1 (int truncation).
    assert max_width_size(3795, 2134, 1600) == (1600, 899)
    # No upscaling when already smaller than the cap.
    assert max_width_size(1200, 800, 1600) == (1200, 800)
    assert max_width_size(1600, 899, 1600) == (1600, 899)
