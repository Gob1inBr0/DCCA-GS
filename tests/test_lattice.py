"""CPU tests for the I5 lattice quantizers and dithering."""

import torch

from scaffold_gs.lattice import (
    lattice_ste,
    make_dither,
    quantize_lattice,
    vq_dequantize,
    vq_quantize,
)


def _brute_grid(dims: int, rng: int = 3):
    vals = torch.arange(-rng, rng + 1, dtype=torch.float32)
    grid = torch.stack(
        torch.meshgrid(*([vals] * dims), indexing="ij"), dim=-1
    ).reshape(-1, dims)
    return grid


def test_d4_nearest_and_parity():
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    coords, point = quantize_lattice(x, "d4")
    assert torch.allclose(coords, point)
    assert (coords.sum(dim=-1) % 2 == 0).all()
    grid = _brute_grid(4, rng=5)
    grid = grid[(grid.sum(-1) % 2 == 0)]  # D4 = even-sum integer vectors
    dist = ((x[:, None, :] - grid[None]) ** 2).sum(-1)
    for i in range(16):
        best_dist = dist[i].min()
        assert (x[i] - point[i]).pow(2).sum() <= best_dist + 1e-4


def test_e8_parity_and_repr():
    torch.manual_seed(1)
    x = torch.randn(48, 8)
    coords, point = quantize_lattice(x, "e8")
    assert torch.allclose(point * 2.0, coords)
    # E8 invariant in the doubled representation: sum(coords) % 4 == 0.
    assert (coords.sum(dim=-1) % 4 == 0).all()
    # The integer candidate must be no worse than half-integer candidates.
    grid = _brute_grid(8, rng=2)
    half = grid + 0.5
    grid = torch.cat(
        [grid[(grid.sum(-1) % 2 == 0)], half[(half.sum(-1) % 2 == 0)]], dim=0
    )  # E8 = even-sum integers plus even-sum half-integers
    dist = ((x[:, None, :] - grid[None]) ** 2).sum(-1)
    for i in range(8):
        assert (x[i] - point[i]).pow(2).sum() <= dist[i].min() + 1e-4


def test_z_rounds():
    x = torch.tensor([[0.4, 0.6, -0.7]])
    coords, point = quantize_lattice(x, "z")
    assert torch.equal(coords, torch.tensor([[0.0, 1.0, -1.0]]))


def test_dither_reproducible():
    idx = torch.arange(5, dtype=torch.int64)
    a = make_dither(7, idx, 1, 4, torch.device("cpu"))
    b = make_dither(7, idx, 1, 4, torch.device("cpu"))
    c = make_dither(8, idx, 1, 4, torch.device("cpu"))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert (a >= -0.5).all() and (a < 0.5).all()


def test_vq_roundtrip_no_dither():
    x = torch.randn(32, 10)
    step = torch.ones(32, 10)
    anchor_idx = torch.arange(32, dtype=torch.int64)
    values, q_eff, coords = vq_quantize(
        x, step, "d4", 4, 0, anchor_idx, 0, False
    )
    back = vq_dequantize(values, q_eff, step, "d4", 4, 0, anchor_idx, 0, False)
    assert torch.equal(back, values)
    assert (values - x).abs().max() < 1.0
    # leftover dims are scalar-quantized: values are multiples of the step.
    assert torch.allclose(values[:, 8:], torch.round(x[:, 8:]))


def test_vq_roundtrip_dither_and_seed():
    torch.manual_seed(3)
    x = torch.randn(32, 10)
    step = torch.full((32, 10), 0.5)
    anchor_idx = torch.arange(32, dtype=torch.int64)
    values, q_eff, coords = vq_quantize(
        x, step, "d4", 4, 11, anchor_idx, 0, True
    )
    back = vq_dequantize(values, q_eff, step, "d4", 4, 11, anchor_idx, 0, True)
    assert (back - x).abs().max() < 1.0
    wrong = vq_dequantize(values, q_eff, step, "d4", 4, 12, anchor_idx, 0, True)
    assert not torch.allclose(back, wrong)


def test_lattice_ste_grad():
    x = torch.randn(8, 4, requires_grad=True)
    step = torch.ones(8, 4)
    anchor_idx = torch.arange(8, dtype=torch.int64)
    y = lattice_ste(x, step, "d4", 4, 0, anchor_idx, 0, False)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x.grad))
