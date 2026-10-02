"""第一层"每步税"修复的逐位等价性测试。

每处修改前的实现原样内联为参照（参照里的 .cuda() 换成 CPU 张量，设备
搬运不影响数值），在随机输入加边界值（0、±1、NaN、低于 1e-6 的值、
打包键的边界坐标）上断言新实现与旧实现 bit-equal。

hacplus 包的 import 链顶层要加载 CUDA 扩展 _gridencoder，没有该扩展的
环境（如本机 CPU 开发环境）自动跳过对应测试；在训练服务器上全量执行。
"""

import numpy as np
import pytest
import torch

from scaffold_gs.datasets import SceneCamera
from scaffold_gs.growth import remove_existing_cells
from scaffold_gs.losses import _gaussian_window, ssim_loss
from scaffold_gs.mlp_quant import _ArithEncoder, _build_table
from scaffold_gs.mlp_quant import arith_decode, arith_encode


# ---------------------------------------------------------------------------
# 参照实现（修改前原样拷贝）
# ---------------------------------------------------------------------------


class _LowBoundOld(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(x.numpy() >= 1e-6, g.numpy() < 0.0)
        t = torch.Tensor(pass_through_if + 0.0)
        return grad1 * t


class _STEBinaryOld(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


def _ssim_reference(img1, img2, window_size=11):
    """修改前的 ssim_loss：每次调用现场重建窗口 + 5 次卷积。"""
    from math import exp

    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * 1.5**2)) for x in range(window_size)]
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(3, 1, window_size, window_size).contiguous().to(img1.device)
    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(img1, window, padding=pad, groups=3)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=pad, groups=3) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return (1.0 - ssim_map).mean()


def _dedup_reference(grid_coords, unique_grid):
    """修改前的候选格子去重：逐块全对比。"""
    remove = torch.zeros(unique_grid.shape[0], dtype=torch.bool)
    chunk_size = 4096
    for start in range(0, unique_grid.shape[0], chunk_size):
        chunk = unique_grid[start : start + chunk_size]
        dup = (chunk.unsqueeze(1) == grid_coords.unsqueeze(0)).all(-1).any(-1)
        remove[start : start + chunk_size] = dup
    return remove


def _arith_encode_old(values):
    """修改前的 arith_encode 循环（list.index 线性扫描）。"""
    vals_l, counts_l, cdf, _total = _build_table(values)
    if len(vals_l) == 1:
        return b""
    enc = _ArithEncoder()
    for v in values.tolist():
        enc.encode(cdf[v], counts_l[vals_l.index(v)])
    return enc.finish()


def _hacplus_imports():
    """ hacplus 的 import 链需要 CUDA 扩展，缺失时跳过测试。"""
    pytest.importorskip("_gridencoder")
    from hacplus.utils.encodings import STE_binary, get_binary_vxl_size
    from hacplus.utils.entropy_models import Low_bound

    return STE_binary, get_binary_vxl_size, Low_bound


# ---------------------------------------------------------------------------
# Low_bound：backward 去 CPU 往返 + 去二次 apply
# ---------------------------------------------------------------------------


def _rand_x_g(n=4096, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, generator=gen) ** 6) * 1e-4  # 大量样本低于 1e-6
    x[:: 97] = 0.0
    g = torch.randn(n, generator=gen)
    return x, g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_low_bound_forward_backward_bitexact(seed):
    _, _, Low_bound = _hacplus_imports()
    x0, g0 = _rand_x_g(seed=seed)

    x_old = x0.clone().requires_grad_(True)
    y_old = _LowBoundOld.apply(x_old)
    (g_old,) = torch.autograd.grad((y_old * g0).sum(), x_old)

    x_new = x0.clone().requires_grad_(True)
    y_new = Low_bound.apply(x_new)
    (g_new,) = torch.autograd.grad((y_new * g0).sum(), x_new)

    assert torch.equal(y_old, y_new)
    assert torch.equal(g_old, g_new)


def test_double_low_bound_removal_bitexact():
    """mix_prob_2/3 里连着两次 Low_bound 与一次在数值上逐位等价（前向与反向）。"""
    _, _, Low_bound = _hacplus_imports()
    x0, g0 = _rand_x_g(seed=7)
    w = torch.randn_like(x0)

    x_old = x0.clone().requires_grad_(True)
    y_old = _LowBoundOld.apply(_LowBoundOld.apply(x_old))
    (g_old,) = torch.autograd.grad((y_old * w).sum(), x_old)

    x_new = x0.clone().requires_grad_(True)
    y_new = Low_bound.apply(x_new)
    (g_new,) = torch.autograd.grad((y_new * w).sum(), x_new)

    assert torch.equal(y_old, y_new)
    assert torch.equal(g_old, g_new)


# ---------------------------------------------------------------------------
# STE_binary / get_binary_vxl_size
# ---------------------------------------------------------------------------


def _ste_binary_cases(seed=11):
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(2048, generator=gen) * 3.0
    specials = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 1.0 + 2**-23, -(1.0 + 2**-23),
         1e30, -1e30, float("inf"), float("-inf"), float("nan")]
    )
    return torch.cat([base, specials])


def test_ste_binary_bitexact():
    STE_binary, _, _ = _hacplus_imports()
    x0 = _ste_binary_cases()
    g0 = torch.nan_to_num(torch.randn_like(x0))

    x_old = x0.clone().requires_grad_(True)
    y_old = _STEBinaryOld.apply(x_old)
    (g_old,) = torch.autograd.grad((y_old * g0).sum(), x_old)

    x_new = x0.clone().requires_grad_(True)
    y_new = STE_binary.apply(x_new)
    (g_new,) = torch.autograd.grad((y_new * g0).sum(), x_new)

    assert torch.equal(y_old, y_new)
    assert torch.equal(g_old, g_new)


def test_get_binary_vxl_size_bitexact():
    _, get_binary_vxl_size, _ = _hacplus_imports()
    gen = torch.Generator().manual_seed(3)
    v = (torch.rand(4096, generator=gen) < 0.37).float()

    ttl_num = v.numel()
    pos_num = torch.sum(v)
    Pg = torch.clamp(pos_num / ttl_num, min=1e-6, max=1 - 1e-6)
    ttl_bit = pos_num * (-torch.log2(Pg)) + (ttl_num - pos_num) * (
        -torch.log2(1 - Pg)
    ) + 32

    pg_new, bit_new, mb_new, num_new = get_binary_vxl_size(v)
    assert torch.equal(pg_new, Pg)
    assert torch.equal(bit_new, ttl_bit)
    assert isinstance(mb_new, torch.Tensor)
    assert torch.equal(mb_new, ttl_bit / 8.0 / 1024 / 1024)
    assert num_new == ttl_num


# ---------------------------------------------------------------------------
# SSIM 窗口缓存
# ---------------------------------------------------------------------------


def test_ssim_window_cache_bitexact():
    gen = torch.Generator().manual_seed(5)
    img1 = torch.rand(1, 3, 64, 64, generator=gen)
    img2 = torch.rand(1, 3, 64, 64, generator=gen)

    ref = _ssim_reference(img1, img2)
    got = ssim_loss(img1, img2)
    assert torch.equal(ref, got)

    # 缓存窗口与现场重建的窗口逐位一致，且第二次拿到的是同一个对象
    w1 = _gaussian_window(11, 1.5, img1.device)
    w2 = _gaussian_window(11, 1.5, img1.device)
    assert w1 is w2


# ---------------------------------------------------------------------------
# 候选格子去重
# ---------------------------------------------------------------------------


def test_remove_existing_cells_bitexact():
    gen = torch.Generator().manual_seed(9)
    grid_coords = torch.randint(-50, 50, (500, 3), generator=gen)
    unique_grid = torch.randint(-60, 60, (300, 3), generator=gen)
    unique_grid[:100] = grid_coords[torch.randint(0, 500, (100,), generator=gen)]

    assert torch.equal(
        remove_existing_cells(grid_coords, unique_grid),
        _dedup_reference(grid_coords, unique_grid),
    )


def test_remove_existing_cells_key_boundary():
    """打包键每轴 21bit：±2^20 走键路径，±(2^20+1) 回退全对比，两边一致。"""
    B = 1 << 20
    grid_coords = torch.tensor([[B, B, B], [-B, -B, -B], [0, 0, 0]])
    unique_grid = torch.tensor(
        [[B, B, B], [-(B + 1), 0, 0], [B + 1, 0, 0], [-B, -B, -B], [0, 0, 1]]
    )
    assert torch.equal(
        remove_existing_cells(grid_coords, unique_grid),
        _dedup_reference(grid_coords, unique_grid),
    )

    grid_coords2 = torch.tensor([[B + 1, 0, 0], [0, 0, 0]])
    assert torch.equal(
        remove_existing_cells(grid_coords2, unique_grid),
        _dedup_reference(grid_coords2, unique_grid),
    )


# ---------------------------------------------------------------------------
# mlp_quant 静态算术编码
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_arith_encode_bitexact_and_roundtrip(seed):
    gen = torch.Generator().manual_seed(seed)
    values = torch.randint(-8, 9, (4096,), generator=gen)

    new_bytes, vals_l, counts_l = arith_encode(values)
    assert new_bytes == _arith_encode_old(values)

    decoded = arith_decode(new_bytes, values.numel(), vals_l, counts_l)
    assert decoded == values.tolist()


# ---------------------------------------------------------------------------
# 相机矩阵缓存
# ---------------------------------------------------------------------------


def test_camera_cache_bitexact():
    from pathlib import Path

    gen = np.random.RandomState(13)
    c2w = gen.randn(4, 4).astype(np.float64)
    c2w[3] = [0.0, 0.0, 0.0, 1.0]
    K = gen.rand(3, 3).astype(np.float64) * 100 + 500

    cam = SceneCamera(
        uid=0, colmap_id=0, image_name="", image_path=Path("."),
        c2w=c2w, K=K, width=16, height=16, split="train", appearance_id=0,
    )
    device = torch.device("cpu")
    v1, k1 = cam.to_gsplat(device)
    v2, k2 = cam.to_gsplat(device)
    assert v1 is v2 and k1 is k2

    c2w_t = torch.from_numpy(c2w).float()
    ref_v = torch.linalg.inv_ex(c2w_t).inverse
    assert torch.equal(v1[0], ref_v)
    assert torch.equal(k1[0], torch.from_numpy(K).float())

    c1 = cam.camera_center(device)
    c2 = cam.camera_center(device)
    assert c1 is c2
    assert torch.equal(c1, c2w_t[:3, 3])
