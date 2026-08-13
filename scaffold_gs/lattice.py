"""Innovation-5 lattice vector quantizers (VQ) and dithered quantization.

Supported lattices (per design doc 5.3):

- ``z``  : integer lattice (degenerates to scalar quantization).
- ``d4`` : D4 lattice in 4D, integer coordinates with even coordinate sum.
- ``e8`` : E8 lattice in 8D (Gosset), represented as ``coords/2`` so every
  lattice point maps to an integer coordinate vector.

Probability model follows the doc's option 1: lattice coordinates are coded
per-dimension with the existing Gaussian entropy coder (the arithmetic coder
receives ``step*point`` values with bin width ``step/divisor``, which makes its
integer symbols exactly the lattice coordinates).
"""

from __future__ import annotations

from typing import Tuple

import torch


def lattice_divisor(lattice: str) -> int:
    """Coordinate scaling: ``point = coords / divisor``."""
    return {"z": 1, "d4": 1, "e8": 2}[lattice]


def _round_parity_fix(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest integer vector with even coordinate sum."""
    r = torch.round(x)
    delta = x - r
    s = (r.sum(dim=-1, keepdim=True) % 2).to(x.dtype)
    idx = delta.abs().argmax(dim=-1, keepdim=True)
    adjust = torch.where(
        delta.gather(-1, idx) > 0,
        torch.ones_like(delta.gather(-1, idx)),
        -torch.ones_like(delta.gather(-1, idx)),
    ) * s
    onehot = torch.zeros_like(x).scatter(-1, idx, 1.0)
    return r + onehot * adjust


def quantize_lattice(
    x_unit: torch.Tensor, lattice: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest lattice point in unit-lattice space.

    Args:
        x_unit: [..., G] values in lattice-unit space.
        lattice: ``z``, ``d4`` or ``e8``.

    Returns:
        coords: [..., G] integer coordinate tensor (int64),
        point: [..., G] float lattice point with ``point = coords / divisor``.
    """
    if lattice == "z":
        coords = torch.round(x_unit)
        return coords, coords
    if lattice == "d4":
        coords = _round_parity_fix(x_unit)
        return coords, coords
    if lattice == "e8":
        r = _round_parity_fix(x_unit)  # integer candidate
        r2 = _round_parity_fix(x_unit - 0.5) + 0.5  # half-integer candidate
        closer = (x_unit - r).pow(2).sum(-1, keepdim=True) <= (
            x_unit - r2
        ).pow(2).sum(-1, keepdim=True)
        point = torch.where(closer, r, r2)
        coords = (point * 2.0).round()
        return coords, point
    raise ValueError(f"unsupported lattice {lattice!r}")


def _hash_mix(h: torch.Tensor) -> torch.Tensor:
    mask = 0xFFFFFFFF
    h = (h ^ (h >> 16)) * 0x85EBCA6B & mask
    h = (h ^ (h >> 13)) * 0xC2B2AE35 & mask
    return (h ^ (h >> 16)) & mask


def make_dither(
    seed: int,
    anchor_idx: torch.Tensor,
    field_id: int,
    dims: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic per-element dither in [-0.5, 0.5).

    Seed is written to the codec header so encode and decode reproduce the
    same dither from (seed, anchor index, field id, element index).
    """
    n = anchor_idx.shape[0]
    idx = anchor_idx.to(torch.int64).unsqueeze(-1)  # [N, 1]
    j = torch.arange(dims, dtype=torch.int64, device=device).unsqueeze(0)  # [1, D]
    h = (
        idx * 0x9E3779B1
        + int(seed) * 0x85EBCA6B
        + int(field_id) * 0xC2B2AE35
        + j * 0x27D4EB2F
    ) & 0xFFFFFFFF
    h = _hash_mix(h)
    u = h.to(torch.float32) * (1.0 / 2**32)  # [0, 1)
    return u - 0.5


def _group_steps(step: torch.Tensor, group_size: int):
    """Per-group geometric-mean steps from a per-element step tensor [N, D].

    Returns (steps_full [N, n_full, g], steps_left [N, r], n_full, r).
    """
    d = step.shape[-1]
    n_full, r = divmod(d, group_size)
    if n_full > 0:
        full = step[..., : n_full * group_size].view(
            *step.shape[:-1], n_full, group_size
        )
        steps_full = torch.exp(torch.log(full.clamp_min(1e-12)).mean(dim=-1, keepdim=True))
    else:
        steps_full = None
    steps_left = step[..., n_full * group_size :] if r > 0 else None
    return steps_full, steps_left, n_full, r


def vq_q_eff(step: torch.Tensor, lattice: str, group_size: int) -> torch.Tensor:
    """Effective per-element entropy bin width for a VQ group layout."""
    divisor = lattice_divisor(lattice)
    d = step.shape[-1]
    n_full, r = divmod(d, group_size)
    steps_full, steps_left, _, _ = _group_steps(step, group_size)
    parts = []
    if n_full > 0:
        sf = steps_full.expand(*step.shape[:-1], n_full, group_size)
        parts.append((sf / divisor).reshape(*step.shape[:-1], n_full * group_size))
    if r > 0:
        parts.append(steps_left)
    return torch.cat(parts, dim=-1)


def vq_apply_guard(
    step: torch.Tensor, max_abs: float, group_size: int
) -> torch.Tensor:
    """Floor the per-group step so lattice coordinates stay within int16.

    The arithmetic coder clamps symbol indices to [0, 32767] (int16), so any
    coordinate beyond ~16k is silently truncated and the roundtrip breaks.
    ``max_abs`` is the global per-field max |x| (written to the codec header);
    with ``step >= max_abs / 8000`` every coordinate satisfies
    ``|coords| <= 8000`` and the range fits int16 on both sides.
    """
    if max_abs is None or max_abs <= 0.0:
        return step
    guard = float(max_abs) / 8000.0
    d = step.shape[-1]
    n_full, r = divmod(d, group_size)
    steps_full, steps_left, _, _ = _group_steps(step, group_size)
    parts = []
    if n_full > 0:
        sf = torch.maximum(
            steps_full, torch.full_like(steps_full, guard)
        ).expand(*step.shape[:-1], n_full, group_size)
        parts.append(sf.reshape(*step.shape[:-1], n_full * group_size))
    if r > 0:
        parts.append(
            torch.maximum(steps_left, torch.full_like(steps_left, guard))
        )
    return torch.cat(parts, dim=-1)


def vq_quantize(
    x: torch.Tensor,
    step: torch.Tensor,
    lattice: str,
    group_size: int,
    seed: int,
    anchor_idx: torch.Tensor,
    field_id: int,
    dither_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VQ forward: lattice-quantize ``x`` (optionally dithered).

    Args:
        x: [N, D] attribute values.
        step: [N, D] per-element quantization step (base or adaptive).
        lattice / group_size: lattice type and grouping.

    Returns:
        values: [N, D] quantized reconstruction ``step*point``,
        q_eff: [N, D] effective per-element bin width for the entropy coder,
        coords: [N, D] integer lattice coordinates (for roundtrip hashing).
    """
    divisor = lattice_divisor(lattice)
    d = x.shape[-1]
    n_full, r = divmod(d, group_size)
    steps_full, steps_left, _, _ = _group_steps(step, group_size)
    out_values = []
    out_q = []
    out_coords = []

    if n_full > 0:
        xf = x[..., : n_full * group_size].view(*x.shape[:-1], n_full, group_size)
        sf = steps_full  # [..., n_full, 1]
        if dither_enabled:
            uf = make_dither(
                seed, anchor_idx, field_id, n_full * group_size, x.device
            ).view(*x.shape[:-1], n_full, group_size)
        else:
            uf = torch.zeros_like(xf)
        coords, point = quantize_lattice((xf + sf * uf) / sf, lattice)
        values = sf * point
        out_values.append(values.reshape(*x.shape[:-1], n_full * group_size))
        out_q.append(
            sf.expand_as(values).reshape(*x.shape[:-1], n_full * group_size)
            / divisor
        )
        out_coords.append(coords.reshape(*x.shape[:-1], n_full * group_size))

    if r > 0:
        xl = x[..., n_full * group_size :]
        sl = steps_left
        if dither_enabled:
            ul = make_dither(
                seed,
                anchor_idx,
                field_id + 1000,
                r,
                x.device,
            )
        else:
            ul = torch.zeros_like(xl)
        xd = (xl + sl * ul) / sl
        coords_l = torch.round(xd)
        values_l = coords_l * sl
        out_values.append(values_l)
        out_q.append(sl)
        out_coords.append(coords_l)

    values = torch.cat(out_values, dim=-1)
    q_eff = torch.cat(out_q, dim=-1)
    coords = torch.cat(out_coords, dim=-1)
    return values, q_eff, coords


def vq_dequantize(
    values: torch.Tensor,
    q_eff: torch.Tensor,
    step: torch.Tensor,
    lattice: str,
    group_size: int,
    seed: int,
    anchor_idx: torch.Tensor,
    field_id: int,
    dither_enabled: bool,
) -> torch.Tensor:
    """Inverse of :func:`vq_quantize`: subtract the same dither."""
    if not dither_enabled:
        return values
    divisor = lattice_divisor(lattice)
    d = values.shape[-1]
    n_full, r = divmod(d, group_size)
    out = []
    if n_full > 0:
        vf = values[..., : n_full * group_size]
        uf = make_dither(
            seed, anchor_idx, field_id, n_full * group_size, values.device
        ).view(*values.shape[:-1], n_full, group_size)
        step_full = q_eff[..., : n_full * group_size] * divisor
        out.append(vf - step_full * uf.reshape_as(vf))
    if r > 0:
        vl = values[..., n_full * group_size :]
        ul = make_dither(seed, anchor_idx, field_id + 1000, r, values.device)
        out.append(vl - q_eff[..., n_full * group_size :] * ul)
    return torch.cat(out, dim=-1)


def lattice_ste(
    x: torch.Tensor,
    step: torch.Tensor,
    lattice: str,
    group_size: int,
    seed: int,
    anchor_idx: torch.Tensor,
    field_id: int,
    dither_enabled: bool,
) -> torch.Tensor:
    """Straight-through lattice quantization for the training forward pass."""
    values, _, _ = vq_quantize(
        x, step, lattice, group_size, seed, anchor_idx, field_id, dither_enabled
    )
    return values + (x - x.detach())
