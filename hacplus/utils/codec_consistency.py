"""Pure helpers shared by PHG codec encoding, decoding, and tests.

This is the PHG v1 minimal subset: formula Q helpers, integer symbols and
bitstream file classification. I1 stores no bitstream payload, so the legacy
``i1_context_*`` files are intentionally classified as unknown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


FORMULA_INPUT_VERSION = "formula_decoder_available_v1"
CODEC_HEADER_FILENAME = "codec_header.json"
CONTENT_AWARE_Q_META_FILENAME = "content_aware_q_meta.json"


def quantization_integer_symbols(
    values,
    *,
    mode: str,
    q_step=None,
    inv_scale=None,
    center=None,
    centered: bool = False,
):
    """Map reconstructed values back to the arithmetic coder's integer symbols.

    PHG v1 only supports ``q_step``; ``inv_scale`` is rejected.
    """
    if mode != "q_step":
        raise ValueError(f"PHG v1 only supports quantization mode 'q_step', got {mode!r}")
    if q_step is None:
        raise ValueError("q_step symbols require q_step")
    if hasattr(values, "detach"):
        import torch

        return torch.round(values / q_step).to(torch.int64)
    return np.rint(np.asarray(values) / np.asarray(q_step)).astype(np.int64)


def classify_codec_file(filename: str) -> Tuple[str, bool]:
    """Return (category, decode_required) for a bitstream-directory file."""
    if filename == "attributes.pth":
        return "raw_attributes", False
    if filename == CONTENT_AWARE_Q_META_FILENAME:
        return "formula_header", True
    if filename == CODEC_HEADER_FILENAME:
        return "codec_header", True
    if filename in {
        "manifest.json",
        "codec_summary.json",
        "size_breakdown.json",
        "codec_roundtrip_diagnostics.json",
    }:
        return "aux", False
    if filename.startswith("i1_context_"):
        return "unknown", False
    if filename.endswith((".json", ".log", ".txt")):
        return "aux", False
    if filename.startswith(("xyz_gpcc", "anchor")):
        return "anchor_gpcc", True
    if filename.startswith(("x_bound_", "bounds")):
        return "core_codec", True
    if filename.startswith(("mlp_codec_", "MLPs")):
        return "mlp", True
    if filename.startswith(("hash", "hash_level")):
        return "hash", True
    if filename.endswith((".b", ".pkl", ".npz", ".pth")):
        return "core_codec", True
    return "unknown", False


def build_formula_complexity_features(
    local_density,
    predicted_mean_scaling,
    predicted_mean_offsets,
    masks,
    n_offsets: int,
):
    """Build the decoder-available I2 formula input without hidden state."""
    n_anchor = int(local_density.shape[0])
    if hasattr(local_density, "detach"):
        import torch

        local_density = local_density.reshape(n_anchor, 1)
        predicted_mean_scaling = predicted_mean_scaling.reshape(n_anchor, 6)
        predicted_mean_offsets = predicted_mean_offsets.reshape(
            n_anchor, int(n_offsets), 3
        )
        scale_anisotropy = torch.std(
            predicted_mean_scaling[:, :3], dim=-1, keepdim=True, unbiased=False
        )
        offset_energy = predicted_mean_offsets.abs().mean(dim=(1, 2)).reshape(n_anchor, 1)
        if masks is None:
            active_mask_ratio = torch.ones_like(local_density)
        else:
            active_mask_ratio = masks.to(dtype=local_density.dtype).reshape(
                n_anchor, int(n_offsets), -1
            ).mean(dim=(1, 2), keepdim=False).reshape(n_anchor, 1)
        zeros_photo = torch.zeros(
            (n_anchor, 4), device=local_density.device, dtype=local_density.dtype
        )
        result = torch.cat(
            [local_density, scale_anisotropy, offset_energy, active_mask_ratio, zeros_photo],
            dim=-1,
        )
    else:
        local_density = np.asarray(local_density).reshape(n_anchor, 1)
        predicted_mean_scaling = np.asarray(predicted_mean_scaling).reshape(n_anchor, 6)
        predicted_mean_offsets = np.asarray(predicted_mean_offsets).reshape(
            n_anchor, int(n_offsets), 3
        )
        scale_anisotropy = predicted_mean_scaling[:, :3].std(axis=-1, keepdims=True)
        offset_energy = np.abs(predicted_mean_offsets).mean(axis=(1, 2)).reshape(n_anchor, 1)
        if masks is None:
            active_mask_ratio = np.ones_like(local_density)
        else:
            active_mask_ratio = np.asarray(masks).reshape(
                n_anchor, int(n_offsets), -1
            ).mean(axis=(1, 2)).reshape(n_anchor, 1)
        zeros_photo = np.zeros((n_anchor, 4), dtype=local_density.dtype)
        result = np.concatenate(
            [local_density, scale_anisotropy, offset_energy, active_mask_ratio, zeros_photo],
            axis=-1,
        )
    if result.shape != (n_anchor, 8):
        raise RuntimeError(f"formula complexity input must be [N, 8], got {tuple(result.shape)}")
    return result


def formula_complexity_multiplier(complexity_logits, strength: float):
    """Apply the shared I2 formula mapping: 1 + tanh(z) * strength."""
    if hasattr(complexity_logits, "detach"):
        import torch

        return 1.0 + torch.tanh(complexity_logits) * float(strength)
    return 1.0 + np.tanh(np.asarray(complexity_logits)) * float(strength)


def stable_lowest_indices(scores, tie_keys, count: int):
    """Deterministic arg-partition by score with a tie-break key."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    tie_keys = np.asarray(tie_keys, dtype=np.int64).reshape(-1)
    if scores.shape != tie_keys.shape:
        raise ValueError(f"score/tie shape mismatch: {scores.shape} vs {tie_keys.shape}")
    count = min(max(int(count), 0), scores.size)
    order = np.lexsort((tie_keys, scores))
    return order[:count].astype(np.int64)
