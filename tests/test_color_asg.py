"""Tests for the two-stage ASG color path.

The default ``color_mode="rgb"`` path must stay byte-for-byte identical to
the old ``mlp_color`` decoder.  The ASG path is only enabled explicitly and is
checked for shape/range, determinism, checkpoint reproduction and MLP
quantization grouping.
"""

from __future__ import annotations

import copy

import pytest
import torch

from scaffold_gs.asg import _orthonormal_basis, evaluate_asg, evaluate_asg_rgb
from scaffold_gs.config import ModelConfig
from scaffold_gs.mlp_quant import MLP_GROUPS, quantize_core_mlps
from scaffold_gs.model import AnchorDecoder, AnchorParams


def _fake_params(n: int = 6, k: int = 3, feat_dim: int = 6) -> AnchorParams:
    p = AnchorParams(k, feat_dim)
    p.anchor = torch.nn.Parameter(torch.randn(n, 3))
    p.offset = torch.nn.Parameter(torch.randn(n, k, 3) * 0.1)
    p.anchor_feat = torch.nn.Parameter(torch.randn(n, feat_dim))
    p.scaling = torch.nn.Parameter(torch.zeros(n, 6))
    p.rotation = torch.nn.Parameter(torch.zeros(n, 4), requires_grad=False)
    p.opacity = torch.nn.Parameter(torch.zeros(n, 1), requires_grad=False)
    return p


def _decoder(**overrides) -> AnchorDecoder:
    kwargs = dict(
        feat_dim=6,
        n_offsets=3,
        appearance_dim=0,
        color_mode="rgb",
        asg_lobes=1,
        asg_latent_dim=8,
        asg_hidden=6,
    )
    kwargs.update(overrides)
    return AnchorDecoder(**kwargs)


def test_colormode_validation():
    with pytest.raises(ValueError):
        ModelConfig(color_mode="sh")
    with pytest.raises(ValueError):
        ModelConfig(asg_lobes=0)
    with pytest.raises(ValueError):
        ModelConfig(asg_latent_dim=0)
    with pytest.raises(ValueError):
        ModelConfig(asg_hidden=0)


def test_rgb_mode_unchanged():
    torch.manual_seed(7)
    decoder = _decoder()
    params = _fake_params()
    assert not hasattr(decoder, "mlp_asg")
    g = decoder.predict_gaussians(params, torch.zeros(3))

    feat = params.anchor_feat
    anchor = params.anchor
    ob_view = anchor - torch.zeros(3)
    ob_view = ob_view / ob_view.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ref = decoder.mlp_color(torch.cat([feat, ob_view], dim=-1)).reshape(
        params.num_anchors, decoder.n_offsets, 3
    )
    selected_ref = ref.reshape(-1, 3)[g.selection_mask]
    assert g.colors.shape == selected_ref.shape
    assert torch.equal(g.colors, selected_ref)


def test_asg_shape_range_and_registered_modules():
    torch.manual_seed(8)
    decoder = _decoder(color_mode="asg")
    # Ensure a non-empty selection for the shape assertion.
    with torch.no_grad():
        decoder.mlp_opacity[2].bias.fill_(0.1)
    g = decoder.predict_gaussians(_fake_params(), torch.zeros(3))
    assert g.colors.shape[1] == 3
    assert g.colors.shape[0] == g.xyz.shape[0]
    assert bool((g.colors >= 0.0).all()) and bool((g.colors <= 1.0).all())
    assert hasattr(decoder, "mlp_asg")
    assert hasattr(decoder, "mlp_color2")
    assert any("mlp_asg" in n for n, _ in decoder.named_parameters())
    assert any("mlp_color2" in n for n, _ in decoder.named_parameters())


def test_asg_determinism_and_roundtrip():
    torch.manual_seed(9)
    decoder = _decoder(color_mode="asg")
    params = _fake_params()
    with torch.no_grad():
        decoder.mlp_opacity[2].bias.fill_(0.1)
    a = decoder.predict_gaussians(params, torch.zeros(3))
    b = decoder.predict_gaussians(params, torch.zeros(3))
    assert torch.equal(a.colors, b.colors)

    sd = copy.deepcopy(decoder.state_dict())
    decoder2 = _decoder(color_mode="asg")
    decoder2.load_state_dict(sd)
    c = decoder2.predict_gaussians(params, torch.zeros(3))
    assert torch.equal(a.colors, c.colors)

    raw = torch.randn(3, 3 * (3 + 1 + 1 + 3 + 8))
    view = torch.randn(3, 3)
    view = view / view.norm(dim=-1, keepdim=True)
    x, y, z = _orthonormal_basis(raw[:, :3])
    assert torch.allclose(
        (x * x).sum(-1)
        + (y * y).sum(-1)
        + (z * z).sum(-1),
        torch.full((3,), 3.0),
        atol=1e-5,
    )
    f1, l1 = evaluate_asg(raw, view, 3, lobes=1, latent_dim=8)
    f2, l2 = evaluate_asg(raw, view, 3, lobes=1, latent_dim=8)
    assert torch.equal(f1, f2)
    assert torch.equal(l1, l2)


def test_mlp_quant_groups_for_asg():
    torch.manual_seed(10)
    decoder = _decoder(color_mode="asg")
    meta = quantize_core_mlps(
        decoder, bits=8, groups=("mlp_asg", "mlp_color2")
    )
    names = set(meta)
    assert any("mlp_asg" in n for n in names)
    assert any("mlp_color2" in n for n in names)
    assert all(meta[n]["group"] in ("mlp_asg", "mlp_color2") for n in names)
    # The group router must not treat mlp_color2 as mlp_color.
    assert "mlp_color" in MLP_GROUPS
    assert "mlp_color2" in MLP_GROUPS
    assert "mlp_asg" in MLP_GROUPS


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ASG HAC++ adapter requires CUDA extensions",
)
def test_hac_core_asg_adapter_smoke():
    from scaffold_gs.hacpp import HACPlusModel

    cfg = ModelConfig(
        model_name="hac_pp",
        feat_dim=4,
        n_offsets=2,
        voxel_size=1.0,
        appearance_dim=0,
        color_mode="asg",
        asg_lobes=1,
        asg_latent_dim=8,
        asg_hidden=4,
    )
    model = HACPlusModel(cfg, "cuda")
    assert hasattr(model.core, "mlp_asg")
    assert hasattr(model.core, "mlp_color2")
    attrs = model.export_attributes()
    restored = HACPlusModel.from_attributes(attrs, "cuda")
    assert hasattr(restored.core, "mlp_asg")
    assert hasattr(restored.core, "mlp_color2")
