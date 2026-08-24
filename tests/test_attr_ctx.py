"""CPU tests for the optional R4 attr-ctx predictor payload."""

import torch

from scaffold_gs.attr_ctx import (
    AttrCtxPredictor,
    load_attr_ctx_payload,
    quantize_attr_ctx_inplace,
    save_attr_ctx_payload,
)


def test_payload_roundtrip(tmp_path):
    torch.manual_seed(0)
    pred = AttrCtxPredictor(feat_dim=50, grid_ctx_dim=48, n_offsets=10, hidden=64)
    n = save_attr_ctx_payload(pred, tmp_path)
    assert n > 0
    quantize_attr_ctx_inplace(pred)
    pred2 = load_attr_ctx_payload(tmp_path, "cpu")
    sd1 = pred.state_dict()
    sd2 = pred2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert sd1[k].shape == sd2[k].shape
        assert torch.allclose(sd1[k], sd2[k], atol=1e-5, rtol=1e-4)


def test_adjust_shapes():
    torch.manual_seed(1)
    from scaffold_gs.attr_ctx import adjust_offsets, adjust_scaling

    pred = AttrCtxPredictor(feat_dim=50, grid_ctx_dim=48, n_offsets=10, hidden=32)
    n = 7
    ctx = torch.randn(n, 48)
    feat_q = torch.randn(n, 50)
    mean_s = torch.randn(n, 6)
    scale_s = torch.rand(n, 6).clamp_min(1e-6)
    ms, ss = adjust_scaling(pred, mean_s, scale_s, feat_q, ctx)
    assert ms.shape == (n, 6) and ss.shape == (n, 6)
    assert (ss > 0).all()
    k3 = 30
    mean_o = torch.randn(n, k3)
    scale_o = torch.rand(n, k3).clamp_min(1e-6)
    scaling_q = torch.randn(n, 6)
    masks = torch.randint(0, 2, (n, 10)).float()
    mo, so = adjust_offsets(pred, mean_o, scale_o, feat_q, scaling_q, masks, ctx)
    assert mo.shape == (n * k3,) and so.shape == (n * k3,)
    assert (so > 0).all()
