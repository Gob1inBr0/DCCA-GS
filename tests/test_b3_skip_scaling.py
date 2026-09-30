"""B3 coarse_ladder_skip_scaling: scaling field excluded from the penalty.

The penalty lives on HACPlusModel but only reads ``self.cfg``, so the
unbound method is exercised against a stand-in namespace — no CUDA model
construction needed.
"""

import math
import types

import torch

from scaffold_gs.hacpp import HACPlusModel


def _fake_self(skip_scaling: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(coarse_ladder_skip_scaling=skip_scaling)
    )


def _inputs():
    torch.manual_seed(0)
    feat = torch.randn(4, 5, 32, requires_grad=True)
    # leaf tensor so the grad-isolation assertion in the skip case is
    # meaningful (a non-leaf result of `randn * 3.0` never gets .grad)
    scaling = (torch.randn(4, 5, 3) * 3.0).requires_grad_(True)
    offsets = torch.randn(4, 5, 9, requires_grad=True)
    q_feat = torch.full((), 0.01)
    q_scaling = torch.full((), 0.001)
    q_offsets = torch.full((), 0.1)
    masks = torch.ones(4, 5, 3)
    return feat, scaling, offsets, q_feat, q_scaling, q_offsets, masks


def _expected_terms(feat, scaling, offsets, qf, qs, qo, masks):
    def align(x, q, L):
        r = x / q
        return ((r - L * torch.round(r / L)) / L).abs()

    mask3 = masks.to(offsets.dtype).repeat(1, 1, 3)
    return {
        "feat": align(feat, qf, 8).mean(),
        "scaling": align(scaling, qs, 2).mean(),
        "offset": (align(offsets, qo, 8) * mask3).sum() / mask3.sum(),
    }


def test_skip_scaling_removes_scaling_term_and_gradient():
    feat, scaling, offsets, qf, qs, qo, masks = _inputs()
    terms = _expected_terms(feat, scaling, offsets, qf, qs, qo, masks)

    pen = HACPlusModel._coarse_ladder_penalty_raw(
        _fake_self(True), feat, scaling, offsets, qf, qs, qo, masks
    )
    expected = terms["feat"] + terms["offset"]
    assert torch.allclose(pen, expected, atol=1e-6)

    pen.backward()
    assert scaling.grad is None, "scaling must not receive gradient when skipped"


def test_default_keeps_all_three_terms():
    feat, scaling, offsets, qf, qs, qo, masks = _inputs()
    terms = _expected_terms(feat, scaling, offsets, qf, qs, qo, masks)

    pen = HACPlusModel._coarse_ladder_penalty_raw(
        _fake_self(False), feat, scaling, offsets, qf, qs, qo, masks
    )
    expected = terms["feat"] + terms["scaling"] + terms["offset"]
    assert torch.allclose(pen, expected, atol=1e-6)
    assert math.isfinite(float(pen))
