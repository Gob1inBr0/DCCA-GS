import numpy as np
import torch
from math import log

from scaffold_gs.config import ModelConfig, OptimConfig
from scaffold_gs.model import MODELS


def _make_model():
    cfg = ModelConfig(voxel_size=0.1, appearance_dim=0)
    model = MODELS["scaffold_gs"](cfg, "cpu")
    pts = np.random.RandomState(3).rand(200, 3).astype(np.float32)
    rgb = np.random.RandomState(4).randint(0, 255, (200, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)
    model.create_optimizer(OptimConfig(max_steps=100, eval_steps=[], save_steps=[]))
    for group in model.optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
    model.optimizer.step()  # initialize Adam states for shape checks
    model.optimizer.zero_grad(set_to_none=True)
    return model


def _check_optimizer_shapes(model):
    for group in model.optimizer.param_groups:
        if group["name"] not in ("anchor", "offset", "anchor_feat", "scaling"):
            continue
        p = group["params"][0]
        state = model.optimizer.state[p]
        assert state["exp_avg"].shape == p.shape
        assert state["exp_avg_sq"].shape == p.shape


def test_grow_anchors():
    model = _make_model()
    before = model.num_anchors
    n = before
    k = model.cfg.n_offsets
    # Give the offsets non-zero values so candidate cells differ from the
    # existing anchor grid (with zero offsets, growth legitimately finds no
    # new cells, exactly like the official implementation).
    with torch.no_grad():
        model.anchor_params.offset.uniform_(-1.0, 1.0)
        model.anchor_params.scaling.fill_(log(0.8))
    model.offset_gradient_accum[: n * k] = 0.1
    model.offset_denom[: n * k] = 100.0
    model.opacity_accum[:] = 0.1
    model.anchor_demon[:] = 100.0

    model.adjust_anchor(
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        min_opacity=0.005,
    )
    assert model.num_anchors > before
    assert model.offset_gradient_accum.shape[0] == model.num_anchors * k
    assert model.offset_denom.shape[0] == model.num_anchors * k
    assert model.opacity_accum.shape[0] == model.num_anchors
    _check_optimizer_shapes(model)


def test_prune_anchors():
    model = _make_model()
    before = model.num_anchors
    model.opacity_accum[:] = 0.0001
    model.anchor_demon[:] = 200.0
    model.offset_denom[:] = 0.0  # no growth this round

    model.adjust_anchor(
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        min_opacity=0.005,
    )
    assert model.num_anchors < before
    _check_optimizer_shapes(model)
