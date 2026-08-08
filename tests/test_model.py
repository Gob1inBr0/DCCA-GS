import numpy as np
import pytest
import torch

from scaffold_gs.config import ModelConfig
from scaffold_gs.model import MODELS, ScaffoldGSModel
from scaffold_gs.utils import voxelize_points


def test_voxelize_points_dedup():
    pts = np.array(
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.99, 0.99, 0.99]],
        dtype=np.float32,
    )
    out = voxelize_points(pts, 0.1)
    assert out.shape == (2, 3)


def _make_model(**overrides):
    cfg = ModelConfig(voxel_size=0.1, appearance_dim=0, **overrides)
    model = MODELS["scaffold_gs"](cfg, "cpu")
    pts = np.random.RandomState(0).rand(200, 3).astype(np.float32)
    rgb = np.random.RandomState(1).randint(0, 255, (200, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)
    return model


def test_init_shapes():
    model = _make_model()
    assert model.anchor_params.anchor.shape[1] == 3
    assert model.anchor_params.offset.shape == (model.num_anchors, 10, 3)
    assert model.anchor_params.anchor_feat.shape[1] == 32
    assert model.anchor_params.scaling.shape[1] == 6
    assert model.anchor_params.rotation.shape == (model.num_anchors, 4)
    assert model.opacity_accum.shape == (model.num_anchors, 1)
    assert model.offset_denom.shape == (model.num_anchors * 10, 1)


def test_decoder_outputs():
    model = _make_model(use_feat_bank=True)
    model.set_appearance(4)
    center = torch.zeros(3)
    g = model.decoder.predict_gaussians(
        model.anchor_params, center, appearance_id=0
    )
    assert g.neural_opacity.shape == (model.num_anchors, 10)
    assert g.selection_mask.shape[0] == model.num_anchors * 10
    assert g.xyz.shape[1] == 3
    assert g.colors.shape[1] == 3
    assert g.scales.shape[1] == 3
    assert g.quats.shape[1] == 4
    assert g.xyz.shape[0] == g.colors.shape[0]


def test_export_attributes_roundtrip():
    model = _make_model()
    attrs = model.export_attributes()
    model2 = ScaffoldGSModel.from_attributes(attrs, "cpu")
    for name in ("anchor", "offset", "anchor_feat", "scaling", "rotation", "opacity"):
        a = getattr(model.anchor_params, name)
        b = getattr(model2.anchor_params, name)
        assert torch.allclose(a, b), name


def test_model_registry():
    assert "scaffold_gs" in MODELS
