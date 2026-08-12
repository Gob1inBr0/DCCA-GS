"""PHG v1 tests: I1 pure-coordinate context and I2 formula quantization."""

from pathlib import Path

import numpy as np
import pytest
import torch


def _hacpp_available() -> bool:
    try:
        import arithmetic  # noqa: F401
        import hacplus  # noqa: F401
        import simple_knn  # noqa: F401
        import _gridencoder  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _hacpp_available() or not torch.cuda.is_available(),
    reason="HAC++ CUDA extensions (gridencoder/arithmetic/simple_knn) required",
)


def _make_model(
    feat_dim: int = 50,
    n_offsets: int = 10,
    hierarchical: bool = False,
    hidden=None,
    layers: int = 1,
):
    from scaffold_gs.config import ModelConfig, OptimConfig
    from scaffold_gs.hacpp import HACPlusModel

    cfg = ModelConfig(
        model_name="hac_pp",
        feat_dim=feat_dim,
        n_offsets=n_offsets,
        voxel_size=0.1,
        appearance_dim=0,
        hierarchical_context=hierarchical,
        hierarchical_context_start_iter=0,
        content_aware_quant=True,
        content_aware_q_mode="formula",
        content_aware_start_iter=0,
        content_aware_ramp_iters=1,
        mlp_complexity_hidden=hidden,
        mlp_complexity_layers=layers,
    )
    model = HACPlusModel(cfg, "cuda")
    pts = (np.random.RandomState(0).rand(300, 3).astype(np.float32) - 0.5)
    rgb = np.random.RandomState(1).randint(0, 255, (300, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)
    model.create_optimizer(
        OptimConfig(
            max_steps=20,
            eval_steps=[],
            save_steps=[],
            lambda_rate=0.004,
        )
    )
    return model


def test_config_defaults_and_reserved_flags():
    from scaffold_gs.config import ModelConfig

    cfg = ModelConfig()
    assert cfg.hierarchical_context is False
    assert cfg.content_aware_quant is True
    assert cfg.content_aware_q_mode == "formula"
    assert cfg.complexity_scale == 0.35
    with pytest.raises(NotImplementedError):
        ModelConfig(vq_enabled=True)
    with pytest.raises(NotImplementedError):
        ModelConfig(dither_enabled=True)
    with pytest.raises(ValueError):
        ModelConfig(content_aware_q_mode="exact")


def test_i1_context_dimension_and_level_determinism():
    model = _make_model(hierarchical=True)
    core = model.core
    core.current_iter = 1
    anchor = core.get_anchor[:8]
    ctx = core.calc_context_feat(anchor, caller="test")
    assert ctx.shape[-1] == core.base_grid_context_dim * 2 + 3
    l1 = core.compute_anchor_level_ids(anchor)
    l2 = core.compute_anchor_level_ids(anchor)
    assert torch.equal(l1, l2)
    assert set(l1.tolist()) <= {0, 1, 2}


def test_mlp_complexity_shape_is_configurable():
    model = _make_model(hidden=16, layers=2)
    mlp = model.core.mlp_complexity
    assert mlp[0].in_features == 8
    assert mlp[0].out_features == 16
    assert mlp[-1].in_features == 16
    assert mlp[-1].out_features == 3


def test_i2_formula_q_is_deterministic_on_encode_and_decode_inputs():
    model = _make_model()
    core = model.core
    core.current_step = 5  # active (start_iter=0)
    anchor = core.get_anchor[:8]
    masks = core.get_mask[:8]
    mean_scaling = torch.rand(8, 6, device="cuda")
    mean_offsets = torch.rand(8, 30, device="cuda")
    q_feat = torch.ones(8, 50, device="cuda")
    q_scaling = torch.ones(8, 6, device="cuda")
    q_offsets = torch.ones(8, 30, device="cuda")

    def q():
        return core._codec_apply_content_aware_quant_params(
            "test",
            anchor,
            masks,
            q_feat,
            q_scaling,
            q_offsets,
            None,
            None,
            None,
            mean_scaling,
            mean_offsets,
        )

    a = q()
    b = q()
    for ta, tb in zip(a[:3], b[:3]):
        assert torch.equal(ta, tb)


def test_roundtrip_writes_header_and_no_i1_files(tmp_path: Path):
    from scaffold_gs.hacpp import HACPlusCodec

    model = _make_model()
    codec = HACPlusCodec()
    meta = codec.encode(model, tmp_path)
    assert meta["total_bits"] > 0
    names = {p.name for p in tmp_path.iterdir()}
    assert "codec_header.json" in names
    assert "content_aware_q_meta.json" in names
    assert not any(name.startswith("i1_context_") for name in names)

    model2 = codec.decode(tmp_path)
    assert model2.core.decoded_version
    assert model2.num_anchors == model.num_anchors
    diag_path = tmp_path / "codec_roundtrip_diagnostics.json"
    assert diag_path.is_file()
