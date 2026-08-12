"""PHG I6 tests: sensitivity supervision, EMA, and q_override round-trip."""

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
    reason="HAC++ CUDA extensions required",
)


def _make_sens_model():
    from scaffold_gs.config import ModelConfig, OptimConfig
    from scaffold_gs.hacpp import HACPlusModel

    cfg = ModelConfig(
        model_name="hac_pp",
        feat_dim=50,
        n_offsets=10,
        voxel_size=0.1,
        appearance_dim=0,
        content_aware_quant=True,
        content_aware_start_iter=0,
        content_aware_ramp_iters=1,
        sensitivity_enabled=True,
        sensitivity_weight=1e-3,
        sensitivity_start_iter=0,
    )
    model = HACPlusModel(cfg, "cuda")
    pts = (np.random.RandomState(0).rand(200, 3).astype(np.float32) - 0.5)
    rgb = np.random.RandomState(1).randint(0, 255, (200, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)
    model.create_optimizer(
        OptimConfig(max_steps=20, eval_steps=[], save_steps=[], lambda_rate=0.004)
    )
    return model


def _make_camera():
    from scaffold_gs.datasets import SceneCamera

    c2w = np.eye(4)
    c2w[2, 3] = -5.0
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
    return SceneCamera(
        uid=0,
        colmap_id=0,
        image_name="cam0",
        image_path=Path("."),
        c2w=c2w,
        K=K,
        width=64,
        height=64,
        split="train",
        appearance_id=0,
    )


def test_sensitivity_config_validation():
    from scaffold_gs.config import ModelConfig

    with pytest.raises(ValueError):
        ModelConfig(sensitivity_enabled=True, sensitivity_weight=0.0)
    with pytest.raises(ValueError):
        ModelConfig(sensitivity_enabled=True, content_aware_quant=False)


def test_sensitivity_gradient_reaches_mlp_complexity():
    model = _make_sens_model()
    cam = _make_camera()
    bg = torch.zeros(3, device="cuda")

    out1 = model.render(cam, bg, is_training=True, retain_grad=True, appearance_id=0, step=12_000)
    assert out1.gaussians.pre_quant_feat is not None, (
        "renderer must forward retain_grad to generate_gaussians"
    )
    assert out1.gaussians.pre_quant_feat.requires_grad
    loss1 = (out1.image**2).mean()
    loss1.backward()
    assert out1.gaussians.pre_quant_feat.grad is not None
    assert out1.gaussians.pre_quant_feat.grad.abs().sum() > 0
    model.accumulate_sensitivity(out1.gaussians)
    assert model.core.sensitivity_feat.abs().sum() > 0
    model.optimizer.zero_grad(set_to_none=True)

    out2 = model.render(cam, bg, is_training=True, retain_grad=True, appearance_id=0, step=12_001)
    assert out2.gaussians.complexity_logits is not None
    sens_loss = model.sensitivity_supervision(out2.gaussians)
    sens_loss.backward()
    grads = [
        p.grad
        for p in model.core.mlp_complexity.parameters()
        if p.grad is not None
    ]
    assert grads, "sensitivity supervision did not reach mlp_complexity"
    assert any(grad.abs().sum() > 0 for grad in grads)


def test_sensitivity_active_after_growth_window():
    """I6 accumulation must keep working after retain_grad is turned off."""
    model = _make_sens_model()
    cam = _make_camera()
    bg = torch.zeros(3, device="cuda")

    out = model.render(
        cam, bg, is_training=True, retain_grad=False, appearance_id=0, step=12_001
    )
    assert out.gaussians.pre_quant_feat is not None
    out.image.sum().backward()
    assert out.gaussians.pre_quant_feat.grad is not None
    model.accumulate_sensitivity(out.gaussians)
    assert model.core.sensitivity_feat.abs().sum() > 0


def test_q_override_roundtrip(tmp_path: Path):
    from scaffold_gs.hacpp import HACPlusCodec

    model = _make_sens_model()
    model.cfg.sensitivity_enabled = False  # standard codec path
    codec = HACPlusCodec()
    meta0 = codec.encode(model, tmp_path / "formula")
    n = int(meta0["num_anchors"])
    ov = np.ones((n, 1), dtype=np.float32) * 1.5
    ov_dir = tmp_path / "sens"
    ov_dir.mkdir()
    np.save(ov_dir / "q_override_feat.npy", ov)
    np.save(ov_dir / "q_override_scaling.npy", ov)
    np.save(ov_dir / "q_override_offsets.npy", ov)
    meta = codec.encode(
        model,
        ov_dir,
        q_override_feat=str(ov_dir / "q_override_feat.npy"),
        q_override_scaling=str(ov_dir / "q_override_scaling.npy"),
        q_override_offsets=str(ov_dir / "q_override_offsets.npy"),
    )
    decoded = codec.decode(
        ov_dir,
        q_override_feat=str(ov_dir / "q_override_feat.npy"),
        q_override_scaling=str(ov_dir / "q_override_scaling.npy"),
        q_override_offsets=str(ov_dir / "q_override_offsets.npy"),
    )
    assert decoded.core.decoded_version
    assert (ov_dir / "codec_roundtrip_diagnostics.json").is_file()
