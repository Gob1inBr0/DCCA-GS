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


def _make_camera(uid: int):
    from scaffold_gs.datasets import SceneCamera

    c2w = np.eye(4)
    c2w[2, 3] = -5.0
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
    return SceneCamera(
        uid=uid,
        colmap_id=uid,
        image_name=f"cam{uid}",
        image_path=Path("."),
        c2w=c2w,
        K=K,
        width=64,
        height=64,
        split="train",
        appearance_id=uid,
    )


def test_hacpp_train_step_and_codec(tmp_path):
    from scaffold_gs.config import ModelConfig, OptimConfig
    from scaffold_gs.hacpp import HACPlusCodec, HACPlusModel

    cfg = ModelConfig(
        model_name="hac_pp",
        feat_dim=50,
        n_offsets=10,
        voxel_size=0.1,
        appearance_dim=0,
    )
    model = HACPlusModel(cfg, "cuda")
    pts = (np.random.RandomState(0).rand(300, 3).astype(np.float32) - 0.5)
    rgb = np.random.RandomState(1).randint(0, 255, (300, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)
    model.create_optimizer(
        OptimConfig(max_steps=20, eval_steps=[], save_steps=[], lambda_rate=0.004)
    )

    cam = _make_camera(0)
    bg = torch.zeros(3, device="cuda")
    out = model.render(cam, bg, is_training=True, retain_grad=True, step=12_000)
    assert out.image.shape == (1, 64, 64, 3)
    assert out.gaussians.xyz.shape[0] > 0

    loss = (out.image**2).mean()
    if out.gaussians.bit_per_param is not None:
        loss = loss + model.rate_loss_term(out.gaussians, 12_000)
    loss.backward()
    model.training_statis(
        out.meta["means2d"],
        (out.meta["radii"] > 0).all(dim=-1),
        out.gaussians,
        out.meta["width"],
        out.meta["gaussian_ids"],
        out.meta["height"],
    )
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)

    # Encode -> decode -> render round-trip.
    codec = HACPlusCodec()
    meta = codec.encode(model, tmp_path)
    assert meta["total_bits"] > 0
    assert (tmp_path / "xyz_gpcc.npz").exists()
    model2 = codec.decode(tmp_path)
    assert model2._view.decoded_version
    assert model2.num_anchors == model.num_anchors
    out2 = model2.render(cam, bg, is_training=False)
    assert out2.image.shape == (1, 64, 64, 3)


def test_pick_channel_group():
    from hacplus.scene.gaussian_model import pick_channel_group

    assert pick_channel_group(50) == 10
    assert pick_channel_group(32) == 8
    assert pick_channel_group(24) == 8
    assert pick_channel_group(16) == 8
    assert pick_channel_group(8) == 8
    assert pick_channel_group(20) == 10
    assert pick_channel_group(30) == 10


@pytest.mark.parametrize("feat_dim", [8, 16, 24, 32, 50])
def test_hacpp_feat_dim_codec_roundtrip(tmp_path, feat_dim):
    from scaffold_gs.config import ModelConfig
    from scaffold_gs.hacpp import HACPlusCodec, HACPlusModel

    cfg = ModelConfig(
        model_name="hac_pp",
        feat_dim=feat_dim,
        n_offsets=10,
        voxel_size=0.1,
        appearance_dim=0,
        content_aware_quant=False,
    )
    model = HACPlusModel(cfg, "cuda")
    pts = (np.random.RandomState(feat_dim).rand(200, 3).astype(np.float32) - 0.5)
    rgb = np.random.RandomState(feat_dim + 1).randint(0, 255, (200, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)

    codec = HACPlusCodec()
    meta = codec.encode(model, tmp_path)
    assert meta["total_bits"] > 0
    model2 = codec.decode(tmp_path)
    assert model2._view.decoded_version
    assert model2._view.anchor_feat.shape[-1] == feat_dim
    cam = _make_camera(0)
    out = model2.render(cam, torch.zeros(3, device="cuda"), is_training=False)
    assert out.image.shape == (1, 64, 64, 3)
