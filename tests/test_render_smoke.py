from pathlib import Path

import numpy as np
import pytest
import torch

from scaffold_gs.config import ModelConfig, OptimConfig
from scaffold_gs.datasets import SceneCamera
from scaffold_gs.model import MODELS


def _make_camera(uid: int) -> SceneCamera:
    c2w = np.eye(4)
    c2w[2, 3] = -5.0  # camera at z=-5 looking toward +Z (gsplat/OpenGL -Z)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_render_and_train_step():
    import gsplat  # noqa: F401

    cfg = ModelConfig(voxel_size=0.1, appearance_dim=0)
    model = MODELS["scaffold_gs"](cfg, "cuda")
    pts = np.random.RandomState(0).rand(500, 3).astype(np.float32) * 2 - 1
    rgb = np.random.RandomState(1).randint(0, 255, (500, 3)).astype(np.uint8)
    model.init_from_pcd(pts, rgb, 1.0)
    model.create_optimizer(OptimConfig(max_steps=10, eval_steps=[], save_steps=[]))

    cam = _make_camera(0)
    bg = torch.zeros(3, device="cuda")
    out = model.render(cam, bg, is_training=True, retain_grad=True)
    assert out.image.shape == (1, 64, 64, 3)
    assert out.visible_mask.sum() > 0
    assert out.gaussians.xyz.shape[0] > 0

    gt = torch.zeros_like(out.image)
    loss = ((out.image - gt) ** 2).mean()
    loss.backward()
    model.training_statis(
        out.meta["means2d"],
        (out.meta["radii"] > 0).all(dim=-1),
        out.gaussians,
        out.meta["width"],
        out.meta["gaussian_ids"],
    )
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)

    out2 = model.render(cam, bg, is_training=True)
    assert out2.image.shape == (1, 64, 64, 3)
