"""概念删除 MVP 冒烟测试（本地 CPU）。

覆盖 scripts/deletion_audit/ 三个模块的本地可运行部分：
  1. 锚点删除机制：逐锚点键过滤、网格参数原样保留、原 checkpoint 不被改动；
  2. SAM2 区域掩码 -> 锚点支持度投影（OpenCV 相机约定）；
  3. 稠密网格泄漏探针：精确采样 + 植入泄漏信号可被统计量识别；
  4. 残留审计打分：掩码内 PSNR 下降 / 全图 PSNR 下降、配对缺失报错。

不依赖 CUDA 扩展；HAC++ 前向路径只在服务器验证
（grid_leakage_probe.probe_features_hacpp 的接入说明见该文件）。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "deletion_audit" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attr_mod = _load("anchor_attribution")
probe_mod = _load("grid_leakage_probe")
score_mod = _load("audit_scoring")


def _make_attrs(n: int = 20, feat_dim: int = 4) -> dict:
    torch.manual_seed(0)
    positions = torch.rand(n, 3)
    positions[3] = torch.tensor([0.5, 0.5, 0.5])
    return {
        "config": {"model_name": "hac_pp"},
        "voxel_size": 0.01,
        "spatial_lr_scale": 0.1,
        "anchor": positions,
        "offset": torch.randn(n, 1, 3),
        "mask": torch.ones(n, 1),
        "anchor_feat": torch.randn(n, feat_dim),
        "scaling": torch.rand(n, 1),
        "rotation": torch.randn(n, 4),
        "opacity": torch.rand(n, 1),
        "x_bound_min": torch.zeros(3),
        "x_bound_max": torch.ones(3),
        "decoder": {
            "grid_xyz": torch.randn(7, 33),
            "plane_xy": torch.arange(5, dtype=torch.float32),
        },
    }


def test_delete_anchors_filters_per_anchor_keys_and_keeps_grid():
    attrs = _make_attrs()
    n = attrs["anchor"].shape[0]
    remove = torch.zeros(n, dtype=torch.bool)
    remove[[3, 7, 9]] = True
    grid_before = attrs["decoder"]["grid_xyz"].clone()

    new_attrs, info = attr_mod.delete_anchors(attrs, remove)

    assert info["n_removed"] == 3 and info["n_kept"] == n - 3
    for key in ("anchor", "offset", "mask", "anchor_feat", "scaling", "rotation", "opacity"):
        assert new_attrs[key].shape[0] == n - 3
        assert key in info["filtered_keys"]
    # 网格/解码器参数必须原样保留（这正是泄漏探针要测量的对象）
    assert torch.equal(new_attrs["decoder"]["grid_xyz"], grid_before)
    assert "decoder" in info["untouched_keys"] and "config" in info["untouched_keys"]
    # 原 checkpoint 不被改动
    assert attrs["anchor"].shape[0] == n


def test_sphere_selection_hits_single_planted_anchor():
    attrs = _make_attrs()
    mask = attr_mod.sphere_mask(attrs["anchor"].float(), (0.5, 0.5, 0.5), 0.01)
    assert mask.tolist() == [i == 3 for i in range(attrs["anchor"].shape[0])]


def test_projection_votes_with_opencv_camera():
    cameras = [
        {  # 相机在原点、朝 +z（OpenCV 惯例），64x64 图，f=50
            "c2w": np.eye(4),
            "K": np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]),
            "width": 64,
            "height": 64,
        },
        {  # 第二视角区域图全 0（无任何区域），不产生投票
            "c2w": np.eye(4),
            "K": np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]),
            "width": 64,
            "height": 64,
        },
    ]
    left_half = np.zeros((64, 64), dtype=np.uint16)
    left_half[:, :32] = 1  # 区域 1 = 图像左半
    maps = [left_half, np.zeros((64, 64), dtype=np.uint16)]

    anchors = torch.tensor(
        [
            [-0.2, 0.0, 5.0],  # 投影到左半 -> 区域 1
            [0.4, 0.0, 5.0],  # 投影到右半 -> 背景
            [0.0, 0.0, -5.0],  # 相机身后
            [0.0, 10.0, 1.0],  # 出画
        ]
    )
    out = attr_mod.mask_project_support_scores(anchors, cameras, maps, target_region_id=1)

    assert out["votes"].tolist() == [1, 0, 0, 0]
    assert out["votes_any"].tolist() == [1, 0, 0, 0]  # 区域图右半为 0（背景），不算"投进任意区域"
    assert out["scores"].tolist() == [0.5, 0.0, 0.0, 0.0]


def test_dense_sample_features_exact_at_voxel_centers():
    g, c = 9, 4
    grid = torch.randn(c, g, g, g)
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    centers = torch.tensor([[i / (g - 1), j / (g - 1), k / (g - 1)] for i in range(g) for j in range(g) for k in range(g)])
    feats = probe_mod.dense_sample_features(grid, lo, hi, centers)
    assert feats.shape == (g**3, c)
    expected = grid.reshape(c, -1).T  # 网格内存序 (x, y, z) 展平
    assert torch.allclose(feats, expected, atol=1e-6)


def test_probe_recovers_planted_leak_and_control_does_not_flag():
    g, c = 9, 4
    torch.manual_seed(1)
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def _run(plant: bool) -> dict:
        grid = 0.1 * torch.randn(c, g, g, g)
        if plant:
            grid[:, 1, 1, 1] = 10.0 * torch.randn(c)  # 被删区域残留高能量特征
        deleted = torch.tensor([[1 / 8.0, 1 / 8.0, 1 / 8.0], [2 / 8.0, 1 / 8.0, 1 / 8.0]])
        surviving = torch.rand(64, 3)
        f_del = probe_mod.dense_sample_features(grid, lo, hi, deleted)
        f_sur = probe_mod.dense_sample_features(grid, lo, hi, surviving)
        return probe_mod.leakage_statistics(f_del, f_sur)

    leak = _run(plant=True)
    clean = _run(plant=False)
    # 植入泄漏：被删区域特征能量显著高于存活区域
    assert leak["norm_ratio_deleted_over_surviving"] > 3.0
    # 干净对照：两侧特征能量同量级（不触发泄漏判读）
    assert 0.5 < clean["norm_ratio_deleted_over_surviving"] < 2.0
    assert set(leak) >= {"norm_deleted_mean", "max_cos_deleted_to_surviving_mean"}


def test_audit_scoring_masked_drop_smaller_than_full_and_missing_pair_raises(tmp_path):
    rng = np.random.default_rng(0)
    dir_o, dir_d, dir_m = tmp_path / "o", tmp_path / "d", tmp_path / "m"
    for d in (dir_o, dir_d, dir_m):
        d.mkdir()
    for name in ("view0.png", "view1.png"):
        base = rng.random((32, 32, 3))
        perturbed = base.copy()
        perturbed[:16] += rng.random((16, 32, 3)) * 0.2  # 只扰动掩码区域（上半图）
        Image.fromarray((base * 255).astype(np.uint8)).save(dir_o / name)
        Image.fromarray((np.clip(perturbed, 0, 1) * 255).astype(np.uint8)).save(dir_d / name)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:16] = 255
        Image.fromarray(mask).save(dir_m / name)

    results = score_mod.score_pair_set(dir_o, dir_d, dir_m)
    assert len(results) == 2
    for r in results:
        assert r["psnr_masked_original_vs_deleted"] < r["psnr_full_original_vs_deleted"]
        assert 0.4 < r["mask_fraction"] < 0.6

    (dir_d / "view1.png").unlink()
    with pytest.raises(FileNotFoundError):
        score_mod.score_pair_set(dir_o, dir_d, dir_m)
