"""Section 3.5 minimal experiment: block-based decoding overhead/benefit.

Usage (5090, HAC_5090_a100 env):

    python scripts/block_decode_experiment.py \
        --artifact-dir runs/mlp_quant_sens_cd8_rest16_110k/b8/bitstreams \
        --data-dir /home/fansonglin/xieliang/chentong/CT_HAC_v1/data/4-28 \
        --max-width 1600 --data-factor 1 --test-every 8 \
        --ks 4 16 64 --render-views 12 \
        --out runs/block_decode_exp/results.json

What it measures (不改训练):
  * 做法 A: anchors split into K Morton-order blocks; per-view visible-block
    ratio via conservative frustum-sphere culling (all val views) and via the
    exact rasterizer prefilter (render subset); full-decode time + peak CUDA
    memory as the lazy-decode baseline; full vs block-culled render time and
    PSNR difference (culling is exact, so PSNR loss should be 0).
  * 做法 B: per-block independent hash + header accounting from hac_meta
    (duplicated (K-1) times), reported as total_MB increment.

The measured render saving is conservative: both paths still run the full
prefilter pass; a real per-block pipeline would also skip prefilter on
invisible blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.hacpp import HACPlusCodec
from scaffold_gs.renderer import _empty_output


def patch_legacy_8dim_complexity() -> None:
    """Decode pre-4D bitstreams (formula_decoder_available_v1, 8-dim input).

    The last four image-side statistics are constant zeros in the legacy
    formula mode, so their weight columns are dead; truncating to the first
    four columns reproduces the encoder's complexity logits bit-exactly.
    """
    from scaffold_gs import hac_core

    orig = hac_core.HACCoreView.load_decoder_state

    def patched(self, state):
        cs = state.get("mlp_complexity")
        if cs is not None and cs.get("0.weight") is not None:
            w = cs["0.weight"]
            if w.shape[-1] == 8:
                cs = dict(cs)
                cs["0.weight"] = w[:, :4].contiguous()
                state = dict(state)
                state["mlp_complexity"] = cs
        return orig(self, state)

    hac_core.HACCoreView.load_decoder_state = patched

    import hacplus.utils.codec_consistency as cc

    cc.FORMULA_INPUT_VERSION = "formula_decoder_available_v1"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def frustum_planes(
    viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int,
    near: float, far: float,
) -> torch.Tensor:
    """Extract 6 normalized OpenGL-style frustum planes from world->camera."""
    view = viewmat[0].to(torch.float64)
    K = K[0].to(torch.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    proj = torch.zeros(4, 4, dtype=torch.float64, device=view.device)
    proj[0, 0] = fx
    proj[1, 1] = fy
    proj[0, 2] = cx
    proj[1, 2] = cy
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    m = proj @ view  # [4,4] world -> clip
    planes = torch.stack(
        [
            m[3] + m[0],
            m[3] - m[0],
            m[3] + m[1],
            m[3] - m[1],
            m[3] + m[2],
            m[3] - m[2],
        ]
    )
    norms = planes[:, :3].norm(dim=1)
    return (planes / norms[:, None]).to(torch.float32)


def sphere_visible(planes: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Conservative sphere-vs-frustum test. planes [6,4], centers [B,3]."""
    b = centers.shape[0]
    hom = torch.cat([centers, torch.ones(b, 1, device=centers.device)], dim=1)
    d = hom @ planes.T  # [B,6]
    return (d > -radii[:, None]).all(dim=1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def render_timed(
    model, camera, background, block_ids: torch.Tensor,
    block_vis: torch.Tensor | None = None,
):
    """Render one camera, returning (output, prefilter_s, total_s).

    ``block_vis`` optionally restricts generation/rasterization to anchors in
    visible blocks (exact culling: the restricted set is a superset of the
    prefilter-visible set, so the image is identical). The prefilter pass
    still runs over all anchors, which keeps the comparison conservative.
    """
    from gsplat.rendering import rasterization

    t0 = time.perf_counter()
    visible_full = model.prefilter_anchors(camera)
    _sync()
    t_pre = time.perf_counter() - t0

    if visible_full.sum() == 0:
        return _empty_output(camera, model.device, model.cfg.n_offsets), t_pre, t_pre
    if block_vis is None:
        restrict = visible_full
    else:
        restrict = visible_full & block_vis[block_ids]
    if restrict.sum() == 0:
        return _empty_output(camera, model.device, model.cfg.n_offsets), t_pre, t_pre

    t0 = time.perf_counter()
    gaussians = model.generate_gaussians(
        camera, visible_mask=restrict, is_training=False
    )
    viewmats, Ks = camera.to_gsplat(model.device)
    render_colors, render_alphas, meta = rasterization(
        means=gaussians.xyz,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
        viewmats=viewmats,
        Ks=Ks,
        width=camera.width,
        height=camera.height,
        near_plane=camera.near_plane,
        far_plane=camera.far_plane,
        backgrounds=background,
        render_mode="RGB",
        packed=True,
        sparse_grad=False,
        sh_degree=None,
        tile_size=int(getattr(model.cfg, "tile_size", 16)),
    )
    _sync()
    t_gr = time.perf_counter() - t0

    out = _empty_output(camera, model.device, model.cfg.n_offsets)
    out.image = render_colors
    out.alpha = render_alphas
    out.meta = meta
    out.gaussians = gaussians
    out.visible_mask = restrict
    return out, t_pre, t_pre + t_gr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--max-width", type=int, default=None)
    p.add_argument("--data-factor", type=int, default=1)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64])
    p.add_argument("--render-views", type=int, default=12)
    p.add_argument("--render-repeats", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--legacy-complexity-8dim",
        action="store_true",
        help="Decode pre-4D bitstreams (8-dim complexity input, v1 header).",
    )
    p.add_argument("--out", default="runs/block_decode_exp/results.json")
    args = p.parse_args()

    device = torch.device(args.device)
    artifact_dir = Path(args.artifact_dir)
    if args.legacy_complexity_8dim:
        patch_legacy_8dim_complexity()

    torch.cuda.reset_peak_memory_stats(device)
    codec = HACPlusCodec()
    t0 = time.perf_counter()
    model = codec.decode(artifact_dir)
    _sync()
    decode_s = time.perf_counter() - t0
    decode_peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    dataset = ColmapDataset(
        data_dir=args.data_dir,
        data_factor=args.data_factor,
        test_every=args.test_every,
        white_background=False,
        preload_images=False,
        max_width=args.max_width,
        cache_images_cpu=False,
        device=args.device,
    )
    val_cams = dataset.val_cameras
    background = dataset.background
    anchors = model.core.get_anchor.detach()
    n = anchors.shape[0]
    print(f"[block_decode] decoded {n} anchors; {len(val_cams)} val views")

    # ---- 做法 B: per-block hash + header accounting (from hac_meta) ----
    meta = json.loads((artifact_dir / "hac_meta.json").read_text())
    bit2mb = 8 * 1024 * 1024
    bit_hash = int(meta["bit_hash"])
    bit_header = int(meta["bit_header"])
    bit_mlp_float = int(meta["bit_mlp"])
    mlp_quant_b = (artifact_dir / "mlp_quant.bin").stat().st_size
    mlp_quant_meta_b = (artifact_dir / "mlp_quant_meta.json").stat().st_size
    quant_total_bits = (
        int(meta["total_bits"]) - bit_mlp_float + 8 * (mlp_quant_b + mlp_quant_meta_b)
    )
    base_mb = quant_total_bits / bit2mb
    overhead_bits = bit_hash + bit_header
    method_b = []
    for k in args.ks:
        inc_bits = (k - 1) * overhead_bits
        method_b.append(
            {
                "K": k,
                "hash_bits": bit_hash,
                "header_bits": bit_header,
                "overhead_bits_per_block": overhead_bits,
                "increment_bits": inc_bits,
                "increment_MiB": round(inc_bits / bit2mb, 6),
                "total_MB": round((quant_total_bits + inc_bits) / bit2mb, 4),
                "pct_increment": round(100.0 * inc_bits / quant_total_bits, 4),
            }
        )
        print(
            f"[做法B] K={k}: +{method_b[-1]['increment_MiB']:.4f} MiB "
            f"(+{method_b[-1]['pct_increment']:.2f}%), "
            f"total {method_b[-1]['total_MB']:.4f} MiB"
        )

    # ---- 做法 A: Morton-order blocks + visibility ----
    results = {
        "scene": str(args.data_dir),
        "bitstream": str(artifact_dir),
        "n_anchors": n,
        "n_val_views": len(val_cams),
        "decode_s": round(decode_s, 3),
        "decode_peak_memory_MiB": round(decode_peak_mib, 1),
        "base_total_MB_quant": round(base_mb, 4),
        "hash_header_pct_of_total": round(
            100.0 * overhead_bits / quant_total_bits, 4
        ),
        "method_B": method_b,
        "method_A": {},
    }

    render_cams = val_cams[: args.render_views]
    for k in args.ks:
        block_size = math.ceil(n / k)
        n_blocks = math.ceil(n / block_size)
        block_ids = torch.arange(n, device=device) // block_size
        block_centers = torch.stack(
            [anchors[block_ids == b].mean(0) for b in range(n_blocks)]
        )
        block_radii = torch.stack(
            [
                (anchors[block_ids == b] - block_centers[b]).norm(dim=1).max()
                for b in range(n_blocks)
            ]
        )

        # Geometric visible ratio over all val views (cheap, no rendering).
        vis_fracs_geo = []
        for cam in val_cams:
            viewmats, Ks = cam.to_gsplat(device)
            planes = frustum_planes(
                viewmats, Ks, cam.width, cam.height, cam.near_plane, cam.far_plane
            )
            vis = sphere_visible(planes, block_centers, block_radii)
            vis_fracs_geo.append(float(vis.float().mean().item()))

        # Exact visible ratio + render timing on the render subset.
        vis_fracs_exact = []
        vis_anchor_fracs = []
        t_pre_list, t_full_list, t_cull_list = [], [], []
        n_anchor_in_vis_blocks = []
        psnr_diffs = []
        for cam in render_cams:
            out_full, t_pre, t_full = render_timed(model, cam, background, block_ids)
            vis_full = out_full.visible_mask
            vis_anchor_fracs.append(float(vis_full.float().mean().item()))
            vis_exact = torch.zeros(n_blocks, dtype=torch.bool, device=device)
            vis_exact[block_ids[vis_full]] = True
            vis_fracs_exact.append(float(vis_exact.float().mean().item()))
            n_anchor_in_vis_blocks.append(
                int(torch.bincount(block_ids, minlength=n_blocks)[vis_exact].sum().item())
            )

            out_cull, t_pre_c, t_cull = render_timed(
                model, cam, background, block_ids, vis_exact
            )
            t_pre_list.append(t_pre)
            t_full_list.append(t_full)
            t_cull_list.append(t_cull)
            psnr_diffs.append(psnr(out_full.image, out_cull.image))

        mean_t_pre = sum(t_pre_list) / len(t_pre_list)
        mean_t_full = sum(t_full_list) / len(t_full_list)
        mean_t_cull = sum(t_cull_list) / len(t_cull_list)
        mean_anchor_frac = (
            sum(n_anchor_in_vis_blocks) / len(n_anchor_in_vis_blocks) / n
        )
        # Hypothetical lazy pipeline: prefilter only anchors in visible blocks
        # (cost scales with anchor count, conservative) + culled generate/raster.
        t_lazy = mean_t_pre * mean_anchor_frac + (
            mean_t_cull - mean_t_pre
        )
        print(f"[做法A] K={k}: geo_visible={sum(vis_fracs_geo)/len(vis_fracs_geo):.3f} "
              f"exact_visible={sum(vis_fracs_exact)/len(vis_fracs_exact):.3f} "
              f"visible_anchor={sum(vis_anchor_fracs)/len(vis_anchor_fracs):.3f} "
              f"psnr_loss={max(psnr_diffs):.4f} "
              f"render {mean_t_full:.3f}s -> lazy {t_lazy:.3f}s")

        results["method_A"][str(k)] = {
            "n_blocks": n_blocks,
            "block_size": block_size,
            "visible_ratio_geometric_mean": round(
                sum(vis_fracs_geo) / len(vis_fracs_geo), 4
            ),
            "visible_ratio_exact_mean": round(
                sum(vis_fracs_exact) / len(vis_fracs_exact), 4
            ),
            "visible_anchor_fraction_exact_mean": round(
                sum(vis_anchor_fracs) / len(vis_anchor_fracs), 4
            ),
            "visible_anchor_fraction_exact_min": round(
                min(vis_anchor_fracs), 4
            ),
            "visible_ratio_geometric_min": round(min(vis_fracs_geo), 4),
            "visible_ratio_geometric_max": round(max(vis_fracs_geo), 4),
            "decode_time_reduction_ceiling_pct": round(
                100.0
                * (1.0 - sum(vis_fracs_geo) / len(vis_fracs_geo)),
                2,
            ),
            "render_full_s": round(mean_t_full, 4),
            "render_culled_s": round(mean_t_cull, 4),
            "prefilter_s": round(mean_t_pre, 4),
            "anchor_fraction_in_visible_blocks": round(mean_anchor_frac, 4),
            "render_speedup_pct_conservative": round(
                100.0
                * (mean_t_full - mean_t_cull)
                / max(mean_t_full, 1e-9),
                2,
            ),
            "render_speedup_pct_lazy_pipeline": round(
                100.0 * (mean_t_full - t_lazy) / max(mean_t_full, 1e-9), 2
            ),
            "psnr_diff_max_dB": round(max(psnr_diffs), 6),
            "psnr_diff_mean_dB": round(sum(psnr_diffs) / len(psnr_diffs), 6),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
