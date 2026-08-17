"""CLI for Scaffold-GS on gsplat.

Subcommands:

- ``train``    : full 30k-iteration reconstruction with anchor growing/pruning.
- ``eval``     : render held-out views and report PSNR/SSIM/LPIPS.
- ``export``   : write stable anchor attributes + official-format PLY/MLPs.
- ``compress`` : (reserved) compress an exported model with HAC/HAC++ codecs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import tyro

from scaffold_gs.codec import CODECS
from scaffold_gs.config import CompressConfig, EvalConfig, ExportConfig, TrainConfig
from scaffold_gs.datasets import ColmapDataset
from scaffold_gs.model import MODELS
from scaffold_gs.trainer import evaluate, load_checkpoint, run_training, save_checkpoint
from scaffold_gs.utils import set_random_seed


def _load_for_io(cfg) -> tuple[Any, Any, int]:
    model, optim_cfg, iteration, _ = load_checkpoint(cfg.ckpt, cfg.device)
    return model, optim_cfg, iteration


def cmd_train(cfg: TrainConfig) -> None:
    set_random_seed(cfg.seed)
    run_training(cfg)


def cmd_eval(cfg: EvalConfig) -> None:
    set_random_seed(42)
    model, _, iteration = _load_for_io(cfg)
    dataset = ColmapDataset(
        data_dir=cfg.data.data_dir,
        data_factor=cfg.data.data_factor,
        test_every=cfg.data.test_every,
        white_background=cfg.data.white_background,
        preload_images=cfg.data.preload_images,
        max_width=cfg.data.max_width,
        cache_images_cpu=cfg.data.cache_images_cpu,
        device=cfg.device,
    )
    out_dir = cfg.out_dir or str(Path(cfg.data.result_dir) / "eval")
    evaluate(model, dataset, out_dir, iteration)


def cmd_export(cfg: ExportConfig) -> None:
    model, _, iteration = _load_for_io(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attrs = model.export_attributes()
    torch.save(attrs, out_dir / "attributes.pth")
    model.save_ply(out_dir / "point_cloud.ply")
    model.save_mlp_checkpoints(out_dir / "mlp_checkpoints")

    total_bytes = _tensor_bytes(attrs)
    print(
        f"[Export @{iteration}] anchors={attrs['anchor'].shape[0]} "
        f"raw_bytes≈{total_bytes / 1e6:.2f} MB -> {out_dir}"
    )


def _tensor_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.nelement() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def cmd_compress(cfg: CompressConfig) -> None:
    model, _, iteration = _load_for_io(cfg)
    out_dir = Path(cfg.out_dir)
    if cfg.codec not in CODECS:
        if cfg.codec == "hac_pp":
            from scaffold_gs.hacpp import HACPlusCodec

            CODECS["hac_pp"] = HACPlusCodec
        else:
            raise NotImplementedError(
                f"Codec '{cfg.codec}' is not implemented yet. "
                "Available: none, hac_pp."
            )
    codec = CODECS[cfg.codec]()
    metadata = codec.encode(
        model,
        out_dir,
        attr_ctx=cfg.attr_ctx,
        mask_keep_ratio=cfg.mask_keep_ratio,
    )
    metadata["iteration"] = iteration
    print(f"[Compress] {metadata}")


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {
            "train": cmd_train,
            "eval": cmd_eval,
            "export": cmd_export,
            "compress": cmd_compress,
        }
    )


if __name__ == "__main__":
    main()
