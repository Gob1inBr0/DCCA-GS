import json
from pathlib import Path
import sys

import torch

from hacplus.utils.encodings_cuda import decoder
from scaffold_gs.config import ModelConfig
from scaffold_gs.hacpp import HACPlusCodec, HACPlusModel

root, audit_root = (Path(argument) for argument in sys.argv[1:])
audit = {}
for arm in ("control_initial", "merged_initial", "smoke", "control", "merged"):
    source = root / arm / "bitstreams"
    dest = audit_root / arm
    dest.mkdir()
    original = torch.load(source / "attributes.pth", map_location="cpu", weights_only=False)
    compact = {k: original[k] for k in ("config", "voxel_size", "spatial_lr_scale")}
    compact["networks"] = {k: v for k, v in original["decoder"].items() if k != "encoding_xyz"}
    del original
    package = dest / "decoder_config_and_weights.pth"
    torch.save(compact, package)
    for f in source.iterdir():
        if f.suffix == ".b" or f.name in ("codec_header.json", "content_aware_q_meta.json",
                                         "xyz_gpcc.npz", "x_bound_min.pkl", "x_bound_max.pkl"):
            (dest / f.name).symlink_to(f)

    config = torch.load(package, map_location="cpu", weights_only=False)
    new = HACPlusModel(ModelConfig(**config["config"]), "cuda")
    new.voxel_size = config["voxel_size"]
    new.spatial_lr_scale = config["spatial_lr_scale"]
    new._view.load_decoder_state({**config["networks"], "encoding_xyz": new.core.encoding_xyz.state_dict()})
    # Attribute probability prediction in the original implementation reads
    # the hash model before decode_attributes installs the decoded hash.
    # Install the binary hash received in the stream before predicting.
    hash_bits = decoder(new.core.get_encoding_params().numel(), str(dest / "hash.b"), "cuda")
    new._view.set_hash_params((hash_bits.float() * 2 - 1).view(-1, new.core.n_features_per_level))
    new.decode_attributes(dest)

    standard = HACPlusCodec().decode(source)
    state_names = ("_anchor", "_offset", "_anchor_feat", "_mask", "_scaling")
    for name in state_names:
        assert torch.equal(getattr(standard.core, name), getattr(new.core, name)), name
    assert torch.equal(standard.core.get_encoding_params(), new.core.get_encoding_params())
    audit[arm] = {
        "compact_bytes": sum(x.stat().st_size for x in dest.iterdir()),
        "decoder_config_and_weights_bytes": package.stat().st_size,
        "coded_anchors": new.num_anchors,
        "active_offsets": int(new.core.get_mask.sum().item()),
        "reconstruction_matches_standard": True,
        "artifact_files": len(list(dest.iterdir())),
    }
    print(arm, json.dumps(audit[arm]), flush=True)
    del new, standard
(audit_root / "results.json").write_text(json.dumps(audit, indent=2) + "\n")
