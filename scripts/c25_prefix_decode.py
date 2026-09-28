#!/usr/bin/env python3
"""Standalone prefix decoder for the c25 layered bitstream.

Consumes ONLY the .bin file written by c25_real_bitstream.py (v1/v2/v3-s5
formats) plus the trained run directory — everything needed to turn a file
prefix into decoded symbols (conditional tables, params, s5 flag-head
weights, ladder layout) is read from the file itself. This is the honest
end-to-end check of the single-file truncatable-stream contract: the
in-encoder roundtrip asserts cover the encoder's own bookkeeping, this
script covers what a receiver would actually do.

Per quality prefix P_p (all fields at level min(p, len_f-1)) it:
  1. decodes the file's chunks (optionally from a PHYSICALLY truncated
     copy — bytes past the prefix are cut off before parsing),
  2. reconstructs symbols by telescoping and dequantizes with the
     production grid Q (from the codec decode, same source as the encoder),
  3. renders N views and computes PSNR, byte-for-byte accounting included,
  4. at full precision asserts the stream symbols equal round(t/Q) of the
     production decode (the strongest "same model" check), and compares
     bytes/PSNR against the encoder JSON when --reference is given.

Run on the training machine (needs scaffold_gs + constriction + torch):
  PYTHONPATH=/home/project2/c25_pylib2 python scripts/c25_prefix_decode.py \
      --bin analysis/s2_prefix_sweep/c25_real_bitstream.bin \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --data-dir <data> --stats <anchor_stats.npz> --field-aware \
      --reference analysis/s2_prefix_sweep/c25_real_bitstream.json
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from c25_container import (prefix_byte_length, read_container,  # noqa: E402
                           split_params)
from c25_real_bitstream import (N_VIEWS, S5FlagHead, _s5_pflag,  # noqa: E402
                                area_rank_groups, bucket_of, decode_chunk,
                                per_anchor_step_tiers, uniform_groups)


def load_s5_heads(blob, dev):
    """state-dict blob -> {level int: S5FlagHead}."""
    states = torch.load(io.BytesIO(blob), map_location="cpu",
                        weights_only=True)
    heads = {}
    for k, sd in states.items():
        h = S5FlagHead()
        h.load_state_dict(sd)
        heads[int(k)] = h.eval().to(dev)
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", dest="bin_in", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", default=None,
                    help="anchor_stats.npz; grouping must match the encoder "
                         "run (contribution-area vs uniform fallback)")
    ap.add_argument("--mapped", default=None,
                    help="decoded->trained row map; auto-generated from the "
                         "checkpoint when missing (same cache as encoder)")
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--views", type=int, default=None,
                    help="render this many val views (default: encoder's 16)")
    ap.add_argument("--prefixes", default="all",
                    help="comma-separated prefix indices, or 'all'")
    ap.add_argument("--truncate", action="store_true",
                    help="physically cut the file at each prefix boundary "
                         "and decode from the truncated bytes")
    ap.add_argument("--reference", default=None,
                    help="encoder JSON (c25_real_bitstream.py --out) to "
                         "compare bytes and PSNR against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scaffold_gs.datasets import ColmapDataset
    from scaffold_gs.hacpp import HACPlusCodec

    bin_path = Path(args.bin_in)
    header, blob, file_chunks = read_container(bin_path)
    n_groups = int(header["groups"])
    fields = tuple(header["fields"])
    fsteps = {f: [int(s) for s in header["steps"][f]] for f in fields}
    s5_levels = set(header.get("s5_levels", []))
    s5_heads = {}
    if blob:
        s5_heads = load_s5_heads(blob, args.device)
        assert set(s5_heads) == s5_levels, "blob/header level mismatch"
    assert len(file_chunks) == len(header["chunks"]), \
        f"file has {len(file_chunks)} chunks, header says " \
        f"{len(header['chunks'])} — truncated file?"
    print(f"[PD] format {header['format']} groups={n_groups} "
          f"steps={fsteps} s5_levels={sorted(s5_levels)}", flush=True)

    # ---- model + production grid Q (same source as the encoder) ----
    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_trained = int(meta["num_anchors_total"])
    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    del codec
    n_alive = model.num_anchors
    dev = model.device if args.device == "cuda" else torch.device(args.device)
    view = model._view
    dec_t = {
        "feat": view.anchor_feat.data.detach().float(),
        "scaling": view.scaling.data.detach().float(),
        "offset": view.offset.data.detach().float(),
    }
    last_q = getattr(model.core, "last_decode_Q", None)
    assert last_q is not None, "codec did not expose last_decode_Q — this " \
        "decoder requires the production-grid Q (post-audit codec)"
    Q_t = {
        "feat": last_q["feat"].to(dev).float(),
        "scaling": last_q["scaling"].to(dev).float(),
        "offset": last_q["offsets"].to(dev).float().reshape(
            dec_t["offset"].shape),
    }
    run_tag = run_dir.name

    # ---- grouping + step tiers: same helpers, same inputs ----
    if args.stats and Path(args.stats).exists():
        area = np.load(args.stats)["area"].astype(np.float64)
        assert area.shape[0] == n_trained
        group_of_trained = area_rank_groups(area, args.groups)
    else:
        group_of_trained = None
        print("[PD] WARNING: no stats — uniform-bucket grouping; must match "
              "how the encoder was run", flush=True)
    if args.mapped and Path(args.mapped).exists():
        mapped = np.load(args.mapped).astype(np.int64)
    else:
        ck = torch.load(run_dir / "ckpts" / "ckpt_30000.pth",
                        map_location="cpu", weights_only=False)
        trained_xyz = ck["model_state"]["_anchor"].to(dev)
        assert trained_xyz.shape[0] == n_trained
        decoded_xyz = model.core.get_anchor
        mapped = np.empty(decoded_xyz.shape[0], dtype=np.int64)
        with torch.no_grad():
            for start in range(0, decoded_xyz.shape[0], 1024):
                end = min(start + 1024, decoded_xyz.shape[0])
                d = torch.cdist(decoded_xyz[start:end], trained_xyz)
                mapped[start:end] = d.min(dim=1)[1].cpu().numpy()
                del d
        mp = bin_path.parent / (run_tag + ".mapped.npy")
        if not mp.exists():
            np.save(mp, mapped)
    assert n_alive == mapped.size
    g_base_alive = (group_of_trained[mapped] if group_of_trained is not None
                    else uniform_groups(n_alive, args.groups))

    sym = {f: torch.round(dec_t[f] / Q_t[f]) for f in fields}
    q_flat = {f: sym[f].cpu().numpy().astype(np.int32).reshape(-1)
              for f in fields}
    logq_flat = {f: np.log(Q_t[f].cpu().numpy().reshape(-1)
                           ).astype(np.float32) for f in fields}
    out_shape = {f: tuple(sym[f].shape) for f in fields}
    g_flat = {}
    for f in fields:
        tiers = per_anchor_step_tiers(Q_t[f].cpu().numpy().reshape(-1),
                                      n_alive)
        n_cols = q_flat[f].size // n_alive
        g_flat[f] = np.repeat(g_base_alive * 3 + tiers, n_cols)

    # ---- fixed payload accounting (same convention as the encoder) ----
    fixed_bytes = (
        int(meta["bit_hash"]) + int(meta["bit_mlp"]) + int(meta["bit_bounds"])
        + int(meta["bit_header"]) + int(meta["bit_masks"])
    ) / 8.0
    geom_bytes = int(meta["bit_anchor"]) / 8.0
    hlen = int.from_bytes(bin_path.read_bytes()[:4], "little")
    header_total = 4 + hlen
    blob_overhead = (4 + len(blob)) if blob else 0
    s5_weight_bytes = len(blob) if blob else 0

    # file-order chunks indexed by (field, level)
    by_key = {}
    for c_hdr, c_body in zip(header["chunks"], file_chunks):
        by_key[(c_hdr["field"], c_hdr["level"])] = c_body
    for f in fields:
        for li in range(len(fsteps[f])):
            assert (f, li) in by_key, f"missing chunk ({f}, L{li}) in file"

    prefixes = ([int(x) for x in args.prefixes.split(",")]
                if args.prefixes != "all"
                else list(range(max(len(fsteps[f]) for f in fields))))

    dataset = ColmapDataset(
        data_dir=args.data_dir, data_factor=args.data_factor, test_every=8,
        max_width=args.max_width, white_background=False,
        preload_images=False, device=str(dev),
    )
    val = dataset.val_cameras
    n_views = args.views or N_VIEWS
    pick = np.unique(np.linspace(0, len(val) - 1, n_views).astype(int))
    cams = [val[i] for i in pick]
    background = dataset.background

    def decode_prefix(p, container_path):
        """Decode prefix p from container_path; returns {field: (acc, level)}."""
        hdr, blb, chs = read_container(container_path)
        # a physically truncated file keeps the full header but only the
        # complete chunks up to the cut: bodies are a head of header order
        assert len(chs) <= len(hdr["chunks"])
        body_by_key = {}
        for c_hdr, c_body in zip(hdr["chunks"], chs):
            body_by_key[(c_hdr["field"], c_hdr["level"])] = c_body
        heads = load_s5_heads(blb, args.device) if blb else {}
        acc = {}
        for f in fields:
            lf = min(p, len(fsteps[f]) - 1)
            a = None
            for li in range(lf + 1):
                body = body_by_key[(f, li)]
                q = q_flat[f]
                if li == 0:
                    layer = np.round(q / fsteps[f][0]).astype(np.int32)
                    ctx = None
                    g_chunk = g_flat[f] // 3
                else:
                    ratio = fsteps[f][li - 1] // fsteps[f][li]
                    layer = (np.round(q / fsteps[f][li]).astype(np.int32)
                             - ratio * a)
                    ctx = bucket_of(a)
                    g_chunk = g_flat[f]
                params = split_params(body["params_raw"], n_groups)
                flag = None
                if heads and f == "offset" and li in heads:
                    flag = _s5_pflag(heads[li], a, logq_flat[f], li,
                                     args.device)
                back = decode_chunk(body["data"], g_chunk, params, ctx,
                                    s4=None, s5_flag=flag)
                assert (back == layer).all(), \
                    f"stream decode mismatch {f} L{li} (prefix {p})"
                a = back if li == 0 else ratio * a + back
            acc[f] = (a, lf)
        return acc

    results = []
    for p in prefixes:
        t0 = time.time()
        if args.truncate:
            cut = prefix_byte_length(bin_path, p)
            tmp = bin_path.with_suffix(f".p{p}.trunc.bin")
            tmp.write_bytes(bin_path.read_bytes()[:cut])
            acc = decode_prefix(p, tmp)
            tmp.unlink()
        else:
            acc = decode_prefix(p, bin_path)

        # full-precision hard contract: stream symbols == production symbols
        if p == max(len(fsteps[f]) - 1 for f in fields):
            for f in fields:
                acc_f, lf = acc[f]
                assert lf == len(fsteps[f]) - 1
                assert np.array_equal(acc_f, q_flat[f]), \
                    f"full-precision stream symbols != production symbols ({f})"

        deq = {}
        for f in fields:
            a, lf = acc[f]
            xhat = (fsteps[f][lf] * a.reshape(out_shape[f])
                    * Q_t[f].cpu().numpy()).astype(np.float32)
            deq[f] = torch.from_numpy(xhat).to(dev)
        view.anchor_feat = torch.nn.Parameter(deq["feat"].contiguous(),
                                              requires_grad=False)
        view.scaling = torch.nn.Parameter(deq["scaling"].contiguous(),
                                          requires_grad=False)
        view.offset = torch.nn.Parameter(deq["offset"].contiguous(),
                                         requires_grad=False)
        psnrs = []
        with torch.no_grad():
            for cam in cams:
                out = model.render(cam, background, is_training=False,
                                   appearance_id=0, step=30000)
                pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
                gt = dataset.get_image(cam).clamp(0.0, 1.0)
                mse = torch.mean((pred - gt) ** 2).item()
                psnrs.append(float("inf") if mse <= 0
                             else -10.0 * np.log10(mse))
                del out

        # byte accounting, prefix-consistent (params only for present chunks)
        cum = 0
        par = 0
        for f in fields:
            lf = min(p, len(fsteps[f]) - 1)
            for li in range(lf + 1):
                body = by_key[(f, li)]
                cum += len(body["data"])
                locs, p0s, bts = split_params(body["params_raw"], n_groups)
                par += (8 + locs.astype(np.int32).tobytes().__len__()
                        + p0s.astype(np.float32).tobytes().__len__()
                        + bts.astype(np.float32).tobytes().__len__())
        real_bytes = (header_total + fixed_bytes + geom_bytes + par
                      + s5_weight_bytes + blob_overhead + cum)
        entry = {
            "prefix": p,
            "field_levels": {f: int(min(p, len(fsteps[f]) - 1))
                             for f in fields},
            "coded_bytes": int(cum),
            "real_bytes": round(real_bytes, 1),
            "psnr_mean": round(float(np.mean(psnrs)), 4),
            "render_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(f"[PD] prefix P{p}: REAL {real_bytes/1e6:6.2f} MB  "
              f"PSNR {entry['psnr_mean']:.3f}"
              f"{' (truncated-file decode)' if args.truncate else ''}",
              flush=True)
        del deq
        torch.cuda.empty_cache()

    verdict = {"bin": str(bin_path), "run": str(run_dir),
               "truncate": bool(args.truncate), "views": n_views,
               "s5_levels": sorted(s5_levels), "results": results}

    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        rmap = {r["prefix"]: r for r in ref["results"]}
        checks = []
        for r in results:
            rr = rmap.get(r["prefix"])
            if rr is None:
                continue
            checks.append({
                "prefix": r["prefix"],
                "bytes_match": abs(rr["real_bytes"] - r["real_bytes"]) <= 0.5,
                "bytes_ref": rr["real_bytes"], "bytes_dec": r["real_bytes"],
                "psnr_ref": rr["psnr_mean"], "psnr_dec": r["psnr_mean"],
                "psnr_abs_diff": round(abs(rr["psnr_mean"] - r["psnr_mean"]),
                                       4),
            })
        ok = all(c["bytes_match"] and c["psnr_abs_diff"] < 0.02
                 for c in checks)
        verdict["reference_checks"] = checks
        verdict["reference_verdict"] = "PASS" if ok else "FAIL"
        print(f"[PD] reference comparison: "
              f"{verdict['reference_verdict']} "
              f"(max dPSNR {max((c['psnr_abs_diff'] for c in checks), default=0):.4f})",
              flush=True)

    out = args.out or str(bin_path.with_suffix(".prefix_decode.json"))
    Path(out).write_text(json.dumps(verdict, indent=1))
    print(f"[PD] wrote {out}")


if __name__ == "__main__":
    main()
