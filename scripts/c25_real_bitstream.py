#!/usr/bin/env python3
"""S3/priority-1: REAL layered bitstream — range-coded, prefix-decodable.

Upgrades the C2.5 accounting experiment to a real file: the three quantized
fields (feat/scaling/offsets) are re-encoded as a nested quality ladder with
an actual range coder (constriction; per-symbol discrete-Gaussian models
parameterized by the 32 contribution groups — the C2.5 zero-training
conditional). The bitstream layout is

  [4B header len][header json] then per chunk (ladder order):
  [4B params len][params: 32x(loc,b) float32][4B data len][data]

so truncating after any chunk boundary yields a valid decodable prefix.
Real cumulative bytes at each boundary are paired with a real render PSNR
computed from bitstream-DECODED symbols — the measured "cumulative bytes ->
PSNR" curve (journal-axis priority 1).

Ladder (power-of-two steps, telescoping reconstruction):
  chunk L0: q0 = round(q/16)                 -> dequant 16*Q*q0
  chunk L1: r1 = q1 - 2*q0, q1 = round(q/8)  -> acc = 2*acc + r1
  chunk L2: r2 = q2 - 2*q1, q2 = round(q/4)  -> acc = 2*acc + r2
  chunk L3: r3 = q3 - 2*q2, q3 = round(q/2)  -> acc = 2*acc + r3
  chunk L4: r4 = q  - 2*q3                   -> acc = q   (exact decoded grid)

Run under the DCCA env python with PYTHONPATH=/home/project2/c25_pylib2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import constriction  # noqa: E402

N_VIEWS = 16
GROUPS = 32
FACTORS = [16, 8, 4, 2]
STEPS = FACTORS + [1]
N_LEVELS = len(STEPS)
FAMILY_RANGE = 4096


def fit_laplace(symbols):
    s = np.asarray(symbols, dtype=np.float64)
    loc = float(np.median(s))
    m = float(np.mean(np.abs(s - loc)))
    if m <= 1e-9:
        return loc, 1e-3
    beta = (np.sqrt(1.0 + 4.0 * m * m) - 1.0) / (2.0 * m)
    b = -1.0 / np.log(max(min(float(beta), 1.0 - 1e-9), 1e-12))
    return loc, max(b, 1e-3)


def model_params(layer, g):
    locs = np.empty(GROUPS, dtype=np.float64)
    bs = np.empty(GROUPS, dtype=np.float64)
    for gid in range(GROUPS):
        locs[gid], bs[gid] = fit_laplace(layer[g == gid])
    return locs[g].astype(np.float32), bs[g].astype(np.float32), (locs, bs)


def family():
    return constriction.stream.model.QuantizedGaussian(-FAMILY_RANGE, FAMILY_RANGE)


def code_chunk(layer, g):
    means, stds, params = model_params(layer, g)
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(layer.astype(np.int32), family(), means, stds)
    comp = enc.get_compressed()
    dec = constriction.stream.queue.RangeDecoder(comp)
    back = dec.decode(family(), means, stds)
    exact = bool((back == layer).all())
    return comp.tobytes(), means, stds, params, back.astype(np.int32), exact


def decode_chunk(data, g, params):
    locs, bs = params
    means = locs[g].astype(np.float32)
    stds = bs[g].astype(np.float32)
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(data, dtype=np.uint32))
    return dec.decode(family(), means, stds).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--mapped",
                    default="analysis/s2_prefix_sweep/prefix_sweep.mapped.npy")
    ap.add_argument("--s2", default="analysis/s2_prefix_sweep/prefix_sweep.json")
    ap.add_argument("--out", default="analysis/s2_prefix_sweep/c25_real_bitstream.json")
    ap.add_argument("--bin", dest="bin_out",
                    default="analysis/s2_prefix_sweep/c25_real_bitstream.bin")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from scaffold_gs.hacpp import HACPlusCodec
    from scaffold_gs.datasets import ColmapDataset

    run_dir = Path(args.run)
    bit_dir = run_dir / "bitstreams"
    meta = json.loads((bit_dir / "hac_meta.json").read_text())
    n_trained = int(meta["num_anchors_total"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = np.load(args.stats)
    area = stats["area"].astype(np.float64)
    assert area.shape[0] == n_trained
    rank = np.empty(n_trained, dtype=np.int64)
    rank[np.argsort(-area, kind="stable")] = np.arange(n_trained)
    group_of = np.minimum(rank * GROUPS // n_trained, GROUPS - 1).astype(np.int16)

    mapped = np.load(args.mapped).astype(np.int64)

    codec = HACPlusCodec()
    model = codec.decode(bit_dir)
    model.eval()
    del codec
    n_alive = model.num_anchors
    dev = model.device
    assert n_alive == mapped.size
    view = model._view
    dec_t = {
        "feat": view.anchor_feat.data.detach().float(),
        "scaling": view.scaling.data.detach().float(),
        "offset": view.offset.data.detach().float(),
    }
    z = np.load(out_path.parent / "c25_q_cache.npz")
    Q_t = {
        "feat": torch.from_numpy(z["Q_feat"]).to(dev),
        "scaling": torch.from_numpy(z["Q_scaling"]).to(dev),
        "offset": torch.from_numpy(z["Q_offsets"]).to(dev),
    }
    print(f"[RB] model decoded (anchors={n_alive}), Q loaded", flush=True)

    fields = ("feat", "scaling", "offset")
    sym = {f: torch.round(dec_t[f] / Q_t[f]) for f in fields}
    q_flat, g_flat, out_shape = {}, {}, {}
    for f in fields:
        qf = sym[f].cpu().numpy().astype(np.int32).reshape(-1)
        n_cols = qf.size // n_alive
        q_flat[f] = qf
        g_flat[f] = np.repeat(group_of[mapped], n_cols)
        out_shape[f] = tuple(sym[f].shape)

    # ---- range-code the ladder (chunks in ladder order) ----
    chunks = []          # dicts: level, field, params, data, decoded
    t0 = time.time()
    for li in range(N_LEVELS):
        for f in fields:
            q = q_flat[f]
            if li == 0:
                layer = np.round(q / STEPS[0]).astype(np.int32)
            else:
                layer = (np.round(q / STEPS[li]).astype(np.int32)
                         - (STEPS[li - 1] // STEPS[li])
                         * np.round(q / STEPS[li - 1]).astype(np.int32))
            data, means, stds, params, back, exact = code_chunk(layer, g_flat[f])
            assert exact, f"roundtrip mismatch level {li} field {f}"
            chunks.append({"level": li, "field": f, "params": params,
                           "data": data, "decoded": back, "bytes": len(data),
                           "means": means, "stds": stds})
            print(f"[RB] coded L{li} {f}: {len(data)/1e6:.2f} MB exact={exact}",
                  flush=True)
    coding_s = time.time() - t0

    # ---- write the real transmit-able file ----
    header = {
        "format": "c25_layered_v1",
        "groups": GROUPS,
        "steps": [int(s) for s in STEPS],
        "fields": list(fields),
        "chunks": [{"level": c["level"], "field": c["field"]} for c in chunks],
        "n_alive": int(n_alive),
    }
    header_bytes = json.dumps(header).encode("utf-8")
    bin_path = Path(args.bin_out)
    with open(bin_path, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(4, "little"))
        fh.write(header_bytes)
        for c in chunks:
            locs, bs = c["params"]
            pb = locs.astype(np.float32).tobytes() + bs.astype(np.float32).tobytes()
            fh.write(len(pb).to_bytes(4, "little"))
            fh.write(pb)
            fh.write(len(c["data"]).to_bytes(4, "little"))
            fh.write(c["data"])
    bin_size = bin_path.stat().st_size
    print(f"[RB] wrote {bin_path} ({bin_size/1e6:.2f} MB)", flush=True)

    # ---- fixed payload carried once (from the real hac_meta accounting) ----
    fixed_bytes = (
        int(meta["bit_hash"]) + int(meta["bit_mlp"]) + int(meta["bit_bounds"])
        + int(meta["bit_header"]) + int(meta["bit_masks"])
    ) / 8.0
    geom_bytes = int(meta["bit_anchor"]) / 8.0
    header_total = 4 + len(header_bytes)
    chunk_params_total = sum(4 + 8 * GROUPS + 4 for _ in chunks)
    chunk_data_total = sum(c["bytes"] for c in chunks)

    # ---- dataset (after model, per trainer convention) ----
    dataset = ColmapDataset(
        data_dir=args.data_dir, data_factor=2, test_every=8,
        white_background=False, preload_images=False, device=str(dev),
    )
    val = dataset.val_cameras
    pick = np.unique(np.linspace(0, len(val) - 1, N_VIEWS).astype(int))
    cams = [val[i] for i in pick]
    background = dataset.background

    # cumulative coded bytes after finishing level li
    cum_by_level, cum = {}, 0
    for c in chunks:
        cum += c["bytes"]
        cum_by_level[c["level"]] = cum

    # ---- render prefixes; full precision FIRST as the sanity anchor ----
    results = []
    for li in [N_LEVELS - 1] + list(range(N_LEVELS - 1)):
        deq = {}
        for f in fields:
            chunks_upto = [c for c in chunks if c["level"] <= li and c["field"] == f]
            acc = chunks_upto[0]["decoded"].astype(np.float64)
            for prev, cur in zip(chunks_upto[:-1], chunks_upto[1:]):
                acc = (STEPS[prev["level"]] // STEPS[cur["level"]]) * acc \
                    + cur["decoded"]
            xhat = (STEPS[li] * acc.reshape(out_shape[f])
                    * Q_t[f].cpu().numpy()).astype(np.float32)
            deq[f] = torch.from_numpy(xhat).to(dev)
        view.anchor_feat = torch.nn.Parameter(deq["feat"].contiguous(),
                                              requires_grad=False)
        view.scaling = torch.nn.Parameter(deq["scaling"].contiguous(),
                                          requires_grad=False)
        view.offset = torch.nn.Parameter(deq["offset"].contiguous(),
                                         requires_grad=False)
        t0 = time.time()
        psnrs = []
        with torch.no_grad():
            for cam in cams:
                out = model.render(cam, background, is_training=False,
                                   appearance_id=0, step=30000)
                pred = out.image[0].permute(2, 0, 1).clamp(0.0, 1.0)
                gt = dataset.get_image(cam).clamp(0.0, 1.0)
                mse = torch.mean((pred - gt) ** 2).item()
                psnrs.append(float("inf") if mse <= 0 else -10.0 * np.log10(mse))
                del out
        real_bytes = (4 + len(header_bytes) + fixed_bytes + geom_bytes
                      + chunk_params_total + cum_by_level[li])
        entry = {
            "level": li,
            "step_factor": int(STEPS[li]),
            "coded_bytes": int(cum_by_level[li]),
            "real_bytes": round(real_bytes, 1),
            "psnr_mean": round(float(np.mean(psnrs)), 4),
            "psnr_per_view": [round(p, 4) for p in psnrs],
            "render_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(f"[RB] prefix@L{li} (Q x{entry['step_factor']}): REAL "
              f"{real_bytes/1e6:6.2f} MB  PSNR {entry['psnr_mean']:.3f}", flush=True)
        if li == N_LEVELS - 1:
            print(f"[RB] SANITY ANCHOR: {entry['psnr_mean']:.3f} dB (need >= 25)")
            assert entry["psnr_mean"] >= 25.0, "sanity anchor failed — INVALID"
        del deq
        torch.cuda.empty_cache()

    payload = {
        "run": str(run_dir), "n_alive": int(n_alive), "groups": GROUPS,
        "steps": [int(s) for s in STEPS],
        "fixed_bytes": fixed_bytes, "geom_bytes": geom_bytes,
        "header_bytes": header_total,
        "chunk_params_total": chunk_params_total,
        "coding_s": round(coding_s, 1),
        "bin_file": str(bin_path), "bin_size": bin_size,
        "results": results,
    }
    if Path(args.s2).exists():
        s2 = json.loads(Path(args.s2).read_text())
        payload["s2_selection_reference"] = [
            {k2: r[k2] for k2 in ("ordering", "prefix", "bytes_est", "psnr_mean")}
            for r in s2["results"]
        ]
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"[RB] wrote {out_path}")


if __name__ == "__main__":
    main()
