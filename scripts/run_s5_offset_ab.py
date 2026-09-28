#!/usr/bin/env python3
"""A/B driver for the offset refine-flag MLP (S5) against the production
static conditional table.

Runs c25_real_bitstream.py twice on the SAME run directory with identical
arguments — the only difference is --s5-offset-mlp — then compares:

  - offset enhancement-layer bytes (chunks with field=offset, level>=1):
    baseline B vs MLP arm S; the trained head weights ride in the S5 file,
    so the honest number is net = (B - S - weights) / B;
  - PSNR per prefix: the flag path is a lossless re-encode (probabilities
    change code length, not decoded values), so arms must match to float
    noise. A PSNR difference means the A/B is broken, not "a tradeoff".

Gate (pre-registered for the S5 production integration): net >= 10% on the
offset enhancement layers and identical PSNR on both scenes.

This is the isomorphic-baseline measurement the earlier learned-probability
attempt lacked: the baseline here IS the production encoder, not a reduced
probe table (2026-09-26 lesson, registered in the negative-result ledger).

Usage (training machine, c25 python + PYTHONPATH=/home/project2/c25_pylib2):
  python scripts/run_s5_offset_ab.py \
      --run /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42 \
      --data-dir /mnt/newproject2/data/1-78 \
      --stats /mnt/newproject2/dcca_runs/ph0c_base_r085_lam0005_s42/anchor_stats.npz \
      --field-aware --out-dir analysis/s5_offset_ab
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_encoder(args, out_json, out_bin, s5):
    cmd = [args.python, str(REPO / "scripts" / "c25_real_bitstream.py"),
           "--run", args.run, "--data-dir", args.data_dir,
           "--device", args.device,
           "--data-factor", str(args.data_factor),
           "--out", str(out_json), "--bin", str(out_bin)]
    if args.stats:
        cmd += ["--stats", args.stats]
    if args.max_width:
        cmd += ["--max-width", str(args.max_width)]
    if args.groups != 32:
        cmd += ["--groups", str(args.groups)]
    if args.field_aware:
        cmd += ["--field-aware"]
    if s5:
        cmd += ["--s5-offset-mlp"]
    print(f"[AB] {'S5  ' if s5 else 'base'}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"[AB] encoder failed ({'s5' if s5 else 'base'} arm), "
                         f"rc={proc.returncode}")
    return time.time() - t0


def offset_enh_bytes(payload):
    return sum(c["bytes"] for c in payload["chunks_summary"]
               if c["field"] == "offset" and c["level"] >= 1)


def per_level_bytes(payload):
    out = {}
    for c in payload["chunks_summary"]:
        if c["field"] == "offset" and c["level"] >= 1:
            out[c["level"]] = c["bytes"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--field-aware", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default="analysis/s5_offset_ab")
    ap.add_argument("--gate-net-pct", type=float, default=10.0,
                    help="pre-registered adoption gate on net saving")
    ap.add_argument("--tag", default=None,
                    help="output tag; default <run name>")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.run).name
    base_json = out_dir / f"{tag}_baseline.json"
    base_bin = out_dir / f"{tag}_baseline.bin"
    s5_json = out_dir / f"{tag}_s5.json"
    s5_bin = out_dir / f"{tag}_s5.bin"

    t_base = run_encoder(args, base_json, base_bin, s5=False)
    t_s5 = run_encoder(args, s5_json, s5_bin, s5=True)
    P_base = json.loads(base_json.read_text())
    P_s5 = json.loads(s5_json.read_text())

    B = offset_enh_bytes(P_base)
    S = offset_enh_bytes(P_s5)
    weights = int(P_s5["s5_weight_bytes"]) + (4 if P_s5["s5_apply"] else 0)
    raw_pct = 100 * (1 - S / max(B, 1))
    net_pct = 100 * (1 - (S + weights) / max(B, 1))

    psnr_checks = []
    for rb, rs in zip(P_base["results"], P_s5["results"]):
        assert rb["prefix"] == rs["prefix"]
        psnr_checks.append({"prefix": rb["prefix"],
                            "base": rb["psnr_mean"], "s5": rs["psnr_mean"],
                            "abs_diff": round(abs(rb["psnr_mean"]
                                                  - rs["psnr_mean"]), 4)})
    psnr_ok = all(c["abs_diff"] < 0.01 for c in psnr_checks)

    per_level = {}
    lb, ls = per_level_bytes(P_base), per_level_bytes(P_s5)
    for lv in sorted(set(lb) | set(ls)):
        per_level[lv] = {"base": lb.get(lv), "s5": ls.get(lv),
                         "saved": (lb.get(lv, 0) - ls.get(lv, 0))}

    net_ok = net_pct >= args.gate_net_pct
    verdict = "PASS" if (net_ok and psnr_ok) else "FAIL"

    summary = {
        "run": args.run, "tag": tag, "field_aware": bool(args.field_aware),
        "groups": args.groups,
        "commands": {"base": f"c25_real_bitstream.py -> {base_json}",
                     "s5": f"c25_real_bitstream.py --s5-offset-mlp -> {s5_json}"},
        "coding_s": {"base": round(t_base, 1), "s5": round(t_s5, 1)},
        "offset_enh_bytes": {"base": int(B), "s5": int(S)},
        "s5_weight_bytes_total": weights,
        "s5_level_params": P_s5.get("s5_level_params", {}),
        "raw_saving_pct": round(raw_pct, 2),
        "net_saving_pct": round(net_pct, 2),
        "gate_net_pct": args.gate_net_pct,
        "per_level": per_level,
        "psnr_checks": psnr_checks,
        "psnr_identical": psnr_ok,
        "net_gate_pass": net_ok,
        "verdict": verdict,
        "full_precision": {"base": P_base["results"][0],
                           "s5": P_s5["results"][0]},
    }
    out = out_dir / f"{tag}_ab_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[AB] offset enhancement layers: base {B/1e6:.3f} MB -> "
          f"s5 {S/1e6:.3f} MB (+{weights/1024:.0f} KB weights) "
          f"= raw {raw_pct:.1f}% / net {net_pct:.1f}%", flush=True)
    for lv in sorted(per_level):
        pl = per_level[lv]
        print(f"[AB]   offset L{lv}: {pl['base']} -> {pl['s5']} "
              f"(saved {pl['saved']})", flush=True)
    print(f"[AB] psnr identical: {psnr_ok} (max diff "
          f"{max((c['abs_diff'] for c in psnr_checks), default=0):.4f})",
          flush=True)
    print(f"[AB] VERDICT: {verdict} (gate: net >= {args.gate_net_pct}%, "
          f"PSNR identical) -> {out}", flush=True)


if __name__ == "__main__":
    main()
