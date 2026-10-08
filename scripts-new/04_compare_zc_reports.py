#!/usr/bin/env python3
"""Compare legacy and UAV-DCCA-ZC layered bitstream JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mb(x: float) -> float:
    return float(x) / 1e6


def _pct(delta: float, base: float) -> float:
    return 100.0 * float(delta) / max(float(base), 1.0)


def _prefix_map(report):
    out = {}
    for row in report.get("results", []):
        out[int(row["prefix"])] = row
    return out


def _field_map(report, key):
    return {str(k): int(v) for k, v in report.get(key, {}).items()}


def compare(base, test, psnr_eps):
    lines = []
    lines.append("== Stream Summary ==")
    lines.append(f"base zc_context: {base.get('zc_context', 'unknown')}")
    lines.append(f"test zc_context: {test.get('zc_context', 'unknown')}")
    lines.append(f"base cells: {base.get('zc_num_cells', 0)}")
    lines.append(f"test cells: {test.get('zc_num_cells', 0)}")
    lines.append("")

    base_p = _prefix_map(base)
    test_p = _prefix_map(test)
    lines.append("== Prefix Bytes / PSNR ==")
    for p in sorted(set(base_p) & set(test_p)):
        b = base_p[p]
        t = test_p[p]
        db = int(t["real_bytes"]) - int(b["real_bytes"])
        dpsnr = float(t["psnr_mean"]) - float(b["psnr_mean"])
        flag = ""
        if abs(dpsnr) > psnr_eps:
            flag = "  PSNR_CHANGED"
        lines.append(
            "P{}: base={:.3f}MB test={:.3f}MB delta={:.3f}MB "
            "({:+.2f}%) dPSNR={:+.4f}{}".format(
                p,
                _mb(b["real_bytes"]),
                _mb(t["real_bytes"]),
                _mb(db),
                _pct(db, b["real_bytes"]),
                dpsnr,
                flag,
            )
        )
    lines.append("")

    for key in ("bytes_by_field", "bytes_by_field_level"):
        bmap = _field_map(base, key)
        tmap = _field_map(test, key)
        lines.append(f"== {key} ==")
        for name in sorted(set(bmap) | set(tmap)):
            b = bmap.get(name, 0)
            t = tmap.get(name, 0)
            db = t - b
            lines.append(
                "{}: base={:.3f}MB test={:.3f}MB delta={:.3f}MB "
                "({:+.2f}%)".format(
                    name, _mb(b), _mb(t), _mb(db), _pct(db, b)
                )
            )
        lines.append("")

    b_cond = int(base.get("conditional_saved_bytes", 0))
    t_cond = int(test.get("conditional_saved_bytes", 0))
    lines.append("== Conditioner Savings ==")
    lines.append(
        "conditional_saved_bytes: base={:.3f}MB test={:.3f}MB delta={:.3f}MB".format(
            _mb(b_cond), _mb(t_cond), _mb(t_cond - b_cond)
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="legacy/off JSON report")
    ap.add_argument("--test", required=True, help="ZCausal JSON report")
    ap.add_argument("--out", default=None, help="optional text report path")
    ap.add_argument("--psnr-eps", type=float, default=1e-4)
    args = ap.parse_args()

    text = compare(_load(args.base), _load(args.test), args.psnr_eps)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
