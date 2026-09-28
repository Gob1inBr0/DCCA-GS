"""Decisions-curve diagnostic (mode-4 prerequisite, runbook §6.3).

Parses PruneLog lines from finished run logs and reports, per run and per
phase (growth = step < reinit_iter, post = step >= reinit_iter), the mean
fraction of the population actually being decided each projection:
    bite_ratio = decisions / N   (decisions = N - kappa)

Validates the hypothesis "a fixed post budget (r_post=0.85) barely bites at
loose lambda" before any lambda-adaptive budget work (mode 4).

Usage:
  python scripts/decisions_curve.py LOG [LOG ...]
"""

from __future__ import annotations

import json
import re
import sys

LINE = re.compile(
    r"cycle=(\d+) step=(\d+) N=(\d+) kappa=(\d+) decisions=(\d+)"
)


def parse(path: str) -> dict:
    rows = []
    for line in open(path, errors="ignore"):
        m = LINE.search(line)
        if m:
            cyc, step, n, kappa, dec = map(int, m.groups())
            rows.append((cyc, step, n, kappa, dec))
    if not rows:
        return {}
    reinit = 15000
    phases = {"growth": [], "post": []}
    for cyc, step, n, kappa, dec in rows:
        ph = "post" if step >= reinit else "growth"
        if n > 0:
            phases[ph].append(dec / n)
    out = {"path": path, "cycles": len(rows)}
    for ph, vals in phases.items():
        if vals:
            out[ph] = {
                "cycles": len(vals),
                "bite_ratio_mean": round(sum(vals) / len(vals), 5),
                "bite_ratio_min": round(min(vals), 5),
                "bite_ratio_max": round(max(vals), 5),
            }
    return out


def main() -> None:
    table = {}
    for path in sys.argv[1:]:
        r = parse(path)
        if r:
            table[path.rsplit("/", 1)[-1].replace(".log", "")] = r
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
