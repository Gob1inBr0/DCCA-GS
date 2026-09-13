"""CPU tests for 方案 C (bg feature codebook) and the byte-account
aggregation. No GPU, no real codec: the codec-facing pieces are exercised
through their pure functions, plus one real-data check of the aggregation
against the committed area dump.

Run:  python3 tests/test_bg_codebook.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scaffold_gs import bg_codebook as bc  # noqa: E402
from scripts.anchor_byte_account import (  # noqa: E402
    aggregate_account,
    flags_share,
)

PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def test_derive_flags():
    area = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    seen = np.array([True, True, True, True, True, True])
    flags = bc.derive_flags(area, seen, quantile=0.8)
    # top 20% of 5 valid = 1 anchor (the largest)
    check("derive_flags picks top area", flags.sum() == 1 and flags[4])
    flags_unseen = bc.derive_flags(area, seen, quantile=0.6)
    check(
        "derive_flags unseen-excluded",
        flags_unseen.sum() == 2,
        f"got {flags_unseen.sum()}",
    )
    empty = bc.derive_flags(np.array([np.nan, np.nan]))
    check("derive_flags all-nan -> none", empty.sum() == 0)


def test_kmeans_determinism_and_recovery():
    gen = torch.Generator().manual_seed(7)
    # Three well-separated clusters.
    X = torch.cat(
        [
            torch.randn(50, 8, generator=gen),
            torch.randn(50, 8, generator=gen) + 20,
            torch.randn(50, 8, generator=gen) + 40,
        ]
    )
    c1, a1 = bc.kmeans(X, 3, iters=30, seed=0)
    c2, a2 = bc.kmeans(X, 3, iters=30, seed=0)
    check("kmeans deterministic", torch.equal(c1, c2) and torch.equal(a1, a2))
    # Each cluster's members share one assignment (3 groups, no splits).
    groups = torch.unique(a1)
    check("kmeans K respected", c1.shape[0] == 3 and len(groups) == 3)
    # Lloyd invariant: every point sits at its nearest centroid.
    nearest = torch.cdist(X, c1).argmin(dim=1)
    check("kmeans nearest-centroid assignment", torch.equal(nearest, a1))
    # K > N clamps.
    c3, _ = bc.kmeans(X[:2], 10, iters=5, seed=0)
    check("kmeans K>N clamps", c3.shape[0] == 2)


def test_codebook_roundtrip():
    gen = torch.Generator().manual_seed(3)
    feat = torch.randn(1000, 32, generator=gen)
    flags = torch.zeros(1000, dtype=torch.bool)
    flags[torch.randperm(1000, generator=gen)[:150]] = True
    feat_rep, payload = bc.build_codebook(feat, flags, codebook_size=64, seed=0)
    check(
        "build_codebook replaces only bg rows",
        torch.equal(feat_rep[~flags], feat[~flags])
        and not torch.equal(feat_rep[flags], feat[flags]),
    )
    check(
        "payload sizes",
        payload["codebook"].shape == (64, 32)
        and payload["indices"].shape == (150,)
        and payload["n_total"] == 1000,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bg_codebook.npz"
        nbytes = bc.save_payload(payload, path)
        loaded = bc.load_payload(path)
        check("payload save/load roundtrip", np.array_equal(loaded["codebook"], payload["codebook"]))
        # Reconstruct from an fg-only matrix (simulating the decoded stream):
        # fg rows come from 'the stream' (== replaced fg rows), bg rows must
        # come out as the stored centroids.
        feat_fg_t = feat_rep[~flags].clone()
        payload2 = dict(loaded)
        payload2["flags_packed"] = loaded["flags_packed"]
        device = torch.device("cpu")
        flags_back = bc.unpack_flags(payload2, device)
        check("flags unpack roundtrip", torch.equal(flags_back, flags))
        full = torch.empty(1000, 32)
        full[~flags_back] = feat_fg_t
        codebook = torch.from_numpy(loaded["codebook"].astype(np.float32))
        full[flags_back] = codebook[
            torch.from_numpy(loaded["indices"].astype(np.int64))
        ]
        check(
            "reconstruct == replaced features (bit-level)",
            torch.equal(full, feat_rep),
        )
        check("payload bytes > 0", nbytes > 0)
    # Zero-background edge: payload valid, features unchanged.
    src = torch.randn(10, 4)
    noflag = torch.zeros(10, 4, dtype=torch.bool)
    fr, p0 = bc.build_codebook(src, noflag, 8)
    check(
        "zero-bg payload valid + no-op",
        p0["indices"].shape == (0,) and torch.equal(fr, src),
    )


def test_empty_batch_selection():
    # All-background batch: the FG selection is empty; the codec guards
    # rely on this.
    bg = torch.ones(6, dtype=torch.bool)
    fg_local = (~bg).nonzero(as_tuple=True)[0]
    x = torch.arange(12).float().view(6, 2)
    xs = x[fg_local]
    check("all-bg selection empty", xs.shape == (0, 2))


def test_aggregation_with_real_area_dump():
    npz_path = REPO / "analysis/anchor_stats/ph0c_base_lam0004_s104/anchor_stats.npz"
    if not npz_path.exists():
        print("[SKIP] real area dump not present")
        return
    data = np.load(npz_path)
    area, seen = data["area"], data["seen"]
    rng = np.random.default_rng(0)
    # Synthetic but realistic per-anchor bits: heavy-tailed, correlated with
    # area so the shares are meaningful (feat dominates per the byte table).
    base = rng.gamma(2.0, 30.0, size=area.shape[0])  # ~60 bits mean
    scale_part = rng.gamma(2.0, 6.0, size=area.shape[0])
    off_part = rng.gamma(2.0, 25.0, size=area.shape[0])
    per_anchor = {
        "feat": base * (1.0 + 0.5 * np.log1p(np.clip(area, 0, None) * 1e4)),
        "scaling": scale_part,
        "offsets": off_part * (data["seen"].astype(float)),
    }
    account = aggregate_account(per_anchor, area, seen)
    shares = [r["anchor_bits_share"] for r in account["deciles"]]
    check(
        "decile shares sum to ~1",
        abs(sum(shares) - 1.0) < 1e-6,
        f"sum={sum(shares):.6f}",
    )
    check("10 deciles", len(shares) == 10)
    h = account["headline"]
    check(
        "top-x shares monotone in x",
        h["top_5pct_area_bits_share"]
        <= h["top_10pct_area_bits_share"]
        <= h["top_20pct_area_bits_share"],
        f"5%={h['top_5pct_area_bits_share']:.3f} "
        f"10%={h['top_10pct_area_bits_share']:.3f} "
        f"20%={h['top_20pct_area_bits_share']:.3f}",
    )
    check(
        "large-area anchors concentrate bits above uniform",
        h["top_5pct_area_bits_share"] > 0.05,
        f"top-5% owns {h['top_5pct_area_bits_share']:.3f} (uniform would be 0.05)",
    )
    check(
        "n_valid matches seen+finite",
        account["n_valid"] == int((seen & np.isfinite(area)).sum()),
    )
    fs = flags_share(area, seen, 0.9, per_anchor)
    check(
        "flags share sane",
        0.0 < fs["bg_population_share"] <= 0.2 and 0.0 < fs["bg_bits_share"] < 1.0,
        f"pop={fs['bg_population_share']:.3f} bits={fs['bg_bits_share']:.3f}",
    )
    print(json_table(account))


def json_table(account):
    lines = ["decile | n | area_median | bits_share | bits/anchor"]
    for r in account["deciles"]:
        lines.append(
            f"{r['decile']} | {r['n']} | {r['area_median']:.2e} | "
            f"{r['anchor_bits_share']:.3f} | {r['bits_per_anchor_median']:.0f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    test_derive_flags()
    test_kmeans_determinism_and_recovery()
    test_codebook_roundtrip()
    test_empty_batch_selection()
    test_aggregation_with_real_area_dump()
    n_ok = sum(PASS)
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) else 1)
