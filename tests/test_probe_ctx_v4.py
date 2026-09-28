"""Tests for probe v4's pure decision-stream helpers.

Run: python -m pytest tests/test_probe_ctx_v4.py -q  (CPU only, <1s)

Covers the three pieces whose correctness the probe verdict rests on:
- the stage-C continuation expansion (k = 1..m, bit = k < m) — a wrong
  expansion silently shifts every continuation bitstream;
- the isomorphic static flag baseline (train-only cell rates with the
  production fallback chain cell -> group -> global);
- the isomorphic geometric-tail baseline (beta from train magnitudes).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_ctx_v4 import (_cont_decisions, static_cont_baseline,
                          static_flag_baseline)

N_BUCKETS = 3


def test_cont_decisions_expansion():
    # d = [0, +1, -2, +3, 0] -> survivors mags [1, 2, 3]
    d = np.array([0, 1, -2, 3, 0], dtype=np.int64)
    g = np.array([9, 9, 15, 21, 9], dtype=np.int64)  # composite ids
    cell = g * N_BUCKETS + np.array([0, 1, 2, 0, 1], dtype=np.int64)
    bit, sym, cel, gg, k, counts = _cont_decisions(d, cell, g)
    assert counts.tolist() == [1, 2, 3]
    assert bit.tolist() == [0, 1, 0, 1, 1, 0]
    assert sym.tolist() == [1, 2, 2, 3, 3, 3]
    assert k.tolist() == [1, 1, 2, 1, 2, 3]
    assert cel.tolist() == (cell[[1, 2, 2, 3, 3, 3]]).tolist()
    assert gg.tolist() == [9, 15, 15, 21, 21, 21]
    assert bit.sum() == 3  # sum(mag - 1)


def test_cont_decisions_all_zero():
    d = np.zeros(5, dtype=np.int64)
    g = np.zeros(5, dtype=np.int64)
    cell = np.zeros(5, dtype=np.int64)
    bit, sym, cel, gg, k, counts = _cont_decisions(d, cell, g)
    assert bit.size == 0 and counts.sum() == 0


def test_static_flag_baseline_fallback_chain():
    rng = np.random.default_rng(0)
    n = 4000
    n_groups, n_cells = 4, 4 * N_BUCKETS
    # anchors 0..399; decisions tied to anchors via sym index
    sym = np.arange(n) // 4  # 4 columns per anchor
    cell = (sym % n_groups) * N_BUCKETS + (sym % N_BUCKETS)
    g = cell // N_BUCKETS
    # cell 0: rate 0.9; cell 1: rate 0.1; others arbitrary
    p_true = np.where(cell == 0, 0.9, np.where(cell == 1, 0.1, 0.5))
    bit = (rng.random(n) < p_true).astype(np.int64)
    # train = everything EXCEPT the anchors holding cell-1 decisions is
    # hard to arrange cleanly; instead make cell 2 absent from train by
    # val-selecting all its anchors
    val_anchor = np.zeros(sym.max() + 1, dtype=bool)
    val_anchor[np.unique(sym[cell == 2])] = True
    val_sel = val_anchor[sym]
    tr_sel = ~val_sel
    p = static_flag_baseline(bit, cell, g, tr_sel, n_cells, n_groups)
    # train-side cells recover their empirical rates (sanity, not exact)
    assert p[cell == 0].mean() > 0.7
    assert p[cell == 1].mean() < 0.3
    # cell 2 unseen in train -> falls back (group or global), stays valid
    assert ((p[cell == 2] > 0) & (p[cell == 2] < 1)).all()
    # fallback rate equals the global train rate for fully-unseen groups
    global_rate = bit[tr_sel & (cell == 3)].mean()
    assert np.isclose(p[cell == 2].mean(), global_rate, atol=0.2)


def test_static_cont_baseline():
    rng = np.random.default_rng(1)
    n_cells = 2
    cell_s = np.array([0] * 500 + [1] * 500)
    mag_s = np.where(cell_s == 0, rng.integers(1, 3, 1000),
                     rng.integers(4, 9, 1000)).astype(np.float64)
    surv_train = np.ones(1000, dtype=bool)
    beta = static_cont_baseline(cell_s, mag_s, surv_train, n_cells)
    m0 = mag_s[cell_s == 0].mean()
    m1 = mag_s[cell_s == 1].mean()
    assert np.isclose(beta[0], min(max(1 - 1 / max(m0, 1), 0.05), 0.98))
    assert np.isclose(beta[1], min(max(1 - 1 / max(m1, 1), 0.05), 0.98))
    assert beta[1] > beta[0]  # heavier tails -> larger continuation rate

    # unseen cell falls back to the global fit
    surv_train2 = surv_train.copy()
    surv_train2[cell_s == 1] = False
    beta2 = static_cont_baseline(cell_s, mag_s, surv_train2, n_cells)
    gm = mag_s[cell_s == 0].mean()  # only cell 0 contributes to train
    assert np.isclose(beta2[1], min(max(1 - 1 / max(gm, 1), 0.05), 0.98))
