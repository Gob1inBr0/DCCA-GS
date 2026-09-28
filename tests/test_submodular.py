"""Unit tests for the submodular greedy selector (v1/v2).

Run: python -m pytest tests/test_submodular.py -q  (CPU only, <1s)

Covers the three properties that matter for the journal-round verdicts:
- kappa=1 picks the exact max-marginal-gain anchor (first greedy step);
- the budget is respected and the selection beats a base-score topk of the
  same size on a crafted instance where coverage and base scores disagree;
- v2 sensitivity weighting actually engages and points in the right
  direction (regression for the 2026-09-19 wiring fix: cover_sens used to
  silently run pure coverage), with sens=0 reproducing v1 exactly.
"""

import torch

from scaffold_gs.submodular import submodular_greedy_select


def _edges(pairs):
    """One view; pairs = list of (anchor, block) repeated per pixel mass."""
    anchors = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    blocks = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return [(blocks, anchors)]


def _objective(keep, pairs):
    """True coverage objective: sum over blocks of max pair mass."""
    best = {}
    for a, b in pairs:
        best[b] = max(best.get(b, 0), 1 if (a, b) not in best else 1)
    # recompute honestly from multiplicities
    mass = {}
    for a, b in pairs:
        mass[(a, b)] = mass.get((a, b), 0) + 1
    covered = {}
    for (a, b), m in mass.items():
        if a in keep:
            covered[b] = max(covered.get(b, 0), m)
    return sum(covered.values())


def test_kappa1_picks_max_gain():
    # anchor0 -> block0 (mass 3), anchor1 -> block1 (mass 5)
    pairs = [(0, 0)] * 3 + [(1, 1)] * 5
    keep, _ = submodular_greedy_select(_edges(pairs), 1, None)
    assert keep.tolist() == [1]


def test_budget_respected_and_beats_base_topk():
    # base scores favor anchors 0/1, but they cover the SAME block 0
    # (mass 2 each); anchors 2/3 cover fresh blocks with mass 3 — greedy
    # must prefer the fresh coverage, base-score topk wastes the budget.
    pairs = [(0, 0)] * 2 + [(1, 0)] * 2 + [(2, 1)] * 3 + [(3, 2)] * 3
    base = torch.tensor([9.0, 8.0, 1.0, 0.5])
    keep, _ = submodular_greedy_select(_edges(pairs), 2, None, base_scores=base)
    assert keep.numel() <= 2
    obj_greedy = _objective(set(keep.tolist()), pairs)
    obj_topk = _objective({0, 1}, pairs)  # base-score top-2 choice
    assert obj_greedy > obj_topk
    assert set(keep.tolist()) == {2, 3}


def test_v2_weighting_points_toward_sensitive_anchor():
    # anchor0: mass 6 on block0; anchor1: mass 5 on block1, sensitivity 2.
    # v1 picks anchor0 (6 > 5); v2 picks anchor1 (5*(1+2)=15 > 6*1).
    pairs = [(0, 0)] * 6 + [(1, 1)] * 5
    sens = torch.tensor([0.0, 2.0])
    keep_v1, _ = submodular_greedy_select(_edges(pairs), 1, None)
    keep_v2, _ = submodular_greedy_select(_edges(pairs), 1, sens)
    assert keep_v1.tolist() == [0]
    assert keep_v2.tolist() == [1]


def test_v2_zero_sens_equals_v1():
    # mass * (1 + 0) must reproduce the v1 objective exactly.
    pairs = [(0, 0)] * 3 + [(1, 1)] * 4 + [(1, 2)] * 2 + [(2, 1)] * 1
    rng = torch.Generator().manual_seed(0)
    base = torch.rand(3, generator=rng)
    sens0 = torch.zeros(3)
    k1, _ = submodular_greedy_select(_edges(pairs), 2, None, base_scores=base)
    k2, _ = submodular_greedy_select(_edges(pairs), 2, sens0, base_scores=base)
    assert set(k1.tolist()) == set(k2.tolist())
