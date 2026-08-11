"""Vendored HAC++ core (official YihangChen-ee/HAC-plus, adapted).

All internal modules use package-relative imports (``hacplus.scene`` /
``hacplus.utils``), so importing this package no longer mutates ``sys.path``.
The CUDA extensions ``_gridencoder``, ``arithmetic`` and ``simple_knn`` must be
installed in the active environment (the 5090 ``HAC_5090_a100`` conda env
already has them).

Only the files required by the model/adapter are vendored; the diff-gaussian
rasterizer is intentionally not included because gsplat2hac renders with gsplat.
"""
