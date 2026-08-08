"""Vendored HAC++ core (official YihangChen-ee/HAC-plus, unmodified).

The official code uses absolute imports such as ``from scene.gaussian_model
import GaussianModel``, so importing this package inserts its own directory at
the front of ``sys.path``. The CUDA extensions ``_gridencoder``, ``arithmetic``
and ``simple_knn`` must be installed in the active environment (the 5090
``HAC_5090_a100`` conda env already has them).

Only the files required by the model/adapter are vendored; the diff-gaussian
rasterizer is intentionally not included because gsplat2hac renders with gsplat.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
