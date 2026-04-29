# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for Kalman Filter Transformers.

This package forwards to the flat layout. New code should import from
``sktime.transformations`` directly (e.g.
``from sktime.transformations.kalman_filter import KalmanFilterTransformerFP``).
"""

import importlib
import sys
import warnings

from sktime.transformations.kalman_filter import (  # noqa: F401
    KalmanFilterTransformerFP,
    KalmanFilterTransformerPK,
)
from sktime.transformations.kalman_filter_base import BaseKalmanFilter  # noqa: F401
from sktime.transformations.simdkalman import (  # noqa: F401
    KalmanFilterTransformerSIMD,
)

__all__ = [
    "BaseKalmanFilter",
    "KalmanFilterTransformerFP",
    "KalmanFilterTransformerPK",
    "KalmanFilterTransformerSIMD",
]

# Make legacy submodule paths resolve to the new flat modules.
#   - ``_base``, ``_kalman_filter``, ``_simdkalman`` were all private files;
#     aliased only for pickle compat (objects pickled with previous releases
#     store the old private ``__module__``).
_legacy_to_new = {
    "_base": "sktime.transformations.kalman_filter_base",
    "_kalman_filter": "sktime.transformations.kalman_filter",
    "_simdkalman": "sktime.transformations.simdkalman",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.series.kalman_filter is deprecated and will be "
    "removed in a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.kalman_filter.KalmanFilterTransformerFP).",
    DeprecationWarning,
    stacklevel=2,
)
