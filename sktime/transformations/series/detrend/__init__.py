# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for detrend transformers.

This module forwards to the flat layout. New code should import directly from
``sktime.transformations`` (e.g. ``from sktime.transformations.mstl import MSTL``).
"""

import importlib
import sys
import warnings

from sktime.transformations.deseasonalize import (  # noqa: F401
    ConditionalDeseasonalizer,
    Deseasonalizer,
    STLTransformer,
)
from sktime.transformations.detrend import Detrender  # noqa: F401
from sktime.transformations.mstl import MSTL  # noqa: F401

__all__ = [
    "ConditionalDeseasonalizer",
    "Deseasonalizer",
    "Detrender",
    "MSTL",
    "STLTransformer",
]

# Make legacy submodule paths resolve to the new flat modules.
#   - ``mstl`` was a public submodule, so this is API-level back-compat.
#   - ``_detrend`` / ``_deseasonalize`` were private; we only alias them so
#     that objects pickled with previous releases (whose ``__module__`` points
#     at the old private path) continue to unpickle successfully.
_legacy_to_new = {
    "mstl": "sktime.transformations.mstl",
    "_detrend": "sktime.transformations.detrend",
    "_deseasonalize": "sktime.transformations.deseasonalize",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.series.detrend is deprecated and will be removed "
    "in a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.mstl.MSTL).",
    DeprecationWarning,
    stacklevel=2,
)
