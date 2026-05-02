# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for hierarchical transformers.

This package forwards to the flat layout. New code should import directly from
``sktime.transformations`` (e.g.
``from sktime.transformations.aggregate import Aggregator``).

Subpackage ``reconcile`` is still housed underneath as a real subpackage
(also a deprecation shim) until that namespace is fully retired.
"""

import importlib
import sys
import warnings

from sktime.transformations.aggregate import Aggregator  # noqa: F401
from sktime.transformations.squeeze_hierarchy import SqueezeHierarchy  # noqa: F401

__all__ = [
    "Aggregator",
    "SqueezeHierarchy",
]

# Make legacy submodule paths resolve to the new flat modules.
# Both were public submodules.
_legacy_to_new = {
    "aggregate": "sktime.transformations.aggregate",
    "squeeze_hierarchy": "sktime.transformations.squeeze_hierarchy",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.hierarchical is deprecated and will be removed in "
    "a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.aggregate.Aggregator).",
    DeprecationWarning,
    stacklevel=2,
)
