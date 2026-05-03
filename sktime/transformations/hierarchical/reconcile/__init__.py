# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for hierarchical reconciliation transformers.

This package forwards to the flat layout. New code should import directly from
``sktime.transformations`` (e.g.
``from sktime.transformations.reconcile import Reconciler``).
"""

import importlib
import sys
import warnings

from sktime.transformations.bottom_up_reconciler import BottomUpReconciler  # noqa: F401
from sktime.transformations.middle_out_reconciler import (
    MiddleOutReconciler,  # noqa: F401
)
from sktime.transformations.optimal_reconciler import (  # noqa: F401
    NonNegativeOptimalReconciler,
    OptimalReconciler,
)
from sktime.transformations.reconcile import Reconciler  # noqa: F401
from sktime.transformations.topdown_reconciler import TopdownReconciler  # noqa: F401

__all__ = [
    "MiddleOutReconciler",
    "BottomUpReconciler",
    "OptimalReconciler",
    "NonNegativeOptimalReconciler",
    "TopdownReconciler",
    "Reconciler",
]

# Make legacy submodule paths resolve to the new flat modules.
# All were private files; aliased only for pickle compat (objects pickled with
# previous releases store the old private ``__module__``).
_legacy_to_new = {
    "_reconcile": "sktime.transformations.reconcile",
    "_bottom_up": "sktime.transformations.bottom_up_reconciler",
    "_middle_out": "sktime.transformations.middle_out_reconciler",
    "_optimal": "sktime.transformations.optimal_reconciler",
    "_topdown": "sktime.transformations.topdown_reconciler",
    "_base": "sktime.transformations._hierarchical_reconcile_base",
    "_utils": "sktime.transformations._hierarchical_reconcile_utils",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.hierarchical.reconcile is deprecated and will be "
    "removed in a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.reconcile.Reconciler).",
    DeprecationWarning,
    stacklevel=2,
)
