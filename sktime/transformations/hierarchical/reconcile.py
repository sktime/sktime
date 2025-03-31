# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements hierarchical reconciliation transformers.

These reconcilers only depend on the structure of the hierarchy.
"""

# TODO(fangelim): This module will be removed in sktime version 0.38.0
__all__ = [
    "Reconciler",
]

from warnings import warn

from sktime.transformations.hierarchical.reconciliation.reconcile_forecasts import (
    ReconcileForecasts as Reconciler,
    _get_s_matrix,
)

warn(
    "Reconciler will be renamed from `Reconciler` to"
    "`ReconcileForecasts`, and moved to sktime.transformations.hierarchical."
    "reconciliation.reconcile_forecasts in sktime version 0.38.0. "
    "To retain prior behaviour, please adjust your imports to use "
    "`ReconcileForecasts`.",
    category=DeprecationWarning,
    stacklevel=2,
)
