"""Bottomup reconciliation."""

from sktime.transformations.hierarchical.reconcile._base import _ReconcilerTransformer
from sktime.transformations.hierarchical.reconcile._utils import (
    _get_bottom_level_idxs,
    _loc_series_idxs,
)

__all__ = [
    "BottomUpReconciler",
]


class BottomUpReconciler(_ReconcilerTransformer):
    """
    Bottom-up reconciliation for hierarchical time series data.

    It aggregates the data to the bottom level and then transforms it back
    to the original hierarchy.

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import (
    ...     BottomUpReconciler)
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = _make_hierarchical()
    >>> pipe = BottomUpReconciler() * NaiveForecaster()
    >>> pipe = pipe.fit(y)
    >>> y_pred = pipe.predict(fh=[1, 2, 3])
    """

    _tags = {
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def _fit_reconciler(self, X, y=None):
        """
        Fit the reconciler.

        Sets the bottom level series index.

        Parameters
        ----------
        X : pd.DataFrame
            The target timeseries
        y : pd.Series
            Exogenous variables, ignored.
        """
        self._bottom_series = _get_bottom_level_idxs(X)

        return self

    def _transform_reconciler(self, X, y=None):
        """
        Filter the bottom level series.

        Parameters
        ----------
        X : pd.DataFrame
            The target timeseries
        y : pd.Series
            Exogenous variables, ignored.
        """
        X_bottom = _loc_series_idxs(X, self._bottom_series)

        return X_bottom

    def _inverse_transform_reconciler(self, X, y=None):
        # Do not need to do anything
        # Since the base reconciler already aggregates and filters the
        # timeseries
        return X
