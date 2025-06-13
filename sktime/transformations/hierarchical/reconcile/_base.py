from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile._utils import (
    _loc_series_idxs,
)

__all__ = ["_ReconcilerTransformer"]


def _has_hierarchy(X):
    """Check if the data has hierarchy.

    To have an hierarchy, it must either:
    - Have more than 2 index levels
    - Have a __total level at the second index level
    """
    n_levels_geq_3 = X.index.nlevels >= 3
    if n_levels_geq_3:
        return True
    n_levels_eq_2 = X.index.nlevels == 2
    if n_levels_eq_2:
        has___total_at_last_non_temporal_level = "__total" in X.index.get_level_values(
            -2
        )
        return has___total_at_last_non_temporal_level
    return False


class _ReconcilerTransformer(BaseTransformer):
    """Base class of reconcilers that follow the reconciliation API.

    In the reconciliation API, `transform` filters and prepares the data to
    forecast and `inverse_transform` reconciles the forecasts.

    Reconcilers work as identity transformations if the data is not
    hierarchical.

    Children should implement:
    - `_fit_reconciler(X, y)`
    - `_transform_reconciler(X, y)`
    - `_inverse_transform_reconciler(X, y)`
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        # estimator type
        # --------------
        "object_type": ["reconciler", "transformer"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        # todo instance wise?
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": True,  # does transformer have inverse
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def _fit(self, X, y=None):
        """
        Fit reconciler.

        Sets `_no_hierarchy` and `_original_series` attributes.
        Calls `_fit_reconciler` to fit the reconciler.

        Parameters
        ----------
        X : pd.DataFrame
            The target timeseries
        y : pd.DataFrame, optional
            Exogenous variables

        Returns
        -------
        self
        """
        self._no_hierarchy = not _has_hierarchy(X)
        if self._no_hierarchy:
            return self
        self._original_series = X.index.droplevel(-1).unique()
        self.aggregator_ = Aggregator(bypass_inverse_transform=False)
        self.aggregator_.fit(X)
        X = self.aggregator_.transform(X)
        return self._fit_reconciler(X, y)

    def _transform(self, X, y=None):
        """
        Transform data.

        Calls `_transform_reconciler` to transform the data.
        If `_no_hierarchy` is True, returns the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The target timeseries
        y : pd.DataFrame, optional
            Exogenous variables

        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        if self._no_hierarchy:
            return X
        X = self.aggregator_.transform(X)
        return self._transform_reconciler(X, y)

    def _inverse_transform(self, X, y):
        """Apply reconciliation.

        Calls `_inverse_transform_reconciler` to reconcile the forecasts.
        If `_no_hierarchy` is True, returns the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The target timeseries
        y : pd.DataFrame, optional
            Exogenous variables

        Returns
        -------
        pd.DataFrame
            Reconciled forecasts
        """
        if self._no_hierarchy:
            return X

        _X = self._inverse_transform_reconciler(X, y)

        # We ensure that all series
        # given in fit are present in the output
        _X = Aggregator(flatten_single_levels=False).fit_transform(_X)
        _X = _loc_series_idxs(_X, self._original_series).sort_index()
        return _X

    def _fit_reconciler(self, X, y):
        """Fit reconciler."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _fit_reconciler."
        )

    def _transform_reconciler(self, X, y):
        """Transform data."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _transform_reconciler."
        )

    def _inverse_transform_reconciler(self, X, y):
        """Apply reconciliation."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _inverse_transform_reconciler."
        )
