"""Topdown hierarchical reconciliation transformer."""

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconciliation._utils import (
    _is_hierarchical_dataframe,
    get_bottom_level_idxs,
    get_total_level_idxs,
    loc_series_idxs,
)

__all__ = ["TopdownShareReconciler"]


class TopdownShareReconciler(BaseTransformer):
    """
    Topdown reconciliation.

    TopdownReconciler is a transformer for hierarchical time series reconciliation
    using the top-down approach.

    Attributes
    ----------
    _tags : dict
        Common tags for the transformer.
    _no_hierarchy : bool
        Indicates if the input series has no hierarchy.
    _original_series : pd.Index
        Original series index without the bottom level.
    _aggregator : Aggregator
        Aggregator instance to aggregate the series.
    _total_series : pd.Index
        Index of the total level series.
    _bottom_series : pd.Index
        Index of the bottom level series.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        # estimator type
        # --------------
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
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        "capability:hierarchical_reconciliation": True,
    }

    def _fit(self, X, y):
        self._no_hierarchy = not _is_hierarchical_dataframe(X)

        if self._no_hierarchy:
            return self

        self._original_series = X.index.droplevel(-1).unique()
        self._aggregator = Aggregator()
        self._aggregator.fit(X)
        X = self._aggregator.transform(X)

        self._total_series = get_total_level_idxs(X)
        self._bottom_series = get_bottom_level_idxs(X)

        return self

    def _transform(self, X, y):
        if self._no_hierarchy:
            return X

        # X = self._aggregator.transform(X)
        X_total = loc_series_idxs(X, self._total_series)
        X_bottom = loc_series_idxs(X, self._bottom_series)

        X_total.index = X_total.index.get_level_values(-1)

        X_total_expanded = X_total.reindex(X_bottom.index.get_level_values(-1))
        X_total_expanded.index = X_bottom.index

        shares = X_bottom / X_total_expanded

        _X = pd.concat([loc_series_idxs(X, self._total_series), shares], axis=0)
        _X = loc_series_idxs(_X, self._original_series).sort_index()

        return _X

    def _inverse_transform(self, X, y):
        if self._no_hierarchy:
            return X

        X_shares = X.copy()
        X_shares.loc[X_shares.index.droplevel(-1).isin(self._total_series)] = 1

        # In the future, we could add an option to keep all levels
        # and propage the shares from top to bottom as in
        # ForecastProportions
        # X_shares = _recursively_propagate_topdown(X_shares)

        # Adjust so that shares sum to 1
        X_shares_not_total = X_shares.loc[
            ~X_shares.index.droplevel(-1).isin(self._total_series)
        ]
        X_shares_not_total = X_shares_not_total / X_shares_not_total.groupby(
            level=-1
        ).transform("sum")
        X_shares.loc[X_shares_not_total.index] = X_shares_not_total

        X_total = loc_series_idxs(X, self._total_series)

        X_total.index = X_total.index.get_level_values(-1)

        # Reindex total series to match bottom series
        X_total_expanded = X_total.loc[X_shares.index.get_level_values(-1)]
        X_total_expanded.index = X_shares.index

        forecasts_from_shares = X_shares * X_total_expanded

        _X = Aggregator(False).fit_transform(forecasts_from_shares)
        _X = loc_series_idxs(_X, self._original_series).sort_index()
        return _X
