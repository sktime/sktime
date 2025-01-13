"""Single-level reconciliation."""

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator

__all__ = ["TopdownReconciler", "BottomUpReconciler", "MiddleOutReconciler"]


def loc_series_idxs(y, idxs):
    return y.loc[y.index.droplevel(-1).isin(idxs)]


def get_bottom_level_idxs(y):
    idx = y.index
    idx = idx.droplevel(-1).unique()
    return idx[idx.get_level_values(-1) != "__total"]


def get_bottom_series(y):
    bottom_idx = get_bottom_level_idxs(y)
    return loc_series_idxs(y, bottom_idx)


def get_total_level_idxs(y):
    nlevels = y.index.droplevel(-1).nlevels
    return pd.Index([tuple(["__total"] * nlevels)])


def get_total_series(y):
    total_idx = get_total_level_idxs(y)
    return loc_series_idxs(y, total_idx)


def get_middle_level_series(y, middle_level):
    idx = y.index
    idx = idx.droplevel(-1).unique()
    idx_middle_level = idx.get_level_values(middle_level)
    idx_below_middle_level = idx.get_level_values(middle_level + 1)
    is_middle_level_mask = (idx_below_middle_level == "__total") & (
        idx_middle_level != "__total"
    )
    return idx[is_middle_level_mask]


def split_middle_levels(y, middle_level):
    """
    Return two series: middle level and above, and middle level and below.
    """
    idx = y.index
    idx = idx.droplevel(-1).unique()
    idx_middle_level = idx.get_level_values(middle_level)
    idx_below_middle_level = idx.get_level_values(middle_level + 1)
    is_middle_or_above_mask = idx_below_middle_level == "__total"
    is_middle_or_below_mask = idx_middle_level != "__total"

    return loc_series_idxs(y, idx[is_middle_or_above_mask]), loc_series_idxs(
        y, idx[is_middle_or_below_mask]
    )


_COMMON_TAGS = {
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
    "capability:inverse_transform": False,  # does transformer have inverse
    "skip-inverse-transform": True,  # is inverse-transform skipped when called?
    "univariate-only": False,  # can the transformer handle multivariate X?
    "handles-missing-data": False,  # can estimator handle missing data?
    "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
    "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
    "transform-returns-same-time-index": False,
}


class TopdownReconciler(BaseTransformer):
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

    _tags = _COMMON_TAGS

    def _fit(self, X, y):
        self._no_hierarchy = X.index.nlevels == 1

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

        X = self._aggregator.transform(X)
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

        X_total = loc_series_idxs(X, self._total_series)
        X_bottom = loc_series_idxs(X, self._bottom_series)

        # Keep only timeindex of total series
        X_total.index = X_total.index.get_level_values(-1)

        # Reindex total series to match bottom series
        X_total_expanded = X_total.reindex(X_bottom.index.get_level_values(-1))
        X_total_expanded.index = X_bottom.index

        forecasts_from_shares = X_bottom * X_total_expanded
        _X = pd.concat(
            [loc_series_idxs(X, self._original_series), forecasts_from_shares], axis=0
        )

        _X = self._aggregator.transform(_X)
        _X = loc_series_idxs(_X, self._original_series).sort_index()
        return _X


class BottomUpReconciler(BaseTransformer):
    """
    Bottom-up reconciliation for hierarchical time series data.

    It aggregates the data to the bottom level and then transforms it back
    to the original hierarchy.

    Attributes
    ----------
    _tags : dict
        Common tags for the transformer.
    _no_hierarchy : bool
        Indicates if the data has no hierarchy (single level).
    _original_series : pd.Index
        The original series index before transformation.
    _aggregator : Aggregator
        An instance of the Aggregator class used for aggregation.
    _bottom_series : pd.Index
        The index of the bottom level series.
    """

    _tags = _COMMON_TAGS

    def _fit(self, X, y):
        self._no_hierarchy = X.index.nlevels == 1

        if self._no_hierarchy:
            return self

        self._original_series = X.index.droplevel(-1).unique()
        self._aggregator = Aggregator()
        self._aggregator.fit(X)
        X = self._aggregator.transform(X)

        self._bottom_series = get_bottom_level_idxs(X)

        return self

    def _transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        X = self._aggregator.transform(X)
        X_bottom = loc_series_idxs(X, self._bottom_series)

        return X_bottom

    def _inverse_transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        X = self._aggregator.transform(X)
        X = loc_series_idxs(X, self._original_series).sort_index()

        return X


class MiddleOutReconciler(BaseTransformer):
    """
    Reconciliation using a middle-out approach.

    Parameters
    ----------
    middle_level : int
        The level at which to split the hierarchy for reconciliation.
    middle_top_reconciler : BaseTransformer
        The transformer to use for the top part of the hierarchy.
    middle_bottom_reconciler : BaseTransformer
        The transformer to use for the bottom part of the hierarchy.
    """

    def __init__(
        self,
        middle_level: int,
        middle_top_reconciler: BaseTransformer,
        middle_bottom_reconciler: BaseTransformer,
    ):
        self.middle_level = middle_level
        self.middle_top_reconciler = middle_top_reconciler
        self.middle_bottom_reconciler = middle_bottom_reconciler

        super().__init__()

    def _fit(self, X, y):
        self._no_hierarchy = X.index.nlevels == 1

        if self._no_hierarchy:
            # TODO(fangelim): Should we raise a warning here?
            return self

        X_middle_top, X_middle_bottom = split_middle_levels(X, self.middle_level)

        self.middle_top_reconciler.fit(X=X_middle_top, y=y)
        self.middle_bottom_reconciler.fit(X=X_middle_bottom, y=y)

        return self

    def _transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        X_middle_top, X_middle_bottom = split_middle_levels(X, self.middle_level)

        X_middle_top = self.middle_top_reconciler.transform(X_middle_top)
        X_middle_bottom = self.middle_bottom_reconciler.transform(X_middle_bottom)

        _X = pd.concat([X_middle_top, X_middle_bottom], axis=0).sort_index()

        # Remove middle level duplicates
        _X = _X[~_X.index.duplicated(keep="first")]
        return _X

    def _inverse_transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        X_middle_top, X_middle_bottom = split_middle_levels(X, self.middle_level)

        X_middle_top = self.middle_top_reconciler.inverse_transform(X_middle_top)
        X_middle_bottom = self.middle_bottom_reconciler.inverse_transform(
            X_middle_bottom
        )

        _X = pd.concat([X_middle_top, X_middle_bottom], axis=0).sort_index()

        # Remove middle level duplicates
        _X = _X[~_X.index.duplicated(keep="first")]
        return _X
