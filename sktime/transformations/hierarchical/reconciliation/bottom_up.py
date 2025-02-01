"""Single-level reconciliation."""

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconciliation._utils import (
    _is_hierarchical_dataframe,
    get_bottom_level_idxs,
    loc_series_idxs,
)

__all__ = [
    "BottomUpReconciler",
]


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

        X = Aggregator(flatten_single_levels=False).fit_transform(X)
        X = loc_series_idxs(X, self._original_series).sort_index()

        return X
