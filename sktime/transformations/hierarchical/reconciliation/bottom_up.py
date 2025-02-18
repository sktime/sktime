"""Single-level reconciliation."""

from sktime.transformations._reconcile import _ReconcilerTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconciliation._utils import (
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

    def _fit_reconciler(self, X, y):
        self._aggregator = Aggregator()
        self._aggregator.fit(X)
        X = self._aggregator.transform(X)

        self._bottom_series = _get_bottom_level_idxs(X)

        return self

    def _transform_reconciler(self, X, y=None):
        X = self._aggregator.transform(X)
        X_bottom = _loc_series_idxs(X, self._bottom_series)

        return X_bottom

    def _inverse_transform_reconciler(self, X, y=None):
        X = Aggregator(flatten_single_levels=False).fit_transform(X)
        X = _loc_series_idxs(X, self._original_series).sort_index()
        return X
