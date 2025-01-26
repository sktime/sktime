"""Forecast-proportions reconciliation."""

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconciliation._drop_redundant_hierarchical_levels import (  # noqa: E501
    _DropRedundantHierarchicalLevels,
)
from sktime.transformations.hierarchical.reconciliation._utils import (
    _is_hierarchical_dataframe,
    _promote_hierarchical_indexes_and_keep_timeindex,
    _recursively_propagate_topdown,
    get_bottom_level_idxs,
    get_total_level_idxs,
    loc_series_idxs,
)

__all__ = ["ForecastProportions"]


class ForecastProportions(BaseTransformer):
    """
    Apply forecast proportions to hierarchical time series.

    Forecast proportions keep the original series during `transform`,
    and propagate the "proportions" of each forecast with respect to its
    total during `inverse_transform`.

    For more information, see "Single level approaches" in [1].

    References
    ----------
    .. [1] Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting:
       principles and practice. OTexts.

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

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        self._no_hierarchy = not _is_hierarchical_dataframe(X)

        if self._no_hierarchy:
            return self

        self._original_series = X.index.droplevel(-1).unique()
        self._aggregator = Aggregator()
        self._aggregator.fit(X)
        X = self._aggregator.transform(X)

        self._drop_redundant_levels = _DropRedundantHierarchicalLevels()
        self._drop_redundant_levels.fit(X)
        X = self._drop_redundant_levels.transform(X)

        self._total_series = get_total_level_idxs(X)
        self._bottom_series = get_bottom_level_idxs(X)

        return self

    def _transform(self, X, y):
        if self._no_hierarchy:
            return X
        X = self._drop_redundant_levels.transform(X)
        return X

    def _inverse_transform(self, X, y):
        if self._no_hierarchy:
            return X

        # In forecast proportions, we have to recursively apply
        # the ratio between the total forecast yhat/sum(yhat_lower)
        # to the bottom level series

        # To do this, we will create a dataframe with this ratio for each
        # series and timesteps

        # Map each series to its total series

        idx_map_parents = X.index.map(_promote_hierarchical_indexes_and_keep_timeindex)
        idx = X.index

        # Df mapping each value to its parent
        X_parents = X.copy()
        # Set total to 0 because of the operations below
        # which would account for it twice
        X_parents.loc[X_parents.index.isin(self._total_series)] = 0
        X_parents.index = idx_map_parents

        # The parent sum according to direct children
        X_parents_bu = X_parents.groupby(
            level=[i for i in range(X.index.nlevels)]
        ).sum()
        X_parents_bu = X_parents_bu.loc[idx_map_parents]
        X_parents_bu.index = idx

        # The forecasted parent total
        X_parents_total = X.loc[idx_map_parents]
        X_parents_total.index = idx

        X_ratios = X_parents_total / X_parents_bu
        X_ratios.index = X.index

        # Now, multiply the ratio down to the bottom level, recursively

        X_ratios_propagated = _recursively_propagate_topdown(X_ratios)
        _X = X_ratios_propagated * X

        _X = self._drop_redundant_levels.inverse_transform(_X)
        _X = loc_series_idxs(_X, self._original_series).sort_index()
        return _X
