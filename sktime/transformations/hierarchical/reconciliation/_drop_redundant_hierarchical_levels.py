from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconciliation._utils import (
    _is_hierarchical_dataframe,
)

__all__ = ["_DropRedundantHierarchicalLevels"]


class _DropRedundantHierarchicalLevels(BaseTransformer):
    """
    Drop redundant levels from multiindex.

    Sometimes, the multiindex can have redundant levels, for example:

    __total, __total, pd.Period("2020-01-01")
    stateA, regionA, pd.Period("2020-01-01")
    stateA, regionB, pd.Period("2020-01-01")

    In this case, stateA is already total at level 0.
    This transformer will drop the first level, as it is redundant.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        # estimator type
        # --------------
        "scitype:transform-input": ["Series", "Hierachical"],
        "scitype:transform-output": ["Series", "Hierarchical", "Panel"],
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
        "capability:hierarchical_reconciliation": False,
    }

    def _fit(self, X, y):
        self._no_hierarchy = not _is_hierarchical_dataframe(X)
        if self._no_hierarchy:
            return self

        self._aggregator = Aggregator(False)
        Xt = self._aggregator.fit_transform(X)

        first_level_with_more_than_one_value = X.index.nlevels
        for level in range(Xt.index.nlevels - 1):
            nuniques = Xt.index.get_level_values(level).drop("__total").nunique()

            if nuniques > 1:
                first_level_with_more_than_one_value = level
                break

        self.levels_to_drop_ = list(range(first_level_with_more_than_one_value))

        self._idx = X.index.droplevel(-1).unique()
        self._dummy_idx_names = ["dummy" + str(i) for i in range(X.index.nlevels)]
        self._idx_names = X.index.names
        return self

    def _transform(self, X, y):
        if self._no_hierarchy:
            return X

        return X.droplevel(self.levels_to_drop_)

    def _inverse_transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        _X = X.copy()

        # To account for when indexes are unnamed
        self._dummy_idx_names = ["dummy" + str(i) for i in range(len(self._idx_names))]
        idx = self._idx.copy()
        idx.names = self._dummy_idx_names[:-1]
        _X.index.names = self._dummy_idx_names[len(self.levels_to_drop_) :]

        new_idx = idx.join(_X.index, how="right")
        _X.index = new_idx

        correct_index_order = list(range(self._idx.nlevels + 1))
        correct_index_order = (
            correct_index_order[X.index.nlevels :]
            + correct_index_order[: X.index.nlevels]
        )
        _X = _X.reorder_levels(correct_index_order)
        _X.index.names = self._idx_names
        return _X
