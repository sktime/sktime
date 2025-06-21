"""Drop redundant levels from multiindex."""

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator

__all__ = ["SqueezeHierarchy"]


class SqueezeHierarchy(BaseTransformer):
    """
    Drop redundant levels from multiindex.

    Sometimes, the multiindex can have redundant levels, for example:

    ```
    __total,__total, __total, pd.Period("2020-01-01")   0.1
    countryA, stateA, regionA, pd.Period("2020-01-01")    0.05
    countryA, stateB, regionB, pd.Period("2020-01-01")    0.05
    ```

    In this case, countryA is already total at level 0.
    This transformer will drop the first level, as it is redundant.

    In cases where the redundant levels have different values,
    an error will be raised.

    Raises
    ------
    ValueError
        If there are values that are not the same for the same index.
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
        "y_inner_mtype": "None",
        "capability:inverse_transform": True,  # does transformer have inverse
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def _fit(self, X, y):
        self._no_hierarchy = X.index.nlevels < 3
        if self._no_hierarchy:
            return self

        self._aggregator = Aggregator(False)
        Xt = self._aggregator.fit_transform(X)

        # Here, we find the first level with more than one value
        # (disconsidering `__total``)
        # i.e. the first non-redundant level
        first_level_with_more_than_one_value = 0
        for level in range(Xt.index.nlevels - 1):
            nuniques = Xt.index.get_level_values(level).drop("__total").nunique()

            if nuniques > 1:
                first_level_with_more_than_one_value = level
                break

        # Convert to negative index
        # This makes possible to transform
        # with series with different number of levels
        self.levels_to_drop_ = [
            x - X.index.nlevels for x in range(first_level_with_more_than_one_value)
        ]

        # Remove level -3 from levels to drop, since Hierarchical representation
        # always have at least 3 levels
        # So if nlevels = 3, no levels will be dropped
        # If nlevels is 4, only the first level will be dropped
        self.levels_to_drop_ = [x for x in self.levels_to_drop_ if x < -3]

        Xt = X.droplevel(self.levels_to_drop_).sort_index()
        self._assert_no_inconsistent_duplicated_indexes(Xt)

        self._idx = X.index.droplevel(-1).unique()

        # We create dummy index names to account for when indexes are unnamed
        self._dummy_idx_names = ["dummy" + str(i) for i in range(X.index.nlevels)]
        self._idx_names = X.index.names
        return self

    def _assert_no_inconsistent_duplicated_indexes(self, X):
        X_mean = X.groupby(level=list(range(X.index.nlevels))).transform("mean")
        diff = (X_mean - X).abs()
        # Greater than zero
        if (diff > 1e-9).any().any():
            raise ValueError(
                "There are values that are not the same for the same index."
            )

    def _transform(self, X, y):
        if self._no_hierarchy:  # or isinstance(X, np.ndarray):
            return X

        levels_to_drop = [x for x in self.levels_to_drop_ if x >= -X.index.nlevels]

        X = X.droplevel(levels_to_drop).sort_index()

        indexes_to_keep = ~X.index.duplicated(keep="first")
        X = X[indexes_to_keep]
        return X

    def _inverse_transform(self, X, y=None):
        if self._no_hierarchy:  # or isinstance(X, np.ndarray):
            return X

        _X = X.copy()

        # To account for when indexes are unnamed we add dummy names
        self._dummy_idx_names = ["dummy" + str(i) for i in range(len(self._idx_names))]
        idx = self._idx.copy()
        idx.names = self._dummy_idx_names[:-1]
        _X.index.names = self._dummy_idx_names[len(self.levels_to_drop_) :]

        new_idx = idx.join(_X.index, how="left")
        new_idx = new_idx.reorder_levels(self._dummy_idx_names)
        _X.index = new_idx

        _X.index.names = self._idx_names
        return _X.sort_index()
