"""Truncation transformer - truncate unequal length panels to lower/upper bounds."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__all__ = ["TruncationTransformer"]
__author__ = ["abostrom"]


class TruncationTransformer(BaseTransformer):
    """
    Truncates unequal length panels between lower/upper length ranges.

    If ``lower`` and ``upper`` are None the transformer will truncate
    each series to ``iloc`` indices from range [0, min_series_length)
    where ``min_series_length`` is the length of the shortest series in the panel
    calculated automatically.

    If ``lower`` is set and ``upper`` is None the transformer will truncate
    each series to ``iloc`` indices from range [lower, min_series_length)
    where ``min_series_length`` is the length of the shortest series in the panel
    calculated automatically.

    If ``upper`` is set and ``lower`` is None the transformer will truncate
    each series to ``iloc`` indices from range [0, upper).

    If both ``lower`` and ``upper`` are set the transformer will truncate
    each series to ``iloc`` indices from range [lower, upper).

    Parameters
    ----------
    lower : int, optional (default=None) minimum length, inclusive
        If None, will find the length of the shortest series and use instead.
    upper : int, optional (default=None) maximum length, exclusive
        Cannot be less than the length of the shortest series in the panel.
        This is used to calculate the range between.
        If None, will find the length of the shortest series and use instead.

    Examples
    --------
    Truncate only unequal length panels in data:
    >>> from sktime.transformations.panel.truncation import TruncationTransformer
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical(same_cutoff=False)
    >>> tt = TruncationTransformer()
    >>> tt.fit(X)
    TruncationTransformer(...)
    >>> X_transformed = tt.transform(X)

    Truncate each panel to first 5 elements:
    >>> from sktime.transformations.panel.truncation import TruncationTransformer
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical(same_cutoff=False)
    >>> tt = TruncationTransformer(lower=5)
    >>> tt.fit(X)
    TruncationTransformer(...)
    >>> X_transformed = tt.transform(X)

    Pick range from index 1 (inclusively) to 3 (exclusively):
    >>> from sktime.transformations.panel.truncation import TruncationTransformer
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical(same_cutoff=False)
    >>> tt = TruncationTransformer(lower=1, upper=3)
    >>> tt.fit(X)
    TruncationTransformer(...)
    >>> X_transformed = tt.transform(X)
    """

    _tags = {
        "authors": ["abostrom"],
        "maintainers": ["abostrom"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
    }

    error_messages = {
        "lower_gt_0": "lower must be greater than or equal to 0",
        "upper_gt_lower": "upper must be greater than lower",
        "upper_le_min_length": "upper must be less than \
            or equal to the length of the shortest series in the panel.",
    }

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        self._validate_parameters()
        super().__init__()

    @staticmethod
    def _get_min_length(X):
        def get_length(input):
            return min(map(lambda series: len(series), input))

        return min(map(get_length, X))

    def _validate_parameters(self):
        if self.lower is not None:
            if self.lower < 0:
                raise ValueError(self.error_messages["lower_gt_0"])
            if self.upper is not None and self.upper <= self.lower:
                raise ValueError(self.error_messages["upper_gt_lower"])

    def _data_to_series_list(self, X):
        n_instances, _ = X.shape
        return [X.iloc[i, :].values for i in range(n_instances)]

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        self.series_list = self._data_to_series_list(X)
        self._min_length = self._get_min_length(self.series_list)
        self._validate_upper_with_data()

        return self

    def _validate_upper_with_data(self):
        if self.upper is not None and self.upper > self._min_length:
            raise ValueError(self.error_messages["upper_le_min_length"])

    def _get_truncation_indices(self):
        """Get the truncation indices based on lower and upper bounds."""
        if self.lower is None and self.upper is None:
            idxs = np.arange(self._min_length)
        elif self.upper is not None and self.lower is None:
            idxs = np.arange(self.upper)
        elif self.upper is None and self.lower is not None:
            idxs = np.arange(self.lower, self._min_length)
        else:
            idxs = np.arange(self.lower, self.upper)
        return idxs

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        idxs = self._get_truncation_indices()
        truncate = [
            pd.Series([series.iloc[idxs] for series in out]) for out in self.series_list
        ]

        Xt = pd.DataFrame(truncate)
        Xt.columns = X.columns
        Xt.index = X.index
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"lower": None, "upper": None},
            {"lower": 5},
            {"lower": 1, "upper": 2},
        ]
        return params
