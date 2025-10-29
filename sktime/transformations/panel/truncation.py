"""Truncation transformer - truncate unequal length panels to lower/upper bounds."""

from sktime.transformations.base import BaseTransformer

__all__ = ["TruncationTransformer"]


class TruncationTransformer(BaseTransformer):
    """Truncates unequal length panels between lower/upper length ranges.

    Truncates each series in ``transform`` to ``iloc`` between integers
    ``lower`` (inclusive) and ``upper`` (exclusive).

    If ``lower`` is ``None``, it is set to ``0``.

    If ``upper`` is ``None``, it is set to the length of the shortest series
    in the panel passed to ``fit``.

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
    >>> tt = TruncationTransformer(upper=5)
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
        "authors": ["abostrom", "Astrael1", "fkiraly"],
        "maintainers": ["Astrael1"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "df-list",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
    }

    error_messages = {
        "lower_gt_0": "lower must be greater than or equal to 0",
        "upper_gt_lower": "upper must be greater than lower",
    }

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        super().__init__()
        self._validate_parameters()

    @staticmethod
    def _get_min_length(X):
        """Get the minimum length of series in a list of np.ndarrays.

        Parameters
        ----------
        X : list of np.ndarrays
            List of arrays to get the minimum length from.

        Returns
        -------
        min_length : int
            Minimum length of series in X.
        """
        return min(x.shape[0] for x in X)

    def _validate_parameters(self):
        if self.lower is not None:
            if self.lower < 0:
                raise ValueError(self.error_messages["lower_gt_0"])
            if self.upper is not None and self.upper <= self.lower:
                raise ValueError(self.error_messages["upper_gt_lower"])

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : list of pd.DataFrame
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        array_list = [x.values for x in X]

        if self.upper is None:
            self._upper = self._get_min_length(array_list)
        else:
            self._upper = self.upper

        if self.lower is None:
            self._lower = 0
        else:
            self._lower = self.lower

        return self

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
        idx = slice(self._lower, self._upper)
        Xt = [x.iloc[idx] for x in X]
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
