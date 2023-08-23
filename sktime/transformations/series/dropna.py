# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Transformer to drop rows or columns containing missing values."""

__author__ = ["hliebert"]

from sktime.transformations.base import BaseTransformer


class DropNA(BaseTransformer):
    """Drop missing values transformation.

    Drops rows or columns with missing values from X. Mostly wraps
    pandas.DataFrame.dropna, but allows specifying thresh as a fraction of
    non-missing observations.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are removed.
        Must be 0 or 'index' for univariate input.

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.

    how : {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.

    thresh : int or float, optional
         If int, require at least that many non-NA values (as in pandas.dropna).
         If float, minimum share of non-NA values for rows/columns to be
         retained. Fraction must be contained within (0,1]. Setting fraction
         to 1.0 is equivalent to setting how='any'. thresh cannot be combined
         with how.

    remember : bool, default False if axis==0, True if axis==1
        If True, drops the same rows/columns in transform as in fit. If false,
        drops rows/columns according to the NAs seen in transform (equivalent
        to PandasTransformAdaptor(method="dropna")).
    """

    _tags = {
        "univariate-only": False,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": False,
    }

    VALID_AXIS_VALUES = [0, "index", 1, "columns"]
    VALID_HOW_VALUES = [None, "any", "all"]
    VALID_THRESH_TYPES = (type(None), int, float)
    VALID_REMEMBER_TYPES = (type(None), bool)

    def __init__(self, axis=0, how=None, thresh=None, remember=None):
        self.axis = self._check_axis(axis)
        self.how = self._check_how(how)
        self.thresh = self._check_thresh(thresh, how)
        self.remember = self._check_remember(remember)
        super().__init__()

        # axis, use numeric axis internally, default rows/index
        if self.axis == "index":
            self._axis = 0
        elif self.axis == "columns":
            self._axis = 1
        else:
            self._axis = self.axis

        # criterion (how/thresh), default to "any" if neither how nor thresh passed
        if (self.how is None) and (self.thresh is None):
            self._how = "any"
        else:
            self._how = how
        self._thresh = self.thresh

        # remember, default to remember dropped columns but not rows
        if self.remember is None:
            self._remember = bool(self._axis)
        else:
            self._remember = self.remember

    def _check_axis(self, axis):
        """Check axis parameter, should be a valid string as per docstring."""
        if axis not in self.VALID_AXIS_VALUES:
            raise ValueError(
                f'invalid axis parameter value encountered: "{axis}", '
                f"axis must be one of: {self.VALID_AXIS_VALUES}"
            )

        return axis

    def _check_how(self, how):
        """Check how parameter, should be a valid string as per docstring."""
        if how not in self.VALID_HOW_VALUES:
            raise ValueError(
                f'invalid how parameter value encountered: "{how}", '
                f"how must be one of: {self.VALID_HOW_VALUES}"
            )

        return how

    def _check_thresh(self, thresh, how):
        """Check thresh parameter, should be a valid value as per docstring."""
        if not isinstance(thresh, self.VALID_THRESH_TYPES) or isinstance(thresh, bool):
            raise TypeError(
                f'invalid thresh parameter value encountered: "{thresh}", '
                f"thresh must be of type: {self.VALID_THRESH_TYPES}"
            )
        if (isinstance(thresh, int) and not (thresh > 0)) or (
            isinstance(thresh, float) and not (0 < thresh <= 1)
        ):
            raise ValueError(
                "thresh must be positive integer or a fraction between zero and one"
            )

        if (how is not None) and (thresh is not None):
            raise TypeError("thresh cannot be set together with how")

        return thresh

    def _check_remember(self, remember):
        """Check remember parameter, should be a valid type as per docstring."""
        if not isinstance(remember, self.VALID_REMEMBER_TYPES):
            raise TypeError(
                f'invalid remember parameter value encountered: "{remember}", '
                f"remember must be of type: {self.VALID_REMEMBER_TYPES}"
            )

        return remember

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("univariate-only")==True:
                guaranteed to have a single column
            if self.get_tag("univariate-only")==False: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        self: reference to self
        """
        self.dropped_index_values_ = None
        self._agg_axis = 1 - self._axis

        mask = None
        if self._how == "any":
            mask = X.isna().any(axis=self._agg_axis)
        elif self._how == "all":
            mask = X.isna().all(axis=self._agg_axis)
        elif isinstance(self._thresh, int):
            mask = X.count(axis=self._agg_axis) < self._thresh
        elif isinstance(self._thresh, float):
            mask = X.notna().mean(axis=self._agg_axis) < self._thresh

        if mask is not None:
            self.dropped_index_values_ = mask.index[mask].to_list()
        else:
            self.dropped_index_values_ = None

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("univariate-only")==True:
                guaranteed to have a single column
            if self.get_tag("univariate-only")==False: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        transformed version of X
        """
        dropped_index_values = self.dropped_index_values_
        agg_axis = self._agg_axis
        axis = self._axis
        how = self._how
        thresh = self._thresh
        remember = self._remember

        if remember:
            if dropped_index_values is not None:
                return X.drop(labels=dropped_index_values, axis=axis)
            else:
                return X
        else:
            if isinstance(thresh, float):
                mask = X.notna().mean(axis=agg_axis) < thresh
                index_to_drop = mask.index[mask]
                return X.drop(labels=index_to_drop, axis=axis)
            elif isinstance(thresh, int):
                return X.dropna(axis=axis, thresh=thresh)
            else:
                return X.dropna(axis=axis, how=how)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"axis": 0, "how": "any", "thresh": None},
            {"axis": 1, "how": "any", "thresh": None},
            {"axis": 0, "how": "all", "thresh": None},
            {"axis": 1, "how": "all", "thresh": None},
            {"axis": 0, "how": None, "thresh": 0.9},
            {"axis": 1, "how": None, "thresh": 0.9},
            {"axis": 1, "how": None, "thresh": 3},
        ]

        return params
