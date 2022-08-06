# -*- coding: utf-8 -*-
"""Interval and window segmenter transformers."""
import math

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.datatypes._panel._convert import (
    _concat_nested_arrays,
    _get_column_names,
    _get_time_index,
)
from sktime.transformations.base import BaseTransformer
from sktime.utils.validation import check_window_length
from sktime.utils.validation.panel import check_X


class IntervalSegmenter(BaseTransformer):
    """Interval segmentation transformer.

    Parameters
    ----------
    intervals : int, np.ndarray or list of np.ndarrays with one for each
    column of input data.
        Intervals to generate.
        - If int, intervals gives the number of generated intervals.
        - If ndarray, 2d np.ndarray [n_intervals, 2] with rows giving
        intervals, the first column giving start points,
        and the second column giving end points of intervals
    """

    _tags = {
        "univariate-only": True,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
    }

    def __init__(self, intervals=10):
        self.intervals = intervals
        self._time_index = []
        self.input_shape_ = ()
        super(IntervalSegmenter, self).__init__()

    def _fit(self, X, y=None):
        """
        Fit transformer, generating random interval indices.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : an instance of self.
        """
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

        n_instances, n_columns, n_timepoints = X.shape
        self.input_shape_ = n_instances, n_columns, n_timepoints

        self._time_index = np.arange(n_timepoints)

        if isinstance(self.intervals, np.ndarray):
            self.intervals_ = list(self.intervals)

        elif isinstance(self.intervals, (int, np.integer)):
            if not self.intervals <= n_timepoints // 2:
                raise ValueError(
                    "The number of intervals must be half the number of time points"
                    " or less"
                )
            self.intervals_ = np.array_split(self._time_index, self.intervals)

        else:
            raise ValueError(
                f"Intervals must be either an integer, an array with "
                f"start and end points, but found: {self.intervals}"
            )
        return self

    def _transform(self, X, y=None):
        """Transform input series.

        Transform X, segments time-series in each column into random
        intervals using interval indices generated
        during `fit`.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one
          column for each generated interval.
        """
        # Tabularise assuming series
        # have equal indexes in any given column
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        # Segment into intervals.
        # TODO generalise to non-equal-index cases
        intervals = []

        # univariate, only a single column name
        column_names = _get_column_names(X)[0]
        new_column_names = []
        for interval in self.intervals_:
            start, end = interval[0], interval[-1]
            if f"{column_names}_{start}_{end}" not in new_column_names:
                interval = X[:, start:end]
                intervals.append(interval)
                new_column_names.append(f"{column_names}_{start}_{end}")

        # Return nested pandas DataFrame.
        Xt = pd.DataFrame(_concat_nested_arrays(intervals))
        Xt.columns = new_column_names
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # small number of intervals for testing
        params = {"intervals": 2}
        return params


class RandomIntervalSegmenter(IntervalSegmenter):
    """Random interval segmenter transformer.

    Transformer that segments time-series into random intervals with
    random starting points and lengths. Some
    intervals may overlap and may be duplicates.

    Parameters
    ----------
    n_intervals : str, int or float
        Number of intervals to generate.
        - If "log", log of m is used where m is length of time series.
        - If "sqrt", sqrt of m is used.
        - If "random", random number of intervals is generated.
        - If int, n_intervals intervals are generated.
        - If float, int(n_intervals * m) is used with n_intervals giving the
        fraction of intervals of the
        time series length.

        For all arguments relative to the length of the time series,
        the generated number of intervals is
        always at least 1.

        Default is "sqrt".

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    _tags = {
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd_Series_Table",
        # which mtypes do _fit/_predict support for y?
    }

    def __init__(
        self, n_intervals="sqrt", min_length=None, max_length=None, random_state=None
    ):
        self.n_intervals = n_intervals
        self.min_length = min_length
        self.max_length = max_length
        self.random_state = random_state
        super(RandomIntervalSegmenter, self).__init__()

    def _fit(self, X, y=None):
        """Fit transformer, generating random interval indices.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : any container with method shape, optional, default=None
            y.shape[0] determines n_timepoints, 1 if None

        Returns
        -------
        self : RandomIntervalSegmenter
            This estimator
        """
        if y is not None:
            n_timepoints = y.shape[0]
        else:
            n_timepoints = 1

        self.min_length = check_window_length(
            self.min_length, n_timepoints, "min_length"
        )
        self.max_length = check_window_length(
            self.max_length, n_timepoints, "max_length"
        )
        if self.min_length is None:
            min_length = 2
        else:
            min_length = self.min_length
        if self.max_length is not None:
            if not min_length < self.max_length:
                raise ValueError("`max_length` must be bigger than `min_length`.")

        self.input_shape_ = X.shape

        # Retrieve time-series indexes from each column.
        # TODO generalise to columns with series of unequal length
        self._time_index = _get_time_index(X)

        # Compute random intervals for each column.
        # TODO if multiple columns are passed, introduce option to compute
        #  one set of shared intervals,
        #  or rely on ColumnTransformer?
        if self.n_intervals == "random":
            if self.min_length is not None or self.max_length is not None:
                raise ValueError(
                    "Setting `min_length` or `max_length` is not yet "
                    "implemented for `n_intervals='random'`."
                )
            self.intervals_ = _rand_intervals_rand_n(
                self._time_index, random_state=self.random_state
            )
        else:
            self.intervals_ = _rand_intervals_fixed_n(
                self._time_index,
                n_intervals=self.n_intervals,
                min_length=min_length,
                max_length=self.max_length,
                random_state=self.random_state,
            )
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # we need to override this, or it inherits from IntervalSegmenter
        #   but this estimator does not have an "intervals" parameter
        return {}


def _rand_intervals_rand_n(x, random_state=None):
    """Sample a random number of intervals.

    Compute a random number of intervals from index (x) with
    random starting points and lengths. Intervals are unique, but may
    overlap.

    Parameters
    ----------
    x : array_like, shape = (n_timepoints,)

    Returns
    -------
    intervals : array-like, shape = (n_intervals, 2)
        2d array containing start and end points of intervals

    References
    ----------
    .. [1] Deng, Houtao, et al. "A time series forest for classification
    and feature extraction."
        Information Sciences 239 (2013): 142-153.
    """
    rng = check_random_state(random_state)
    starts = []
    ends = []
    n_timepoints = x.shape[0]  # series length
    W = rng.randint(1, n_timepoints, size=int(np.sqrt(n_timepoints)))
    for w in W:
        size = n_timepoints - w + 1
        start = rng.randint(size, size=int(np.sqrt(size)))
        starts.extend(start)
        for s in start:
            end = s + w
            ends.append(end)
    return np.column_stack([starts, ends])


def _rand_intervals_fixed_n(
    x, n_intervals, min_length=1, max_length=None, random_state=None
):
    """Sample a fixed number of intervals.

    Compute a fixed number (n) of intervals from index (x) with
    random starting points and lengths. Intervals may overlap and may
    not be unique.

    Parameters
    ----------
    x : array_like, shape = (n_timepoints,)
        Array containing the time-series index.
    n_intervals : 'sqrt', 'log', float or int

    Returns
    -------
    intervals : array-like, shape = (n_intervals, 2)
        2d array containing start and end points of intervals
    """
    rng = check_random_state(random_state)
    n_timepoints = x.shape[0]
    n_intervals = _get_n_from_n_timepoints(n_timepoints, n_intervals)
    starts = rng.randint(0, n_timepoints - min_length + 1, size=(n_intervals,))
    if max_length is None:
        max_length = n_timepoints - starts
    ends = rng.randint(starts + min_length, starts + max_length + 1)
    return np.column_stack([starts, ends])


class SlidingWindowSegmenter(BaseTransformer):
    """Sliding window segmenter transformer.

    This class is to transform a univariate series into a
    multivariate one by extracting sets of subsequences.
    It does this by firstly padding the time series on either end
    floor(window_length/2) times. Then it performs a sliding
    window of size window_length and hop size 1.

    e.g. if window_length = 3

    S = 1,2,3,4,5, floor(3/2) = 1 so S would be padded as

    1,1,2,3,4,5,5

    then SlidingWindowSegmenter would extract the following:

    (1,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,5)

    the time series is now a multivariate one.

    Parameters
    ----------
        window_length : int, optional, default=5.
            length of sliding window interval

    Returns
    -------
        df : pandas dataframe of shape
             [n_instances, n_timepoints]

    Proposed in the ShapeDTW algorithm.
    """

    _tags = {
        "univariate-only": True,
        "fit_is_empty": True,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
    }

    def __init__(self, window_length=5):
        self.window_length = window_length
        super(SlidingWindowSegmenter, self).__init__()

    def _transform(self, X, y=None):
        """Transform time series.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        dims: a pandas data frame of shape = [n_instances, n_timepoints]
        """
        # get the number of attributes and instances
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        n_timepoints = X.shape[1]
        n_instances = X.shape[0]

        # Check the parameters are appropriate
        self._check_parameters(n_timepoints)

        pad_amnt = math.floor(self.window_length / 2)
        padded_data = np.zeros((n_instances, n_timepoints + (2 * pad_amnt)))

        # Pad both ends of X
        for i in range(n_instances):
            padded_data[i] = np.pad(X[i], pad_amnt, mode="edge")

        subsequences = np.zeros((n_instances, n_timepoints, self.window_length))

        # Extract subsequences
        for i in range(n_instances):
            subsequences[i] = self._extract_subsequences(padded_data[i], n_timepoints)

        # Convert this into a panda's data frame
        df = pd.DataFrame()
        for i in range(len(subsequences)):
            inst = subsequences[i]
            data = []
            for j in range(len(inst)):
                data.append(pd.Series(inst[j]))
            df[i] = data

        return df.transpose()

    def _extract_subsequences(self, instance, n_timepoints):
        """Extract a set of subsequences from a list of instances.

        Adopted from -
        https://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-
        construction-from-1d-array/4924433#4924433

        """
        shape = (n_timepoints, self.window_length)
        strides = (instance.itemsize, instance.itemsize)
        return np.lib.stride_tricks.as_strided(instance, shape=shape, strides=strides)

    def _check_parameters(self, n_timepoints):
        """Check the values of parameters for interval segmenter.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.window_length, int):
            if self.window_length <= 0:
                raise ValueError(
                    "window_length must have the \
                                  value of at least 1"
                )
        else:
            raise TypeError(
                "window_length must be an 'int'. \
                            Found '"
                + type(self.window_length).__name__
                + "' instead."
            )


def _get_n_from_n_timepoints(n_timepoints, n="sqrt"):
    """Get number of intervals from number of time points.

    Helpful to compute number of intervals relative to time series length,
    e.g. using floats or functions.

    Parameters
    ----------
    n_timepoints : int
    n : {int, float, str, callable}

    Returns
    -------
    n_intervals_ : int
        Computed number of intervals
    """
    # check input: n_timepoints
    if not np.issubdtype(type(n_timepoints), np.dtype(int).type):
        raise ValueError(
            f"`n_timepoints` must be an integer, but found: " f"{type(n_timepoints)}"
        )
    if not n_timepoints >= 1:
        raise ValueError(f"`n_timepoints` must be >= 1, but found: {n_timepoints}")

    # compute number of splits
    allowed_strings = ["sqrt", "log"]

    # integer
    if np.issubdtype(type(n), np.dtype(int).type):
        if not n <= n_timepoints:
            raise ValueError(
                f"If `n_intervals` is an integer, it must be smaller "
                f"than `n_timepoints`, but found:  `n_intervals`={n} "
                f"and `n_timepoints`={n_timepoints}"
            )
        if n < 1:
            raise ValueError(
                f"If `n_intervals` is an integer, "
                f"`n_intervals` must be >= 1, but found: {n}"
            )
        n_intervals_ = n

    # function
    elif callable(n):
        n_intervals_ = n(n_timepoints)

    # string
    elif isinstance(n, str):
        if n not in allowed_strings:
            raise ValueError(
                f"If `n_intervals` is a string, `n_intervals` must be "
                f"in {allowed_strings}, but found: {n}"
            )
        str_func_map = {"sqrt": np.sqrt, "log": np.log}
        func = str_func_map[n]
        n_intervals_ = func(n_timepoints)

    # float
    elif isinstance(n, float):
        if not (0 < n <= 1):
            raise ValueError(
                f"If `n_intervals` is a float, `n_intervals` must be > 0 "
                f"and <= 1, but found: {n}"
            )
        n_intervals_ = n * n_timepoints

    else:
        raise ValueError(
            f"`n_intervals` must be either one of the allowed string options "
            f"in "
            f"{allowed_strings}, an integer or a float number."
        )

    # make sure n_intervals is an integer and there is at least one interval
    n_intervals_ = np.maximum(1, int(n_intervals_))
    return n_intervals_
