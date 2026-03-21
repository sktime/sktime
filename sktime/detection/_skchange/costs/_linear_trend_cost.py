"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a best fit linear trend line within each interval.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.validation.parameters import check_data_column
from .base import BaseCost


@njit
def fit_linear_trend(time_steps: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Parameters
    ----------
    x : np.ndarray
        1D array of data points

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    mean_t = np.mean(time_steps)
    centered_time_steps = time_steps - mean_t
    mean_value = np.mean(values)

    # Calculate linear regression denominator = sum((t-mean_t)²):
    denominator = np.sum(np.square(centered_time_steps))

    # Calculate linear regression numerator = sum((t-mean_t)*(x-mean_x)):
    numerator = np.sum(centered_time_steps * (values - mean_value))

    slope = numerator / denominator
    intercept = mean_value - slope * mean_t

    return slope, intercept


@njit
def fit_indexed_linear_trend(xs: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Assuming the time steps are [0, 1, 2, ..., n-1], we can optimize the calculation.

    Parameters
    ----------
    xs : np.ndarray
        1D array of data points

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    n_samples = len(xs)

    # For evenly spaced time steps [0, 1, 2, ..., n-1],
    # the mean time step is (n-1)/2.
    mean_t = (n_samples - 1) / 2.0

    # Optimized calculation for denominator:
    # sum of (t - mean_t)^2 = n*(n^2-1)/12
    denominator = n_samples * (n_samples * n_samples - 1) / 12.0

    # Calculate numerator: sum((t-mean_t)*(x-mean_x))
    # numerator = np.sum((np.arange(n) - mean_t) * (xs - mean_x))
    mean_x = np.mean(xs)
    numerator = 0.0
    for i in range(n_samples):
        numerator += (i - mean_t) * (xs[i] - mean_x)

    slope = numerator / denominator
    intercept = mean_x - slope * mean_t

    return slope, intercept


@njit
def linear_trend_cost_mle(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Evaluate the linear trend cost with optimized parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    ts : np.ndarray
        Time points the data points were observed at.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment_times = times[start:end]
        for col in range(n_columns):
            segment_data = X[start:end, col]
            slope, intercept = fit_linear_trend(
                time_steps=segment_times, values=segment_data
            )
            costs[i, col] = np.sum(
                np.square(segment_data - (intercept + slope * segment_times))
            )

    return costs


@njit
def linear_trend_cost_index_times_mle(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the linear trend cost with optimized parameters.

    Assumes that the data is observed at time steps [0, 1, 2, ..., n-1]
    within each segment, with n being the length of the segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            segment_data = X[start:end, col]

            slope, intercept = fit_indexed_linear_trend(segment_data)
            linear_trend_values = intercept + slope * np.arange(segment_data.shape[0])
            costs[i, col] = np.sum(np.square(segment_data - linear_trend_values))

    return costs


@njit
def linear_trend_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    time_steps: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Evaluate the linear trend cost with fixed parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    time_steps : np.ndarray
        Time points the data points were observed at.
    params : np.ndarray
        Fixed parameters for linear trend: [[slope_1, intercept_1],
                                            [slope_2, intercept_2],
                                             ...]
        where each pair corresponds to a column in X.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)

    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for col in range(n_columns):
        slope, intercept = params[col, :]
        for i in range(n_intervals):
            start, end = starts[i], ends[i]
            segment_data = X[start:end, col]
            segment_ts = time_steps[start:end]

            linear_trend_values = intercept + slope * segment_ts
            costs[i, col] = np.sum(np.square(segment_data - linear_trend_values))

    return costs


class LinearTrendCost(BaseCost):
    """Linear trend cost function.

    This cost function calculates the sum of squared errors between data points
    and a linear trend line fitted to each interval. For each interval and each column,
    a straight line is fitted (or provided as a fixed parameter) and the squared
    differences between the actual data points and the fitted line are summed.

    By default the time steps are assumed to be [0, 1, 2, ..., (start - end) - 1] for
    each segment. If a time column is provided, time steps are taken from that column.

    Inspired by [1]_ who propose the same cost function for detecting changes in
    piecewise-linear signals, but within an optimization problem which enforces
    continuity of the linear trend across segments. To achieve similar results
    in the `skchange` pacakge, we recommend using the `ContinuousLinearTrendScore`
    within a change detection algorithm such as `SeededBinarySegmentation` or
    `MovingWindow`.

    Parameters
    ----------
    param : array-like, optional (default=None)
        Fixed parameters for the cost calculation in the form:

            [[slope_1, intercept_1],

            [slope_2, intercept_2],

            ...],

        i.e. with shape: (n_columns, 2). Each pair of parameters corresponds to a column
        in the input data. If None, the optimal parameters (i.e., the best-fit line) are
        calculated for each interval.
    time_column : str or int, optional (default=None)
        By default time steps are assumed to be evenly spaced with unit distance.
        If a time column is provided, its time steps are used to calculate the
        linear trends. The time column values must be convertible to `float` datatype.
        If your time column is of `datetime.datetime` type, you can e.g.
        transform your `"datetime"` as
        ```
        ## Sane way of creating a numeric time column:
        df: pd.DataFrame
        df["time_elapsed"] = df["datetime"] - df.loc[0, "datetime"]
        df["num_timestamp"] = (
            df["time_elapsed"].dt.total_seconds() /\

            pd.Timedelta(hours=1).total_seconds()
        )  # Convert to hours
        ```
        and then pass `time_column = "num_timestamp"`.
    shared_linear_trend : bool, optional (default=False)
        If True, the same fixed linear trend parameters are used for all columns.

    References
    ----------
    .. [1] Fearnhead, P., & Grose, D. (2024). cpop: Detecting Changes in \
    Piecewise-Linear Signals. Journal of Statistical Software, 109(7), 1-30.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
    }

    def __init__(
        self,
        param=None,
        time_column: str | int | None = None,
        share_fixed_trend: bool = False,
    ):
        super().__init__(param)
        self.time_column = time_column
        self.share_fixed_trend = share_fixed_trend

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method stores the input data for later cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        if self.time_column is not None:
            self.time_column_idx = check_data_column(
                self.time_column, "Time", X, self._X_columns
            )
        else:
            self.time_column_idx = None

        if self.time_column_idx is not None:
            # If a time column is provided, convert it to float
            # dtype for numba compatibility:
            self._time_stamps = X[:, self.time_column_idx].astype(float)
            # Start at time zero for the first data point:
            self._time_stamps -= self._time_stamps[0]
        elif self.param is not None and self.time_column_idx is None:
            # If using fixed parameters, we need a time column:
            self._time_stamps = np.arange(X.shape[0])
        else:
            # No provided time column or fixed parameters, so we assume
            # the time steps are [0, 1, 2, ..., n-1] for each segment.
            self._time_stamps = None

        if self.time_column_idx is not None:
            trend_columns = np.delete(np.arange(X.shape[1]), self.time_column_idx)
            self._trend_data = X[:, trend_columns]
        else:
            self._trend_data = X

        self._param = self._check_param(self.param, self._trend_data)
        if self.param is not None:
            self._trend_params = self._param
        else:
            self._trend_params = None

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal linear trend parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        For each interval, a straight line is fitted and the squared differences
        between the actual data points and the fitted line are summed.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data.
        """
        if self._time_stamps is not None:
            return linear_trend_cost_mle(
                starts, ends, self._trend_data, self._time_stamps
            )
        else:
            return linear_trend_cost_index_times_mle(starts, ends, self._trend_data)

    def _evaluate_fixed_param(self, starts, ends) -> np.ndarray:
        """Evaluate the cost for fixed linear trend parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        For each interval, the squared differences between the actual data points and
        the fixed linear trend are summed.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data.
        """
        return linear_trend_cost_fixed(
            starts, ends, self._trend_data, self._time_stamps, self._trend_params
        )

    def _check_fixed_param(self, param, trend_data: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : array-like
            Fixed parameters for the linear trend: `[[slope_1, intercept_1],\
                                                    [slope_2, intercept_2],\
                                                     ...]`,
            where each pair of parameters corresponds to a column in `trend_data`.
        trend_data : np.ndarray
            Input data to evaluate LinearTrendCost on. Must be a 2D array.

        Returns
        -------
        param: np.ndarray
            Fixed parameters for the cost calculation.
        """
        # Convert to numpy array if not already
        param_array = np.asarray(param, dtype=float)

        # If shared_linear_trend is True, allow providing a single
        # [slope, intercept] pair that will be applied to all columns.
        if self.share_fixed_trend and param_array.size == 2:
            # Repeat the parameters for each column:
            param_array = np.tile(param_array.reshape(1, 2), (trend_data.shape[1], 1))
            return param_array

        # Otherwise, check that we have the right number of parameters (2 per column)
        if param_array.size != 2 * trend_data.shape[1]:
            raise ValueError(
                f"Expected {2 * trend_data.shape[1]} parameters "
                f"(2 per column), but got {param_array.size}."
            )

        if param_array.ndim == 1 and trend_data.shape[1] == 1:
            # Got a 1D array for a single column, reshape to 2D.
            param_array = param_array.reshape(-1, 2)

        if param_array.ndim != 2 or param_array.shape[1] != 2:
            raise ValueError("Fixed parameters must convertible to shape (n_cols, 2).")

        return param_array

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        For linear trend fitting, we need at least 3 points.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        return 3

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the cost function.
        """
        # For each column we need 2 parameters: slope and intercept
        return 2 * p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params = [
            {"param": None},
            # Fixed parameter shape depends on the data:
            {
                "param": np.zeros((1, 2)),
                "share_fixed_trend": True,
            },  # slope=0, intercept=0 for all columns.
        ]
        return params
