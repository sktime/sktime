# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Linear trend cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost


def _fit_linear_trend(time_steps, values):
    """OLS fit of a linear trend: returns (slope, intercept)."""
    mean_t = np.mean(time_steps)
    centered_t = time_steps - mean_t
    mean_v = np.mean(values)
    denominator = np.sum(np.square(centered_t))
    numerator = np.sum(centered_t * (values - mean_v))
    slope = numerator / denominator
    intercept = mean_v - slope * mean_t
    return slope, intercept


def _fit_indexed_linear_trend(xs):
    """OLS fit assuming evenly spaced time [0, 1, ..., n-1]."""
    n = len(xs)
    mean_t = (n - 1) / 2.0
    denominator = n * (n * n - 1) / 12.0
    mean_x = np.mean(xs)
    numerator = 0.0
    for i in range(n):
        numerator += (i - mean_t) * (xs[i] - mean_x)
    slope = numerator / denominator
    intercept = mean_x - slope * mean_t
    return slope, intercept


def _linear_trend_cost_mle(starts, ends, X, times):
    """Linear trend cost with optimal parameters and explicit time column."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment_times = times[start:end]
        for col in range(n_columns):
            segment_data = X[start:end, col]
            slope, intercept = _fit_linear_trend(segment_times, segment_data)
            costs[i, col] = np.sum(
                np.square(segment_data - (intercept + slope * segment_times))
            )
    return costs


def _linear_trend_cost_index_mle(starts, ends, X):
    """Linear trend cost with optimal parameters using [0..n-1] time steps."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            segment_data = X[start:end, col]
            slope, intercept = _fit_indexed_linear_trend(segment_data)
            trend = intercept + slope * np.arange(len(segment_data), dtype=float)
            costs[i, col] = np.sum(np.square(segment_data - trend))
    return costs


def _linear_trend_cost_fixed(starts, ends, X, time_steps, params):
    """Linear trend cost with fixed slope/intercept parameters."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    for col in range(n_columns):
        slope, intercept = params[col, :]
        for i in range(n_intervals):
            start, end = starts[i], ends[i]
            segment_data = X[start:end, col]
            segment_ts = time_steps[start:end]
            costs[i, col] = np.sum(
                np.square(segment_data - (intercept + slope * segment_ts))
            )
    return costs


class LinearTrendCost(BaseCost):
    """Linear trend cost function.

    Calculates the sum of squared errors between data points and a
    linear trend line fitted to each interval.

    Parameters
    ----------
    param : array-like, optional (default=None)
        Fixed parameters ``[[slope_1, intercept_1], ...]``,
        shape ``(n_columns, 2)``.
    time_column : int, optional (default=None)
        Column index to use as time.  If ``None``, time steps are
        assumed to be ``[0, 1, ..., n-1]`` within each segment.
    share_fixed_trend : bool, optional (default=False)
        If True, a single ``[slope, intercept]`` pair is broadcast
        to all data columns.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
    }

    def __init__(self, param=None, time_column=None, share_fixed_trend=False):
        self.time_column = time_column
        self.share_fixed_trend = share_fixed_trend
        super().__init__(param)

    def _resolve_time_and_data(self, X):
        """Extract time stamps and trend data columns from X."""
        if self.time_column is not None:
            tc = self.time_column
            time_stamps = X[:, tc].astype(float)
            time_stamps = time_stamps - time_stamps[0]
            trend_cols = [c for c in range(X.shape[1]) if c != tc]
            trend_data = X[:, trend_cols]
        else:
            time_stamps = None
            trend_data = X
        return time_stamps, trend_data

    def _evaluate_optim_param(self, X, starts, ends):
        time_stamps, trend_data = self._resolve_time_and_data(X)
        if time_stamps is not None:
            return _linear_trend_cost_mle(starts, ends, trend_data, time_stamps)
        return _linear_trend_cost_index_mle(starts, ends, trend_data)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        time_stamps, trend_data = self._resolve_time_and_data(X)
        if time_stamps is None:
            time_stamps = np.arange(X.shape[0], dtype=float)
        return _linear_trend_cost_fixed(starts, ends, trend_data, time_stamps, param)

    def _check_fixed_param(self, param, X):
        _, trend_data = self._resolve_time_and_data(X)
        param_array = np.asarray(param, dtype=float)
        n_trend_cols = trend_data.shape[1]

        if self.share_fixed_trend and param_array.size == 2:
            param_array = np.tile(param_array.reshape(1, 2), (n_trend_cols, 1))
            return param_array

        if param_array.size != 2 * n_trend_cols:
            raise ValueError(
                f"Expected {2 * n_trend_cols} parameters (2 per column), "
                f"got {param_array.size}."
            )
        if param_array.ndim == 1 and n_trend_cols == 1:
            param_array = param_array.reshape(-1, 2)
        if param_array.ndim != 2 or param_array.shape[1] != 2:
            raise ValueError(
                "Fixed parameters must be convertible to shape (n_cols, 2)."
            )
        return param_array

    @property
    def min_size(self):
        return 3

    def get_model_size(self, p):
        return 2 * p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"param": None},
            {"param": np.zeros((1, 2)), "share_fixed_trend": True},
        ]
