# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Linear trend cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection.costs._base import BaseCost


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
    centered_t = np.arange(n, dtype=float) - mean_t
    numerator = np.sum(centered_t * (xs - mean_x))
    slope = numerator / denominator
    intercept = mean_x - slope * mean_t
    return slope, intercept


# maximal number of elements of the window arrays processed at once,
# to bound the memory used when many long intervals have the same length
_MAX_BLOCK_ELEMENTS = 2**22


def _group_intervals_by_length(starts, ends):
    """Group intervals of equal length.

    Parameters
    ----------
    starts, ends : np.ndarray of int, shape (n_intervals,)

    Returns
    -------
    groups : list of np.ndarray of int
        indices of the intervals, one array per distinct interval length;
        all intervals in a group have the same length ``ends - starts``
    """
    lengths = ends - starts
    order = np.argsort(lengths, kind="stable")
    boundaries = np.flatnonzero(np.diff(lengths[order])) + 1
    return np.split(order, boundaries)


def _iter_windows(groups, starts, ends, X, time_steps=None):
    """Iterate over groups of intervals of equal length, as window arrays.

    Parameters
    ----------
    groups : list of np.ndarray of int
        as returned by ``_group_intervals_by_length``
    starts, ends : np.ndarray of int, shape (n_intervals,)
    X : np.ndarray of shape (n_samples, n_columns)
    time_steps : np.ndarray of shape (n_samples,), optional

    Yields
    ------
    idx : np.ndarray of int
        indices of the intervals in the block
    windows : np.ndarray of shape (len(idx), length, n_columns)
        ``windows[j]`` is ``X[starts[idx[j]]:ends[idx[j]]]``
    time_windows : np.ndarray of shape (len(idx), length), or None
        ``time_windows[j]`` is ``time_steps[starts[idx[j]]:ends[idx[j]]]``,
        only yielded if ``time_steps`` is given
    """
    n_columns = X.shape[1]
    for group in groups:
        length = ends[group[0]] - starts[group[0]]
        offsets = np.arange(length)
        block_size = max(1, _MAX_BLOCK_ELEMENTS // (n_columns * length))
        for block_start in range(0, len(group), block_size):
            idx = group[block_start : block_start + block_size]
            positions = starts[idx, None] + offsets
            windows = X[positions]
            if time_steps is None:
                yield idx, windows, None
            else:
                yield idx, windows, time_steps[positions]


def _use_grouped_evaluation(groups, n_intervals):
    """Whether evaluating intervals grouped by length pays off.

    Stacking the intervals of equal length into one array has a per-group
    overhead, so it is only used if the intervals share lengths, i.e., if
    there are at least four intervals per distinct length on average.
    """
    return 4 * len(groups) <= n_intervals


def _linear_trend_cost_mle(starts, ends, X, times):
    """Linear trend cost with optimal parameters and explicit time column."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    groups = _group_intervals_by_length(starts, ends)
    if not _use_grouped_evaluation(groups, n_intervals):
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

    for idx, windows, time_windows in _iter_windows(groups, starts, ends, X, times):
        # windows: (n_group, length, n_columns), time_windows: (n_group, length)
        centered_t = time_windows - np.mean(time_windows, axis=1, keepdims=True)
        centered_x = windows - np.mean(windows, axis=1, keepdims=True)
        denominator = np.sum(np.square(centered_t), axis=1)
        numerator = (centered_t[:, None, :] @ centered_x)[:, 0, :]
        slope = numerator / denominator[:, None]
        residuals = centered_x - slope[:, None, :] * centered_t[:, :, None]
        costs[idx] = np.sum(np.square(residuals), axis=1)
    return costs


def _linear_trend_cost_index_mle(starts, ends, X):
    """Linear trend cost with optimal parameters using [0..n-1] time steps."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    groups = _group_intervals_by_length(starts, ends)
    if not _use_grouped_evaluation(groups, n_intervals):
        for i in range(n_intervals):
            start, end = starts[i], ends[i]
            time_steps = np.arange(end - start, dtype=float)
            for col in range(n_columns):
                segment_data = X[start:end, col]
                slope, intercept = _fit_indexed_linear_trend(segment_data)
                costs[i, col] = np.sum(
                    np.square(segment_data - (intercept + slope * time_steps))
                )
        return costs

    for idx, windows, _ in _iter_windows(groups, starts, ends, X):
        # windows: (n_group, length, n_columns)
        n = windows.shape[1]
        centered_t = np.arange(n, dtype=float) - (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        centered_x = windows - np.mean(windows, axis=1, keepdims=True)
        slope = (centered_t @ centered_x) / denominator
        residuals = centered_x - slope[:, None, :] * centered_t[None, :, None]
        costs[idx] = np.sum(np.square(residuals), axis=1)
    return costs


def _linear_trend_cost_fixed(starts, ends, X, time_steps, params):
    """Linear trend cost with fixed slope/intercept parameters."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    slope = params[:, 0]
    intercept = params[:, 1]

    groups = _group_intervals_by_length(starts, ends)
    if not _use_grouped_evaluation(groups, n_intervals):
        for col in range(n_columns):
            for i in range(n_intervals):
                start, end = starts[i], ends[i]
                segment_data = X[start:end, col]
                segment_ts = time_steps[start:end]
                costs[i, col] = np.sum(
                    np.square(segment_data - (intercept[col] + slope[col] * segment_ts))
                )
        return costs

    for idx, windows, time_windows in _iter_windows(
        groups, starts, ends, X, time_steps
    ):
        # windows: (n_group, length, n_columns), time_windows: (n_group, length)
        trend = (
            intercept[None, None, :] + slope[None, None, :] * time_windows[:, :, None]
        )
        costs[idx] = np.sum(np.square(windows - trend), axis=1)
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

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.detection.costs import LinearTrendCost
    >>> X = np.array([0., 1., 2., 3., 10., 9., 8., 7.]).reshape(-1, 1)
    >>> cost = LinearTrendCost()
    >>> cost.evaluate(X, [[0, 4], [4, 8], [0, 8]]).round(2)
    array([[ 0.  ],
           [ 0.  ],
           [33.33]])

    Each of the first two intervals lies exactly on a straight line, so the
    residuals, and hence the cost, are zero. The third interval contains the
    change in slope and cannot be fitted by a single line.

    Passing ``param`` evaluates the cost at a fixed ``[slope, intercept]``
    instead of fitting the trend to the interval:

    >>> cost = LinearTrendCost(param=np.array([[1.0, 0.0]]),
    ...                        share_fixed_trend=True)
    >>> cost.evaluate(X, [[0, 4], [4, 8], [0, 8]]).round(2)
    array([[ 0.],
           [56.],
           [56.]])
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
