# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Continuous piecewise linear trend change score."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def _continuous_piecewise_linear_trend_squared_contrast(
    signal,
    first_interval_inclusive_start,
    second_interval_inclusive_start,
    non_inclusive_end,
):
    """Squared contrast for continuous piecewise linear trend.

    Parameters
    ----------
    signal : np.ndarray
        1D data segment (already sliced to [start:end]).
    first_interval_inclusive_start : int
        Start of the first interval (inclusive).
    second_interval_inclusive_start : int
        Start of the second interval (inclusive).
    non_inclusive_end : int
        End of the interval (exclusive).

    Returns
    -------
    float
        Squared contrast value.
    """
    s = first_interval_inclusive_start - 1
    e = non_inclusive_end - 1
    b = second_interval_inclusive_start - 1 + 1

    l = e - s  # noqa: E741
    alpha = np.sqrt(
        6.0 / (l * (l**2 - 1) * (1 + (e - b + 1) * (b - s) + (e - b) * (b - s - 1)))
    )
    beta = np.sqrt(((e - b + 1.0) * (e - b)) / ((b - s - 1.0) * (b - s)))

    first_interval_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_interval_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)
    second_interval_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_interval_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    contrast = 0.0
    for t in range(s + 1, b + 1):
        contrast += (
            alpha * beta * (first_interval_slope * t - first_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    for t in range(b + 1, e + 1):
        contrast += (
            (-alpha / beta) * (second_interval_slope * t - second_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    return contrast**2


def _analytical_cont_piecewise_linear_trend_score(starts, splits, ends, X):
    """Evaluate using the analytical solution for evenly-spaced data.

    Parameters
    ----------
    starts : np.ndarray
        Start indices (inclusive).
    splits : np.ndarray
        Split indices.
    ends : np.ndarray
        End indices (exclusive).
    X : np.ndarray
        2D data array.

    Returns
    -------
    np.ndarray
        Scores of shape (n_intervals, n_columns).
    """
    scores = np.zeros((len(starts), X.shape[1]))
    for i in range(len(starts)):
        start, split, end = starts[i], splits[i], ends[i]
        for j in range(X.shape[1]):
            scores[i, j] = _continuous_piecewise_linear_trend_squared_contrast(
                X[start:end, j],
                first_interval_inclusive_start=start,
                second_interval_inclusive_start=split,
                non_inclusive_end=end,
            )
    return scores


def _lin_reg_cont_piecewise_linear_trend_score(starts, splits, ends, X, times):
    """Evaluate using linear regression for non-uniform time steps.

    Parameters
    ----------
    starts, splits, ends : np.ndarray
        Interval boundaries.
    X : np.ndarray
        2D data array.
    times : np.ndarray
        1D time stamps.

    Returns
    -------
    np.ndarray
        Scores of shape (n_intervals, n_columns).
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        n = end - start

        trend_data = np.zeros((n, 3))
        trend_data[:, 0] = 1.0
        trend_data[:, 1] = times[start:end]
        trend_data[(split - start) :, 2] = times[split:end] - times[split]

        split_res = np.linalg.lstsq(trend_data, X[start:end, :], rcond=None)
        if n == 3 and split_res[2] == 3:
            split_squared_residuals = np.zeros((n_columns,))
        else:
            split_squared_residuals = split_res[1] if len(split_res[1]) > 0 else None

        joint_res = np.linalg.lstsq(
            trend_data[:, np.array([0, 1])], X[start:end, :], rcond=None
        )
        joint_squared_residuals = joint_res[1] if len(joint_res[1]) > 0 else None

        if split_squared_residuals is None or joint_squared_residuals is None:
            scores[i, :] = np.nan
        else:
            scores[i, :] = joint_squared_residuals - split_squared_residuals

    return scores


class ContinuousLinearTrendScore(BaseIntervalScorer):
    """Continuous linear trend change score.

    Calculates the difference in squared error between observed data and:

    - a two-parameter linear trend across the whole interval, and
    - a three-parameter linear trend with a kink at the split point.

    Intended for use with the narrowest-over-threshold (NOT) segment selection
    method [1]_. Accessible within ``SeededBinarySegmentation`` by passing
    ``selection_method="narrowest"``.

    By default, time steps are assumed to be evenly spaced and an analytical
    solution is used. If ``time_column`` is given, linear regression is used
    instead.

    Parameters
    ----------
    time_column : int or None, default=None
        Column index for time stamps. If provided, those values are used as
        time stamps instead of assuming even spacing.

    References
    ----------
    .. [1] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019).
       Narrowest-over-threshold detection of multiple change points and
       change-point-like features.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "task": "change_score",
    }

    def __init__(self, time_column=None):
        self.time_column = time_column
        super().__init__()

    @property
    def min_size(self):
        return 2

    def get_model_size(self, p):
        return 2 * p

    def _evaluate(self, X, cuts):
        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]

        if self.time_column is not None:
            idx = self.time_column
            times = X[:, idx].astype(np.float64)
            times = times - times[0]
            data_cols = np.delete(np.arange(X.shape[1]), idx)
            data = X[:, data_cols]
            return _lin_reg_cont_piecewise_linear_trend_score(
                starts, splits, ends, data, times
            )
        return _analytical_cont_piecewise_linear_trend_score(starts, splits, ends, X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{}, {}]
