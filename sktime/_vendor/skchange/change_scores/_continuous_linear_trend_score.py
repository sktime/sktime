"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a best fit linear trend line within each interval.
"""

import numpy as np

from ..base import BaseIntervalScorer
from ..utils.numba import njit
from ..utils.validation.parameters import check_data_column


@njit
def lin_reg_cont_piecewise_linear_trend_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate the continuous linear trend cost.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    times : np.ndarray
        Time steps corresponding to the data points. If the data points
        are evenly spaced, instead call the analytical score function.

    Returns
    -------
    scores : np.ndarray
        A 2D array of scores. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        split_interval_trend_data = np.zeros((end - start, 3))
        split_interval_trend_data[:, 0] = 1.0  # Intercept

        # Whole interval slope:
        split_interval_trend_data[:, 1] = times[start:end]  # Time steps

        # Change in slope from the 'split' index:
        split_interval_trend_data[(split - start) :, 2] = (
            times[split:end] - times[split]
        )

        ### Compute scores for all data columns at once:
        # Calculate the slope and intercept for the whole interval:
        split_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data, X[start:end, :]
        )
        if (end - start) == 3 and split_interval_linreg_res[2] == 3:
            # If the interval is only 3 points long (the minimum),
            # and the model matrix is full rank, the residuals are zero.
            split_interval_squared_residuals = np.zeros((n_columns,))
        else:
            split_interval_squared_residuals = split_interval_linreg_res[1]

        # By only regressing onto the first two columns, we can calculate the cost
        # without allowing for a change in slope at the split point.
        joint_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data[:, np.array([0, 1])], X[start:end, :]
        )
        joint_interval_squared_residuals = joint_interval_linreg_res[1]

        # If either of the linear regression solutions failed, return NaN.
        if (len(split_interval_squared_residuals) == 0) or (
            len(joint_interval_squared_residuals) == 0
        ):
            scores[i, :] = np.nan
        else:
            scores[i, :] = (
                joint_interval_squared_residuals - split_interval_squared_residuals
            )

    return scores


@njit
def continuous_piecewise_linear_trend_squared_contrast(
    signal: np.ndarray,
    first_interval_inclusive_start: int,
    second_interval_inclusive_start: int,
    non_inclusive_end: int,
):
    # Assume 'start' is the first index of the data, perform inner product with the
    # desired segment of the data to get the cost.
    assert (
        first_interval_inclusive_start
        < second_interval_inclusive_start
        < non_inclusive_end - 1
    )

    ## Translate named parameters to the NOT-paper sytax.
    ## We are zero-indexing the data, whilst the paper is one-indexing.
    s = first_interval_inclusive_start - 1
    e = non_inclusive_end - 1

    # Add one to NOT-syntax split index to account for the difference
    # in definition of where the change in slope starts from. They
    # let the change in slop take effect after the point before the split.
    b = second_interval_inclusive_start - 1 + 1

    l = e - s
    alpha = np.sqrt(
        6.0 / (l * (l**2 - 1) * (1 + (e - b + 1) * (b - s) + (e - b) * (b - s - 1)))
    )
    beta = np.sqrt(((e - b + 1.0) * (e - b)) / ((b - s - 1.0) * (b - s)))

    first_interval_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_interval_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)

    second_interval_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_interval_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    # Accumulate the contrast value inner product:
    contrast = 0.0
    for t in range(s + 1, b + 1):
        contrast += (
            alpha * beta * (first_interval_slope * t - first_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    for t in range(b + 1, e + 1):
        contrast += (
            (-alpha / beta) * (second_interval_slope * t - second_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    return np.square(contrast)


@njit
def analytical_cont_piecewise_linear_trend_score(
    starts: np.ndarray, splits: np.ndarray, ends: np.ndarray, X: np.ndarray
):
    """Evaluate the continuous piecewise linear trend cost.

    Using the analytical solution, this function evaluates the cost for
    `X[start:end]` for each each `[start, split, end]` triplett in `cuts`.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    scores : np.ndarray
        A 2D array of scores. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    scores = np.zeros((len(starts), X.shape[1]))
    for i in range(len(starts)):
        start, split, end = starts[i], splits[i], ends[i]
        for j in range(X.shape[1]):
            scores[i, j] = continuous_piecewise_linear_trend_squared_contrast(
                X[start:end, j],
                first_interval_inclusive_start=start,
                second_interval_inclusive_start=split,
                non_inclusive_end=end,
            )

    return scores


class ContinuousLinearTrendScore(BaseIntervalScorer):
    """
    Continuous linear trend change score.

    Calculates the difference in squared error between observed data and:
    - a two-parameter linear trend across the whole interval, and
    - a three-parameter linear trend with a kink at the split point.

    Intended for use with the NOT segment selection method as developed by
    Baranowski et al. [1]_. Accessible within the `SeededBinarySegmentation`
    change detector by passing `selection_method="narrowest"`.

    By default, time steps are assumed to be evenly spaced.
    In this case, an analytical solution is used to calculate the score for each column
    in the data, as described in [1]_.

    If a time column is provided, its values are used as time stamps for calculating the
    linear trends. In this case, two linear regression problems are solved for each
    interval: One with a kink at the split point and one without.

    Parameters
    ----------
    time_column : str, optional
        Name of the time column in the data. The columns must be convertible to a float.
        If provided, its values are used as time stamps for calculating the piecewise
        linear trends.
        If not provided, the time steps are assumed to be evenly spaced.

    References
    ----------
    .. [1] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold
       detection of multiple change points and change-point-like features. Journal of
       the Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "task": "change_score",
    }

    def __init__(
        self,
        time_column: str | None = None,
    ):
        super().__init__()
        self.time_column = time_column
        self.time_column_idx = None
        self._time_stamps = None

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
            # Need time column as float data for numba compatibility:
            self._time_stamps = X[:, self.time_column_idx].astype(np.float64)
            # Start at time zero for first data point:
            self._time_stamps -= self._time_stamps[0]
        else:
            # No provided time column or fixed parameters, so we assume
            # the time steps are [0, 1, 2, ..., n-1] for each segment.
            self._time_stamps = None

        if self.time_column_idx is not None:
            piecewise_linear_trend_columns = np.delete(
                np.arange(X.shape[1]), self.time_column_idx
            )
            self._piecewise_linear_trend_data = X[:, piecewise_linear_trend_columns]
        else:
            self._piecewise_linear_trend_data = X
        return self

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the continuous piecewise linear trend scores.

        Evaluates the score on `X[start:end]` for each each `[start, split, end]`
        triplett in cuts.  On each interval, the difference in summed squared
        residuals between the best fit linear trend accross the whole interval
        and the best fit linear trend with a kink at the split point
        is calculated. The score is calculated for each column in the data.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the ``start``, the second is the ``split``, and the
            third is the ``end`` of the interval to evaluate the score on.

        Returns
        -------
        scores : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data.
        """
        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]
        if self.time_column is None:
            scores = analytical_cont_piecewise_linear_trend_score(
                # scores = linear_trend_score(
                starts=starts,
                splits=splits,
                ends=ends,
                X=self._piecewise_linear_trend_data,
            )
        else:
            scores = lin_reg_cont_piecewise_linear_trend_score(
                starts=starts,
                splits=splits,
                ends=ends,
                X=self._piecewise_linear_trend_data,
                times=self._time_stamps,
            )
        return scores

    @property
    def min_size(self) -> int:
        """Minimum number of samples required on each side of a split point to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        To solve for a linear trend, we need at least 2 points on each side
        of the split point. Due to the construction of the score where we
        enfore continuity continuity at the split point, we need at least 1 point
        on the left side of the split point and 1 points on the right side of the
        split point. The minimum size of the total interval is therefore 3.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        # Need at least a difference of 1 between the start and split
        # indices to evaluate the cost, and at least a difference of 2
        # between the split and end indices to evaluate the cost.
        return 2

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
        # In each interval we need 2 parameters: slope and intercept.
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
        params = [{}, {}]
        return params
