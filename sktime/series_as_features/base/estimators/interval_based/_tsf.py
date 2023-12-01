"""Time Series Forest (TSF) Classifier."""

__author__ = [
    "TonyBagnall",
    "kkoziara",
    "luiszugasti",
    "kanand77",
    "mloning",
    "Oleksii Kachaiev",
]
__all__ = [
    "BaseTimeSeriesForest",
    "_transform",
    "_get_intervals",
    "_fit_estimator",
]

import math
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state

from sktime.base._base import _clone_estimator
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation import check_n_jobs


class BaseTimeSeriesForest:
    """Base time series forest classifier."""

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        n_jobs=1,
        inner_series_length: Optional[int] = None,
        random_state=None,
    ):
        super().__init__(
            self._base_estimator,
            n_estimators=n_estimators,
        )

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.n_jobs = n_jobs
        self.inner_series_length = inner_series_length
        # The following set in method fit
        self.n_classes = 0
        self.series_length = 0
        self.n_intervals = 0
        self.estimators_ = []
        self.intervals_ = []
        self.classes_ = []

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    @property
    def _estimator(self):
        """Access first parameter in self, self inheriting from sklearn BaseForest.

        The attribute was renamed from base_estimator to estimator in sklearn 1.2.0.
        """
        import sklearn
        from packaging.specifiers import SpecifierSet

        sklearn_version = sklearn.__version__

        if sklearn_version in SpecifierSet(">=1.2.0"):
            return self.estimator
        else:
            return self.base_estimator

    def _fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        Xt: np.ndarray or pd.DataFrame
            Panel training data.
        y : np.ndarray
            The class labels.

        Returns
        -------
        self : object
            An fitted instance of the classifier
        """
        X = X.squeeze(1)
        n_instances, self.series_length = X.shape

        n_jobs = check_n_jobs(self.n_jobs)

        rng = check_random_state(self.random_state)

        self.n_classes = np.unique(y).shape[0]

        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.n_intervals = int(math.sqrt(self.series_length))
        if self.n_intervals == 0:
            self.n_intervals = 1
        if self.series_length < self.min_interval:
            self.min_interval = self.series_length

        self.intervals_ = [
            _get_intervals(
                self.n_intervals,
                self.min_interval,
                self.series_length,
                rng,
                self.inner_series_length,
            )
            for _ in range(self.n_estimators)
        ]

        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(_fit_estimator)(
                _clone_estimator(self._estimator, rng), X, y, self.intervals_[i]
            )
            for i in range(self.n_estimators)
        )

        self._is_fitted = True
        return self

    def _get_fitted_params(self):
        return {
            "classes": self.classes_,
            "intervals": self.intervals_,
            "estimators": self.estimators_,
        }


def _get_intervals(
    n_intervals: int,
    min_interval: int,
    series_length: int,
    rng: RandomState,
    inner_series_length: Optional[int] = None,
) -> np.ndarray:
    """Generate random intervals for given parameters.

    Parameters
    ----------
    n_intervals : int
        Number of intervals to generate.
    min_interval : int
        Minimum length of an interval.
    series_length : int
        Length of the series.
    rng : RandomState
        Random number generator.
    inner_series_length : int, optional (default=None)
        Length of the inner series, define the maximum of an interval
        and forces intervals to be contained in disjoint segments of
        length inner_series_length. If None, defaults to series_length.

    Returns
    -------
    intervals_starts_and_end_matrix : np.ndarray
        Matrix of shape (n_intervals, 2) where each row represents an
        interval and contains its start and end.
    """
    interval_max_length = (
        series_length if inner_series_length is None else inner_series_length
    )
    capped_min_interval = (
        interval_max_length if min_interval >= interval_max_length else min_interval
    )
    number_of_inner_intervals = series_length // interval_max_length
    intervals_starts_and_end_matrix = np.zeros((n_intervals, 2), dtype=int)
    for interval_index in range(n_intervals):
        inner_intervals_step = rng.randint(number_of_inner_intervals)
        current_interval_start = (
            inner_intervals_step * interval_max_length
            + rng.randint(max(1, interval_max_length - capped_min_interval))
        )
        current_interval_length = compute_interval_length(
            capped_min_interval,
            current_interval_start,
            inner_intervals_step,
            interval_max_length,
            rng,
        )
        current_interval_end = current_interval_start + current_interval_length
        intervals_starts_and_end_matrix[interval_index, :] = [
            current_interval_start,
            current_interval_end,
        ]
    return intervals_starts_and_end_matrix


def compute_interval_length(
    capped_min_interval: int,
    current_interval_start: int,
    inner_intervals_step: int,
    interval_max_length: int,
    rng: RandomState,
) -> int:
    if (
        capped_min_interval
        < interval_max_length * (inner_intervals_step + 1) - current_interval_start
    ):
        current_interval_length = max(
            capped_min_interval,
            rng.randint(
                interval_max_length * (inner_intervals_step + 1)
                - current_interval_start
                - 1,
            ),
        )
    elif (
        capped_min_interval
        == interval_max_length * (inner_intervals_step + 1) - current_interval_start
    ):
        current_interval_length = capped_min_interval
    else:
        highest_possible_interval_length = (
            interval_max_length * (inner_intervals_step + 1) - current_interval_start
        )
        raise ValueError(
            f"low({capped_min_interval}) > "
            f"high({highest_possible_interval_length}): "
            f"Decrease capped_min_interval({capped_min_interval}) "
            f"or increase interval_max_length({interval_max_length})"
        )
    return current_interval_length


def _transform(X, intervals):
    """Transform X for given intervals.

    Compute the mean, standard deviation and slope for given intervals of input data X.

    Parameters
    ----------
    Xt: np.ndarray or pd.DataFrame
        Panel data to transform.
    intervals : np.ndarray
        Intervals containing start and end values.

    Returns
    -------
    Xt: np.ndarray or pd.DataFrame
     Transformed X, containing the mean, std and slope for each interval
    """
    n_instances, _ = X.shape
    n_intervals, _ = intervals.shape
    transformed_x = np.empty(shape=(3 * n_intervals, n_instances), dtype=np.float32)
    for j in range(n_intervals):
        X_slice = X[:, intervals[j][0] : intervals[j][1]]
        means = np.mean(X_slice, axis=1)
        std_dev = np.std(X_slice, axis=1)
        slope = _slope(X_slice, axis=1)
        transformed_x[3 * j] = means
        transformed_x[3 * j + 1] = std_dev
        transformed_x[3 * j + 2] = slope

    return transformed_x.T


def _fit_estimator(estimator, X, y, intervals):
    """Fit an estimator on input data (X, y)."""
    transformed_x = _transform(X, intervals)
    return estimator.fit(transformed_x, y)
