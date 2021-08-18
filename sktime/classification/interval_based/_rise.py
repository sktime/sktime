# -*- coding: utf-8 -*-
"""Random Interval Spectral Forest (RISE).

Implementation of Deng's Time Series Forest, with minor changes.
"""

__author__ = ["Tony Bagnall", "Yi-Xuan Xu"]
__all__ = ["RandomIntervalSpectralForest", "acf", "matrix_acf", "ps"]


from numba import int64, prange, jit
import numpy as np
from joblib import Parallel
from joblib import delayed
from sklearn.base import clone
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state

from sktime.classification.base import BaseClassifier
from sklearn.ensemble._base import _partition_estimators
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


def _transform(X, interval, lag):
    """Compute the ACF and PS for given intervals of input data X."""
    n_instances, _ = X.shape
    acf_x = np.empty(shape=(n_instances, lag))
    ps_len = _round_to_nearest_power_of_two(interval[1] - interval[0])
    ps_x = np.empty(shape=(n_instances, ps_len))
    for j in range(n_instances):
        interval_x = X[j, interval[0] : interval[1]]
        acf_x[j] = acf(interval_x, lag)
        ps_x[j] = ps(interval_x, n=ps_len * 2)
    # interval_x = X[:, interval[0]: interval[1]]
    # acf_x = matrix_acf(interval_x, n_instances, lag)
    # ps_x = _ps(interval_x, n=ps_len*2)
    transformed_x = np.concatenate((ps_x, acf_x), axis=1)

    return transformed_x


def _parallel_build_trees(X, y, tree, interval, lag, acf_min_values):
    """Private function used to fit a single tree in parallel."""
    temp_lag = lag
    if temp_lag > interval[1] - interval[0] - acf_min_values:
        temp_lag = interval[1] - interval[0] - acf_min_values
    if temp_lag < 0:
        temp_lag = 1
    temp_lag = int(temp_lag)
    transformed_x = _transform(X, interval, temp_lag)
    tree.fit(transformed_x, y)

    return temp_lag, tree


def _predict_proba_for_estimator(X, estimator, interval, lag):
    """Private function used to predict class probabilities in parallel."""
    transformed_x = _transform(X, interval, lag)
    return estimator.predict_proba(transformed_x)


def _make_estimator(base_estimator, random_state=None):
    """Make and configure a copy of the `base_estimator` attribute.

    Warning: This method should be used to properly instantiate new
    sub-estimators.
    """
    estimator = clone(base_estimator)
    estimator.set_params(**{"random_state": random_state})
    return estimator


def _select_interval(min_interval, max_interval, series_length, rng, method=3):
    """Private function used to select an interval for a single tree."""
    interval = np.empty(2, dtype=int)
    if method == 0:
        interval[0] = rng.randint(series_length - min_interval)
        interval[1] = rng.randint(interval[0] + min_interval, series_length)
    else:
        if rng.randint(2):
            interval[0] = rng.randint(series_length - min_interval)
            interval_range = min(series_length - interval[0], max_interval)
            length = rng.randint(min_interval, interval_range)
            interval[1] = interval[0] + length
        else:
            interval[1] = rng.randint(min_interval, series_length)
            interval_range = min(interval[1], max_interval)
            length = (
                3
                if interval_range == min_interval
                else rng.randint(min_interval, interval_range)
            )
            interval[0] = interval[1] - length
    return interval


def _produce_intervals(
    n_estimators, min_interval, max_interval, series_length, rng, method=3
):
    """Private function used to produce intervals for all trees."""
    intervals = np.empty((n_estimators, 2), dtype=int)
    if method == 0:
        # just keep it as a backup, untested
        intervals[:, 0] = rng.randint(series_length - min_interval, size=n_estimators)
        intervals[:, 1] = rng.randint(
            intervals[:, 0] + min_interval, series_length, size=n_estimators
        )
    elif method == 3:
        bools = rng.randint(2, size=n_estimators)
        true = np.where(bools == 1)[0]
        intervals[true, 0] = rng.randint(series_length - min_interval, size=true.size)
        interval_range = np.fmin(series_length - intervals[true, 0], max_interval)
        length = rng.randint(min_interval, interval_range)
        intervals[true, 1] = intervals[true, 0] + length

        false = np.where(bools == 0)[0]
        intervals[false, 1] = rng.randint(min_interval, series_length, size=false.size)
        interval_range = np.fmin(intervals[false, 1], max_interval)
        min_mask = interval_range == min_interval
        length = np.empty(false.size)
        length[min_mask] = 3
        length[~min_mask] = rng.randint(min_interval, interval_range[~min_mask])
        intervals[false, 0] = intervals[false, 1] - length

    return intervals


class RandomIntervalSpectralForest(ForestClassifier, BaseClassifier):
    """Random Interval Spectral Forest (RISE).

    Input: n series length m
    for each tree
        sample a random intervals
        take the ACF and PS over this interval, and concatenate features
        build tree on new features
    ensemble the trees through averaging probabilities.

    Parameters
    ----------
    n_estimators : int, optional (default=200)
        The number of trees in the forest.
    min_interval : int, optional (default=16)
        The minimum width of an interval.
    acf_lag : int, optional (default=100)
        The maximum number of autocorrelation terms to use.
    acf_min_values : int, optional (default=4)
        Never use fewer than this number of terms to find a correlation.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    n_classes : int
        The number of classes, extracted from the data.
    n_estimators : array of shape = [n_estimators] of DecisionTree classifiers
    intervals : array of shape = [n_estimators][2]
        Stores indexes of start and end points for all classifiers.

    Notes
    -----
    ..[1] Jason Lines, Sarah Taylor and Anthony Bagnall, "Time Series Classification
    with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles",
      ACM Transactions on Knowledge and Data Engineering, 12(5): 2018
    https://dl.acm.org/doi/10.1145/3182382
    Java implementation
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/frequency_based/RISE.java
    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    # TO DO: handle missing values, unequal length series and multivariate
    # problems

    def __init__(
        self,
        n_estimators=500,
        max_interval=0,
        min_interval=16,
        acf_lag=100,
        acf_min_values=4,
        n_jobs=None,
        random_state=None,
    ):
        super(RandomIntervalSpectralForest, self).__init__(
            base_estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=n_estimators,
        )
        self.n_estimators = n_estimators
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values
        self.n_jobs = n_jobs
        self.random_state = random_state

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    @property
    def feature_importances_(self):
        """Feature importance not supported for the RISE classifier."""
        raise NotImplementedError(
            "The impurity-based feature importances of "
            "RandomIntervalSpectralForest is currently not supported."
        )

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

        using random intervals and spectral features.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples. If a Pandas data frame is passed it
            must have a single column (i.e., univariate classification).
            RISE has no bespoke method for multivariate classification as yet.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        n_instances, self.series_length = X.shape
        self.min_interval_, self.max_interval_ = self.min_interval, self.max_interval
        if self.max_interval_ not in range(1, self.series_length):
            self.max_interval_ = self.series_length
        if self.min_interval_ not in range(1, self.series_length + 1):
            self.min_interval_ = self.series_length // 2

        rng = check_random_state(self.random_state)

        self.estimators_ = []
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        # self.intervals = _produce_intervals(
        #     self.n_estimators,
        #     self.min_interval,
        #     self.max_interval,
        #     self.series_length,
        #     rng
        # )
        self.intervals = np.empty((self.n_estimators, 2), dtype=int)
        self.intervals[:] = [
            _select_interval(
                self.min_interval_, self.max_interval_, self.series_length, rng
            )
            for _ in range(self.n_estimators)
        ]

        # Check lag against global properties
        self.acf_lag_ = self.acf_lag
        if self.acf_lag > self.series_length - self.acf_min_values:
            self.acf_lag_ = self.series_length - self.acf_min_values
        if self.acf_lag < 0:
            self.acf_lag_ = 1
        self.lags = np.zeros(self.n_estimators, dtype=int)

        trees = [
            _make_estimator(
                self.base_estimator, random_state=rng.randint(np.iinfo(np.int32).max)
            )
            for _ in range(self.n_estimators)
        ]

        # Parallel loop
        worker_rets = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_build_trees)(
                X,
                y,
                tree,
                self.intervals[i],
                self.acf_lag_,
                self.acf_min_values,
            )
            for i, tree in enumerate(trees)
        )

        # Collect lags and newly grown trees
        for i, (lag, tree) in enumerate(worker_rets):
            self.lags[i] = lag
            self.estimators_.append(tree)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Find predictions for all cases in X.

        Built on top of `predict_proba`.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The input samples. If a Pandas data frame is passed it must have a
            single column (i.e., univariate classification). RISE has no
            bespoke method for multivariate classification as yet.

        Returns
        -------
        y : array of shape = [n_instances]
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return np.asarray([self.classes_[np.argmax(prob)] for prob in proba])

    def predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The input samples. If a Pandas data frame is passed it must have a
            single column (i.e., univariate classification). RISE has no
            bespoke method for multivariate classification as yet.

        Attributes
        ----------
        n_instances : int
            Number of cases to classify.
        n_columns : int
            Number of attributes in X, must match `series_length` determined
            in `fit`.

        Returns
        -------
        output : array of shape = [n_instances, n_classes]
            The class probabilities of all cases.
        """
        # Check data
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        n_instances, n_columns = X.shape
        if n_columns != self.series_length:
            raise TypeError(
                "ERROR number of attributes in the train does not match "
                "that in the test data."
            )

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        all_proba = Parallel(n_jobs=n_jobs)(
            delayed(_predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                self.intervals[i],
                self.lags[i],
            )
            for i in range(self.n_estimators)
        )

        return np.sum(all_proba, axis=0) / self.n_estimators


@jit(parallel=True, cache=True, nopython=True)
def acf(x, max_lag):
    """Autocorrelation function transform.

    currently calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [max_lag]
    """
    y = np.empty(max_lag)
    length = len(x)
    for lag in prange(1, max_lag + 1):
        # Do it ourselves to avoid zero variance warnings
        lag_length = length - lag
        x1, x2 = x[:-lag], x[lag:]
        s1 = np.sum(x1)
        s2 = np.sum(x2)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        ss1 = np.sum(x1 * x1)
        ss2 = np.sum(x2 * x2)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        if v1_is_zero and v2_is_zero:  # Both zero variance,
            # so must be 100% correlated
            y[lag - 1] = 1
        elif v1_is_zero or v2_is_zero:  # One zero variance
            # the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)
        # _x = np.vstack((x[:-lag], x[lag:]))
        # s = np.sum(_x, axis=1)
        # ss = np.sum(_x * _x, axis=1)
        # v = ss - s * s / l
        # zero_variances = v <= 1e-9
        # i = lag - 1
        # if np.all(zero_variances):  # Both zero variance,
        #     # so must be 100% correlated
        #     y[i] = 1
        # elif np.any(zero_variances):  # One zero variance
        #     # the other not
        #     y[i] = 0
        # else:
        #     m = _x - s.reshape(2, 1) / l
        #     y[i] = (m[0] @ m[1]) / np.sqrt(np.prod(v))

    return y


#        y[lag - 1] = np.corrcoef(x[lag:], x[:-lag])[0][1]
#        if np.isnan(y[lag - 1]) or np.isinf(y[lag-1]):
#            y[lag-1]=0

# @jit(parallel=True, cache=True, nopython=True)
# def _acf(x, max_lag):
#     y = np.empty(max_lag)
#     length = len(x)
#     n = length - np.arange(1, max_lag + 1)
#     # _x = np.array([x[:-1], x[:0:-1]])
#     # from_end_to_lag = slice(-1, -max_lag - 1, -1)
#     # cs = np.cumsum(_x, axis=1)[:, from_end_to_lag]
#     # cm = cs / n
#     # css = np.cumsum(_x * _x, axis=1)[:, from_end_to_lag]
#     # cv = css - cs
#
#     a, b = x[:-1], x[:0:-1]
#     from_end_to_lag = slice(-1, -max_lag - 1, -1)
#     cs1 = np.cumsum(a)[from_end_to_lag] / n
#     cs2 = np.cumsum(b)[from_end_to_lag] / n
#     css1 = np.cumsum(a * a)[from_end_to_lag] / n
#     css2 = np.cumsum(b * b)[from_end_to_lag] / n
#     cv1 = css1 - cs1 * cs1
#     cv2 = css2 - cs2 * cs2
#     covar = cv1 * cv2
#
#     for lag in prange(1, max_lag + 1):
#         idx = lag - 1
#         m1, m2, l = cs1[idx], cs2[idx], n[idx]
#         y[idx] = np.sum((x[:-lag] - m1) * (x[lag:] - m2)) / l
#     # both_zero = (cv1 <= 1e-9) & (cv2 <= 1e-9)
#     # one_zero = (cv1 <= 1e-9) ^ (cv2 <= 1e-9)
#     cv1_is_zero, cv2_is_zero = cv1 <= 1e-9, cv2 <= 1e-9
#     non_zero = ~cv1_is_zero & ~cv2_is_zero
#     y[cv1_is_zero & cv2_is_zero] = 1  # Both zero variance,
#     # so must be 100% correlated
#     y[cv1_is_zero ^ cv2_is_zero] = 0  # One zero variance
#     # the other not
#     y[non_zero] /= np.sqrt(covar[non_zero])
#
#     return y

# @jit(parallel=True, cache=True, nopython=True)
def matrix_acf(x, num_cases, max_lag):
    """Autocorrelation function transform.

    Calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [num_cases, interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [num_cases,max_lag]

    """
    y = np.empty(shape=(num_cases, max_lag))
    length = x.shape[1]
    for lag in prange(1, max_lag + 1):
        # Could just do it ourselves ... TO TEST
        #            s1=np.sum(x[:-lag])/x.shape()[0]
        #            ss1=s1*s1
        #            s2=np.sum(x[lag:])
        #            ss2=s2*s2
        #
        lag_length = length - lag
        x1, x2 = x[:, :-lag], x[:, lag:]
        s1 = np.sum(x1, axis=1)
        s2 = np.sum(x2, axis=1)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        s12 = np.sum(x1 * x2, axis=1)
        ss1 = np.sum(x1 * x1, axis=1)
        ss2 = np.sum(x2 * x2, axis=1)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v12 = s12 - s1 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        non_zero = ~v1_is_zero & ~v2_is_zero
        # y[:, lag - 1] = np.sum((x1 - m1[:, None]) *
        # (x2 - m2[:, None]), axis=1)
        y[v1_is_zero & v2_is_zero, lag - 1] = 1  # Both zero variance,
        # so must be 100% correlated
        y[v1_is_zero ^ v2_is_zero, lag - 1] = 0  # One zero variance
        # the other not
        var = (v1 * v2)[non_zero]
        y[non_zero, lag - 1] = v12[non_zero] / np.sqrt(var)
    #     # y[lag - 1] = np.corrcoef(x[:, lag:], x[:, -lag])[0][1]
    #     # if np.isnan(y[lag - 1]) or np.isinf(y[lag - 1]):
    #     #     y[lag - 1] = 0
    return y


def ps(x, sign=1, n=None, pad="mean"):
    """Power spectrum transformer.

    Power spectrum transform, currently calculated using np function.
    It would be worth looking at ff implementation, see difference in speed
    to java.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    sign : {-1, 1}, default = 1
    n : int, default=None
    pad : str or function, default='mean'
        controls the mode of the pad function
        see numpy.pad for more details
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns
    -------
    y : array-like shape = [len(x)/2]
    """
    x_len = x.shape[-1]
    x_is_1d = x.ndim == 1
    # pad or slice series if length is not of power of 2 or n is specified
    if x_len & (x_len - 1) != 0 or n:
        # round n (or the length of x) to next power of 2
        # when n is not specified
        if not n:
            n = _round_to_nearest_power_of_two(x_len)
        # pad series up to n when n is larger otherwise slice series up to n
        if n > x_len:
            pad_length = (0, n - x_len) if x_is_1d else ((0, 0), (0, n - x_len))
            x_in_power_2 = np.pad(x, pad_length, mode=pad)
        else:
            x_in_power_2 = x[:n] if x_is_1d else x[:, :n]
    else:
        x_in_power_2 = x
    # use sign to determine inverse or normal fft
    # using the norm in numpy fft function
    # backward = normal fft, forward = inverse fft (divide by n after fft)
    # note: use the following code when upgrade numpy to 1.20
    # norm = "backward" if sign > 0 else "forward"
    # fft = np.fft.rfft(x_in_power_2, norm=norm)
    if sign < 0:
        x_in_power_2 /= n
    fft = np.fft.rfft(x_in_power_2)
    fft = fft[:-1] if x_is_1d else fft[:, :-1]
    return np.abs(fft)


@jit("int64(int64)", cache=True, nopython=True)
def _round_to_nearest_power_of_two(n):
    return int64(1 << round(np.log2(n)))
