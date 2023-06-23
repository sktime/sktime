"""Random Interval Spectral Ensemble (RISE)."""

__author__ = ["TonyBagnall"]
__all__ = [
    "RandomIntervalSpectralEnsemble",
    "ps",
]

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble._base import _partition_estimators
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_random_state

from sktime.classification.base import BaseClassifier


def _transform(X, interval, lag):
    """Compute the ACF and PS for given intervals of input data X."""
    from sktime.classification.interval_based._rise_numba import (
        _round_to_nearest_power_of_two,
        acf,
    )

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


class RandomIntervalSpectralEnsemble(BaseClassifier):
    """Random Interval Spectral Ensemble (RISE).

    Input: n series length m
    For each tree
        - sample a random intervals
        - take the ACF and PS over this interval, and concatenate features
        - build tree on new features
    Ensemble the trees through averaging probabilities.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of trees in the forest.
    min_interval : int, default=16
        The minimum width of an interval.
    acf_lag : int, default=100
        The maximum number of autocorrelation terms to use.
    acf_min_values : int, default=4
        Never use fewer than this number of terms to find a correlation.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.
    intervals_ : array of shape = [n_estimators][2]
        Stores indexes of start and end points for all classifiers.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/interval_based/RISE.java>`_.

    References
    ----------
    .. [1] Jason Lines, Sarah Taylor and Anthony Bagnall, "Time Series Classification
       with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based
       Ensembles", ACM Transactions on Knowledge and Data Engineering, 12(5): 2018
    """

    _tags = {
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "interval",
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        n_estimators=500,
        max_interval=0,
        min_interval=16,
        acf_lag=100,
        acf_min_values=4,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.intervals_ = []

        self.base_estimator = DecisionTreeClassifier(random_state=random_state)

        super().__init__()

    @property
    def feature_importances_(self):
        """Feature importance not supported for the RISE classifier."""
        raise NotImplementedError(
            "The impurity-based feature importances of "
            "RandomIntervalSpectralForest is currently not supported."
        )

    def _fit(self, X, y):
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
        X = X.squeeze(1)
        n_instances, self.series_length = X.shape
        self.min_interval_, self.max_interval_ = self.min_interval, self.max_interval
        if self.max_interval_ not in range(1, self.series_length):
            self.max_interval_ = self.series_length
        if self.min_interval_ not in range(1, self.series_length + 1):
            self.min_interval_ = self.series_length // 2

        rng = check_random_state(self.random_state)

        self.estimators_ = []
        # self.intervals = _produce_intervals(
        #     self.n_estimators,
        #     self.min_interval,
        #     self.max_interval,
        #     self.series_length,
        #     rng
        # )
        self.intervals_ = np.empty((self.n_estimators, 2), dtype=int)
        self.intervals_[:] = [
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
        worker_rets = Parallel(n_jobs=self._threads_to_use)(
            delayed(_parallel_build_trees)(
                X,
                y,
                tree,
                self.intervals_[i],
                self.acf_lag_,
                self.acf_min_values,
            )
            for i, tree in enumerate(trees)
        )

        # Collect lags and newly grown trees
        for i, (lag, tree) in enumerate(worker_rets):
            self.lags[i] = lag
            self.estimators_.append(tree)

        return self

    def _predict(self, X) -> np.ndarray:
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
        proba = self._predict_proba(X)
        return np.asarray([self.classes_[np.argmax(prob)] for prob in proba])

    def _predict_proba(self, X) -> np.ndarray:
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
        X = X.squeeze(1)

        n_instances, n_columns = X.shape
        if n_columns != self.series_length:
            raise ValueError(
                "ERROR number of attributes in the train does not match "
                "that in the test data."
            )

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self._threads_to_use)

        # Parallel loop
        all_proba = Parallel(n_jobs=n_jobs)(
            delayed(_predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                self.intervals_[i],
                self.lags[i],
            )
            for i in range(self.n_estimators)
        )

        return np.sum(all_proba, axis=0) / self.n_estimators

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10}
        else:
            return {
                "n_estimators": 2,
                "acf_lag": 10,
                "min_interval": 5,
            }


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
    from sktime.classification.interval_based._rise_numba import (
        _round_to_nearest_power_of_two,
    )

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
