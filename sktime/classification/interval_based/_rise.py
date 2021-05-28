# -*- coding: utf-8 -*-
"""Random Interval Spectral Forest (RISE).

Implementation of Deng's Time Series Forest, with minor changes.
"""

__author__ = ["Tony Bagnall", "Yi-Xuan Xu"]
__all__ = ["RandomIntervalSpectralForest", "acf", "matrix_acf", "ps"]

import math

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
    ps_len = (interval[1] - interval[0]) / 2
    ps_x = np.empty(shape=(n_instances, int(ps_len)))
    for j in range(n_instances):
        acf_x[j] = acf(X[j, interval[0] : interval[1]], lag)
        ps_x[j] = ps(X[j, interval[0] : interval[1]])
    transformed_x = np.concatenate((acf_x, ps_x), axis=1)

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
        n_estimators=200,
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

        rng = check_random_state(self.random_state)

        self.estimators_ = []
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.intervals = np.zeros((self.n_estimators, 2), dtype=int)
        self.intervals[0][0] = 0
        self.intervals[0][1] = self.series_length
        for i in range(1, self.n_estimators):
            self.intervals[i][0] = rng.randint(self.series_length - self.min_interval)
            self.intervals[i][1] = rng.randint(
                self.intervals[i][0] + self.min_interval, self.series_length
            )

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
        """Find predictions for all cases in X. Built on top of `predict_proba.

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

        Local variables
        ---------------
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

    Return
    ----------
    y : array-like shape = [max_lag]
    """
    y = np.zeros(max_lag)
    length = len(x)
    for lag in range(1, max_lag + 1):
        # Do it ourselves to avoid zero variance warnings
        s1 = np.sum(x[:-lag])
        ss1 = np.sum(np.square(x[:-lag]))
        s2 = np.sum(x[lag:])
        ss2 = np.sum(np.square(x[lag:]))
        s1 = s1 / (length - lag)
        s2 = s2 / (length - lag)
        y[lag - 1] = np.sum((x[:-lag] - s1) * (x[lag:] - s2))
        y[lag - 1] = y[lag - 1] / (length - lag)
        v1 = ss1 / (length - lag) - s1 * s1
        v2 = ss2 / (length - lag) - s2 * s2
        if v1 <= 1e-9 and v2 <= 1e-9:  # Both zero variance,
            # so must be 100% correlated
            y[lag - 1] = 1
        elif v1 <= 1e-9 or v2 <= 1e-9:  # One zero variance
            # the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = y[lag - 1] / (math.sqrt(v1) * math.sqrt(v2))
    return np.array(y)


#        y[lag - 1] = np.corrcoef(x[lag:], x[:-lag])[0][1]
#        if np.isnan(y[lag - 1]) or np.isinf(y[lag-1]):
#            y[lag-1]=0


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

    Return
    ----------
    y : array-like shape = [num_cases,max_lag]

    """
    y = np.empty(shape=(num_cases, max_lag))
    for lag in range(1, max_lag + 1):
        # Could just do it ourselves ... TO TEST
        #            s1=np.sum(x[:-lag])/x.shape()[0]
        #            ss1=s1*s1
        #            s2=np.sum(x[lag:])
        #            ss2=s2*s2
        #
        y[lag - 1] = np.corrcoef(x[:, lag:], x[:, -lag])[0][1]
        if np.isnan(y[lag - 1]) or np.isinf(y[lag - 1]):
            y[lag - 1] = 0
    return y


def ps(x):
    """Power spectrum transform.

    Calculated using np function. It would be worth looking at ff implementation,
    see difference in speed to java.

    Parameters
    ----------
    x : array-like shape = [interval_width]

    Return
    ----------
    y : array-like shape = [len(x)/2]
    """
    fft = np.fft.fft(x)
    fft = fft.real * fft.real + fft.imag * fft.imag
    fft = fft[: int(len(x) / 2)]
    return np.array(fft)
