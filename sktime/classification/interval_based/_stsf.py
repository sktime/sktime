# -*- coding: utf-8 -*-
""" Supervised Time Series Forest Classifier (STSF).
"""

__author__ = ["Matthew Middlehurst"]
__all__ = ["SupervisedTimeSeriesForest"]

import math

import numpy as np
from joblib import Parallel
from joblib import delayed
from scipy import stats, signal
from sklearn.base import clone
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class SupervisedTimeSeriesForest(ForestClassifier, BaseClassifier):
    """Time series forest classifier.

    A time series forest is an ensemble of decision trees built on random intervals.
     Overview: Input n series length m
     for each tree
         sample sqrt(m) intervals
         find mean, sd and slope for each interval, concatenate to form new
         data set
         build decision tree on new data set
     ensemble the trees with averaged probability estimates

     This implementation deviates from the original in minor ways. It samples
     intervals with replacement and does not use the splitting criteria tiny
     refinement described in [1]. This is an intentionally stripped down, non
     configurable version for use as a hive-cote component. For a configurable
     tree based ensemble, see sktime.classifiers.ensemble.TimeSeriesForestClassifier

     TO DO: handle missing values, unequal length series and multivariate
     problems

     Parameters
     ----------
     n_estimators    : int, ensemble size, optional (default = 200)
     min_interval    : int, minimum width of an interval, optional (default
     to 3)
     n_jobs          : int, optional (default=1)
         The number of jobs to run in parallel for both `fit` and `predict`.
         ``-1`` means using all processors.
     random_state    : int, seed for random, optional (default = none)

     Attributes
     ----------
     n_classes    : int, extracted from the data
     num_atts     : int, extracted from the data
     n_intervals  : int, sqrt(num_atts)
     classifiers  : array of shape = [n_estimators] of DecisionTree
     classifiers
     intervals    : array of shape = [n_estimators][n_intervals][2] stores
     indexes of all start and end points for all classifiers
     dim_to_use   : int, the column of the panda passed to use (can be
     passed a multidimensional problem, but will only use one)

     References
     ----------
     .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
     classification and feature extraction",Information Sciences, 239, 2013
     Java implementation
     https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/TSF.java
     Arxiv version of the paper: https://arxiv.org/abs/1302.2277
    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        n_estimators=200,
        n_jobs=1,
        random_state=None,
    ):
        super(SupervisedTimeSeriesForest, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_estimators,
        )

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_classes = 0
        self.series_length = 0
        self.n_intervals = 0
        self.estimators_ = []
        self.intervals_ = []
        self.classes_ = []

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y) using random
        intervals and summary features
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. RISE has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(
            X,
            y,
            enforce_univariate=True,
            coerce_to_numpy=True,
        )
        X = X.squeeze(1)
        n_instances, self.series_length = X.shape

        rng = check_random_state(self.random_state)

        cls, class_counts = np.unique(y, return_counts=True)
        self.n_classes = class_counts.shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)

        balance_cases = np.narray([])
        average = n_instances/self.n_classes
        for i, c in enumerate(cls):
            if class_counts[i] < average:
                cls_idx = np.where(y == c)[0]
                balance_cases = np.concatenate(
                    rng.choice(cls_idx, size=class_counts[i]-average),
                    balance_cases,
                )

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(
                X,
                X_p,
                X_d,
                y,
                np.concatenate(
                    rng.choice(n_instances, size=n_instances),
                    balance_cases,
                ),
                i,
            )
            for i in range(self.n_estimators)
        )

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        proba = self.predict_proba(X)
        return np.asarray([self.classes_[np.argmax(prob)] for prob in proba])

    def predict_proba(self, X):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : nd.array of shape = (n_instances, n_classes)
            Predicted probabilities
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        _, series_length = X.shape
        if series_length != self.series_length:
            raise TypeError(
                " ERROR number of attributes in the train does not match "
                "that in the test data"
            )

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)

        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_proba_for_estimator)(
                X, X_p, X_d, i
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        return output

    @staticmethod
    def _transform(X, intervals):
        """
        Compute the mean, median, standard deviation, slope, iqr, min and max
        for given intervals of input data X.
        """
        n_instances, _ = X.shape
        n_intervals, _ = intervals.shape
        transformed_x = np.empty(shape=(7 * n_intervals, n_instances), dtype=np.float32)
        for j in range(n_intervals):
            X_slice = X[:, intervals[j][0] : intervals[j][1]]
            # mean
            transformed_x[7 * j] = np.mean(X_slice, axis=1)
            # median
            transformed_x[7 * j + 1] = np.median(X_slice, axis=1)
            # standard deviation
            transformed_x[7 * j + 2] = np.std(X_slice, axis=1)
            # slope
            transformed_x[7 * j + 3] = _slope(X_slice, axis=1)
            # interquartile range
            transformed_x[7 * j + 4] = stats.iqr(X_slice, axis=1)
            # min
            transformed_x[7 * j + 5] = np.min(X_slice, axis=1)
            # max
            transformed_x[7 * j + 6] = np.max(X_slice, axis=1)

        return transformed_x.T

    def _get_intervals(self, X, y, rng):
        """
        Generate random intervals for given parameters.
        """
        intervals = np.zeros((n_intervals, 2), dtype=int)
        for j in range(n_intervals):
            intervals[j][0] = rng.randint(series_length - min_interval)
            length = rng.randint(series_length - intervals[j][0] - 1)
            if length < min_interval:
                length = min_interval
            intervals[j][1] = intervals[j][0] + length
        return intervals

    def _supervised_interval_search(self, ):
        print(":)")

    def _fit_estimator(self, X, X_p, X_d, y, bag, i):
        """
        Fit an estimator - a clone of base_estimator - on input data (X, y)
        transformed using the randomly generated intervals.
        """
        estimator = clone(self.base_estimator)
        estimator.set_params(random_state=self.random_state * 37 * i)

        rng = check_random_state(self.random_state * 37 * i)
        transformed_x = self._transform(X[bag], self._get_intervals(X[bag], y, rng))
        transformed_x_p = self._transform(
            X_p[bag],
            self._get_intervals(X_p[bag], y, rng),
        )
        transformed_x_d = self._transform(
            X_d[bag],
            self._get_intervals(X_d[bag], y, rng),
        )

        return estimator.fit(transformed_x, y)

    def _predict_proba_for_estimator(self, X, X_p, X_d, i):
        """
        Find probability estimates for each class for all cases in X using
        given estimator and intervals.
        """
        transformed_x = self._transform(X, self.intervals_[i][0])
        transformed_x_p = self._transform(X_p, self.intervals_[i][1])
        transformed_x_d = self._transform(X_d, self.intervals_[i][2])



        return self.estimators_[i].predict_proba(transformed_x)
