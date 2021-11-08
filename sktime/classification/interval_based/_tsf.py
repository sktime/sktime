# -*- coding: utf-8 -*-
"""Time Series Forest (TSF) Classifier.

Interval based TSF classifier, extracts basic summary features from random intervals.
"""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti", "kanand77"]
__all__ = ["TimeSeriesForestClassifier"]

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sktime.classification.base import BaseClassifier
from sktime.series_as_features.base.estimators.interval_based import (
    BaseTimeSeriesForest,
)
from sktime.series_as_features.base.estimators.interval_based._tsf import _transform
from sktime.utils.validation.panel import check_X


class TimeSeriesForestClassifier(
    BaseTimeSeriesForest, ForestClassifier, BaseClassifier
):
    """Time series forest classifier.

    A time series forest is an ensemble of decision trees built on random intervals.
    Overview: Input n series length m.
    For each tree
        - sample sqrt(m) intervals,
        - find mean, std and slope for each interval, concatenate to form new
        data set,
        - build decision tree on new data set.
    Ensemble the trees with averaged probability estimates.

    This implementation deviates from the original in minor ways. It samples
    intervals with replacement and does not use the splitting criteria tiny
    refinement described in [1].

    This is an intentionally stripped down, non
    configurable version for use as a hive-cote component. For a configurable
    tree based ensemble, see sktime.classifiers.ensemble.TimeSeriesForestClassifier

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_interval : int, default=3
        Minimum length of an interval.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/TSF.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction",Information Sciences, 239, 2013

    Examples
    --------
    >>> from sktime.classification.interval_based import TimeSeriesForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = TimeSeriesForestClassifier(n_estimators=10)
    >>> clf.fit(X_train, y_train)
    TimeSeriesForestClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _base_estimator = DecisionTreeClassifier(criterion="entropy")

    def predict(self, X):
        """Find predictions for all cases in X. Built on top of predict_proba.

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
        """Find probability estimates for each class for all cases in X.

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
            raise ValueError(
                "The number of time points in the training data does not match "
                "that in the test data."
            )
        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba)(X, self.estimators_[i], self.intervals_[i])
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        return output


def _predict_proba(X, estimator, intervals):
    """Find probability estimates for each class for all cases in X."""
    Xt = _transform(X, intervals)
    return estimator.predict_proba(Xt)
