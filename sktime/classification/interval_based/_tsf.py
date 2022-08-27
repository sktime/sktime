# -*- coding: utf-8 -*-
"""Time Series Forest (TSF) Classifier.

Interval based TSF classifier, extracts basic summary features from random intervals.
"""

__author__ = ["kkoziara", "luiszugasti", "kanand77"]
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
    >>> clf = TimeSeriesForestClassifier(n_estimators=5)
    >>> clf.fit(X_train, y_train)
    TimeSeriesForestClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _base_estimator = DecisionTreeClassifier(criterion="entropy")

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        n_jobs=1,
        random_state=None,
    ):
        super(TimeSeriesForestClassifier, self).__init__(
            min_interval=min_interval,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        BaseClassifier.__init__(self)

    def fit(self, X, y, **kwargs):
        """Wrap fit to call BaseClassifier.fit.

        This is a fix to get around the problem with multiple inheritance. The
        problem is that if we just override _fit, this class inherits the fit from
        the sklearn class BaseTimeSeriesForest. This is the simplest solution,
        albeit a little hacky.
        """
        return BaseClassifier.fit(self, X=X, y=y, **kwargs)

    def predict(self, X, **kwargs) -> np.ndarray:
        """Wrap predict to call BaseClassifier.predict."""
        return BaseClassifier.predict(self, X=X, **kwargs)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        """Wrap predict_proba to call BaseClassifier.predict_proba."""
        return BaseClassifier.predict_proba(self, X=X, **kwargs)

    def _fit(self, X, y):
        BaseTimeSeriesForest._fit(self, X=X, y=y)

    def _predict(self, X) -> np.ndarray:
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

    def _predict_proba(self, X) -> np.ndarray:
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
        X = X.squeeze(1)
        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_single_classifier_proba)(
                X, self.estimators_[i], self.intervals_[i]
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        return output

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
            return {"n_estimators": 2}


def _predict_single_classifier_proba(X, estimator, intervals):
    """Find probability estimates for each class for all cases in X."""
    Xt = _transform(X, intervals)
    return estimator.predict_proba(Xt)
