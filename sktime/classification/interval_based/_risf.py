# -*- coding: utf-8 -*-
"""Random Interval Spectral Ensemble (RISE)."""

__author__ = ["Yi-Xuan Xu"]

__all__ = ["RandomIntervalSpectralForest"]

import numpy as np
from deprecated.sphinx import deprecated
from joblib import Parallel, delayed
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.classification.interval_based._rise import (
    _make_estimator,
    _parallel_build_trees,
    _predict_proba_for_estimator,
    _select_interval,
)


@deprecated(
    version="0.8.1",
    reason="RandomIntervalSpectralForest will be removed in v0.10.0",
    category=FutureWarning,
)
class RandomIntervalSpectralForest(ForestClassifier, BaseClassifier):
    """Random Interval Spectral Forest (RISF).

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

    def fit(self, X, y, **kwargs):
        """Wrap fit to call BaseClassifier.fit.

        This is a fix to get around the problem with multiple inheritance. The
        problem is that if we just override _fit, this class inherits the fit from
        the sklearn class ForestClassifier. This is the simplest solution,
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
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
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

    def _predict(self, X):
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

    def _predict_proba(self, X):
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
