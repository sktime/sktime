# -*- coding: utf-8 -*-
"""Canonical Interval Forest Classifier (CIF)."""

__author__ = ["Matthew Middlehurst"]
__all__ = ["CanonicalIntervalForest"]

import numpy as np
import math

from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution


from sktime.contrib._continuous_interval_tree import (
    _cif_feature,
    ContinuousIntervalTree,
)
from sktime.transformations.panel.catch22 import Catch22
from sktime.utils.validation.panel import check_X, check_X_y
from sktime.classification.base import BaseClassifier


class CanonicalIntervalForest(BaseClassifier):
    """Canonical Interval Forest Classifier.

    Interval based forest making use of the catch22 feature set on randomly
    selected intervals.

    Overview: Input n series length m
    for each tree
        sample sqrt(m) intervals
        subsample att_subsample_size tsf/catch22 attributes randomly
        randomly select dimension for each interval
        calculate attributes for each interval, concatenate to form new
        data set
        build decision tree on new data set
    ensemble the trees with averaged probability estimates

    This implementation deviates from the original in minor ways. Predictions
    are made using summed probabilities instead of majority vote
    and it does not use the splitting criteria tiny refinement described in
    deng13forest by default.

    Parameters
    ----------
    n_estimators       : int, ensemble size, optional (default to 200)
    n_intervals        : int, number of intervals to extract, optional (default to
    sqrt(series_length)*sqrt(n_dims))
    att_subsample_size : int, number of catch22/tsf attributes to subsample
    per classifier, optional (default to 8)
    min_interval       : int, minimum width of an interval, optional (default
    to 3)
    max_interval       : int, maximum width of an interval, optional (default
    to series_length/2)
    n_jobs             : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state       : int, seed for random, optional (default to no seed)

    Attributes
    ----------
    n_classes      : int, extracted from the data
    n_instances    : int, extracted from the data
    n_dims         : int, extracted from the data
    series_length  : int, extracted from the data
    classifiers    : array of shape = [n_estimators] of DecisionTree
    atts           : array of shape = [n_estimators][att_subsample_size]
    catch22/tsf attribute indexes for all classifiers
    intervals      : array of shape = [n_estimators][n_intervals][2] stores
    indexes of all start and end points for all classifiers
    dims           : array of shape = [n_estimators][n_intervals] stores
    the dimension to extract from for each interval

    Notes
    -----
    ..[1] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
    Interval Forest (CIF) Classifier for Time Series Classification."
        IEEE International Conference on Big Data 2020

    For the original Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/CIF.java

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        min_interval=3,
        max_interval=None,
        n_estimators=200,
        n_intervals=None,
        att_subsample_size=8,
        base_estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.base_estimator = base_estimator

        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.att_subsample_size = att_subsample_size

        self.random_state = random_state
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.__n_intervals = n_intervals
        self.__max_interval = max_interval
        self.classifiers = []
        self.atts = []
        self.intervals = []
        self.dims = []
        self.classes_ = []
        self.tree = None

        super(CanonicalIntervalForest, self).__init__()

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

         Uses random intervals and catch22/tsf summary features.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,n_dimensions,
        series_length] or shape = [n_instances,series_length]
        The training input samples.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        self.n_instances, self.n_dims, self.series_length = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if self.base_estimator is None or self.base_estimator == "DTC":
            self.tree = DecisionTreeClassifier(criterion="entropy")
        elif self.base_estimator == "CIT":
            self.tree = ContinuousIntervalTree()
        elif isinstance(self.base_estimator, BaseEstimator):
            self.tree = self.base_estimator
        else:
            raise ValueError("DrCIF invalid base estimator given.")

        if self.n_intervals is None:
            self.__n_intervals = int(
                math.sqrt(self.series_length) * math.sqrt(self.n_dims)
            )
        if self.__n_intervals <= 0:
            self.__n_intervals = 1

        if self.series_length < self.min_interval:
            self.min_interval = self.series_length

        if self.max_interval is None:
            self.__max_interval = self.series_length / 2
        if self.__max_interval < self.min_interval:
            self.__max_interval = self.min_interval

        fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
            )
            for i in range(self.n_estimators)
        )

        self.classifiers, self.intervals, self.dims, self.atts = zip(*fit)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances,n_dimensions,series_length]

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances,n_dimensions,series_length]

        Local variables
        ----------
        n_test_instances     : int, number of cases to classify
        series_length    : int, number of attributes in X, must match
        series_length determined in fit

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)

        n_test_instances, _, series_length = X.shape
        if series_length != self.series_length:
            raise TypeError(
                "ERROR number of attributes in the train does not match "
                "that in the test data"
            )

        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.classifiers[i],
                self.intervals[i],
                self.dims[i],
                self.atts[i],
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        return output

    def _fit_estimator(self, X, y, idx):
        c22 = Catch22(outlier_norm=True)
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (idx + 1)
        rng = check_random_state(rs)

        transformed_x = np.empty(
            shape=(self.att_subsample_size * self.__n_intervals, self.n_instances),
            dtype=np.float32,
        )

        atts = rng.choice(25, self.att_subsample_size, replace=False)
        dims = rng.choice(self.n_dims, self.__n_intervals, replace=True)
        intervals = np.zeros((self.__n_intervals, 2), dtype=int)

        # Find the random intervals for classifier i and concatenate
        # features
        for j in range(0, self.__n_intervals):
            if rng.random() < 0.5:
                intervals[j][0] = rng.randint(0, self.series_length - self.min_interval)
                len_range = min(
                    self.series_length - intervals[j][0],
                    self.__max_interval,
                )
                length = (
                    rng.randint(0, len_range - self.min_interval) + self.min_interval
                )
                intervals[j][1] = intervals[j][0] + length
            else:
                intervals[j][1] = (
                    rng.randint(0, self.series_length - self.min_interval)
                    + self.min_interval
                )
                len_range = min(intervals[j][1], self.__max_interval)
                length = (
                    rng.randint(0, len_range - self.min_interval) + self.min_interval
                    if len_range - self.min_interval > 0
                    else self.min_interval
                )
                intervals[j][0] = intervals[j][1] - length

            for a in range(0, self.att_subsample_size):
                transformed_x[self.att_subsample_size * j + a] = _cif_feature(
                    X, intervals[j], dims[j], atts[a], c22
                )

        tree = clone(self.tree)
        tree.set_params(random_state=rs)
        transformed_x = transformed_x.T
        transformed_x = transformed_x.round(8)
        transformed_x = np.nan_to_num(transformed_x, False, 0, 0, 0)
        tree.fit(transformed_x, y)

        return [tree, intervals, dims, atts]

    def _predict_proba_for_estimator(self, X, classifier, intervals, dims, atts):
        c22 = Catch22(outlier_norm=True)
        if isinstance(self.tree, ContinuousIntervalTree):
            return classifier.predict_proba_cif(X, c22, intervals, dims, atts)
        else:
            transformed_x = np.empty(
                shape=(self.att_subsample_size * self.__n_intervals, X.shape[0]),
                dtype=np.float32,
            )

            for j in range(0, self.__n_intervals):
                for a in range(0, self.att_subsample_size):
                    transformed_x[self.att_subsample_size * j + a] = _cif_feature(
                        X, intervals[j], dims[j], atts[a], c22
                    )

            transformed_x = transformed_x.T
            transformed_x.round(8)
            np.nan_to_num(transformed_x, False, 0, 0, 0)

            return classifier.predict_proba(transformed_x)

    def temporal_importance_curves(self):
        if not isinstance(self.tree, ContinuousIntervalTree):
            raise ValueError(
                "CIF base estimator for temporal importance curves must"
                " be ContinuousIntervalTree."
            )

        curves = np.zeros((25, self.n_dims, self.series_length))
        for i, tree in enumerate(self.classifiers):
            splits, gains = tree.tree_splits_gain()

            for n, split in enumerate(splits):
                gain = gains[n]
                interval = int(split / self.att_subsample_size)
                att = self.atts[i][int(split % self.att_subsample_size)]
                dim = self.dims[i][interval]

                for j in range(
                    self.intervals[i][interval][0], self.intervals[i][interval][1] + 1
                ):
                    curves[att][dim][j] += gain
