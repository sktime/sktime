# -*- coding: utf-8 -*-
"""Diverse Representation Canonical Interval Forest Classifier (DrCIF)."""

__author__ = ["Matthew Middlehurst"]
__all__ = ["DrCIF"]

import math
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.contrib.classification_intervals._continuous_interval_tree import (
    _drcif_feature,
    ContinuousIntervalTree,
)
from sktime.transformations.panel.catch22 import Catch22
from sktime.utils.validation.panel import check_X, check_X_y


class DrCIF(BaseClassifier):
    """Diverse Representation Canonical Interval Forest Classifier (DrCIF).

    Interval based forest making use of the catch22 feature set on randomly
    selected intervals on the base series, periodogram representation and
    differences representation.

    Overview: Input n series length m
    for each tree
        sample 4 + (sqrt(m)*sqrt(d)) / 3 intervals per representation
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
    n_intervals        : int or size 3 list, number of intervals to extract per
    representation, optional (default to 4 + (sqrt(representation_length)*sqrt(n_dims))
    / 3)
    att_subsample_size : int, number of catch22/tsf attributes to subsample
    per classifier, optional (default to 10)
    min_interval       : int or size 3 list, minimum width of an interval
    per representation, optional (default to 4)
    max_interval       : int or size 3 list, maximum width of an interval
    per representation, optional (default to representation_length / 2)
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
    For the original Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/DrCIF.java

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
        min_interval=None,
        max_interval=None,
        n_estimators=200,
        n_intervals=None,
        att_subsample_size=10,
        base_estimator=None,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.base_estimator = base_estimator

        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.att_subsample_size = att_subsample_size

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data
        self.transformed_data = []

        self.random_state = random_state
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.__n_intervals = []
        self.__max_interval = []
        self.__min_interval = []
        self.total_intervals = 0
        self.classifiers = []
        self.intervals = []
        self.atts = []
        self.dims = []
        self.classes_ = []
        self.tree = None

        super(DrCIF, self).__init__()

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

         Uses random intervals and catch22/basic summary features

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

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if self.base_estimator is None or self.base_estimator == "DTC":
            self.tree = DecisionTreeClassifier(criterion="entropy")
        elif self.base_estimator == "CIT":
            self.tree = ContinuousIntervalTree()
        elif isinstance(self.base_estimator, BaseEstimator):
            self.tree = self.base_estimator
        else:
            raise ValueError("DrCIF invalid base estimator given.")

        X_p = np.zeros(
            (
                self.n_instances,
                self.n_dims,
                int(
                    math.pow(2, math.ceil(math.log(self.series_length, 2)))
                    - self.series_length
                ),
            )
        )
        X_p = np.concatenate((X, X_p), axis=2)
        X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        X_d = np.diff(X, 1)

        if self.n_intervals is None:
            self.__n_intervals = [None, None, None]
            self.__n_intervals[0] = 4 + int(
                (math.sqrt(self.series_length) * math.sqrt(self.n_dims)) / 3
            )
            self.__n_intervals[1] = 4 + int(
                (math.sqrt(X_p.shape[2]) * math.sqrt(self.n_dims)) / 3
            )
            self.__n_intervals[2] = 4 + int(
                (math.sqrt(X_d.shape[2]) * math.sqrt(self.n_dims)) / 3
            )
        elif isinstance(self.n_intervals, int):
            self.__n_intervals = [self.n_intervals, self.n_intervals, self.n_intervals]
        elif isinstance(self.n_intervals, list) and len(self.n_intervals) == 3:
            self.__n_intervals = self.n_intervals
        else:
            raise ValueError("DrCIF n_intervals must be an int or list of length 3.")
        for i, n in enumerate(self.__n_intervals):
            if n <= 0:
                self.__n_intervals[i] = 1

        if self.min_interval is None:
            self.__min_interval = [4, 4, 4]
        elif isinstance(self.min_interval, int):
            self.__min_interval = [
                self.min_interval,
                self.min_interval,
                self.min_interval,
            ]
        elif isinstance(self.min_interval, list) and len(self.min_interval) == 3:
            self.__min_interval = self.min_interval
        else:
            raise ValueError("DrCIF min_interval must be an int or list of length 3.")
        if self.series_length < self.__min_interval[0]:
            self.__min_interval[0] = self.series_length
        if X_p.shape[2] < self.__min_interval[1]:
            self.__min_interval[1] = X_p.shape[2]
        if X_d.shape[2] < self.__min_interval[2]:
            self.__min_interval[2] = X_d.shape[2]

        if self.max_interval is None:
            self.__max_interval = [
                self.series_length / 2,
                X_p.shape[2] / 2,
                X_d.shape[2] / 2,
            ]
        elif isinstance(self.max_interval, int):
            self.__max_interval = [
                self.max_interval,
                self.max_interval,
                self.max_interval,
            ]
        elif isinstance(self.max_interval, list) and len(self.max_interval) == 3:
            self.__max_interval = self.max_interval
        else:
            raise ValueError("DrCIF max_interval must be an int or list of length 3.")
        for i, n in enumerate(self.__max_interval):
            if n < self.__min_interval[i]:
                self.__max_interval[i] = self.__min_interval[i]

        self.total_intervals = sum(self.__n_intervals)

        if time_limit > 0:
            self.n_estimators = 0
            self.classifiers = []
            self.intervals = []
            self.atts = []
            self.dims = []
            self.transformed_data = []

            while (
                train_time < time_limit
                and self.n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fit_estimator)(
                        X,
                        X_p,
                        X_d,
                        y,
                        i,
                    )
                    for i in range(self.n_jobs)
                )

                (
                    classifiers,
                    intervals,
                    dims,
                    atts,
                    transformed_data,
                ) = zip(*fit)

                self.classifiers += classifiers
                self.intervals += intervals
                self.atts += atts
                self.dims += dims
                self.transformed_data += transformed_data

                self.n_estimators += self.n_jobs
                train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_estimator)(
                    X,
                    X_p,
                    X_d,
                    y,
                    i,
                )
                for i in range(self.n_estimators)
            )

            (
                self.classifiers,
                self.intervals,
                self.dims,
                self.atts,
                self.transformed_data,
            ) = zip(*fit)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Find predictions for all cases in X. Built on top of predict_proba.

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
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances,n_dimensions,series_length]

        Local variables
        ----------
        n_test_instances     : int, number of cases to classify
        series_length    : int, number of attributes in X, must match
        _num_atts determined in fit

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

        X_p = np.zeros(
            (
                n_test_instances,
                self.n_dims,
                int(
                    math.pow(2, math.ceil(math.log(self.series_length, 2)))
                    - self.series_length
                ),
            )
        )
        X_p = np.concatenate((X, X_p), axis=2)
        X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        X_d = np.diff(X, 1)

        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_proba_for_estimator)(
                X,
                X_p,
                X_d,
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

    def _get_train_probs(self, X, y):
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        n_instances, n_dims, series_length = X.shape

        if (
            n_instances != self.n_instances
            or n_dims != self.n_dims
            or series_length != self.series_length
        ):
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        p = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_probas_for_estimator)(
                y,
                i,
            )
            for i in range(self.n_estimators)
        )
        y_probas, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(n_instances)
        for oob in oobs:
            for inst in oob:
                divisors[inst] += 1

        for i in range(n_instances):
            results[i] = (
                np.ones(self.n_classes) * (1 / self.n_classes)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes) * divisors[i])
            )

        return results

    def _fit_estimator(self, X, X_p, X_d, y, idx):
        c22 = Catch22(outlier_norm=True)
        T = [X, X_p, X_d]
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (idx + 1)
        rng = check_random_state(rs)

        transformed_x = np.empty(
            shape=(self.att_subsample_size * self.total_intervals, self.n_instances),
            dtype=np.float32,
        )

        atts = rng.choice(29, self.att_subsample_size, replace=False)
        dims = rng.choice(self.n_dims, self.total_intervals, replace=True)
        intervals = np.zeros((self.total_intervals, 2), dtype=int)

        p = 0
        j = 0
        for r in range(0, len(T)):
            transform_length = T[r].shape[2]

            # Find the random intervals for classifier i, transformation r
            # and concatenate features
            for _ in range(0, self.__n_intervals[r]):
                if rng.random() < 0.5:
                    intervals[j][0] = rng.randint(
                        0, transform_length - self.__min_interval[r]
                    )
                    len_range = min(
                        transform_length - intervals[j][0],
                        self.__max_interval[r],
                    )
                    length = (
                        rng.randint(0, len_range - self.__min_interval[r])
                        + self.__min_interval[r]
                    )
                    intervals[j][1] = intervals[j][0] + length
                else:
                    intervals[j][1] = (
                        rng.randint(0, transform_length - self.__min_interval[r])
                        + self.__min_interval[r]
                    )
                    len_range = min(intervals[j][1], self.__max_interval[r])
                    length = (
                        rng.randint(0, len_range - self.__min_interval[r])
                        + self.__min_interval[r]
                        if len_range - self.__min_interval[r] > 0
                        else self.__min_interval[r]
                    )
                    intervals[j][0] = intervals[j][1] - length

                for a in range(0, self.att_subsample_size):
                    transformed_x[p] = _drcif_feature(
                        T[r], intervals[j], dims[j], atts[a], c22
                    )
                    p += 1

                j += 1

        tree = _clone_estimator(self.tree, random_state=rs)
        transformed_x = transformed_x.T
        transformed_x = transformed_x.round(8)
        transformed_x = np.nan_to_num(transformed_x, False, 0, 0, 0)
        tree.fit(transformed_x, y)

        return [
            tree,
            intervals,
            dims,
            atts,
            transformed_x if self.save_transformed_data else None,
        ]

    def _predict_proba_for_estimator(
        self, X, X_p, X_d, classifier, intervals, dims, atts
    ):
        c22 = Catch22(outlier_norm=True)
        if isinstance(self.tree, ContinuousIntervalTree):
            return classifier.predict_proba_drcif(
                X, X_p, X_d, c22, self.__n_intervals, intervals, dims, atts
            )
        else:
            T = [X, X_p, X_d]

            transformed_x = np.empty(
                shape=(self.att_subsample_size * self.total_intervals, X.shape[0]),
                dtype=np.float32,
            )

            p = 0
            j = 0
            for r in range(0, len(T)):
                for _ in range(0, self.__n_intervals[r]):
                    for a in range(0, self.att_subsample_size):
                        transformed_x[p] = _drcif_feature(
                            T[r], intervals[j], dims[j], atts[a], c22
                        )
                        p += 1
                    j += 1

            transformed_x = transformed_x.T
            transformed_x.round(8)
            np.nan_to_num(transformed_x, False, 0, 0, 0)

            return classifier.predict_proba(transformed_x)

    def _train_probas_for_estimator(self, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (idx + 1)
        rng = check_random_state(rs)

        indices = range(self.n_instances)
        subsample = rng.choice(self.n_instances, size=self.n_instances)
        oob = [n for n in indices if n not in subsample]

        clf = _clone_estimator(self.tree, rs)
        clf.fit(self.transformed_data[idx][subsample], y[subsample])
        probas = clf.predict_proba(self.transformed_data[idx][oob])

        results = np.zeros((self.n_instances, self.n_classes))
        for n, proba in enumerate(probas):
            results[oob[n]] += proba

        return [results, oob]
