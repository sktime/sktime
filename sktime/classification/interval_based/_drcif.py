# -*- coding: utf-8 -*-
"""DrCIF classifier.

interval based DrCIF classifier extracting catch22 features from random intervals on
periodogram and differences representations as well as the base series.
"""

__author__ = ["MatthewMiddlehurst"]
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
from sktime.contrib.vector_classifiers._continuous_interval_tree import (
    _drcif_feature,
    ContinuousIntervalTree,
)
from sktime.transformations.panel.catch22 import Catch22
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X_y


class DrCIF(BaseClassifier):
    """Diverse Representation Canonical Interval Forest Classifier (DrCIF).

    Extension of the CIF algorithm using multple representations. Implementation of the
    interval based forest making use of the catch22 feature set on randomly selected
    intervals on the base series, periodogram representation and differences
    representation described in the HIVE-COTE 2.0 paper Middlehurst et al (2021). [1]_

    Overview: Input "n" series with "d" dimensions of length "m".
    For each tree
        - Sample n_intervals intervals per representation of random position and length
        - Subsample att_subsample_size catch22 or summary statistic attributes randomly
        - Randomly select dimension for each interval
        - Calculate attributes for each interval from its representation, concatenate
          to form new data set
        - Build decision tree on new data set
    Ensemble the trees with averaged probability estimates

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_intervals : int, length 3 list of int or None, default=None
        Number of intervals to extract per representation per tree as an int for all
        representations or list for individual settings, if None extracts
        (4 + (sqrt(representation_length) * sqrt(n_dims)) / 3) intervals.
    att_subsample_size : int, default=10
        Number of catch22 or summary statistic attributes to subsample per tree.
    min_interval : int or length 3 list of int, default=4
        Minimum length of an interval per representation as an int for all
        representations or list for individual settings.
    max_interval : int, length 3 list of int or None, default=None
        Maximum length of an interval per representation as an int for all
        representations or list for individual settings, if None set to
        (representation_length / 2).
    base_estimator : BaseEstimator or str, default="DTC"
        Base estimator for the ensemble, can be supplied a sklearn BaseEstimator or a
        string for suggested options.
        "DTC" uses the sklearn DecisionTreeClassifier using entropy as a splitting
        measure.
        "CIT" uses the sktime ContinuousIntervalTree, an implementation of the original
        tree used with embedded attribute processing for faster predictions.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_instances : int
        The number of train cases.
    n_dims : int
        The number of dimensions per case.
    series_length : int
        The length of each series.
    classes_ : list
        The classes labels.
    total_intervals : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals : list of shape (n_estimators) of ndarray with shape (total_intervals,2)
        Stores indexes of each intervals start and end points for all classifiers.
    atts : list of shape (n_estimators) of array with shape (att_subsample_size)
        Attribute indexes of the subsampled catch22 or summary statistic for all
        classifiers.
    dims : list of shape (n_estimators) of array with shape (total_intervals)
        The dimension to extract attributes from each interval for all classifiers.
    transformed_data : list of shape (n_estimators) of ndarray with shape
    (n_instances,total_intervals * att_subsample_size)
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    See Also
    --------
    CanonicalIntervalForest

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/DrCIF.java>`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." arXiv preprint arXiv:2104.07551 (2021).

    Examples
    --------
    >>> from sktime.classification.interval_based import DrCIF
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = DrCIF(n_estimators=10)
    >>> clf.fit(X_train, y_train)
    DrCIF(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": True,
        "capability:contractable": True,
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=None,
        att_subsample_size=10,
        min_interval=4,
        max_interval=None,
        base_estimator="DTC",
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.att_subsample_size = att_subsample_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.base_estimator = base_estimator

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.classes_ = []
        self.total_intervals = 0
        self.estimators_ = []
        self.intervals = []
        self.atts = []
        self.dims = []
        self.transformed_data = []

        self._n_estimators = n_estimators
        self._n_intervals = n_intervals
        self._att_subsample_size = att_subsample_size
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._base_estimator = base_estimator
        self._n_jobs = n_jobs

        super(DrCIF, self).__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances, self.n_dims, self.series_length = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if self.base_estimator == "DTC":
            self._base_estimator = DecisionTreeClassifier(criterion="entropy")
        elif self.base_estimator == "CIT":
            self._base_estimator = ContinuousIntervalTree()
        elif isinstance(self.base_estimator, BaseEstimator):
            self._base_estimator = self.base_estimator
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
            self._n_intervals = [None, None, None]
            self._n_intervals[0] = 4 + int(
                (math.sqrt(self.series_length) * math.sqrt(self.n_dims)) / 3
            )
            self._n_intervals[1] = 4 + int(
                (math.sqrt(X_p.shape[2]) * math.sqrt(self.n_dims)) / 3
            )
            self._n_intervals[2] = 4 + int(
                (math.sqrt(X_d.shape[2]) * math.sqrt(self.n_dims)) / 3
            )
        elif isinstance(self.n_intervals, int):
            self._n_intervals = [self.n_intervals, self.n_intervals, self.n_intervals]
        elif isinstance(self.n_intervals, list) and len(self.n_intervals) == 3:
            self._n_intervals = self.n_intervals
        else:
            raise ValueError("DrCIF n_intervals must be an int or list of length 3.")
        for i, n in enumerate(self._n_intervals):
            if n <= 0:
                self._n_intervals[i] = 1

        if self.att_subsample_size > 25:
            self._att_subsample_size = 25

        if isinstance(self.min_interval, int):
            self._min_interval = [
                self.min_interval,
                self.min_interval,
                self.min_interval,
            ]
        elif isinstance(self.min_interval, list) and len(self.min_interval) == 3:
            self._min_interval = self.min_interval
        else:
            raise ValueError("DrCIF min_interval must be an int or list of length 3.")
        if self.series_length < self._min_interval[0]:
            self._min_interval[0] = self.series_length
        if X_p.shape[2] < self._min_interval[1]:
            self._min_interval[1] = X_p.shape[2]
        if X_d.shape[2] < self._min_interval[2]:
            self._min_interval[2] = X_d.shape[2]

        if self.max_interval is None:
            self._max_interval = [
                self.series_length / 2,
                X_p.shape[2] / 2,
                X_d.shape[2] / 2,
            ]
        elif isinstance(self.max_interval, int):
            self._max_interval = [
                self.max_interval,
                self.max_interval,
                self.max_interval,
            ]
        elif isinstance(self.max_interval, list) and len(self.max_interval) == 3:
            self._max_interval = self.max_interval
        else:
            raise ValueError("DrCIF max_interval must be an int or list of length 3.")
        for i, n in enumerate(self._max_interval):
            if n < self._min_interval[i]:
                self._max_interval[i] = self._min_interval[i]

        self.total_intervals = sum(self._n_intervals)

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self.intervals = []
            self.atts = []
            self.dims = []
            self.transformed_data = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs)(
                    delayed(self._fit_estimator)(
                        X,
                        X_p,
                        X_d,
                        y,
                        i,
                    )
                    for i in range(self._n_jobs)
                )

                (
                    estimators,
                    intervals,
                    dims,
                    atts,
                    transformed_data,
                ) = zip(*fit)

                self.estimators_ += estimators
                self.intervals += intervals
                self.atts += atts
                self.dims += dims
                self.transformed_data += transformed_data

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self._n_jobs)(
                delayed(self._fit_estimator)(
                    X,
                    X_p,
                    X_d,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            (
                self.estimators_,
                self.intervals,
                self.dims,
                self.atts,
                self.transformed_data,
            ) = zip(*fit)

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X):
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

        y_probas = Parallel(n_jobs=self._n_jobs)(
            delayed(self._predict_proba_for_estimator)(
                X,
                X_p,
                X_d,
                self.estimators_[i],
                self.intervals[i],
                self.dims[i],
                self.atts[i],
            )
            for i in range(self._n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self._n_estimators
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

        p = Parallel(n_jobs=self._n_jobs)(
            delayed(self._train_probas_for_estimator)(
                y,
                i,
            )
            for i in range(self._n_estimators)
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
            shape=(self._att_subsample_size * self.total_intervals, self.n_instances),
            dtype=np.float32,
        )

        atts = rng.choice(29, self._att_subsample_size, replace=False)
        dims = rng.choice(self.n_dims, self.total_intervals, replace=True)
        intervals = np.zeros((self.total_intervals, 2), dtype=int)

        p = 0
        j = 0
        for r in range(0, len(T)):
            transform_length = T[r].shape[2]

            # Find the random intervals for classifier i, transformation r
            # and concatenate features
            for _ in range(0, self._n_intervals[r]):
                if rng.random() < 0.5:
                    intervals[j][0] = rng.randint(
                        0, transform_length - self._min_interval[r]
                    )
                    len_range = min(
                        transform_length - intervals[j][0],
                        self._max_interval[r],
                    )
                    length = (
                        rng.randint(0, len_range - self._min_interval[r])
                        + self._min_interval[r]
                    )
                    intervals[j][1] = intervals[j][0] + length
                else:
                    intervals[j][1] = (
                        rng.randint(0, transform_length - self._min_interval[r])
                        + self._min_interval[r]
                    )
                    len_range = min(intervals[j][1], self._max_interval[r])
                    length = (
                        rng.randint(0, len_range - self._min_interval[r])
                        + self._min_interval[r]
                        if len_range - self._min_interval[r] > 0
                        else self._min_interval[r]
                    )
                    intervals[j][0] = intervals[j][1] - length

                for a in range(0, self._att_subsample_size):
                    transformed_x[p] = _drcif_feature(
                        T[r], intervals[j], dims[j], atts[a], c22
                    )
                    p += 1

                j += 1

        tree = _clone_estimator(self._base_estimator, random_state=rs)
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
        if isinstance(self._base_estimator, ContinuousIntervalTree):
            return classifier._predict_proba_drcif(
                X, X_p, X_d, c22, self._n_intervals, intervals, dims, atts
            )
        else:
            T = [X, X_p, X_d]

            transformed_x = np.empty(
                shape=(self._att_subsample_size * self.total_intervals, X.shape[0]),
                dtype=np.float32,
            )

            p = 0
            j = 0
            for r in range(0, len(T)):
                for _ in range(0, self._n_intervals[r]):
                    for a in range(0, self._att_subsample_size):
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

        clf = _clone_estimator(self._base_estimator, rs)
        clf.fit(self.transformed_data[idx][subsample], y[subsample])
        probas = clf.predict_proba(self.transformed_data[idx][oob])

        results = np.zeros((self.n_instances, self.n_classes))
        for n, proba in enumerate(probas):
            results[oob[n]] += proba

        return [results, oob]
