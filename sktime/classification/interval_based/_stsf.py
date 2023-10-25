"""Supervised Time Series Forest Classifier (STSF).

Interval based STSF classifier extracting summary features from intervals selected
through a supervised process.
"""

__author__ = ["matthewmiddlehurst"]
__all__ = ["SupervisedTimeSeriesForest"]

import math

import numpy as np
from joblib import Parallel, delayed
from scipy import signal, stats
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.utils.slope_and_trend import _slope


class SupervisedTimeSeriesForest(BaseClassifier):
    """Supervised Time Series Forest (STSF).

    An ensemble of decision trees built on intervals selected through a supervised
    process as described in _[1].
    Overview: Input n series length m
    For each tree
        - sample X using class-balanced bagging
        - sample intervals for all 3 representations and 7 features using supervised
        - method
        - find mean, median, std, slope, iqr, min and max using their corresponding
        - interval for each rperesentation, concatenate to form new data set
        - build decision tree on new data set
    Ensemble the trees with averaged probability estimates.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_instances_ : int
        The number of train cases.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    intervals : array-like of shape [n_estimators][3][7][n_intervals][2]
        Stores indexes of all start and end points for all estimators. Each estimator
        contains indexes for each representation and feature combination.
    estimators_ : list of shape (n_estimators) of DecisionTreeClassifier
        The collections of estimators trained in fit.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/STSF.java>`_.

    References
    ----------
    .. [1] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020

    Examples
    --------
    >>> from sktime.classification.interval_based import SupervisedTimeSeriesForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = SupervisedTimeSeriesForest(n_estimators=5)
    >>> clf.fit(X_train, y_train)
    SupervisedTimeSeriesForest(n_estimators=5)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "interval",
    }

    def __init__(
        self,
        n_estimators=200,
        n_jobs=1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_instances_ = 0
        self.series_length_ = 0
        self.estimators_ = []
        self.intervals_ = []

        self._base_estimator = DecisionTreeClassifier(criterion="entropy")
        self._stats = [np.mean, np.median, np.std, _slope, stats.iqr, np.min, np.max]

        super().__init__()

    def _fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

        Uses supervised intervals and summary features

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. STSF has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        X = X.squeeze(1)

        self.n_instances_, self.series_length_ = X.shape
        rng = check_random_state(self.random_state)
        cls, class_counts = np.unique(y, return_counts=True)

        self.intervals_ = [[[] for _ in range(3)] for _ in range(self.n_estimators)]

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)

        balance_cases = np.zeros(0, dtype=np.int32)
        average = math.floor(self.n_instances_ / self.n_classes_)
        for i, c in enumerate(cls):
            if class_counts[i] < average:
                cls_idx = np.where(y == c)[0]
                balance_cases = np.concatenate(
                    (rng.choice(cls_idx, size=average - class_counts[i]), balance_cases)
                )

        fit = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._fit_estimator)(
                X,
                X_p,
                X_d,
                y,
                balance_cases,
                i,
            )
            for i in range(self.n_estimators)
        )

        self.estimators_, self.intervals_ = zip(*fit)

        return self

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
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

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

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)

        y_probas = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._predict_proba_for_estimator)(
                X,
                X_p,
                X_d,
                self.intervals_[i],
                self.estimators_[i],
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self.n_estimators
        )
        return output

    def _transform(self, X, intervals):
        """Compute summary stats.

        Find the mean, median, standard deviation, slope, iqr, min and max using
        intervals of input data X generated for each.
        """
        n_instances, _ = X.shape
        total_intervals = 0

        for i in range(len(self._stats)):
            total_intervals += len(intervals[i])
        transformed_x = np.zeros((total_intervals, n_instances), dtype=np.float32)

        p = 0
        for i, f in enumerate(self._stats):
            n_intervals = len(intervals[i])

            for j in range(n_intervals):
                X_slice = X[:, intervals[i][j][0] : intervals[i][j][1]]
                transformed_x[p] = f(X_slice, axis=1)
                p += 1

        return transformed_x.T

    def _get_intervals(self, X, y, rng):
        """Generate intervals using a recursive function and random split point."""
        n_instances, series_length = X.shape
        split_point = (
            int(series_length / 2)
            if series_length <= 8
            else rng.randint(4, series_length - 4)
        )

        cls, class_counts = np.unique(y, return_counts=True)

        s = StandardScaler()
        X_norm = s.fit_transform(X)

        intervals = []
        for function in self._stats:
            function_intervals = []
            self._supervised_interval_search(
                X_norm,
                y,
                function,
                function_intervals,
                cls,
                class_counts,
                0,
                split_point + 1,
            )
            self._supervised_interval_search(
                X_norm,
                y,
                function,
                function_intervals,
                cls,
                class_counts,
                split_point + 1,
                series_length,
            )
            intervals.append(function_intervals)

        return intervals

    def _supervised_interval_search(
        self, X, y, function, function_intervals, classes, class_counts, start, end
    ):
        """Recursive function for finding intervals for a feature using fisher score.

        Given a start and end point the series is split in half and both intervals are
        evaluated. The half with the higher score is retained and used as the new start
        and end for a recursive call.
        """
        series_length = end - start
        if series_length < 4:
            return

        e = start + int(series_length / 2)

        X_l = function(X[:, start:e], axis=1)
        X_r = function(X[:, e:end], axis=1)

        s1 = _fisher_score(X_l, y, classes, class_counts)
        s2 = _fisher_score(X_r, y, classes, class_counts)

        if s2 < s1:
            function_intervals.append([start, e])
            self._supervised_interval_search(
                X,
                y,
                function,
                function_intervals,
                classes,
                class_counts,
                start,
                e,
            )
        else:
            function_intervals.append([e, end])
            self._supervised_interval_search(
                X,
                y,
                function,
                function_intervals,
                classes,
                class_counts,
                e,
                end,
            )

    def _fit_estimator(self, X, X_p, X_d, y, balance_cases, idx):
        """Fit an estimator - a clone of base_estimator - on input data (X, y).

        Transformed using the supervised intervals for each feature and representation.
        """
        estimator = clone(self._base_estimator)
        rs = 5465 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        estimator.set_params(random_state=rs)
        rng = check_random_state(rs)

        class_counts = np.zeros(0)
        while class_counts.shape[0] != self.n_classes_:
            bag = np.concatenate(
                (rng.choice(self.n_instances_, size=self.n_instances_), balance_cases)
            )
            _, class_counts = np.unique(y[bag], return_counts=True)
        n_instances = bag.shape[0]
        bag = bag.astype(int)

        transformed_x = np.zeros((n_instances, 0), dtype=np.float32)

        intervals = self._get_intervals(X[bag], y[bag], rng)
        transformed_x = np.concatenate(
            (transformed_x, self._transform(X[bag], intervals)),
            axis=1,
        )

        intervals_p = self._get_intervals(X_p[bag], y[bag], rng)
        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_p[bag], intervals_p)),
            axis=1,
        )

        intervals_d = self._get_intervals(X_d[bag], y[bag], rng)
        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_d[bag], intervals_d)),
            axis=1,
        )

        return [
            estimator.fit(transformed_x, y[bag]),
            [intervals, intervals_p, intervals_d],
        ]

    def _predict_proba_for_estimator(self, X, X_p, X_d, intervals, estimator):
        """Find probability estimates for each class for all cases in X."""
        n_instances, _ = X.shape
        transformed_x = np.zeros((n_instances, 0), dtype=np.float32)

        transformed_x = np.concatenate(
            (transformed_x, self._transform(X, intervals[0])),
            axis=1,
        )

        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_p, intervals[1])),
            axis=1,
        )

        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_d, intervals[2])),
            axis=1,
        )

        return estimator.predict_proba(transformed_x)

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


def _fisher_score(X, y, classes, class_counts):
    """Fisher score for feature selection."""
    a = 0
    b = 0

    x_mean = np.mean(X)

    for i, cls in enumerate(classes):
        X_cls = X[np.where(y == cls)]
        xy_mean = np.mean(X_cls)
        xy_std = np.std(X_cls)

        a += class_counts[i] * (xy_mean - x_mean) ** 2
        b += class_counts[i] * xy_std**2

    return 0 if b == 0 else a / b
