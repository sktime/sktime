"""CIF classifier.

Interval based CIF classifier extracting catch22 features from random intervals.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["CanonicalIntervalForest"]

import math

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.sklearn._continuous_interval_tree import (
    ContinuousIntervalTree,
)
from sktime.transformations.panel.catch22 import Catch22


class CanonicalIntervalForest(BaseClassifier):
    """Canonical Interval Forest Classifier (CIF).

    Implementation of the nterval based forest making use of the catch22 feature set
    on randomly selected intervals described in Middlehurst et al. (2020). [1]_

    Overview: Input "n" series with "d" dimensions of length "m".
    For each tree
        - Sample n_intervals intervals of random position and length
        - Subsample att_subsample_size catch22 or summary statistic attributes randomly
        - Randomly select dimension for each interval
        - Calculate attributes for each interval, concatenate to form new
          data set
        - Build decision tree on new data set
    ensemble the trees with averaged probability estimates

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_intervals : int or None, default=None
        Number of intervals to extract per tree, if None extracts
        (sqrt(series_length) * sqrt(n_dims)) intervals.
    att_subsample_size : int, default=8
        Number of catch22 or summary statistic attributes to subsample per tree.
    min_interval : int, default=3
        Minimum length of an interval.
    max_interval : int or None, default=None
        Maximum length of an interval, if None set to (series_length / 2).
    base_estimator : BaseEstimator or str, default="DTC"
        Base estimator for the ensemble, can be supplied a sklearn BaseEstimator or a
        string for suggested options.
        "DTC" uses the sklearn DecisionTreeClassifier using entropy as a splitting
        measure.
        "CIT" uses the sktime ContinuousIntervalTree, an implementation of the original
        tree used with embedded attribute processing for faster predictions.
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
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of ndarray with shape (n_intervals,2)
        Stores indexes of each intervals start and end points for all classifiers.
    atts_ : list of shape (n_estimators) of array with shape (att_subsample_size)
        Attribute indexes of the subsampled catch22 or summary statistic for all
        classifiers.
    dims_ : list of shape (n_estimators) of array with shape (n_intervals)
        The dimension to extract attributes from each interval for all classifiers.

    See Also
    --------
    DrCIF

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/CIF.java>`_.

    References
    ----------
    .. [1] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020

    Examples
    --------
    >>> from sktime.classification.interval_based import CanonicalIntervalForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True) # doctest: +SKIP
    >>> clf = CanonicalIntervalForest(
    ...     n_estimators=3, n_intervals=2, att_subsample_size=2
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    CanonicalIntervalForest(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "interval",
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=None,
        att_subsample_size=8,
        min_interval=3,
        max_interval=None,
        base_estimator="CIT",
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.att_subsample_size = att_subsample_size
        self.base_estimator = base_estimator

        self.random_state = random_state
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.estimators_ = []
        self.intervals_ = []
        self.atts_ = []
        self.dims_ = []

        self._n_intervals = n_intervals
        self._att_subsample_size = att_subsample_size
        self._max_interval = max_interval
        self._min_interval = min_interval
        self._base_estimator = base_estimator

        super().__init__()

    def _fit(self, X, y):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.base_estimator.lower() == "dtc":
            self._base_estimator = DecisionTreeClassifier(criterion="entropy")
        elif self.base_estimator.lower() == "cit":
            self._base_estimator = ContinuousIntervalTree()
        elif isinstance(self.base_estimator, BaseEstimator):
            self._base_estimator = self.base_estimator
        else:
            raise ValueError("DrCIF invalid base estimator given.")

        if self.n_intervals is None:
            self._n_intervals = int(
                math.sqrt(self.series_length_) * math.sqrt(self.n_dims_)
            )
        if self._n_intervals <= 0:
            self._n_intervals = 1

        if self.att_subsample_size > 25:
            self._att_subsample_size = 25

        if self.series_length_ <= self.min_interval:
            self._min_interval = self.series_length_ - 1
        elif self.min_interval < 3:
            self._min_interval = 3

        if self.max_interval is None:
            self._max_interval = self.series_length_ / 2
        if self._max_interval < self._min_interval:
            self._max_interval = self._min_interval

        fit = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
            )
            for i in range(self.n_estimators)
        )

        self.estimators_, self.intervals_, self.dims_, self.atts_ = zip(*fit)

        return self

    def _predict(self, X) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        n_test_instances, _, series_length = X.shape
        if series_length != self.series_length_:
            raise ValueError(
                "ERROR number of attributes in the train does not match "
                "that in the test data"
            )

        y_probas = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                self.intervals_[i],
                self.dims_[i],
                self.atts_[i],
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self.n_estimators
        )
        return output

    def _fit_estimator(self, X, y, idx):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _drcif_feature,
        )

        c22 = Catch22(outlier_norm=True)
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        transformed_x = np.empty(
            shape=(self._att_subsample_size * self._n_intervals, self.n_instances_),
            dtype=np.float32,
        )

        atts = rng.choice(25, self._att_subsample_size, replace=False)
        dims = rng.choice(self.n_dims_, self._n_intervals, replace=True)
        intervals = np.zeros((self._n_intervals, 2), dtype=int)

        # Find the random intervals for classifier i and concatenate
        # features
        for j in range(0, self._n_intervals):
            if rng.random() < 0.5:
                intervals[j][0] = rng.randint(
                    0, self.series_length_ - self._min_interval
                )
                len_range = min(
                    self.series_length_ - intervals[j][0],
                    self._max_interval,
                )
                length = (
                    rng.randint(0, len_range - self._min_interval) + self._min_interval
                    if len_range - self._min_interval > 0
                    else self._min_interval
                )
                intervals[j][1] = intervals[j][0] + length
            else:
                intervals[j][1] = (
                    rng.randint(0, self.series_length_ - self._min_interval)
                    + self._min_interval
                )
                len_range = min(intervals[j][1], self._max_interval)
                length = (
                    rng.randint(0, len_range - self._min_interval) + self._min_interval
                    if len_range - self._min_interval > 0
                    else self._min_interval
                )
                intervals[j][0] = intervals[j][1] - length

            for a in range(0, self._att_subsample_size):
                transformed_x[self._att_subsample_size * j + a] = _drcif_feature(
                    X, intervals[j], dims[j], atts[a], c22, case_id=j
                )

        tree = _clone_estimator(self._base_estimator, random_state=rs)
        transformed_x = transformed_x.T
        transformed_x = transformed_x.round(8)
        if self.base_estimator == "CIT":
            transformed_x = np.nan_to_num(
                transformed_x, False, posinf=np.nan, neginf=np.nan
            )
        else:
            transformed_x = np.nan_to_num(transformed_x, False, 0, 0, 0)
        tree.fit(transformed_x, y)

        return [tree, intervals, dims, atts]

    def _predict_proba_for_estimator(self, X, classifier, intervals, dims, atts):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _drcif_feature,
        )

        c22 = Catch22(outlier_norm=True)
        if isinstance(self._base_estimator, ContinuousIntervalTree):
            return classifier._predict_proba_cif(X, c22, intervals, dims, atts)
        else:
            transformed_x = np.empty(
                shape=(self._att_subsample_size * self._n_intervals, X.shape[0]),
                dtype=np.float32,
            )

            for j in range(0, self._n_intervals):
                for a in range(0, self._att_subsample_size):
                    transformed_x[self._att_subsample_size * j + a] = _drcif_feature(
                        X, intervals[j], dims[j], atts[a], c22, case_id=j
                    )

            transformed_x = transformed_x.T
            transformed_x.round(8)
            np.nan_to_num(transformed_x, False, 0, 0, 0)

            return classifier.predict_proba(transformed_x)

    def _temporal_importance_curves(self, normalise_time_points=False):
        if not isinstance(self._base_estimator, ContinuousIntervalTree):
            raise ValueError(
                "CIF base estimator for temporal importance curves must"
                " be ContinuousIntervalTree."
            )

        curves = np.zeros((25, self.n_dims_, self.series_length_))
        if normalise_time_points:
            counts = np.zeros((25, self.n_dims_, self.series_length_))

        for i, tree in enumerate(self.estimators_):
            splits, gains = tree.tree_node_splits_and_gain()

            for n, split in enumerate(splits):
                gain = gains[n]
                interval = int(split / self._att_subsample_size)
                att = self.atts_[i][int(split % self._att_subsample_size)]
                dim = self.dims_[i][interval]

                for j in range(
                    self.intervals_[i][interval][0], self.intervals_[i][interval][1] + 1
                ):
                    curves[att][dim][j] += gain
                    if normalise_time_points:
                        counts[att][dim][j] += 1

        if normalise_time_points:
            counts = counts / self.n_estimators / self._n_intervals
            curves /= counts

        return curves

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
            return {"n_estimators": 10, "n_intervals": 2, "att_subsample_size": 4}
        else:
            return {"n_estimators": 2, "n_intervals": 2, "att_subsample_size": 2}
