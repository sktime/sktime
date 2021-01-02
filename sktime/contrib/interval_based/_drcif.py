# -*- coding: utf-8 -*-
""" Diverse Representation Canonical Interval Forest Classifier (DrCIF).
"""

__author__ = ["Matthew Middlehurst"]
__all__ = ["DrCIF"]

import numpy as np
import math

from scipy import signal
from scipy import stats
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import clone
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X, check_X_y
from sktime.classification.base import BaseClassifier

_check_soft_dependencies("catch22")
from sktime.contrib.transformations.catch22_features import Catch22  # noqa: E402


class DrCIF(ForestClassifier, BaseClassifier):
    """Diverse Representation Canonical Interval Forest Classifier.

    Interval based forest making use of the catch22 feature set on randomly
    selected intervals on the base series, periodogram representation and
    differences representation.

    Overview: Input n series length m
    for each tree
        sample 4 + (sqrt(m)*sqrt(d))/3 intervals per representation
        subsample att_subsample_size tsf/catch22 attributes randomly
        randomly select dimension for each interval
        calculate attributes for each interval, concatenate to form new
        data set
        build decision tree on new data set
    ensemble the trees with averaged probability estimates

    This implementation deviates from the original in minor ways. Predictions
    are made using summed probabilites instead of majority vote
    and it does not use the splitting criteria tiny refinement described in
    deng13forest.

    For the original Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/DrCIF.java

    !!!
    The catch22 package is currently unstable, results will differ between operating
    systems. MacOS is presumed to be the correct version, with only minor differences
    for Windows.
    !!!

    Parameters
    ----------
    n_estimators       : int, ensemble size, optional (default to 500)
    n_intervals         : int, number of intervals to extract, optional (default to
    sqrt(series_length)*sqrt(n_dims))
    random_state       : int, seed for random, optional (default to no seed)
    att_subsample_size : int, number of catch22/tsf attributes to subsample
    per classifier, optional (default to 8)
    min_interval       : int, minimum width of an interval, optional (default
    to 3)
    max_interval       : int, maximum width of an interval, optional (default
    to series_length/2)

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
    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        min_interval=3,
        max_interval=None,
        n_estimators=500,
        n_intervals=None,
        att_subsample_size=10,
        random_state=None,
    ):
        super(DrCIF, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_estimators,
        )

        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.att_subsample_size = att_subsample_size

        self.random_state = random_state

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.total_intervals = 0
        self.classifiers = []
        self.atts = []
        self.intervals = []
        self.dims = []
        self.classes_ = []

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y) using random
        intervals and catch22/tsf summary features
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,series_length]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification).
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        rng = check_random_state(self.random_state)

        self.n_instances, self.n_dims, self.series_length = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.classifiers = []
        self.atts = []

        if self.n_intervals is None:
            self.n_intervals = 4 + int(
                (math.sqrt(self.series_length) * math.sqrt(self.n_dims)) / 3
            )
        if self.n_intervals <= 0:
            self.n_intervals = 1
        if self.series_length < self.min_interval:
            self.min_interval = self.series_length

        if self.max_interval is None:
            self.max_interval = self.series_length / 2
        else:
            self.max_interval = self.max_interval
        if self.max_interval < self.min_interval:
            self.max_interval = self.min_interval

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)
        T = [X, X_p, X_d]
        self.total_intervals = self.n_intervals * 2 + int(self.n_intervals / 2)

        self.intervals = np.zeros(
            (self.n_estimators, self.att_subsample_size * self.total_intervals, 2),
            dtype=int,
        )

        c22 = Catch22()

        for i in range(0, self.n_estimators):
            transformed_x = np.empty(
                shape=(
                    self.att_subsample_size * self.total_intervals,
                    self.n_instances,
                ),
                dtype=np.float32,
            )

            self.atts.append(rng.choice(29, self.att_subsample_size, replace=False))
            self.dims.append([])

            p = 0
            j = 0
            for r in range(0, len(T)):
                transform_length = T[r].shape[2]
                transform_n_intervals = (
                    int(self.n_intervals / 2) if r == 1 else self.n_intervals
                )

                # Find the random intervals for classifier i, transformation r
                # and concatenate features
                for _ in range(0, transform_n_intervals):
                    self.dims[i].append(rng.choice(self.n_dims))

                    if rng.random() < 0.5:
                        self.intervals[i][j][0] = rng.randint(
                            0, transform_length - self.min_interval
                        )
                        len_range = min(
                            transform_length - self.intervals[i][j][0],
                            self.max_interval,
                        )
                        length = (
                            rng.randint(0, len_range - self.min_interval)
                            + self.min_interval
                        )
                        self.intervals[i][j][1] = self.intervals[i][j][0] + length
                    else:
                        self.intervals[i][j][1] = (
                            rng.randint(0, transform_length - self.min_interval)
                            + self.min_interval
                        )
                        len_range = min(self.intervals[i][j][1], self.max_interval)
                        length = (
                            rng.randint(0, len_range - self.min_interval)
                            + self.min_interval
                            if len_range - self.min_interval > 0
                            else self.min_interval
                        )
                        self.intervals[i][j][0] = self.intervals[i][j][1] - length

                    for a in range(0, self.att_subsample_size):
                        transformed_x[p] = self.__scif_feature(T[r], i, j, a, c22)
                        p += 1

                    j += 1

            tree = clone(self.base_estimator)
            tree.set_params(**{"random_state": self.random_state})
            transformed_x = transformed_x.T
            np.nan_to_num(transformed_x, False, 0, 0, 0)
            tree.fit(transformed_x, y)
            self.classifiers.append(tree)

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
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

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
        sums = np.zeros((n_test_instances, self.n_classes), dtype=np.float64)

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)
        T = [X, X_p, X_d]

        c22 = Catch22()

        for i in range(0, self.n_estimators):
            transformed_x = np.empty(
                shape=(
                    self.att_subsample_size * self.total_intervals,
                    n_test_instances,
                ),
                dtype=np.float32,
            )

            p = 0
            j = 0
            for r in range(0, len(T)):
                transform_n_intervals = (
                    int(self.n_intervals / 2) if r == 1 else self.n_intervals
                )

                for _ in range(0, transform_n_intervals):
                    for a in range(0, self.att_subsample_size):
                        transformed_x[p] = self.__scif_feature(T[r], i, j, a, c22)
                        p += 1
                    j += 1

            transformed_x = transformed_x.T
            np.nan_to_num(transformed_x, False, 0, 0, 0)
            sums += self.classifiers[i].predict_proba(transformed_x)

        output = sums / (np.ones(self.n_classes) * self.n_estimators)
        return output

    def __scif_feature(self, X, i, j, a, c22):
        if self.atts[i][a] == 22:
            # mean
            return np.mean(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        if self.atts[i][a] == 23:
            # median
            return np.median(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        elif self.atts[i][a] == 24:
            # std_dev
            return np.std(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        elif self.atts[i][a] == 25:
            # slope
            return _slope(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        elif self.atts[i][a] == 26:
            # iqr
            return stats.iqr(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        elif self.atts[i][a] == 27:
            # min
            return np.min(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        elif self.atts[i][a] == 28:
            # max
            return np.max(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                axis=1,
            )
        else:
            return c22._transform_single_feature(
                X[
                    :,
                    self.dims[i][j],
                    self.intervals[i][j][0] : self.intervals[i][j][1],
                ],
                feature=a,
            )
