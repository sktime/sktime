""" Canonical Interval Forest Classifier (CIF).
"""
import math
from sklearn import clone
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.transformers.series_as_features.catch22_features import Catch22

__author__ = ["Matthew Middlehurst"]
__all__ = ["CanonicalIntervalForest"]

import random

import numpy as np
from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X_y, check_X
from sktime.classification.base import BaseClassifier


class CanonicalIntervalForest(ForestClassifier, BaseClassifier):
    """Canonical Interval Forest Classifier.

    @article{middlehurst2020canonical,
      title={The Canonical Interval Forest (CIF) Classifier for Time Series
      Classification},
      author={Middlehurst, Matthew and Large, James and Bagnall, Anthony},
      journal={arXiv preprint arXiv:2008.09172},
      year={2020}
    }

    Interval based forest making use of the catch22 feature set on randomly
    selected intervals.

    Overview: Input n series length m
    for each tree
        sample sqrt(m) intervals
        subsample att_subsample_size tsf/catch22 attributes randomly
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
    /tsml/classifiers/interval_based/CIF.java


    Parameters
    ----------
    n_estimators       : int, ensemble size, optional (default = 500)
    random_state       : int, seed for random, optional (default to no seed,
    I think!)
    att_subsample_size : int, number of catch22/tsf attributes to subsample
    per classifier
    min_interval       : int, minimum width of an interval, optional (default
    to 3)
    max_interval       : int, maximum width of an interval, optional (default
    to series_length/2)

    Attributes
    ----------
    n_classes      : int, extracted from the data
    n_instances    : int, extracted from the data
    series_length  : int, extracted from the data
    n_intervals    : int, sqrt(series_length)
    classifiers    : array of shape = [n_estimators] of DecisionTree
    self.atts      : array of shape = [n_estimators][att_subsample_size]
    catch22/tsf attribute indexes for all classifiers
    intervals      : array of shape = [n_estimators][n_intervals][2] stores
    indexes of all start and end points for all classifiers

    """

    def __init__(self,
                 random_state=None,
                 min_interval=3,
                 max_interval=None,
                 n_estimators=500,
                 att_subsample_size=8
                 ):
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.att_subsample_size = att_subsample_size

        self.random_state = random_state

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.series_length = 0
        self.n_intervals = 0
        self.classifiers = []
        self.atts = []
        self.intervals = []
        self.classes_ = []

        super(CanonicalIntervalForest, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_estimators)

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

        X, y = check_X_y(X, y, enforce_univariate=True)
        X = tabularize(X, return_array=True)

        rng = check_random_state(self.random_state)

        self.n_instances, self.series_length = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.classifiers = []
        self.atts = []

        self.n_intervals = int(math.sqrt(self.series_length))
        if self.n_intervals == 0:
            self.n_intervals = 1
        if self.series_length < self.min_interval:
            self.min_interval = self.series_length
        self.intervals = np.zeros((self.n_estimators, self.att_subsample_size
                                   * self.n_intervals, 2), dtype=int)

        if self.max_interval is None:
            max_interval_length = self.series_length / 2
            if max_interval_length < self.min_interval:
                max_interval_length = self.min_interval
        else:
            max_interval_length = self.max_interval

        c22 = Catch22()

        for i in range(0, self.n_estimators):
            transformed_x = np.empty(shape=(
                self.att_subsample_size * self.n_intervals, self.n_instances))

            self.atts.append(rng.choice(25, self.att_subsample_size,
                                        replace=False))

            # Find the random intervals for classifier i and concatentate
            # features
            for j in range(0, self.n_intervals):
                if rng.random() < 0.5:
                    self.intervals[i][j][0] = \
                        rng.randint(0, self.series_length - self.min_interval)
                    len_range = min(self.series_length -
                                    self.intervals[i][j][0],
                                    max_interval_length)
                    length = rng.randint(0, len_range - self.min_interval) + \
                        self.min_interval
                    self.intervals[i][j][1] = self.intervals[i][j][0] + length
                else:
                    self.intervals[i][j][1] = \
                        rng.randint(0, self.series_length -
                                    self.min_interval) + self.min_interval
                    len_range = min(self.intervals[i][j][1],
                                    max_interval_length)
                    length = rng.randint(0, len_range - self.min_interval) \
                        + self.min_interval \
                        if len_range - self.min_interval > 0 \
                        else self.min_interval
                    self.intervals[i][j][0] = self.intervals[i][j][1] - length

                for a in range(0, self.att_subsample_size):
                    if self.atts[i][a] == 22:
                        # mean
                        transformed_x[self.att_subsample_size * j + a] = \
                            np.mean(X[:, self.intervals[i][j][0]:
                                    self.intervals[i][j][1]], axis=1)
                    elif self.atts[i][a] == 23:
                        # std_dev
                        transformed_x[self.att_subsample_size * j + a] = \
                            np.std(X[:, self.intervals[i][j][0]:
                                     self.intervals[i][j][1]], axis=1)
                    elif self.atts[i][a] == 24:
                        # slope
                        transformed_x[self.att_subsample_size * j + a] = \
                            self._lsq_fit(X[:, self.intervals[i][j][0]:
                                            self.intervals[i][j][1]])
                    else:
                        transformed_x[self.att_subsample_size * j + a] = \
                            c22._transform_single_feature(
                                X[:, self.intervals[i][j][0]:
                                  self.intervals[i][j][1]], feature=a)

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
        proba = self.predict_proba(X)
        return [self.classes_[int(np.random.choice(
            np.flatnonzero(prob == prob.max())))]
                for prob in proba]

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
        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)

        n_test_instances, series_length = X.shape
        if series_length != self.series_length:
            raise TypeError(
                " ERROR number of attributes in the train does not match "
                "that in the test data")
        sums = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)

        c22 = Catch22()

        for i in range(0, self.n_estimators):
            transformed_x = np.empty(
                shape=(self.att_subsample_size * self.n_intervals,
                       n_test_instances), dtype=np.float32)
            for j in range(0, self.n_intervals):
                for a in range(0, self.att_subsample_size):
                    if self.atts[i][a] == 22:
                        # mean
                        transformed_x[self.att_subsample_size * j + a] = \
                            np.mean(X[:, self.intervals[i][j][0]:
                                    self.intervals[i][j][1]], axis=1)
                    elif self.atts[i][a] == 23:
                        # std_dev
                        transformed_x[self.att_subsample_size * j + a] = \
                            np.std(X[:, self.intervals[i][j][0]:
                                     self.intervals[i][j][1]], axis=1)
                    elif self.atts[i][a] == 24:
                        # slope
                        transformed_x[self.att_subsample_size * j + a] = \
                            self._lsq_fit(X[:, self.intervals[i][j][0]:
                                            self.intervals[i][j][1]])
                    else:
                        transformed_x[self.att_subsample_size * j + a] = \
                            c22._transform_single_feature(
                                X[:, self.intervals[i][j][0]:
                                  self.intervals[i][j][1]], feature=a)

            transformed_x = transformed_x.T
            np.nan_to_num(transformed_x, False, 0, 0, 0)
            sums += self.classifiers[i].predict_proba(transformed_x)

        output = sums / (np.ones(self.n_classes) * self.n_estimators)
        return output

    def _lsq_fit(self, Y):
        """ Find the slope for each series (row) of Y
        Parameters
        ----------
        Y: array of shape = [n_samps, interval_size]

        Returns
        ----------
        slope: array of shape = [n_samps]

        """
        x = np.arange(Y.shape[1]) + 1
        slope = (np.mean(x * Y, axis=1) - np.mean(x) * np.mean(Y, axis=1)) / (
                (x * x).mean() - x.mean() ** 2)
        return slope
