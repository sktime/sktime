# -*- coding: utf-8 -*-
""" catch22 Forest Classifier.

A forest classifier based on catch22 features
"""

__author__ = ["Matthew Middlehurst"]
__all__ = ["Catch22ForestClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.panel import check_X

_check_soft_dependencies("catch22")
from catch22 import catch22_all  # noqa: E402


class Catch22ForestClassifier(BaseClassifier):
    """Canonical Time-series Characteristics (catch22).

    Overview: Input n series length m. Transforms series into the 22 catch22
    features [1] extracted from the hctsa toolbox[2] and builds a random forest
    classifier on them.

    Parameters
    ----------
    n_estimators            : int, number of trees in the random forest
    bootstrap               : bool, if true draw samples with replacement
    n_jobs                  : int or None, number of jobs to run in parallel
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    bagging_classifier      : trained forest classifier

    Notes
    -----
    ..[1] Fulcher, B. D., & Jones, N. S. (2017). hctsa: A computational framework
    for automated time-series phenotyping using massive feature extraction.
    Cell systems, 5(5), 527-531.

    ..[2] Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their
    methods. Journal of the Royal Society Interface, 10(83), 20130048.

    Original Catch22ForestClassifier:
    https://github.com/chlubba/sktime-catch22
    catch22 package implementations:
    https://github.com/chlubba/catch22
    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/transformers/Catch22.java
    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(self, n_estimators=100, n_jobs=None, random_state=None):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifier = None
        self.n_timestep_ = 0
        self.n_dims_ = 0
        self.classes_ = []
        super(Catch22ForestClassifier, self).__init__()

    def fit(self, X, y):
        """Fit a random catch22 feature forest classifier.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        c22_list = []
        for i in range(n_instances):
            series = X[i, :]
            c22_dict = catch22_all(series)
            c22_list.append(c22_dict["values"])

        self.classifier = RandomForestClassifier(
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )

        X_c22 = np.array(c22_list)
        np.nan_to_num(X_c22, False, 0, 0, 0)

        self.classifier.fit(X_c22, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Make predictions for all cases in X.

        Parameters
        ----------
        X : The testing input samples of shape [n_instances,1].

        Returns
        -------
        output : numpy array of shape = [n_instances]
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        c22_list = []
        for i in range(n_instances):
            series = X[i, :]
            c22_dict = catch22_all(series)
            c22_list.append(c22_dict["values"])

        X_c22 = np.array(c22_list)
        np.nan_to_num(X_c22, False, 0, 0, 0)

        return self.classifier.predict(X_c22)

    def predict_proba(self, X):
        """Make class probability estimates on each case in X.

        Parameters
        ----------
        X - pandas dataframe of testing data of shape [n_instances,1].

        Returns
        -------
        output : numpy array of shape =
                [n_instances, num_classes] of probabilities
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        n_instances = X.shape[0]
        X = np.reshape(X, (n_instances, -1))

        c22_list = []
        for i in range(n_instances):
            series = X[i, :]
            c22_dict = catch22_all(series)
            c22_list.append(c22_dict["values"])

        X_c22 = np.array(c22_list)
        np.nan_to_num(X_c22, False, 0, 0, 0)

        return self.classifier.predict_proba(X_c22)
