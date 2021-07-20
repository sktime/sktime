# -*- coding: utf-8 -*-
"""Catch22 Forest Classifier.

A forest classifier based on catch22 features
"""

__author__ = ["Matthew Middlehurst"]
__all__ = ["Catch22Classifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import class_distribution

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.catch22 import Catch22
from sktime.utils.validation.panel import check_X_y, check_X


class Catch22Classifier(BaseClassifier):
    """Canonical Time-series Characteristics (catch22).

    Overview: Input n series length m. Transforms series into the 22 catch22
    features [1] extracted from the hctsa toolbox[2] and builds a random forest
    classifier on them.

    Parameters
    ----------
    n_estimators            : int, number of trees in the random forest (default=200)
    outlier_norm            : boolean, normalise each series for the outlier catch22
    features which can take a while to process otherwise (default=False)
    n_jobs                  : int or None, number of jobs to run in parallel (default=1)
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    classifier              : trained random forest classifier

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
    catch22 package C, MATLAB and wrapped Python implementations:
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

    def __init__(
        self,
        outlier_norm=False,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.outlier_norm = outlier_norm

        self.estimator = (
            RandomForestClassifier(n_estimators=200, n_jobs=n_jobs)
            if estimator is None
            else estimator
        )

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.transformer_ = None
        self.estimator_ = None
        self.classes_ = []
        self.n_classes = 0
        super(Catch22Classifier, self).__init__()

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
        X, y = check_X_y(X, y)
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.n_classes = np.unique(y).shape[0]

        self.transformer_ = Catch22(outlier_norm=self.outlier_norm)

        self.estimator_ = _clone_estimator(self.estimator, self.random_state)
        X_t = self.transformer_.fit_transform(X, y)
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)
        self.estimator_.fit(X_t, y)

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
        X = check_X(X)

        X_t = self.transformer_.transform(X)
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)
        return self.estimator_.predict(X_t)

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
        X = check_X(X)

        X_t = self.transformer_.transform(X)
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)

        m = getattr(self.estimator_, "predict_proba", None)
        if callable(m):
            return self.estimator_.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes))
            preds = self.estimator_.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists
