# -*- coding: utf-8 -*-
"""
"""

__author__ = ["Matthew Middlehurst"]
__all__ = ["TSFreshClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import class_distribution

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
from sktime.utils.validation.panel import check_X, check_X_y


class TSFreshClassifier(BaseClassifier):
    """"""

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
        default_fc_parameters="efficient",
        relevant_feature_extractor=False,
        estimator=None,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.relevant_feature_extractor = relevant_feature_extractor

        self.estimator = (
            RandomForestClassifier(n_estimators=200, n_jobs=n_jobs)
            if estimator is None
            else estimator
        )

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self.transformer_ = None
        self.estimator_ = None
        self.classes_ = []
        self.n_classes = 0
        super(TSFreshClassifier, self).__init__()

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

        self.transformer_ = (
            TSFreshRelevantFeatureExtractor(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self.n_jobs,
                chunksize=self.chunksize,
            )
            if self.relevant_feature_extractor
            else TSFreshFeatureExtractor(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self.n_jobs,
                chunksize=self.chunksize,
            )
        )

        if self.verbose < 2:
            self.transformer_.show_warnings = False
            if self.verbose < 1:
                self.transformer_.disable_progressbar = True

        self.estimator_ = _clone_estimator(self.estimator, self.random_state)
        X_t = self.transformer_.fit_transform(X, y)
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

        return self.estimator_.predict(self.transformer_.transform(X))

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

        m = getattr(self.estimator_, "predict_proba", None)
        if callable(m):
            return self.estimator_.predict_proba(self.transformer_.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes))
            preds = self.estimator_.predict(self.transformer_.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists
