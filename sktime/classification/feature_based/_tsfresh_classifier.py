# -*- coding: utf-8 -*-
"""TSFresh Classifier.

Pipeline classifier using the TSFresh transformer and an estimator.
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
    """Canonical Time-series Characteristics (catch22) classifier.

    This classifier simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.


    Parameters
    ----------
    outlier_norm : bool, default=False
        Normalise each series during the two outlier catch22 features, which can take a
        while to process for large values
    estimator : sklearn classifier, default=RandomForestClassifier
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    n_jobs= : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    :py:class:`TSFreshFeatureExtractor`

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh–a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843

    Example
    -------
    >>> from sktime.classification.feature_based import TSFreshClassifier
    >>> from sktime.datasets import load_italy_power_demand
    >>> X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    >>> X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    >>> clf = TSFreshClassifier()
    >>> clf.fit(X_train, y_train)
    TSFreshClassifier(...)
    >>> y_pred = clf.predict(X_test)
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
