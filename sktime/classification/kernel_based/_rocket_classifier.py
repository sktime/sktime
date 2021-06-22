# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (ROCKET)."""

__author__ = "Matthew Middlehurst"
__all__ = ["ROCKETClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import Rocket
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class ROCKETClassifier(BaseClassifier):
    """Classifier wrapped for the ROCKET transformer using RidgeClassifierCV.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=10,000)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    classifier              : ROCKET classifier
    n_classes               : extracted from the data

    Notes
    -----
    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Francois and Webb,
      Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java
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
        num_kernels=10000,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifier = None

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(ROCKETClassifier, self).__init__()

    def fit(self, X, y):
        """Build a pipeline containing the ROCKET transformer and RidgeClassifierCV.

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

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifier = rocket_pipeline = make_pipeline(
            Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
        rocket_pipeline.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Find predictions for all cases in X.

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
        self.check_is_fitted()
        X = check_X(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
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
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        self.check_is_fitted()
        X = check_X(X)

        dists = np.zeros((X.shape[0], self.n_classes))
        preds = self.classifier.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
