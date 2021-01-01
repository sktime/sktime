# -*- coding: utf-8 -*-
"""
"""

__author__ = "Matthew Middlehurst"
__all__ = ["ROCKETClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import Rocket
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class ROCKETClassifier(BaseClassifier):
    """

    Parameters
    ----------
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------

    Notes
    -----

    """

    # Capability tags
    _tags = {"multivariate": True, "unequal_length": False, "missing_values": False}

    def __init__(
        self,
        num_kernels=10000,
        ensemble=False,
        ensemble_size=25,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.ensemble = ensemble
        self.ensemble_size = ensemble_size
        self.random_state = random_state

        self.classifiers = []
        self.weights = []
        self.weight_sum = 0

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(ROCKETClassifier, self).__init__()

    def fit(self, X, y):
        """

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

        if self.ensemble:
            for i in range(self.ensemble_size):
                rocket_pipeline = make_pipeline(
                    Rocket(num_kernels=self.num_kernels,
                           random_state=self.random_state),
                    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
                )
                rocket_pipeline.fit(X, y)
                self.classifiers.append(rocket_pipeline)
                self.weights.append(rocket_pipeline.steps[1][1].best_score_)
                self.weight_sum = self.weight_sum + self.weights[i]
        else:
            rocket_pipeline = make_pipeline(
                Rocket(num_kernels=self.num_kernels, random_state=self.random_state),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
            )
            rocket_pipeline.fit(X, y)
            self.classifiers.append(rocket_pipeline)

        self._is_fitted = True
        return self

    def predict(self, X):
        if self.ensemble:
            rng = check_random_state(self.random_state)
            return np.array(
                [
                    self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                    for prob in self.predict_proba(X)
                ]
            )
        else:
            return self.classifiers[0].predict(X)

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X)

        if self.ensemble:
            sums = np.zeros((X.shape[0], self.n_classes))

            for n, clf in enumerate(self.classifiers):
                preds = clf.predict(X)
                for i in range(0, X.shape[0]):
                    sums[i, self.class_dictionary[preds[i]]] += self.weights[n]

            dists = sums / (np.ones(self.n_classes) * self.weight_sum)
        else:
            dists = np.zeros((X.shape[0], self.n_classes))
            preds = self.classifiers[0].predict(X)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
