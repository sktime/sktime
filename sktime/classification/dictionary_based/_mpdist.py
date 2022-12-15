# -*- coding: utf-8 -*-
"""HYDRA classifier.

HYDRA: Competing convolutional kernels for fast and accurate time series classification
By Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
https://arxiv.org/abs/2203.13652
"""

__author__ = ["patrickzib"]
__all__ = ["MPDist"]

import numpy as np
import stumpy
from sklearn.metrics import pairwise

from sktime.classification.base import BaseClassifier
from sktime.utils.validation.panel import check_X


class MPDist(BaseClassifier):
    """Sktime-MPDist k-NN classifier-adaptor."""

    _tags = {
        "capability:multithreading": True,
        "classifier_type": "distance",
    }

    def __init__(self, window=10, n_jobs=1):
        self.window = window
        self.n_jobs = n_jobs
        super(MPDist, self).__init__()

    def _fit(self, X, y):
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        self._X_train = X
        self._y_train = y

    def _predict(self, X) -> np.ndarray:
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        distance_matrix = pairwise.pairwise_distances(
            X,
            self._X_train,
            metric=(lambda x, y: stumpy.mpdist(x, y, self.window)),
            n_jobs=self.n_jobs,
        )

        return self._y_train[np.argmin(distance_matrix, axis=1)]
