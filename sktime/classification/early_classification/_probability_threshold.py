# -*- coding: utf-8 -*-
"""Probability Threshold Early Classifier.

hi
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ProbabilityThresholdEarlyClassifier"]

import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.feature_based import Catch22Classifier


class ProbabilityThresholdEarlyClassifier(BaseClassifier):
    """Probability Threshold Early Classifier.

    #points not inclusive

    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        probability_threshold=0.85,
        consecutive_predictions=1,
        estimator=None,
        classification_points=None,
        n_jobs=1,
        random_state=None,
    ):
        self.probability_threshold = probability_threshold
        self.consecutive_predictions = consecutive_predictions
        self.estimator = estimator
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []

        super(ProbabilityThresholdEarlyClassifier, self).__init__()

    def _fit(self, X, y):
        _, _, series_length = X.shape

        self._classification_points = (
            copy.deepcopy(self.classification_points)
            if self.classification_points is not None
            else [round(series_length / i) for i in range(1, 21)]
        )
        # remove duplicates
        self._classification_points = list(set(self._classification_points))
        # remove classification points that are less than 3
        self._classification_points = [i for i in self._classification_points if i >= 3]

        fit = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
            )
            for i in range(len(self._classification_points))
        )

        self._estimators = zip(*fit)

        return self

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X):
        return 0

    def _fit_estimator(self, X, y, i):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (i + 1)
        rng = check_random_state(rs)

        estimator = _clone_estimator(
            Catch22Classifier() if self.estimator is None else self.estimator,
            rng,
        )

        estimator.fit(X[:, :, : self._classification_points[i]], y)

        return estimator

    def _predict_proba_for_estimator(self, X):
        return 0
