# -*- coding: utf-8 -*-
"""Probability Threshold Early Classifier.

"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ProbabilityThresholdEarlyClassifier"]

from sktime.classification.base import BaseClassifier


class ProbabilityThresholdEarlyClassifier(BaseClassifier):
    """Probability Threshold Early Classifier."""

    _tags = {}

    def __init__(
        self,
        outlier_norm=False,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.outlier_norm = outlier_norm
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._transformer = None
        self._estimator = None

        super(ProbabilityThresholdEarlyClassifier, self).__init__()
