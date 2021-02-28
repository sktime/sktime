# -*- coding: utf-8 -*-
"""
    Time Series Forest Regressor (TSF).
"""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti"]
__all__ = ["TimeSeriesForest"]

from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sktime.classification.base import BaseClassifier
from sktime.series_as_features.base.estimators.interval_based._tsf import (
    BaseTimeSeriesForest,
)


class TimeSeriesForest(BaseTimeSeriesForest, ForestClassifier, BaseClassifier):

    # Capabilities: data types this classifier can handle
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        n_jobs=1,
        random_state=None,
    ):
        super(TimeSeriesForest, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_estimators,
        )

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.n_jobs = n_jobs
        # The following set in method fit
        self.n_classes = 0
        self.series_length = 0
        self.n_intervals = 0
        self.estimators_ = []
        self.intervals_ = []
        self.classes_ = []

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False
