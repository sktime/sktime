# -*- coding: utf-8 -*-
"""
    Time Series Forest Regressor (TSF).
"""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti", "kanand77"]
__all__ = ["TimeSeriesForestClassifier"]

from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sktime.classification.base import BaseClassifier
from sktime.series_as_features.base.estimators.interval_based._tsf import (
    BaseTimeSeriesForest,
)


class TimeSeriesForestClassifier(
    BaseTimeSeriesForest, ForestClassifier, BaseClassifier
):

    base_estimator = DecisionTreeClassifier(criterion="entropy")
