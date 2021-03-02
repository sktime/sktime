# -*- coding: utf-8 -*-
"""
    Time Series Forest Regressor (TSF).
"""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti", "kanand77"]
__all__ = ["TimeSeriesForestRegressor"]

from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sktime.regression.base import BaseRegressor
from sktime.series_as_features.base.estimators.interval_based._tsf import (
    BaseTimeSeriesForest,
)


class TimeSeriesForestRegressor(BaseTimeSeriesForest, ForestRegressor, BaseRegressor):

    base_estimator = DecisionTreeRegressor(criterion="entropy")
