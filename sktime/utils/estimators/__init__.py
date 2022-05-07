# -*- coding: utf-8 -*-
"""Utility estimator classes for testing and debugging."""

__author__ = ["ltsaprounis"]

__all__ = [
    "MockForecaster",
    "MockUnivariateForecasterLogger",
    "make_mock_estimator",
]

from sktime.utils.estimators._base import make_mock_estimator
from sktime.utils.estimators._forecasters import (
    MockForecaster,
    MockUnivariateForecasterLogger,
)
