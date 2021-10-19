#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics to assess performance on forecasting task.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

__author__ = ["Markus Löning", "Tomasz Chodakowski", "Martin Walter"]
__all__ = [
    "make_forecasting_scorer",
    "MeanAbsoluteScaledError",
    "MedianAbsoluteScaledError",
    "MeanSquaredScaledError",
    "MedianSquaredScaledError",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "MedianAbsoluteError",
    "MedianSquaredError",
    "MeanAbsolutePercentageError",
    "MedianAbsolutePercentageError",
    "MeanSquaredPercentageError",
    "MedianSquaredPercentageError",
    "MeanRelativeAbsoluteError",
    "MedianRelativeAbsoluteError",
    "GeometricMeanRelativeAbsoluteError",
    "GeometricMeanRelativeSquaredError",
    "MeanAsymmetricError",
    "RelativeLoss",
    "mean_absolute_scaled_error",
    "median_absolute_scaled_error",
    "mean_squared_scaled_error",
    "median_squared_scaled_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "median_squared_error",
    "mean_absolute_percentage_error",
    "median_absolute_percentage_error",
    "mean_squared_percentage_error",
    "median_squared_percentage_error",
    "mean_relative_absolute_error",
    "median_relative_absolute_error",
    "geometric_mean_relative_absolute_error",
    "geometric_mean_relative_squared_error",
    "mean_asymmetric_error",
    "relative_loss",
]

from sktime.performance_metrics.forecasting._classes import (
    make_forecasting_scorer,
    MeanAbsoluteScaledError,
    MedianAbsoluteScaledError,
    MeanSquaredScaledError,
    MedianSquaredScaledError,
    MeanAbsoluteError,
    MeanSquaredError,
    MedianAbsoluteError,
    MedianSquaredError,
    MeanAbsolutePercentageError,
    MedianAbsolutePercentageError,
    MeanSquaredPercentageError,
    MedianSquaredPercentageError,
    MeanRelativeAbsoluteError,
    MedianRelativeAbsoluteError,
    GeometricMeanRelativeAbsoluteError,
    GeometricMeanRelativeSquaredError,
    MeanAsymmetricError,
    RelativeLoss,
)
from sktime.performance_metrics.forecasting._functions import (
    mean_absolute_scaled_error,
    median_absolute_scaled_error,
    mean_squared_scaled_error,
    median_squared_scaled_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    median_squared_error,
    mean_absolute_percentage_error,
    median_absolute_percentage_error,
    mean_squared_percentage_error,
    median_squared_percentage_error,
    mean_relative_absolute_error,
    median_relative_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    mean_asymmetric_error,
    relative_loss,
)
