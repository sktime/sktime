#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics to assess performance on forecasting task.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

__author__ = ["mloning", "tch", "RNKuhns", "fkiraly", "aiwalter", "markussagen"]

__all__ = [
    "make_forecasting_scorer",
    "GeometricMeanAbsoluteError",
    "GeometricMeanSquaredError",
    "GeometricMeanRelativeAbsoluteError",
    "GeometricMeanRelativeSquaredError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanAbsoluteScaledError",
    "MeanAsymmetricError",
    "MeanLinexError",
    "MeanRelativeAbsoluteError",
    "MeanSquaredError",
    "MeanSquaredErrorPercentage",
    "MeanSquaredPercentageError",
    "MeanSquaredScaledError",
    "MedianAbsoluteError",
    "MedianAbsolutePercentageError",
    "MedianAbsoluteScaledError",
    "MedianRelativeAbsoluteError",
    "MedianSquaredError",
    "MedianSquaredPercentageError",
    "MedianSquaredScaledError",
    "RelativeLoss",
    "mean_absolute_scaled_error",
    "median_absolute_scaled_error",
    "mean_squared_scaled_error",
    "median_squared_scaled_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "median_squared_error",
    "geometric_mean_absolute_error",
    "geometric_mean_squared_error",
    "mean_absolute_percentage_error",
    "median_absolute_percentage_error",
    "mean_squared_percentage_error",
    "median_squared_percentage_error",
    "mean_relative_absolute_error",
    "median_relative_absolute_error",
    "geometric_mean_relative_absolute_error",
    "geometric_mean_relative_squared_error",
    "mean_asymmetric_error",
    "mean_linex_error",
    "relative_loss",
]

from sktime.performance_metrics.forecasting._base import make_forecasting_scorer
from sktime.performance_metrics.forecasting._functions import (
    geometric_mean_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    geometric_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_asymmetric_error,
    mean_linex_error,
    mean_relative_absolute_error,
    mean_squared_error,
    mean_squared_percentage_error,
    mean_squared_scaled_error,
    median_absolute_error,
    median_absolute_percentage_error,
    median_absolute_scaled_error,
    median_relative_absolute_error,
    median_squared_error,
    median_squared_percentage_error,
    median_squared_scaled_error,
    relative_loss,
)
from sktime.performance_metrics.forecasting._gmae import GeometricMeanAbsoluteError
from sktime.performance_metrics.forecasting._gmrelae import (
    GeometricMeanRelativeAbsoluteError,
)
from sktime.performance_metrics.forecasting._gmrelse import (
    GeometricMeanRelativeSquaredError,
)
from sktime.performance_metrics.forecasting._gmse import GeometricMeanSquaredError
from sktime.performance_metrics.forecasting._mae import MeanAbsoluteError
from sktime.performance_metrics.forecasting._mape import MeanAbsolutePercentageError
from sktime.performance_metrics.forecasting._mase import MeanAbsoluteScaledError
from sktime.performance_metrics.forecasting._masyme import MeanAsymmetricError
from sktime.performance_metrics.forecasting._medae import MedianAbsoluteError
from sktime.performance_metrics.forecasting._medape import MedianAbsolutePercentageError
from sktime.performance_metrics.forecasting._medase import MedianAbsoluteScaledError
from sktime.performance_metrics.forecasting._medrelae import MedianRelativeAbsoluteError
from sktime.performance_metrics.forecasting._medse import MedianSquaredError
from sktime.performance_metrics.forecasting._medspe import MedianSquaredPercentageError
from sktime.performance_metrics.forecasting._medsse import MedianSquaredScaledError
from sktime.performance_metrics.forecasting._mlinex import MeanLinexError
from sktime.performance_metrics.forecasting._mrelae import MeanRelativeAbsoluteError
from sktime.performance_metrics.forecasting._mse import MeanSquaredError
from sktime.performance_metrics.forecasting._msep import MeanSquaredErrorPercentage
from sktime.performance_metrics.forecasting._mspe import MeanSquaredPercentageError
from sktime.performance_metrics.forecasting._msse import MeanSquaredScaledError
from sktime.performance_metrics.forecasting._rell import RelativeLoss
