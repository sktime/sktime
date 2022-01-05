# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Import all time series forecasting functionality available in sktime."""

__author__ = ["mloning"]
__all__ = [
    "ForecastingHorizon",
    "load_lynx",
    "load_longley",
    "load_airline",
    "load_shampoo_sales",
    "CutoffSplitter",
    "ForecastingGridSearchCV",
    "ForecastingRandomizedSearchCV",
    "SlidingWindowSplitter",
    "SingleWindowSplitter",
    "ExpandingWindowSplitter",
    "temporal_train_test_split",
    "NaiveForecaster",
    "ExponentialSmoothing",
    "ThetaForecaster",
    "AutoARIMA",
    "ARIMA",
    "AutoETS",
    "Prophet",
    "TrendForecaster",
    "PolynomialTrendForecaster",
    "TransformedTargetForecaster",
    "MultiplexForecaster",
    "Deseasonalizer",
    "EnsembleForecaster",
    "Detrender",
    "pd",
    "np",
    "plot_series",
    "NormalHedgeEnsemble",
    "NNLSEnsemble",
    "OnlineEnsembleForecaster",
    "evaluate",
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
    "relative_loss",
    "mean_asymmetric_error",
]

import numpy as np
import pandas as pd

from sktime.datasets import load_airline, load_longley, load_lynx, load_shampoo_sales
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    MultiplexForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.online_learning._online_ensemble import OnlineEnsembleForecaster
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NNLSEnsemble,
    NormalHedgeEnsemble,
)
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
from sktime.performance_metrics.forecasting import (
    GeometricMeanRelativeAbsoluteError,
    GeometricMeanRelativeSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanAsymmetricError,
    MeanRelativeAbsoluteError,
    MeanSquaredError,
    MeanSquaredPercentageError,
    MeanSquaredScaledError,
    MedianAbsoluteError,
    MedianAbsolutePercentageError,
    MedianAbsoluteScaledError,
    MedianRelativeAbsoluteError,
    MedianSquaredError,
    MedianSquaredPercentageError,
    MedianSquaredScaledError,
    RelativeLoss,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    make_forecasting_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_asymmetric_error,
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
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series
