#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
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
    "PolynomialTrendForecaster",
    "TransformedTargetForecaster",
    "MultiplexForecaster",
    "Deseasonalizer",
    "ReducedForecaster",
    "EnsembleForecaster",
    "Detrender",
    "sMAPE",
    "MASE",
    "smape_loss",
    "mase_loss",
    "pd",
    "np",
    "plot_series",
    "NormalHedgeEnsemble",
    "NNLSEnsemble",
    "OnlineEnsembleForecaster",
    "evaluate",
]

import numpy as np
import pandas as pd

from sktime.datasets import load_airline
from sktime.datasets import load_longley
from sktime.datasets import load_lynx
from sktime.datasets import load_shampoo_sales
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose import ReducedForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.compose import MultiplexForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import CutoffSplitter
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.online_learning._online_ensemble import (
    OnlineEnsembleForecaster,
)
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NNLSEnsemble,
)
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NormalHedgeEnsemble,
)
from sktime.performance_metrics.forecasting import MASE
from sktime.performance_metrics.forecasting import mase_loss
from sktime.performance_metrics.forecasting import sMAPE
from sktime.performance_metrics.forecasting import smape_loss
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.utils.plotting import plot_series
