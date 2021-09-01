#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements composite forecasters."""

__author__ = ["mloning"]

__all__ = [
    "ColumnEnsembleForecaster",
    "EnsembleForecaster",
    "AutoEnsembleForecaster",
    "TransformedTargetForecaster",
    "ForecastingPipeline",
    "DirectTabularRegressionForecaster",
    "DirectTimeSeriesRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "DirRecTabularRegressionForecaster",
    "DirRecTimeSeriesRegressionForecaster",
    "StackingForecaster",
    "MultiplexForecaster",
    "ReducedForecaster",
    "make_reduction",
]

from sktime.forecasting.compose._column_ensemble import ColumnEnsembleForecaster
from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.compose._ensemble import AutoEnsembleForecaster
from sktime.forecasting.compose._pipeline import TransformedTargetForecaster
from sktime.forecasting.compose._pipeline import ForecastingPipeline
from sktime.forecasting.compose._reduce import DirRecTabularRegressionForecaster
from sktime.forecasting.compose._reduce import DirRecTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import DirectTabularRegressionForecaster
from sktime.forecasting.compose._reduce import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import MultioutputTabularRegressionForecaster
from sktime.forecasting.compose._reduce import MultioutputTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveTabularRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose._stack import StackingForecaster
from sktime.forecasting.compose._multiplexer import MultiplexForecaster
from sktime.forecasting.compose._reduce import ReducedForecaster
from sktime.forecasting.compose._reduce import make_reduction
