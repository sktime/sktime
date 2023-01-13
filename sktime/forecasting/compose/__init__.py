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
    "ForecastX",
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
    "make_reduction",
    "BaggingForecaster",
    "ForecastByLevel",
    "Permute",
]

from sktime.forecasting.compose._bagging import BaggingForecaster
from sktime.forecasting.compose._column_ensemble import ColumnEnsembleForecaster
from sktime.forecasting.compose._ensemble import (
    AutoEnsembleForecaster,
    EnsembleForecaster,
)
from sktime.forecasting.compose._grouped import ForecastByLevel
from sktime.forecasting.compose._multiplexer import MultiplexForecaster
from sktime.forecasting.compose._pipeline import (
    ForecastingPipeline,
    ForecastX,
    Permute,
    TransformedTargetForecaster,
)
from sktime.forecasting.compose._reduce import (
    DirectTabularRegressionForecaster,
    DirectTimeSeriesRegressionForecaster,
    DirRecTabularRegressionForecaster,
    DirRecTimeSeriesRegressionForecaster,
    MultioutputTabularRegressionForecaster,
    MultioutputTimeSeriesRegressionForecaster,
    RecursiveTabularRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster,
    make_reduction,
)
from sktime.forecasting.compose._stack import StackingForecaster
