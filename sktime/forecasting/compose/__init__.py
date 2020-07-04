#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD

__author__ = ["Markus Löning"]

from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.compose._pipeline import TransformedTargetForecaster
from sktime.forecasting.compose._reduce import DirectRegressionForecaster
from sktime.forecasting.compose._reduce import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import ReducedRegressionForecaster
from sktime.forecasting.compose._reduce import ReducedTimeSeriesRegressionForecaster
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

__all__ = [
    "EnsembleForecaster",
    "TransformedTargetForecaster",
    "DirectRegressionForecaster",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "ReducedRegressionForecaster",
    "ReducedTimeSeriesRegressionForecaster",
    "StackingForecaster",
]

from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.compose._pipeline import TransformedTargetForecaster
from sktime.forecasting.compose._reduce import DirectRegressionForecaster
from sktime.forecasting.compose._reduce import \
    DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveRegressionForecaster
from sktime.forecasting.compose._reduce import \
    RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import ReducedRegressionForecaster
from sktime.forecasting.compose._reduce import \
    ReducedTimeSeriesRegressionForecaster
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
from sktime.forecasting.compose._stack import StackingForecaster
