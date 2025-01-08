#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements composite forecasters."""

__all__ = [
    "HierarchyEnsembleForecaster",
    "ColumnEnsembleForecaster",
    "EnsembleForecaster",
    "TransformSelectForecaster",
    "FallbackForecaster",
    "AutoEnsembleForecaster",
    "TransformedTargetForecaster",
    "ForecastingPipeline",
    "ForecastX",
    "GroupbyCategoryForecaster",
    "DirectTabularRegressionForecaster",
    "DirectTimeSeriesRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "DirRecTabularRegressionForecaster",
    "DirRecTimeSeriesRegressionForecaster",
    "DirectReductionForecaster",
    "RecursiveReductionForecaster",
    "StackingForecaster",
    "MultiplexForecaster",
    "make_reduction",
    "BaggingForecaster",
    "FhPlexForecaster",
    "ForecastByLevel",
    "Permute",
    "YfromX",
    "SkforecastAutoreg",
    "IgnoreX",
]

from sktime.forecasting.compose._bagging import BaggingForecaster
from sktime.forecasting.compose._column_ensemble import ColumnEnsembleForecaster
from sktime.forecasting.compose._ensemble import (
    AutoEnsembleForecaster,
    EnsembleForecaster,
)
from sktime.forecasting.compose._fallback import FallbackForecaster
from sktime.forecasting.compose._fhplex import FhPlexForecaster
from sktime.forecasting.compose._grouped import (
    ForecastByLevel,
    GroupbyCategoryForecaster,
)
from sktime.forecasting.compose._hierarchy_ensemble import HierarchyEnsembleForecaster
from sktime.forecasting.compose._ignore_x import IgnoreX
from sktime.forecasting.compose._multiplexer import MultiplexForecaster
from sktime.forecasting.compose._pipeline import (
    ForecastingPipeline,
    ForecastX,
    Permute,
    TransformedTargetForecaster,
)
from sktime.forecasting.compose._reduce import (
    DirectReductionForecaster,
    DirectTabularRegressionForecaster,
    DirectTimeSeriesRegressionForecaster,
    DirRecTabularRegressionForecaster,
    DirRecTimeSeriesRegressionForecaster,
    MultioutputTabularRegressionForecaster,
    MultioutputTimeSeriesRegressionForecaster,
    RecursiveReductionForecaster,
    RecursiveTabularRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster,
    YfromX,
    make_reduction,
)
from sktime.forecasting.compose._skforecast_reduce import (
    SkforecastAutoreg,
    SkforecastRecursive,
)
from sktime.forecasting.compose._stack import StackingForecaster
from sktime.forecasting.compose._transform_select_forecaster import (
    TransformSelectForecaster,
)
