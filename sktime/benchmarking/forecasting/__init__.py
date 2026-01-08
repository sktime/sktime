# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecasting benchmarking utilities."""

__all__ = [
    "TimeSeriesSimulator",
    "ModelComparisonBenchmark",
]

from sktime.benchmarking.forecasting._benchmark import ModelComparisonBenchmark
from sktime.benchmarking.forecasting._simulator import TimeSeriesSimulator
