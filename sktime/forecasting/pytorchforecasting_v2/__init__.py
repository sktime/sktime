# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to pytorch-forecasting v2 estimators.

This package contains sktime estimators that wrap PTF v2 models.
"""

from sktime.forecasting.pytorchforecasting_v2.tft import PytorchForecastingTFTV2

__all__ = ["PytorchForecastingTFTV2"]
__author__ = ["vedantag17"]
