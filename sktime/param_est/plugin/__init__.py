# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Plugin composites for parameter estimators."""

__all__ = [
    "PluginParamsForecaster",
    "PluginParamsTransformer",
]

from sktime.param_est.plugin._forecaster import PluginParamsForecaster
from sktime.param_est.plugin._transformer import PluginParamsTransformer
