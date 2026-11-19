"""Simons: small, open time series foundation models for probabilistic forecasting.

Source: https://github.com/Abel-ai-lab/Simons
Vendored at commit 0d0042f8da45140172bd36980be08654c0017d8b under Apache-2.0.
Model weights are not bundled; see the Simons model license.
sktime migration: 2026-11-19.
Original authors: Abel AI Lab (https://github.com/Abel-ai-lab),
Shu-Wan (https://github.com/Shu-Wan), and ruizhu-lgtm
(https://github.com/ruizhu-lgtm).
"""

from .config import SimonsConfig
from .covariates import encode_categorical, prepare_covariates
from .forecast import QUANTILES, forecast, forecast_items, forecast_series
from .model import Simons

__all__ = [
    "QUANTILES",
    "Simons",
    "SimonsConfig",
    "encode_categorical",
    "forecast",
    "forecast_items",
    "forecast_series",
    "prepare_covariates",
]
__version__ = "0.1.2"
