# Copyright 2026 Abel AI Lab.
# Source: https://github.com/Abel-ai-lab/Simons
# Apache-2.0; vendored from Abel-ai-lab/Simons at commit 8e338e915d26c8960713889ef12ab1cf5c08d591.
# Model weights are not bundled; see the Simons model license.
# ruff: noqa

"""Simons: small, open time series foundation models for probabilistic forecasting."""

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
__version__ = "0.1.1"
