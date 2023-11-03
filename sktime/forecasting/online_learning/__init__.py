# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements algorithms for creating online ensembles of forecasters."""

__author__ = ["magittan"]

__all__ = [
    "NormalHedgeEnsemble",
    "NNLSEnsemble",
    "OnlineEnsembleForecaster",
]

from sktime.forecasting.online_learning._online_ensemble import OnlineEnsembleForecaster
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NNLSEnsemble,
    NormalHedgeEnsemble,
)
