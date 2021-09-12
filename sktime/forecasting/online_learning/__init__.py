# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implments algorithms for creating online ensembles of forecasters."""

__author__ = ["William Zheng"]

__all__ = [
    "NormalHedgeEnsemble",
    "NNLSEnsemble",
    "OnlineEnsembleForecaster",
]

from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NormalHedgeEnsemble,
    NNLSEnsemble,
)
from sktime.forecasting.online_learning._online_ensemble import (
    OnlineEnsembleForecaster,
)
