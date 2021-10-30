# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for adapting other forecasters to sktime framework."""

__author__ = ["Markus Löning"]
__all__ = [
    "_ProphetAdapter",
    "_StatsModelsAdapter",
    "_TbatsAdapter",
    "_PmdArimaAdapter",
]

from sktime.forecasting.base.adapters._fbprophet import _ProphetAdapter
from sktime.forecasting.base.adapters._statsmodels import _StatsModelsAdapter
from sktime.forecasting.base.adapters._tbats import _TbatsAdapter
from sktime.forecasting.base.adapters._pmdarima import _PmdArimaAdapter
