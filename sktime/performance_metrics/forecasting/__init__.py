#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD

__author__ = ["Markus Löning"]

from sktime.performance_metrics.forecasting._functions import mase_loss, smape_loss
from sktime.performance_metrics.forecasting._classes import MASE, sMAPE, make_forecasting_scorer
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "MASE",
    "sMAPE",
    "mase_loss",
    "smape_loss",
    "make_forecasting_scorer"
]

from sktime.performance_metrics.forecasting._classes import MASE
from sktime.performance_metrics.forecasting._classes import \
    make_forecasting_scorer
from sktime.performance_metrics.forecasting._classes import sMAPE
from sktime.performance_metrics.forecasting._functions import mase_loss
from sktime.performance_metrics.forecasting._functions import smape_loss
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
