# -*- coding: utf-8 -*-
"""Register estimators to compete in benchmarking."""

from sktime.benchmarking import estimator_registry
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster

estimator_registry.register(
    id="Naive-v1",
    entry_point=NaiveForecaster,
    kwargs={"strategy": "mean", "sp": 12},
)

estimator_registry.register(
    id="ETS-v1",
    entry_point=AutoETS,
    kwargs={"auto": "True", "n_jobs": -1, "sp": 12},
)
