#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for MultiplexForecaster and associated dunders."""

__author__ = ["aiwalter", "miraep8"]

from sktime.datasets import load_shampoo_sales
from sktime.forecasting.all import (
    AutoETS,
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    MultiplexForecaster,
    NaiveForecaster,
)
from sktime.forecasting.model_evaluation import evaluate
from sktime.utils.validation.forecasting import check_scoring


def _score_forecasters(forecasters, cv, y):
    """Will evaluate all the forecasters on y and return the name of best."""
    scoring = check_scoring(None)
    scoring_name = f"test_{scoring.name}"
    score = None
    for name, forecaster in forecasters:
        results = evaluate(forecaster, cv, y)
        results = results.mean()
        new_score = float(results[scoring_name])
        if not score or new_score < score:
            score = new_score
            best_name = name
    return best_name


def test_multiplex():
    """Test results of MultiplexForecaster.

    Because MultiplexForecaster should essentially just be a framework for
    comparing different models/selecting which model does best, we can check
    that it performs as expected.
    """
    y = load_shampoo_sales()
    forecasters = [
        ("ets", AutoETS()),
        ("naive", NaiveForecaster()),
    ]
    multiplex_forecaster = MultiplexForecaster(forecasters=forecasters)
    forecaster_names = multiplex_forecaster.get_forecaster_names()
    # check that get_forcaster_names performs as expected:
    assert forecaster_names == [name for name, _ in forecasters]
    cv = ExpandingWindowSplitter(start_with_window=True, step_length=12)
    gscv = ForecastingGridSearchCV(
        cv=cv,
        param_grid={"selected_forecaster": forecaster_names},
        forecaster=multiplex_forecaster,
    )
    gscv.fit(y)
    gscv_best_name = gscv.best_forecaster_.selected_forecaster
    best_name = _score_forecasters(forecasters, cv, y)
    assert gscv_best_name == best_name
