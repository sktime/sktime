#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for MultiplexForecaster and associated dunders."""

__author__ = ["miraep8"]

import pytest

from sktime.datasets import load_shampoo_sales
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import MultiplexForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.validation._dependencies import _check_estimator_deps
from sktime.utils.validation.forecasting import check_scoring


def _score_forecasters(forecasters, cv, y):
    """Will evaluate all the forecasters on y and return the name of best."""
    scoring = check_scoring(None)
    scoring_name = f"test_{scoring.name}"
    score = None
    for name, forecaster in forecasters:
        results = evaluate(forecaster, cv, y)
        results = results.mean(numeric_only=True)
        new_score = float(results[scoring_name])
        if not score or new_score < score:
            score = new_score
            best_name = name
    return best_name


@pytest.mark.skipif(
    not _check_estimator_deps(ThetaForecaster, severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_multiplex_forecaster_alone():
    """Test results of MultiplexForecaster.

    Because MultiplexForecaster is in many ways a wrapper for an underlying
    forecaster - we can confirm that if the selected_forecaster is set that the
    MultiplexForecaster performs as expected.
    """
    from numpy.testing import assert_array_equal

    y = load_shampoo_sales()
    # Note - we select two forecasters which are deterministic.
    forecaster_tuples = [
        ("naive", NaiveForecaster()),
        ("theta", ThetaForecaster()),
    ]
    forecaster_names = [name for name, _ in forecaster_tuples]
    forecasters = [forecaster for _, forecaster in forecaster_tuples]
    multiplex_forecaster = MultiplexForecaster(forecasters=forecaster_tuples)
    fh_test = [1, 2, 3]
    # for each of the forecasters - check that the wrapped forecaster predictions
    # agree with the unwrapped forecaster predictions!
    for ind, name in enumerate(forecaster_names):
        # make a copy to ensure we don't reference the same objectL
        test_forecaster = forecasters[ind].clone()
        test_forecaster.fit(y)
        multiplex_forecaster.selected_forecaster = name
        # Note- MultiplexForecaster will make a copy of the forecaster before fitting.
        multiplex_forecaster.fit(y)
        y_pred_indiv = test_forecaster.predict(fh=fh_test)
        y_pred_multi = multiplex_forecaster.predict(fh=fh_test)
        assert_array_equal(y_pred_indiv, y_pred_multi)


def test_multiplex_with_grid_search():
    """Test MultiplexForecaster perfromas as expected with ForecastingGridSearchCV.

    Because the typical use case of MultiplexForecaster is to use it with the
    ForecastingGridSearchCV forecaster - here we simply test that the best
    "selected_forecaster" for MultiplexForecaster found using ForecastingGridSearchCV
    is the same forecaster we would find if we evaluated all the forecasters in
    MultiplexForecaster independently.
    """
    y = load_shampoo_sales()
    forecasters = [
        ("naive1", NaiveForecaster()),
        ("naive2", NaiveForecaster(strategy="mean")),
    ]
    multiplex_forecaster = MultiplexForecaster(forecasters=forecasters)
    forecaster_names = [name for name, _ in forecasters]
    cv = ExpandingWindowSplitter(step_length=12)
    gscv = ForecastingGridSearchCV(
        cv=cv,
        param_grid={"selected_forecaster": forecaster_names},
        forecaster=multiplex_forecaster,
    )
    gscv.fit(y)
    gscv_best_name = gscv.best_forecaster_.selected_forecaster
    best_name = _score_forecasters(forecasters, cv, y)
    assert gscv_best_name == best_name


@pytest.mark.skipif(
    not _check_estimator_deps(AutoARIMA, severity="none"),
    reason="skip test if required soft dependency for AutoARIMA not available",
)
def test_multiplex_or_dunder():
    """Test that the MultiplexForecaster magic "|" dunder methodbahves as expected.

    A MultiplexForecaster can be created by using the "|" dunder method on
    either forecaster or MultiplexForecaster objects. Here we test that it performs
    as expected on all the use cases, and raises the expected error in some others.
    """
    # test a simple | example with two forecasters:
    multiplex_two_forecaster = AutoETS() | NaiveForecaster()
    assert isinstance(multiplex_two_forecaster, MultiplexForecaster)
    assert len(multiplex_two_forecaster.forecasters) == 2
    # now test that | also works on two MultiplexForecasters:
    multiplex_one = MultiplexForecaster([("arima", AutoARIMA()), ("ets", AutoETS())])
    multiplex_two = MultiplexForecaster(
        [("theta", ThetaForecaster()), ("naive", NaiveForecaster())]
    )
    multiplex_two_multiplex = multiplex_one | multiplex_two
    assert isinstance(multiplex_two_multiplex, MultiplexForecaster)
    assert len(multiplex_two_multiplex.forecasters) == 4
    # last we will check 3 forecaster with the same name - should check both that
    # MultiplexForecaster | forecaster works, and that ensure_unique_names works
    multiplex_same_name_three_test = (
        NaiveForecaster(strategy="last")
        | NaiveForecaster(strategy="mean")
        | NaiveForecaster(strategy="drift")
    )
    assert isinstance(multiplex_same_name_three_test, MultiplexForecaster)
    assert len(multiplex_same_name_three_test.forecasters) == 3
    forecaster_param_names = multiplex_same_name_three_test._get_estimator_names(
        multiplex_same_name_three_test._forecasters
    )
    assert len(set(forecaster_param_names)) == 3

    # test we get a ValueError if we try to | with anything else:
    with pytest.raises(TypeError):
        multiplex_one | "this shouldn't work"
