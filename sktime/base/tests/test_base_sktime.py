# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseObject universal base class that require sktime or sklearn imports."""

__author__ = ["fkiraly"]


def test_get_fitted_params_sklearn():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj sktime component returns expected nested params
    """
    from sktime.datasets import load_airline
    from sktime.forecasting.trend import TrendForecaster

    y = load_airline()
    f = TrendForecaster().fit(y)

    params = f.get_fitted_params()

    assert "regressor__coef" in params.keys()
    assert "regressor" in params.keys()
    assert "regressor__intercept" in params.keys()
