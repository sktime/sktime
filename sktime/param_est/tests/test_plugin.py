# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for parameter plugin transformers."""

__author__ = ["fkiraly"]

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.param_est.fixed import FixedParams
from sktime.param_est.plugin import PluginParamsForecaster
from sktime.param_est.seasonality import SeasonalityACF
from sktime.transformations.series.difference import Differencer
from sktime.utils.validation._dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_seasonality_acf():
    """Test PluginParamsForecaster on airline data. Same as docstring example."""
    y = load_airline()

    sp_est = Differencer() * SeasonalityACF()
    fcst = NaiveForecaster()
    sp_auto = PluginParamsForecaster(sp_est, fcst)
    sp_auto.fit(y, fh=[1, 2, 3])
    assert sp_auto.forecaster_.get_params()["sp"] == 12

    PluginParamsForecaster(
        FixedParams({"foo": 12}), NaiveForecaster(), params={"foo": "sp"}
    )
