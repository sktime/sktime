#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for Bagging Forecasters."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.compose import BaggingForecaster

# from sktime.forecasting.compose._bagging import _calculate_data_quantiles
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.bootstrap import STLBootstrapTransformer
from sktime.transformations.series.boxcox import LogTransformer

y = load_airline()


@pytest.mark.parametrize("transformer", [LogTransformer, NaiveForecaster])
def test_bagging_forecaster_transformer_type_error(transformer):
    """Test that the right exception is raised for invalid transformer."""
    with pytest.raises(TypeError) as ex:
        f = BaggingForecaster(
            bootstrapping_transformer=transformer, forecaster=NaiveForecaster(sp=12)
        )
        f.fit(y)
        msg = (
            "bootstrap_transformer in BaggingForecaster should be a Transformer "
            "that take as input a Series and output a Panel."
        )
        assert msg == ex.value


@pytest.mark.parametrize("forecaster", [LogTransformer])
def test_bagging_forecaster_forecaster_type_error(forecaster):
    """Test that the right exception is raised for invalid forecaster."""
    with pytest.raises(TypeError) as ex:
        f = BaggingForecaster(
            bootstrapping_transformer=STLBootstrapTransformer(sp=12),
            forecaster=forecaster,
        )
        f.fit(y)
        msg = "forecaster in BaggingForecaster should be an sktime Forecaster"
        assert msg == ex.value
