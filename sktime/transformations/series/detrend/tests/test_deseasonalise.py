#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pytest
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import TEST_SPS
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.utils._testing.forecasting import make_forecasting_problem
from statsmodels.tsa.seasonal import seasonal_decompose

MODELS = ["additive", "multiplicative"]

y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.parametrize("sp", TEST_SPS)
def test_deseasonalised_values(sp):
    transformer = Deseasonalizer(sp=sp)
    transformer.fit(y_train)
    actual = transformer.transform(y_train)

    r = seasonal_decompose(y_train, period=sp)
    expected = y_train - r.seasonal
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("model", MODELS)
def test_transform_time_index(sp, model):
    transformer = Deseasonalizer(sp=sp, model=model)
    transformer.fit(y_train)
    yt = transformer.transform(y_test)
    np.testing.assert_array_equal(yt.index, y_test.index)


@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("model", MODELS)
def test_inverse_transform_time_index(sp, model):
    transformer = Deseasonalizer(sp=sp, model=model)
    transformer.fit(y_train)
    yit = transformer.inverse_transform(y_test)
    np.testing.assert_array_equal(yit.index, y_test.index)


@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("model", MODELS)
def test_transform_inverse_transform_equivalence(sp, model):
    transformer = Deseasonalizer(sp=sp, model=model)
    transformer.fit(y_train)
    yit = transformer.inverse_transform(transformer.transform(y_train))
    np.testing.assert_array_equal(y_train.index, yit.index)
    np.testing.assert_array_almost_equal(y_train, yit)
