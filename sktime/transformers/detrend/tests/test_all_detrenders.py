#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np
import pytest
from sktime.transformers.detrend import Detrender, Deseasonaliser
from sktime.utils.testing.base import _construct_instance
from sktime.utils.testing.forecasting import make_forecasting_problem

DETRENDERS = [
    Deseasonaliser,
    Detrender,
]

# testing data
y_train, y_test = make_forecasting_problem()


@pytest.mark.parametrize("Transformer", DETRENDERS)
def test_transform_time_index(Transformer):
    t = _construct_instance(Transformer)
    t.fit(y_train)
    yt = t.transform(y_test)
    np.testing.assert_array_equal(yt.index, y_test.index)


@pytest.mark.parametrize("Transformer", DETRENDERS)
def test_inverse_transform_time_index(Transformer):
    t = _construct_instance(Transformer)
    t.fit(y_train)
    yit = t.inverse_transform(y_test)
    np.testing.assert_array_equal(yit.index, y_test.index)


@pytest.mark.parametrize("Transformer", DETRENDERS)
def test_transform_inverse_transform_equivalence(Transformer):
    t = _construct_instance(Transformer)
    t.fit(y_train)
    yit = t.inverse_transform(t.transform(y_train))
    np.testing.assert_array_equal(y_train.index, yit.index)
    np.testing.assert_allclose(y_train.values, yit.values)
