#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

import numpy as np
import pytest
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformers.single_series.adapt import \
    SingleSeriesTransformAdaptor
from sktime.transformers.single_series.boxcox import BoxCoxTransformer
from sktime.transformers.single_series.detrend import ConditionalDeseasonalizer
from sktime.transformers.single_series.detrend import Deseasonalizer
from sktime.transformers.single_series.detrend import Detrender
from sktime.utils._testing import _construct_instance
from sktime.utils._testing.forecasting import make_forecasting_problem

SINGLE_SERIES_TRANSFORMERS = [
    Deseasonalizer,
    ConditionalDeseasonalizer,
    Detrender,
    SingleSeriesTransformAdaptor,
    BoxCoxTransformer
]

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.parametrize("Transformer", SINGLE_SERIES_TRANSFORMERS)
def test_transform_time_index(Transformer):
    t = _construct_instance(Transformer)
    t.fit(y_train)
    yt = t.transform(y_test)
    np.testing.assert_array_equal(yt.index, y_test.index)


@pytest.mark.parametrize("Transformer", SINGLE_SERIES_TRANSFORMERS)
def test_inverse_transform_time_index(Transformer):
    t = _construct_instance(Transformer)
    t.fit(y_train)
    yit = t.inverse_transform(y_test)
    np.testing.assert_array_equal(yit.index, y_test.index)


@pytest.mark.parametrize("Transformer", SINGLE_SERIES_TRANSFORMERS)
def test_transform_inverse_transform_equivalence(Transformer):
    t = _construct_instance(Transformer)
    t.fit(y_train)
    yit = t.inverse_transform(t.transform(y_train))
    np.testing.assert_array_equal(y_train.index, yit.index)
    np.testing.assert_allclose(y_train.values, yit.values)
