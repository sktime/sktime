#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pytest
from sktime.tests.test_all_estimators import EXCLUDED_ESTIMATORS
from sktime.utils import all_estimators
from sktime.utils._testing import _construct_instance
from sktime.utils._testing import _make_args

ALL_SERIES_AS_FEATURES_TRANSFORMERS = [
    e[1] for e in all_estimators("series_as_features_transformer")
    if e[0] not in EXCLUDED_ESTIMATORS
]


@pytest.mark.parametrize("Transformer", ALL_SERIES_AS_FEATURES_TRANSFORMERS)
def test_transformed_data_has_same_index_as_input_data(Transformer):
    transformer = _construct_instance(Transformer)
    X, y = _make_args(transformer, "fit")
    Xt = transformer.fit_transform(X, y)
    np.testing.assert_array_equal(X.index, Xt.index)
