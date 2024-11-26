#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
<<<<<<< HEAD
"""Tests that RecursiveReductionForecaster can handle (1) index with missing values
- e.g. Jan, Feb, Apr, ... etc. and (2) hierarchical data with unequal lengths."""
=======
"""Test that RecursiveReductionForecaster can handle index with missing values
- e.g. Jan, Feb, Apr, ... etc."""
>>>>>>> bff872fa9 (added test robust to missing index values)

__author__ = ["ericjb"]

import numpy as np
from sklearn.neural_network import MLPRegressor

from sktime.datasets import load_airline
from sktime.datatypes import get_cutoff
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose._reduce import RecursiveReductionForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils._testing.forecasting import (
    _assert_correct_columns,
    _assert_correct_pred_time_index,
)
<<<<<<< HEAD
from sktime.utils._testing.hierarchical import _make_hierarchical
=======
>>>>>>> bff872fa9 (added test robust to missing index values)

# warnings.filterwarnings("ignore", \
# message="RecursiveReductionForecaster is experimental")
# warnings.filterwarnings("ignore", category=DataConversionWarning)
# warnings.filterwarnings("ignore", message="X does not have valid feature names, \
# but MLPRegressor was fitted with feature names")


def test_missing_index_with_recursive_reduction():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    y = load_airline()
    TEST_SIZE = 6
    train, test = temporal_train_test_split(y, test_size=TEST_SIZE)
    SP = 12  # SP = Seasonal Period
    regressor = MLPRegressor(
        hidden_layer_sizes=(7,),
        shuffle=False,
        activation="relu",
        max_iter=2000,
        solver="lbfgs",
    )
    mlp_fc = RecursiveReductionForecaster(regressor, window_length=SP)
    train = train.drop(train.index[-3])
    mlp_fc.fit(train)
    fh = ForecastingHorizon(test.index, is_relative=False)
    pred = mlp_fc.predict(fh)

    cutoff = get_cutoff(train, return_index=True)
    _assert_correct_pred_time_index(pred.index, cutoff, fh)
    _assert_correct_columns(pred, train)
<<<<<<< HEAD


def test_unequal_indices_recursive_reduction():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    y = _make_hierarchical(hierarchy_levels=(8,), min_timepoints=11, max_timepoints=13)

    TEST_SIZE = 3
    train, test = temporal_train_test_split(y, test_size=TEST_SIZE)
    regressor = MLPRegressor(
        hidden_layer_sizes=(7,),
        shuffle=False,
        activation="relu",
        max_iter=2000,
        solver="lbfgs",
    )
    mlp_fc = RecursiveReductionForecaster(regressor, window_length=4)
    mlp_fc.fit(train)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    pred = mlp_fc.predict(fh)

    expected_idx = fh.get_expected_pred_idx(y=train)
    assert expected_idx.equals(pred.index)
=======
>>>>>>> bff872fa9 (added test robust to missing index values)
