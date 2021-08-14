#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Differencer functionality."""

__author__ = ["Ryan Kuhns"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.transformations.series.difference import Differencer
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal

y_airline = load_airline()
y_airline_df = pd.concat([y_airline, y_airline], axis=1)
y_airline_df.columns = ["Passengers 1", "Passengers 2"]

test_cases = [y_airline, y_airline_df]
lags_to_test = [1, 12, (3), [5], np.array([7]), (8, 3), [1, 12], np.array([5, 7, 1])]


@pytest.mark.parametrize("y", test_cases)
@pytest.mark.parametrize("lags", lags_to_test)
def test_differencer_same_series(y, lags):
    transformer = Differencer(lags=lags)
    y_transform = transformer.fit_transform(y)
    y_reconstructed = transformer.inverse_transform(y_transform)

    # Reconstruction should return the reconstructed series for same indices
    # that are in the `Z` timeseries passed to inverse_transform
    _assert_array_almost_equal(y.loc[y_reconstructed.index], y_reconstructed)


@pytest.mark.parametrize("y", test_cases)
@pytest.mark.parametrize("lags", lags_to_test)
def test_differencer_remove_missing_false(y, lags):
    transformer = Differencer(lags=lags, drop_na=False)
    y_transform = transformer.fit_transform(y)
    y_reconstructed = transformer.inverse_transform(y_transform)

    _assert_array_almost_equal(y, y_reconstructed)


@pytest.mark.parametrize("y", test_cases)
@pytest.mark.parametrize("lags", lags_to_test)
def test_differencer_prediction(y, lags):
    y_train = y.iloc[:-12].copy()
    y_true = y.iloc[-12:].copy()

    transformer = Differencer(lags=[1, 12])
    y_transform = transformer.fit_transform(y)

    # Use the actual transformed values as predictions since we know we should
    # be able to convert them to the units of the original series and exactly
    # match the y_true values for this period
    y_pred = y_transform.iloc[-12:].copy()

    # Redo the transformer's fit and transformation
    # Now the transformer doesn't know anything about the values in y_true
    # This simulates use-case with a forecasting pipeline
    y_transform = transformer.fit_transform(y_train)

    y_pred_inv = transformer.inverse_transform(y_pred)

    _assert_array_almost_equal(y_true, y_pred_inv)
